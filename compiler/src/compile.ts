import { randomUUID } from "node:crypto";
import {
  lstat,
  mkdir,
  open,
  readFile,
  readlink,
  rename,
  unlink,
} from "node:fs/promises";
import path from "node:path";

import ts from "typescript";

import { analyzeSemaSite, type SemaSiteAnalysis } from "./analyze-site.js";
import { createSourceIrBundle, type SourceIrBundle } from "./bundle-ir.js";
import { resolveDefinitionConfiguration } from "./definition.js";
import {
  buildSemaExecutionPlan,
  ExecutionPlanError,
  type ExecutionPlanSiteDescriptor,
} from "./execution-plan.js";
import {
  computeSemanticSha256,
  createFunctionId,
  normalizeProjectRelativeSourcePath,
  serializeIrBundle,
  sha256Hex,
  stringifyExactJson,
} from "./identity.js";
import {
  createFirstBeforeSemaRewriteTransformer,
  type PlannedSemaRewrite,
} from "./rewrite.js";
import {
  findMalformedSemaSites,
  findSemaSites,
  type MalformedSemaSite,
  type SemaSite,
} from "./sema-sites.js";
import {
  createSourceNeuralFunctionIr,
  type SourceNeuralFunctionIr,
} from "./source-ir.js";
import { compareBytes } from "./semantic-json.js";

const MALFORMED_SITE_DIAGNOSTIC = 9100;
const COMPILATION_DIAGNOSTIC = 9130;
const DEFAULT_BUNDLE_NAME = "semantscript.ir.v1.json";
const textEncoder = new TextEncoder();

export interface PlanSemaCompilationOptions {
  readonly projectRoot: string;
  readonly encoderRef?: string;
  readonly adapterRef?: string;
  readonly readSourceBytes?: (
    sourceFile: ts.SourceFile,
  ) => Uint8Array | Promise<Uint8Array>;
}

export interface PlannedSemaSite extends PlannedSemaRewrite {
  readonly analysis: SemaSiteAnalysis;
  readonly semanticSha256: string;
  readonly duplicateOrdinal: number;
  readonly record: SourceNeuralFunctionIr;
}

export interface SemaCompilationPlan {
  readonly sites: readonly PlannedSemaSite[];
  readonly bundle: SourceIrBundle;
  readonly bundleText: string;
}

interface SealedPlanSite extends PlannedSemaRewrite {
  readonly semanticSha256: string;
  readonly duplicateOrdinal: number;
  readonly recordText: string;
}

interface CompilationPlanSeal {
  readonly sites: readonly SealedPlanSite[];
  readonly bundleText: string;
}

export type PlanSemaCompilationResult =
  | { readonly ok: true; readonly value: SemaCompilationPlan }
  | { readonly ok: false; readonly diagnostics: readonly ts.Diagnostic[] };

export interface EmitSemaCompilationOptions {
  readonly bundlePath?: string;
}

export interface EmittedSemaCompilation {
  readonly plan: SemaCompilationPlan;
  readonly emittedFiles: readonly string[];
  readonly bundlePath: string;
}

export type EmitSemaCompilationResult =
  | { readonly ok: true; readonly value: EmittedSemaCompilation }
  | { readonly ok: false; readonly diagnostics: readonly ts.Diagnostic[] };

const planSeals = new WeakMap<SemaCompilationPlan, CompilationPlanSeal>();

export interface CompileSemantScriptProgramOptions
  extends PlanSemaCompilationOptions, EmitSemaCompilationOptions {}

export async function planSemaCompilation(
  program: ts.Program,
  options: PlanSemaCompilationOptions,
): Promise<PlanSemaCompilationResult> {
  const sites = findSemaSites(program)
    .map((site) => ({
      site,
      normalizedPath: normalizeProjectRelativeSourcePath(
        options.projectRoot,
        site.sourceFile.fileName,
      ),
    }))
    .sort(compareLocatedSites);
  const diagnostics: ts.Diagnostic[] = findMalformedSemaSites(program).map(
    malformedSiteDiagnostic,
  );
  const prepared: PreparedSite[] = [];
  const sourceDigests = new Map<string, Promise<string>>();
  const readSourceBytes = options.readSourceBytes ?? defaultReadSourceBytes;

  for (const { site, normalizedPath } of sites) {
    const analysisResult = analyzeSemaSite(program, site);

    if (!analysisResult.ok) {
      diagnostics.push(...analysisResult.diagnostics);
      continue;
    }

    const analysis = analysisResult.value;
    const definitionResult = resolveDefinitionConfiguration(
      program,
      site,
      analysis.ir.inputs,
      analysis.ir.output,
    );

    if (!definitionResult.ok) {
      diagnostics.push(...definitionResult.diagnostics);
      continue;
    }

    let sourceDigest = sourceDigests.get(site.sourceFile.fileName);

    if (!sourceDigest) {
      sourceDigest = Promise.resolve(readSourceBytes(site.sourceFile)).then(
        (bytes) => sha256Hex(bytes),
      );
      sourceDigests.set(site.sourceFile.fileName, sourceDigest);
    }

    prepared.push({
      site,
      normalizedPath,
      analysis,
      configuration: definitionResult.value,
      sourceSha256: await sourceDigest,
    });
  }

  if (diagnostics.length > 0) {
    return { ok: false, diagnostics };
  }

  const duplicateCounts = new Map<string, number>();
  const plannedSites: PlannedSemaSite[] = [];

  for (const preparedSite of prepared) {
    const { analysis, configuration, normalizedPath, site, sourceSha256 } =
      preparedSite;
    const configuredAnalysis: SemaSiteAnalysis = {
      ...analysis,
      ir: { ...analysis.ir, template: configuration.definition.template },
    };
    const runtime = {
      resultMode: analysis.ir.resultMode,
      confidenceThreshold: configuration.confidenceThreshold,
      fallbackRef: null,
      synchronous: true as const,
    };
    const semanticSha256 = computeSemanticSha256({
      irVersion: 1,
      definition: configuration.definition,
      inputs: analysis.ir.inputs,
      output: analysis.ir.output,
      runtime,
    });
    const duplicateKey = `${normalizedPath}\u0000${semanticSha256}`;
    const duplicateOrdinal = duplicateCounts.get(duplicateKey) ?? 0;
    duplicateCounts.set(duplicateKey, duplicateOrdinal + 1);
    const functionId = createFunctionId(
      normalizedPath,
      duplicateOrdinal,
      semanticSha256,
    );
    const record = createSourceNeuralFunctionIr(configuredAnalysis, {
      id: functionId,
      semanticSha256,
      sourcePath: normalizedPath,
      sourceSha256,
      encoderRef: options.encoderRef ?? "encoder.main",
      adapterRef: options.adapterRef ?? "adapter.application",
      headRefs: createHeadRefs(
        functionId,
        analysis.ir.output.kind === "scalar"
          ? 1
          : analysis.ir.output.fields.length,
      ),
      confidenceThreshold: configuration.confidenceThreshold,
      examples: configuration.definition.examples,
      constraints: configuration.definition.constraints,
    });

    plannedSites.push({
      sourceFile: site.sourceFile,
      start: site.location.start,
      end: site.location.end,
      functionId,
      inputNames: analysis.ir.inputs.map(({ name }) => name),
      analysis: configuredAnalysis,
      semanticSha256,
      duplicateOrdinal,
      record,
    });
  }

  const records = plannedSites.map(({ record }) => record);
  let executionPlan;

  try {
    executionPlan = buildSemaExecutionPlan(
      program.getTypeChecker(),
      plannedSites.map(createExecutionPlanDescriptor),
      { sourceFiles: program.getSourceFiles() },
    );
  } catch (error) {
    if (!(error instanceof ExecutionPlanError)) {
      throw error;
    }

    return {
      ok: false,
      diagnostics: [executionPlanDiagnostic(error, plannedSites[0])],
    };
  }

  const bundle = createSourceIrBundle(records, executionPlan);
  const plan: SemaCompilationPlan = {
    sites: plannedSites,
    bundle,
    bundleText: serializeIrBundle(bundle),
  };
  planSeals.set(plan, sealCompilationPlan(plan));
  return {
    ok: true,
    value: plan,
  };
}

export async function emitSemaCompilation(
  program: ts.Program,
  plan: SemaCompilationPlan,
  options: EmitSemaCompilationOptions = {},
): Promise<EmitSemaCompilationResult> {
  const preEmitDiagnostics = ts.getPreEmitDiagnostics(program);

  if (hasErrors(preEmitDiagnostics)) {
    return { ok: false, diagnostics: preEmitDiagnostics };
  }

  const bundlePath = resolveBundlePath(program, options.bundlePath);
  await assertBundlePathDoesNotOverwriteSource(program, bundlePath);
  const planSnapshot = assertPlanMatchesProgram(program, plan);
  const outputs = new Map<string, string>();
  const transformer = createFirstBeforeSemaRewriteTransformer(
    planSnapshot.sites,
  );
  const emitResult = program.emit(
    undefined,
    (fileName, data, writeByteOrderMark) => {
      const contents = `${writeByteOrderMark ? "\uFEFF" : ""}${data}`;
      outputs.set(
        path.resolve(fileName),
        sanitizeEmittedOutput(fileName, contents),
      );
    },
    undefined,
    false,
    { before: [transformer] },
  );
  const emitDiagnostics = [...emitResult.diagnostics];

  if (emitResult.emitSkipped || hasErrors(emitDiagnostics)) {
    if (emitDiagnostics.length === 0) {
      emitDiagnostics.push(
        configurationDiagnostic("TypeScript skipped JavaScript emission"),
      );
    }

    return { ok: false, diagnostics: emitDiagnostics };
  }

  if (
    planSnapshot.sites.length > 0 &&
    ![...outputs.keys()].some(isJavaScriptOutput)
  ) {
    return {
      ok: false,
      diagnostics: [
        configurationDiagnostic("TypeScript emitted no JavaScript output"),
      ],
    };
  }

  if (await anyCanonicalPathMatches([...outputs.keys()], bundlePath)) {
    throw new RangeError(
      "IR bundle path conflicts with a TypeScript output path",
    );
  }

  const emittedFiles = [...outputs.keys()].sort(compareStrings);
  await assertFileTargetIsNotDirectory(bundlePath, "IR bundle");
  await Promise.all(
    emittedFiles.map(async (fileName) =>
      assertFileTargetIsNotDirectory(fileName, "TypeScript output"),
    ),
  );
  const stagedFiles: StagedTextFile[] = [];

  try {
    const stagedBundle = await stageTextFile(
      bundlePath,
      planSnapshot.bundleText,
    );
    stagedFiles.push(stagedBundle);

    for (const fileName of emittedFiles) {
      const contents = outputs.get(fileName);

      if (contents === undefined) {
        throw new Error(`missing captured TypeScript output ${fileName}`);
      }

      stagedFiles.push(await stageTextFile(fileName, contents));
    }

    for (const staged of stagedFiles.slice(1)) {
      await commitStagedFile(staged);
    }

    await commitStagedFile(stagedBundle);
  } catch (error) {
    await Promise.all(
      stagedFiles.map(({ temporary }) =>
        unlink(temporary).catch(() => undefined),
      ),
    );
    throw error;
  }

  return { ok: true, value: { plan, emittedFiles, bundlePath } };
}

export async function compileSemantScriptProgram(
  program: ts.Program,
  options: CompileSemantScriptProgramOptions,
): Promise<EmitSemaCompilationResult> {
  const planned = await planSemaCompilation(program, options);

  if (!planned.ok) {
    return planned;
  }

  return emitSemaCompilation(program, planned.value, options);
}

interface PreparedSite {
  readonly site: SemaSite;
  readonly normalizedPath: string;
  readonly analysis: SemaSiteAnalysis;
  readonly configuration: Extract<
    ReturnType<typeof resolveDefinitionConfiguration>,
    { readonly ok: true }
  >["value"];
  readonly sourceSha256: string;
}

function compareLocatedSites(
  left: { readonly site: SemaSite; readonly normalizedPath: string },
  right: { readonly site: SemaSite; readonly normalizedPath: string },
): number {
  const pathOrder = compareBytes(
    textEncoder.encode(left.normalizedPath),
    textEncoder.encode(right.normalizedPath),
  );
  return pathOrder || left.site.location.start - right.site.location.start;
}

function createHeadRefs(functionId: string, count: number): readonly string[] {
  const identity = functionId.slice(3);
  return Array.from(
    { length: count },
    (_, index) => `head.${identity}.${index.toString().padStart(3, "0")}`,
  );
}

function createExecutionPlanDescriptor(
  site: PlannedSemaSite,
): ExecutionPlanSiteDescriptor {
  const { template } = site.analysis.site.node;
  const expressions = ts.isTemplateExpression(template)
    ? template.templateSpans.map(({ expression }) => expression)
    : [];

  if (expressions.length !== site.inputNames.length) {
    throw new RangeError(
      "analyzed sema inputs do not match their template expressions",
    );
  }

  return {
    functionId: site.functionId,
    normalizedSourcePath: site.record.source.path,
    start: site.start,
    expression: site.analysis.site.node,
    inputs: site.inputNames.map((name, index) => {
      const expression = expressions[index];
      if (expression === undefined) {
        throw new RangeError("missing analyzed sema input expression");
      }
      return { name, expression };
    }),
  };
}

async function defaultReadSourceBytes(
  sourceFile: ts.SourceFile,
): Promise<Uint8Array> {
  return readFile(sourceFile.fileName);
}

function resolveBundlePath(
  program: ts.Program,
  requested: string | undefined,
): string {
  if (requested) {
    return path.resolve(requested);
  }

  const outDir = program.getCompilerOptions().outDir;

  if (!outDir) {
    throw new TypeError(
      "bundlePath is required when the TypeScript program has no outDir",
    );
  }

  return path.resolve(outDir, DEFAULT_BUNDLE_NAME);
}

function sealCompilationPlan(plan: SemaCompilationPlan): CompilationPlanSeal {
  return {
    sites: plan.sites.map((site) => ({
      sourceFile: site.sourceFile,
      start: site.start,
      end: site.end,
      functionId: site.functionId,
      inputNames: [...site.inputNames],
      semanticSha256: site.semanticSha256,
      duplicateOrdinal: site.duplicateOrdinal,
      recordText: stringifyExactJson(site.record),
    })),
    bundleText: plan.bundleText,
  };
}

function assertPlanMatchesProgram(
  program: ts.Program,
  plan: SemaCompilationPlan,
): CompilationPlanSeal {
  const discoveredSites = findSemaSites(program);

  if (plan.sites.length !== discoveredSites.length) {
    throw new RangeError(
      `compilation plan contains ${String(plan.sites.length)} rewrites for ${String(discoveredSites.length)} discovered sema sites`,
    );
  }

  const remaining = new Map<ts.SourceFile, Set<string>>();

  for (const site of discoveredSites) {
    const keys = remaining.get(site.sourceFile) ?? new Set<string>();
    keys.add(siteKey(site.location.start, site.location.end));
    remaining.set(site.sourceFile, keys);
  }

  for (const site of plan.sites) {
    if (program.getSourceFile(site.sourceFile.fileName) !== site.sourceFile) {
      throw new RangeError(
        "compilation plan contains a source file from another Program",
      );
    }

    const keys = remaining.get(site.sourceFile);

    if (!keys?.delete(siteKey(site.start, site.end))) {
      throw new RangeError(
        "compilation plan does not match the Program's discovered sema sites",
      );
    }

    if (
      site.record.id !== site.functionId ||
      site.record.semanticSha256 !== site.semanticSha256
    ) {
      throw new RangeError(
        "compilation plan rewrite identity does not match its IR record",
      );
    }

    if (
      site.inputNames.length !== site.record.inputs.length ||
      site.inputNames.some(
        (name, index) => name !== site.record.inputs[index]?.name,
      )
    ) {
      throw new RangeError(
        "compilation plan rewrite inputs do not match its IR record",
      );
    }
  }

  if ([...remaining.values()].some((keys) => keys.size > 0)) {
    throw new RangeError("compilation plan omits a discovered sema site");
  }

  const plannedRecords = plan.sites.map(({ record }) => record);
  const bundleFunctionsText = stringifyExactJson(plan.bundle.functions);
  const validatedBundle = createSourceIrBundle(
    plan.bundle.functions,
    plan.bundle.executionPlan,
  );
  const currentBundleText = serializeIrBundle(validatedBundle);

  if (
    stringifyExactJson(plannedRecords) !== bundleFunctionsText ||
    plan.bundleText !== currentBundleText
  ) {
    throw new RangeError(
      "compilation plan bundle and bundle text are inconsistent",
    );
  }

  const seal = planSeals.get(plan);

  if (!seal) {
    throw new TypeError(
      "emit requires the exact plan returned by planSemaCompilation",
    );
  }

  if (
    seal.bundleText !== plan.bundleText ||
    seal.sites.length !== plan.sites.length ||
    seal.sites.some((expected, index) => {
      const actual = plan.sites[index];
      return (
        !actual ||
        expected.sourceFile !== actual.sourceFile ||
        expected.start !== actual.start ||
        expected.end !== actual.end ||
        expected.functionId !== actual.functionId ||
        expected.semanticSha256 !== actual.semanticSha256 ||
        expected.duplicateOrdinal !== actual.duplicateOrdinal ||
        expected.recordText !== stringifyExactJson(actual.record) ||
        expected.inputNames.length !== actual.inputNames.length ||
        expected.inputNames.some(
          (name, inputIndex) => name !== actual.inputNames[inputIndex],
        )
      );
    })
  ) {
    throw new RangeError("compilation plan was modified after planning");
  }

  return seal;
}

function siteKey(start: number, end: number): string {
  return `${String(start)}:${String(end)}`;
}

async function assertBundlePathDoesNotOverwriteSource(
  program: ts.Program,
  bundlePath: string,
): Promise<void> {
  if (
    await anyCanonicalPathMatches(
      program.getSourceFiles().map(({ fileName }) => fileName),
      bundlePath,
    )
  ) {
    throw new RangeError(
      "IR bundle path conflicts with a TypeScript source file",
    );
  }
}

async function assertFileTargetIsNotDirectory(
  fileName: string,
  description: string,
): Promise<void> {
  try {
    if ((await lstat(fileName)).isDirectory()) {
      throw new RangeError(
        `${description} path must not be an existing directory`,
      );
    }
  } catch (error) {
    if (!isMissingPathError(error)) {
      throw error;
    }
  }
}

async function anyCanonicalPathMatches(
  candidates: readonly string[],
  target: string,
): Promise<boolean> {
  const targetKey = comparisonPath(await canonicalizePath(target));
  const candidateKeys = await Promise.all(
    candidates.map(async (candidate) =>
      comparisonPath(await canonicalizePath(candidate)),
    ),
  );
  return candidateKeys.includes(targetKey);
}

async function canonicalizePath(fileName: string): Promise<string> {
  return await canonicalizePathSegments(path.resolve(fileName), new Set());
}

async function canonicalizePathSegments(
  absolutePath: string,
  seenLinks: Set<string>,
): Promise<string> {
  const root = path.parse(absolutePath).root;
  const segments = path
    .relative(root, absolutePath)
    .split(path.sep)
    .filter(Boolean);
  let current = root;

  for (const [index, segment] of segments.entries()) {
    const candidate = path.join(current, segment);

    try {
      const status = await lstat(candidate);

      if (!status.isSymbolicLink()) {
        current = candidate;
        continue;
      }

      if (seenLinks.has(candidate) || seenLinks.size >= 40) {
        throw new RangeError(
          `too many symbolic links while resolving ${absolutePath}`,
        );
      }

      seenLinks.add(candidate);
      const linkTarget = await readlink(candidate);
      const resolvedTarget = path.resolve(path.dirname(candidate), linkTarget);
      return await canonicalizePathSegments(
        path.join(resolvedTarget, ...segments.slice(index + 1)),
        seenLinks,
      );
    } catch (error) {
      if (!isMissingPathError(error)) {
        throw error;
      }

      return path.join(current, ...segments.slice(index));
    }
  }

  return current;
}

function comparisonPath(fileName: string): string {
  return ts.sys.useCaseSensitiveFileNames ? fileName : fileName.toLowerCase();
}

function isMissingPathError(error: unknown): boolean {
  return (
    error instanceof Error &&
    "code" in error &&
    (error.code === "ENOENT" || error.code === "ENOTDIR")
  );
}

function sanitizeEmittedOutput(fileName: string, contents: string): string {
  if (fileName.endsWith(".map")) {
    const parsed = JSON.parse(contents) as Record<string, unknown>;
    delete parsed["sourcesContent"];
    return JSON.stringify(parsed);
  }

  return contents.replace(
    /(sourceMappingURL=data:application\/json(?:;charset=[^;,]+)?;base64,)([A-Za-z0-9+/=]+)/gu,
    (_match, prefix: string, payload: string) => {
      const parsed = JSON.parse(
        Buffer.from(payload, "base64").toString("utf8"),
      ) as Record<string, unknown>;
      delete parsed["sourcesContent"];
      return `${prefix}${Buffer.from(JSON.stringify(parsed)).toString("base64")}`;
    },
  );
}

interface StagedTextFile {
  readonly target: string;
  readonly temporary: string;
}

async function stageTextFile(
  fileName: string,
  contents: string,
): Promise<StagedTextFile> {
  const directory = path.dirname(fileName);
  await mkdir(directory, { recursive: true });
  const temporary = path.join(
    directory,
    `.${path.basename(fileName)}.${process.pid.toString()}.${randomUUID()}.tmp`,
  );

  try {
    const handle = await open(temporary, "wx");

    try {
      await handle.writeFile(contents, "utf8");
      await handle.sync();
    } finally {
      await handle.close();
    }

    return { target: fileName, temporary };
  } catch (error) {
    await unlink(temporary).catch(() => undefined);
    throw error;
  }
}

async function commitStagedFile(staged: StagedTextFile): Promise<void> {
  await rename(staged.temporary, staged.target);
}

function malformedSiteDiagnostic(
  site: MalformedSemaSite,
): ts.DiagnosticWithLocation {
  const siteText = site.node.getText(site.sourceFile);
  const prefix = `${site.location.fileName}:${String(site.location.line)}:${String(site.location.column)}`;

  return {
    category: ts.DiagnosticCategory.Error,
    code: MALFORMED_SITE_DIAGNOSTIC,
    file: site.sourceFile,
    start: site.location.start,
    length: site.location.end - site.location.start,
    messageText: `${prefix}: ${site.reason} (site: ${siteText})`,
  };
}

function configurationDiagnostic(messageText: string): ts.Diagnostic {
  return {
    category: ts.DiagnosticCategory.Error,
    code: COMPILATION_DIAGNOSTIC,
    file: undefined,
    start: undefined,
    length: undefined,
    messageText,
  };
}

function executionPlanDiagnostic(
  error: ExecutionPlanError,
  site: PlannedSemaSite | undefined,
): ts.Diagnostic {
  return {
    category: ts.DiagnosticCategory.Error,
    code: COMPILATION_DIAGNOSTIC + 1,
    file: site?.sourceFile,
    start: site?.start,
    length: site === undefined ? undefined : site.end - site.start,
    messageText: `${error.code}: ${error.message}`,
  };
}

function hasErrors(diagnostics: readonly ts.Diagnostic[]): boolean {
  return diagnostics.some(
    ({ category }) => category === ts.DiagnosticCategory.Error,
  );
}

function isJavaScriptOutput(fileName: string): boolean {
  return /\.(?:c|m)?js$/u.test(fileName);
}

function compareStrings(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
