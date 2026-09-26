import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import type {
  SemaCallObservation,
  SemaDiagnosticResult,
  SemaFallback,
} from "@semantscript/core";

import {
  ConstraintEvaluationError,
  evaluatePredicate,
  jsonStrictEqual,
} from "./constraint-eval.js";
import {
  bundleCandidates,
  resolveArtifactRoot,
  resolveBundlePath,
} from "./defaults.js";
import { CliUsageError, stringOption, type CliIo } from "./io.js";
import {
  readArtifactRelease,
  readJson,
  type ArtifactRelease,
  type ManifestFunctionProvenance,
  type ManifestFunctionSummary,
} from "./manifest.js";
import { DISTANCE_DESCRIPTION, nearest } from "./nearest.js";
import { importModule, resolveArguments } from "./run.js";
import { formatRatio, shortId } from "./table.js";
import { findTrainingCases, type TrainingCase } from "./training-cache.js";

const DEFAULT_NEIGHBOURS = 5;
const DEFAULT_CACHE_DIR = ".semantscript/cache";
const INPUT_PREVIEW_CHARACTERS = 160;

interface BundleConstraint {
  readonly kind: string;
  readonly source: string;
  readonly predicate: unknown;
  readonly output: unknown;
}

interface BundleExample {
  readonly inputs: unknown;
  readonly output: unknown;
}

interface BundleFunction {
  readonly id: string;
  readonly source: string | null;
  readonly examples: readonly BundleExample[];
  readonly constraints: readonly BundleConstraint[];
}

interface ConstraintReport {
  readonly total: number;
  readonly active: readonly {
    readonly index: number;
    readonly kind: string;
    readonly source: string;
    readonly output: unknown;
    /** Whether the answer obeys it; null when there is no answer to check. */
    readonly satisfied: boolean | null;
  }[];
  readonly inactive: number;
  readonly errors: readonly {
    readonly index: number;
    readonly source: string;
    readonly reason: string;
  }[];
}

interface NeighbourReport {
  readonly distance: number;
  readonly inputs: unknown;
  readonly output: unknown;
}

interface TrainingReport {
  readonly datasetSha256: string | null;
  readonly datasetPath: string | null;
  readonly adversarialPath: string | null;
  readonly release: boolean;
  readonly teacher: string | null;
  readonly total: number;
  readonly otherDatasets: number;
  readonly nearest: readonly (NeighbourReport & {
    readonly origin: TrainingCase["origin"];
    readonly detail: string | null;
  })[];
}

interface ExplainedCall {
  readonly functionId: string;
  readonly inArtifact: boolean;
  readonly inBundle: boolean;
  readonly source: string | null;
  readonly inputs: unknown;
  readonly resultMode: string | null;
  readonly answer: SemaDiagnosticResult | null;
  readonly threshold: {
    readonly confidenceThreshold: number;
    readonly met: boolean;
    readonly fallbackRef: string | null;
  } | null;
  readonly verification: ManifestFunctionSummary | null;
  readonly provenance: ManifestFunctionProvenance | null;
  readonly constraints: ConstraintReport | null;
  readonly examples: {
    readonly total: number;
    readonly nearest: readonly NeighbourReport[];
  } | null;
  readonly training: TrainingReport | null;
}

interface ExplainDocument {
  readonly module: string;
  readonly export: string;
  readonly arguments: readonly unknown[];
  readonly artifact: {
    readonly root: string;
    readonly release: string;
    readonly manifestSha256: string;
    readonly createdAt: string | null;
    readonly application: { readonly id: string; readonly version: string };
  };
  readonly bundle: string | null;
  readonly cacheDir: string;
  readonly distance: string;
  readonly calls: readonly ExplainedCall[];
  /** Artifact functions the bundle no longer has: the likely earlier version of a changed expression. */
  readonly staleArtifactFunctions: readonly string[];
  readonly result: unknown;
  readonly error: { readonly name: string; readonly message: string } | null;
  readonly ok: boolean;
}

/**
 * `semantscript explain`: call one export the way `run` does, with every sema
 * call's calibrated distribution recorded, and report for each call the
 * answer, the constraints active for its input, the nearest gold examples and
 * training cases, and the release and verification the answer came from.
 */
export async function explainCommand(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const { values, positionals } = parseArgs({
    args: [...args],
    options: {
      artifact: { type: "string" },
      bundle: { type: "string" },
      "cache-dir": { type: "string" },
      neighbors: { type: "string" },
      call: { type: "string" },
      input: { type: "string" },
      "input-file": { type: "string" },
      json: { type: "boolean" },
    },
    allowPositionals: true,
  });
  const modulePath = positionals[0];
  if (modulePath === undefined || positionals.length !== 1) {
    throw new CliUsageError("explain takes exactly one module path");
  }
  const exportName = stringOption(values, "call");
  if (exportName === undefined) {
    throw new CliUsageError("explain needs --call <export>");
  }
  if (values.input !== undefined && values["input-file"] !== undefined) {
    throw new CliUsageError("pass either --input or --input-file, not both");
  }
  const neighbours = parseNeighbours(values.neighbors);
  const root = resolveArtifactRoot(values, io);
  const cacheDir = resolve(
    io.cwd,
    stringOption(values, "cache-dir") ?? DEFAULT_CACHE_DIR,
  );
  const bundlePath = findBundle(values, io);
  const callArguments = await resolveArguments(
    values.input,
    values["input-file"],
    io.cwd,
  );

  const release = await readArtifactRelease(root);
  const bundle = bundlePath === null ? undefined : await readBundle(bundlePath);

  const observations: SemaCallObservation[] = [];
  const core = await import("@semantscript/core");
  const handle = await core.loadSemaArtifact(root, {
    diagnostics: "always",
    observe: (observation) => {
      observations.push(observation);
    },
    fallbacks: topAnswerFallbacks(release),
  });
  let result: unknown;
  let error: unknown;
  let threw = false;
  try {
    const namespace = await importModule(modulePath, io.cwd);
    const target = namespace[exportName];
    if (typeof target !== "function") {
      io.stderr(`${modulePath} has no function export named ${exportName}\n`);
      return 1;
    }
    try {
      result = await Promise.resolve(
        (target as (...callArgs: unknown[]) => unknown)(...callArguments),
      );
    } catch (caught: unknown) {
      threw = true;
      error = caught;
    }
  } finally {
    await handle.close();
  }

  const calls: ExplainedCall[] = [];
  for (const observation of observations) {
    calls.push(
      await explainCall(observation, release, bundle, cacheDir, neighbours),
    );
  }
  const staleArtifactFunctions =
    bundle === undefined
      ? []
      : release.functions.map((fn) => fn.id).filter((id) => !bundle.has(id));
  const expectedError =
    threw &&
    (error instanceof core.SemaConfidenceError ||
      error instanceof core.SemaUnknownFunctionError);
  const ok =
    (!threw || expectedError) && calls.every((call) => call.inArtifact);
  const document: ExplainDocument = {
    module: resolve(io.cwd, modulePath),
    export: exportName,
    arguments: callArguments,
    artifact: {
      root,
      release: release.release,
      manifestSha256: release.manifestSha256,
      createdAt: release.createdAt,
      application: {
        id: release.applicationId,
        version: release.applicationVersion,
      },
    },
    bundle: bundlePath,
    cacheDir,
    distance: DISTANCE_DESCRIPTION,
    calls,
    staleArtifactFunctions,
    result: threw ? null : (result ?? null),
    error: threw ? errorSummary(error) : null,
    ok,
  };
  if (values.json === true) {
    io.stdout(`${JSON.stringify(document, null, 2)}\n`);
  } else {
    io.stdout(renderExplanation(document, threw && isRuntimeNotLoaded(error)));
  }
  return ok ? 0 : 1;
}

/**
 * Matches the runtime's not-loaded error by name or code, not `instanceof`:
 * the hint exists for a module that reached another copy of
 * `@semantscript/core`, whose error class is a different object from the
 * CLI's own.
 */
function isRuntimeNotLoaded(error: unknown): boolean {
  if (typeof error !== "object" || error === null) return false;
  const { name, code } = error as { name?: unknown; code?: unknown };
  return (
    code === "SEMA_RUNTIME_NOT_LOADED" || name === "SemaRuntimeNotLoadedError"
  );
}

function parseNeighbours(value: string | undefined): number {
  if (value === undefined) return DEFAULT_NEIGHBOURS;
  const count = Number(value);
  if (!Number.isInteger(count) || count < 0) {
    throw new CliUsageError("--neighbors must be a non-negative integer");
  }
  return count;
}

/** `--bundle`, else the build's bundle when one exists; explain still runs without one. */
function findBundle(
  values: Readonly<Record<string, string | boolean | undefined>>,
  io: CliIo,
): string | null {
  if (stringOption(values, "bundle") !== undefined) {
    return resolveBundlePath(values, io);
  }
  return bundleCandidates(io.cwd).find((path) => existsSync(path)) ?? null;
}

async function readBundle(
  path: string,
): Promise<ReadonlyMap<string, BundleFunction>> {
  const bundle = record(await readJson(path));
  if (bundle["kind"] !== "semantscript.ir-bundle") {
    throw new Error(`${path} is not a semantscript.ir-bundle`);
  }
  const functions = new Map<string, BundleFunction>();
  for (const entry of list(bundle["functions"])) {
    const fn = record(entry);
    const id = fn["id"];
    if (typeof id !== "string") continue;
    const source = record(fn["source"]);
    const definition = record(fn["definition"]);
    functions.set(id, {
      id,
      source:
        typeof source["path"] === "string"
          ? `${source["path"]}${typeof source["line"] === "number" ? `:${String(source["line"])}` : ""}`
          : null,
      examples: list(definition["examples"]).map((example) => ({
        inputs: record(example)["inputs"],
        output: record(example)["output"],
      })),
      constraints: list(definition["constraints"]).map((constraint) => {
        const raw = record(constraint);
        return {
          kind: String(raw["kind"]),
          source: String(raw["source"]),
          predicate: raw["predicate"],
          output: raw["output"],
        };
      }),
    });
  }
  return functions;
}

/**
 * Explain never runs the application's fallbacks (the CLI cannot import them),
 * but the runtime refuses to load an artifact whose fallbacks are unregistered.
 * Each stand-in returns the model's own top answer, which is always a valid
 * result, and the report says that the application's fallback decides instead.
 */
function topAnswerFallbacks(
  release: ArtifactRelease,
): ReadonlyMap<string, SemaFallback> {
  const fallbacks = new Map<string, SemaFallback>();
  for (const provenance of release.provenance.values()) {
    if (provenance.fallbackRef !== null) {
      fallbacks.set(provenance.fallbackRef, (_inputs, diagnostic) => {
        return diagnostic.value;
      });
    }
  }
  return fallbacks;
}

async function explainCall(
  observation: SemaCallObservation,
  release: ArtifactRelease,
  bundle: ReadonlyMap<string, BundleFunction> | undefined,
  cacheDir: string,
  neighbours: number,
): Promise<ExplainedCall> {
  const { functionId } = observation;
  const inputs = jsonForm(observation.inputs);
  const definition = bundle?.get(functionId);
  const verification =
    release.functions.find((fn) => fn.id === functionId) ?? null;
  const provenance = release.provenance.get(functionId) ?? null;
  const answer =
    observation.kind === "answered" ? (observation.diagnostic ?? null) : null;
  const scalar = answer !== null && !("fields" in answer);
  const threshold =
    observation.kind === "answered" && observation.confidenceThreshold !== null
      ? {
          confidenceThreshold: observation.confidenceThreshold,
          met:
            answer !== null &&
            confidenceOf(answer) >= observation.confidenceThreshold,
          fallbackRef: provenance?.fallbackRef ?? null,
        }
      : null;
  const training = await findTrainingCases(
    cacheDir,
    functionId,
    provenance?.datasetSha256 ?? null,
  );
  return {
    functionId,
    inArtifact: observation.kind === "answered",
    inBundle: definition !== undefined,
    source: definition?.source ?? null,
    inputs,
    resultMode: observation.kind === "answered" ? observation.resultMode : null,
    answer,
    threshold,
    verification,
    provenance,
    constraints:
      definition === undefined
        ? null
        : explainConstraints(
            definition.constraints,
            inputs,
            answer === null ? undefined : { value: answer.value, scalar },
          ),
    examples:
      definition === undefined
        ? null
        : {
            total: definition.examples.length,
            nearest: nearest(
              inputs,
              definition.examples,
              (example) => example.inputs,
              neighbours,
            ).map(({ candidate, distance }) => ({
              distance,
              inputs: candidate.inputs,
              output: candidate.output,
            })),
          },
    training:
      training === undefined
        ? null
        : {
            datasetSha256: training.datasetSha256,
            datasetPath: training.datasetPath,
            adversarialPath: training.adversarialPath,
            release: training.release,
            teacher: training.teacher,
            total: training.cases.length,
            otherDatasets: training.otherDatasets,
            nearest: nearest(
              inputs,
              training.cases,
              (entry) => entry.inputs,
              neighbours,
            ).map(({ candidate, distance }) => ({
              distance,
              inputs: candidate.inputs,
              output: candidate.output,
              origin: candidate.origin,
              detail: candidate.detail,
            })),
          },
  };
}

function explainConstraints(
  constraints: readonly BundleConstraint[],
  inputs: unknown,
  answer: { readonly value: unknown; readonly scalar: boolean } | undefined,
): ConstraintReport {
  const active: {
    index: number;
    kind: string;
    source: string;
    output: unknown;
    satisfied: boolean | null;
  }[] = [];
  const errors: { index: number; source: string; reason: string }[] = [];
  let inactive = 0;
  constraints.forEach((constraint, index) => {
    let holds: boolean;
    try {
      holds = evaluatePredicate(constraint.predicate, record(inputs));
    } catch (error: unknown) {
      if (!(error instanceof ConstraintEvaluationError)) throw error;
      errors.push({ index, source: constraint.source, reason: error.message });
      return;
    }
    if (!holds) {
      inactive += 1;
      return;
    }
    // v1 constraints bind scalar outputs, compared with JavaScript strict equality.
    const equal =
      answer === undefined || !answer.scalar
        ? null
        : jsonStrictEqual(answer.value, constraint.output);
    active.push({
      index,
      kind: constraint.kind,
      source: constraint.source,
      output: constraint.output,
      satisfied:
        equal === null ? null : constraint.kind === "always" ? equal : !equal,
    });
  });
  return { total: constraints.length, active, inactive, errors };
}

function confidenceOf(answer: SemaDiagnosticResult): number {
  return "fields" in answer ? answer.minimumFieldConfidence : answer.confidence;
}

function renderExplanation(
  document: ExplainDocument,
  otherRuntime: boolean,
): string {
  const lines: string[] = [];
  const { artifact } = document;
  lines.push(
    `explain ${document.export}(${document.arguments.map((argument) => preview(argument)).join(", ")})`,
    `release  ${artifact.release} (${artifact.application.id}@${artifact.application.version}${artifact.createdAt === null ? "" : `, built ${artifact.createdAt}`})`,
    `         manifest ${artifact.manifestSha256}`,
    `bundle   ${document.bundle ?? "none found: constraints and gold examples are not shown (pass --bundle or run semantscript build)"}`,
  );
  if (document.calls.length === 0) {
    lines.push(
      "",
      document.error === null
        ? "the export made no sema call for this input"
        : "no sema call was dispatched before the export failed",
    );
  }
  document.calls.forEach((call, index) => {
    lines.push("", ...renderCall(call, index, document));
  });
  if (document.staleArtifactFunctions.length > 0 && document.bundle !== null) {
    lines.push(
      "",
      `artifact functions the bundle no longer has (earlier versions of changed expressions): ${document.staleArtifactFunctions.map(shortId).join(", ")}`,
    );
  }
  lines.push("");
  if (document.error === null) {
    lines.push(
      `result   ${document.result === null ? "null" : preview(document.result)}`,
    );
  } else {
    lines.push(`error    ${document.error.name}: ${document.error.message}`);
    if (otherRuntime) {
      lines.push(
        "         the module reached a different @semantscript/core than the CLI loaded the artifact into; run the CLI installed in the application (npx semantscript explain)",
      );
    }
  }
  lines.push(`distance ${document.distance}`);
  return `${lines.join("\n")}\n`;
}

function renderCall(
  call: ExplainedCall,
  index: number,
  document: ExplainDocument,
): string[] {
  const lines = [
    `call ${String(index + 1)}: ${call.functionId}${call.source === null ? "" : ` (${call.source})`}`,
    `  inputs       ${preview(call.inputs)}`,
  ];
  if (!call.inArtifact) {
    lines.push(
      `  missing      the loaded artifact has no function ${shortId(call.functionId)}: ${
        call.inBundle
          ? "the expression changed since the artifact was trained"
          : "neither the artifact nor the bundle knows it (a stale build or a different artifact)"
      }; run semantscript build and semantscript train, then explain again`,
    );
  }
  if (call.answer !== null) {
    lines.push(...renderAnswer(call.answer));
  }
  if (call.resultMode === "diagnostic") {
    lines.push(
      "  mode         sema.withConfidence: the program receives the whole diagnostic result",
    );
  }
  if (call.threshold !== null) {
    const { confidenceThreshold, met, fallbackRef } = call.threshold;
    lines.push(
      `  threshold    @confidence ${String(confidenceThreshold)}: ${
        call.resultMode === "diagnostic"
          ? `${met ? "met" : "below"}, but sema.withConfidence returns the diagnostic result either way (the threshold does not change what the program receives)`
          : met
            ? "met, the program receives the value"
            : fallbackRef === null
              ? "below, the program receives SemaConfidenceError"
              : `below, the application's fallback ${fallbackRef} decides (explain shows the model's answer)`
      }`,
    );
  }
  if (call.constraints === null) {
    lines.push(
      `  constraints  not shown: ${document.bundle === null ? "no bundle" : "the bundle does not describe this function (rebuild)"}`,
    );
  } else {
    lines.push(...renderConstraints(call.constraints));
  }
  if (call.examples !== null) {
    lines.push(
      `  gold examples nearest the input (${String(call.examples.nearest.length)} of ${String(call.examples.total)})`,
      ...call.examples.nearest.map(
        (entry) =>
          `    ${entry.distance.toFixed(3)}  ${preview(entry.output)}  ${preview(entry.inputs)}`,
      ),
    );
  }
  lines.push(...renderTraining(call.training, call.inArtifact));
  if (call.verification !== null) {
    const fn = call.verification;
    const heads =
      fn.heads.length > 1
        ? `, per field ${fn.heads.map((head) => `${head.outputPath.join(".")} ${formatRatio(head.accuracy)}`).join(", ")}`
        : "";
    lines.push(
      `  verification ${fn.status}: accuracy ${formatRatio(fn.accuracy)}, ECE ${formatRatio(fn.ece)}, Brier ${formatRatio(fn.brier)}, pair consistency ${formatRatio(fn.pairConsistency)}, ${String(fn.attestedCases)} attested cases, ${String(fn.constraintViolations)} constraint violation${fn.constraintViolations === 1 ? "" : "s"}${heads}`,
    );
  }
  if (call.provenance !== null) {
    const { teacher, baseModel, datasetSha256 } = call.provenance;
    lines.push(
      `  trained      teacher ${teacher ?? "unrecorded"}, base model ${baseModel ?? "unrecorded"}, dataset ${datasetSha256 ?? "unrecorded"}`,
    );
  }
  return lines;
}

function renderAnswer(answer: SemaDiagnosticResult): string[] {
  if ("fields" in answer) {
    const lines = [
      `  value        ${preview(answer.value)}`,
      `  confidence   ${formatRatio(answer.minimumFieldConfidence)} (lowest field), uncertainty ${formatRatio(answer.maximumFieldUncertainty)} (highest field)`,
    ];
    for (const [field, result] of Object.entries(answer.fields)) {
      lines.push(
        `  field ${field}: ${preview(result.value)} confidence ${formatRatio(result.confidence)}, ${renderDistribution(result.distribution)}`,
      );
    }
    return lines;
  }
  return [
    `  value        ${preview(answer.value)}`,
    `  confidence   ${formatRatio(answer.confidence)}, uncertainty ${formatRatio(answer.uncertainty)}${answer.expectedValue === null ? "" : `, expected value ${String(answer.expectedValue)}`}`,
    `  distribution ${renderDistribution(answer.distribution)}`,
  ];
}

function renderDistribution(
  distribution: readonly {
    readonly value: unknown;
    readonly probability: number;
  }[],
): string {
  return [...distribution]
    .sort((left, right) => right.probability - left.probability)
    .map((entry) => `${preview(entry.value)} ${formatRatio(entry.probability)}`)
    .join(" | ");
}

function renderConstraints(report: ConstraintReport): string[] {
  if (report.total === 0) {
    return ["  constraints  none declared"];
  }
  const lines = [
    `  constraints  ${String(report.active.length)} active of ${String(report.total)} (${String(report.inactive)} inactive${report.errors.length === 0 ? "" : `, ${String(report.errors.length)} could not be evaluated`})`,
  ];
  for (const entry of report.active) {
    const verdict =
      entry.satisfied === null
        ? "no answer to check"
        : entry.satisfied
          ? "satisfied"
          : "VIOLATED by the answer";
    lines.push(
      `    ${entry.kind} ${preview(entry.output)} when ${entry.source}: ${verdict}`,
    );
  }
  for (const entry of report.errors) {
    lines.push(
      `    constraint ${String(entry.index)} (${entry.source}) could not be evaluated: ${entry.reason}`,
    );
  }
  return lines;
}

function renderTraining(
  training: TrainingReport | null,
  inArtifact: boolean,
): string[] {
  if (training === null) {
    return [
      inArtifact
        ? "  training     no cached dataset for this function (pass --cache-dir, or the cache was cleared)"
        : "  training     none: this function id has never been trained",
    ];
  }
  const dataset =
    training.datasetSha256 === null
      ? "the release's dataset is not cached; adversarial sidecar only"
      : `${training.release ? "the release's dataset" : "NOT the release's dataset (it is not cached)"} ${training.datasetSha256.slice(0, 12)}…`;
  const others =
    training.otherDatasets > 0
      ? `, ${String(training.otherDatasets)} other cached dataset${training.otherDatasets === 1 ? "" : "s"} for this id`
      : "";
  return [
    `  training cases nearest the input (${String(training.nearest.length)} of ${String(training.total)}; ${dataset}${training.adversarialPath === null ? "" : " plus its adversarial sidecar"}${others})`,
    ...training.nearest.map(
      (entry) =>
        `    ${entry.distance.toFixed(3)}  ${preview(entry.output)}  ${origin(entry)}  ${preview(entry.inputs)}`,
    ),
  ];
}

function origin(entry: {
  readonly origin: TrainingCase["origin"];
  readonly detail: string | null;
}): string {
  if (entry.origin === "gold") return "gold";
  if (entry.origin === "synthetic") {
    return entry.detail === null ? "synthetic" : `synthetic (${entry.detail})`;
  }
  return entry.detail === null ? "adversarial" : `adversarial ${entry.detail}`;
}

function preview(value: unknown): string {
  const text = value === undefined ? "undefined" : JSON.stringify(value);
  return text.length > INPUT_PREVIEW_CHARACTERS
    ? `${text.slice(0, INPUT_PREVIEW_CHARACTERS - 1)}…`
    : text;
}

function jsonForm(value: unknown): unknown {
  try {
    return JSON.parse(JSON.stringify(value)) as unknown;
  } catch {
    return null;
  }
}

function errorSummary(error: unknown): {
  readonly name: string;
  readonly message: string;
} {
  return error instanceof Error
    ? { name: error.name, message: error.message }
    : { name: "Error", message: String(error) };
}

function record(value: unknown): Readonly<Record<string, unknown>> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Readonly<Record<string, unknown>>)
    : {};
}

function list(value: unknown): readonly unknown[] {
  return Array.isArray(value) ? (value as readonly unknown[]) : [];
}
