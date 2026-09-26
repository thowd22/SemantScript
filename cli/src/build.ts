import { dirname, join, relative, resolve } from "node:path";
import { parseArgs } from "node:util";

import { compileSemantScriptProgram } from "@semantscript/compiler";
import ts from "typescript";

import { CliUsageError, type CliIo } from "./io.js";
import { remedyText } from "./remedy.js";

const APPLICATION_ID = /^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/u;

/**
 * `semantscript build`: compile every `.sem.ts` site of a TypeScript project
 * into runtime calls plus one IR bundle, under the application's encoder and
 * adapter refs.
 */
export async function buildCommand(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const { values } = parseArgs({
    args: [...args],
    options: {
      project: { type: "string", short: "p" },
      application: { type: "string" },
      bundle: { type: "string" },
      "domain-depth": { type: "string", multiple: true },
      "route-domains": { type: "boolean" },
    },
    allowPositionals: false,
  });
  const domainDepths = parseDomainDepths(values["domain-depth"] ?? []);
  const result = await compileProject(
    {
      ...(values.project === undefined ? {} : { project: values.project }),
      ...(values.application === undefined
        ? {}
        : { application: values.application }),
      ...(values.bundle === undefined ? {} : { bundle: values.bundle }),
      ...(domainDepths === undefined ? {} : { domainDepths }),
      ...(values["route-domains"] === true ? { routeDomains: true } : {}),
    },
    io,
  );
  return result.status;
}

/** `--domain-depth name=n`, repeatable: the shared-encoder layers a domain runs. */
export function parseDomainDepths(
  entries: readonly string[],
): Readonly<Record<string, number>> | undefined {
  if (entries.length === 0) return undefined;
  const depths: Record<string, number> = {};
  for (const entry of entries) {
    const separator = entry.indexOf("=");
    const name = separator < 0 ? "" : entry.slice(0, separator);
    const depth = Number(entry.slice(separator + 1));
    if (
      !/^[a-z][a-z0-9-]*$/u.test(name) ||
      !Number.isInteger(depth) ||
      depth < 1
    ) {
      throw new CliUsageError(
        `--domain-depth expects name=<positive integer> with a lowercase name, not ${JSON.stringify(entry)}`,
      );
    }
    depths[name] = depth;
  }
  return depths;
}

export interface CompileProjectOptions {
  readonly project?: string;
  readonly application?: string;
  readonly bundle?: string;
  readonly domainDepths?: Readonly<Record<string, number>>;
  readonly routeDomains?: boolean;
}

export interface CompileProjectResult {
  readonly status: number;
  readonly projectRoot: string;
  /** The project's TypeScript sources, for watchers. */
  readonly sourceFiles: readonly string[];
  readonly bundlePath?: string;
}

/** The build behind `semantscript build` and `dev`: prints the summary or diagnostics. */
export async function compileProject(
  options: CompileProjectOptions,
  io: CliIo,
): Promise<CompileProjectResult> {
  const application = options.application ?? "application";
  if (!APPLICATION_ID.test(application)) {
    throw new CliUsageError(
      "--application must be lowercase letters, digits and single dashes (for example refund-app)",
    );
  }
  const configPath = resolve(io.cwd, options.project ?? "tsconfig.json");
  const projectRoot = dirname(configPath);
  const failed = (
    sourceFiles: readonly string[] = [],
  ): CompileProjectResult => ({
    status: 1,
    projectRoot,
    sourceFiles,
  });
  const configDiagnostics: ts.Diagnostic[] = [];
  const host: ts.ParseConfigFileHost = {
    ...ts.sys,
    onUnRecoverableConfigFileDiagnostic: (diagnostic) => {
      configDiagnostics.push(diagnostic);
    },
  };
  const parsed = ts.getParsedCommandLineOfConfigFile(configPath, {}, host);
  if (parsed === undefined || configDiagnostics.length > 0) {
    io.stderr(
      configDiagnostics.length > 0
        ? formatDiagnostics(configDiagnostics, io.cwd)
        : `unable to read ${configPath}\n`,
    );
    io.stderr(`next: ${await remedyText("build-no-tsconfig")}\n`);
    return failed();
  }
  if (parsed.errors.length > 0) {
    io.stderr(formatDiagnostics(parsed.errors, io.cwd));
    return failed();
  }
  const sourceFiles = parsed.fileNames.map((fileName) => resolve(fileName));
  const outDir = parsed.options.outDir;
  if (outDir === undefined) {
    io.stderr(
      `${configPath} must set compilerOptions.outDir; next: ${await remedyText("build-no-outdir", { config: configPath })}\n`,
    );
    return failed(sourceFiles);
  }
  const program = ts.createProgram({
    rootNames: parsed.fileNames,
    options: parsed.options,
    ...(parsed.projectReferences === undefined
      ? {}
      : { projectReferences: parsed.projectReferences }),
  });
  const preEmit = ts.getPreEmitDiagnostics(program);
  if (preEmit.length > 0) {
    io.stderr(formatDiagnostics(preEmit, io.cwd));
    return failed(sourceFiles);
  }
  const bundlePath = resolve(
    io.cwd,
    options.bundle ?? join(outDir, "semantscript.ir.v1.json"),
  );
  const result = await compileSemantScriptProgram(program, {
    projectRoot,
    encoderRef: `encoder.${application}`,
    adapterRef: `adapter.${application}`,
    bundlePath,
    ...(options.domainDepths === undefined
      ? {}
      : { domainDepths: options.domainDepths }),
    ...(options.routeDomains === true ? { routeDomains: true } : {}),
  });
  if (!result.ok) {
    io.stderr(formatDiagnostics(result.diagnostics, io.cwd));
    return failed(sourceFiles);
  }
  const { plan, emittedFiles } = result.value;
  const lines = [
    `compiled ${String(plan.bundle.functions.length)} neural function(s) for application ${application}`,
  ];
  for (const record of plan.bundle.functions) {
    const heads = record.model.heads.length;
    lines.push(
      `  ${record.id}  ${record.source.path}:${String(record.source.line)}:${String(record.source.column)}  ${record.output.tsType}  (${String(heads)} head${heads === 1 ? "" : "s"})`,
    );
  }
  const domains = plan.bundle.executionPlan.domains;
  if (domains !== undefined) {
    lines.push(
      `domains: ${domains
        .map(
          (domain) =>
            `${domain.name} (${String(domain.functionIds.length)} function${domain.functionIds.length === 1 ? "" : "s"}, ${domain.encoderDepth === null ? "full depth" : `depth ${String(domain.encoderDepth)}`})`,
        )
        .join(", ")}`,
    );
  }
  lines.push(
    `execution plan: ${String(plan.bundle.executionPlan.stages.length)} stage(s)`,
    `emitted ${String(emittedFiles.length)} file(s) under ${relative(io.cwd, resolve(projectRoot, outDir)) || "."}`,
    `bundle: ${relative(io.cwd, result.value.bundlePath) || result.value.bundlePath}`,
  );
  io.stdout(`${lines.join("\n")}\n`);
  return {
    status: 0,
    projectRoot,
    sourceFiles,
    bundlePath: result.value.bundlePath,
  };
}

function formatDiagnostics(
  diagnostics: readonly ts.Diagnostic[],
  cwd: string,
): string {
  return ts.formatDiagnostics(diagnostics, {
    getCanonicalFileName: (fileName) => fileName,
    getCurrentDirectory: () => cwd,
    getNewLine: () => "\n",
  });
}
