import { dirname, join, relative, resolve } from "node:path";
import { parseArgs } from "node:util";

import { compileSemantScriptProgram } from "@semantscript/compiler";
import ts from "typescript";

import { CliUsageError, type CliIo } from "./io.js";

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
    },
    allowPositionals: false,
  });
  const application = values.application ?? "application";
  if (!APPLICATION_ID.test(application)) {
    throw new CliUsageError(
      "--application must be lowercase letters, digits and single dashes (for example refund-app)",
    );
  }
  const configPath = resolve(io.cwd, values.project ?? "tsconfig.json");
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
    return 1;
  }
  if (parsed.errors.length > 0) {
    io.stderr(formatDiagnostics(parsed.errors, io.cwd));
    return 1;
  }
  const outDir = parsed.options.outDir;
  if (outDir === undefined) {
    io.stderr(`${configPath} must set compilerOptions.outDir\n`);
    return 1;
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
    return 1;
  }
  const projectRoot = dirname(configPath);
  const bundlePath = resolve(
    io.cwd,
    values.bundle ?? join(outDir, "semantscript.ir.v1.json"),
  );
  const result = await compileSemantScriptProgram(program, {
    projectRoot,
    encoderRef: `encoder.${application}`,
    adapterRef: `adapter.${application}`,
    bundlePath,
  });
  if (!result.ok) {
    io.stderr(formatDiagnostics(result.diagnostics, io.cwd));
    return 1;
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
  lines.push(
    `execution plan: ${String(plan.bundle.executionPlan.stages.length)} stage(s)`,
    `emitted ${String(emittedFiles.length)} file(s) under ${relative(io.cwd, resolve(projectRoot, outDir)) || "."}`,
    `bundle: ${relative(io.cwd, result.value.bundlePath) || result.value.bundlePath}`,
  );
  io.stdout(`${lines.join("\n")}\n`);
  return 0;
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
