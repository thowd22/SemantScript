import { mkdir, readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { compileSemantScriptProgram } from "@semantscript/compiler";
import ts from "typescript";

const programDirectory = dirname(fileURLToPath(import.meta.url));
const defaultSourcePath = join(programDirectory, "refund-with-confidence.sem.ts");

/**
 * Compile one single-function sema program from this directory. The default
 * is the canonical refund decision; the companion risk function used by the
 * shared-encoder experiment compiles the same way under the same application
 * encoder and adapter refs, so both can share one artifact.
 */
export async function compileRefundProgram(outputDirectory, sourceFile = "refund-with-confidence.sem.ts") {
  if (typeof outputDirectory !== "string" || outputDirectory.length === 0) {
    throw new TypeError("outputDirectory must be a nonempty string");
  }
  if (typeof sourceFile !== "string" || !/^[a-z][a-z0-9-]*\.sem\.ts$/u.test(sourceFile)) {
    throw new TypeError("sourceFile must name a .sem.ts program in the program directory");
  }
  const sourcePath = sourceFile === "refund-with-confidence.sem.ts" ? defaultSourcePath : join(programDirectory, sourceFile);

  const outDir = resolve(outputDirectory);
  await mkdir(outDir, { recursive: true });
  const program = ts.createProgram({
    rootNames: [sourcePath],
    options: {
      declaration: true,
      inlineSources: true,
      module: ts.ModuleKind.NodeNext,
      moduleResolution: ts.ModuleResolutionKind.NodeNext,
      noEmitOnError: true,
      outDir,
      rootDir: programDirectory,
      sourceMap: true,
      strict: true,
      target: ts.ScriptTarget.ES2023,
    },
  });
  const result = await compileSemantScriptProgram(program, {
    projectRoot: programDirectory,
    bundlePath: join(outDir, "semantscript.ir.v1.json"),
    encoderRef: "encoder.refund-benchmark",
    adapterRef: "adapter.refund-benchmark",
  });

  if (!result.ok) {
    throw new Error(formatDiagnostics(result.diagnostics));
  }

  const [record] = result.value.plan.bundle.functions;
  if (record === undefined || result.value.plan.bundle.functions.length !== 1) {
    throw new Error("refund benchmark source must compile to exactly one neural function");
  }

  return {
    bundlePath: result.value.bundlePath,
    bundleText: result.value.plan.bundleText,
    emittedFiles: result.value.emittedFiles,
    functionId: record.id,
    semanticSha256: record.semanticSha256,
  };
}

function formatDiagnostics(diagnostics) {
  const host = {
    getCanonicalFileName: (fileName) => fileName,
    getCurrentDirectory: () => programDirectory,
    getNewLine: () => "\n",
  };
  return ts.formatDiagnosticsWithColorAndContext(diagnostics, host);
}

async function main() {
  const outputDirectory = process.argv[2];
  const sourceFile = process.argv[3];
  if (outputDirectory === undefined) {
    throw new Error("usage: node compile-program.mjs <output-directory>");
  }
  const compiled = sourceFile === undefined
    ? await compileRefundProgram(outputDirectory)
    : await compileRefundProgram(outputDirectory, sourceFile);
  const exactBundle = await readFile(compiled.bundlePath, "utf8");
  if (exactBundle !== compiled.bundleText) {
    throw new Error("emitted bundle bytes differ from the compiler plan");
  }
  process.stdout.write(
    `${JSON.stringify({
      bundlePath: compiled.bundlePath,
      functionId: compiled.functionId,
      semanticSha256: compiled.semanticSha256,
    })}\n`,
  );
}

const invokedPath = process.argv[1];
if (invokedPath !== undefined && import.meta.url === pathToFileURL(invokedPath).href) {
  await main();
}
