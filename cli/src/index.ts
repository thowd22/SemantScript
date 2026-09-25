#!/usr/bin/env node
import { realpathSync } from "node:fs";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { buildCommand } from "./build.js";
import { devCommand } from "./dev.js";
import { doctorCommand } from "./doctor.js";
import { initCommand } from "./init.js";
import { CliUsageError, processIo, type CliIo } from "./io.js";
import { runCommand } from "./run.js";
import { testCommand } from "./test-command.js";
import { trainCommand } from "./train.js";

export const USAGE = `usage: semantscript <command> [options]

  init   [--tool next|vite|esbuild|tsc] [--no-example] [--no-doctor]
         [--python <exe>] [--trainer-module <module>]
         wire the compiler into the project's build tool and add a starter expression,
         then check the environment
  doctor [--python <exe>] [--teacher <teacher.toml>] [--probe request|free|none]
         [--device auto|cpu|cuda] [--trainer-module <module>] [--no-teacher]
         [--runtime] [--json]
         check Node, the native bindings, Python, the trainer, PyTorch and its
         device, ONNX Runtime and the teacher, one line each with the fix
  build  [--project tsconfig.json] [--application <id>] [--bundle <path>]
         [--route-domains] [--domain-depth <name>=<layers>]...
         compile .sem.ts sites to runtime calls and one IR bundle
  train  [--bundle <path>] [--artifact <root>] [--teacher <teacher.toml>]
         [--cache-dir <dir>] [--report <path>] [--python <exe>] [--cases <n>]
         [--epochs <n>] [--batch-size <n>] [--learning-rate <x>] [--seed <n>]
         [--device <name>] [--select-best-epoch] [--ece-threshold <x>]
         [--no-preflight] ...
         check the Python side and the teacher, then generate data, train,
         verify and export an artifact from the bundle
  dev    [build and train options] [--debounce <ms>] [--once]
         build and train, then watch the sources and repeat on every save
  test   [--artifact <root>] [--bundle <path>] [--json]
         report each function's verification and replay the bundle's examples
  run    [--artifact <root>] <module.js> [--call <export>] [--input <json> | --input-file <path>]
         load the artifact, import the compiled module and call an export

  defaults: the bundle is the build's semantscript.ir.v1.json, the artifact is
  .semantscript/artifact (or SEMANTSCRIPT_ARTIFACT), the teacher is teacher.toml
  or the Anthropic backend when ANTHROPIC_API_KEY is set, and the Python
  interpreter is --python, else SEMANTSCRIPT_PYTHON, else python3.
`;

export {
  buildCommand,
  devCommand,
  doctorCommand,
  initCommand,
  runCommand,
  testCommand,
  trainCommand,
};
export { compileProject } from "./build.js";
export { detectBuildTool, type BuildTool } from "./init.js";
export {
  DEFAULT_ARTIFACT_PATH,
  findTeacherConfig,
  pythonPath,
  resolveArtifactRoot,
  resolveBundlePath,
  resolvePython,
  resolveTeacherConfig,
} from "./defaults.js";
export {
  checkNode,
  checkRuntimeBindings,
  parseDoctorReport,
  renderChecks,
  runPythonDoctor,
  type DoctorCheck,
} from "./doctor.js";
export { canonical } from "./test-command.js";
export { renderTrainReport } from "./train.js";
export { readArtifactSummary } from "./manifest.js";
export type { CliIo } from "./io.js";

const COMMANDS: Readonly<
  Record<string, (args: readonly string[], io: CliIo) => Promise<number>>
> = {
  init: initCommand,
  doctor: doctorCommand,
  build: buildCommand,
  train: trainCommand,
  dev: devCommand,
  test: testCommand,
  run: runCommand,
};

/** Dispatch one invocation; returns the process exit status. */
export async function runCli(
  argv: readonly string[],
  io: CliIo = processIo(),
): Promise<number> {
  const [command, ...rest] = argv;
  if (
    command === undefined ||
    command === "--help" ||
    command === "-h" ||
    command === "help"
  ) {
    io.stdout(USAGE);
    return command === undefined ? 2 : 0;
  }
  if (
    Object.hasOwn(COMMANDS, command) &&
    (rest.includes("--help") || rest.includes("-h"))
  ) {
    io.stdout(USAGE);
    return 0;
  }
  const handler = Object.hasOwn(COMMANDS, command)
    ? COMMANDS[command]
    : undefined;
  if (handler === undefined) {
    io.stderr(`unknown command ${command}\n${USAGE}`);
    return 2;
  }
  try {
    return await handler(rest, io);
  } catch (error: unknown) {
    if (error instanceof CliUsageError) {
      io.stderr(`${error.message}\n${USAGE}`);
      return 2;
    }
    if (
      error instanceof Error &&
      error.name === "TypeError" &&
      "code" in error
    ) {
      // node:util parseArgs reports unknown or malformed options as TypeErrors.
      io.stderr(`${error.message}\n${USAGE}`);
      return 2;
    }
    io.stderr(`${error instanceof Error ? error.message : String(error)}\n`);
    return 1;
  }
}

function invokedDirectly(): boolean {
  const entry = process.argv[1];
  if (entry === undefined) return false;
  try {
    return realpathSync(entry) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (invokedDirectly()) {
  runCli(process.argv.slice(2)).then(
    (code) => {
      process.exitCode = code;
    },
    (error: unknown) => {
      process.stderr.write(
        `${error instanceof Error ? error.message : String(error)}\n`,
      );
      process.exitCode = 1;
    },
  );
}
