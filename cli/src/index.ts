#!/usr/bin/env node
import { realpathSync } from "node:fs";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { buildCommand } from "./build.js";
import { devCommand } from "./dev.js";
import { initCommand } from "./init.js";
import { CliUsageError, processIo, type CliIo } from "./io.js";
import { runCommand } from "./run.js";
import { testCommand } from "./test-command.js";
import { trainCommand } from "./train.js";

export const USAGE = `usage: semantscript <command> [options]

  init   [--tool next|vite|esbuild|tsc] [--no-example]
         wire the compiler into the project's build tool and add a starter expression
  build  [--project tsconfig.json] [--application <id>] [--bundle <path>]
         compile .sem.ts sites to runtime calls and one IR bundle
  train  [--bundle <path>] [--artifact <root>] [--teacher <teacher.toml>]
         [--cache-dir <dir>] [--report <path>] [--python <exe>] [--cases <n>]
         [--epochs <n>] [--batch-size <n>] [--learning-rate <x>] [--seed <n>]
         [--device <name>] [--select-best-epoch] [--ece-threshold <x>] ...
         generate data, train, verify and export an artifact from the bundle
  dev    [build and train options] [--debounce <ms>] [--once]
         build and train, then watch the sources and repeat on every save
  test   [--artifact <root>] [--bundle <path>] [--json]
         report each function's verification and replay the bundle's examples
  run    [--artifact <root>] <module.js> [--call <export>] [--input <json> | --input-file <path>]
         load the artifact, import the compiled module and call an export

  defaults: the bundle is the build's semantscript.ir.v1.json, the artifact is
  .semantscript/artifact (or SEMANTSCRIPT_ARTIFACT), the teacher is teacher.toml
  or the Anthropic backend when ANTHROPIC_API_KEY is set.
`;

export {
  buildCommand,
  devCommand,
  initCommand,
  runCommand,
  testCommand,
  trainCommand,
};
export { compileProject } from "./build.js";
export { detectBuildTool, type BuildTool } from "./init.js";
export {
  DEFAULT_ARTIFACT_PATH,
  resolveArtifactRoot,
  resolveBundlePath,
  resolveTeacherConfig,
} from "./defaults.js";
export { canonical } from "./test-command.js";
export { pythonPath, renderTrainReport } from "./train.js";
export { readArtifactSummary } from "./manifest.js";
export type { CliIo } from "./io.js";

const COMMANDS: Readonly<
  Record<string, (args: readonly string[], io: CliIo) => Promise<number>>
> = {
  init: initCommand,
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
