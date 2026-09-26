#!/usr/bin/env node
import { realpathSync } from "node:fs";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { buildCommand } from "./build.js";
import { devCommand } from "./dev.js";
import { doctorCommand } from "./doctor.js";
import { initCommand } from "./init.js";
import { CliUsageError, processIo, type CliIo } from "./io.js";
import { packageCommand } from "./package.js";
import { releasesCommand } from "./releases.js";
import { runCommand } from "./run.js";
import { teacherCommand } from "./teacher.js";
import { testCommand } from "./test-command.js";
import { trainCommand } from "./train.js";

export const USAGE = `usage: semantscript <command> [options]

  init   [--tool next|vite|esbuild|tsc] [--no-example] [--no-doctor]
         [--teacher anthropic|openrouter|ollama|constraints] [--teacher-model <id>]
         [--python <exe>] [--trainer-module <module>]
         wire the compiler into the project's build tool, add a starter expression
         and write .semantscript/teacher.toml (no key), then check the environment
  doctor [--python <exe>] [--teacher <teacher.toml>|constraints] [--probe request|free|none]
         [--device auto|cpu|cuda] [--trainer-module <module>] [--no-teacher]
         [--runtime] [--json]
         check Node, the native bindings, Python, the trainer, PyTorch and its
         device, ONNX Runtime and the teacher, one line each with the fix
  build  [--project tsconfig.json] [--application <id>] [--bundle <path>]
         [--route-domains] [--domain-depth <name>=<layers>]...
         compile .sem.ts sites to runtime calls and one IR bundle
  train  [--bundle <path>] [--artifact <root>] [--teacher <teacher.toml>|constraints]
         [--cache-dir <dir>] [--report <path>] [--python <exe>] [--cases <n>]
         [--epochs <n>] [--batch-size <n>] [--learning-rate <x>] [--seed <n>]
         [--device <name>] [--select-best-epoch] [--ece-threshold <x>]
         [--estimate] [--max-cost-usd <x>] [--no-preflight] ...
         check the Python side and the teacher, then generate data, train,
         verify and export an artifact from the bundle; --estimate prints the
         teacher's requests, tokens, USD and time without calling it, and
         --max-cost-usd stops before the request that would pass the cap
  teacher probe [--teacher <teacher.toml>|constraints] [--python <exe>] [--json]
         send one small request through the teacher and report the model,
         latency, tokens and cost
  dev    [build and train options] [--debounce <ms>] [--once]
         build and train, then watch the sources and repeat on every save
  test   [--artifact <root>] [--bundle <path>] [--json]
         report each function's verification and replay the bundle's examples
  run    [--artifact <root>] <module.js> [--call <export>] [--input <json> | --input-file <path>]
         load the artifact, import the compiled module and call an export
  releases [list] [--artifact <root>] [--json]
         list every release under the artifact root: date, digest, per-function
         verification and which one current.json names
  releases show <release> [--artifact <root>] [--json]
         one release's per-function verification
  releases rollback [<release>] [--artifact <root>] [--dry-run] [--json]
  releases promote <release> [--artifact <root>] [--dry-run] [--json]
         verify a release and point current.json at it atomically (rollback
         without a name picks the release created before the current one);
         a process loaded with watch: true swaps to it
  releases prune [--keep <n>] [--older-than <30d|12h|90m>] [--artifact <root>]
         [--dry-run] [--json]
         remove releases beyond the n newest and/or older than the age; never
         the current one
  package [--project <dir>] [--dist <dir>] [--artifact <root>] [--out <dir>]
         [--include <path>]... [--target lambda-zip|lambda-image|cloud-run-functions
         | --max-bytes <n>] [--platform <os>] [--arch <cpu>] [--force] [--json]
         write a deployable directory: dist/ without the IR bundle, production
         node_modules pruned to the platform's native bindings, the current
         release only and a manifest of digests; print its size by part and,
         over the target, which size lever would fit it

  defaults: the bundle is the build's semantscript.ir.v1.json, the artifact is
  .semantscript/artifact (or SEMANTSCRIPT_ARTIFACT), the teacher is teacher.toml
  or the Anthropic backend when ANTHROPIC_API_KEY is set (--teacher constraints
  labels with the expressions' own constraints, no key), and the Python
  interpreter is --python, else SEMANTSCRIPT_PYTHON, else python3 (python on
  Windows).
`;

export {
  buildCommand,
  devCommand,
  doctorCommand,
  initCommand,
  packageCommand,
  releasesCommand,
  runCommand,
  teacherCommand,
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
  teacherToml,
  TEACHER_CHOICES,
  type TeacherChoice,
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
export { renderEstimate, renderTrainReport } from "./train.js";
export {
  parseTeacherProbe,
  renderTeacherProbe,
  type TeacherProbeResult,
} from "./teacher.js";
export {
  pointerBytes,
  readArtifactSummary,
  readPointer,
  ReleaseError,
  writePointer,
  type ArtifactPointer,
} from "./manifest.js";
export { resolveRelease, scanReleases, type ReleaseEntry } from "./releases.js";
export {
  MEASURED_ENCODER,
  PACKAGE_KIND,
  PACKAGE_MANIFEST,
  PACKAGE_TARGETS,
  PackageError,
  packageLevers,
  pruneNativeBindings,
  type PackageLever,
  type PackageTarget,
} from "./package.js";
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
  teacher: teacherCommand,
  releases: releasesCommand,
  package: packageCommand,
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
