import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, relative } from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

import {
  DEFAULT_TEACHER_MODEL,
  DEFAULT_TRAINER_MODULE,
  PYTHON_ENVIRONMENT_VARIABLE,
  findTeacherConfig,
  pythonPath,
  resolvePython,
} from "./defaults.js";
import {
  CliUsageError,
  listOf,
  objectOf,
  stringOf,
  stringOption,
  type CliIo,
  type OptionValues,
} from "./io.js";

export const DOCTOR_REPORT_KIND = "semantscript.doctor-report";
export const DOCTOR_REPORT_VERSION = 1;
/** The oldest Node release the workspace supports (package.json `engines`). */
export const MINIMUM_NODE_VERSION = "22.13.0";
export const NODE_CHECK_IDS = ["node", "runtime-bindings"] as const;
/** The checks the Python side reports, in order (semantscript_trainer/doctor.py). */
export const PYTHON_CHECK_IDS = [
  "python",
  "trainer",
  "model",
  "torch",
  "device",
  "onnxruntime",
  "platform-env",
  "teacher-config",
  "teacher-key",
  "teacher-probe",
] as const;
const CHECK_STATUSES = ["pass", "fail", "warn", "skip"] as const;
const PROBE_MODES = ["request", "free", "none"] as const;
const NATIVE_PACKAGES = ["onnxruntime-node", "tokenizers"] as const;
/** The oldest Python the trainer imports on (it uses PEP 695 `type` statements). */
export const MINIMUM_PYTHON_VERSION = "3.12";
const PYTHON_FIX =
  "install Python 3.12 or later, or point the CLI at one with --python <exe> or SEMANTSCRIPT_PYTHON";
/** Checks whose fix is to install into, or pick, another interpreter. */
const INTERPRETER_CHECK_IDS: readonly string[] = [
  "python",
  "trainer",
  "model",
  "torch",
  "onnxruntime",
];

export type CheckStatus = (typeof CHECK_STATUSES)[number];
export type ProbeMode = (typeof PROBE_MODES)[number];

/** One line of the doctor report. */
export interface DoctorCheck {
  readonly id: string;
  readonly status: CheckStatus;
  readonly summary: string;
  readonly fix: string | null;
}

export interface PythonDoctorOptions {
  readonly python: string;
  readonly trainerModule: string;
  /** The teacher file to check; undefined with `defaultTeacherModel` checks the generated default. */
  readonly teacher?: string;
  readonly defaultTeacherModel?: string;
  readonly checkTeacher: boolean;
  readonly probe: ProbeMode;
  readonly device?: string;
  /** Retry imports under other environment variables only when one fails (the train preflight). */
  readonly quick: boolean;
}

/**
 * `semantscript doctor`: check Node, the native runtime bindings, the Python
 * interpreter, the trainer, PyTorch and its device, ONNX Runtime, the platform
 * environment and the teacher, one line each with the fix. Exits 1 when any
 * check fails.
 */
export async function doctorCommand(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const { values } = parseArgs({
    args: [...args],
    options: {
      python: { type: "string" },
      "trainer-module": { type: "string" },
      teacher: { type: "string" },
      probe: { type: "string" },
      device: { type: "string" },
      "no-teacher": { type: "boolean" },
      runtime: { type: "boolean" },
      json: { type: "boolean" },
    },
    allowPositionals: false,
  });
  const probe = parseProbe(stringOption(values, "probe") ?? "request");
  const checks = await collectChecks(values, io, {
    probe,
    quick: false,
    runtimeOnly: values.runtime === true,
  });
  if (values.json === true) {
    io.stdout(
      `${JSON.stringify(
        {
          kind: DOCTOR_REPORT_KIND,
          reportVersion: DOCTOR_REPORT_VERSION,
          checks,
        },
        null,
        2,
      )}\n`,
    );
  } else {
    io.stdout(`semantscript doctor: ${io.cwd}\n${renderChecks(checks)}`);
  }
  return checks.some((check) => check.status === "fail") ? 1 : 0;
}

/** Every check `doctor` (and `init`) runs, Node first, then the Python side. */
export async function collectChecks(
  values: OptionValues,
  io: CliIo,
  options: {
    readonly probe: ProbeMode;
    readonly quick: boolean;
    readonly runtimeOnly?: boolean;
  },
): Promise<readonly DoctorCheck[]> {
  const checks: DoctorCheck[] = [checkNode(), await checkRuntimeBindings()];
  if (options.runtimeOnly === true) return checks;
  const teacher = findTeacherConfig(values, io);
  const anthropicKey = io.env["ANTHROPIC_API_KEY"];
  const python = resolvePython(values, io);
  const pythonChecks = await runPythonDoctor(
    {
      python,
      trainerModule:
        stringOption(values, "trainer-module") ?? DEFAULT_TRAINER_MODULE,
      ...(teacher !== undefined
        ? { teacher }
        : anthropicKey !== undefined && anthropicKey.length > 0
          ? { defaultTeacherModel: DEFAULT_TEACHER_MODEL }
          : {}),
      checkTeacher: values["no-teacher"] !== true,
      probe: options.probe,
      ...(stringOption(values, "device") === undefined
        ? {}
        : { device: stringOption(values, "device") ?? "auto" }),
      quick: options.quick,
    },
    io,
  );
  const venv = unusedVirtualEnvironment(values, io);
  checks.push(
    ...(venv === undefined
      ? pythonChecks
      : withVirtualEnvironmentHint(pythonChecks, python, venv)),
  );
  return checks;
}

/**
 * A `.venv` interpreter near `cwd` when neither `--python` nor
 * SEMANTSCRIPT_PYTHON chose the interpreter, else undefined.
 */
export function unusedVirtualEnvironment(
  values: OptionValues,
  io: CliIo,
): string | undefined {
  const defaulted =
    stringOption(values, "python") === undefined &&
    (io.env[PYTHON_ENVIRONMENT_VARIABLE] ?? "") === "";
  const found = defaulted ? findVirtualEnvironment(io.cwd) : undefined;
  return found === undefined ? undefined : relative(io.cwd, found);
}

/**
 * The interpreter of a `.venv` in `cwd` or one of its parents, if there is
 * one. The CLI does not pick it up by itself (an activated venv puts its own
 * python3 first on PATH), so doctor names it when the default interpreter
 * lacks the packages.
 */
export function findVirtualEnvironment(cwd: string): string | undefined {
  const relative =
    process.platform === "win32"
      ? join(".venv", "Scripts", "python.exe")
      : join(".venv", "bin", "python");
  let directory = cwd;
  for (;;) {
    const candidate = join(directory, relative);
    if (existsSync(candidate)) return candidate;
    const parent = dirname(directory);
    if (parent === directory) return undefined;
    directory = parent;
  }
}

/** Append "use the .venv interpreter" to every failed interpreter-level fix. */
export function withVirtualEnvironmentHint(
  checks: readonly DoctorCheck[],
  python: string,
  venv: string,
): readonly DoctorCheck[] {
  const hint = `if the packages are in ${venv} rather than ${python} (the default interpreter), pass --python ${venv} or set the SEMANTSCRIPT_PYTHON environment variable to ${venv}`;
  return checks.map((check) =>
    check.status === "fail" && INTERPRETER_CHECK_IDS.includes(check.id)
      ? {
          ...check,
          fix: check.fix === null ? hint : `${check.fix}; or ${hint}`,
        }
      : check,
  );
}

/**
 * The train preflight: the Python and teacher checks without a billed request.
 * Prints the lines to stderr and returns false when any check fails.
 */
export async function preflight(
  options: {
    readonly python: string;
    readonly trainerModule: string;
    readonly teacher: string;
    readonly device?: string;
    /** A `.venv` interpreter the default interpreter may be missing (see unusedVirtualEnvironment). */
    readonly virtualEnvironment?: string;
  },
  io: CliIo,
  command = "train",
): Promise<boolean> {
  const started = Date.now();
  const { virtualEnvironment, ...doctorOptions } = options;
  const reported = await runPythonDoctor(
    { ...doctorOptions, checkTeacher: true, probe: "free", quick: true },
    io,
  );
  const checks =
    virtualEnvironment === undefined
      ? reported
      : withVirtualEnvironmentHint(
          reported,
          options.python,
          virtualEnvironment,
        );
  const seconds = ((Date.now() - started) / 1000).toFixed(1);
  const failed = checks.some((check) => check.status === "fail");
  io.stderr(
    `semantscript ${command}: environment preflight (${seconds} s)\n${renderChecks(checks)}`,
  );
  if (failed) {
    io.stderr(
      `semantscript ${command}: stopped before training; fix the failed checks above (semantscript doctor explains each one) or pass --no-preflight\n`,
    );
  }
  return !failed;
}

/** `node`: the running Node release against the workspace's `engines` floor. */
export function checkNode(
  version: string = process.versions.node,
): DoctorCheck {
  const summary = `Node ${version} (${process.platform}-${process.arch})`;
  return compareVersions(version, MINIMUM_NODE_VERSION) >= 0
    ? { id: "node", status: "pass", summary, fix: null }
    : {
        id: "node",
        status: "fail",
        summary: `${summary} is older than ${MINIMUM_NODE_VERSION}`,
        fix: `install Node ${MINIMUM_NODE_VERSION} or later (the repository's .nvmrc pins the tested release; nvm install)`,
      };
}

/**
 * `runtime-bindings`: resolve onnxruntime-node and tokenizers the way the
 * runtime does (from `@semantscript/core`) and import them, which loads their
 * native binaries for this platform.
 */
export async function checkRuntimeBindings(
  from?: string,
): Promise<DoctorCheck> {
  const platform = `${process.platform}-${process.arch}`;
  let base: string;
  try {
    base = from ?? import.meta.resolve("@semantscript/core");
  } catch (error: unknown) {
    return {
      id: "runtime-bindings",
      status: "fail",
      summary: `@semantscript/core does not resolve: ${message(error)}`,
      fix: "npm install in the project (the CLI loads the runtime from @semantscript/core)",
    };
  }
  const require = createRequire(base);
  const loaded: string[] = [];
  for (const name of NATIVE_PACKAGES) {
    try {
      const path = require.resolve(name);
      await import(pathToFileURL(path).href);
      loaded.push(`${name} ${packageVersion(path, name)}`);
    } catch (error: unknown) {
      return {
        id: "runtime-bindings",
        status: "fail",
        summary: `${name} does not load on ${platform}: ${firstLine(message(error))}`,
        fix: `run npm install (or npm rebuild ${NATIVE_PACKAGES.join(" ")}) on this machine; the native binaries are per platform, so a node_modules copied from another OS or architecture does not load`,
      };
    }
  }
  return {
    id: "runtime-bindings",
    status: "pass",
    summary: `${loaded.join(" and ")} loaded for ${platform}`,
    fix: null,
  };
}

/**
 * Spawn `<python> -m <trainer-module> doctor --json` with the trainer's
 * PYTHONPATH and validate its report. An interpreter or trainer that does not
 * start becomes a failed check; a report that breaks the contract throws.
 */
export async function runPythonDoctor(
  options: PythonDoctorOptions,
  io: CliIo,
): Promise<readonly DoctorCheck[]> {
  const args = ["-m", options.trainerModule, "doctor", "--json"];
  args.push("--probe", options.probe);
  if (options.quick) args.push("--quick");
  if (!options.checkTeacher) args.push("--no-teacher");
  if (options.teacher !== undefined) args.push("--teacher", options.teacher);
  else if (options.defaultTeacherModel !== undefined)
    args.push("--default-teacher-model", options.defaultTeacherModel);
  if (options.device !== undefined) args.push("--device", options.device);

  const outcome = await capture(options.python, args, io);
  if (outcome.error !== undefined) {
    return withSkipped(
      {
        id: "python",
        status: "fail",
        summary: `unable to run ${options.python}: ${outcome.error.message}`,
        fix: PYTHON_FIX,
      },
      "the interpreter does not start",
    );
  }
  const reportLine = outcome.stdout.trim().split("\n").at(-1) ?? "";
  if ((outcome.status === 0 || outcome.status === 1) && reportLine !== "") {
    let document: unknown;
    try {
      document = JSON.parse(reportLine);
    } catch {
      throw new Error(
        `${options.python} -m ${options.trainerModule} doctor printed no JSON report: ${firstLine(reportLine)}`,
      );
    }
    return parseDoctorReport(document);
  }
  // The trainer did not start: say which piece is missing.
  const stderrTail =
    outcome.stderr.trim().split("\n").at(-1) || "no error output";
  const version = await capture(
    options.python,
    ["-c", "import sys; print(sys.version.split()[0])"],
    io,
  );
  const found = version.stdout.trim();
  const pythonCheck: DoctorCheck =
    version.status === 0
      ? compareVersions(found, MINIMUM_PYTHON_VERSION) >= 0
        ? {
            id: "python",
            status: "pass",
            summary: `Python ${found} (${options.python})`,
            fix: null,
          }
        : {
            id: "python",
            status: "fail",
            summary: `Python ${found} (${options.python}) is older than ${MINIMUM_PYTHON_VERSION}, which the trainer needs`,
            fix: PYTHON_FIX,
          }
      : {
          id: "python",
          status: "fail",
          summary: `${options.python} does not run a Python program: ${stderrTail}`,
          fix: PYTHON_FIX,
        };
  const moduleRoot = options.trainerModule.split(".")[0] ?? "";
  const missingModule = new RegExp(
    `No module named '?${moduleRoot.replaceAll(/[.*+?^${}()|[\]\\]/gu, "\\$&")}`,
    "u",
  ).test(outcome.stderr);
  const noDoctor = /invalid choice: 'doctor'/u.test(outcome.stderr);
  const trainerCheck: DoctorCheck = missingModule
    ? {
        id: "trainer",
        status: "fail",
        summary: `${options.python} cannot import ${options.trainerModule}: ${stderrTail}`,
        fix: `pip install -e ".[training]" from the SemantScript checkout into ${options.python}, or run the CLI from the checkout (it adds trainer/src and model/src to PYTHONPATH)`,
      }
    : noDoctor
      ? {
          id: "trainer",
          status: "fail",
          summary: `${options.trainerModule} has no doctor command`,
          fix: "update the SemantScript trainer to the release that matches this CLI, or pass --no-preflight to train without the checks",
        }
      : {
          id: "trainer",
          status: "fail",
          summary: `${options.python} -m ${options.trainerModule} doctor exited with status ${String(outcome.status)}: ${stderrTail}`,
          fix: `run ${options.python} -m ${options.trainerModule} doctor to see the full error`,
        };
  return withSkipped(
    pythonCheck,
    pythonCheck.status === "fail"
      ? "the interpreter cannot run the trainer"
      : "the trainer does not start",
    pythonCheck.status === "fail" ? undefined : trainerCheck,
  );
}

/** Validate the Python side's closed report. */
export function parseDoctorReport(document: unknown): readonly DoctorCheck[] {
  const report = objectOf(document, "doctor report");
  if (report["kind"] !== DOCTOR_REPORT_KIND) {
    throw new Error(`doctor report.kind must be ${DOCTOR_REPORT_KIND}`);
  }
  if (report["reportVersion"] !== DOCTOR_REPORT_VERSION) {
    throw new Error(
      `doctor report.reportVersion must be ${String(DOCTOR_REPORT_VERSION)}`,
    );
  }
  const checks = listOf(report["checks"], "doctor report.checks").map(
    (entry, index): DoctorCheck => {
      const path = `doctor report.checks[${String(index)}]`;
      const check = objectOf(entry, path);
      const extra = Object.keys(check).filter(
        (key) => !["id", "status", "summary", "fix"].includes(key),
      );
      if (extra.length > 0) {
        throw new Error(`${path} has unknown keys: ${extra.join(", ")}`);
      }
      const id = stringOf(check["id"], `${path}.id`);
      if (!(PYTHON_CHECK_IDS as readonly string[]).includes(id)) {
        throw new Error(
          `${path}.id ${JSON.stringify(id)} is not a known check`,
        );
      }
      const status = stringOf(check["status"], `${path}.status`);
      if (!(CHECK_STATUSES as readonly string[]).includes(status)) {
        throw new Error(
          `${path}.status must be one of ${CHECK_STATUSES.join(", ")}`,
        );
      }
      const fix = check["fix"];
      if (fix !== null && typeof fix !== "string") {
        throw new Error(`${path}.fix must be a string or null`);
      }
      return {
        id,
        status: status as CheckStatus,
        summary: stringOf(check["summary"], `${path}.summary`),
        fix,
      };
    },
  );
  const ids = checks.map((check) => check.id);
  if (new Set(ids).size !== ids.length) {
    throw new Error("doctor report.checks repeats a check id");
  }
  return checks;
}

/** One line per check, `fix:` under every line that did not pass, then the totals. */
export function renderChecks(checks: readonly DoctorCheck[]): string {
  const lines: string[] = [];
  for (const check of checks) {
    lines.push(
      `  ${check.status.padEnd(5)} ${check.id.padEnd(17)} ${check.summary}`,
    );
    if (check.status !== "pass" && check.fix !== null) {
      lines.push(`        fix: ${check.fix}`);
    }
  }
  const count = (status: CheckStatus): number =>
    checks.filter((check) => check.status === status).length;
  const warnings = count("warn");
  lines.push(
    `${String(count("pass"))} passed, ${String(warnings)} ${warnings === 1 ? "warning" : "warnings"}, ${String(count("fail"))} failed, ${String(count("skip"))} skipped`,
  );
  return `${lines.join("\n")}\n`;
}

function parseProbe(value: string): ProbeMode {
  if (!(PROBE_MODES as readonly string[]).includes(value)) {
    throw new CliUsageError(`--probe must be one of ${PROBE_MODES.join(", ")}`);
  }
  return value as ProbeMode;
}

function withSkipped(
  first: DoctorCheck,
  reason: string,
  second?: DoctorCheck,
): readonly DoctorCheck[] {
  const given = second === undefined ? [first] : [first, second];
  const skipped = PYTHON_CHECK_IDS.filter(
    (id) => !given.some((check) => check.id === id),
  ).map((id): DoctorCheck => ({
    id,
    status: "skip",
    summary: `not checked: ${reason}`,
    fix: null,
  }));
  return [...given, ...skipped];
}

function capture(
  command: string,
  args: readonly string[],
  io: CliIo,
): Promise<{
  readonly status: number;
  readonly stdout: string;
  readonly stderr: string;
  readonly error?: Error;
}> {
  return new Promise((resolvePromise) => {
    const out: Buffer[] = [];
    const err: Buffer[] = [];
    const child = spawn(command, args, {
      cwd: io.cwd,
      // The output is decoded as UTF-8 below; without this a piped Python
      // on Windows writes the locale code page (cp1252 and the like) and
      // fails on, or garbles, a non-ASCII user name or path.
      env: {
        ...io.env,
        PYTHONPATH: pythonPath(io.env["PYTHONPATH"]),
        PYTHONIOENCODING: "utf-8",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    const abort = (): void => {
      child.kill();
    };
    io.signal?.addEventListener("abort", abort, { once: true });
    child.stdout.on("data", (chunk: Buffer) => out.push(chunk));
    child.stderr.on("data", (chunk: Buffer) => err.push(chunk));
    child.once("error", (error) => {
      io.signal?.removeEventListener("abort", abort);
      resolvePromise({ status: 1, stdout: "", stderr: "", error });
    });
    child.once("close", (code) => {
      io.signal?.removeEventListener("abort", abort);
      resolvePromise({
        status: code ?? 1,
        stdout: Buffer.concat(out).toString("utf8"),
        stderr: Buffer.concat(err).toString("utf8"),
      });
    });
  });
}

function packageVersion(resolved: string, name: string): string {
  let directory = dirname(resolved);
  for (;;) {
    const manifest = join(directory, "package.json");
    if (existsSync(manifest)) {
      try {
        const parsed = JSON.parse(readFileSync(manifest, "utf8")) as {
          name?: unknown;
          version?: unknown;
        };
        if (parsed.name === name && typeof parsed.version === "string") {
          return parsed.version;
        }
      } catch {
        // keep walking up
      }
    }
    const parent = dirname(directory);
    if (parent === directory) return "(version unknown)";
    directory = parent;
  }
}

function compareVersions(left: string, right: string): number {
  const parse = (value: string): number[] =>
    value.split(".").map((part) => Number.parseInt(part, 10) || 0);
  const a = parse(left);
  const b = parse(right);
  for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
    const difference = (a[index] ?? 0) - (b[index] ?? 0);
    if (difference !== 0) return difference;
  }
  return 0;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function firstLine(text: string): string {
  return text.split("\n")[0] ?? "";
}
