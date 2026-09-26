import { spawn } from "node:child_process";
import { mkdir, stat, writeFile } from "node:fs/promises";
import { constants } from "node:os";
import { dirname, resolve } from "node:path";
import { StringDecoder } from "node:string_decoder";
import { parseArgs } from "node:util";

import { VERSION as COMPILER_VERSION } from "@semantscript/compiler";

import {
  DEFAULT_TRAINER_MODULE,
  pythonPath,
  resolveArtifactRoot,
  resolveBundlePath,
  resolvePython,
  resolveTeacherConfig,
} from "./defaults.js";
import { capture, preflight, unusedVirtualEnvironment } from "./doctor.js";
import {
  CliUsageError,
  listOf,
  numberOf,
  objectOf,
  stringOf,
  stringOption,
  type CliIo,
  type OptionValues,
} from "./io.js";
import { readJson } from "./manifest.js";
import { remedyText } from "./remedy.js";
import { formatRatio, renderTable, shortId } from "./table.js";

/** Options handed through to the Python driver unchanged. */
const PASSTHROUGH_STRING = [
  "cases",
  "application-id",
  "application-version",
  "compiler-version",
  "encoder-name",
  "encoder-revision",
  "epochs",
  "batch-size",
  "learning-rate",
  "max-sequence-length",
  "evaluation-ratio",
  "seed",
  "device",
  "head-architecture",
  "ece-threshold",
  "max-constraint-violation-rate",
  "counterfactual-ratio",
  "adapter-bottleneck-size",
  "max-cost-usd",
  "seed-attempts",
  "seed-retry-margin",
] as const;
const PASSTHROUGH_BOOLEAN = [
  "local-files-only",
  "select-best-epoch",
  "no-cache",
  "full",
] as const;

export const TRAIN_OPTIONS: Record<
  string,
  { readonly type: "string" | "boolean" }
> = {
  bundle: { type: "string" },
  artifact: { type: "string" },
  teacher: { type: "string" },
  "cache-dir": { type: "string" },
  report: { type: "string" },
  python: { type: "string" },
  "trainer-module": { type: "string" },
  "no-preflight": { type: "boolean" },
  estimate: { type: "boolean" },
  ...Object.fromEntries(
    PASSTHROUGH_STRING.map((name) => [name, { type: "string" }]),
  ),
  ...Object.fromEntries(
    PASSTHROUGH_BOOLEAN.map((name) => [name, { type: "boolean" }]),
  ),
};

/**
 * `semantscript train`: hand the IR bundle to the Python trainer, which
 * generates data, trains, verifies and exports, then render its report.
 */
export async function trainCommand(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const { values } = parseArgs({
    args: [...args],
    options: TRAIN_OPTIONS,
    allowPositionals: false,
  });
  return runTrain(values, io);
}

/**
 * The training run behind `semantscript train` and `dev`. Unless
 * `--no-preflight` is passed (or `options.preflight` is false), the doctor's
 * Python and teacher checks run first, without a billed teacher request, so a
 * missing interpreter, package, key or model stops the run in seconds.
 * `--estimate` prints what the teacher would cost and exits without the
 * preflight, the teacher or any training.
 */
export async function runTrain(
  values: OptionValues,
  io: CliIo,
  options: { readonly preflight?: boolean; readonly command?: string } = {},
): Promise<number> {
  const bundle = resolveBundlePath(values, io);
  const artifact = resolveArtifactRoot(values, io);
  const teacher = resolveTeacherConfig(values, io);
  const cacheDir = resolve(
    io.cwd,
    stringOption(values, "cache-dir") ?? ".semantscript/cache",
  );
  const report = resolve(
    io.cwd,
    stringOption(values, "report") ?? `${artifact}.report.json`,
  );
  const python = resolvePython(values, io);
  const trainerModule =
    stringOption(values, "trainer-module") ?? DEFAULT_TRAINER_MODULE;
  const maxCost = stringOption(values, "max-cost-usd");
  if (maxCost !== undefined) {
    const parsed = Number(maxCost);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      throw new CliUsageError(
        "--max-cost-usd must be a positive number of USD",
      );
    }
  }
  const estimate = values["estimate"] === true;
  const doctor = trainerDoctorCommand(values, trainerModule);
  // The trainer names this command in the fixes it prints itself.
  const trainerIo: CliIo = {
    ...io,
    env: { ...io.env, SEMANTSCRIPT_DOCTOR_COMMAND: doctor },
  };
  if (
    !estimate &&
    values["no-preflight"] !== true &&
    options.preflight !== false
  ) {
    const device = stringOption(values, "device");
    const virtualEnvironment = unusedVirtualEnvironment(values, io);
    const ready = await preflight(
      {
        python,
        trainerModule,
        teacher,
        ...(device === undefined ? {} : { device }),
        ...(virtualEnvironment === undefined ? {} : { virtualEnvironment }),
      },
      io,
      options.command,
    );
    if (!ready) return 1;
  }

  const commandArgs = [
    "-m",
    trainerModule,
    "train",
    "--bundle",
    bundle,
    "--artifact",
    artifact,
    "--teacher",
    teacher,
    "--cache-dir",
    cacheDir,
    "--report",
    report,
  ];
  for (const name of PASSTHROUGH_STRING) {
    const value = values[name];
    if (typeof value === "string") commandArgs.push(`--${name}`, value);
  }
  // The manifest records which compiler produced the bundle: the installed
  // @semantscript/compiler unless the caller names another.
  if (typeof values["compiler-version"] !== "string") {
    commandArgs.push("--compiler-version", COMPILER_VERSION);
  }
  for (const name of PASSTHROUGH_BOOLEAN) {
    if (values[name] === true) commandArgs.push(`--${name}`);
  }
  if (estimate) {
    commandArgs.push("--estimate");
    const outcome = await capture(python, commandArgs, trainerIo);
    if (outcome.error !== undefined) {
      io.stderr(unableToRun(python, outcome.error, doctor));
      return 1;
    }
    if (outcome.signal !== undefined && io.signal?.aborted !== true) {
      io.stderr(outcome.stderr);
      io.stderr(
        `semantscript train: the trainer was killed by ${outcome.signal}; next: ${remedyText("trainer-killed", { doctor })}\n`,
      );
      // The shell's convention, as for a training run: 128 plus the signal number.
      return (
        128 +
        ((constants.signals[outcome.signal as NodeJS.Signals] as
          number | undefined) ?? 0)
      );
    }
    if (outcome.status !== 0) {
      const filter = new TrainerStderr((text) => {
        io.stderr(text);
      });
      filter.push(outcome.stderr);
      filter.end();
      const traceback = filter.fatalTraceback();
      if (traceback !== undefined) {
        io.stderr(
          await trainerFailureLine(traceback, {
            python,
            doctor,
            path: tracebackPath(report),
            io,
          }),
        );
      }
      return outcome.status;
    }
    if (outcome.stderr.length > 0) io.stderr(outcome.stderr);
    let document: unknown;
    try {
      document = JSON.parse(outcome.stdout);
    } catch {
      io.stderr(
        `${python} -m ${trainerModule} train --estimate printed no JSON estimate\n`,
      );
      return 1;
    }
    io.stdout(renderEstimate(document));
    return 0;
  }

  io.stderr(`semantscript train: ${python} ${commandArgs.join(" ")}\n`);
  const before = await fileStamp(report);
  const filter = new TrainerStderr((text) => {
    io.stderr(text);
  });
  const outcome = await runProcess(python, commandArgs, trainerIo, filter);
  if (outcome.error !== undefined) {
    io.stderr(unableToRun(python, outcome.error, doctor));
    return 1;
  }
  const status = outcome.status;
  if (outcome.signal !== undefined && io.signal?.aborted !== true) {
    // Killed by the operating system (the OOM killer, a native crash): no
    // Python exception to blame, and a traceback it survived earlier is not
    // the cause. Forward what was held and name the signal.
    filter.release();
    io.stderr(
      `semantscript train: the trainer was killed by ${outcome.signal}; next: ${remedyText("trainer-killed", { doctor })}\n`,
    );
    return status;
  }
  // A report the trainer did not rewrite is an earlier run's: never render it.
  const after = await fileStamp(report);
  const fresh = after !== undefined && after !== before;
  if (status !== 0 && !fresh && io.signal?.aborted !== true) {
    // The trainer stopped before writing a report. A handled failure already
    // printed its `error: …; next: …` line; an uncaught exception becomes one
    // line that names the fix, with the traceback kept in a file.
    const traceback = filter.fatalTraceback();
    if (traceback !== undefined) {
      io.stderr(
        await trainerFailureLine(traceback, {
          python,
          doctor,
          path: tracebackPath(report),
          io,
        }),
      );
    }
    return status;
  }
  filter.release();
  if (!fresh) {
    if (status === 0)
      io.stderr(
        `the trainer exited successfully but wrote no report at ${report}\n`,
      );
    return status === 0 ? 1 : status;
  }
  let document: unknown;
  try {
    document = await readJson(report);
  } catch {
    if (status === 0)
      io.stderr(
        `the trainer exited successfully but wrote an unreadable report at ${report}\n`,
      );
    return status === 0 ? 1 : status;
  }
  io.stdout(renderTrainReport(document));
  return status;
}

/**
 * Runs the trainer with its stderr forwarded through `filter` as it arrives,
 * so progress shows per expression.
 */
function runProcess(
  command: string,
  args: readonly string[],
  io: CliIo,
  filter: TrainerStderr,
): Promise<{
  readonly status: number;
  readonly error?: Error;
  /** The signal that killed the trainer, when one did. */
  readonly signal?: string;
}> {
  return new Promise((resolvePromise) => {
    const child = spawn(command, args, {
      cwd: io.cwd,
      env: {
        ...io.env,
        PYTHONPATH: pythonPath(io.env["PYTHONPATH"]),
        PYTHONIOENCODING: "utf-8",
      },
      stdio: ["ignore", "ignore", "pipe"],
    });
    const decoder = new StringDecoder("utf8");
    child.stderr.on("data", (chunk: Buffer) => {
      filter.push(decoder.write(chunk));
    });
    const abort = (): void => {
      child.kill();
    };
    io.signal?.addEventListener("abort", abort, { once: true });
    child.once("error", (error) => {
      io.signal?.removeEventListener("abort", abort);
      resolvePromise({ status: 1, error });
    });
    child.once("close", (code, signal) => {
      io.signal?.removeEventListener("abort", abort);
      filter.push(decoder.end());
      filter.end();
      if (code === null && signal !== null) {
        // The shell's convention: 128 plus the signal number.
        resolvePromise({
          status: 128 + constants.signals[signal],
          signal,
        });
        return;
      }
      resolvePromise({ status: code ?? 1 });
    });
  });
}

const TRACEBACK_START = "Traceback (most recent call last):";
/** `python3: No module named semantscript_trainer.cli` and `Error while finding module specification for ...`. */
const MODULE_LAUNCH_FAILURE =
  /^(?!(?:error|warning|note):).+?: (?:No module named |Error while finding module specification for )|^Error while finding module specification for /u;
/** The lines that join two tracebacks of one chained exception. */
const CHAINED_EXCEPTION =
  /^(?:During handling of the above exception|The above exception was the direct cause)/u;

/**
 * Splits the trainer's stderr into lines (a carriage return ends one too, so
 * progress bars redraw) and forwards each as soon as it is complete, except a
 * Python traceback. A traceback is held back until the process shows whether it
 * was fatal: when an ordinary line follows it, the trainer went on (a logged
 * warning), so it is released in place; at exit, `fatalTraceback()` hands the
 * caller a traceback that ended the run, and `release()` prints one that did not.
 */
export class TrainerStderr {
  #pending = "";
  /** The traceback being held, from its first line. */
  #held: string[] | undefined;
  /** Whether the held traceback's exception line has arrived. */
  #complete = false;
  /** The last traceback released in place, while no `error:` line followed it. */
  #released: string | undefined;
  readonly #forward: (text: string) => void;

  constructor(forward: (text: string) => void) {
    this.#forward = forward;
  }

  /** The traceback held at the end of the output, if any. */
  get traceback(): string | undefined {
    return this.#held?.join("");
  }

  /**
   * The traceback that ended the run, for a process that exited nonzero
   * without a report: the held one, or one already released in place that no
   * handled `error:` line followed. The held text is not forwarded.
   */
  fatalTraceback(): string | undefined {
    const held = this.traceback;
    this.#held = undefined;
    return held ?? this.#released;
  }

  /** Forward a held traceback: the run finished without dying on it. */
  release(): void {
    if (this.#held === undefined) return;
    const held = this.#held;
    this.#held = undefined;
    this.#released = held.join("");
    for (const line of held) this.#forward(line);
  }

  push(text: string): void {
    this.#pending += text;
    let end = this.#lineEnd();
    while (end >= 0) {
      this.#line(this.#pending.slice(0, end + 1));
      this.#pending = this.#pending.slice(end + 1);
      end = this.#lineEnd();
    }
  }

  end(): void {
    if (this.#pending.length > 0) this.#line(`${this.#pending}\n`);
    this.#pending = "";
  }

  #lineEnd(): number {
    const newline = this.#pending.indexOf("\n");
    const carriage = this.#pending.indexOf("\r");
    if (carriage < 0) return newline;
    // Keep `\r\n` together; a lone `\r` (a progress bar redraw) ends a line.
    if (carriage === this.#pending.length - 1) return newline;
    if (this.#pending[carriage + 1] === "\n") {
      return newline >= 0 && newline < carriage ? newline : carriage + 1;
    }
    return newline >= 0 && newline < carriage ? newline : carriage;
  }

  #line(line: string): void {
    const text = line.trimEnd();
    if (this.#held !== undefined) {
      if (this.#continues(text)) {
        this.#held.push(line);
        return;
      }
      // An ordinary line after the exception: the trainer went on.
      this.release();
    }
    if (text === TRACEBACK_START || MODULE_LAUNCH_FAILURE.test(text)) {
      this.#held = [line];
      this.#complete = text !== TRACEBACK_START;
      return;
    }
    if (text.startsWith("error:")) this.#released = undefined;
    this.#forward(line);
  }

  /** Whether `text` still belongs to the held traceback. */
  #continues(text: string): boolean {
    if (text === TRACEBACK_START) {
      this.#complete = false;
      return true;
    }
    if (!this.#complete) {
      if (text.length > 0 && !/^\s/u.test(text)) this.#complete = true;
      return true;
    }
    return (
      text.length === 0 || /^\s/u.test(text) || CHAINED_EXCEPTION.test(text)
    );
  }
}

/** The doctor check that installs each module the trainer imports; anything else is the trainer's. */
const MODULE_CHECKS: Readonly<Record<string, string>> = {
  semantscript_model: "model",
  torch: "torch",
  transformers: "torch",
  tokenizers: "torch",
  safetensors: "torch",
  numpy: "torch",
  huggingface_hub: "torch",
  onnx: "onnxruntime",
  onnxruntime: "onnxruntime",
  onnxscript: "onnxruntime",
};

export interface TrainerFailure {
  /** `ModuleNotFoundError: No module named 'torch'`, the exception line. */
  readonly exception: string;
  readonly remedy: "module-missing" | "python-too-old" | "trainer-crash";
  /** The module that did not import (module-missing only). */
  readonly module?: string;
  /** The `semantscript doctor` check that covers it. */
  readonly check: string;
}

/** Classify a trainer traceback (or launch failure) by its final exception line. */
export function classifyTrainerFailure(traceback: string): TrainerFailure {
  const lines = traceback
    .split("\n")
    .map((line) => line.trimEnd())
    .filter((line) => line.length > 0);
  let exception =
    [...lines].reverse().find((line) => !/^\s/u.test(line)) ?? "unknown error";
  const specification = /\((ModuleNotFoundError: .*)\)$/u.exec(exception);
  if (specification?.[1] !== undefined) exception = specification[1];
  const missing =
    /No module named '?([A-Za-z0-9_.]+)'?/u.exec(exception)?.[1] ?? undefined;
  if (missing !== undefined) {
    if (!exception.startsWith("ModuleNotFoundError")) {
      exception = `ModuleNotFoundError: No module named '${missing}'`;
    }
    return {
      exception,
      remedy: "module-missing",
      module: missing,
      check: MODULE_CHECKS[missing.split(".")[0] ?? missing] ?? "trainer",
    };
  }
  if (exception.startsWith("SyntaxError")) {
    return { exception, remedy: "python-too-old", check: "python" };
  }
  return { exception, remedy: "trainer-crash", check: "trainer" };
}

/** `<report without .json>.traceback.txt`, next to the report. */
function tracebackPath(report: string): string {
  return `${report.replace(/\.json$/u, "")}.traceback.txt`;
}

/** Identifies one version of a file, or undefined when there is none. */
async function fileStamp(path: string): Promise<string | undefined> {
  try {
    const stats = await stat(path, { bigint: true });
    return `${String(stats.ino)}:${String(stats.size)}:${String(stats.mtimeNs)}:${String(stats.ctimeNs)}`;
  } catch {
    return undefined;
  }
}

/**
 * `semantscript doctor`, with the `--python`, `--trainer-module` and
 * `--teacher` this run passed, so following the advice checks the same
 * interpreter, module and teacher file.
 */
export function trainerDoctorCommand(
  values: OptionValues,
  trainerModule: string,
): string {
  const parts = ["semantscript doctor"];
  const python = stringOption(values, "python");
  if (python !== undefined) parts.push(`--python ${shellWord(python)}`);
  if (trainerModule !== DEFAULT_TRAINER_MODULE) {
    parts.push(`--trainer-module ${shellWord(trainerModule)}`);
  }
  const teacher = stringOption(values, "teacher");
  if (teacher !== undefined) parts.push(`--teacher ${shellWord(teacher)}`);
  return parts.join(" ");
}

function shellWord(value: string): string {
  return /^[\w@%+=:,./\\-]+$/u.test(value)
    ? value
    : `"${value.replaceAll('"', '\\"')}"`;
}

/**
 * The one line `train` prints in place of a trainer traceback, naming the
 * doctor check to run; the traceback itself goes to `path`.
 */
async function trainerFailureLine(
  traceback: string,
  options: {
    readonly python: string;
    readonly doctor: string;
    readonly path: string;
    readonly io: CliIo;
  },
): Promise<string> {
  const { python, doctor, path, io } = options;
  let failure = classifyTrainerFailure(traceback);
  if (
    failure.remedy === "python-too-old" &&
    (await pythonIsCurrent(python, io))
  ) {
    // A 3.12 interpreter parses the trainer: the broken file is the trainer's.
    failure = { ...failure, remedy: "trainer-crash", check: "trainer" };
  }
  let saved = false;
  try {
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, traceback);
    saved = true;
  } catch {
    // The line below still names the fix; only the file is missing.
  }
  const fix =
    failure.remedy === "module-missing"
      ? remedyText("module-missing", {
          doctor,
          check: failure.check,
          python,
          module: failure.module ?? "",
        })
      : failure.remedy === "python-too-old"
        ? remedyText("python-too-old", { doctor, python })
        : remedyText("trainer-crash", { doctor });
  return `semantscript train: the trainer stopped: ${failure.exception}; next: ${fix}${saved ? `; full traceback in ${path}` : ""}\n`;
}

/** Whether `python` is 3.12 or later (false when it cannot say). */
async function pythonIsCurrent(python: string, io: CliIo): Promise<boolean> {
  const outcome = await capture(
    python,
    ["-c", "import sys; print(sys.version_info >= (3, 12))"],
    io,
  );
  return outcome.status === 0 && outcome.stdout.trim() === "True";
}

/** `unable to run <python>: spawn <python> ENOENT; next: ...` */
function unableToRun(python: string, error: Error, doctor: string): string {
  return `unable to run ${python}: ${error.message}; next: ${remedyText("python-missing", { doctor })}\n`;
}

/** Render the trainer's JSON report as the per-function table `semantscript train` prints. */
export function renderTrainReport(document: unknown): string {
  const report = objectOf(document, "report");
  const status = stringOf(report["status"], "report.status");
  const rows = listOf(report["functions"], "report.functions").map(
    (entry, index) => {
      const path = `report.functions[${String(index)}]`;
      const fn = objectOf(entry, path);
      const dataset = objectOf(fn["dataset"], `${path}.dataset`);
      const training = objectOf(fn["training"], `${path}.training`);
      const verification = objectOf(fn["verification"], `${path}.verification`);
      const metrics = objectOf(
        verification["metrics"],
        `${path}.verification.metrics`,
      );
      const adversarial = fn["adversarial"];
      const adversarialCases =
        adversarial === null || adversarial === undefined
          ? 0
          : numberOf(
              objectOf(adversarial, `${path}.adversarial`)["cases"],
              `${path}.adversarial.cases`,
            );
      const heldOut = training["heldOutAccuracy"];
      return [
        shortId(stringOf(fn["id"], `${path}.id`)),
        typeof fn["sourcePath"] === "string" ? fn["sourcePath"] : "",
        typeof fn["cache"] === "string" ? fn["cache"] : "trained",
        String(numberOf(dataset["cases"], `${path}.dataset.cases`)),
        String(adversarialCases),
        typeof heldOut === "number" ? formatRatio(heldOut) : "-",
        stringOf(verification["status"], `${path}.verification.status`),
        formatRatio(
          numberOf(
            metrics["accuracy"],
            `${path}.verification.metrics.accuracy`,
          ),
        ),
        formatRatio(
          numberOf(metrics["ece"], `${path}.verification.metrics.ece`),
        ),
        String(
          numberOf(
            verification["attestedCases"],
            `${path}.verification.attestedCases`,
          ),
        ),
        String(
          numberOf(
            metrics["constraintViolations"],
            `${path}.verification.metrics.constraintViolations`,
          ),
        ),
      ];
    },
  );
  const lines = [
    renderTable(
      [
        "function",
        "source",
        "cache",
        "cases",
        "adversarial",
        "held-out acc",
        "verification",
        "accuracy",
        "ece",
        "attested",
        "violations",
      ],
      rows,
    ),
  ];
  for (const entry of listOf(report["functions"], "report.functions")) {
    const fn = objectOf(entry, "report.functions[]");
    const verification = objectOf(
      fn["verification"],
      "report.functions[].verification",
    );
    const failures = verification["failures"];
    if (!Array.isArray(failures) || failures.length === 0) continue;
    lines.push(
      `${shortId(stringOf(fn["id"], "report.functions[].id"))} verification failures:\n`,
    );
    for (const failure of failures as readonly unknown[]) {
      lines.push(
        `  ${typeof failure === "string" ? failure.replaceAll("\n", "\n  ") : JSON.stringify(failure)}\n`,
      );
    }
  }
  lines.push(...renderSeedRetry(report));
  const cache = report["cache"];
  if (cache !== null && cache !== undefined) {
    const summary = objectOf(cache, "report.cache");
    const directory = summary["directory"];
    const reused = String(numberOf(summary["reused"], "report.cache.reused"));
    const trained = String(
      numberOf(summary["trained"], "report.cache.trained"),
    );
    const where = typeof directory === "string" ? ` (${directory})` : " (off)";
    lines.push(`build cache: ${reused} reused, ${trained} trained${where}\n`);
  }
  const teacher = report["teacher"];
  if (
    teacher !== null &&
    typeof teacher === "object" &&
    !Array.isArray(teacher)
  ) {
    const spend = (teacher as Record<string, unknown>)["spend"];
    if (spend !== null && spend !== undefined) {
      lines.push(renderSpend(objectOf(spend, "report.teacher.spend")));
    }
  }
  const artifact = report["artifact"];
  if (artifact !== null && artifact !== undefined) {
    const record = objectOf(artifact, "report.artifact");
    lines.push(
      `artifact: ${stringOf(record["root"], "report.artifact.root")} (release ${stringOf(record["manifestSha256"], "report.artifact.manifestSha256")})\n`,
    );
  }
  lines.push(
    status === "reused"
      ? "train reused: nothing changed, no training performed\n"
      : `train ${status}\n`,
  );
  return lines.join("");
}

/**
 * The seed retry: one row per function per training attempt when the build
 * retrained with another seed, the seed the release published, and why the
 * retry stopped when it did. Reports from before the retry carry none of it.
 */
function renderSeedRetry(report: Readonly<Record<string, unknown>>): string[] {
  const lines: string[] = [];
  const attempts = Array.isArray(report["attempts"])
    ? listOf(report["attempts"], "report.attempts")
    : [];
  if (attempts.length > 1) {
    const rows: string[][] = [];
    for (const [index, entry] of attempts.entries()) {
      const path = `report.attempts[${String(index)}]`;
      const attempt = objectOf(entry, path);
      const functions = listOf(attempt["functions"], `${path}.functions`);
      for (const [position, item] of functions.entries()) {
        const fnPath = `${path}.functions[${String(position)}]`;
        const fn = objectOf(item, fnPath);
        const violations = numberOf(
          fn["constraintViolations"],
          `${fnPath}.constraintViolations`,
        );
        const records = fn["records"];
        const rate = fn["violationRate"];
        rows.push([
          String(numberOf(attempt["attempt"], `${path}.attempt`)),
          String(numberOf(attempt["seed"], `${path}.seed`)),
          shortId(stringOf(fn["id"], `${fnPath}.id`)),
          stringOf(fn["status"], `${fnPath}.status`),
          formatRatio(numberOf(fn["accuracy"], `${fnPath}.accuracy`)),
          typeof records === "number" && typeof rate === "number"
            ? `${String(violations)}/${String(records)} (${(rate * 100).toFixed(2)}%)`
            : String(violations),
          formatRatio(numberOf(fn["ece"], `${fnPath}.ece`)),
        ]);
      }
    }
    lines.push(
      "training attempts:\n",
      renderTable(
        [
          "attempt",
          "seed",
          "function",
          "status",
          "accuracy",
          "violations",
          "ece",
        ],
        rows,
      ),
    );
    if (report["status"] === "passed" && typeof report["seed"] === "number") {
      lines.push(
        `seed retry: published seed ${String(report["seed"])} after ${String(attempts.length)} attempts\n`,
      );
    }
  }
  const retry = report["retry"];
  if (retry !== null && typeof retry === "object" && !Array.isArray(retry)) {
    const reason = (retry as Record<string, unknown>)["stopReason"];
    if (typeof reason === "string") {
      lines.push(`seed retry stopped: ${reason}\n`);
    }
  }
  return lines;
}

/** `teacher: 596 requests (12 replayed), USD 1.0213 of the USD 5 cap`. */
function renderSpend(spend: Readonly<Record<string, unknown>>): string {
  const requests = numberOf(spend["requests"], "report.teacher.spend.requests");
  const replayed =
    typeof spend["replayed"] === "number" && spend["replayed"] > 0
      ? ` (${String(spend["replayed"])} replayed)`
      : "";
  const cost =
    typeof spend["costUsd"] === "number"
      ? `USD ${spend["costUsd"].toFixed(4)}`
      : "USD unknown";
  const cap =
    typeof spend["maxCostUsd"] === "number"
      ? ` of the USD ${String(spend["maxCostUsd"])} cap`
      : "";
  return `teacher: ${String(requests)} request${requests === 1 ? "" : "s"}${replayed}, ${cost}${cap}\n`;
}

/** Render `train --estimate`'s JSON as one row per expression plus the total. */
export function renderEstimate(document: unknown): string {
  const estimate = objectOf(document, "estimate");
  if (estimate["kind"] !== "semantscript.train-estimate") {
    throw new Error("estimate.kind must be semantscript.train-estimate");
  }
  const price = objectOf(estimate["price"], "estimate.price");
  const teacher = objectOf(estimate["teacher"], "estimate.teacher");
  const row = (
    name: string,
    source: string,
    entry: Readonly<Record<string, unknown>>,
    path: string,
  ): string[] => {
    const expected = numberOf(
      entry["expectedRequests"],
      `${path}.expectedRequests`,
    );
    const maximum = numberOf(
      entry["maximumRequests"],
      `${path}.maximumRequests`,
    );
    const cost = numberOf(entry["costUsd"], `${path}.costUsd`);
    const maximumCost = numberOf(
      entry["maximumCostUsd"],
      `${path}.maximumCostUsd`,
    );
    const batch = numberOf(entry["batchRequests"], `${path}.batchRequests`);
    return [
      name,
      source,
      expected === 0 && maximum === 0
        ? "0"
        : `${String(expected)} (max ${String(maximum)})`,
      numberOf(entry["inputTokens"], `${path}.inputTokens`).toLocaleString(
        "en-US",
      ),
      numberOf(
        entry["cacheReadTokens"],
        `${path}.cacheReadTokens`,
      ).toLocaleString("en-US"),
      numberOf(entry["outputTokens"], `${path}.outputTokens`).toLocaleString(
        "en-US",
      ),
      `${cost.toFixed(2)} (max ${maximumCost.toFixed(2)})`,
      formatDuration(numberOf(entry["seconds"], `${path}.seconds`)) +
        (batch > 0 && typeof entry["maximumSeconds"] === "number"
          ? ` (max ${formatDuration(entry["maximumSeconds"])})`
          : ""),
    ];
  };
  const functions = listOf(estimate["functions"], "estimate.functions").map(
    (entry, index) => {
      const path = `estimate.functions[${String(index)}]`;
      const fn = objectOf(entry, path);
      const cached = objectOf(fn["cached"], `${path}.cached`);
      const reused =
        cached["dataset"] === true && cached["adversarial"] !== false
          ? " (cached)"
          : "";
      return row(
        shortId(stringOf(fn["id"], `${path}.id`)),
        (typeof fn["sourcePath"] === "string" ? fn["sourcePath"] : "") + reused,
        fn,
        path,
      );
    },
  );
  functions.push(
    row(
      "total",
      "",
      objectOf(estimate["total"], "estimate.total"),
      "estimate.total",
    ),
  );
  const model =
    typeof teacher["model"] === "string"
      ? `${stringOf(teacher["backend"], "estimate.teacher.backend")} ${teacher["model"]}${teacher["fallback"] === true ? " (fallback)" : ""}`
      : stringOf(teacher["backend"], "estimate.teacher.backend");
  const lines = [
    `teacher: ${model}; price: ${stringOf(price["source"], "estimate.price.source")} (USD ${String(numberOf(price["inputUsdPerMillion"], "estimate.price.inputUsdPerMillion"))} in / ${String(numberOf(price["outputUsdPerMillion"], "estimate.price.outputUsdPerMillion"))} out per million tokens)\n`,
    renderTable(
      [
        "function",
        "source",
        "requests",
        "input tokens",
        "cached",
        "output tokens",
        "USD",
        "time",
      ],
      functions,
    ),
    `time per request: ${String(numberOf(estimate["secondsPerRequest"], "estimate.secondsPerRequest"))} s (${stringOf(estimate["secondsSource"], "estimate.secondsSource")}); tokens are characters / ${String(numberOf(estimate["charactersPerToken"], "estimate.charactersPerToken"))}; no teacher request was sent\n`,
  ];
  const totals = objectOf(estimate["total"], "estimate.total");
  const batched = numberOf(
    totals["batchRequests"],
    "estimate.total.batchRequests",
  );
  if (batched > 0) {
    lines.push(
      `${String(batched)} request${batched === 1 ? " goes" : "s go"} through the Message Batches API at half price; the time counts about 1 h per batch (most batches finish within an hour) and the maximum counts the full poll_timeout_seconds (24 h by default) per batch the run can submit\n`,
    );
  }
  const cap = estimate["maxCostUsd"];
  if (typeof cap === "number") {
    const total = objectOf(estimate["total"], "estimate.total");
    const expected = numberOf(total["costUsd"], "estimate.total.costUsd");
    const maximum = numberOf(
      total["maximumCostUsd"],
      "estimate.total.maximumCostUsd",
    );
    lines.push(
      expected > cap
        ? `--max-cost-usd ${String(cap)} is below the expected USD ${expected.toFixed(2)}: the run will stop before it finishes\n`
        : maximum > cap
          ? `--max-cost-usd ${String(cap)} covers the expected cost but not the maximum (USD ${maximum.toFixed(2)})\n`
          : `--max-cost-usd ${String(cap)} covers the maximum cost\n`,
    );
  }
  return lines.join("");
}

function formatDuration(seconds: number): string {
  if (seconds < 90) return `${seconds.toFixed(0)} s`;
  if (seconds < 5400) return `${(seconds / 60).toFixed(0)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}
