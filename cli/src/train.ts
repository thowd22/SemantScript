import { spawn } from "node:child_process";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

import {
  DEFAULT_TRAINER_MODULE,
  pythonPath,
  resolveArtifactRoot,
  resolveBundlePath,
  resolvePython,
  resolveTeacherConfig,
} from "./defaults.js";
import { preflight } from "./doctor.js";
import {
  listOf,
  numberOf,
  objectOf,
  stringOf,
  stringOption,
  type CliIo,
  type OptionValues,
} from "./io.js";
import { readJson } from "./manifest.js";
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
  if (values["no-preflight"] !== true && options.preflight !== false) {
    const device = stringOption(values, "device");
    const ready = await preflight(
      {
        python,
        trainerModule,
        teacher,
        ...(device === undefined ? {} : { device }),
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
  for (const name of PASSTHROUGH_BOOLEAN) {
    if (values[name] === true) commandArgs.push(`--${name}`);
  }

  io.stderr(`semantscript train: ${python} ${commandArgs.join(" ")}\n`);
  const outcome = await runProcess(python, commandArgs, io);
  if (outcome.error !== undefined) {
    io.stderr(`unable to run ${python}: ${outcome.error.message}\n`);
    return 1;
  }
  const status = outcome.status;
  let document: unknown;
  try {
    document = await readJson(report);
  } catch {
    if (status === 0)
      io.stderr(
        `the trainer exited successfully but wrote no report at ${report}\n`,
      );
    return status === 0 ? 1 : status;
  }
  io.stdout(renderTrainReport(document));
  return status;
}

/** Runs the trainer with its stderr streamed to the terminal so progress shows per expression. */
function runProcess(
  command: string,
  args: readonly string[],
  io: CliIo,
): Promise<{ readonly status: number; readonly error?: Error }> {
  return new Promise((resolvePromise) => {
    const child = spawn(command, args, {
      cwd: io.cwd,
      env: { ...io.env, PYTHONPATH: pythonPath(io.env["PYTHONPATH"]) },
      stdio: ["ignore", "ignore", "inherit"],
    });
    const abort = (): void => {
      child.kill();
    };
    io.signal?.addEventListener("abort", abort, { once: true });
    child.once("error", (error) => {
      io.signal?.removeEventListener("abort", abort);
      resolvePromise({ status: 1, error });
    });
    child.once("close", (code) => {
      io.signal?.removeEventListener("abort", abort);
      resolvePromise({ status: code ?? 1 });
    });
  });
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
