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
  for (const name of PASSTHROUGH_BOOLEAN) {
    if (values[name] === true) commandArgs.push(`--${name}`);
  }
  if (estimate) {
    commandArgs.push("--estimate");
    const outcome = await capture(python, commandArgs, io);
    if (outcome.error !== undefined) {
      io.stderr(`unable to run ${python}: ${outcome.error.message}\n`);
      return 1;
    }
    if (outcome.stderr.length > 0) io.stderr(outcome.stderr);
    if (outcome.status !== 0) return outcome.status;
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
