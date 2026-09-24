import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { delimiter, dirname, join, resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import {
  listOf,
  numberOf,
  objectOf,
  requireString,
  stringOf,
  stringOption,
  type CliIo,
} from "./io.js";
import { readJson } from "./manifest.js";
import { formatRatio, renderTable, shortId } from "./table.js";

const REPOSITORY_ROOT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const DEFAULT_TRAINER_MODULE = "semantscript_trainer.cli";

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
] as const;
const PASSTHROUGH_BOOLEAN = ["local-files-only", "select-best-epoch"] as const;

const OPTIONS: Record<string, { readonly type: "string" | "boolean" }> = {
  bundle: { type: "string" },
  artifact: { type: "string" },
  teacher: { type: "string" },
  "cache-dir": { type: "string" },
  report: { type: "string" },
  python: { type: "string" },
  "trainer-module": { type: "string" },
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
    options: OPTIONS,
    allowPositionals: false,
  });
  const bundle = resolve(io.cwd, requireString(values, "bundle"));
  const artifact = resolve(io.cwd, requireString(values, "artifact"));
  const teacher = resolve(io.cwd, requireString(values, "teacher"));
  const cacheDir = resolve(
    io.cwd,
    stringOption(values, "cache-dir") ?? ".semantscript/cache",
  );
  const report = resolve(
    io.cwd,
    stringOption(values, "report") ?? `${artifact}.report.json`,
  );
  const python =
    stringOption(values, "python") ??
    io.env["SEMANTSCRIPT_PYTHON"] ??
    (process.platform === "win32" ? "python" : "python3");
  const trainerModule =
    stringOption(values, "trainer-module") ?? DEFAULT_TRAINER_MODULE;

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
  const result = spawnSync(python, commandArgs, {
    cwd: io.cwd,
    env: { ...io.env, PYTHONPATH: pythonPath(io.env["PYTHONPATH"]) },
    stdio: ["ignore", "ignore", "inherit"],
  });
  if (result.error !== undefined) {
    io.stderr(`unable to run ${python}: ${result.error.message}\n`);
    return 1;
  }
  const status = result.status ?? 1;
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

/** The monorepo's trainer and model sources when the CLI runs from the checkout. */
export function pythonPath(existing: string | undefined): string {
  const entries: string[] = [];
  if (
    existsSync(join(REPOSITORY_ROOT, "trainer", "src", "semantscript_trainer"))
  ) {
    entries.push(
      join(REPOSITORY_ROOT, "trainer", "src"),
      join(REPOSITORY_ROOT, "model", "src"),
    );
    const localPackages = join(REPOSITORY_ROOT, ".python-packages");
    if (existsSync(localPackages)) entries.push(localPackages);
  }
  if (existing !== undefined && existing.length > 0) entries.push(existing);
  return entries.join(delimiter);
}

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
      return [
        shortId(stringOf(fn["id"], `${path}.id`)),
        typeof fn["sourcePath"] === "string" ? fn["sourcePath"] : "",
        String(numberOf(dataset["cases"], `${path}.dataset.cases`)),
        String(adversarialCases),
        formatRatio(
          numberOf(
            training["heldOutAccuracy"],
            `${path}.training.heldOutAccuracy`,
          ),
        ),
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
  const artifact = report["artifact"];
  if (artifact !== null && artifact !== undefined) {
    const record = objectOf(artifact, "report.artifact");
    lines.push(
      `artifact: ${stringOf(record["root"], "report.artifact.root")} (release ${stringOf(record["manifestSha256"], "report.artifact.manifestSha256")})\n`,
    );
  }
  lines.push(`train ${status}\n`);
  return lines.join("");
}
