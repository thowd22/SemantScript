import { resolve } from "node:path";
import { parseArgs } from "node:util";

import {
  DEFAULT_TRAINER_MODULE,
  resolveArtifactRoot,
  resolvePython,
} from "./defaults.js";
import {
  CliUsageError,
  listOf,
  numberOf,
  objectOf,
  stringOf,
  stringOption,
  type CliIo,
} from "./io.js";
import { readJson, readPointer, ReleaseError } from "./manifest.js";
import {
  formatBytes,
  releasesCommand,
  resolveRelease,
  scanReleases,
} from "./releases.js";
import { formatRatio, renderTable, shortId } from "./table.js";
import {
  fileStamp,
  runProcess,
  trainerDoctorCommand,
  trainerFailureLine,
  TrainerStderr,
  tracebackPath,
  unableToRun,
} from "./train.js";

export const DERIVE_REPORT_KIND = "semantscript.derive-report";

/** Options of `releases derive`; releases.ts lists the value ones in its VALUE_OPTIONS. */
export const DERIVE_OPTIONS = {
  int8: { type: "boolean" },
  artifact: { type: "string" },
  "cache-dir": { type: "string" },
  report: { type: "string" },
  python: { type: "string" },
  "trainer-module": { type: "string" },
  "weight-type": { type: "string" },
  "per-channel": { type: "boolean" },
  "reduce-range": { type: "boolean" },
  "max-attested-disagreements": { type: "string" },
  "max-decision-change-rate": { type: "string" },
  "ece-threshold": { type: "string" },
  promote: { type: "boolean" },
  json: { type: "boolean" },
} as const;

/**
 * `semantscript releases derive --int8 [<release>]`: run the trainer's
 * `derive-int8`, which quantizes the release's encoder graphs, checks the
 * int8 chain against the float32 one on the release's own records from the
 * build cache and publishes the result as a new release beside it. The
 * pointer is left alone unless `--promote` is given; the gate's refusal exits
 * 2 with the figures and the next command.
 */
export async function deriveRelease(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const { values, positionals } = parseArgs({
    args: [...args],
    options: DERIVE_OPTIONS,
    allowPositionals: true,
  });
  const [spec, ...extra] = positionals;
  if (extra.length > 0) {
    throw new CliUsageError("releases derive takes at most one release");
  }
  if (values.int8 !== true) {
    throw new CliUsageError(
      "releases derive needs --int8: int8 dynamic quantization is the one derivation",
    );
  }
  const weightType = values["weight-type"];
  if (
    weightType !== undefined &&
    weightType !== "int8" &&
    weightType !== "uint8"
  ) {
    throw new CliUsageError(
      `--weight-type must be int8 or uint8, got ${weightType}`,
    );
  }
  const attested = values["max-attested-disagreements"];
  if (attested !== undefined && !/^[0-9]+$/u.test(attested)) {
    throw new CliUsageError(
      `--max-attested-disagreements must be a whole number, got ${attested}`,
    );
  }
  for (const name of ["max-decision-change-rate", "ece-threshold"] as const) {
    const value = values[name];
    if (value === undefined) continue;
    const parsed = Number(value);
    if (
      value.trim() === "" ||
      !Number.isFinite(parsed) ||
      parsed < 0 ||
      parsed > 1
    ) {
      throw new CliUsageError(
        `--${name} must be a number from 0 to 1, got ${value}`,
      );
    }
  }

  const root = resolveArtifactRoot(values, io);
  let source: string;
  if (spec === undefined) {
    const pointer = await readPointer(root);
    if (pointer === undefined) {
      throw new ReleaseError(
        "POINTER_INVALID",
        `${resolve(root, "current.json")} does not exist; name the release to derive from`,
      );
    }
    source = pointer.manifestSha256;
  } else {
    source = resolveRelease(await scanReleases(root), spec).digest;
  }
  const cacheDir = resolve(
    io.cwd,
    stringOption(values, "cache-dir") ?? ".semantscript/cache",
  );
  const report = resolve(
    io.cwd,
    stringOption(values, "report") ?? `${root}.derive-report.json`,
  );
  const python = resolvePython(values, io);
  const trainerModule =
    stringOption(values, "trainer-module") ?? DEFAULT_TRAINER_MODULE;
  const doctor = trainerDoctorCommand(values, trainerModule);
  const trainerIo: CliIo = {
    ...io,
    env: { ...io.env, SEMANTSCRIPT_DOCTOR_COMMAND: doctor },
  };
  const commandArgs = [
    "-m",
    trainerModule,
    "derive-int8",
    "--artifact",
    root,
    "--release",
    source,
    "--cache-dir",
    cacheDir,
    "--report",
    report,
  ];
  for (const name of [
    "weight-type",
    "max-attested-disagreements",
    "max-decision-change-rate",
    "ece-threshold",
  ] as const) {
    const value = values[name];
    if (typeof value === "string") commandArgs.push(`--${name}`, value);
  }
  for (const name of ["per-channel", "reduce-range"] as const) {
    if (values[name] === true) commandArgs.push(`--${name}`);
  }

  io.stderr(
    `semantscript releases derive: ${python} ${commandArgs.join(" ")}\n`,
  );
  const before = await fileStamp(report);
  const filter = new TrainerStderr((text) => {
    io.stderr(text);
  });
  const outcome = await runProcess(python, commandArgs, trainerIo, filter);
  if (outcome.error !== undefined) {
    io.stderr(unableToRun(python, outcome.error, doctor));
    return 1;
  }
  if (outcome.signal !== undefined && io.signal?.aborted !== true) {
    filter.release();
    io.stderr(
      `semantscript releases derive: the trainer was killed by ${outcome.signal}\n`,
    );
    return outcome.status;
  }
  const after = await fileStamp(report);
  const fresh = after !== undefined && after !== before;
  if (!fresh) {
    const traceback = filter.fatalTraceback();
    if (traceback !== undefined) {
      io.stderr(
        (
          await trainerFailureLine(traceback, {
            python,
            doctor,
            path: tracebackPath(report),
            io,
          })
        ).replace(/^semantscript train:/u, "semantscript releases derive:"),
      );
    } else if (outcome.status === 0) {
      io.stderr(
        `the trainer exited successfully but wrote no report at ${report}\n`,
      );
    }
    return outcome.status === 0 ? 1 : outcome.status;
  }
  filter.release();
  let document: Readonly<Record<string, unknown>>;
  try {
    document = objectOf(await readJson(report), "report");
    if (document["kind"] !== DERIVE_REPORT_KIND) {
      throw new Error(`report.kind must be ${DERIVE_REPORT_KIND}`);
    }
  } catch (error: unknown) {
    io.stderr(
      `the trainer wrote an unreadable derive report at ${report}: ${error instanceof Error ? error.message : String(error)}\n`,
    );
    return outcome.status === 0 ? 1 : outcome.status;
  }
  const published = document["status"] === "published";
  const derived = published
    ? stringOf(
        objectOf(document["derived"], "report.derived")["manifestSha256"],
        "report.derived.manifestSha256",
      )
    : undefined;
  const artifactFlag =
    stringOption(values, "artifact") === undefined ? "" : ` --artifact ${root}`;
  if (values.json === true) {
    io.stdout(`${JSON.stringify(document, null, 2)}\n`);
  } else {
    io.stdout(
      renderDeriveReport(document, {
        promoting: values.promote === true,
        artifactFlag,
      }),
    );
  }
  if (!published || derived === undefined) {
    return outcome.status === 0 ? 1 : outcome.status;
  }
  if (values.promote === true) {
    return releasesCommand(
      [
        "promote",
        derived,
        "--artifact",
        root,
        ...(values.json === true ? ["--json"] : []),
      ],
      io,
    );
  }
  return outcome.status;
}

/** Render a `semantscript.derive-report`: the figures per function, the sizes and the next command. */
export function renderDeriveReport(
  document: Readonly<Record<string, unknown>>,
  options: {
    readonly promoting?: boolean;
    readonly artifactFlag?: string;
  } = {},
): string {
  const status = stringOf(document["status"], "report.status");
  const source = objectOf(document["source"], "report.source");
  const settings = objectOf(document["settings"], "report.settings");
  const quantization = objectOf(
    document["quantization"],
    "report.quantization",
  );
  const sourceDigest = stringOf(
    source["manifestSha256"],
    "report.source.manifestSha256",
  );
  const lines: string[] = [];
  const flags = `weights ${stringOf(settings["weightType"], "report.settings.weightType")}, per-channel ${settings["perChannel"] === true ? "yes" : "no"}, reduce-range ${settings["reduceRange"] === true ? "yes" : "no"}`;
  const derivedRecord =
    document["derived"] === null || document["derived"] === undefined
      ? undefined
      : objectOf(document["derived"], "report.derived");
  lines.push(
    derivedRecord === undefined
      ? `int8 derivation of releases/sha256-${sourceDigest.slice(0, 12)} (${flags})`
      : `derived releases/sha256-${stringOf(derivedRecord["manifestSha256"], "report.derived.manifestSha256")} from releases/sha256-${sourceDigest} (int8-dynamic, ${flags})`,
  );
  const functions = listOf(
    quantization["functions"],
    "report.quantization.functions",
  );
  const ratio = (value: unknown): string =>
    typeof value === "number" ? formatRatio(value) : "-";
  const rows = functions.map((entry, index) => {
    const path = `report.quantization.functions[${String(index)}]`;
    const fn = objectOf(entry, path);
    return [
      shortId(stringOf(fn["id"], `${path}.id`)),
      String(numberOf(fn["recordsChecked"], `${path}.recordsChecked`)),
      String(numberOf(fn["attestedRecords"], `${path}.attestedRecords`)),
      String(
        numberOf(fn["argmaxDisagreements"], `${path}.argmaxDisagreements`),
      ),
      String(
        numberOf(fn["attestedDisagreements"], `${path}.attestedDisagreements`),
      ),
      ratio(fn["sourceEce"]),
      ratio(fn["quantizedEce"]),
    ];
  });
  const records = numberOf(
    quantization["recordsChecked"],
    "report.quantization.recordsChecked",
  );
  const changes = numberOf(
    quantization["argmaxDisagreements"],
    "report.quantization.argmaxDisagreements",
  );
  rows.push([
    "total",
    String(records),
    String(
      numberOf(
        quantization["attestedRecords"],
        "report.quantization.attestedRecords",
      ),
    ),
    String(changes),
    String(
      numberOf(
        quantization["attestedDisagreements"],
        "report.quantization.attestedDisagreements",
      ),
    ),
    ratio(quantization["sourceEce"]),
    ratio(quantization["quantizedEce"]),
  ]);
  lines.push(
    renderTable(
      [
        "function",
        "records",
        "attested",
        "decisions changed",
        "attested changed",
        "float32 ece",
        "int8 ece",
      ],
      rows,
    ).trimEnd(),
  );
  lines.push(
    `tolerance: ${String(numberOf(settings["attestedDisagreementTolerance"], "report.settings.attestedDisagreementTolerance"))} attested change(s), decision change rate ${String(numberOf(settings["argmaxDisagreementTolerance"], "report.settings.argmaxDisagreementTolerance"))}, ECE ${String(numberOf(settings["eceThreshold"], "report.settings.eceThreshold"))}`,
  );
  const recordSources = listOf(
    document["recordSources"],
    "report.recordSources",
  ).map((entry, index) => {
    const item = objectOf(entry, `report.recordSources[${String(index)}]`);
    return `${String(item["kind"])} ${String(item["sha256"]).slice(0, 12)} (${String(item["records"])} for ${shortId(String(item["functionId"]))})`;
  });
  lines.push(`verified on: ${recordSources.join(", ")}`);
  const heldOut = document["heldOut"];
  if (
    heldOut !== null &&
    typeof heldOut === "object" &&
    !Array.isArray(heldOut)
  ) {
    const note = (heldOut as Record<string, unknown>)["note"];
    if (typeof note === "string") lines.push(`held-out: ${note}`);
  }
  const encoderLine = `encoder ${formatBytes(numberOf(quantization["sourceEncoderByteLength"], "report.quantization.sourceEncoderByteLength"))} -> ${formatBytes(numberOf(quantization["quantizedEncoderByteLength"], "report.quantization.quantizedEncoderByteLength"))}`;
  if (derivedRecord === undefined) {
    lines.push(encoderLine);
  } else {
    lines.push(
      `${encoderLine}; release ${formatBytes(numberOf(source["bytes"], "report.source.bytes"))} -> ${formatBytes(numberOf(derivedRecord["bytes"], "report.derived.bytes"))}`,
    );
  }
  if (status === "published") {
    const digest = stringOf(
      derivedRecord?.["manifestSha256"],
      "report.derived.manifestSha256",
    );
    lines.push(
      options.promoting === true
        ? "published beside the source release; promoting it:"
        : `published beside the source release; current.json is unchanged\nnext: semantscript releases promote ${digest.slice(0, 12)}${options.artifactFlag ?? ""} (semantscript releases promote ${sourceDigest.slice(0, 12)} returns to the float32 release)`,
    );
  } else {
    lines.push(
      `derive ${status}: nothing was published and current.json is unchanged`,
    );
    for (const failure of listOf(document["failures"], "report.failures")) {
      lines.push(`  ${String(failure)}`);
    }
    if (typeof document["next"] === "string") {
      lines.push(`next: ${document["next"]}`);
    }
  }
  return `${lines.join("\n")}\n`;
}
