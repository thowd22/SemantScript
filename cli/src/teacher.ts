import { resolve } from "node:path";
import { parseArgs } from "node:util";

import {
  DEFAULT_TRAINER_MODULE,
  findTeacherConfig,
  resolvePython,
} from "./defaults.js";
import { capture } from "./doctor.js";
import {
  CliUsageError,
  objectOf,
  stringOf,
  stringOption,
  type CliIo,
} from "./io.js";

export const TEACHER_PROBE_KIND = "semantscript.teacher-probe";

/** One probe request's outcome, as `python -m semantscript_trainer.cli teacher probe --json` prints it. */
export interface TeacherProbeResult {
  readonly ok: boolean;
  readonly backend: string;
  readonly model: string | null;
  readonly baseUrl: string | null;
  readonly requestSent: boolean;
  readonly latencySeconds: number | null;
  readonly inputTokens: number | null;
  readonly outputTokens: number | null;
  readonly costUsd: number | null;
  readonly priceSource: string | null;
  readonly summary: string;
  readonly fix: string | null;
}

/**
 * `semantscript teacher probe`: send one small request through the configured
 * teacher (`--teacher`, else the teacher file `train` would use) and report the
 * model, the latency, the tokens and the cost. The constraints teacher sends
 * nothing; a mixed constraints teacher probes its fallback. Exits 1 when the
 * request fails.
 */
export async function teacherCommand(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const [subcommand, ...rest] = args;
  if (subcommand !== "probe") {
    throw new CliUsageError(
      "teacher needs a subcommand: semantscript teacher probe",
    );
  }
  const { values } = parseArgs({
    args: rest,
    options: {
      teacher: { type: "string" },
      python: { type: "string" },
      "trainer-module": { type: "string" },
      "cache-dir": { type: "string" },
      json: { type: "boolean" },
    },
    allowPositionals: false,
  });
  const teacher = findTeacherConfig(values, io);
  if (teacher === undefined) {
    throw new CliUsageError(
      "no teacher to probe: pass --teacher <teacher.toml>|constraints, or run semantscript init --teacher anthropic|openrouter|ollama|constraints to write .semantscript/teacher.toml",
    );
  }
  const python = resolvePython(values, io);
  const trainerModule =
    stringOption(values, "trainer-module") ?? DEFAULT_TRAINER_MODULE;
  const cacheDir = resolve(
    io.cwd,
    stringOption(values, "cache-dir") ?? ".semantscript/cache",
  );
  const outcome = await capture(
    python,
    [
      "-m",
      trainerModule,
      "teacher",
      "probe",
      "--teacher",
      teacher,
      "--cache-dir",
      cacheDir,
      "--json",
    ],
    io,
  );
  if (outcome.error !== undefined) {
    io.stderr(`unable to run ${python}: ${outcome.error.message}\n`);
    return 1;
  }
  const line = outcome.stdout.trim().split("\n").at(-1) ?? "";
  let result: TeacherProbeResult;
  try {
    result = parseTeacherProbe(JSON.parse(line));
  } catch (error: unknown) {
    io.stderr(outcome.stderr);
    io.stderr(
      `${python} -m ${trainerModule} teacher probe printed no result${error instanceof Error ? `: ${error.message}` : ""}\n`,
    );
    return 1;
  }
  if (values.json === true) {
    io.stdout(
      `${JSON.stringify({ kind: TEACHER_PROBE_KIND, probeVersion: 1, ...result }, null, 2)}\n`,
    );
  } else {
    io.stdout(renderTeacherProbe(teacher, result));
  }
  return result.ok ? 0 : 1;
}

/** Validate the Python side's probe result. */
export function parseTeacherProbe(document: unknown): TeacherProbeResult {
  const value = objectOf(document, "teacher probe");
  if (value["kind"] !== TEACHER_PROBE_KIND || value["probeVersion"] !== 1) {
    throw new Error(`teacher probe must be a ${TEACHER_PROBE_KIND} version 1`);
  }
  const numberOrNull = (name: string): number | null => {
    const item = value[name];
    if (item === null || item === undefined) return null;
    if (typeof item !== "number" || !Number.isFinite(item)) {
      throw new Error(`teacher probe ${name} must be a number or null`);
    }
    return item;
  };
  const stringOrNull = (name: string): string | null => {
    const item = value[name];
    if (item === null || item === undefined) return null;
    return stringOf(item, `teacher probe ${name}`);
  };
  if (typeof value["ok"] !== "boolean") {
    throw new Error("teacher probe ok must be a boolean");
  }
  return {
    ok: value["ok"],
    backend: stringOf(value["backend"], "teacher probe backend"),
    model: stringOrNull("model"),
    baseUrl: stringOrNull("baseUrl"),
    requestSent: value["requestSent"] === true,
    latencySeconds: numberOrNull("latencySeconds"),
    inputTokens: numberOrNull("inputTokens"),
    outputTokens: numberOrNull("outputTokens"),
    costUsd: numberOrNull("costUsd"),
    priceSource: stringOrNull("priceSource"),
    summary: stringOf(value["summary"], "teacher probe summary"),
    fix: stringOrNull("fix"),
  };
}

export function renderTeacherProbe(
  teacher: string,
  result: TeacherProbeResult,
): string {
  const via =
    result.baseUrl === null ? "" : ` via ${new URL(result.baseUrl).host}`;
  const lines = [
    `semantscript teacher probe: ${teacher}`,
    `  backend  ${result.backend}`,
    `  model    ${result.model === null ? "(none)" : `${result.model}${via}`}`,
  ];
  if (result.requestSent) {
    lines.push(
      `  latency  ${result.latencySeconds === null ? "-" : `${result.latencySeconds.toFixed(2)} s`}`,
      `  tokens   ${result.inputTokens === null ? "?" : String(result.inputTokens)} in / ${result.outputTokens === null ? "?" : String(result.outputTokens)} out`,
    );
  }
  lines.push(
    `  cost     ${result.costUsd === null ? "unknown" : `USD ${result.costUsd.toFixed(6)}`}${result.priceSource === null ? "" : ` (${result.priceSource})`}`,
    `  ${(result.ok ? "ok" : "failed").padEnd(8)} ${result.summary}`,
  );
  if (result.fix !== null) lines.push(`  fix      ${result.fix}`);
  return `${lines.join("\n")}\n`;
}
