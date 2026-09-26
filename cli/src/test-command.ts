import { resolve } from "node:path";
import { parseArgs } from "node:util";

import { resolveArtifactRoot } from "./defaults.js";
import { listOf, objectOf, stringOf, type CliIo } from "./io.js";
import {
  readArtifactSummary,
  readJson,
  type ArtifactSummary,
} from "./manifest.js";
import {
  renderTable,
  shortId,
  VERIFICATION_HEADERS,
  verificationCells,
} from "./table.js";

interface ExampleFailure {
  readonly index: number;
  readonly expected: unknown;
  readonly actual: unknown;
  readonly error?: string;
}

interface FunctionExamples {
  readonly functionId: string;
  readonly present: boolean;
  readonly passed: number;
  readonly total: number;
  readonly failures: readonly ExampleFailure[];
}

/**
 * `semantscript test`: report the verification every function shipped with,
 * then replay the bundle's examples through the loaded artifact.
 */
export async function testCommand(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const { values } = parseArgs({
    args: [...args],
    options: {
      artifact: { type: "string" },
      bundle: { type: "string" },
      json: { type: "boolean" },
    },
    allowPositionals: false,
  });
  const root = resolveArtifactRoot(values, io);
  const summary = await readArtifactSummary(root);
  const examples =
    values.bundle === undefined
      ? undefined
      : await replayExamples(root, resolve(io.cwd, values.bundle), summary);
  const verified = summary.functions.every((fn) => fn.status === "passed");
  const replayed =
    examples === undefined ||
    examples.every((entry) => entry.present && entry.failures.length === 0);
  const ok = verified && replayed;

  if (values.json === true) {
    io.stdout(
      `${JSON.stringify(
        {
          artifact: { root, manifestSha256: summary.manifestSha256 },
          application: {
            id: summary.applicationId,
            version: summary.applicationVersion,
          },
          functions: summary.functions.map((fn) => ({
            ...fn,
            examples:
              examples?.find((entry) => entry.functionId === fn.id) ?? null,
          })),
          missingFunctions: (examples ?? [])
            .filter((entry) => !entry.present)
            .map((entry) => entry.functionId),
          ok,
        },
        null,
        2,
      )}\n`,
    );
    return ok ? 0 : 1;
  }

  const rows = summary.functions.map((fn) => {
    const replay = examples?.find((entry) => entry.functionId === fn.id);
    return [
      ...verificationCells(fn),
      replay === undefined
        ? "-"
        : `${String(replay.passed)}/${String(replay.total)}`,
    ];
  });
  const lines = [
    `artifact ${summary.manifestSha256} (${summary.applicationId}@${summary.applicationVersion})`,
    renderTable([...VERIFICATION_HEADERS, "examples"], rows).trimEnd(),
  ];
  for (const entry of examples ?? []) {
    if (!entry.present) {
      lines.push(`${shortId(entry.functionId)}: absent from the artifact`);
      continue;
    }
    for (const failure of entry.failures) {
      const detail =
        failure.error === undefined
          ? `expected ${JSON.stringify(failure.expected)}, got ${JSON.stringify(failure.actual)}`
          : failure.error;
      lines.push(
        `${shortId(entry.functionId)} example ${String(failure.index)}: ${detail}`,
      );
    }
  }
  lines.push(`test ${ok ? "passed" : "failed"}`);
  io.stdout(`${lines.join("\n")}\n`);
  return ok ? 0 : 1;
}

async function replayExamples(
  root: string,
  bundlePath: string,
  summary: ArtifactSummary,
): Promise<readonly FunctionExamples[]> {
  const bundle = objectOf(await readJson(bundlePath), "bundle");
  const functions = listOf(bundle["functions"], "bundle.functions").map(
    (entry, index) => {
      const path = `bundle.functions[${String(index)}]`;
      const fn = objectOf(entry, path);
      const definition = objectOf(fn["definition"], `${path}.definition`);
      return {
        id: stringOf(fn["id"], `${path}.id`),
        examples: listOf(
          definition["examples"],
          `${path}.definition.examples`,
        ).map((example, position) => {
          const record = objectOf(
            example,
            `${path}.definition.examples[${String(position)}]`,
          );
          return {
            inputs: objectOf(
              record["inputs"],
              `${path}.definition.examples[${String(position)}].inputs`,
            ),
            output: record["output"],
          };
        }),
      };
    },
  );
  const { loadSemaArtifact } = await import("@semantscript/core");
  const handle = await loadSemaArtifact(root);
  try {
    return functions.map((fn) => {
      const shipped = summary.functions.find(
        (candidate) => candidate.id === fn.id,
      );
      if (shipped === undefined || !handle.functionIds.has(fn.id)) {
        return {
          functionId: fn.id,
          present: false,
          passed: 0,
          total: fn.examples.length,
          failures: [],
        };
      }
      const failures: ExampleFailure[] = [];
      fn.examples.forEach((example, index) => {
        try {
          const raw = handle.call<unknown>(fn.id, example.inputs);
          const actual =
            shipped.resultMode === "diagnostic"
              ? objectOf(raw, "diagnostic")["value"]
              : raw;
          if (canonical(actual) !== canonical(example.output)) {
            failures.push({ index, expected: example.output, actual });
          }
        } catch (error: unknown) {
          failures.push({
            index,
            expected: example.output,
            actual: undefined,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      });
      return {
        functionId: fn.id,
        present: true,
        passed: fn.examples.length - failures.length,
        total: fn.examples.length,
        failures,
      };
    });
  } finally {
    await handle.close();
  }
}

/** Key-order-independent JSON text so object outputs compare by value. */
export function canonical(value: unknown): string {
  return value === undefined ? "undefined" : JSON.stringify(sortKeys(value));
}

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) return (value as readonly unknown[]).map(sortKeys);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
        .map(([key, entry]) => [key, sortKeys(entry)]),
    );
  }
  return value;
}
