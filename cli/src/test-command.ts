import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

import {
  BUNDLE_FILE_NAME,
  bundleCandidates,
  resolveArtifactRoot,
} from "./defaults.js";
import { CliUsageError, listOf, objectOf, stringOf, type CliIo } from "./io.js";
import {
  readArtifactSummary,
  readJson,
  type ArtifactSummary,
} from "./manifest.js";
import { remedyText } from "./remedy.js";
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

interface BundleFunction {
  readonly id: string;
  readonly examples: readonly {
    readonly inputs: Record<string, unknown>;
    readonly output: unknown;
  }[];
}

/**
 * `semantscript test`: report the verification every function shipped with,
 * run the runtime's load checks (pointer, manifest and resource digests, safe
 * paths) on the release, then compare the artifact's functions with the
 * build's bundle and replay its examples through the loaded artifact.
 * `--no-bundle` skips the bundle; without `--bundle` the build's bundle is
 * found the way `run` and `explain` find it.
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
      "no-bundle": { type: "boolean" },
      json: { type: "boolean" },
    },
    allowPositionals: false,
  });
  if (values.bundle !== undefined && values["no-bundle"] === true) {
    throw new CliUsageError("--bundle and --no-bundle cannot be combined");
  }
  const root = resolveArtifactRoot(values, io);
  const summary = await readSummary(root);
  const verified = summary.functions.every((fn) => fn.status === "passed");
  // The runtime refuses a release with an unverified function before any
  // digest check, so for one of those the unverified next: line is the fix.
  if (verified) await checkArtifact(root);
  const bundlePath =
    values["no-bundle"] === true ? null : findBundle(values.bundle, io);
  const bundle =
    bundlePath === null ? undefined : await readBundleFunctions(bundlePath);
  const examples =
    bundle === undefined
      ? undefined
      : await replayExamples(root, bundle, summary, verified);
  const unbundled =
    bundle === undefined
      ? []
      : summary.functions
          .map((fn) => fn.id)
          .filter((id) => !bundle.some((fn) => fn.id === id));
  const replayed =
    examples === undefined ||
    examples.every((entry) => entry.present && entry.failures.length === 0);
  const ok = verified && replayed && unbundled.length === 0;
  const next: string[] = [];
  if (!verified) next.push(remedyText("test-function-unverified"));
  if (examples?.some((entry) => !entry.present) === true) {
    next.push(remedyText("test-function-absent"));
  }
  if (unbundled.length > 0) next.push(remedyText("test-function-unbundled"));
  if (
    examples?.some((entry) => entry.present && entry.failures.length > 0) ===
    true
  ) {
    next.push(remedyText("test-example-mismatch"));
  }

  if (values.json === true) {
    io.stdout(
      `${JSON.stringify(
        {
          artifact: { root, manifestSha256: summary.manifestSha256 },
          application: {
            id: summary.applicationId,
            version: summary.applicationVersion,
          },
          bundle: bundlePath,
          functions: summary.functions.map((fn) => ({
            ...fn,
            examples:
              examples?.find((entry) => entry.functionId === fn.id) ?? null,
          })),
          missingFunctions: (examples ?? [])
            .filter((entry) => !entry.present)
            .map((entry) => entry.functionId),
          unbundledFunctions: unbundled,
          ok,
          next,
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
  ];
  if (bundlePath !== null) lines.push(`bundle ${bundlePath}`);
  lines.push(
    renderTable([...VERIFICATION_HEADERS, "examples"], rows).trimEnd(),
  );
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
  for (const id of unbundled) {
    lines.push(`${shortId(id)}: in the artifact but not in the bundle`);
  }
  for (const step of next) lines.push(`next: ${step}`);
  lines.push(`test ${ok ? "passed" : "failed"}`);
  io.stdout(`${lines.join("\n")}\n`);
  return ok ? 0 : 1;
}

/**
 * `--bundle`, else the first bundle the build wrote (the tsconfig outDir, then
 * the conventional output directories, as `run` and `explain` look). No build
 * output is a failure that names `semantscript build`.
 */
function findBundle(requested: string | undefined, io: CliIo): string {
  if (requested !== undefined) return resolve(io.cwd, requested);
  const candidates = bundleCandidates(io.cwd);
  const found = candidates.find((candidate) => existsSync(candidate));
  if (found === undefined) {
    throw new Error(
      `no ${BUNDLE_FILE_NAME} under ${candidates
        .map((candidate) => candidate.slice(0, -BUNDLE_FILE_NAME.length - 1))
        .join(", ")}; next: ${remedyText("test-no-build")}`,
    );
  }
  return found;
}

/**
 * The runtime's own load checks on the release `current.json` names, without
 * starting ONNX sessions: pointer, manifest digest and schema, ABI, resource
 * sizes and digests, symlinks and non-regular files. A failure keeps the
 * runtime's `ArtifactLoadError` code and remedy.
 */
async function checkArtifact(root: string): Promise<void> {
  const { checkSemaArtifact } = await import("@semantscript/core");
  try {
    await checkSemaArtifact(root);
  } catch (error: unknown) {
    if (isArtifactLoadError(error)) {
      throw new Error(
        `artifact at ${root} fails the runtime's load checks: ${error.code}: ${error.detail}; next: ${error.remedy}`,
        { cause: error },
      );
    }
    throw error;
  }
}

function isArtifactLoadError(
  error: unknown,
): error is { code: string; detail: string; remedy: string } {
  return (
    error instanceof Error &&
    error.name === "ArtifactLoadError" &&
    "code" in error &&
    typeof error.code === "string" &&
    "detail" in error &&
    typeof error.detail === "string" &&
    "remedy" in error &&
    typeof error.remedy === "string"
  );
}

/**
 * The artifact's summary. With no `current.json` at the root the error names
 * `semantscript train`; a pointer or release that does not read names
 * `semantscript releases rollback`, as `run` does for the same artifact.
 */
async function readSummary(root: string): Promise<ArtifactSummary> {
  try {
    return await readArtifactSummary(root);
  } catch (error: unknown) {
    if (
      typeof error === "object" &&
      error !== null &&
      "code" in error &&
      error.code === "ENOENT" &&
      "path" in error &&
      error.path === join(root, "current.json")
    ) {
      throw new Error(
        `no artifact at ${root} (current.json is missing); next: ${remedyText("test-no-artifact", { root })}`,
        { cause: error },
      );
    }
    throw new Error(
      `${error instanceof Error ? error.message : String(error)}; next: ${remedyText("test-artifact-unreadable")}`,
      { cause: error },
    );
  }
}

async function readBundleFunctions(
  bundlePath: string,
): Promise<readonly BundleFunction[]> {
  const bundle = objectOf(await readJson(bundlePath), "bundle");
  return listOf(bundle["functions"], "bundle.functions").map((entry, index) => {
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
  });
}

/**
 * Each bundle function's examples replayed through the loaded artifact. A
 * function the artifact does not carry is `present: false`; with `load`
 * false (an unverified release, which the runtime refuses) nothing is
 * replayed and only the absent functions are listed.
 */
async function replayExamples(
  root: string,
  functions: readonly BundleFunction[],
  summary: ArtifactSummary,
  load: boolean,
): Promise<readonly FunctionExamples[]> {
  if (!load) {
    return functions
      .filter(
        (fn) => !summary.functions.some((shipped) => shipped.id === fn.id),
      )
      .map((fn) => ({
        functionId: fn.id,
        present: false,
        passed: 0,
        total: fn.examples.length,
        failures: [],
      }));
  }
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
