import { createHash } from "node:crypto";
import path from "node:path";

import type { SourceIrBundle } from "./bundle-ir.js";
import type { SourceNeuralFunctionIr } from "./source-ir.js";
import { compareBytes, semanticJsonBytes } from "./semantic-json.js";

const encoder = new TextEncoder();
const SHA256 = /^[a-f0-9]{64}$/;
const WINDOWS_ABSOLUTE_PATH = /^(?:[A-Za-z]:[\\/]|\\\\)/;

export type NeuralFunctionSemanticProjection = Pick<
  SourceNeuralFunctionIr,
  "irVersion" | "definition" | "inputs" | "output" | "runtime"
>;

/** Returns a lowercase SHA-256 digest for the exact bytes supplied. */
export function sha256Hex(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

/** Returns a SHA-256 digest over semantscript.semantic-json/v1 bytes. */
export function semanticJsonSha256(value: unknown): string {
  return sha256Hex(semanticJsonBytes(value));
}

/** Selects the fields that IR.md defines as a neural function's semantic identity. */
export function createSemanticProjection(
  source: NeuralFunctionSemanticProjection,
): NeuralFunctionSemanticProjection {
  return {
    irVersion: source.irVersion,
    definition: source.definition,
    inputs: source.inputs,
    output: source.output,
    runtime: source.runtime,
  };
}

/** The SHA-256 of a NeuralFunction record's semantic identity projection (`irVersion`, `definition`, `inputs`, `output`, `runtime`). */
export function computeSemanticSha256(
  source: NeuralFunctionSemanticProjection,
): string {
  return semanticJsonSha256(createSemanticProjection(source));
}

/**
 * Makes a case-preserving, slash-separated path relative to the compiler project
 * root and rejects paths outside that root. This is lexical containment; callers
 * that follow symlinks must resolve them before calling this function.
 */
export function normalizeProjectRelativeSourcePath(
  projectRoot: string,
  sourcePath: string,
): string {
  if (projectRoot.length === 0) {
    throw new TypeError("compiler project root must not be empty");
  }

  if (sourcePath.length === 0) {
    throw new TypeError("source path must not be empty");
  }

  const pathApi = usesWindowsPathRules(projectRoot, sourcePath)
    ? path.win32
    : path;
  const absoluteRoot = pathApi.resolve(projectRoot);
  const absoluteSource = pathApi.isAbsolute(sourcePath)
    ? pathApi.resolve(sourcePath)
    : pathApi.resolve(absoluteRoot, sourcePath);
  const relative = pathApi.relative(absoluteRoot, absoluteSource);

  if (
    relative.length === 0 ||
    pathApi.isAbsolute(relative) ||
    relative === ".." ||
    relative.startsWith(`..${pathApi.sep}`)
  ) {
    throw new RangeError(
      "source path must name a file inside the compiler project root",
    );
  }

  const normalized = relative.replaceAll("\\", "/");
  assertNormalizedSourcePath(normalized);
  return normalized;
}

/** Computes the stable IR.md v1 source-expression identity. */
export function createFunctionId(
  normalizedSourcePath: string,
  duplicateOrdinal: number,
  semanticSha256: string,
): string {
  assertNormalizedSourcePath(normalizedSourcePath);

  if (!Number.isSafeInteger(duplicateOrdinal) || duplicateOrdinal < 0) {
    throw new RangeError(
      "duplicate ordinal must be a non-negative safe integer",
    );
  }

  if (!SHA256.test(semanticSha256)) {
    throw new TypeError(
      "semantic SHA-256 must contain 64 lowercase hexadecimal characters",
    );
  }

  return `nf_${semanticJsonSha256([
    "semantscript-function-id",
    1,
    normalizedSourcePath,
    duplicateOrdinal,
    semanticSha256,
  ])}`;
}

/**
 * Serializes a JSON value with UTF-8-sorted object keys, two-space indentation,
 * LF line endings, one trailing LF, and a lexical `-0` for negative zero.
 */
export function stringifyExactJson(value: unknown): string {
  return `${serializeJsonValue(value, 0, new Set())}\n`;
}

/** Deterministic on-disk form for a validated, versioned source IR bundle. */
export function serializeIrBundle(bundle: SourceIrBundle): string {
  return stringifyExactJson(bundle);
}

function usesWindowsPathRules(
  projectRoot: string,
  sourcePath: string,
): boolean {
  return (
    WINDOWS_ABSOLUTE_PATH.test(projectRoot) ||
    WINDOWS_ABSOLUTE_PATH.test(sourcePath)
  );
}

function assertNormalizedSourcePath(value: string): void {
  assertUnicodeScalarString(value);

  const segments = value.split("/");

  if (
    value.length === 0 ||
    value.includes("\\") ||
    path.posix.isAbsolute(value) ||
    WINDOWS_ABSOLUTE_PATH.test(value) ||
    segments.some(
      (segment) => segment.length === 0 || segment === "." || segment === "..",
    )
  ) {
    throw new TypeError(
      "normalized source path must be a project-relative slash-separated path without dot segments",
    );
  }
}

function serializeJsonValue(
  value: unknown,
  depth: number,
  active: Set<object>,
): string {
  if (value === null) {
    return "null";
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new TypeError("exact JSON numbers must be finite");
    }

    return Object.is(value, -0) ? "-0" : String(value);
  }

  if (typeof value === "string") {
    assertUnicodeScalarString(value);
    return JSON.stringify(value);
  }

  if (typeof value !== "object") {
    throw new TypeError(
      "exact JSON values must contain only JSON-compatible data",
    );
  }

  if (active.has(value)) {
    throw new TypeError("exact JSON values must not contain cycles");
  }

  active.add(value);

  try {
    if (Array.isArray(value)) {
      return serializeArray(value, depth, active);
    }

    const prototype = Object.getPrototypeOf(value) as unknown;

    if (prototype !== Object.prototype && prototype !== null) {
      throw new TypeError("exact JSON objects must be plain objects");
    }

    return serializeObject(value, depth, active);
  } finally {
    active.delete(value);
  }
}

function serializeArray(
  value: readonly unknown[],
  depth: number,
  active: Set<object>,
): string {
  const keys = Object.keys(value);

  if (
    keys.length !== value.length ||
    keys.some((key, index) => key !== String(index)) ||
    Object.getOwnPropertySymbols(value).length > 0
  ) {
    throw new TypeError(
      "exact JSON arrays must be dense and must not contain enumerable non-index properties",
    );
  }

  if (value.length === 0) {
    return "[]";
  }

  const indentation = "  ".repeat(depth + 1);
  const closingIndentation = "  ".repeat(depth);
  const items = value.map((_, index) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));

    if (!descriptor || !("value" in descriptor)) {
      throw new TypeError("exact JSON serialization does not invoke accessors");
    }

    return `${indentation}${serializeJsonValue(descriptor.value, depth + 1, active)}`;
  });
  return `[\n${items.join(",\n")}\n${closingIndentation}]`;
}

function serializeObject(
  value: object,
  depth: number,
  active: Set<object>,
): string {
  if (Object.getOwnPropertySymbols(value).length > 0) {
    throw new TypeError(
      "exact JSON objects must not have symbol-keyed properties",
    );
  }

  const descriptors = Object.getOwnPropertyDescriptors(value);
  const keys = Object.keys(descriptors)
    .filter((key) => descriptors[key]?.enumerable)
    .sort((left, right) =>
      compareBytes(encoder.encode(left), encoder.encode(right)),
    );

  if (keys.length === 0) {
    return "{}";
  }

  const indentation = "  ".repeat(depth + 1);
  const closingIndentation = "  ".repeat(depth);
  const entries = keys.map((key) => {
    assertUnicodeScalarString(key);
    const descriptor = descriptors[key];

    if (!descriptor || !("value" in descriptor)) {
      throw new TypeError("exact JSON serialization does not invoke accessors");
    }

    return `${indentation}${JSON.stringify(key)}: ${serializeJsonValue(
      descriptor.value,
      depth + 1,
      active,
    )}`;
  });
  return `{\n${entries.join(",\n")}\n${closingIndentation}}`;
}

function assertUnicodeScalarString(value: string): void {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);

    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);

      if (next < 0xdc00 || next > 0xdfff) {
        throw new TypeError(
          "exact JSON strings cannot contain unpaired UTF-16 surrogates",
        );
      }

      index += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw new TypeError(
        "exact JSON strings cannot contain unpaired UTF-16 surrogates",
      );
    }
  }
}
