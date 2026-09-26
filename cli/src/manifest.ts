import { randomBytes } from "node:crypto";
import { lstat, open, readFile, rename, unlink } from "node:fs/promises";
import { join } from "node:path";
import process from "node:process";

import { listOf, numberOf, objectOf, stringOf } from "./io.js";

export interface ManifestHeadSummary {
  readonly outputPath: readonly string[];
  readonly accuracy: number;
  readonly pairConsistency: number;
  readonly ece: number;
}

export interface ManifestFunctionSummary {
  readonly id: string;
  readonly resultMode: string;
  readonly status: string;
  readonly accuracy: number;
  readonly ece: number;
  readonly brier: number;
  readonly pairConsistency: number;
  readonly attestedCases: number;
  readonly exampleFailures: number;
  readonly constraintViolations: number;
  /** The training seed the release was published at; null before seeds were recorded. */
  readonly seed: number | null;
  /** The held-out constraint check; null for a release from before it. */
  readonly heldOutConstraints: HeldOutConstraintsSummary | null;
  readonly heads: readonly ManifestHeadSummary[];
}

/**
 * How an int8 release was derived, from its first quantized encoder's
 * `onnx.quantization` block; the figures are null on a release derived
 * before the manifest recorded them.
 */
export interface ReleaseDerivation {
  readonly precision: string;
  readonly sourceManifestSha256: string;
  readonly weightType: string;
  readonly perChannel: boolean;
  readonly reduceRange: boolean;
  readonly attestedDisagreementTolerance: number;
  readonly argmaxDisagreementTolerance: number;
  readonly eceThreshold: number;
  readonly recordsChecked: number | null;
  readonly attestedRecords: number | null;
  readonly decisionChanges: number | null;
  readonly attestedDecisionChanges: number | null;
  readonly quantizedEce: number | null;
}

/** `violations` of `sampleSize` held-out inputs broke a constraint (drawn with `seed`). */
export interface HeldOutConstraintsSummary {
  readonly sampleSize: number;
  readonly violations: number;
  readonly violationRate: number;
  readonly seed: number;
}

export interface ArtifactSummary {
  readonly root: string;
  readonly release: string;
  readonly manifestSha256: string;
  /** The manifest's `build.createdAt`: when the trainer exported the release. */
  readonly createdAt: string;
  readonly applicationId: string;
  readonly applicationVersion: string;
  readonly functions: readonly ManifestFunctionSummary[];
  /** Present for a release derived by `releases derive --int8`. */
  readonly derivation?: ReleaseDerivation;
}

/**
 * Read the pointer and manifest of a published artifact without loading its
 * models: the recorded per-function verification is what `semantscript test`
 * reports before it replays examples through the runtime.
 */
export async function readArtifactSummary(
  root: string,
): Promise<ArtifactSummary> {
  const pointer = objectOf(
    await readJson(join(root, "current.json")),
    "current.json",
  );
  const release = stringOf(pointer["release"], "current.json.release");
  const manifestSha256 = stringOf(
    pointer["manifestSha256"],
    "current.json.manifestSha256",
  );
  return summarizeManifest(
    root,
    release,
    manifestSha256,
    await readJson(join(root, release, "manifest.json")),
  );
}

/** Summarize one parsed manifest: application, build date and per-function verification. */
export function summarizeManifest(
  root: string,
  release: string,
  manifestSha256: string,
  value: unknown,
): ArtifactSummary {
  const manifest = objectOf(value, "manifest");
  if (manifest["kind"] !== "semantscript.application-artifact") {
    throw new Error("manifest.kind must be semantscript.application-artifact");
  }
  const application = objectOf(manifest["application"], "manifest.application");
  const build = objectOf(manifest["build"], "manifest.build");
  const functions = listOf(manifest["functions"], "manifest.functions").map(
    (entry, index) =>
      summarizeFunction(entry, `manifest.functions[${String(index)}]`),
  );
  return {
    root,
    release,
    manifestSha256,
    createdAt: stringOf(build["createdAt"], "manifest.build.createdAt"),
    applicationId: stringOf(application["id"], "manifest.application.id"),
    applicationVersion: stringOf(
      application["version"],
      "manifest.application.version",
    ),
    functions,
    ...derivationOf(manifest),
  };
}

/** The derivation record of a release with a quantized encoder; empty otherwise. */
function derivationOf(manifest: Readonly<Record<string, unknown>>): {
  derivation?: ReleaseDerivation;
} {
  const resources = Array.isArray(manifest["resources"])
    ? (manifest["resources"] as readonly unknown[])
    : [];
  for (const resource of resources) {
    const record = recordOf(resource);
    const onnx = recordOf(record["onnx"]);
    const precision = onnx["precision"];
    if (
      record["role"] !== "encoder" ||
      typeof precision !== "string" ||
      precision === "float32"
    ) {
      continue;
    }
    const quantization = recordOf(onnx["quantization"]);
    const verification = recordOf(quantization["verification"]);
    const count = (value: unknown): number | null =>
      typeof value === "number" ? value : null;
    return {
      derivation: {
        precision,
        sourceManifestSha256: String(quantization["sourceManifestSha256"]),
        weightType: String(quantization["weightType"]),
        perChannel: quantization["perChannel"] === true,
        reduceRange: quantization["reduceRange"] === true,
        attestedDisagreementTolerance:
          count(quantization["attestedDisagreementTolerance"]) ?? 0,
        argmaxDisagreementTolerance:
          count(quantization["argmaxDisagreementTolerance"]) ?? 0,
        eceThreshold: count(quantization["eceThreshold"]) ?? 0,
        recordsChecked: count(verification["recordsChecked"]),
        attestedRecords: count(verification["attestedRecords"]),
        decisionChanges: count(verification["decisionChanges"]),
        attestedDecisionChanges: count(verification["attestedDecisionChanges"]),
        quantizedEce: count(verification["quantizedEce"]),
      },
    };
  }
  return {};
}

function recordOf(value: unknown): Readonly<Record<string, unknown>> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Readonly<Record<string, unknown>>)
    : {};
}

/** `int8-dynamic from sha256-<12>: 0 of 128 decisions changed (0 of 6 attested)`. */
export function derivationLine(derivation: ReleaseDerivation): string {
  const figures =
    derivation.recordsChecked === null
      ? "figures not recorded"
      : `${String(derivation.decisionChanges)} of ${String(derivation.recordsChecked)} decisions changed (${String(derivation.attestedDecisionChanges)} of ${String(derivation.attestedRecords)} attested)`;
  return `${derivation.precision} from sha256-${derivation.sourceManifestSha256.slice(0, 12)}: ${figures}; tolerance ${String(derivation.attestedDisagreementTolerance)} attested, rate ${String(derivation.argmaxDisagreementTolerance)}, ECE ${String(derivation.eceThreshold)}`;
}

export const POINTER_KIND = "semantscript.artifact-pointer";
const DIGEST = /^[a-f0-9]{64}$/u;

export interface ArtifactPointer {
  readonly release: string;
  readonly manifestSha256: string;
}

/**
 * A release-management failure with a stable code; the CLI reports it with
 * exit status 1.
 */
export class ReleaseError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(`${code}: ${message}`);
    this.name = "ReleaseError";
    this.code = code;
  }
}

/**
 * Read `current.json` strictly (schemas/artifact-pointer.v1.schema.json);
 * `undefined` when it does not exist. A symlinked or non-regular pointer, or
 * one whose release and digest disagree, is a POINTER_INVALID ReleaseError.
 */
export async function readPointer(
  root: string,
): Promise<ArtifactPointer | undefined> {
  const file = join(root, "current.json");
  try {
    const stats = await lstat(file);
    if (!stats.isFile()) {
      throw new ReleaseError(
        "POINTER_INVALID",
        `${file} must be a regular file, not a symlink or directory`,
      );
    }
  } catch (error: unknown) {
    if (isEnoent(error)) return undefined;
    throw error;
  }
  let value: unknown;
  try {
    value = JSON.parse(await readFile(file, "utf8")) as unknown;
  } catch (error: unknown) {
    throw new ReleaseError(
      "POINTER_INVALID",
      `${file} is not JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  const pointer =
    value !== null && typeof value === "object" && !Array.isArray(value)
      ? (value as Readonly<Record<string, unknown>>)
      : undefined;
  const keys = pointer === undefined ? [] : Object.keys(pointer).sort();
  const digest = pointer?.["manifestSha256"];
  if (
    pointer === undefined ||
    keys.join(",") !== "kind,manifestSha256,pointerVersion,release" ||
    pointer["kind"] !== POINTER_KIND ||
    pointer["pointerVersion"] !== 1 ||
    typeof digest !== "string" ||
    !DIGEST.test(digest) ||
    pointer["release"] !== `releases/sha256-${digest}`
  ) {
    throw new ReleaseError(
      "POINTER_INVALID",
      `${file} is not a ${POINTER_KIND} v1 whose release is releases/sha256-<manifestSha256>`,
    );
  }
  return { release: `releases/sha256-${digest}`, manifestSha256: digest };
}

/** The pointer bytes the trainer's exporter writes: sorted keys, indent 2, trailing newline. */
export function pointerBytes(manifestSha256: string): string {
  if (!DIGEST.test(manifestSha256)) {
    throw new Error(`not a manifest digest: ${manifestSha256}`);
  }
  return `${JSON.stringify(
    {
      kind: POINTER_KIND,
      manifestSha256,
      pointerVersion: 1,
      release: `releases/sha256-${manifestSha256}`,
    },
    null,
    2,
  )}\n`;
}

/**
 * Point `current.json` at a release atomically: write an exclusive temporary
 * file beside it, fsync it, rename it over `current.json` and fsync the
 * directory, so a reader (and a runtime watching the pointer) sees either the
 * old pointer or the new one, never a partial file. No release is touched.
 */
export async function writePointer(
  root: string,
  manifestSha256: string,
): Promise<void> {
  const bytes = pointerBytes(manifestSha256);
  const destination = join(root, "current.json");
  try {
    const stats = await lstat(destination);
    if (!stats.isFile()) {
      throw new ReleaseError(
        "POINTER_INVALID",
        `${destination} must be a regular file or absent`,
      );
    }
  } catch (error: unknown) {
    if (!isEnoent(error)) throw error;
  }
  const temporary = join(
    root,
    `.current-${String(process.pid)}-${randomBytes(8).toString("hex")}.tmp`,
  );
  try {
    const handle = await open(temporary, "wx", 0o600);
    try {
      await handle.writeFile(bytes, "utf8");
      await handle.sync();
    } finally {
      await handle.close();
    }
    await rename(temporary, destination);
    await fsyncDirectory(root);
  } finally {
    await unlink(temporary).catch((error: unknown) => {
      if (!isEnoent(error)) throw error;
    });
  }
}

async function fsyncDirectory(directory: string): Promise<void> {
  // Windows cannot open a directory for fsync; its rename is already durable
  // enough through MoveFileEx, as the trainer's exporter also assumes.
  if (process.platform === "win32") return;
  const handle = await open(directory, "r");
  try {
    await handle.sync();
  } finally {
    await handle.close();
  }
}

export function isEnoent(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}

function summarizeFunction(
  value: unknown,
  path: string,
): ManifestFunctionSummary {
  const fn = objectOf(value, path);
  const runtime = objectOf(fn["runtime"], `${path}.runtime`);
  const verification = objectOf(fn["verification"], `${path}.verification`);
  const provenance =
    fn["trainingProvenance"] === undefined
      ? {}
      : objectOf(fn["trainingProvenance"], `${path}.trainingProvenance`);
  const seed =
    provenance["seed"] === undefined
      ? null
      : numberOf(provenance["seed"], `${path}.trainingProvenance.seed`);
  const heldOutConstraints =
    verification["heldOutConstraints"] === undefined
      ? null
      : summarizeHeldOut(
          verification["heldOutConstraints"],
          `${path}.verification.heldOutConstraints`,
        );
  const heads = listOf(fn["heads"], `${path}.heads`).map((entry, index) => {
    const headPath = `${path}.heads[${String(index)}]`;
    const head = objectOf(entry, headPath);
    const headVerification = objectOf(
      head["verification"],
      `${headPath}.verification`,
    );
    const calibration = objectOf(
      head["calibration"],
      `${headPath}.calibration`,
    );
    return {
      outputPath: listOf(head["outputPath"], `${headPath}.outputPath`).map(
        (segment, position) =>
          stringOf(segment, `${headPath}.outputPath[${String(position)}]`),
      ),
      accuracy: numberOf(
        headVerification["accuracy"],
        `${headPath}.verification.accuracy`,
      ),
      pairConsistency: numberOf(
        headVerification["pairConsistency"],
        `${headPath}.verification.pairConsistency`,
      ),
      ece: numberOf(calibration["ece"], `${headPath}.calibration.ece`),
    };
  });
  return {
    id: stringOf(fn["id"], `${path}.id`),
    resultMode: stringOf(runtime["resultMode"], `${path}.runtime.resultMode`),
    status: stringOf(verification["status"], `${path}.verification.status`),
    accuracy: numberOf(
      verification["accuracy"],
      `${path}.verification.accuracy`,
    ),
    ece: numberOf(verification["ece"], `${path}.verification.ece`),
    brier: numberOf(verification["brier"], `${path}.verification.brier`),
    pairConsistency: numberOf(
      verification["pairConsistency"],
      `${path}.verification.pairConsistency`,
    ),
    attestedCases: numberOf(
      verification["attestedCases"],
      `${path}.verification.attestedCases`,
    ),
    exampleFailures: numberOf(
      verification["exampleFailures"],
      `${path}.verification.exampleFailures`,
    ),
    constraintViolations: numberOf(
      verification["constraintViolations"],
      `${path}.verification.constraintViolations`,
    ),
    seed,
    heldOutConstraints,
    heads,
  };
}

function summarizeHeldOut(
  value: unknown,
  path: string,
): HeldOutConstraintsSummary {
  const held = objectOf(value, path);
  return {
    sampleSize: numberOf(held["sampleSize"], `${path}.sampleSize`),
    violations: numberOf(held["violations"], `${path}.violations`),
    violationRate: numberOf(held["violationRate"], `${path}.violationRate`),
    seed: numberOf(held["seed"], `${path}.seed`),
  };
}

export async function readJson(file: string): Promise<unknown> {
  return JSON.parse(await readFile(file, "utf8")) as unknown;
}

/** What a release records about how one function was trained and when it gives way. */
export interface ManifestFunctionProvenance {
  readonly confidenceThreshold: number | null;
  readonly fallbackRef: string | null;
  readonly datasetSha256: string | null;
  readonly teacher: string | null;
  readonly baseModel: string | null;
}

export interface ArtifactRelease extends ArtifactSummary {
  readonly provenance: ReadonlyMap<string, ManifestFunctionProvenance>;
}

/**
 * The summary plus the release's build time and each function's training
 * provenance and confidence policy: which dataset, teacher and base model an
 * answer came from (`semantscript explain`).
 */
export async function readArtifactRelease(
  root: string,
): Promise<ArtifactRelease> {
  const summary = await readArtifactSummary(root);
  const manifest = objectOf(
    await readJson(join(root, summary.release, "manifest.json")),
    "manifest",
  );
  const provenance = new Map<string, ManifestFunctionProvenance>();
  for (const entry of listOf(manifest["functions"], "manifest.functions")) {
    const fn = objectOf(entry, "manifest function");
    const runtime = optionalObject(fn["runtime"]);
    const training = optionalObject(fn["trainingProvenance"]);
    provenance.set(stringOf(fn["id"], "manifest function id"), {
      confidenceThreshold:
        typeof runtime["confidenceThreshold"] === "number"
          ? runtime["confidenceThreshold"]
          : null,
      fallbackRef: optionalString(runtime["fallbackRef"]),
      datasetSha256: optionalString(training["datasetSha256"]),
      teacher: optionalString(training["teacher"]),
      baseModel: optionalString(training["baseModel"]),
    });
  }
  return { ...summary, provenance };
}

function optionalObject(value: unknown): Readonly<Record<string, unknown>> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Readonly<Record<string, unknown>>)
    : {};
}

function optionalString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}
