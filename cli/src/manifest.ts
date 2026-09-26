import { readFile } from "node:fs/promises";
import { join } from "node:path";

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
  readonly heads: readonly ManifestHeadSummary[];
}

export interface ArtifactSummary {
  readonly root: string;
  readonly release: string;
  readonly manifestSha256: string;
  readonly applicationId: string;
  readonly applicationVersion: string;
  readonly functions: readonly ManifestFunctionSummary[];
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
  const manifest = objectOf(
    await readJson(join(root, release, "manifest.json")),
    "manifest",
  );
  if (manifest["kind"] !== "semantscript.application-artifact") {
    throw new Error("manifest.kind must be semantscript.application-artifact");
  }
  const application = objectOf(manifest["application"], "manifest.application");
  const functions = listOf(manifest["functions"], "manifest.functions").map(
    (entry, index) =>
      summarizeFunction(entry, `manifest.functions[${String(index)}]`),
  );
  return {
    root,
    release,
    manifestSha256,
    applicationId: stringOf(application["id"], "manifest.application.id"),
    applicationVersion: stringOf(
      application["version"],
      "manifest.application.version",
    ),
    functions,
  };
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
    heads,
  };
}

export async function readJson(file: string): Promise<unknown> {
  return JSON.parse(await readFile(file, "utf8")) as unknown;
}
