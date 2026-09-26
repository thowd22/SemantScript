import type { ApplicationArtifactManifestV1 } from "./artifact-types.js";
import type { InferenceResponsePlan } from "./inference-protocol.js";

/**
 * In-process answers for stub artifacts (TASK-14.8). A stub artifact is an
 * ordinary on-disk release whose functions carry stub provenance; the runtime
 * loads it through the normal path and, after the real worker has run, asks
 * the answerer registered under the release's manifest digest for the value.
 * Answers never leave the process that registered them, so a stub release
 * copied anywhere else refuses to load.
 */

/** The teacher and base model a stub artifact's functions record. */
export const STUB_PROVENANCE = "semantscript-stub";

export interface StubAnswerRequest {
  readonly functionId: string;
  readonly inputs: Readonly<Record<string, unknown>>;
  readonly canonicalInput: Uint8Array;
  /** The function's heads and whether its call returns diagnostics, as the worker sees them. */
  readonly plan: InferenceResponsePlan;
}

/** Returns the decoded inference result the worker would have returned. */
export type StubAnswerer = (request: StubAnswerRequest) => unknown;

export type SemaStubErrorReason =
  | "unknown-function"
  | "unresolved-function"
  | "invalid-value"
  | "invalid-confidence"
  | "unanswered"
  | "unmatched-input"
  | "unregistered"
  | "invalid-bundle"
  | "occupied-directory";

/** A stub artifact was described or used inconsistently with the IR bundle. */
export class SemaStubError extends Error {
  readonly code = "SEMA_STUB_INVALID";
  readonly reason: SemaStubErrorReason;
  readonly functionId: string | undefined;

  constructor(
    reason: SemaStubErrorReason,
    message: string,
    functionId?: string,
  ) {
    super(message);
    this.name = "SemaStubError";
    this.reason = reason;
    this.functionId = functionId;
  }
}

const answerers = new Map<string, StubAnswerer>();

export function registerStubAnswerer(
  manifestSha256: string,
  answerer: StubAnswerer,
): void {
  answerers.set(manifestSha256, answerer);
}

export function unregisterStubAnswerer(manifestSha256: string): void {
  answerers.delete(manifestSha256);
}

export function isStubManifest(
  manifest: ApplicationArtifactManifestV1,
): boolean {
  return manifest.functions.some(
    (fn) =>
      fn.trainingProvenance.teacher === STUB_PROVENANCE ||
      fn.trainingProvenance.baseModel === STUB_PROVENANCE,
  );
}

/**
 * The answerer for a release about to be activated: undefined for a trained
 * artifact, and a typed refusal for a stub whose answers are not registered in
 * this process.
 */
export function stubAnswererFor(
  manifest: ApplicationArtifactManifestV1,
  manifestSha256: string,
): StubAnswerer | undefined {
  const answerer = answerers.get(manifestSha256);
  if (answerer !== undefined) {
    return answerer;
  }
  if (isStubManifest(manifest)) {
    throw new SemaStubError(
      "unregistered",
      `artifact ${manifestSha256} is a SemantScript test stub with no answers registered in this process; stubs load only through @semantscript/core/testing`,
    );
  }
  return undefined;
}
