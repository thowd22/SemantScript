/**
 * Structured-cloneable inference plan consumed by the runtime worker.
 *
 * The artifact loader owns validation of the manifest and resources. This plan
 * deliberately contains only the verified bytes and normalized v1 bindings the
 * inference worker needs.
 */
export interface StagedInferencePlan {
  readonly kind: "onnx";
  readonly tokenizerJson: Uint8Array;
  /** The application's encoder: the full shared stack, or its deepest exported prefix. */
  readonly encoderModel: Uint8Array;
  readonly encoderAbi: InferenceOnnxAbi;
  /**
   * Depth-routed encoder prefixes beyond the application's encoder, keyed by
   * ref; a function names one through `encoderRef` (TASK-6.7).
   */
  readonly encoders?: readonly StagedEncoderPlan[];
  readonly adapters: readonly StagedAdapterPlan[];
  readonly functions: readonly StagedFunctionPlan[];
  readonly maximumSequenceLength: number;
}

export interface StagedEncoderPlan {
  readonly ref: string;
  readonly model: Uint8Array;
  readonly abi: InferenceOnnxAbi;
}

export interface StagedAdapterPlan {
  readonly ref: string;
  readonly model: Uint8Array;
  readonly abi: InferenceOnnxAbi;
}

export interface StagedFunctionPlan {
  readonly id: string;
  readonly adapterRef: string;
  /** The encoder prefix this function's adapter reads; absent means the application's encoder. */
  readonly encoderRef?: string;
  readonly diagnosticsRequired: boolean;
  readonly heads: readonly StagedHeadPlan[];
}

export interface StagedHeadPlan extends InferenceHeadPlan {
  readonly model: Uint8Array;
  readonly abi: InferenceOnnxAbi;
}

export interface InferenceOnnxAbi {
  readonly inputs: readonly InferenceTensorDescriptor[];
  readonly outputs: readonly InferenceTensorDescriptor[];
}

export interface InferenceTensorDescriptor {
  readonly name: string;
  readonly dtype: "int64" | "float16" | "float32";
  readonly shape: readonly (number | string)[];
}

export type InferenceSupportValue = boolean | string | number;

export type InferencePlainValue =
  InferenceSupportValue | Readonly<Record<string, InferenceSupportValue>>;

export type InferenceExpectedValueMode = "none" | "zero-based-rank" | "numeric";

export interface InferenceDistributionEntry {
  readonly value: InferenceSupportValue;
  readonly probability: number;
}

export interface InferenceScalarDiagnostic {
  readonly value: InferenceSupportValue;
  readonly confidence: number;
  readonly uncertainty: number;
  readonly distribution: readonly InferenceDistributionEntry[];
  readonly expectedValue: number | null;
}

export interface InferenceObjectDiagnostic {
  readonly value: Readonly<Record<string, InferenceSupportValue>>;
  readonly minimumFieldConfidence: number;
  readonly maximumFieldUncertainty: number;
  readonly fields: Readonly<Record<string, InferenceScalarDiagnostic>>;
}

export type InferenceDiagnosticResult =
  InferenceScalarDiagnostic | InferenceObjectDiagnostic;

export type InferenceResultValue =
  InferencePlainValue | InferenceDiagnosticResult;

export type InferenceWorkerResult =
  | { readonly kind: "value"; readonly result: InferencePlainValue }
  | { readonly kind: "scalar"; readonly result: InferenceScalarDiagnostic }
  | { readonly kind: "object"; readonly result: InferenceObjectDiagnostic };

// JSON's longest finite binary64 spelling is currently 25 UTF-8 bytes (for
// example, -0.0000063354006458703095). Keep the response quota independent of
// engine-specific shortest-number choices by reserving that full width.
export const MAXIMUM_SERIALIZED_FINITE_NUMBER_BYTES = 25;

const utf8Encoder = new TextEncoder();

/**
 * Serializes the deliberately narrow inference result protocol. Unlike
 * JSON.stringify, this preserves IEEE-754 negative zero as the valid JSON
 * number literal `-0`; every other finite number uses JSON's exact round-trip
 * representation.
 */
export function stringifyInferenceResult(value: InferenceWorkerResult): string {
  switch (value.kind) {
    case "value":
      return `{"kind":"value","result":${stringifyInferencePlainValue(value.result)}}`;
    case "scalar":
      return `{"kind":"scalar","result":${stringifyScalarDiagnostic(value.result)}}`;
    case "object":
      return `{"kind":"object","result":${stringifyObjectDiagnostic(value.result)}}`;
  }
}

function stringifyInferencePlainValue(value: InferencePlainValue): string {
  if (typeof value !== "object") {
    return stringifyInferenceSupportValue(value);
  }
  return `{${Object.entries(value)
    .map(
      ([field, fieldValue]) =>
        `${JSON.stringify(field)}:${stringifyInferenceSupportValue(fieldValue)}`,
    )
    .join(",")}}`;
}

function stringifyScalarDiagnostic(value: InferenceScalarDiagnostic): string {
  const distribution = value.distribution
    .map(
      (entry) =>
        `{"value":${stringifyInferenceSupportValue(entry.value)},` +
        `"probability":${stringifyFiniteNumber(entry.probability)}}`,
    )
    .join(",");
  const expectedValue =
    value.expectedValue === null
      ? "null"
      : stringifyFiniteNumber(value.expectedValue);
  return (
    `{"value":${stringifyInferenceSupportValue(value.value)},` +
    `"confidence":${stringifyFiniteNumber(value.confidence)},` +
    `"uncertainty":${stringifyFiniteNumber(value.uncertainty)},` +
    `"distribution":[${distribution}],"expectedValue":${expectedValue}}`
  );
}

function stringifyObjectDiagnostic(value: InferenceObjectDiagnostic): string {
  const plainValue = stringifyInferencePlainValue(value.value);
  const fields = Object.entries(value.fields)
    .map(
      ([field, diagnostic]) =>
        `${JSON.stringify(field)}:${stringifyScalarDiagnostic(diagnostic)}`,
    )
    .join(",");
  return (
    `{"value":${plainValue},` +
    `"minimumFieldConfidence":${stringifyFiniteNumber(value.minimumFieldConfidence)},` +
    `"maximumFieldUncertainty":${stringifyFiniteNumber(value.maximumFieldUncertainty)},` +
    `"fields":{${fields}}}`
  );
}

export function stringifyInferenceSupportValue(
  value: InferenceSupportValue,
): string {
  if (typeof value === "number") {
    return stringifyFiniteNumber(value);
  }
  return JSON.stringify(value);
}

function stringifyFiniteNumber(value: number): string {
  if (!Number.isFinite(value)) {
    throw new TypeError("inference result numbers must be finite");
  }
  return Object.is(value, -0) ? "-0" : JSON.stringify(value);
}

export interface InferenceHeadPlan {
  readonly outputPath: readonly [] | readonly [string];
  readonly parameterization: "binary-sigmoid" | "categorical-softmax";
  readonly support: readonly InferenceSupportValue[];
  readonly temperature: number;
  readonly expectedValueMode: InferenceExpectedValueMode;
}

/** Deterministic worker backend used by bounded unit tests without model files. */
export interface TestInferencePlan {
  readonly kind: "test";
  readonly functions: readonly TestFunctionPlan[];
  readonly initializationDelayMilliseconds?: number;
  readonly initializationError?: string;
}

export interface TestFunctionPlan {
  readonly id: string;
  readonly diagnosticsRequired: boolean;
  readonly heads: readonly TestHeadPlan[];
  readonly delayMilliseconds?: number;
  readonly error?: string;
}

export interface TestHeadPlan extends InferenceHeadPlan {
  readonly logits: readonly number[];
}

export interface InferenceResponsePlan {
  readonly diagnosticsRequired: boolean;
  readonly heads: readonly InferenceHeadPlan[];
}

/**
 * Returns a conservative UTF-8 upper bound for one success response. Once the
 * bound exceeds `cap`, the function saturates at `cap + 1` so callers can reject
 * oversized artifacts without constructing a worst-case payload.
 */
export function maximumInferenceResponseBytes(
  plan: InferenceResponsePlan,
  cap = Number.MAX_SAFE_INTEGER - 1,
): number {
  if (!Number.isSafeInteger(cap) || cap < 1 || cap >= Number.MAX_SAFE_INTEGER) {
    throw new RangeError(
      "inference response byte cap must be a positive safe integer",
    );
  }
  const counter = new ResponseSizeCounter(cap);
  const scalar =
    plan.heads.length === 1 && plan.heads[0]?.outputPath.length === 0;
  counter.add(plan.diagnosticsRequired ? 26 : 25);
  if (plan.diagnosticsRequired) {
    if (scalar) {
      const head = plan.heads[0];
      if (head === undefined) {
        throw new TypeError(
          "scalar inference response plan is missing its head",
        );
      }
      addScalarDiagnosticBytes(counter, head);
    } else {
      addObjectDiagnosticBytes(counter, plan.heads);
    }
  } else {
    addPlainValueBytes(counter, plan.heads, scalar);
  }
  counter.add(1);
  return counter.value;
}

class ResponseSizeCounter {
  readonly #cap: number;
  value = 0;

  constructor(cap: number) {
    this.#cap = cap;
  }

  get exceeded(): boolean {
    return this.value > this.#cap;
  }

  get remaining(): number {
    return this.exceeded ? 0 : this.#cap - this.value;
  }

  add(bytes: number): void {
    if (this.value > this.#cap || bytes > this.#cap - this.value) {
      this.value = this.#cap + 1;
      return;
    }
    this.value += bytes;
  }
}

function addPlainValueBytes(
  counter: ResponseSizeCounter,
  heads: readonly InferenceHeadPlan[],
  scalar: boolean,
): void {
  if (scalar) {
    const head = heads[0];
    if (head === undefined) {
      throw new TypeError("scalar inference response plan is missing its head");
    }
    addMaximumSupportValueBytes(counter, head);
    return;
  }

  counter.add(2);
  for (let index = 0; index < heads.length; index += 1) {
    if (counter.exceeded) return;
    const head = heads[index];
    const field = head?.outputPath[0];
    if (head === undefined || field === undefined) {
      throw new TypeError(
        "object inference response heads require one-segment output paths",
      );
    }
    if (index > 0) counter.add(1);
    counter.add(utf8Length(JSON.stringify(field)) + 1);
    addMaximumSupportValueBytes(counter, head);
  }
}

function addScalarDiagnosticBytes(
  counter: ResponseSizeCounter,
  head: InferenceHeadPlan,
): void {
  counter.add(9);
  addMaximumSupportValueBytes(counter, head);
  counter.add(14 + MAXIMUM_SERIALIZED_FINITE_NUMBER_BYTES);
  counter.add(15 + MAXIMUM_SERIALIZED_FINITE_NUMBER_BYTES);
  counter.add(17);
  for (let index = 0; index < head.support.length; index += 1) {
    if (counter.exceeded) return;
    if (index > 0) counter.add(1);
    counter.add(9 + supportValueBytes(head.support[index]) + 15);
    counter.add(MAXIMUM_SERIALIZED_FINITE_NUMBER_BYTES + 1);
  }
  counter.add(18);
  counter.add(
    head.expectedValueMode === "none"
      ? 4
      : MAXIMUM_SERIALIZED_FINITE_NUMBER_BYTES,
  );
  counter.add(1);
}

function addObjectDiagnosticBytes(
  counter: ResponseSizeCounter,
  heads: readonly InferenceHeadPlan[],
): void {
  counter.add(9);
  addPlainValueBytes(counter, heads, false);
  counter.add(26 + MAXIMUM_SERIALIZED_FINITE_NUMBER_BYTES);
  counter.add(27 + MAXIMUM_SERIALIZED_FINITE_NUMBER_BYTES);
  counter.add(11);
  for (let index = 0; index < heads.length; index += 1) {
    if (counter.exceeded) return;
    const head = heads[index];
    const field = head?.outputPath[0];
    if (head === undefined || field === undefined) {
      throw new TypeError(
        "object inference response heads require one-segment output paths",
      );
    }
    if (index > 0) counter.add(1);
    counter.add(utf8Length(JSON.stringify(field)) + 1);
    addScalarDiagnosticBytes(counter, head);
  }
  counter.add(2);
}

function addMaximumSupportValueBytes(
  counter: ResponseSizeCounter,
  head: InferenceHeadPlan,
): void {
  let maximum = 0;
  for (const value of head.support) {
    maximum = Math.max(maximum, supportValueBytes(value));
    if (maximum > counter.remaining) {
      counter.add(maximum);
      return;
    }
  }
  if (maximum === 0) {
    throw new TypeError("inference head support must not be empty");
  }
  counter.add(maximum);
}

function supportValueBytes(value: InferenceSupportValue | undefined): number {
  if (value === undefined) {
    throw new TypeError("inference head support contains an absent value");
  }
  return utf8Length(stringifyInferenceSupportValue(value));
}

function utf8Length(value: string): number {
  return utf8Encoder.encode(value).byteLength;
}

export type InferenceWorkerPlan = StagedInferencePlan | TestInferencePlan;

export const INFERENCE_CONTROL = {
  state: 0,
  sequence: 1,
  payloadLength: 2,
  errorCode: 3,
  /** Model passes the last successful single call performed, after in-scope sharing. */
  passesEncoder: 4,
  passesAdapter: 5,
  passesHead: 6,
  length: 7,
} as const;

export const INFERENCE_STATE = {
  idle: 0,
  pending: 1,
  success: 2,
  error: 3,
} as const;

export const INFERENCE_ERROR = {
  none: 0,
  unknownFunction: 1,
  invalidInput: 2,
  backend: 3,
  invalidResult: 4,
  busy: 5,
  responseTooLarge: 6,
  protocol: 7,
} as const;

export type InferenceWorkerErrorCode =
  | "unknown-function"
  | "invalid-input"
  | "backend"
  | "invalid-result"
  | "busy"
  | "response-too-large"
  | "protocol";

export interface InitializeInferenceMessage {
  readonly kind: "initialize";
  readonly plan: InferenceWorkerPlan;
}

export interface InvokeInferenceMessage {
  readonly kind: "invoke";
  readonly sequence: number;
  readonly functionId: string;
  readonly canonicalInput: Uint8Array;
  readonly controlBuffer: SharedArrayBuffer;
  readonly responseBuffer: SharedArrayBuffer;
  /**
   * A request scope (TASK-8.1): within one scope, sentence and function
   * embeddings are kept so sibling calls over the same input share passes.
   */
  readonly scopeId?: number;
}

export interface EndScopeInferenceMessage {
  readonly kind: "end-scope";
  readonly scopeId: number;
}

export interface InferenceStageRequest {
  readonly functionId: string;
  readonly canonicalInput: Uint8Array;
}

/** Model passes one stage actually performed: the fusion is observable, not assumed. */
export interface InferenceStagePasses {
  readonly encoder: number;
  readonly adapter: number;
  readonly head: number;
}

export interface InvokeStageInferenceMessage {
  readonly kind: "invoke-stage";
  readonly sequence: number;
  readonly requests: readonly InferenceStageRequest[];
  readonly controlBuffer: SharedArrayBuffer;
  readonly responseBuffer: SharedArrayBuffer;
}

export interface ShutdownInferenceMessage {
  readonly kind: "shutdown";
}

export type InferenceWorkerRequest =
  | InitializeInferenceMessage
  | InvokeInferenceMessage
  | InvokeStageInferenceMessage
  | EndScopeInferenceMessage
  | ShutdownInferenceMessage;

export const MAXIMUM_STAGE_REQUESTS = 64;

export interface InferenceReadyMessage {
  readonly kind: "ready";
  readonly functionIds: readonly string[];
}

export interface InferenceInitializationErrorMessage {
  readonly kind: "initialization-error";
  readonly message: string;
}

export type InferenceWorkerResponse =
  InferenceReadyMessage | InferenceInitializationErrorMessage;

export function workerErrorCodeName(code: number): InferenceWorkerErrorCode {
  switch (code) {
    case INFERENCE_ERROR.unknownFunction:
      return "unknown-function";
    case INFERENCE_ERROR.invalidInput:
      return "invalid-input";
    case INFERENCE_ERROR.backend:
      return "backend";
    case INFERENCE_ERROR.invalidResult:
      return "invalid-result";
    case INFERENCE_ERROR.busy:
      return "busy";
    case INFERENCE_ERROR.responseTooLarge:
      return "response-too-large";
    case INFERENCE_ERROR.protocol:
    case INFERENCE_ERROR.none:
    default:
      return "protocol";
  }
}
