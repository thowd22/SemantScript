import { isMarkedAsUntransferable, Worker } from "node:worker_threads";

import {
  INFERENCE_CONTROL,
  type InferenceStagePasses,
  type InferenceStageRequest,
  MAXIMUM_STAGE_REQUESTS,
  INFERENCE_STATE,
  type InferenceDistributionEntry,
  type InferenceExpectedValueMode,
  type InferenceHeadPlan,
  type InferenceObjectDiagnostic,
  type InferencePlainValue,
  type InferenceResponsePlan,
  type InferenceResultValue,
  type InferenceScalarDiagnostic,
  type InferenceSupportValue,
  type InferenceWorkerErrorCode,
  type InferenceWorkerPlan,
  type InferenceWorkerResponse,
  workerErrorCodeName,
} from "./inference-protocol.js";
import { parseStrictJson } from "./strict-json.js";

const DEFAULT_INITIALIZATION_TIMEOUT_MILLISECONDS = 30_000;
const DEFAULT_INFERENCE_TIMEOUT_MILLISECONDS = 30_000;
const DEFAULT_MAXIMUM_INPUT_BYTES = 1_048_576;
const DEFAULT_RESPONSE_BUFFER_BYTES = 65_536;
const DEFAULT_WORKER_MEMORY_LIMIT_MEGABYTES = 512;
const MAXIMUM_SEQUENCE = 0x7fff_ffff;
const MAXIMUM_RESPONSE_JSON_BYTES = 1_048_576;
const RESPONSE_FLOAT_TOLERANCE_ULPS = 8;

const textDecoder = new TextDecoder("utf-8", { fatal: true });
const textEncoder = new TextEncoder();

export interface InferenceRuntimeOptions {
  readonly initializationTimeoutMilliseconds?: number;
  readonly inferenceTimeoutMilliseconds?: number;
  readonly maximumInputBytes?: number;
  readonly responseBufferBytes?: number;
  readonly workerMemoryLimitMegabytes?: number;
  /**
   * Transfer ownership of exact, standalone ArrayBuffers to the worker. This is
   * enabled by default to avoid cloning large model files. After initialization
   * starts, transferred views in the staged plan are detached.
   */
  readonly transferModelBuffers?: boolean;
}

export interface InferenceStageResult {
  /** One decoded result per request, in request order. */
  readonly results: readonly unknown[];
  readonly passes: InferenceStagePasses;
}

export interface InferenceRuntime {
  readonly functionIds: ReadonlySet<string>;
  readonly maximumInputBytes: number;
  /** Model passes the last successful `call` performed, after in-scope sharing. */
  readonly lastPasses: InferenceStagePasses | undefined;
  call(
    functionId: string,
    canonicalInput: Uint8Array,
    scopeId?: number,
  ): unknown;
  /** Lets the worker drop the embeddings a request scope kept. */
  endScope(scopeId: number): void;
  /** Executes several functions as one stage; identical inputs share the encoder pass. */
  callStage(requests: readonly InferenceStageRequest[]): InferenceStageResult;
  close(): Promise<void>;
}

export type SemaInferenceErrorCode =
  | InferenceWorkerErrorCode
  | "closed"
  | "initialization"
  | "invalid-input"
  | "protocol"
  | "timeout"
  | "unknown-function"
  | "worker-failed";

/** The inference worker failed; `code` distinguishes initialization, input, timeout, protocol and backend failures. */
export class SemaInferenceError extends Error {
  readonly code: SemaInferenceErrorCode;

  constructor(
    code: SemaInferenceErrorCode,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "SemaInferenceError";
    this.code = code;
  }
}

/** The inference worker could not start (typically a missing native ONNX Runtime or tokenizer binding). */
export class SemaInferenceInitializationError extends SemaInferenceError {
  constructor(message: string, options?: ErrorOptions) {
    super("initialization", message, options);
    this.name = "SemaInferenceInitializationError";
  }
}

/** The inference worker rejected a request's inputs. */
export class SemaInferenceInputError extends SemaInferenceError {
  constructor(message: string, options?: ErrorOptions) {
    super("invalid-input", message, options);
    this.name = "SemaInferenceInputError";
  }
}

/** An inference call exceeded its timeout. */
export class SemaInferenceTimeoutError extends SemaInferenceError {
  constructor(message: string, options?: ErrorOptions) {
    super("timeout", message, options);
    this.name = "SemaInferenceTimeoutError";
  }
}

/** The called function id is not in the loaded artifact. */
export class SemaUnknownFunctionError extends SemaInferenceError {
  readonly functionId: string;

  constructor(functionId: string) {
    super(
      "unknown-function",
      `unknown semantic function ${JSON.stringify(functionId)}`,
    );
    this.name = "SemaUnknownFunctionError";
    this.functionId = functionId;
  }
}

/**
 * Starts and completely initializes the worker before returning a synchronous
 * inference facade. The plan is structural so artifact-loader implementations
 * can adapt their validated types without an import cycle.
 */
export async function createInferenceRuntime(
  plan: InferenceWorkerPlan,
  options: InferenceRuntimeOptions = {},
): Promise<InferenceRuntime> {
  const normalizedOptions = normalizeOptions(options);
  let responseSchemas: ReadonlyMap<string, InferenceResponseSchema>;
  try {
    responseSchemas = buildInferenceResponseSchemas(plan);
  } catch (error) {
    throw new SemaInferenceInitializationError(
      "invalid inference response plan",
      { cause: error },
    );
  }
  const worker = new Worker(new URL("./inference-worker.js", import.meta.url), {
    execArgv: workerExecArgv(process.execArgv),
    resourceLimits: {
      maxOldGenerationSizeMb: normalizedOptions.workerMemoryLimitMegabytes,
    },
  });

  let functionIds: readonly string[];
  try {
    functionIds = await initializeWorker(worker, plan, normalizedOptions);
    assertInitializedFunctionIds(functionIds, responseSchemas);
  } catch (error) {
    await worker.terminate();
    if (error instanceof SemaInferenceError) {
      throw error;
    }
    throw new SemaInferenceInitializationError(
      "failed to initialize the inference worker",
      {
        cause: error,
      },
    );
  }

  return new WorkerInferenceRuntime(worker, responseSchemas, normalizedOptions);
}

interface NormalizedInferenceRuntimeOptions {
  readonly initializationTimeoutMilliseconds: number;
  readonly inferenceTimeoutMilliseconds: number;
  readonly maximumInputBytes: number;
  readonly responseBufferBytes: number;
  readonly transferModelBuffers: boolean;
  readonly workerMemoryLimitMegabytes: number;
}

class WorkerInferenceRuntime implements InferenceRuntime {
  readonly functionIds: ReadonlySet<string>;
  readonly maximumInputBytes: number;
  readonly #worker: Worker;
  readonly #control: Int32Array;
  readonly #response: Uint8Array;
  readonly #responseSchemas: ReadonlyMap<string, InferenceResponseSchema>;
  readonly #inferenceTimeoutMilliseconds: number;
  #sequence = 0;
  #closed = false;
  #fault: string | undefined;

  constructor(
    worker: Worker,
    responseSchemas: ReadonlyMap<string, InferenceResponseSchema>,
    options: NormalizedInferenceRuntimeOptions,
  ) {
    this.#worker = worker;
    this.#responseSchemas = responseSchemas;
    this.functionIds = new Set(responseSchemas.keys());
    this.#control = new Int32Array(
      new SharedArrayBuffer(
        Int32Array.BYTES_PER_ELEMENT * INFERENCE_CONTROL.length,
      ),
    );
    this.#response = new Uint8Array(
      new SharedArrayBuffer(options.responseBufferBytes),
    );
    this.#inferenceTimeoutMilliseconds = options.inferenceTimeoutMilliseconds;
    this.maximumInputBytes = options.maximumInputBytes;

    worker.on("error", (error) => {
      this.#fault ??= error.message;
    });
    worker.on("exit", (code) => {
      if (!this.#closed) {
        this.#fault ??= `inference worker exited unexpectedly with code ${String(code)}`;
      }
    });
  }

  lastPasses: InferenceStagePasses | undefined = undefined;

  endScope(scopeId: number): void {
    if (this.#closed || this.#fault !== undefined) return;
    try {
      this.#worker.postMessage({ kind: "end-scope", scopeId });
    } catch {
      // A scope that cannot be released only keeps a few tensors until the worker closes.
    }
  }

  call(
    functionId: string,
    canonicalInput: Uint8Array,
    scopeId?: number,
  ): unknown {
    this.#assertUsable();
    const responseSchema = this.#responseSchemas.get(functionId);
    if (responseSchema === undefined) {
      throw new SemaUnknownFunctionError(functionId);
    }
    validateCanonicalInput(canonicalInput, this.maximumInputBytes);

    const sequence = this.#nextSequence();
    this.#response.fill(0);
    Atomics.store(this.#control, INFERENCE_CONTROL.payloadLength, 0);
    Atomics.store(this.#control, INFERENCE_CONTROL.errorCode, 0);
    Atomics.store(this.#control, INFERENCE_CONTROL.sequence, sequence);
    Atomics.store(
      this.#control,
      INFERENCE_CONTROL.state,
      INFERENCE_STATE.pending,
    );

    try {
      this.#worker.postMessage({
        kind: "invoke",
        sequence,
        functionId,
        canonicalInput,
        controlBuffer: this.#control.buffer,
        responseBuffer: this.#response.buffer,
        ...(scopeId === undefined ? {} : { scopeId }),
      });
    } catch (error) {
      this.#fault = `failed to send inference request: ${errorMessage(error)}`;
      throw new SemaInferenceError("worker-failed", this.#fault, {
        cause: error,
      });
    }

    const waitResult = Atomics.wait(
      this.#control,
      INFERENCE_CONTROL.state,
      INFERENCE_STATE.pending,
      this.#inferenceTimeoutMilliseconds,
    );
    if (waitResult === "timed-out") {
      this.#fault =
        `inference request ${String(sequence)} timed out after ` +
        `${String(this.#inferenceTimeoutMilliseconds)}ms`;
      throw new SemaInferenceTimeoutError(this.#fault);
    }

    const observedSequence = Atomics.load(
      this.#control,
      INFERENCE_CONTROL.sequence,
    );
    const state = Atomics.load(this.#control, INFERENCE_CONTROL.state);
    const payloadLength = Atomics.load(
      this.#control,
      INFERENCE_CONTROL.payloadLength,
    );
    if (
      observedSequence !== sequence ||
      (state !== INFERENCE_STATE.success && state !== INFERENCE_STATE.error) ||
      payloadLength < 0 ||
      payloadLength > this.#response.length
    ) {
      this.#fault =
        "the inference worker returned an invalid synchronization response";
      throw new SemaInferenceError("protocol", this.#fault);
    }

    try {
      const responseBytes = this.#response.subarray(0, payloadLength);
      if (state === INFERENCE_STATE.error) {
        let payload: string;
        try {
          payload = textDecoder.decode(responseBytes);
        } catch (error) {
          this.#fault =
            "the inference worker returned an invalid UTF-8 response";
          throw new SemaInferenceError("protocol", this.#fault, {
            cause: error,
          });
        }
        const workerCode = workerErrorCodeName(
          Atomics.load(this.#control, INFERENCE_CONTROL.errorCode),
        );
        if (workerCode === "unknown-function") {
          throw new SemaUnknownFunctionError(functionId);
        }
        throw new SemaInferenceError(
          workerCode,
          payload || "the inference worker failed",
        );
      }

      try {
        const value = parseInferenceResultBytes(responseBytes, responseSchema);
        this.lastPasses = {
          encoder: Atomics.load(this.#control, INFERENCE_CONTROL.passesEncoder),
          adapter: Atomics.load(this.#control, INFERENCE_CONTROL.passesAdapter),
          head: Atomics.load(this.#control, INFERENCE_CONTROL.passesHead),
        };
        return value;
      } catch (error) {
        this.#fault = "the inference worker returned an invalid result payload";
        throw new SemaInferenceError("protocol", this.#fault, { cause: error });
      }
    } finally {
      Atomics.store(
        this.#control,
        INFERENCE_CONTROL.state,
        INFERENCE_STATE.idle,
      );
    }
  }

  callStage(requests: readonly InferenceStageRequest[]): InferenceStageResult {
    this.#assertUsable();
    if (
      !isRequestList(requests) ||
      requests.length === 0 ||
      requests.length > MAXIMUM_STAGE_REQUESTS
    ) {
      throw new SemaInferenceInputError(
        `a stage carries between 1 and ${String(MAXIMUM_STAGE_REQUESTS)} requests`,
      );
    }
    const schemas = requests.map((request) => {
      const responseSchema = this.#responseSchemas.get(request.functionId);
      if (responseSchema === undefined) {
        throw new SemaUnknownFunctionError(request.functionId);
      }
      validateCanonicalInput(request.canonicalInput, this.maximumInputBytes);
      return responseSchema;
    });
    // Every single-call result fits the configured response buffer, so a stage
    // of N fits N buffers plus the framing header.
    const response = new Uint8Array(
      new SharedArrayBuffer(
        this.#response.length * requests.length + 64 + 24 * requests.length,
      ),
    );

    const sequence = this.#nextSequence();
    Atomics.store(this.#control, INFERENCE_CONTROL.payloadLength, 0);
    Atomics.store(this.#control, INFERENCE_CONTROL.errorCode, 0);
    Atomics.store(this.#control, INFERENCE_CONTROL.sequence, sequence);
    Atomics.store(
      this.#control,
      INFERENCE_CONTROL.state,
      INFERENCE_STATE.pending,
    );
    try {
      this.#worker.postMessage({
        kind: "invoke-stage",
        sequence,
        requests: requests.map((request) => ({
          functionId: request.functionId,
          canonicalInput: request.canonicalInput,
        })),
        controlBuffer: this.#control.buffer,
        responseBuffer: response.buffer,
      });
    } catch (error) {
      this.#fault = `failed to send inference request: ${errorMessage(error)}`;
      throw new SemaInferenceError("worker-failed", this.#fault, {
        cause: error,
      });
    }

    const waitResult = Atomics.wait(
      this.#control,
      INFERENCE_CONTROL.state,
      INFERENCE_STATE.pending,
      this.#inferenceTimeoutMilliseconds,
    );
    if (waitResult === "timed-out") {
      this.#fault =
        `inference request ${String(sequence)} timed out after ` +
        `${String(this.#inferenceTimeoutMilliseconds)}ms`;
      throw new SemaInferenceTimeoutError(this.#fault);
    }
    const observedSequence = Atomics.load(
      this.#control,
      INFERENCE_CONTROL.sequence,
    );
    const state = Atomics.load(this.#control, INFERENCE_CONTROL.state);
    const payloadLength = Atomics.load(
      this.#control,
      INFERENCE_CONTROL.payloadLength,
    );
    if (
      observedSequence !== sequence ||
      (state !== INFERENCE_STATE.success && state !== INFERENCE_STATE.error) ||
      payloadLength < 0 ||
      payloadLength > response.length
    ) {
      this.#fault =
        "the inference worker returned an invalid synchronization response";
      throw new SemaInferenceError("protocol", this.#fault);
    }
    try {
      const responseBytes = response.subarray(0, payloadLength);
      if (state === INFERENCE_STATE.error) {
        let payload: string;
        try {
          payload = textDecoder.decode(responseBytes);
        } catch (error) {
          this.#fault =
            "the inference worker returned an invalid UTF-8 response";
          throw new SemaInferenceError("protocol", this.#fault, {
            cause: error,
          });
        }
        const workerCode = workerErrorCodeName(
          Atomics.load(this.#control, INFERENCE_CONTROL.errorCode),
        );
        if (workerCode === "unknown-function") {
          throw new SemaUnknownFunctionError(requests[0]?.functionId ?? "");
        }
        throw new SemaInferenceError(
          workerCode,
          payload || "the inference worker failed",
        );
      }
      try {
        return decodeStageResponse(responseBytes, schemas);
      } catch (error) {
        this.#fault = "the inference worker returned an invalid result payload";
        throw new SemaInferenceError("protocol", this.#fault, { cause: error });
      }
    } finally {
      Atomics.store(
        this.#control,
        INFERENCE_CONTROL.state,
        INFERENCE_STATE.idle,
      );
    }
  }

  async close(): Promise<void> {
    if (this.#closed) {
      return;
    }
    this.#closed = true;
    const exit = waitForWorkerExit(this.#worker, 5_000);
    try {
      this.#worker.postMessage({ kind: "shutdown" });
      await exit;
    } catch {
      // terminate() below is the authoritative cleanup path after a worker fault.
    } finally {
      await this.#worker.terminate();
    }
  }

  #assertUsable(): void {
    if (this.#closed) {
      throw new SemaInferenceError("closed", "the inference runtime is closed");
    }
    if (this.#fault !== undefined) {
      throw new SemaInferenceError("worker-failed", this.#fault);
    }
  }

  #nextSequence(): number {
    this.#sequence =
      this.#sequence === MAXIMUM_SEQUENCE ? 1 : this.#sequence + 1;
    return this.#sequence;
  }
}

interface InferenceResponseHeadSchema {
  readonly field: string | undefined;
  readonly support: readonly InferenceSupportValue[];
  readonly expectedValueMode: InferenceExpectedValueMode;
}

interface InferenceResponseSchema {
  readonly kind: "value" | "scalar" | "object";
  readonly scalar: boolean;
  readonly heads: readonly InferenceResponseHeadSchema[];
}

function buildInferenceResponseSchemas(
  plan: InferenceWorkerPlan,
): ReadonlyMap<string, InferenceResponseSchema> {
  const schemas = new Map<string, InferenceResponseSchema>();
  for (const functionPlan of plan.functions) {
    if (typeof functionPlan.id !== "string" || functionPlan.id.length === 0) {
      throw new TypeError(
        "inference response function id must be a non-empty string",
      );
    }
    if (schemas.has(functionPlan.id)) {
      throw new TypeError(
        `duplicate inference response function id ${JSON.stringify(functionPlan.id)}`,
      );
    }
    schemas.set(functionPlan.id, buildInferenceResponseSchema(functionPlan));
  }
  return schemas;
}

function buildInferenceResponseSchema(
  plan: InferenceResponsePlan,
): InferenceResponseSchema {
  if (typeof plan.diagnosticsRequired !== "boolean") {
    throw new TypeError(
      "inference response diagnosticsRequired must be boolean",
    );
  }
  if (plan.heads.length === 0) {
    throw new TypeError(
      "inference response plan must contain at least one head",
    );
  }

  const scalar =
    plan.heads.length === 1 && plan.heads[0]?.outputPath.length === 0;
  const fields = new Set<string>();
  const insertionOrder: InferenceResponseHeadSchema[] = [];
  for (const head of plan.heads) {
    let field: string | undefined;
    if (scalar) {
      if (head.outputPath.length !== 0) {
        throw new TypeError(
          "scalar inference response head must have an empty output path",
        );
      }
    } else {
      field = head.outputPath.length === 1 ? head.outputPath[0] : undefined;
      if (field === undefined || fields.has(field)) {
        throw new TypeError(
          "object inference response fields must be unique single segments",
        );
      }
      fields.add(field);
    }
    insertionOrder.push(buildInferenceResponseHeadSchema(head, field));
  }

  let enumerableHeads: readonly InferenceResponseHeadSchema[] = insertionOrder;
  if (!scalar) {
    const enumerableOrder: Record<string, InferenceResponseHeadSchema> = {};
    for (const head of insertionOrder) {
      const field = head.field ?? missingSchemaField();
      Object.defineProperty(enumerableOrder, field, {
        configurable: true,
        enumerable: true,
        value: head,
        writable: true,
      });
    }
    // JSON object serialization and parsing use ECMAScript own-key order, which
    // puts integer-index field names before other fields regardless of insertion.
    enumerableHeads = Object.values(enumerableOrder);
  }
  const heads = Object.freeze(enumerableHeads);

  return Object.freeze({
    kind: plan.diagnosticsRequired ? (scalar ? "scalar" : "object") : "value",
    scalar,
    heads,
  });
}

function buildInferenceResponseHeadSchema(
  head: InferenceHeadPlan,
  field: string | undefined,
): InferenceResponseHeadSchema {
  if (!Array.isArray(head.support) || head.support.length < 2) {
    throw new TypeError(
      "inference response support must contain at least two values",
    );
  }
  const support: InferenceSupportValue[] = [];
  const supportKeys = new Set<string>();
  for (const value of head.support) {
    if (!isInferenceSupportValue(value)) {
      throw new TypeError(
        "inference response support contains an invalid value",
      );
    }
    const key = inferenceSupportKey(value);
    if (supportKeys.has(key)) {
      throw new TypeError("inference response support values must be unique");
    }
    supportKeys.add(key);
    support.push(value);
  }
  const expectedValueMode: unknown = head.expectedValueMode;
  if (
    expectedValueMode !== "none" &&
    expectedValueMode !== "zero-based-rank" &&
    expectedValueMode !== "numeric"
  ) {
    throw new TypeError("inference response expected-value mode is invalid");
  }
  if (
    expectedValueMode === "zero-based-rank" &&
    !support.every((value) => typeof value === "string")
  ) {
    throw new TypeError(
      "zero-based-rank response support must contain only strings",
    );
  }
  if (
    expectedValueMode === "numeric" &&
    !support.every((value) => typeof value === "number")
  ) {
    throw new TypeError("numeric response support must contain only numbers");
  }
  return Object.freeze({
    field,
    support: Object.freeze(support),
    expectedValueMode,
  });
}

function assertInitializedFunctionIds(
  actual: readonly string[],
  schemas: ReadonlyMap<string, InferenceResponseSchema>,
): void {
  const expected = [...schemas.keys()];
  if (
    !Array.isArray(actual) ||
    actual.length !== expected.length ||
    actual.some((functionId, index) => functionId !== expected[index])
  ) {
    throw new SemaInferenceInitializationError(
      "inference worker function ids do not match the staged response schemas",
    );
  }
}

function missingSchemaField(): never {
  throw new TypeError("object inference response schema is missing a field");
}

/** Test-facing entry point for the same plan-aware success decoder used by the runtime. */
function isRequestList(value: readonly InferenceStageRequest[]): boolean {
  // Plain JavaScript callers can pass anything; keep the runtime check without
  // letting Array.isArray widen the element type to any.
  return Array.isArray(value);
}

function decodeStageResponse(
  bytes: Uint8Array,
  schemas: readonly InferenceResponseSchema[],
): InferenceStageResult {
  const newline = bytes.indexOf(0x0a);
  if (newline < 0) throw new Error("stage response has no header line");
  const header = JSON.parse(
    textDecoder.decode(bytes.subarray(0, newline)),
  ) as unknown;
  if (header === null || typeof header !== "object" || Array.isArray(header)) {
    throw new Error("stage response header is not an object");
  }
  const { passes, lengths } = header as { passes?: unknown; lengths?: unknown };
  if (
    !Array.isArray(lengths) ||
    lengths.length !== schemas.length ||
    lengths.some(
      (length) => !Number.isSafeInteger(length) || (length as number) < 0,
    )
  ) {
    throw new Error("stage response lengths do not match the request count");
  }
  if (passes === null || typeof passes !== "object" || Array.isArray(passes)) {
    throw new Error("stage response passes are missing");
  }
  const counts = passes as {
    encoder?: unknown;
    adapter?: unknown;
    head?: unknown;
  };
  for (const value of [counts.encoder, counts.adapter, counts.head]) {
    if (!Number.isSafeInteger(value) || (value as number) < 0)
      throw new Error("stage pass counts are invalid");
  }
  let offset = newline + 1;
  const results: unknown[] = [];
  for (const [index, schema] of schemas.entries()) {
    const length = lengths[index] as number;
    if (offset + length > bytes.length)
      throw new Error("stage response is truncated");
    results.push(
      parseInferenceResultBytes(
        bytes.subarray(offset, offset + length),
        schema,
      ),
    );
    offset += length;
  }
  if (offset !== bytes.length)
    throw new Error("stage response has trailing bytes");
  return {
    results,
    passes: {
      encoder: counts.encoder as number,
      adapter: counts.adapter as number,
      head: counts.head as number,
    },
  };
}

export function parseInferenceResultPayload(
  payload: string,
  plan: InferenceResponsePlan,
): InferenceResultValue {
  return parseInferenceResultBytes(
    textEncoder.encode(payload),
    buildInferenceResponseSchema(plan),
  );
}

function parseInferenceResultBytes(
  payload: Uint8Array,
  schema: InferenceResponseSchema,
): InferenceResultValue {
  const value = parseStrictJson(payload, {
    maximumBytes: MAXIMUM_RESPONSE_JSON_BYTES,
    maximumDepth: 32,
    maximumNodes: 1_000_000,
  });
  const wire = inferenceRecord(value, "inference response");
  exactOrderedKeys(wire, ["kind", "result"], "inference response");
  if (wire["kind"] !== schema.kind) {
    throw new TypeError(
      `inference response kind must be ${JSON.stringify(schema.kind)}`,
    );
  }
  if (schema.kind === "value") {
    return parsePlainValue(wire["result"], schema, "inference value");
  }
  if (schema.kind === "scalar") {
    const head = schema.heads[0];
    if (head === undefined)
      throw new TypeError("scalar inference schema is missing its head");
    return parseScalarDiagnostic(
      wire["result"],
      head,
      "scalar inference diagnostic",
    );
  }
  return parseObjectDiagnostic(wire["result"], schema);
}

function parsePlainValue(
  value: unknown,
  schema: InferenceResponseSchema,
  description: string,
): InferencePlainValue {
  if (schema.scalar) {
    const head = schema.heads[0];
    if (head === undefined)
      throw new TypeError("scalar inference schema is missing its head");
    return plannedSupportValue(value, head, description);
  }
  const result = inferenceRecord(value, description);
  const fields = schema.heads.map(({ field }) => field ?? missingSchemaField());
  exactOrderedKeys(result, fields, description);
  const parsed: Record<string, InferenceSupportValue> = {};
  for (const head of schema.heads) {
    const field = head.field ?? missingSchemaField();
    Object.defineProperty(parsed, field, {
      configurable: true,
      enumerable: true,
      value: plannedSupportValue(
        result[field],
        head,
        `${description}.${JSON.stringify(field)}`,
      ),
      writable: true,
    });
  }
  return parsed;
}

function plannedSupportValue(
  value: unknown,
  head: InferenceResponseHeadSchema,
  description: string,
): InferenceSupportValue {
  const parsed = supportValue(value, description);
  if (!head.support.some((candidate) => sameSupportValue(candidate, parsed))) {
    throw new TypeError(`${description} is outside the planned support`);
  }
  return parsed;
}

function parseScalarDiagnostic(
  value: unknown,
  head: InferenceResponseHeadSchema,
  description: string,
): InferenceScalarDiagnostic {
  const result = inferenceRecord(value, description);
  exactOrderedKeys(
    result,
    ["value", "confidence", "uncertainty", "distribution", "expectedValue"],
    description,
  );
  const selectedValue = plannedSupportValue(
    result["value"],
    head,
    `${description}.value`,
  );
  const confidence = unitNumber(
    result["confidence"],
    `${description}.confidence`,
  );
  const uncertainty = unitNumber(
    result["uncertainty"],
    `${description}.uncertainty`,
  );
  const entries = result["distribution"];
  if (!Array.isArray(entries) || entries.length !== head.support.length) {
    throw new TypeError(
      `${description}.distribution must contain exactly ${String(head.support.length)} entries`,
    );
  }
  const distribution = entries.map((entry, index) =>
    parseDistributionEntry(
      entry,
      head.support[index],
      `${description}.distribution[${String(index)}]`,
    ),
  );
  const expectedValue = parseExpectedValue(
    result["expectedValue"],
    head,
    distribution,
    `${description}.expectedValue`,
  );

  let selectedIndex = 0;
  const probabilitySum = compensatedSum(
    distribution.length,
    (index) => distribution[index]?.probability ?? 0,
  );
  for (let index = 1; index < distribution.length; index += 1) {
    if (
      (distribution[index]?.probability ?? Number.NEGATIVE_INFINITY) >
      (distribution[selectedIndex]?.probability ?? Number.NEGATIVE_INFINITY)
    ) {
      selectedIndex = index;
    }
  }
  const selectedEntry = distribution[selectedIndex];
  if (
    selectedEntry === undefined ||
    !sameSupportValue(selectedEntry.value, selectedValue) ||
    !Object.is(selectedEntry.probability, confidence)
  ) {
    throw new TypeError(
      `${description} top-1 value and confidence do not match its distribution`,
    );
  }
  const tolerance = responseFloatTolerance(distribution.length);
  if (Math.abs(probabilitySum - 1) > tolerance) {
    throw new TypeError(`${description} probabilities do not sum to one`);
  }
  const calculatedUncertainty = decodedNormalizedEntropy(distribution);
  if (Math.abs(uncertainty - calculatedUncertainty) > tolerance) {
    throw new TypeError(
      `${description} uncertainty does not match its distribution`,
    );
  }
  return {
    value: selectedValue,
    confidence,
    uncertainty,
    distribution,
    expectedValue,
  };
}

function decodedNormalizedEntropy(
  distribution: readonly InferenceDistributionEntry[],
): number {
  let sum = 0;
  let correction = 0;
  for (const { probability } of distribution) {
    const value = probability === 0 ? 0 : -probability * Math.log(probability);
    const next = sum + value;
    correction +=
      Math.abs(sum) >= Math.abs(value)
        ? sum - next + value
        : value - next + sum;
    sum = next;
  }
  const entropy = sum + correction;
  if (entropy === 0) return 0;
  return Math.min(1, Math.max(0, entropy / Math.log(distribution.length)));
}

function parseDistributionEntry(
  value: unknown,
  expectedSupport: InferenceSupportValue | undefined,
  description: string,
): InferenceDistributionEntry {
  if (expectedSupport === undefined) {
    throw new TypeError(`${description} has no planned support value`);
  }
  const entry = inferenceRecord(value, description);
  exactOrderedKeys(entry, ["value", "probability"], description);
  const parsedValue = supportValue(entry["value"], `${description}.value`);
  if (!sameSupportValue(parsedValue, expectedSupport)) {
    throw new TypeError(
      `${description}.value does not match planned support order`,
    );
  }
  return {
    value: parsedValue,
    probability: unitNumber(entry["probability"], `${description}.probability`),
  };
}

function parseObjectDiagnostic(
  value: unknown,
  schema: InferenceResponseSchema,
): InferenceObjectDiagnostic {
  const description = "object inference diagnostic";
  const result = inferenceRecord(value, description);
  exactOrderedKeys(
    result,
    ["value", "minimumFieldConfidence", "maximumFieldUncertainty", "fields"],
    description,
  );
  const plainValue = parsePlainValue(
    result["value"],
    schema,
    `${description}.value`,
  );
  if (isInferenceSupportValue(plainValue))
    throw new TypeError(`${description}.value must be an object`);
  const fieldsValue = inferenceRecord(
    result["fields"],
    `${description}.fields`,
  );
  const fieldKeys = schema.heads.map(
    ({ field }) => field ?? missingSchemaField(),
  );
  exactOrderedKeys(fieldsValue, fieldKeys, `${description}.fields`);

  const fields: Record<string, InferenceScalarDiagnostic> = {};
  let minimumFieldConfidence = 1;
  let maximumFieldUncertainty = 0;
  for (const head of schema.heads) {
    const field = head.field ?? missingSchemaField();
    const diagnostic = parseScalarDiagnostic(
      fieldsValue[field],
      head,
      `${description}.fields[${JSON.stringify(field)}]`,
    );
    if (
      !sameSupportValue(
        plainValue[field] as InferenceSupportValue,
        diagnostic.value,
      )
    ) {
      throw new TypeError(
        `${description}.value does not match its field diagnostic`,
      );
    }
    Object.defineProperty(fields, field, {
      configurable: true,
      enumerable: true,
      value: diagnostic,
      writable: true,
    });
    minimumFieldConfidence = Math.min(
      minimumFieldConfidence,
      diagnostic.confidence,
    );
    maximumFieldUncertainty = Math.max(
      maximumFieldUncertainty,
      diagnostic.uncertainty,
    );
  }

  const encodedMinimum = unitNumber(
    result["minimumFieldConfidence"],
    `${description}.minimumFieldConfidence`,
  );
  const encodedMaximum = unitNumber(
    result["maximumFieldUncertainty"],
    `${description}.maximumFieldUncertainty`,
  );
  if (
    !Object.is(encodedMinimum, minimumFieldConfidence) ||
    !Object.is(encodedMaximum, maximumFieldUncertainty)
  ) {
    throw new TypeError(`${description} aggregates do not match its fields`);
  }
  return {
    value: plainValue,
    minimumFieldConfidence: encodedMinimum,
    maximumFieldUncertainty: encodedMaximum,
    fields,
  };
}

function inferenceRecord(
  value: unknown,
  description: string,
): Record<string, unknown> {
  if (value === null || Array.isArray(value) || typeof value !== "object") {
    throw new TypeError(`${description} must be an object`);
  }
  return value as Record<string, unknown>;
}

function exactOrderedKeys(
  value: Readonly<Record<string, unknown>>,
  expected: readonly string[],
  description: string,
): void {
  const keys = Object.keys(value);
  if (
    keys.length !== expected.length ||
    keys.some((key, index) => key !== expected[index])
  ) {
    throw new TypeError(`${description} has invalid fields`);
  }
}

function parseExpectedValue(
  value: unknown,
  head: InferenceResponseHeadSchema,
  distribution: readonly InferenceDistributionEntry[],
  description: string,
): number | null {
  if (head.expectedValueMode === "none") {
    if (value !== null)
      throw new TypeError(`${description} must be null for nominal support`);
    return null;
  }
  if (!isFiniteNumber(value)) {
    throw new TypeError(
      `${description} must be a finite number for ordinal support`,
    );
  }
  const calculated = decodedExpectedValue(head, distribution);
  const scale = expectedValueScale(head, calculated, value);
  if (
    Math.abs(value - calculated) >
    responseFloatTolerance(distribution.length, scale)
  ) {
    throw new TypeError(
      `${description} does not match its distribution and planned support`,
    );
  }
  return value;
}

function decodedExpectedValue(
  head: InferenceResponseHeadSchema,
  distribution: readonly InferenceDistributionEntry[],
): number {
  if (head.expectedValueMode === "zero-based-rank") {
    return compensatedSum(
      distribution.length,
      (index) => (distribution[index]?.probability ?? 0) * index,
    );
  }
  let scale = 1;
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (const supportValue of head.support) {
    if (typeof supportValue !== "number") {
      throw new TypeError(
        "numeric expected-value support must contain only numbers",
      );
    }
    scale = Math.max(scale, Math.abs(supportValue));
    minimum = Math.min(minimum, supportValue);
    maximum = Math.max(maximum, supportValue);
  }
  const normalized = compensatedSum(distribution.length, (index) => {
    const supportValue = head.support[index];
    if (typeof supportValue !== "number") {
      throw new TypeError(
        "numeric expected-value support must contain only numbers",
      );
    }
    return (distribution[index]?.probability ?? 0) * (supportValue / scale);
  });
  return clamp(normalized, minimum / scale, maximum / scale) * scale;
}

function expectedValueScale(
  head: InferenceResponseHeadSchema,
  calculated: number,
  encoded: number,
): number {
  let scale = Math.max(1, Math.abs(calculated), Math.abs(encoded));
  if (head.expectedValueMode === "zero-based-rank") {
    return Math.max(scale, head.support.length - 1);
  }
  for (const value of head.support) {
    if (typeof value === "number") scale = Math.max(scale, Math.abs(value));
  }
  return scale;
}

/**
 * Decoder recomputation follows the worker's compensated sums. Eight binary64
 * epsilons per summed term, scaled to the result magnitude, cover only normal
 * operation-order rounding and do not mask a materially different response.
 */
function responseFloatTolerance(termCount: number, scale = 1): number {
  return Math.max(
    Number.EPSILON,
    RESPONSE_FLOAT_TOLERANCE_ULPS *
      Math.max(1, termCount) *
      Number.EPSILON *
      Math.max(1, scale),
  );
}

function compensatedSum(
  length: number,
  valueAt: (index: number) => number,
): number {
  let sum = 0;
  let correction = 0;
  for (let index = 0; index < length; index += 1) {
    const value = valueAt(index);
    const next = sum + value;
    correction +=
      Math.abs(sum) >= Math.abs(value)
        ? sum - next + value
        : value - next + sum;
    sum = next;
  }
  return sum + correction;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function supportValue(
  value: unknown,
  description: string,
): InferenceSupportValue {
  if (!isInferenceSupportValue(value)) {
    throw new TypeError(`${description} must be a finite support value`);
  }
  return value;
}

function unitNumber(value: unknown, description: string): number {
  if (
    !isFiniteNumber(value) ||
    value < 0 ||
    value > 1 ||
    Object.is(value, -0)
  ) {
    throw new TypeError(`${description} must be a number in [0,1]`);
  }
  return value;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function sameSupportValue(
  left: InferenceSupportValue,
  right: InferenceSupportValue,
): boolean {
  return typeof left === "number" && typeof right === "number"
    ? Object.is(left, right)
    : left === right;
}

function inferenceSupportKey(value: InferenceSupportValue): string {
  if (typeof value === "number") {
    return `number:${Object.is(value, -0) ? "-0" : value.toString()}`;
  }
  return `${typeof value}:${String(value)}`;
}

function isInferenceSupportValue(
  value: unknown,
): value is InferenceSupportValue {
  return (
    typeof value === "boolean" ||
    typeof value === "string" ||
    (typeof value === "number" && Number.isFinite(value))
  );
}

async function initializeWorker(
  worker: Worker,
  plan: InferenceWorkerPlan,
  options: NormalizedInferenceRuntimeOptions,
): Promise<readonly string[]> {
  const transferList = options.transferModelBuffers
    ? collectTransferableBuffers(plan)
    : [];

  return new Promise<readonly string[]>((resolve, reject) => {
    let settled = false;
    const finish = (callback: () => void): void => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeout);
      worker.off("message", onMessage);
      worker.off("error", onError);
      worker.off("exit", onExit);
      callback();
    };
    const onMessage = (response: InferenceWorkerResponse): void => {
      if (response.kind === "ready") {
        finish(() => {
          resolve(response.functionIds);
        });
      } else {
        finish(() => {
          reject(new SemaInferenceInitializationError(response.message));
        });
      }
    };
    const onError = (error: Error): void => {
      finish(() => {
        reject(
          new SemaInferenceInitializationError(
            `inference worker failed: ${error.message}`,
            {
              cause: error,
            },
          ),
        );
      });
    };
    const onExit = (code: number): void => {
      finish(() => {
        reject(
          new SemaInferenceInitializationError(
            `inference worker exited during initialization with code ${String(code)}`,
          ),
        );
      });
    };
    const timeout = setTimeout(() => {
      finish(() => {
        reject(
          new SemaInferenceTimeoutError(
            "inference worker initialization timed out after " +
              `${String(options.initializationTimeoutMilliseconds)}ms`,
          ),
        );
      });
    }, options.initializationTimeoutMilliseconds);

    worker.on("message", onMessage);
    worker.on("error", onError);
    worker.on("exit", onExit);
    try {
      worker.postMessage({ kind: "initialize", plan }, transferList);
    } catch (error) {
      finish(() => {
        reject(
          new SemaInferenceInitializationError(
            "failed to send the staged plan to the inference worker",
            {
              cause: error,
            },
          ),
        );
      });
    }
  });
}

function collectTransferableBuffers(plan: InferenceWorkerPlan): ArrayBuffer[] {
  if (plan.kind !== "onnx") {
    return [];
  }

  const buffers = new Set<ArrayBuffer>();
  const consider = (bytes: Uint8Array): void => {
    const buffer = bytes.buffer;
    if (
      buffer instanceof ArrayBuffer &&
      bytes.byteOffset === 0 &&
      bytes.byteLength === buffer.byteLength &&
      !isMarkedAsUntransferable(buffer)
    ) {
      buffers.add(buffer);
    }
  };

  consider(plan.tokenizerJson);
  consider(plan.encoderModel);
  for (const encoder of plan.encoders ?? []) {
    consider(encoder.model);
  }
  for (const adapter of plan.adapters) {
    consider(adapter.model);
  }
  for (const functionPlan of plan.functions) {
    for (const head of functionPlan.heads) {
      consider(head.model);
    }
  }
  return [...buffers];
}

function validateCanonicalInput(
  input: Uint8Array,
  maximumInputBytes: number,
): void {
  if (!(input instanceof Uint8Array) || input.length === 0) {
    throw new SemaInferenceInputError(
      "canonical input must be a non-empty Uint8Array",
    );
  }
  if (input.length > maximumInputBytes) {
    throw new SemaInferenceInputError(
      `canonical input is ${String(input.length)} bytes; maximum is ${String(maximumInputBytes)}`,
    );
  }
}

function normalizeOptions(
  options: InferenceRuntimeOptions,
): NormalizedInferenceRuntimeOptions {
  return {
    initializationTimeoutMilliseconds: positiveIntegerOption(
      options.initializationTimeoutMilliseconds,
      DEFAULT_INITIALIZATION_TIMEOUT_MILLISECONDS,
      "initializationTimeoutMilliseconds",
      600_000,
    ),
    inferenceTimeoutMilliseconds: positiveIntegerOption(
      options.inferenceTimeoutMilliseconds,
      DEFAULT_INFERENCE_TIMEOUT_MILLISECONDS,
      "inferenceTimeoutMilliseconds",
      600_000,
    ),
    maximumInputBytes: positiveIntegerOption(
      options.maximumInputBytes,
      DEFAULT_MAXIMUM_INPUT_BYTES,
      "maximumInputBytes",
      16_777_216,
    ),
    responseBufferBytes: positiveIntegerOption(
      options.responseBufferBytes,
      DEFAULT_RESPONSE_BUFFER_BYTES,
      "responseBufferBytes",
      1_048_576,
      64,
    ),
    transferModelBuffers: options.transferModelBuffers ?? true,
    workerMemoryLimitMegabytes: positiveIntegerOption(
      options.workerMemoryLimitMegabytes,
      DEFAULT_WORKER_MEMORY_LIMIT_MEGABYTES,
      "workerMemoryLimitMegabytes",
      32_768,
      64,
    ),
  };
}

function workerExecArgv(arguments_: readonly string[]): string[] {
  const valueFlags = [
    "--input-type",
    "--max-old-space-size",
    "--max_old_space_size",
    "--max-semi-space-size",
    "--max_semi_space_size",
    "--stack-size",
  ];
  const result: string[] = [];
  for (let index = 0; index < arguments_.length; index += 1) {
    const argument = arguments_[index] ?? "";
    const flag = valueFlags.find(
      (candidate) =>
        argument === candidate || argument.startsWith(`${candidate}=`),
    );
    if (flag === undefined) {
      result.push(argument);
    } else if (argument === flag && arguments_[index + 1] !== undefined) {
      index += 1;
    }
  }
  return result;
}

function positiveIntegerOption(
  value: number | undefined,
  defaultValue: number,
  name: string,
  maximum: number,
  minimum = 1,
): number {
  const selected = value ?? defaultValue;
  if (
    !Number.isSafeInteger(selected) ||
    selected < minimum ||
    selected > maximum
  ) {
    throw new RangeError(
      `${name} must be a safe integer between ${String(minimum)} and ${String(maximum)}`,
    );
  }
  return selected;
}

async function waitForWorkerExit(
  worker: Worker,
  timeoutMilliseconds: number,
): Promise<void> {
  await new Promise<void>((resolve) => {
    let settled = false;
    const finish = (): void => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeout);
      worker.off("exit", finish);
      resolve();
    };
    const timeout = setTimeout(finish, timeoutMilliseconds);
    worker.once("exit", finish);
  });
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
