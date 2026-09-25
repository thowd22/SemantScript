import { parentPort } from "node:worker_threads";

import type { InferenceSession, Tensor as OrtTensor } from "onnxruntime-node";
import type { Tokenizer } from "tokenizers";

import {
  INFERENCE_CONTROL,
  INFERENCE_ERROR,
  INFERENCE_STATE,
  type InferenceDistributionEntry,
  type InferenceHeadPlan,
  type InferenceOnnxAbi,
  type InferenceScalarDiagnostic,
  type InferenceSupportValue,
  type InferenceWorkerResult,
  type InferenceWorkerPlan,
  type InferenceWorkerRequest,
  type InferenceWorkerResponse,
  type InferenceStagePasses,
  type InferenceStageRequest,
  type InvokeInferenceMessage,
  type InvokeStageInferenceMessage,
  MAXIMUM_STAGE_REQUESTS,
  type StagedFunctionPlan,
  type StagedInferencePlan,
  type TestFunctionPlan,
  type TestInferencePlan,
  stringifyInferenceResult,
} from "./inference-protocol.js";

const port = parentPort ?? missingParentPort();

const textDecoder = new TextDecoder("utf-8", { fatal: true });
const textEncoder = new TextEncoder();

interface StageOutcome {
  readonly results: readonly InferenceWorkerResult[];
  readonly passes: InferenceStagePasses;
}

interface InferenceBackend {
  readonly functionIds: readonly string[];
  invoke(
    functionId: string,
    canonicalInput: Uint8Array,
    scopeId?: number,
  ): Promise<InferenceWorkerResult>;
  /** Passes of the last invoke, after any in-scope sharing. */
  readonly lastPasses: InferenceStagePasses;
  /** Drops what a request scope kept. */
  endScope(scopeId: number): void;
  /** One stage: identical canonical inputs share one encoder pass and one adapter pass per adapter. */
  invokeStage(
    requests: readonly InferenceStageRequest[],
  ): Promise<StageOutcome>;
  close(): Promise<void>;
}

class WorkerInvocationError extends Error {
  readonly errorCode: number;

  constructor(errorCode: number, message: string) {
    super(message);
    this.name = "WorkerInvocationError";
    this.errorCode = errorCode;
  }
}

let backend: InferenceBackend | undefined;
let initializing = false;
let invocationInProgress = false;

port.on("message", (request: InferenceWorkerRequest) => {
  void handleRequest(request);
});

async function handleRequest(request: InferenceWorkerRequest): Promise<void> {
  switch (request.kind) {
    case "initialize":
      await initialize(request.plan);
      return;
    case "invoke":
      await invoke(request);
      return;
    case "invoke-stage":
      await invokeStage(request);
      return;
    case "end-scope":
      backend?.endScope(request.scopeId);
      return;
    case "shutdown":
      await shutdown();
      return;
  }
}

async function initialize(plan: InferenceWorkerPlan): Promise<void> {
  if (initializing || backend !== undefined) {
    postInitializationError(
      "the inference worker was initialized more than once",
    );
    return;
  }

  initializing = true;
  try {
    const initialized =
      plan.kind === "onnx"
        ? await OnnxBackend.create(plan)
        : await TestBackend.create(plan);
    backend = initialized;
    const response: InferenceWorkerResponse = {
      kind: "ready",
      functionIds: initialized.functionIds,
    };
    port.postMessage(response);
  } catch (error) {
    postInitializationError(errorMessage(error));
  } finally {
    initializing = false;
  }
}

function postInitializationError(message: string): void {
  const response: InferenceWorkerResponse = {
    kind: "initialization-error",
    message,
  };
  port.postMessage(response);
}

async function invoke(request: InvokeInferenceMessage): Promise<void> {
  let control: Int32Array;
  let response: Uint8Array;
  try {
    control = new Int32Array(request.controlBuffer);
    response = new Uint8Array(request.responseBuffer);
  } catch (error) {
    // There is no safe synchronization target if the shared buffers are invalid.
    throw new Error(
      `invalid inference synchronization buffers: ${errorMessage(error)}`,
      {
        cause: error,
      },
    );
  }

  if (
    control.length !== INFERENCE_CONTROL.length ||
    Atomics.load(control, INFERENCE_CONTROL.state) !==
      INFERENCE_STATE.pending ||
    Atomics.load(control, INFERENCE_CONTROL.sequence) !== request.sequence
  ) {
    complete(
      control,
      response,
      INFERENCE_STATE.error,
      INFERENCE_ERROR.protocol,
      "invalid invocation sequence",
    );
    return;
  }

  if (invocationInProgress) {
    complete(
      control,
      response,
      INFERENCE_STATE.error,
      INFERENCE_ERROR.busy,
      "the inference worker is busy",
    );
    return;
  }

  const activeBackend = backend;
  if (activeBackend === undefined) {
    complete(
      control,
      response,
      INFERENCE_STATE.error,
      INFERENCE_ERROR.protocol,
      "the inference worker is not initialized",
    );
    return;
  }

  invocationInProgress = true;
  try {
    const value = await activeBackend.invoke(
      request.functionId,
      request.canonicalInput,
      request.scopeId,
    );
    const passes = activeBackend.lastPasses;
    Atomics.store(control, INFERENCE_CONTROL.passesEncoder, passes.encoder);
    Atomics.store(control, INFERENCE_CONTROL.passesAdapter, passes.adapter);
    Atomics.store(control, INFERENCE_CONTROL.passesHead, passes.head);
    complete(
      control,
      response,
      INFERENCE_STATE.success,
      INFERENCE_ERROR.none,
      stringifyInferenceResult(value),
    );
  } catch (error) {
    const errorCode =
      error instanceof WorkerInvocationError
        ? error.errorCode
        : INFERENCE_ERROR.backend;
    complete(
      control,
      response,
      INFERENCE_STATE.error,
      errorCode,
      errorMessage(error),
    );
  } finally {
    invocationInProgress = false;
  }
}

async function invokeStage(
  request: InvokeStageInferenceMessage,
): Promise<void> {
  let control: Int32Array;
  let response: Uint8Array;
  try {
    control = new Int32Array(request.controlBuffer);
    response = new Uint8Array(request.responseBuffer);
  } catch (error) {
    throw new Error(
      `invalid inference synchronization buffers: ${errorMessage(error)}`,
      {
        cause: error,
      },
    );
  }
  if (
    control.length !== INFERENCE_CONTROL.length ||
    Atomics.load(control, INFERENCE_CONTROL.state) !==
      INFERENCE_STATE.pending ||
    Atomics.load(control, INFERENCE_CONTROL.sequence) !== request.sequence
  ) {
    complete(
      control,
      response,
      INFERENCE_STATE.error,
      INFERENCE_ERROR.protocol,
      "invalid invocation sequence",
    );
    return;
  }
  if (invocationInProgress) {
    complete(
      control,
      response,
      INFERENCE_STATE.error,
      INFERENCE_ERROR.busy,
      "the inference worker is busy",
    );
    return;
  }
  const activeBackend = backend;
  if (activeBackend === undefined) {
    complete(
      control,
      response,
      INFERENCE_STATE.error,
      INFERENCE_ERROR.protocol,
      "the inference worker is not initialized",
    );
    return;
  }
  if (
    !Array.isArray(request.requests) ||
    request.requests.length === 0 ||
    request.requests.length > MAXIMUM_STAGE_REQUESTS
  ) {
    complete(
      control,
      response,
      INFERENCE_STATE.error,
      INFERENCE_ERROR.protocol,
      "a stage carries between 1 and 64 requests",
    );
    return;
  }

  invocationInProgress = true;
  try {
    const outcome = await activeBackend.invokeStage(request.requests);
    completeBytes(
      control,
      response,
      INFERENCE_STATE.success,
      INFERENCE_ERROR.none,
      frameStageOutcome(outcome),
    );
  } catch (error) {
    const errorCode =
      error instanceof WorkerInvocationError
        ? error.errorCode
        : INFERENCE_ERROR.backend;
    complete(
      control,
      response,
      INFERENCE_STATE.error,
      errorCode,
      errorMessage(error),
    );
  } finally {
    invocationInProgress = false;
  }
}

/**
 * Stage responses are framed as one JSON header line (`passes` and the byte
 * length of every result) followed by the results' exact single-call wire
 * payloads, so each result is decoded by the same parser a single call uses.
 */
function frameStageOutcome(outcome: StageOutcome): Uint8Array {
  const payloads = outcome.results.map((value) =>
    textEncoder.encode(stringifyInferenceResult(value)),
  );
  const header = textEncoder.encode(
    `${JSON.stringify({ passes: outcome.passes, lengths: payloads.map((payload) => payload.length) })}\n`,
  );
  const framed = new Uint8Array(
    header.length +
      payloads.reduce((total, payload) => total + payload.length, 0),
  );
  framed.set(header, 0);
  let offset = header.length;
  for (const payload of payloads) {
    framed.set(payload, offset);
    offset += payload.length;
  }
  return framed;
}

function complete(
  control: Int32Array,
  response: Uint8Array,
  state: number,
  errorCode: number,
  payload: string,
): void {
  completeBytes(
    control,
    response,
    state,
    errorCode,
    textEncoder.encode(payload),
  );
}

function completeBytes(
  control: Int32Array,
  response: Uint8Array,
  state: number,
  errorCode: number,
  bytes: Uint8Array,
): void {
  let encoded = bytes;
  if (encoded.length > response.length) {
    state = INFERENCE_STATE.error;
    errorCode = INFERENCE_ERROR.responseTooLarge;
    encoded = textEncoder.encode(
      "the inference response exceeded the shared response buffer",
    );
  }

  const payloadLength = Math.min(encoded.length, response.length);
  response.fill(0);
  response.set(encoded.subarray(0, payloadLength));
  Atomics.store(control, INFERENCE_CONTROL.payloadLength, payloadLength);
  Atomics.store(control, INFERENCE_CONTROL.errorCode, errorCode);
  Atomics.store(control, INFERENCE_CONTROL.state, state);
  Atomics.notify(control, INFERENCE_CONTROL.state, 1);
}

async function shutdown(): Promise<void> {
  const activeBackend = backend;
  backend = undefined;
  if (activeBackend !== undefined) {
    await activeBackend.close();
  }
  port.close();
}

class TestBackend implements InferenceBackend {
  readonly functionIds: readonly string[];
  readonly #functions: ReadonlyMap<string, TestFunctionPlan>;
  readonly #scopes = new Map<number, Set<string>>();
  lastPasses: InferenceStagePasses = { encoder: 0, adapter: 0, head: 0 };

  endScope(scopeId: number): void {
    this.#scopes.delete(scopeId);
  }

  private constructor(functions: ReadonlyMap<string, TestFunctionPlan>) {
    this.#functions = functions;
    this.functionIds = [...functions.keys()];
  }

  static async create(plan: TestInferencePlan): Promise<TestBackend> {
    if (plan.initializationDelayMilliseconds !== undefined) {
      await delay(
        validateDelay(
          plan.initializationDelayMilliseconds,
          "initialization delay",
        ),
      );
    }
    if (plan.initializationError !== undefined) {
      throw new Error(plan.initializationError);
    }

    const functions = new Map<string, TestFunctionPlan>();
    for (const functionPlan of plan.functions) {
      validateFunctionId(functionPlan.id);
      validateDiagnosticsRequired(functionPlan.diagnosticsRequired);
      if (functions.has(functionPlan.id)) {
        throw new Error(
          `duplicate inference function id ${JSON.stringify(functionPlan.id)}`,
        );
      }
      validateHeads(functionPlan.heads, true);
      functions.set(functionPlan.id, functionPlan);
    }
    return new TestBackend(functions);
  }

  async invoke(
    functionId: string,
    canonicalInput: Uint8Array,
    scopeId?: number,
  ): Promise<InferenceWorkerResult> {
    const text = decodeCanonicalInput(canonicalInput);
    const functionPlan = this.#functions.get(functionId);
    if (functionPlan === undefined) {
      throw new WorkerInvocationError(
        INFERENCE_ERROR.unknownFunction,
        `unknown semantic function ${JSON.stringify(functionId)}`,
      );
    }
    // In a scope, a repeated input costs no encoder or adapter pass.
    let shared = false;
    if (scopeId !== undefined) {
      let seen = this.#scopes.get(scopeId);
      if (seen === undefined) {
        seen = new Set();
        this.#scopes.set(scopeId, seen);
      }
      shared = seen.has(text);
      seen.add(text);
    }
    this.lastPasses = {
      encoder: shared ? 0 : 1,
      adapter: shared ? 0 : 1,
      head: functionPlan.heads.length,
    };
    if (functionPlan.delayMilliseconds !== undefined) {
      await delay(
        validateDelay(functionPlan.delayMilliseconds, "inference delay"),
      );
    }
    if (functionPlan.error !== undefined) {
      throw new WorkerInvocationError(
        INFERENCE_ERROR.backend,
        functionPlan.error,
      );
    }

    return mapHeadOutputs(
      functionPlan.heads,
      functionPlan.heads.map((head) => head.logits),
      functionPlan.diagnosticsRequired,
    );
  }

  async invokeStage(
    requests: readonly InferenceStageRequest[],
  ): Promise<StageOutcome> {
    const results: InferenceWorkerResult[] = [];
    const distinctInputs = new Set<string>();
    let headPasses = 0;
    for (const request of requests) {
      distinctInputs.add(decodeCanonicalInput(request.canonicalInput));
      results.push(
        await this.invoke(request.functionId, request.canonicalInput),
      );
      headPasses += this.#functions.get(request.functionId)?.heads.length ?? 0;
    }
    return {
      results,
      passes: {
        encoder: distinctInputs.size,
        adapter: distinctInputs.size,
        head: headPasses,
      },
    };
  }

  async close(): Promise<void> {}
}

class OnnxBackend implements InferenceBackend {
  readonly functionIds: readonly string[];
  readonly #ort: typeof import("onnxruntime-node");
  readonly #tokenizer: Tokenizer;
  readonly #encoder: InferenceSession;
  /** Depth-routed encoder prefixes by ref, beside the application's encoder (TASK-6.7). */
  readonly #encoders: ReadonlyMap<string, InferenceSession>;
  readonly #adapters: ReadonlyMap<string, InferenceSession>;
  readonly #functions: ReadonlyMap<string, LoadedFunction>;
  readonly #maximumSequenceLength: number;
  /** Per request scope: the tensors kept so sibling calls share passes (TASK-8.1). */
  readonly #scopes = new Map<number, ScopeCache>();
  lastPasses: InferenceStagePasses = { encoder: 0, adapter: 0, head: 0 };

  endScope(scopeId: number): void {
    const cache = this.#scopes.get(scopeId);
    if (cache === undefined) return;
    this.#scopes.delete(scopeId);
    for (const tensor of cache.sentence.values()) tensor.dispose();
    for (const tensor of cache.function.values()) tensor.dispose();
  }

  private constructor(
    ort: typeof import("onnxruntime-node"),
    tokenizer: Tokenizer,
    encoder: InferenceSession,
    encoders: ReadonlyMap<string, InferenceSession>,
    adapters: ReadonlyMap<string, InferenceSession>,
    functions: ReadonlyMap<string, LoadedFunction>,
    maximumSequenceLength: number,
  ) {
    this.#ort = ort;
    this.#tokenizer = tokenizer;
    this.#encoder = encoder;
    this.#encoders = encoders;
    this.#adapters = adapters;
    this.#functions = functions;
    this.#maximumSequenceLength = maximumSequenceLength;
    this.functionIds = [...functions.keys()];
  }

  static async create(plan: StagedInferencePlan): Promise<OnnxBackend> {
    if (
      !Number.isSafeInteger(plan.maximumSequenceLength) ||
      plan.maximumSequenceLength < 1 ||
      plan.maximumSequenceLength > 8192
    ) {
      throw new Error(
        "maximumSequenceLength must be an integer from 1 through 8192",
      );
    }

    const ort = await import("onnxruntime-node");
    ort.env.logLevel = "error";
    const { Tokenizer: TokenizerConstructor } = await import("tokenizers");
    const tokenizerJson = textDecoder.decode(plan.tokenizerJson);
    const tokenizer = TokenizerConstructor.fromString(tokenizerJson);
    tokenizer.setTruncation(plan.maximumSequenceLength);
    tokenizer.disablePadding();
    const sessions: InferenceSession[] = [];

    try {
      const encoder = await createSession(ort, plan.encoderModel);
      sessions.push(encoder);
      assertSessionAbi(encoder, plan.encoderAbi, "encoder");
      const encoders = new Map<string, InferenceSession>();
      for (const encoderPlan of plan.encoders ?? []) {
        if (encoders.has(encoderPlan.ref)) {
          throw new Error(
            `duplicate encoder reference ${JSON.stringify(encoderPlan.ref)}`,
          );
        }
        const prefix = await createSession(ort, encoderPlan.model);
        sessions.push(prefix);
        assertSessionAbi(
          prefix,
          encoderPlan.abi,
          `encoder ${JSON.stringify(encoderPlan.ref)}`,
        );
        encoders.set(encoderPlan.ref, prefix);
      }

      const adapters = new Map<string, InferenceSession>();
      for (const adapterPlan of plan.adapters) {
        if (adapters.has(adapterPlan.ref)) {
          throw new Error(
            `duplicate adapter reference ${JSON.stringify(adapterPlan.ref)}`,
          );
        }
        const adapter = await createSession(ort, adapterPlan.model);
        sessions.push(adapter);
        assertSessionAbi(
          adapter,
          adapterPlan.abi,
          `adapter ${JSON.stringify(adapterPlan.ref)}`,
        );
        adapters.set(adapterPlan.ref, adapter);
      }

      const functions = new Map<string, LoadedFunction>();
      for (const functionPlan of plan.functions) {
        // Each function's encoder (the application's or its depth prefix) must feed its adapter.
        const functionEncoder =
          functionPlan.encoderRef === undefined
            ? encoder
            : encoders.get(functionPlan.encoderRef);
        if (functionEncoder === undefined) {
          throw new Error(
            `function ${JSON.stringify(functionPlan.id)} references unknown encoder ${JSON.stringify(functionPlan.encoderRef)}`,
          );
        }
        const functionAdapter = adapters.get(functionPlan.adapterRef);
        if (functionAdapter !== undefined) {
          assertSessionEdge(
            functionEncoder,
            "sentence_embedding",
            functionAdapter,
            "sentence_embedding",
            `encoder to adapter ${JSON.stringify(functionPlan.adapterRef)} for ${JSON.stringify(functionPlan.id)}`,
          );
        }
        const loaded = await loadFunction(
          ort,
          functionPlan,
          adapters,
          sessions,
        );
        if (functions.has(functionPlan.id)) {
          throw new Error(
            `duplicate inference function id ${JSON.stringify(functionPlan.id)}`,
          );
        }
        functions.set(functionPlan.id, loaded);
      }

      return new OnnxBackend(
        ort,
        tokenizer,
        encoder,
        encoders,
        adapters,
        functions,
        plan.maximumSequenceLength,
      );
    } catch (error) {
      await Promise.allSettled(
        sessions.map(async (session) => session.release()),
      );
      throw error;
    }
  }

  async invoke(
    functionId: string,
    canonicalInput: Uint8Array,
    scopeId?: number,
  ): Promise<InferenceWorkerResult> {
    const outcome = await this.invokeStage(
      [{ functionId, canonicalInput }],
      scopeId === undefined ? undefined : this.#scope(scopeId),
    );
    const [result] = outcome.results;
    if (result === undefined) {
      throw new WorkerInvocationError(
        INFERENCE_ERROR.backend,
        "stage of one produced no result",
      );
    }
    this.lastPasses = outcome.passes;
    return result;
  }

  #scope(scopeId: number): ScopeCache {
    let cache = this.#scopes.get(scopeId);
    if (cache === undefined) {
      cache = { sentence: new Map(), function: new Map() };
      this.#scopes.set(scopeId, cache);
    }
    return cache;
  }

  async invokeStage(
    requests: readonly InferenceStageRequest[],
    scope?: ScopeCache,
  ): Promise<StageOutcome> {
    const plans = requests.map((request) => {
      const functionPlan = this.#functions.get(request.functionId);
      if (functionPlan === undefined) {
        throw new WorkerInvocationError(
          INFERENCE_ERROR.unknownFunction,
          `unknown semantic function ${JSON.stringify(request.functionId)}`,
        );
      }
      return functionPlan;
    });
    const texts = requests.map((request) =>
      decodeCanonicalInput(request.canonicalInput),
    );

    const sentenceEmbeddings = new Map<string, OrtTensor>();
    const functionEmbeddings = new Map<string, OrtTensor>();
    const passes = { encoder: 0, adapter: 0, head: 0 };
    const results: InferenceWorkerResult[] = [];
    try {
      // One encoder pass per distinct (encoder, canonical input): functions of
      // one depth share their prefix pass.
      for (const [index, functionPlan] of plans.entries()) {
        const text = texts[index] ?? "";
        const encoderKey = `${functionPlan.encoderRef ?? ""}\u0000${text}`;
        if (sentenceEmbeddings.has(encoderKey)) continue;
        // A request scope keeps embeddings so later calls of the request share them.
        const kept = scope?.sentence.get(encoderKey);
        if (kept !== undefined) {
          sentenceEmbeddings.set(encoderKey, kept);
          continue;
        }
        const embedding = await this.#encode(text, functionPlan.encoderRef);
        sentenceEmbeddings.set(encoderKey, embedding);
        if (scope !== undefined)
          keepInScope(scope.sentence, encoderKey, embedding);
        passes.encoder += 1;
      }
      // One adapter pass per distinct (input, adapter).
      for (const [index, functionPlan] of plans.entries()) {
        const text = texts[index] ?? "";
        const key = `${functionPlan.adapterRef}\u0000${text}`;
        if (functionEmbeddings.has(key)) continue;
        const keptFunction = scope?.function.get(key);
        if (keptFunction !== undefined) {
          functionEmbeddings.set(key, keptFunction);
          continue;
        }
        const adapter = this.#adapters.get(functionPlan.adapterRef);
        if (adapter === undefined) {
          throw new WorkerInvocationError(
            INFERENCE_ERROR.backend,
            `adapter ${JSON.stringify(functionPlan.adapterRef)} is not loaded`,
          );
        }
        const sentenceEmbedding = sentenceEmbeddings.get(
          `${functionPlan.encoderRef ?? ""}\u0000${text}`,
        );
        if (sentenceEmbedding === undefined) {
          throw new WorkerInvocationError(
            INFERENCE_ERROR.backend,
            "stage lost a sentence embedding",
          );
        }
        let adapterOutput;
        try {
          adapterOutput = await adapter.run({
            sentence_embedding: sentenceEmbedding,
          });
        } catch (error) {
          throw new WorkerInvocationError(
            INFERENCE_ERROR.backend,
            `ONNX inference failed: ${errorMessage(error)}`,
          );
        }
        const functionEmbedding = requireFloatTensor(
          adapterOutput["function_embedding"],
          "adapter output function_embedding",
        );
        functionEmbeddings.set(key, functionEmbedding);
        if (scope !== undefined)
          keepInScope(scope.function, key, functionEmbedding);
        passes.adapter += 1;
      }
      // Every head of every function.
      for (const [index, functionPlan] of plans.entries()) {
        const text = texts[index] ?? "";
        const functionEmbedding = functionEmbeddings.get(
          `${functionPlan.adapterRef}\u0000${text}`,
        );
        if (functionEmbedding === undefined) {
          throw new WorkerInvocationError(
            INFERENCE_ERROR.backend,
            "stage lost a function embedding",
          );
        }
        const evaluations: HeadEvaluation[] = [];
        for (const head of functionPlan.heads) {
          let headOutput;
          try {
            headOutput = await head.session.run({
              function_embedding: functionEmbedding,
            });
          } catch (error) {
            throw new WorkerInvocationError(
              INFERENCE_ERROR.backend,
              `ONNX inference failed: ${errorMessage(error)}`,
            );
          }
          const logitsTensor = requireFloatTensor(
            headOutput["logits"],
            "head output logits",
          );
          try {
            evaluations.push(
              evaluateHead(
                head.plan,
                Array.from(logitsTensor.data, Number),
                functionPlan.diagnosticsRequired,
              ),
            );
          } finally {
            logitsTensor.dispose();
          }
          passes.head += 1;
        }
        results.push(
          mapHeadEvaluations(
            functionPlan.heads.map((head) => head.plan),
            evaluations,
            functionPlan.diagnosticsRequired,
          ),
        );
      }
      return { results, passes };
    } finally {
      // Tensors a scope keeps live until the scope ends; the rest go now.
      for (const tensor of functionEmbeddings.values()) {
        if (!ownedByScope(scope?.function, tensor)) tensor.dispose();
      }
      for (const tensor of sentenceEmbeddings.values()) {
        if (!ownedByScope(scope?.sentence, tensor)) tensor.dispose();
      }
    }
  }

  async #encode(
    canonicalText: string,
    encoderRef?: string,
  ): Promise<OrtTensor> {
    const session =
      encoderRef === undefined ? this.#encoder : this.#encoders.get(encoderRef);
    if (session === undefined) {
      throw new WorkerInvocationError(
        INFERENCE_ERROR.backend,
        `encoder ${JSON.stringify(encoderRef)} is not loaded`,
      );
    }
    let encoding;
    try {
      encoding = await this.#tokenizer.encode(canonicalText, null, {
        addSpecialTokens: true,
      });
    } catch (error) {
      throw new WorkerInvocationError(
        INFERENCE_ERROR.invalidInput,
        `tokenization failed: ${errorMessage(error)}`,
      );
    }
    const ids = encoding.getIds();
    const attentionMask = encoding.getAttentionMask();
    validateTokens(ids, attentionMask, this.#maximumSequenceLength);
    const inputIds = new BigInt64Array(ids.length);
    const inputMask = new BigInt64Array(attentionMask.length);
    for (let index = 0; index < ids.length; index += 1) {
      inputIds[index] = BigInt(ids[index] ?? 0);
      inputMask[index] = BigInt(attentionMask[index] ?? 0);
    }
    const idsTensor = new this.#ort.Tensor("int64", inputIds, [1, ids.length]);
    const maskTensor = new this.#ort.Tensor("int64", inputMask, [
      1,
      attentionMask.length,
    ]);
    try {
      const encoderOutput = await session.run({
        input_ids: idsTensor,
        attention_mask: maskTensor,
      });
      return requireFloatTensor(
        encoderOutput["sentence_embedding"],
        "encoder output sentence_embedding",
      );
    } catch (error) {
      if (error instanceof WorkerInvocationError) throw error;
      throw new WorkerInvocationError(
        INFERENCE_ERROR.backend,
        `ONNX inference failed: ${errorMessage(error)}`,
      );
    } finally {
      idsTensor.dispose();
      maskTensor.dispose();
    }
  }

  async close(): Promise<void> {
    for (const scopeId of [...this.#scopes.keys()]) this.endScope(scopeId);
    const sessions: InferenceSession[] = [
      this.#encoder,
      ...this.#encoders.values(),
      ...this.#adapters.values(),
    ];
    for (const functionPlan of this.#functions.values()) {
      sessions.push(...functionPlan.heads.map((head) => head.session));
    }
    await Promise.allSettled(
      sessions.map(async (session) => session.release()),
    );
  }
}

interface ScopeCache {
  readonly sentence: Map<string, OrtTensor>;
  readonly function: Map<string, OrtTensor>;
}

/** A scope keeps at most this many embeddings of each kind; a request has few inputs. */
const SCOPE_CACHE_ENTRIES = 64;

function keepInScope(
  cache: Map<string, OrtTensor>,
  key: string,
  tensor: OrtTensor,
): void {
  if (cache.size >= SCOPE_CACHE_ENTRIES) {
    const oldest = cache.keys().next();
    if (!oldest.done) {
      cache.get(oldest.value)?.dispose();
      cache.delete(oldest.value);
    }
  }
  cache.set(key, tensor);
}

function ownedByScope(
  cache: Map<string, OrtTensor> | undefined,
  tensor: OrtTensor,
): boolean {
  if (cache === undefined) return false;
  for (const kept of cache.values()) if (kept === tensor) return true;
  return false;
}

interface LoadedHead {
  readonly plan: InferenceHeadPlan;
  readonly session: InferenceSession;
}

interface LoadedFunction {
  readonly adapterRef: string;
  readonly encoderRef?: string;
  readonly diagnosticsRequired: boolean;
  readonly heads: readonly LoadedHead[];
}

async function loadFunction(
  ort: typeof import("onnxruntime-node"),
  functionPlan: StagedFunctionPlan,
  adapters: ReadonlyMap<string, InferenceSession>,
  sessions: InferenceSession[],
): Promise<LoadedFunction> {
  validateFunctionId(functionPlan.id);
  validateDiagnosticsRequired(functionPlan.diagnosticsRequired);
  if (!adapters.has(functionPlan.adapterRef)) {
    throw new Error(
      `function ${JSON.stringify(functionPlan.id)} references unknown adapter ${JSON.stringify(functionPlan.adapterRef)}`,
    );
  }
  validateHeads(functionPlan.heads, false);

  const heads: LoadedHead[] = [];
  const adapter = adapters.get(functionPlan.adapterRef);
  if (adapter === undefined) {
    throw new Error(
      `function ${JSON.stringify(functionPlan.id)} references an unloaded adapter`,
    );
  }
  for (const headPlan of functionPlan.heads) {
    const session = await createSession(ort, headPlan.model);
    sessions.push(session);
    assertSessionAbi(
      session,
      headPlan.abi,
      `head for ${JSON.stringify(functionPlan.id)}`,
    );
    assertSessionEdge(
      adapter,
      "function_embedding",
      session,
      "function_embedding",
      `adapter to head for ${JSON.stringify(functionPlan.id)}`,
    );
    heads.push({ plan: headPlan, session });
  }
  return {
    adapterRef: functionPlan.adapterRef,
    ...(functionPlan.encoderRef === undefined
      ? {}
      : { encoderRef: functionPlan.encoderRef }),
    diagnosticsRequired: functionPlan.diagnosticsRequired,
    heads,
  };
}

async function createSession(
  ort: typeof import("onnxruntime-node"),
  model: Uint8Array,
): Promise<InferenceSession> {
  if (!(model instanceof Uint8Array) || model.byteLength === 0) {
    throw new Error("ONNX model bytes must be a non-empty Uint8Array");
  }
  return ort.InferenceSession.create(model, { executionProviders: ["cpu"] });
}

function assertSessionAbi(
  session: InferenceSession,
  expected: InferenceOnnxAbi,
  description: string,
): void {
  if (
    !sameMetadata(session.inputMetadata, expected.inputs) ||
    !sameMetadata(session.outputMetadata, expected.outputs)
  ) {
    throw new Error(
      `${description} tensor metadata does not match its manifest ABI: expected inputs ` +
        `${JSON.stringify(expected.inputs)} and outputs ${JSON.stringify(expected.outputs)}, received inputs ` +
        `${JSON.stringify(session.inputMetadata)} and outputs ${JSON.stringify(session.outputMetadata)}`,
    );
  }
}

function sameMetadata(
  actual: readonly InferenceSession.ValueMetadata[],
  expected: InferenceOnnxAbi["inputs"],
): boolean {
  return (
    actual.length === expected.length &&
    actual.every((metadata, index) => {
      const descriptor = expected[index];
      return (
        descriptor !== undefined &&
        metadata.isTensor &&
        metadata.name === descriptor.name &&
        metadata.type === descriptor.dtype &&
        metadata.shape.length === descriptor.shape.length &&
        metadata.shape.every((dimension, dimensionIndex) =>
          dimensionMatches(dimension, descriptor.shape[dimensionIndex]),
        )
      );
    })
  );
}

function dimensionMatches(
  actual: number | string,
  expected: number | string | undefined,
): boolean {
  if (typeof expected === "number") {
    return actual === expected;
  }
  if (expected === "BATCH") {
    return actual === 1 || (typeof actual === "string" && actual.length > 0);
  }
  if (expected === "SEQUENCE") {
    return typeof actual === "string" && actual.length > 0;
  }
  if (expected === "HIDDEN") {
    return (
      (typeof actual === "number" &&
        Number.isSafeInteger(actual) &&
        actual > 0) ||
      (typeof actual === "string" && actual.length > 0)
    );
  }
  return actual === expected;
}

function assertSessionEdge(
  producer: InferenceSession,
  outputName: string,
  consumer: InferenceSession,
  inputName: string,
  description: string,
): void {
  const output = producer.outputMetadata.find(
    (metadata) => metadata.name === outputName,
  );
  const input = consumer.inputMetadata.find(
    (metadata) => metadata.name === inputName,
  );
  if (
    output === undefined ||
    input === undefined ||
    !output.isTensor ||
    !input.isTensor ||
    output.type !== input.type ||
    !actualShapesCompatible(output.shape, input.shape)
  ) {
    throw new Error(`${description} tensor metadata is incompatible`);
  }
}

function actualShapesCompatible(
  left: readonly (number | string)[],
  right: readonly (number | string)[],
): boolean {
  if (left.length !== 2 || right.length !== 2) {
    return false;
  }

  const leftBatch = left[0];
  const rightBatch = right[0];
  const leftHidden = left[1];
  const rightHidden = right[1];
  if (
    leftBatch === undefined ||
    rightBatch === undefined ||
    leftHidden === undefined ||
    rightHidden === undefined
  ) {
    return false;
  }

  return (
    validBatchDimension(leftBatch) &&
    validBatchDimension(rightBatch) &&
    validHiddenDimension(leftHidden) &&
    validHiddenDimension(rightHidden) &&
    (typeof leftHidden === "string" ||
      typeof rightHidden === "string" ||
      leftHidden === rightHidden)
  );
}

function validBatchDimension(dimension: number | string): boolean {
  return (
    dimension === 1 || (typeof dimension === "string" && dimension.length > 0)
  );
}

function validHiddenDimension(dimension: number | string): boolean {
  return (
    (typeof dimension === "number" &&
      Number.isSafeInteger(dimension) &&
      dimension > 0) ||
    (typeof dimension === "string" && dimension.length > 0)
  );
}

function requireFloatTensor(
  value: OrtTensor | undefined,
  description: string,
): OrtTensor {
  if (value === undefined || value.type !== "float32") {
    value?.dispose();
    throw new WorkerInvocationError(
      INFERENCE_ERROR.invalidResult,
      `${description} is not a float32 tensor`,
    );
  }
  if (
    value.dims.length !== 2 ||
    value.dims[0] !== 1 ||
    (value.dims[1] ?? 0) < 1
  ) {
    value.dispose();
    throw new WorkerInvocationError(
      INFERENCE_ERROR.invalidResult,
      `${description} does not have shape [1, WIDTH]`,
    );
  }
  return value;
}

function validateTokens(
  ids: readonly number[],
  mask: readonly number[],
  maximumLength: number,
): void {
  if (
    ids.length === 0 ||
    ids.length !== mask.length ||
    ids.length > maximumLength
  ) {
    throw new WorkerInvocationError(
      INFERENCE_ERROR.invalidInput,
      `tokenizer produced an invalid sequence length ${String(ids.length)}; maximum is ${String(maximumLength)}`,
    );
  }
  for (let index = 0; index < ids.length; index += 1) {
    const id = ids[index];
    const maskValue = mask[index];
    if (!Number.isSafeInteger(id) || id === undefined || id < 0) {
      throw new WorkerInvocationError(
        INFERENCE_ERROR.invalidInput,
        `tokenizer produced invalid token id at ${String(index)}`,
      );
    }
    if (
      (maskValue !== 0 && maskValue !== 1) ||
      !Number.isSafeInteger(maskValue)
    ) {
      throw new WorkerInvocationError(
        INFERENCE_ERROR.invalidInput,
        `tokenizer produced invalid attention mask at ${String(index)}`,
      );
    }
  }
}

function decodeCanonicalInput(input: Uint8Array): string {
  if (!(input instanceof Uint8Array) || input.length === 0) {
    throw new WorkerInvocationError(
      INFERENCE_ERROR.invalidInput,
      "canonical input must be non-empty UTF-8 bytes",
    );
  }
  try {
    return textDecoder.decode(input);
  } catch (error) {
    throw new WorkerInvocationError(
      INFERENCE_ERROR.invalidInput,
      `canonical input is not valid UTF-8: ${errorMessage(error)}`,
    );
  }
}

function validateHeads(
  heads: readonly (InferenceHeadPlan & {
    readonly logits?: readonly number[];
  })[],
  requireLogits: boolean,
): void {
  if (heads.length === 0) {
    throw new Error("an inference function must have at least one head");
  }

  const scalar = heads.length === 1 && heads[0]?.outputPath.length === 0;
  const objectFields = new Set<string>();
  for (const head of heads) {
    if (!scalar) {
      const field =
        head.outputPath.length === 1 ? head.outputPath[0] : undefined;
      if (field === undefined || objectFields.has(field)) {
        throw new Error(
          "object head output paths must be unique single segments",
        );
      }
      objectFields.add(field);
    }

    if (!Number.isFinite(head.temperature) || head.temperature <= 0) {
      throw new Error("head temperature must be a positive finite number");
    }
    validateSupport(head);
    validateExpectedValueMode(head);
    if (requireLogits) {
      if (head.logits === undefined || !head.logits.every(Number.isFinite)) {
        throw new Error("test head logits must be finite numbers");
      }
      const expected =
        head.parameterization === "binary-sigmoid" ? 1 : head.support.length;
      if (head.logits.length !== expected) {
        throw new Error(
          `test head requires ${String(expected)} logits but received ${String(head.logits.length)}`,
        );
      }
    }
  }
}

function validateExpectedValueMode(head: InferenceHeadPlan): void {
  const mode: unknown = head.expectedValueMode;
  if (mode !== "none" && mode !== "zero-based-rank" && mode !== "numeric") {
    throw new Error("head expected-value mode is invalid");
  }
  if (
    head.parameterization === "binary-sigmoid" &&
    head.expectedValueMode !== "none"
  ) {
    throw new Error(
      "binary-sigmoid heads cannot produce an ordinal expected value",
    );
  }
  if (
    head.expectedValueMode === "zero-based-rank" &&
    !head.support.every((value) => typeof value === "string")
  ) {
    throw new Error("zero-based-rank expected values require string support");
  }
  if (
    head.expectedValueMode === "numeric" &&
    !head.support.every((value) => typeof value === "number")
  ) {
    throw new Error("numeric expected values require numeric support");
  }
}

function validateSupport(head: InferenceHeadPlan): void {
  if (head.parameterization === "binary-sigmoid") {
    if (
      head.support.length !== 2 ||
      head.support[0] !== false ||
      head.support[1] !== true
    ) {
      throw new Error("binary-sigmoid support must be [false, true]");
    }
    return;
  }
  if (head.support.length < 2) {
    throw new Error(
      "categorical-softmax support must contain at least two values",
    );
  }
  for (const value of head.support) {
    if (
      (typeof value !== "string" &&
        typeof value !== "number" &&
        typeof value !== "boolean") ||
      (typeof value === "number" && !Number.isFinite(value))
    ) {
      throw new Error("head support contains an invalid runtime value");
    }
  }
}

interface HeadEvaluation {
  readonly value: InferenceSupportValue;
  readonly diagnostic?: InferenceScalarDiagnostic;
}

function mapHeadOutputs(
  heads: readonly InferenceHeadPlan[],
  logitsByHead: readonly (readonly number[])[],
  diagnosticsRequired: boolean,
): InferenceWorkerResult {
  if (heads.length !== logitsByHead.length) {
    throw new WorkerInvocationError(
      INFERENCE_ERROR.invalidResult,
      "the head result count does not match the plan",
    );
  }
  const evaluations = heads.map((head, index) =>
    evaluateHead(head, logitsByHead[index] ?? [], diagnosticsRequired),
  );
  return mapHeadEvaluations(heads, evaluations, diagnosticsRequired);
}

function mapHeadEvaluations(
  heads: readonly InferenceHeadPlan[],
  evaluations: readonly HeadEvaluation[],
  diagnosticsRequired: boolean,
): InferenceWorkerResult {
  if (heads.length !== evaluations.length) {
    throw new WorkerInvocationError(
      INFERENCE_ERROR.invalidResult,
      "the head result count does not match the plan",
    );
  }
  if (heads.length === 1 && heads[0]?.outputPath.length === 0) {
    const evaluation = evaluations[0];
    if (evaluation === undefined) {
      throw new WorkerInvocationError(
        INFERENCE_ERROR.invalidResult,
        "scalar head produced no value",
      );
    }
    if (!diagnosticsRequired) {
      return { kind: "value", result: evaluation.value };
    }
    if (evaluation.diagnostic === undefined) {
      throw new WorkerInvocationError(
        INFERENCE_ERROR.invalidResult,
        "scalar head omitted diagnostics",
      );
    }
    return { kind: "scalar", result: evaluation.diagnostic };
  }

  const valueResult: Record<string, InferenceSupportValue> = {};
  const fieldResults: Record<string, InferenceScalarDiagnostic> = {};
  let minimumFieldConfidence = 1;
  let maximumFieldUncertainty = 0;
  for (let index = 0; index < heads.length; index += 1) {
    const field = heads[index]?.outputPath[0];
    const evaluation = evaluations[index];
    if (field === undefined || evaluation === undefined) {
      throw new WorkerInvocationError(
        INFERENCE_ERROR.invalidResult,
        "invalid flat-object head mapping",
      );
    }
    defineResultField(valueResult, field, evaluation.value);
    if (diagnosticsRequired) {
      const diagnostic = evaluation.diagnostic;
      if (diagnostic === undefined) {
        throw new WorkerInvocationError(
          INFERENCE_ERROR.invalidResult,
          "object head omitted diagnostics",
        );
      }
      defineResultField(fieldResults, field, diagnostic);
      minimumFieldConfidence = Math.min(
        minimumFieldConfidence,
        diagnostic.confidence,
      );
      maximumFieldUncertainty = Math.max(
        maximumFieldUncertainty,
        diagnostic.uncertainty,
      );
    }
  }

  if (!diagnosticsRequired) {
    return { kind: "value", result: valueResult };
  }
  return {
    kind: "object",
    result: {
      value: valueResult,
      minimumFieldConfidence,
      maximumFieldUncertainty,
      fields: fieldResults,
    },
  };
}

function defineResultField<T>(
  target: Record<string, T>,
  field: string,
  value: T,
): void {
  Object.defineProperty(target, field, {
    configurable: true,
    enumerable: true,
    value,
    writable: true,
  });
}

function evaluateHead(
  head: InferenceHeadPlan,
  logits: readonly number[],
  diagnosticsRequired: boolean,
): HeadEvaluation {
  if (!logits.every(Number.isFinite)) {
    throw new WorkerInvocationError(
      INFERENCE_ERROR.invalidResult,
      "head logits contain a non-finite value",
    );
  }

  let selectedIndex: number;
  let probabilities: number[] | undefined;
  if (head.parameterization === "binary-sigmoid") {
    if (logits.length !== 1) {
      throw new WorkerInvocationError(
        INFERENCE_ERROR.invalidResult,
        "binary-sigmoid head must emit one logit",
      );
    }
    const scaled = (logits[0] ?? 0) / head.temperature;
    const probabilityTrue = stableSigmoid(scaled);
    selectedIndex = probabilityTrue > 0.5 ? 1 : 0;
    if (diagnosticsRequired) {
      probabilities = [1 - probabilityTrue, probabilityTrue];
    }
  } else {
    if (logits.length !== head.support.length) {
      throw new WorkerInvocationError(
        INFERENCE_ERROR.invalidResult,
        `categorical-softmax head emitted ${String(logits.length)} logits for ` +
          `${String(head.support.length)} support values`,
      );
    }
    const rawMaximumIndex = argmax(logits);
    const calibratedProbabilities = stableSoftmax(
      logits,
      rawMaximumIndex,
      head.temperature,
    );
    selectedIndex = argmax(calibratedProbabilities);
    if (diagnosticsRequired) {
      probabilities = calibratedProbabilities;
    }
  }

  const selected = head.support[selectedIndex];
  if (selected === undefined) {
    throw new WorkerInvocationError(
      INFERENCE_ERROR.invalidResult,
      "head selected an absent support value",
    );
  }
  if (probabilities === undefined) {
    return { value: selected };
  }

  const confidence = probabilities[selectedIndex];
  if (confidence === undefined || !Number.isFinite(confidence)) {
    throw new WorkerInvocationError(
      INFERENCE_ERROR.invalidResult,
      "head confidence is invalid",
    );
  }
  const distribution: InferenceDistributionEntry[] = head.support.map(
    (value, index) => ({
      value,
      probability: probabilities[index] ?? 0,
    }),
  );
  return {
    value: selected,
    diagnostic: {
      value: selected,
      confidence,
      uncertainty: normalizedEntropy(probabilities),
      distribution,
      expectedValue: expectedValue(head, probabilities),
    },
  };
}

function stableSigmoid(value: number): number {
  return value >= 0
    ? 1 / (1 + Math.exp(-value))
    : Math.exp(value) / (1 + Math.exp(value));
}

function stableSoftmax(
  logits: readonly number[],
  selectedIndex: number,
  temperature: number,
): number[] {
  const maximum = logits[selectedIndex];
  if (maximum === undefined) {
    throw new WorkerInvocationError(
      INFERENCE_ERROR.invalidResult,
      "categorical head has no maximum logit",
    );
  }

  const probabilities = new Array<number>(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    probabilities[index] = Math.exp(
      ((logits[index] ?? maximum) - maximum) / temperature,
    );
  }
  const total = compensatedSum(
    probabilities.length,
    (index) => probabilities[index] ?? 0,
  );
  if (!Number.isFinite(total) || total <= 0) {
    throw new WorkerInvocationError(
      INFERENCE_ERROR.invalidResult,
      "categorical probabilities cannot be normalized",
    );
  }
  for (let index = 0; index < probabilities.length; index += 1) {
    probabilities[index] = (probabilities[index] ?? 0) / total;
  }
  return probabilities;
}

function normalizedEntropy(probabilities: readonly number[]): number {
  if (probabilities.length <= 1) {
    return 0;
  }
  const entropy = compensatedSum(probabilities.length, (index) => {
    const probability = probabilities[index] ?? 0;
    return probability === 0 ? 0 : -probability * Math.log(probability);
  });
  if (!Number.isFinite(entropy)) {
    throw new WorkerInvocationError(
      INFERENCE_ERROR.invalidResult,
      "head entropy is invalid",
    );
  }
  if (entropy === 0) {
    return 0;
  }
  return clamp(entropy / Math.log(probabilities.length), 0, 1);
}

function expectedValue(
  head: InferenceHeadPlan,
  probabilities: readonly number[],
): number | null {
  if (head.expectedValueMode === "none") {
    return null;
  }
  if (head.expectedValueMode === "zero-based-rank") {
    return compensatedSum(
      probabilities.length,
      (index) => (probabilities[index] ?? 0) * index,
    );
  }

  let scale = 1;
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (const supportValue of head.support) {
    if (typeof supportValue !== "number") {
      throw new WorkerInvocationError(
        INFERENCE_ERROR.invalidResult,
        "numeric ordinal support is invalid",
      );
    }
    scale = Math.max(scale, Math.abs(supportValue));
    minimum = Math.min(minimum, supportValue);
    maximum = Math.max(maximum, supportValue);
  }
  const normalized = compensatedSum(probabilities.length, (index) => {
    const supportValue = head.support[index];
    if (typeof supportValue !== "number") {
      throw new WorkerInvocationError(
        INFERENCE_ERROR.invalidResult,
        "numeric ordinal support is invalid",
      );
    }
    return (probabilities[index] ?? 0) * (supportValue / scale);
  });
  const result = clamp(normalized, minimum / scale, maximum / scale) * scale;
  if (!Number.isFinite(result)) {
    throw new WorkerInvocationError(
      INFERENCE_ERROR.invalidResult,
      "ordinal expected value is invalid",
    );
  }
  return result;
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

function argmax(values: readonly number[]): number {
  let selected = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (
      (values[index] ?? Number.NEGATIVE_INFINITY) >
      (values[selected] ?? Number.NEGATIVE_INFINITY)
    ) {
      selected = index;
    }
  }
  return selected;
}

function validateDiagnosticsRequired(value: boolean): void {
  if (typeof value !== "boolean") {
    throw new Error("inference function diagnosticsRequired must be boolean");
  }
}

function validateFunctionId(functionId: string): void {
  if (typeof functionId !== "string" || functionId.length === 0) {
    throw new Error("inference function id must be a non-empty string");
  }
}

function validateDelay(value: number, description: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > 60_000) {
    throw new Error(
      `${description} must be an integer between 0 and 60000 milliseconds`,
    );
  }
  return value;
}

async function delay(milliseconds: number): Promise<void> {
  await new Promise<void>((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function missingParentPort(): never {
  throw new Error("the SemantScript inference worker requires a parent port");
}
