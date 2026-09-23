import type {
  ModelProvenance,
  OllamaExecutionBackend,
  RefundInputs,
} from "../types.js";
import { REFUND_SYSTEM_PINS, REFUND_TASK_SPEC_SHA256 } from "../policy.js";
import {
  DEFAULT_BASELINE_TIMEOUT_MS,
  REFUND_OUTPUT_SCHEMA,
  REFUND_PROMPT_VERSION,
  BaselineAdapterError,
  adapterProvenance,
  assertResponseModel,
  buildRefundPrompt,
  invalid,
  invokeWithTimeout,
  parseStructuredPredictionJson,
  pinnedModelProvenance,
  refused,
  responseObject,
  validateTimeout,
  type BaselinePrediction,
  type RefundBaselineAdapter,
} from "./common.js";

export const OLLAMA_QWEN_MODELS = Object.freeze({
  "ollama-1b": REFUND_SYSTEM_PINS["ollama-1b"].model.name,
  "ollama-7b": REFUND_SYSTEM_PINS["ollama-7b"].model.name,
} as const);

export const OLLAMA_QWEN_MANIFEST_SHA256 = Object.freeze({
  "ollama-1b": REFUND_SYSTEM_PINS["ollama-1b"].model.revision,
  "ollama-7b": REFUND_SYSTEM_PINS["ollama-7b"].model.revision,
} as const);

export const OLLAMA_QWEN_WEIGHT_SHA256 = Object.freeze({
  "ollama-1b": REFUND_SYSTEM_PINS["ollama-1b"].model.artifactSha256,
  "ollama-7b": REFUND_SYSTEM_PINS["ollama-7b"].model.artifactSha256,
} as const);

export type OllamaBaselineRole = keyof typeof OLLAMA_QWEN_MODELS;
export type OllamaQwenModel = (typeof OLLAMA_QWEN_MODELS)[OllamaBaselineRole];

export interface OllamaGenerateRequest {
  readonly model: OllamaQwenModel;
  readonly system: string;
  readonly prompt: string;
  readonly stream: false;
  readonly think: false;
  readonly format: typeof REFUND_OUTPUT_SCHEMA;
  readonly options: {
    readonly temperature: 0;
    readonly seed: 0;
    readonly num_predict: 256;
  };
}

export interface OllamaGenerateResponse {
  readonly model: string;
  readonly response: string;
  readonly done: boolean;
  readonly done_reason: string;
}

export interface OllamaTransport {
  generate(
    request: OllamaGenerateRequest,
    signal: AbortSignal,
  ): Promise<OllamaGenerateResponse>;
  resolveExecutionBackend(
    model: OllamaQwenModel,
    signal: AbortSignal,
  ): Promise<OllamaExecutionBackend>;
}

export interface OllamaAdapterOptions<Role extends OllamaBaselineRole> {
  readonly role: Role;
  readonly model: ModelProvenance;
  readonly transport: OllamaTransport;
  readonly timeoutMs?: number;
}

export function createOllamaQwenAdapter<Role extends OllamaBaselineRole>(
  options: OllamaAdapterOptions<Role>,
): RefundBaselineAdapter<Role> {
  const requestedModel = OLLAMA_QWEN_MODELS[options.role];
  const timeoutMs = validateTimeout(options.timeoutMs ?? DEFAULT_BASELINE_TIMEOUT_MS);
  const model = pinnedModelProvenance(options.model, {
    provider: "ollama",
    name: requestedModel,
    version: "qwen2.5",
    revision: OLLAMA_QWEN_MANIFEST_SHA256[options.role],
  });
  if (model.artifactSha256 !== OLLAMA_QWEN_WEIGHT_SHA256[options.role]) {
    throw new BaselineAdapterError(
      "invalid-configuration",
      `model.artifactSha256 must pin the ${options.role} model-weight blob ${OLLAMA_QWEN_WEIGHT_SHA256[options.role]}`,
    );
  }
  const configuration = Object.freeze({
    provider: "ollama",
    role: options.role,
    model,
    promptVersion: REFUND_PROMPT_VERSION,
    taskSpecSha256: REFUND_TASK_SPEC_SHA256,
    timeoutMs,
    structuredOutputSchema: REFUND_OUTPUT_SCHEMA,
    decoding: Object.freeze({ temperature: 0, seed: 0, numPredict: 256 }),
  });
  const provenance = adapterProvenance("refund-ollama-json-schema", configuration);

  return Object.freeze({
    role: options.role,
    model,
    adapter: provenance,
    taskSpecSha256: REFUND_TASK_SPEC_SHA256,
    trainingEvidence: null,
    async resolveExecutionBackend(
      signal: AbortSignal,
    ): Promise<OllamaExecutionBackend> {
      return invokeWithTimeout(
        `Ollama ${requestedModel} execution-backend inspection`,
        timeoutMs,
        signal,
        async (transportSignal) =>
          options.transport.resolveExecutionBackend(requestedModel, transportSignal),
      );
    },
    async predict(inputs: RefundInputs, signal?: AbortSignal): Promise<BaselinePrediction> {
      const prompt = buildRefundPrompt(inputs);
      const request: OllamaGenerateRequest = Object.freeze({
        model: requestedModel,
        system: prompt.system,
        prompt: prompt.user,
        stream: false,
        think: false,
        format: REFUND_OUTPUT_SCHEMA,
        options: Object.freeze({ temperature: 0, seed: 0, num_predict: 256 }),
      });
      const response = await invokeWithTimeout(
        `Ollama ${requestedModel}`,
        timeoutMs,
        signal,
        async (transportSignal) => options.transport.generate(request, transportSignal),
      );
      const responseRecord = responseObject(response, [
        "model",
        "response",
        "done",
        "done_reason",
      ]);
      assertResponseModel(responseRecord["model"], requestedModel);
      if (responseRecord["done"] !== true) {
        invalid("response.done", "must be true");
      }
      if (
        responseRecord["done_reason"] === "refusal" ||
        responseRecord["done_reason"] === "refused"
      ) {
        refused("Ollama");
      }
      if (responseRecord["done_reason"] !== "stop") {
        invalid("response.done_reason", 'must be "stop"');
      }
      return parseStructuredPredictionJson(responseRecord["response"]);
    },
  });
}
