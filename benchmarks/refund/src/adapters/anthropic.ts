import type {
  AnthropicExecutionBackend,
  ModelProvenance,
  RefundInputs,
} from "../types.js";
import { REFUND_SYSTEM_PINS, REFUND_TASK_SPEC_SHA256 } from "../policy.js";
import {
  DEFAULT_BASELINE_TIMEOUT_MS,
  REFUND_OUTPUT_SCHEMA,
  REFUND_PROMPT_VERSION,
  adapterProvenance,
  assertResponseModel,
  buildRefundPrompt,
  invalid,
  invokeWithTimeout,
  parseStructuredPrediction,
  pinnedModelProvenance,
  refused,
  responseObject,
  validateTimeout,
  type BaselinePrediction,
  type RefundBaselineAdapter,
} from "./common.js";

export const ANTHROPIC_SONNET_MODEL =
  REFUND_SYSTEM_PINS["structured-api"].model.name;

export interface AnthropicStructuredRequest {
  readonly model: typeof ANTHROPIC_SONNET_MODEL;
  readonly maxTokens: 256;
  readonly system: string;
  readonly messages: readonly [
    {
      readonly role: "user";
      readonly content: string;
    },
  ];
  readonly outputConfig: {
    readonly format: {
      readonly type: "json_schema";
      readonly name: "refund_decision";
      readonly schema: typeof REFUND_OUTPUT_SCHEMA;
    };
  };
}

export interface AnthropicStructuredResponse {
  readonly model: string;
  readonly stopReason: string;
  readonly output: unknown;
}

export interface AnthropicTransport {
  readonly executionBackend: AnthropicExecutionBackend;
  generate(
    request: AnthropicStructuredRequest,
    signal: AbortSignal,
  ): Promise<AnthropicStructuredResponse>;
}

export interface AnthropicAdapterOptions {
  readonly model: ModelProvenance;
  readonly transport: AnthropicTransport;
  readonly timeoutMs?: number;
}

export function createAnthropicSonnetAdapter(
  options: AnthropicAdapterOptions,
): RefundBaselineAdapter<"structured-api"> {
  const timeoutMs = validateTimeout(options.timeoutMs ?? DEFAULT_BASELINE_TIMEOUT_MS);
  const model = pinnedModelProvenance(options.model, {
    provider: "anthropic",
    name: ANTHROPIC_SONNET_MODEL,
    version: ANTHROPIC_SONNET_MODEL,
    revision: ANTHROPIC_SONNET_MODEL,
  });
  const configuration = Object.freeze({
    provider: "anthropic",
    role: "structured-api",
    model,
    promptVersion: REFUND_PROMPT_VERSION,
    taskSpecSha256: REFUND_TASK_SPEC_SHA256,
    timeoutMs,
    maximumOutputTokens: 256,
    structuredOutputSchema: REFUND_OUTPUT_SCHEMA,
    sampling: "provider-default",
  });
  const provenance = adapterProvenance("refund-anthropic-structured-output", configuration);

  return Object.freeze({
    role: "structured-api",
    model,
    adapter: provenance,
    taskSpecSha256: REFUND_TASK_SPEC_SHA256,
    trainingEvidence: null,
    resolveExecutionBackend(): Promise<AnthropicExecutionBackend> {
      return Promise.resolve(options.transport.executionBackend);
    },
    async predict(inputs: RefundInputs, signal?: AbortSignal): Promise<BaselinePrediction> {
      const prompt = buildRefundPrompt(inputs);
      const request: AnthropicStructuredRequest = Object.freeze({
        model: ANTHROPIC_SONNET_MODEL,
        maxTokens: 256,
        system: prompt.system,
        messages: Object.freeze([
          Object.freeze({
            role: "user",
            content: prompt.user,
          }),
        ] as const),
        outputConfig: Object.freeze({
          format: Object.freeze({
            type: "json_schema",
            name: "refund_decision",
            schema: REFUND_OUTPUT_SCHEMA,
          }),
        }),
      });
      const response = await invokeWithTimeout(
        `Anthropic ${ANTHROPIC_SONNET_MODEL}`,
        timeoutMs,
        signal,
        async (transportSignal) => options.transport.generate(request, transportSignal),
      );
      const responseRecord = responseObject(response, ["model", "stopReason", "output"]);
      assertResponseModel(responseRecord["model"], ANTHROPIC_SONNET_MODEL);
      if (
        responseRecord["stopReason"] === "refusal" ||
        responseRecord["stopReason"] === "refused"
      ) {
        refused("Anthropic");
      }
      if (responseRecord["stopReason"] !== "end_turn") {
        invalid("response.stopReason", 'must be "end_turn"');
      }
      return parseStructuredPrediction(responseRecord["output"]);
    },
  });
}
