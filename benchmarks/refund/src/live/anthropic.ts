import {
  type AnthropicStructuredRequest,
  type AnthropicStructuredResponse,
  type AnthropicTransport,
} from "../adapters/anthropic.js";
import {
  LiveTransportError,
  configuredBaseUrl,
  denseArray,
  fetchBoundedJson,
  plainRecord,
  type FetchImplementation,
} from "./http.js";

export const DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com" as const;
export const ANTHROPIC_API_VERSION = "2023-06-01" as const;
export const MAXIMUM_ANTHROPIC_RESPONSE_BYTES = 65_536 as const;

export interface LiveAnthropicTransportOptions {
  readonly apiKey?: string;
  readonly baseUrl?: string;
  readonly fetchImplementation?: FetchImplementation;
}

export function createLiveAnthropicTransport(
  options: LiveAnthropicTransportOptions = {},
): AnthropicTransport {
  const apiKey = validateApiKey(options.apiKey ?? process.env["ANTHROPIC_API_KEY"]);
  const baseUrl = configuredBaseUrl(
    options.baseUrl ?? DEFAULT_ANTHROPIC_BASE_URL,
    "Anthropic",
    ["https:"],
  );
  if (baseUrl.origin !== DEFAULT_ANTHROPIC_BASE_URL) {
    throw new LiveTransportError(
      "configuration",
      "Anthropic credentials may be sent only to the canonical API origin",
    );
  }
  const fetchImplementation = options.fetchImplementation ?? fetch;

  return Object.freeze({
    executionBackend: Object.freeze({
      kind: "anthropic-api",
      apiVersion: ANTHROPIC_API_VERSION,
      endpoint: DEFAULT_ANTHROPIC_BASE_URL,
      placement: "provider-managed",
    }),
    async generate(
      request: AnthropicStructuredRequest,
      signal: AbortSignal,
    ): Promise<AnthropicStructuredResponse> {
      const body = Object.freeze({
        model: request.model,
        max_tokens: request.maxTokens,
        system: request.system,
        messages: request.messages,
        output_config: Object.freeze({
          format: Object.freeze({
            type: request.outputConfig.format.type,
            schema: request.outputConfig.format.schema,
          }),
        }),
      });
      const response = await fetchBoundedJson(
        fetchImplementation,
        new URL("/v1/messages", baseUrl),
        Object.freeze({
          method: "POST",
          headers: Object.freeze({
            accept: "application/json",
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
            "x-api-key": apiKey,
          }),
          body: JSON.stringify(body),
          signal,
        }),
        MAXIMUM_ANTHROPIC_RESPONSE_BYTES,
        "Anthropic Messages API",
      );
      return mapAnthropicResponse(response);
    },
  });
}

function mapAnthropicResponse(value: unknown): AnthropicStructuredResponse {
  const root = plainRecord(value, "Anthropic Messages API");
  const model = requiredString(root, "model");
  const stopReason = requiredString(root, "stop_reason");
  if (stopReason === "refusal" || stopReason === "refused") {
    return { model, stopReason: "refusal", output: null };
  }
  if (stopReason !== "end_turn") {
    return { model, stopReason, output: null };
  }

  const blocks = denseArray(root["content"], "Anthropic Messages API content");
  const textBlocks = blocks.filter((block) => {
    const record = plainRecord(block, "Anthropic Messages API content block");
    return record["type"] === "text";
  });
  if (textBlocks.length !== 1) {
    throw new LiveTransportError(
      "invalid-response",
      "Anthropic structured response must contain exactly one text block",
    );
  }
  const textBlock = plainRecord(textBlocks[0], "Anthropic Messages API text block");
  const text = requiredString(textBlock, "text");
  let output: unknown;
  try {
    output = JSON.parse(text) as unknown;
  } catch {
    throw new LiveTransportError(
      "invalid-response",
      "Anthropic structured response text is not valid JSON",
    );
  }
  return { model, stopReason, output };
}

function validateApiKey(value: string | undefined): string {
  if (
    value === undefined ||
    value.length === 0 ||
    value.length > 512 ||
    value.includes("\r") ||
    value.includes("\n")
  ) {
    throw new LiveTransportError(
      "configuration",
      "Anthropic API key is missing or invalid",
    );
  }
  return value;
}

function requiredString(
  record: Readonly<Record<string, unknown>>,
  key: string,
): string {
  const value = record[key];
  if (typeof value !== "string" || value.length === 0) {
    throw new LiveTransportError(
      "invalid-response",
      `Anthropic Messages API response field ${key} must be a nonempty string`,
    );
  }
  return value;
}
