import {
  OLLAMA_QWEN_MANIFEST_SHA256,
  OLLAMA_QWEN_MODELS,
  type OllamaBaselineRole,
  type OllamaQwenModel,
  type OllamaGenerateRequest,
  type OllamaGenerateResponse,
  type OllamaTransport,
} from "../adapters/ollama.js";
import type { OllamaExecutionBackend } from "../types.js";
import {
  LiveTransportError,
  configuredBaseUrl,
  denseArray,
  fetchBoundedJson,
  plainRecord,
  type FetchImplementation,
} from "./http.js";

export const DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434" as const;
export const MAXIMUM_OLLAMA_TAGS_RESPONSE_BYTES = 1_048_576 as const;
export const MAXIMUM_OLLAMA_GENERATE_RESPONSE_BYTES = 65_536 as const;
export const MAXIMUM_OLLAMA_RUNTIME_RESPONSE_BYTES = 1_048_576 as const;

export interface LiveOllamaTransportOptions {
  readonly baseUrl?: string;
  readonly fetchImplementation?: FetchImplementation;
}

export function createLiveOllamaTransport(
  options: LiveOllamaTransportOptions = {},
): OllamaTransport {
  const baseUrl = configuredBaseUrl(
    options.baseUrl ?? DEFAULT_OLLAMA_BASE_URL,
    "Ollama",
    ["http:", "https:"],
  );
  if (
    baseUrl.protocol === "http:" &&
    !["127.0.0.1", "localhost", "[::1]"].includes(baseUrl.hostname)
  ) {
    throw new LiveTransportError(
      "configuration",
      "Ollama plain HTTP is allowed only for a loopback host",
    );
  }
  const fetchImplementation = options.fetchImplementation ?? fetch;

  return Object.freeze({
    async resolveExecutionBackend(
      model: OllamaQwenModel,
      signal: AbortSignal,
    ): Promise<OllamaExecutionBackend> {
      const role = roleForModel(model);
      const [versionValue, runningValue] = await Promise.all([
        fetchBoundedJson(
          fetchImplementation,
          new URL("/api/version", baseUrl),
          Object.freeze({
            method: "GET",
            headers: Object.freeze({ accept: "application/json" }),
            signal,
          }),
          MAXIMUM_OLLAMA_RUNTIME_RESPONSE_BYTES,
          "Ollama server version",
        ),
        fetchBoundedJson(
          fetchImplementation,
          new URL("/api/ps", baseUrl),
          Object.freeze({
            method: "GET",
            headers: Object.freeze({ accept: "application/json" }),
            signal,
          }),
          MAXIMUM_OLLAMA_RUNTIME_RESPONSE_BYTES,
          "Ollama running-model state",
        ),
      ]);
      const version = stringField(
        plainRecord(versionValue, "Ollama server version"),
        "version",
      );
      const running = plainRecord(runningValue, "Ollama running-model state");
      const models = denseArray(
        running["models"],
        "Ollama running-model state models",
      );
      const matches = models.filter((entry) => {
        const record = plainRecord(entry, "Ollama running-model state entry");
        return record["name"] === model || record["model"] === model;
      });
      if (matches.length !== 1) {
        throw new LiveTransportError(
          "invalid-response",
          "Ollama running-model state must contain exactly one exact pinned model entry",
        );
      }
      const match = plainRecord(matches[0], "Ollama running-model state entry");
      if (match["digest"] !== OLLAMA_QWEN_MANIFEST_SHA256[role]) {
        throw new LiveTransportError(
          "invalid-response",
          "Ollama running model digest does not match the pinned manifest",
        );
      }
      const modelTotalBytes = safeByteCount(match, "size");
      const modelGpuBytes = safeByteCount(match, "size_vram");
      if (modelTotalBytes === 0 || modelGpuBytes > modelTotalBytes) {
        throw new LiveTransportError(
          "invalid-response",
          "Ollama running-model byte counts are inconsistent",
        );
      }
      const modelCpuBytes = modelTotalBytes - modelGpuBytes;
      const placement =
        modelGpuBytes === 0 ? "cpu" : modelCpuBytes === 0 ? "gpu" : "hybrid";
      return Object.freeze({
        kind: "ollama",
        serverVersion: version,
        placement,
        modelTotalBytes,
        modelCpuBytes,
        modelGpuBytes,
      });
    },
    async generate(
      request: OllamaGenerateRequest,
      signal: AbortSignal,
    ): Promise<OllamaGenerateResponse> {
      const role = roleForModel(request.model);
      const tags = await fetchBoundedJson(
        fetchImplementation,
        new URL("/api/tags", baseUrl),
        Object.freeze({
          method: "GET",
          headers: Object.freeze({ accept: "application/json" }),
          signal,
        }),
        MAXIMUM_OLLAMA_TAGS_RESPONSE_BYTES,
        "Ollama model-list",
      );
      verifyManifestDigest(
        tags,
        request.model,
        OLLAMA_QWEN_MANIFEST_SHA256[role],
      );

      const response = await fetchBoundedJson(
        fetchImplementation,
        new URL("/api/generate", baseUrl),
        Object.freeze({
          method: "POST",
          headers: Object.freeze({
            accept: "application/json",
            "content-type": "application/json",
          }),
          body: JSON.stringify(request),
          signal,
        }),
        MAXIMUM_OLLAMA_GENERATE_RESPONSE_BYTES,
        "Ollama generate",
      );
      const record = plainRecord(response, "Ollama generate");
      return {
        model: stringField(record, "model"),
        response: stringField(record, "response"),
        done: booleanField(record, "done"),
        done_reason: stringField(record, "done_reason"),
      };
    },
  });
}

function safeByteCount(
  record: Readonly<Record<string, unknown>>,
  key: string,
): number {
  const value = record[key];
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new LiveTransportError(
      "invalid-response",
      `Ollama running-model field ${key} must be a non-negative safe integer`,
    );
  }
  return value as number;
}

function roleForModel(model: string): OllamaBaselineRole {
  for (const role of Object.keys(OLLAMA_QWEN_MODELS) as OllamaBaselineRole[]) {
    if (OLLAMA_QWEN_MODELS[role] === model) {
      return role;
    }
  }
  throw new LiveTransportError(
    "configuration",
    "Ollama transport received a model outside the pinned Qwen benchmark set",
  );
}

function verifyManifestDigest(
  value: unknown,
  model: string,
  expectedDigest: string,
): void {
  const root = plainRecord(value, "Ollama model-list");
  const models = denseArray(root["models"], "Ollama model-list models");
  const matches = models.filter((entry) => {
    const record = plainRecord(entry, "Ollama model-list entry");
    return record["name"] === model || record["model"] === model;
  });
  if (matches.length !== 1) {
    throw new LiveTransportError(
      "invalid-response",
      "Ollama model-list must contain exactly one exact pinned model entry",
    );
  }
  const match = plainRecord(matches[0], "Ollama model-list entry");
  if (match["digest"] !== expectedDigest) {
    throw new LiveTransportError(
      "invalid-response",
      "Ollama model manifest digest does not match the pinned benchmark digest",
    );
  }
}

function stringField(
  record: Readonly<Record<string, unknown>>,
  key: string,
): string {
  const value = record[key];
  if (typeof value !== "string") {
    throw new LiveTransportError(
      "invalid-response",
      `Ollama generate response field ${key} must be a string`,
    );
  }
  return value;
}

function booleanField(
  record: Readonly<Record<string, unknown>>,
  key: string,
): boolean {
  const value = record[key];
  if (typeof value !== "boolean") {
    throw new LiveTransportError(
      "invalid-response",
      `Ollama generate response field ${key} must be a boolean`,
    );
  }
  return value;
}
