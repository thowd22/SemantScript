export const MAXIMUM_ERROR_RESPONSE_BYTES = 65_536 as const;

export type FetchImplementation = typeof fetch;

export class LiveTransportError extends Error {
  readonly code:
    | "configuration"
    | "http-status"
    | "invalid-response"
    | "process-failure"
    | "response-too-large";

  constructor(
    code: LiveTransportError["code"],
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "LiveTransportError";
    this.code = code;
  }
}

export async function fetchBoundedJson(
  fetchImplementation: FetchImplementation,
  url: URL,
  init: RequestInit,
  maximumBytes: number,
  provider: string,
): Promise<unknown> {
  let response: Response;
  try {
    response = await fetchImplementation(url, init);
  } catch {
    throw new LiveTransportError(
      "invalid-response",
      `${provider} request failed before a response was received`,
    );
  }

  let bytes: Uint8Array;
  try {
    bytes = await readBoundedBody(
      response,
      response.ok ? maximumBytes : MAXIMUM_ERROR_RESPONSE_BYTES,
    );
  } catch (error) {
    if (error instanceof LiveTransportError) {
      throw error;
    }
    throw new LiveTransportError(
      "invalid-response",
      `${provider} response could not be read`,
    );
  }

  if (!response.ok) {
    throw new LiveTransportError(
      "http-status",
      `${provider} request failed with HTTP status ${String(response.status)}`,
    );
  }

  try {
    return JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(bytes),
    ) as unknown;
  } catch {
    throw new LiveTransportError(
      "invalid-response",
      `${provider} returned invalid UTF-8 JSON`,
    );
  }
}

export function configuredBaseUrl(
  value: string,
  provider: string,
  allowedProtocols: readonly string[],
): URL {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new LiveTransportError(
      "configuration",
      `${provider} base URL is invalid`,
    );
  }
  if (!allowedProtocols.includes(url.protocol)) {
    throw new LiveTransportError(
      "configuration",
      `${provider} base URL must use ${allowedProtocols.join(" or ")}`,
    );
  }
  if (
    url.username !== "" ||
    url.password !== "" ||
    url.search !== "" ||
    url.hash !== "" ||
    (url.pathname !== "" && url.pathname !== "/")
  ) {
    throw new LiveTransportError(
      "configuration",
      `${provider} base URL must contain only scheme, host, and optional port`,
    );
  }
  return url;
}

export function plainRecord(
  value: unknown,
  provider: string,
): Readonly<Record<string, unknown>> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new LiveTransportError(
      "invalid-response",
      `${provider} response must be an object`,
    );
  }
  const prototype = Object.getPrototypeOf(value) as unknown;
  if (prototype !== Object.prototype && prototype !== null) {
    throw new LiveTransportError(
      "invalid-response",
      `${provider} response must be a plain object`,
    );
  }
  return value as Readonly<Record<string, unknown>>;
}

export function denseArray(
  value: unknown,
  provider: string,
): readonly unknown[] {
  if (!Array.isArray(value)) {
    throw new LiveTransportError(
      "invalid-response",
      `${provider} response must be an array`,
    );
  }
  const names = Object.getOwnPropertyNames(value).filter(
    (name) => name !== "length",
  );
  if (
    Object.getPrototypeOf(value) !== Array.prototype ||
    Object.getOwnPropertySymbols(value).length > 0 ||
    names.length !== value.length ||
    names.some((name, index) => name !== String(index))
  ) {
    throw new LiveTransportError(
      "invalid-response",
      `${provider} response must be a dense plain array`,
    );
  }
  return value;
}

async function readBoundedBody(
  response: Response,
  maximumBytes: number,
): Promise<Uint8Array> {
  const contentLength = response.headers.get("content-length");
  if (contentLength !== null) {
    const parsed = Number(contentLength);
    if (Number.isFinite(parsed) && parsed > maximumBytes) {
      throw new LiveTransportError(
        "response-too-large",
        `provider response exceeds the ${String(maximumBytes)} byte limit`,
      );
    }
  }
  if (response.body === null) {
    return new Uint8Array();
  }

  const chunks: Uint8Array[] = [];
  let length = 0;
  const body = response.body as unknown as AsyncIterable<unknown>;
  for await (const value of body) {
    if (!(value instanceof Uint8Array)) {
      throw new LiveTransportError(
        "invalid-response",
        "provider response body emitted a non-byte chunk",
      );
    }
    length += value.byteLength;
    if (length > maximumBytes) {
      throw new LiveTransportError(
        "response-too-large",
        `provider response exceeds the ${String(maximumBytes)} byte limit`,
      );
    }
    chunks.push(value);
  }

  const result = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}
