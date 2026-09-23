import { semanticJsonSha256 } from "@semantscript/compiler";

import { contractInternals } from "../contracts.js";
import {
  REFUND_SUPPORT,
  type AdapterProvenance,
  type BenchmarkSystemRole,
  type DistributionEntry,
  type ExecutionBackendProvenance,
  type ModelProvenance,
  type RefundDecision,
  type RefundInputs,
  type SemantScriptTrainingEvidence,
} from "../types.js";
import {
  REFUND_BASELINE_POLICY,
  REFUND_TASK_SPEC_SHA256,
} from "../policy.js";

export const BASELINE_ADAPTER_VERSION = "2" as const;
export const REFUND_PROMPT_VERSION = "refund-decision.v2" as const;
export const DEFAULT_BASELINE_TIMEOUT_MS = 120_000 as const;
export const MAXIMUM_STRUCTURED_RESPONSE_BYTES = 16_384 as const;
/**
 * Generative baselines state probabilities as rounded decimals, so sums such as
 * 0.99 or 1.02 are rounding, not malformed output. Sums inside this tolerance are
 * renormalized into the sealed distribution; anything further off is invalid.
 */
export const PROBABILITY_SUM_TOLERANCE = 0.1 as const;

export const REFUND_OUTPUT_SCHEMA = Object.freeze({
  type: "object",
  additionalProperties: false,
  required: Object.freeze(["decision", "probabilities"]),
  properties: Object.freeze({
    decision: Object.freeze({ type: "string", enum: REFUND_SUPPORT }),
    probabilities: Object.freeze({
      type: "object",
      additionalProperties: false,
      required: REFUND_SUPPORT,
      properties: Object.freeze(
        Object.fromEntries(
          REFUND_SUPPORT.map((value) => [
            value,
            Object.freeze({ type: "number", minimum: 0, maximum: 1 }),
          ]),
        ),
      ),
    }),
  }),
} as const);

const SHA256 = /^[a-f0-9]{64}$/;

export type GenerativeBaselineRole = "ollama-1b" | "ollama-7b" | "structured-api";
export type BaselineRole = GenerativeBaselineRole | "laya";

export interface BaselinePrediction {
  readonly value: RefundDecision;
  readonly distribution: readonly DistributionEntry[];
}

export interface RefundBaselineAdapter<Role extends BaselineRole = BaselineRole> {
  readonly role: Role;
  readonly model: ModelProvenance;
  readonly adapter: AdapterProvenance;
  readonly taskSpecSha256: string;
  readonly trainingEvidence: SemantScriptTrainingEvidence | null;
  resolveExecutionBackend(signal: AbortSignal): Promise<ExecutionBackendProvenance>;
  predict(inputs: RefundInputs, signal?: AbortSignal): Promise<BaselinePrediction>;
}

export type BaselineAdapterErrorCode =
  | "aborted"
  | "invalid-configuration"
  | "invalid-response"
  | "refused-output"
  | "timeout"
  | "transport-error";

export class BaselineAdapterError extends Error {
  readonly code: BaselineAdapterErrorCode;

  constructor(code: BaselineAdapterErrorCode, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "BaselineAdapterError";
    this.code = code;
  }
}

export function canonicalRefundInput(inputs: RefundInputs): string {
  validateRefundInputs(inputs);
  return JSON.stringify({
    customer: {
      priorRefunds: inputs.customer.priorRefunds,
      tier: inputs.customer.tier,
    },
    order: {
      ageDays: inputs.order.ageDays,
      status: inputs.order.status,
      total: inputs.order.total,
    },
  });
}

export function buildRefundPrompt(inputs: RefundInputs): {
  readonly system: string;
  readonly user: string;
} {
  return Object.freeze({
    system:
      "Apply the complete refund policy and hard constraints supplied by the user. Return only the required structured decision and a complete probability distribution. The decision must be the first option in support order among tied maximum probabilities.",
    user: [
      `Prompt protocol: ${REFUND_PROMPT_VERSION}`,
      `Task specification SHA-256: ${REFUND_TASK_SPEC_SHA256}`,
      `Policy: ${REFUND_BASELINE_POLICY.policy}.`,
      "Hard constraints:",
      ...REFUND_BASELINE_POLICY.hardConstraints.map(
        (constraint, index) => `${String(index + 1)}. ${constraint}`,
      ),
      'Input domain: order.status is exactly "paid" or "fraudulent".',
      `Support order: ${REFUND_SUPPORT.join(", ")}`,
      `Refund input (canonical refund JSON): ${canonicalRefundInput(inputs)}`,
    ].join("\n"),
  });
}

export function parseStructuredPrediction(value: unknown): BaselinePrediction {
  const root = exactObject(value, "response", ["decision", "probabilities"]);
  const decision = refundDecision(root.decision, "response.decision");
  const probabilities = exactObject(
    root.probabilities,
    "response.probabilities",
    REFUND_SUPPORT,
  );
  const weights = REFUND_SUPPORT.map((supportValue) =>
    probability(probabilities[supportValue], `response.probabilities.${supportValue}`),
  );
  const sum = weights.reduce((total, weight) => total + weight, 0);
  if (sum === 0) {
    invalid("response.probabilities", "must assign positive mass to at least one value");
  }
  if (Math.abs(sum - 1) > PROBABILITY_SUM_TOLERANCE) {
    invalid(
      "response.probabilities",
      `must sum to 1 within ${String(PROBABILITY_SUM_TOLERANCE)}`,
    );
  }
  return predictionFromWeights(decision, weights);
}

export function parseStructuredPredictionJson(value: unknown): BaselinePrediction {
  if (typeof value !== "string" || value.length === 0) {
    invalid("response", "must be a nonempty JSON string");
  }
  if (new TextEncoder().encode(value).byteLength > MAXIMUM_STRUCTURED_RESPONSE_BYTES) {
    invalid(
      "response",
      `must not exceed ${String(MAXIMUM_STRUCTURED_RESPONSE_BYTES)} UTF-8 bytes`,
    );
  }
  let decoded: unknown;
  try {
    decoded = JSON.parse(value) as unknown;
  } catch (error) {
    throw new BaselineAdapterError("invalid-response", "response is not valid JSON", {
      cause: error,
    });
  }
  return parseStructuredPrediction(decoded);
}

export function predictionFromLogits(
  selectedIndex: unknown,
  logitsValue: unknown,
): BaselinePrediction {
  if (!Number.isSafeInteger(selectedIndex) || (selectedIndex as number) < 0) {
    invalid("response.selectedIndex", "must be a non-negative safe integer");
  }
  const logits = denseNumbers(logitsValue, "response.logits", false);
  if (logits.length !== REFUND_SUPPORT.length) {
    invalid(
      "response.logits",
      `must contain exactly ${String(REFUND_SUPPORT.length)} logits`,
    );
  }
  const maximum = Math.max(...logits);
  const weights = logits.map((logit) => Math.exp(logit - maximum));
  const expectedIndex = stableArgmax(weights);
  if (selectedIndex !== expectedIndex) {
    invalid(
      "response.selectedIndex",
      "must be the stable support-order argmax of logits",
    );
  }
  const decision = REFUND_SUPPORT[expectedIndex];
  if (decision === undefined) {
    invalid("response.selectedIndex", "is outside the refund support");
  }
  return predictionFromWeights(decision, weights);
}

export function adapterProvenance(
  name: string,
  configuration: Readonly<{ taskSpecSha256: string }>,
): AdapterProvenance {
  if (configuration.taskSpecSha256 !== REFUND_TASK_SPEC_SHA256) {
    configurationError(
      "configuration.taskSpecSha256",
      `must bind the canonical refund task specification ${REFUND_TASK_SPEC_SHA256}`,
    );
  }
  return Object.freeze({
    name,
    version: BASELINE_ADAPTER_VERSION,
    configurationSha256: semanticJsonSha256(configuration),
  });
}

export function pinnedModelProvenance(
  value: ModelProvenance,
  expected: {
    readonly provider: string;
    readonly name: string;
    readonly version: string;
    readonly revision?: string;
  },
): ModelProvenance {
  try {
    contractInternals.validateModelProvenance(value, "model");
  } catch (error) {
    throw new BaselineAdapterError("invalid-configuration", errorMessage(error), {
      cause: error,
    });
  }
  for (const key of ["provider", "name", "version"] as const) {
    if (value[key] !== expected[key]) {
      configurationError(`model.${key}`, `must be ${JSON.stringify(expected[key])}`);
    }
  }
  if (expected.revision !== undefined && value.revision !== expected.revision) {
    configurationError("model.revision", `must be ${JSON.stringify(expected.revision)}`);
  }
  if (value.revision === "latest" || value.revision.endsWith(":latest")) {
    configurationError("model.revision", "must be immutable, not latest");
  }
  return contractInternals.frozenClone(value);
}

export function validateTimeout(value: number): number {
  if (!Number.isSafeInteger(value) || value <= 0 || value > 600_000) {
    configurationError("timeoutMs", "must be a positive safe integer no greater than 600000");
  }
  return value;
}

export async function invokeWithTimeout<T>(
  label: string,
  timeoutMs: number,
  callerSignal: AbortSignal | undefined,
  operation: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  if (callerSignal?.aborted === true) {
    throw new BaselineAdapterError("aborted", `${label} was aborted before dispatch`);
  }

  const controller = new AbortController();
  const timeoutReason = Object.freeze({ kind: "baseline-adapter-timeout" });
  const onCallerAbort = (): void => {
    controller.abort(callerSignal?.reason);
  };
  callerSignal?.addEventListener("abort", onCallerAbort, { once: true });
  const timer = setTimeout(() => {
    controller.abort(timeoutReason);
  }, timeoutMs);

  let rejectAbort: ((reason: BaselineAdapterError) => void) | undefined;
  const aborted = new Promise<never>((_resolve, reject) => {
    rejectAbort = reject;
  });
  const onAbort = (): void => {
    rejectAbort?.(
      new BaselineAdapterError(
        controller.signal.reason === timeoutReason ? "timeout" : "aborted",
        controller.signal.reason === timeoutReason
          ? `${label} exceeded its ${String(timeoutMs)}ms timeout`
          : `${label} was aborted`,
      ),
    );
  };
  controller.signal.addEventListener("abort", onAbort, { once: true });

  try {
    return await Promise.race([operation(controller.signal), aborted]);
  } catch (error) {
    if (error instanceof BaselineAdapterError) {
      throw error;
    }
    if (controller.signal.aborted && controller.signal.reason === timeoutReason) {
      throw new BaselineAdapterError(
        "timeout",
        `${label} exceeded its ${String(timeoutMs)}ms timeout`,
        { cause: error },
      );
    }
    if (controller.signal.aborted) {
      throw new BaselineAdapterError("aborted", `${label} was aborted`, { cause: error });
    }
    throw new BaselineAdapterError("transport-error", `${label} failed`, { cause: error });
  } finally {
    clearTimeout(timer);
    callerSignal?.removeEventListener("abort", onCallerAbort);
    controller.signal.removeEventListener("abort", onAbort);
  }
}

export function assertResponseModel(actual: unknown, expected: string): void {
  if (actual !== expected) {
    invalid("response.model", `must exactly match requested model ${JSON.stringify(expected)}`);
  }
}

export function assertExactRevision(actual: unknown, expected: string): void {
  if (actual !== expected) {
    invalid(
      "response.checkpointRevision",
      `must exactly match pinned revision ${JSON.stringify(expected)}`,
    );
  }
}

export function responseObject(
  value: unknown,
  requiredKeys: readonly string[],
): Readonly<Record<string, unknown>> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    invalid("response", "must be an object");
  }
  const prototype = Object.getPrototypeOf(value) as unknown;
  if (prototype !== Object.prototype && prototype !== null) {
    invalid("response", "must be a plain object");
  }
  if (Object.getOwnPropertySymbols(value).length > 0) {
    invalid("response", "must not contain symbol properties");
  }
  const record = value as Record<string, unknown>;
  const names = Object.getOwnPropertyNames(record);
  for (const name of names) {
    const descriptor = Object.getOwnPropertyDescriptor(record, name);
    if (!descriptor?.enumerable || !("value" in descriptor)) {
      invalid(`response.${name}`, "must be an enumerable data property");
    }
  }
  for (const key of requiredKeys) {
    if (!names.includes(key)) {
      invalid("response", `must contain ${JSON.stringify(key)}`);
    }
  }
  return record;
}

export function invalid(path: string, message: string): never {
  throw new BaselineAdapterError("invalid-response", `${path}: ${message}`);
}

export function refused(provider: string): never {
  throw new BaselineAdapterError(
    "refused-output",
    `${provider} refused or declined the fixed benchmark input`,
  );
}

export function validateRole(role: BenchmarkSystemRole, expected: BaselineRole): void {
  if (role !== expected) {
    configurationError("role", `must be ${JSON.stringify(expected)}`);
  }
}

function predictionFromWeights(
  claimedDecision: RefundDecision,
  weights: readonly number[],
): BaselinePrediction {
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  if (!Number.isFinite(total) || total <= 0) {
    invalid("response", "distribution mass must be positive and finite");
  }
  const probabilities = weights.map((weight) => weight / total);
  const bestIndex = stableArgmax(probabilities);
  if (claimedDecision !== REFUND_SUPPORT[bestIndex]) {
    invalid("response.decision", "must be the stable support-order argmax of probabilities");
  }
  const distribution: DistributionEntry[] = REFUND_SUPPORT.map((value, index) => {
    const probabilityAtIndex = probabilities[index];
    if (probabilityAtIndex === undefined) {
      invalid("response", "distribution cardinality does not match refund support");
    }
    return {
      value,
      probability: probabilityAtIndex,
    };
  });
  return contractInternals.frozenClone({
    value: claimedDecision,
    distribution,
  });
}

function stableArgmax(values: readonly number[]): number {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (const [index, value] of values.entries()) {
    if (value > bestValue) {
      bestIndex = index;
      bestValue = value;
    }
  }
  return bestIndex;
}

function validateRefundInputs(value: unknown): asserts value is RefundInputs {
  const inputs = exactObject(value, "inputs", ["customer", "order"]);
  const customer = exactObject(inputs.customer, "inputs.customer", ["priorRefunds", "tier"]);
  if (
    !Number.isSafeInteger(customer.priorRefunds) ||
    (customer.priorRefunds as number) < 0 ||
    Object.is(customer.priorRefunds, -0)
  ) {
    configurationError("inputs.customer.priorRefunds", "must be a non-negative safe integer");
  }
  if (customer.tier !== "enterprise" && customer.tier !== "standard") {
    configurationError("inputs.customer.tier", 'must be "enterprise" or "standard"');
  }
  const order = exactObject(inputs.order, "inputs.order", ["ageDays", "status", "total"]);
  for (const key of ["ageDays", "total"] as const) {
    if (
      typeof order[key] !== "number" ||
      !Number.isFinite(order[key]) ||
      order[key] < 0 ||
      Object.is(order[key], -0)
    ) {
      configurationError(`inputs.order.${key}`, "must be a non-negative finite number");
    }
  }
  if (order.status !== "fraudulent" && order.status !== "paid") {
    configurationError("inputs.order.status", 'must be "fraudulent" or "paid"');
  }
}

function probability(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    invalid(path, "must be a finite probability from 0 through 1");
  }
  return value;
}

function denseNumbers(value: unknown, path: string, probabilities: boolean): readonly number[] {
  const names = Array.isArray(value)
    ? Object.getOwnPropertyNames(value).filter((name) => name !== "length")
    : [];
  if (
    !Array.isArray(value) ||
    Object.getPrototypeOf(value) !== Array.prototype ||
    Object.getOwnPropertySymbols(value).length > 0 ||
    names.length !== value.length ||
    names.some((name, index) => name !== String(index))
  ) {
    invalid(path, "must be a dense array");
  }
  return value.map((entry, index) => {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor?.enumerable || !("value" in descriptor)) {
      invalid(`${path}[${String(index)}]`, "must be an enumerable data property");
    }
    if (typeof entry !== "number" || !Number.isFinite(entry)) {
      invalid(`${path}[${String(index)}]`, "must be a finite number");
    }
    if (probabilities && (entry < 0 || entry > 1)) {
      invalid(`${path}[${String(index)}]`, "must be a probability");
    }
    return entry;
  });
}

function exactObject<const Keys extends readonly string[]>(
  value: unknown,
  path: string,
  keys: Keys,
): { [Key in Keys[number]]: unknown } {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    invalid(path, "must be an object");
  }
  const prototype = Object.getPrototypeOf(value) as unknown;
  if (prototype !== Object.prototype && prototype !== null) {
    invalid(path, "must be a plain object");
  }
  if (Object.getOwnPropertySymbols(value).length > 0) {
    invalid(path, "must not contain symbol properties");
  }
  const names = Object.getOwnPropertyNames(value);
  if (names.length !== keys.length || keys.some((key) => !names.includes(key))) {
    invalid(path, `must contain exactly: ${keys.join(", ")}`);
  }
  for (const key of keys) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor?.enumerable || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "must be an enumerable data property");
    }
  }
  return value as { [Key in Keys[number]]: unknown };
}

function configurationError(path: string, message: string): never {
  throw new BaselineAdapterError("invalid-configuration", `${path}: ${message}`);
}

function refundDecision(value: unknown, path: string): RefundDecision {
  if (value !== "approve" && value !== "deny" && value !== "review") {
    invalid(path, "must be a canonical refund decision");
  }
  return value;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function requireSha256(value: string, path: string): string {
  if (!SHA256.test(value)) {
    configurationError(path, "must be 64 lowercase hexadecimal characters");
  }
  return value;
}
