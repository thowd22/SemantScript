import { performance } from "node:perf_hooks";

import { semanticJsonSha256 } from "@semantscript/compiler";

import { sealPredictionSet, validateRefundDataset } from "./contracts.js";
import type { BaselinePrediction } from "./adapters/common.js";
import {
  REFUND_SUPPORT,
  REQUIRED_SYSTEM_ROLES,
  type AdapterProvenance,
  type BenchmarkSystemRole,
  type EnvironmentProvenance,
  type ExecutionBackendProvenance,
  type MemoryScope,
  type ModelProvenance,
  type RefundBenchmarkDatasetV1,
  type RefundInputs,
  type RefundPrediction,
  type RefundPredictionSetV1,
  type SemantScriptTrainingEvidence,
  type SystemProvenance,
} from "./types.js";
import { contractInternals } from "./contracts.js";

export const MAXIMUM_WARMUP_ITERATIONS = 10_000 as const;

export type BenchmarkRunnerErrorCode =
  | "aborted"
  | "invalid-configuration"
  | "invalid-memory-sample"
  | "invalid-output"
  | "non-monotonic-clock";

export class BenchmarkRunnerError extends Error {
  readonly code: BenchmarkRunnerErrorCode;

  constructor(
    code: BenchmarkRunnerErrorCode,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "BenchmarkRunnerError";
    this.code = code;
  }
}

/**
 * The runner accepts this structural interface so the SemantScript system and
 * every fixed baseline use precisely the same execution path.
 */
export interface RefundPredictionAdapter<
  Role extends BenchmarkSystemRole = BenchmarkSystemRole,
> {
  readonly role: Role;
  readonly model: ModelProvenance;
  readonly adapter: AdapterProvenance;
  readonly taskSpecSha256: string;
  readonly trainingEvidence: SemantScriptTrainingEvidence | null;
  resolveExecutionBackend(signal: AbortSignal): Promise<ExecutionBackendProvenance>;
  predict(inputs: RefundInputs, signal?: AbortSignal): Promise<BaselinePrediction>;
}

export interface MonotonicClock {
  now(): number;
}

/**
 * A sampler owns any polling needed to observe the sampled peak between start and
 * stop. Its scope is recorded verbatim in the sealed measurement protocol.
 */
export interface PeakMemorySampler {
  readonly scope: MemoryScope;
  start(signal: AbortSignal): void | Promise<void>;
  stop(): number | Promise<number>;
}

export interface RefundBenchmarkRunnerOptions<
  Role extends BenchmarkSystemRole = BenchmarkSystemRole,
> {
  readonly dataset: RefundBenchmarkDatasetV1;
  readonly adapter: RefundPredictionAdapter<Role>;
  readonly environment: EnvironmentProvenance;
  readonly warmupIterations: number;
  readonly warmupInputs: readonly RefundInputs[];
  readonly memorySampler: PeakMemorySampler;
  readonly clock?: MonotonicClock;
  readonly signal?: AbortSignal;
}

interface ValidatedConfiguration<Role extends BenchmarkSystemRole> {
  readonly dataset: RefundBenchmarkDatasetV1;
  readonly adapter: RefundPredictionAdapter<Role>;
  readonly role: Role;
  readonly model: ModelProvenance;
  readonly adapterProvenance: AdapterProvenance;
  readonly taskSpecSha256: string;
  readonly trainingEvidence: SemantScriptTrainingEvidence | null;
  readonly environment: EnvironmentProvenance;
  readonly warmupIterations: number;
  readonly warmupInputs: readonly RefundInputs[];
  readonly warmupInputOrderSha256: string;
  readonly memorySampler: PeakMemorySampler;
  readonly clock: MonotonicClock;
  readonly callerSignal: AbortSignal | undefined;
}

const defaultClock: MonotonicClock = Object.freeze({
  now: (): number => performance.now(),
});

const supportedRoles = new Set<BenchmarkSystemRole>(REQUIRED_SYSTEM_ROLES);

export function runRefundBenchmark<Role extends BenchmarkSystemRole>(
  options: RefundBenchmarkRunnerOptions<Role>,
): Promise<RefundPredictionSetV1>;
export async function runRefundBenchmark(
  options: unknown,
): Promise<RefundPredictionSetV1> {
  const configuration = validateConfiguration(options);
  const controller = new AbortController();
  const callerSignal = configuration.callerSignal;
  const onCallerAbort = (): void => {
    controller.abort(callerSignal?.reason);
  };
  callerSignal?.addEventListener("abort", onCallerAbort, { once: true });

  let samplerStarted = false;
  try {
    throwIfAborted(callerSignal, "benchmark was aborted before warmup");

    for (let index = 0; index < configuration.warmupIterations; index += 1) {
      throwIfAborted(controller.signal, "benchmark was aborted during warmup");
      const warmupInput = configuration.warmupInputs[
        index % configuration.warmupInputs.length
      ];
      if (warmupInput === undefined) {
        configurationError("warmupInputs", "must contain a warmup input");
      }
      validateAdapterPrediction(
        await configuration.adapter.predict(warmupInput, controller.signal),
      );
      throwIfAborted(controller.signal, "benchmark was aborted during warmup");
    }

    throwIfAborted(controller.signal, "benchmark was aborted before measurement");
    samplerStarted = true;
    await configuration.memorySampler.start(controller.signal);
    throwIfAborted(controller.signal, "benchmark was aborted before measurement");

    let lastClockReading: number | undefined;
    const readClock = (): number => {
      const reading = configuration.clock.now();
      if (!Number.isFinite(reading)) {
        throw new BenchmarkRunnerError(
          "non-monotonic-clock",
          "clock.now() must return a finite number",
        );
      }
      if (lastClockReading !== undefined && reading < lastClockReading) {
        throw new BenchmarkRunnerError(
          "non-monotonic-clock",
          "clock.now() moved backwards during the measured run",
        );
      }
      lastClockReading = reading;
      return reading;
    };

    const measuredStartedAt = readClock();
    const predictions: RefundPrediction[] = [];
    for (const benchmarkCase of configuration.dataset.cases) {
      throwIfAborted(controller.signal, "benchmark was aborted during measurement");
      const startedAt = readClock();
      const rawPrediction = await configuration.adapter.predict(
        benchmarkCase.inputs,
        controller.signal,
      );
      throwIfAborted(controller.signal, "benchmark was aborted during measurement");
      const prediction = validateAdapterPrediction(rawPrediction);
      const completedAt = readClock();
      predictions.push({
        caseId: benchmarkCase.id,
        inputSha256: benchmarkCase.inputSha256,
        value: prediction.value,
        distribution: prediction.distribution,
        latencyMs: completedAt - startedAt,
      });
    }
    const measuredCompletedAt = readClock();
    const measuredDurationMs = measuredCompletedAt - measuredStartedAt;
    if (measuredDurationMs <= 0) {
      throw new BenchmarkRunnerError(
        "non-monotonic-clock",
        "the measured run duration must be greater than zero",
      );
    }

    samplerStarted = false;
    const peakMemoryBytes = validatePeakMemory(
      await configuration.memorySampler.stop(),
    );
    const executionBackend = await configuration.adapter.resolveExecutionBackend(
      controller.signal,
    );

    return sealPredictionSet({
      kind: "semantscript.refund-benchmark-predictions",
      predictionVersion: 1,
      benchmark: "refund-decision",
      datasetSha256: configuration.dataset.payloadSha256,
      support: REFUND_SUPPORT,
      system: {
        role: configuration.role,
        taskSpecSha256: configuration.taskSpecSha256,
        model: configuration.model,
        adapter: configuration.adapterProvenance,
        trainingEvidence: configuration.trainingEvidence,
        executionBackend,
      },
      environment: configuration.environment,
      protocol: {
        concurrency: 1,
        warmupIterations: configuration.warmupIterations,
        warmupInputOrderSha256: configuration.warmupInputOrderSha256,
        measuredDurationMs,
        memoryScope: configuration.memorySampler.scope,
        peakMemoryBytes,
      },
      predictions,
    });
  } catch (error) {
    if (!controller.signal.aborted) {
      controller.abort(error);
    }
    throw error;
  } finally {
    if (samplerStarted) {
      try {
        await configuration.memorySampler.stop();
      } catch {
        // Preserve the primary run failure. No prediction set can escape this path.
      }
    }
    callerSignal?.removeEventListener("abort", onCallerAbort);
  }
}

function validateConfiguration(
  options: unknown,
): ValidatedConfiguration<BenchmarkSystemRole> {
  if (options === null || typeof options !== "object") {
    configurationError("options", "must be an object");
  }
  const raw = options as Record<string, unknown>;
  const dataset = validateRefundDataset(raw["dataset"]);
  const warmupIterations = raw["warmupIterations"];
  if (
    !Number.isSafeInteger(warmupIterations) ||
    (warmupIterations as number) < 0 ||
    (warmupIterations as number) > MAXIMUM_WARMUP_ITERATIONS
  ) {
    configurationError(
      "warmupIterations",
      `must be a non-negative safe integer no greater than ${String(MAXIMUM_WARMUP_ITERATIONS)}`,
    );
  }
  const adapterValue = raw["adapter"];
  if (
    adapterValue === null ||
    typeof adapterValue !== "object" ||
    !("predict" in adapterValue) ||
    typeof adapterValue.predict !== "function" ||
    !("resolveExecutionBackend" in adapterValue) ||
    typeof adapterValue.resolveExecutionBackend !== "function"
  ) {
    configurationError(
      "adapter",
      "must provide predict and resolveExecutionBackend functions",
    );
  }
  const adapterRecord = adapterValue as Record<string, unknown>;
  const role = adapterRecord["role"];
  if (typeof role !== "string" || !supportedRoles.has(role as BenchmarkSystemRole)) {
    configurationError("adapter.role", "must be a supported benchmark system role");
  }
  const predictionAdapter = adapterValue as RefundPredictionAdapter;
  const system = validateAndFreezeSystem(role as BenchmarkSystemRole, adapterRecord);
  const environment = validateAndFreezeEnvironment(raw["environment"]);

  const warmupInputs = validateWarmupInputs(
    raw["warmupInputs"],
    warmupIterations as number,
    dataset,
  );
  const warmupInputOrderSha256 = semanticJsonSha256(warmupInputs);

  const memorySamplerValue = raw["memorySampler"];
  if (
    memorySamplerValue === null ||
    typeof memorySamplerValue !== "object" ||
    !("start" in memorySamplerValue) ||
    typeof memorySamplerValue.start !== "function" ||
    !("stop" in memorySamplerValue) ||
    typeof memorySamplerValue.stop !== "function"
  ) {
    configurationError("memorySampler", "must provide start and stop functions");
  }
  const memorySamplerRecord = memorySamplerValue as Record<string, unknown>;
  if (
    memorySamplerRecord["scope"] !== "client-only" &&
    memorySamplerRecord["scope"] !== "process-tree"
  ) {
    configurationError(
      "memorySampler.scope",
      'must be "client-only" or "process-tree"',
    );
  }
  const memorySampler = memorySamplerValue as PeakMemorySampler;
  const configuredClock = raw["clock"];
  const clockValue = configuredClock === undefined ? defaultClock : configuredClock;
  if (
    clockValue === null ||
    typeof clockValue !== "object" ||
    !("now" in clockValue) ||
    typeof clockValue.now !== "function"
  ) {
    configurationError("clock", "must provide a now function");
  }
  const clock = clockValue as MonotonicClock;
  const signal = raw["signal"];
  if (signal !== undefined && !isAbortSignal(signal)) {
    configurationError("signal", "must be an AbortSignal");
  }

  return Object.freeze({
    dataset,
    adapter: predictionAdapter,
    role: role as BenchmarkSystemRole,
    model: system.model,
    adapterProvenance: system.adapter,
    taskSpecSha256: system.taskSpecSha256,
    trainingEvidence: system.trainingEvidence,
    environment,
    warmupIterations: warmupIterations as number,
    warmupInputs,
    warmupInputOrderSha256,
    memorySampler,
    clock,
    callerSignal: signal,
  });
}

function validateAndFreezeSystem(
  role: BenchmarkSystemRole,
  adapter: Record<string, unknown>,
): SystemProvenance {
  try {
    const system = {
      role,
      taskSpecSha256: adapter["taskSpecSha256"],
      model: adapter["model"],
      adapter: adapter["adapter"],
      trainingEvidence: adapter["trainingEvidence"],
      executionBackend: placeholderExecutionBackend(role),
    };
    contractInternals.validateSystemProvenance(system, "adapter");
    return contractInternals.frozenClone(system) as SystemProvenance;
  } catch (error) {
    configurationError("adapter", errorMessage(error), error);
  }
}

function placeholderExecutionBackend(
  role: BenchmarkSystemRole,
): ExecutionBackendProvenance {
  switch (role) {
    case "semantscript":
      return { kind: "semantscript-node", runtime: "onnxruntime-node", device: "cpu" };
    case "ollama-1b":
    case "ollama-7b":
      return {
        kind: "ollama",
        serverVersion: "configuration-validation-only",
        placement: "cpu",
        modelTotalBytes: 1,
        modelCpuBytes: 1,
        modelGpuBytes: 0,
      };
    case "structured-api":
      return {
        kind: "anthropic-api",
        apiVersion: "2023-06-01",
        endpoint: "https://api.anthropic.com",
        placement: "provider-managed",
      };
    case "laya":
      return { kind: "laya", runtime: "python", device: "cpu" };
  }
}

function validateWarmupInputs(
  value: unknown,
  iterations: number,
  dataset: RefundBenchmarkDatasetV1,
): readonly RefundInputs[] {
  let values: readonly unknown[];
  try {
    values = contractInternals.denseArray(value, "warmupInputs", 20_000);
  } catch (error) {
    configurationError("warmupInputs", errorMessage(error), error);
  }
  if ((iterations === 0) !== (values.length === 0)) {
    configurationError(
      "warmupInputs",
      "must be empty exactly when warmupIterations is zero",
    );
  }
  const finalDigests = new Set(dataset.cases.map(({ inputSha256 }) => inputSha256));
  const seenDigests = new Set<string>();
  const inputs = values.map((input, index) => {
    const path = `warmupInputs[${String(index)}]`;
    try {
      contractInternals.validateRefundInputs(input, path);
    } catch (error) {
      configurationError(path, errorMessage(error), error);
    }
    const digest = semanticJsonSha256(input);
    if (finalDigests.has(digest)) {
      configurationError(path, "must not duplicate an exact final evaluation input");
    }
    if (seenDigests.has(digest)) {
      configurationError(path, "duplicates another warmup input");
    }
    seenDigests.add(digest);
    return input as RefundInputs;
  });
  return contractInternals.frozenClone(inputs);
}

function validateAndFreezeEnvironment(
  value: unknown,
): EnvironmentProvenance {
  try {
    contractInternals.validateEnvironment(value, "environment");
    return contractInternals.frozenClone(value) as EnvironmentProvenance;
  } catch (error) {
    configurationError("environment", errorMessage(error), error);
  }
}

function validateAdapterPrediction(value: unknown): BaselinePrediction {
  try {
    const prediction = exactObject(value, "prediction", ["value", "distribution"]);
    if (!isRefundDecision(prediction.value)) {
      outputError("prediction.value", "must be a canonical refund decision");
    }
    const distribution = denseArray(prediction.distribution, "prediction.distribution");
    if (distribution.length !== REFUND_SUPPORT.length) {
      outputError(
        "prediction.distribution",
        `must contain exactly ${String(REFUND_SUPPORT.length)} entries`,
      );
    }
    let total = 0;
    let bestIndex = 0;
    let bestProbability = Number.NEGATIVE_INFINITY;
    const validatedDistribution = REFUND_SUPPORT.map((supportValue, index) => {
      const entry = exactObject(distribution[index], `prediction.distribution[${String(index)}]`, [
        "value",
        "probability",
      ]);
      if (entry.value !== supportValue) {
        outputError(
          `prediction.distribution[${String(index)}].value`,
          `must be ${JSON.stringify(supportValue)}`,
        );
      }
      if (
        typeof entry.probability !== "number" ||
        !Number.isFinite(entry.probability) ||
        entry.probability < 0 ||
        entry.probability > 1
      ) {
        outputError(
          `prediction.distribution[${String(index)}].probability`,
          "must be a finite probability from 0 through 1",
        );
      }
      total += entry.probability;
      if (entry.probability > bestProbability) {
        bestProbability = entry.probability;
        bestIndex = index;
      }
      return Object.freeze({ value: supportValue, probability: entry.probability });
    });
    if (Math.abs(total - 1) > 1e-12) {
      outputError("prediction.distribution", "probabilities must sum to 1 within 1e-12");
    }
    if (prediction.value !== REFUND_SUPPORT[bestIndex]) {
      outputError(
        "prediction.value",
        "must be the stable support-order argmax of distribution",
      );
    }
    return Object.freeze({
      value: prediction.value,
      distribution: Object.freeze(validatedDistribution),
    });
  } catch (error) {
    if (error instanceof BenchmarkRunnerError) {
      throw error;
    }
    throw new BenchmarkRunnerError(
      "invalid-output",
      `adapter returned an invalid prediction: ${errorMessage(error)}`,
      { cause: error },
    );
  }
}

function exactObject<const Keys extends readonly string[]>(
  value: unknown,
  path: string,
  keys: Keys,
): { [Key in Keys[number]]: unknown } {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    outputError(path, "must be an object");
  }
  const prototype = Object.getPrototypeOf(value) as unknown;
  if (prototype !== Object.prototype && prototype !== null) {
    outputError(path, "must be a plain object");
  }
  if (Object.getOwnPropertySymbols(value).length > 0) {
    outputError(path, "must not contain symbol properties");
  }
  const names = Object.getOwnPropertyNames(value);
  if (names.length !== keys.length || keys.some((key) => !names.includes(key))) {
    outputError(path, `must contain exactly: ${keys.join(", ")}`);
  }
  for (const key of keys) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor?.enumerable || !("value" in descriptor)) {
      outputError(`${path}.${key}`, "must be an enumerable data property");
    }
  }
  return value as { [Key in Keys[number]]: unknown };
}

function denseArray(value: unknown, path: string): readonly unknown[] {
  if (!Array.isArray(value)) {
    outputError(path, "must be an array");
  }
  const names = Object.getOwnPropertyNames(value).filter((name) => name !== "length");
  if (
    Object.getOwnPropertySymbols(value).length > 0 ||
    names.length !== value.length ||
    names.some((name, index) => name !== String(index))
  ) {
    outputError(path, "must be dense and contain no extra properties");
  }
  for (let index = 0; index < value.length; index += 1) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor?.enumerable || !("value" in descriptor)) {
      outputError(`${path}[${String(index)}]`, "must be an enumerable data property");
    }
  }
  return value;
}

function validatePeakMemory(value: unknown): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new BenchmarkRunnerError(
      "invalid-memory-sample",
      "memorySampler.stop() must return a non-negative safe integer byte count",
    );
  }
  return value as number;
}

function throwIfAborted(signal: AbortSignal | undefined, message: string): void {
  if (signal?.aborted === true) {
    throw new BenchmarkRunnerError("aborted", message, {
      cause: signal.reason,
    });
  }
}

function isAbortSignal(value: unknown): value is AbortSignal {
  const candidate = value as Record<string, unknown>;
  return (
    value !== null &&
    typeof value === "object" &&
    "aborted" in candidate &&
    typeof candidate["addEventListener"] === "function" &&
    typeof candidate["removeEventListener"] === "function"
  );
}

function isRefundDecision(value: unknown): value is (typeof REFUND_SUPPORT)[number] {
  return value === "approve" || value === "deny" || value === "review";
}

function outputError(path: string, message: string): never {
  throw new BenchmarkRunnerError("invalid-output", `${path}: ${message}`);
}

function configurationError(path: string, message: string, cause?: unknown): never {
  throw new BenchmarkRunnerError(
    "invalid-configuration",
    `${path}: ${message}`,
    cause === undefined ? undefined : { cause },
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
