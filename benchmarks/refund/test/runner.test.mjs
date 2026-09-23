import assert from "node:assert/strict";
import test from "node:test";

import { semanticJsonSha256 } from "@semantscript/compiler";

import {
  BenchmarkRunnerError,
  MAXIMUM_WARMUP_ITERATIONS,
  runRefundBenchmark,
} from "../dist/runner.js";
import {
  REFUND_SYSTEM_PINS,
  REFUND_TASK_SPEC_SHA256,
} from "../dist/policy.js";
import { makeDataset } from "./fixtures.mjs";

const environment = Object.freeze({
  capturedAt: "2026-09-23T13:00:00Z",
  operatingSystem: "runner-test-os",
  architecture: "runner-test-arch",
  cpu: "runner-test-cpu",
  accelerator: null,
  memoryBytes: 16_384,
  runtimeVersions: Object.freeze([
    Object.freeze({ name: "node", version: "test-only" }),
  ]),
  evidenceSha256: semanticJsonSha256("runner-test-environment"),
});

const model = Object.freeze({
  provider: REFUND_SYSTEM_PINS.semantscript.model.provider,
  name: REFUND_SYSTEM_PINS.semantscript.model.name,
  version: REFUND_SYSTEM_PINS.semantscript.model.version,
  revision: semanticJsonSha256("runner-model-revision"),
  artifactSha256: semanticJsonSha256("runner-model-artifact"),
});

const adapterProvenance = Object.freeze({
  name: REFUND_SYSTEM_PINS.semantscript.adapter.name,
  version: REFUND_SYSTEM_PINS.semantscript.adapter.version,
  configurationSha256: semanticJsonSha256("runner-adapter-configuration"),
});

const trainingEvidence = Object.freeze({
  trainingLedgerSha256: semanticJsonSha256("runner-ledger"),
  artifactTrainingDatasetSha256: semanticJsonSha256("runner-training-dataset"),
  artifactTrainingKeySha256: model.revision,
  releaseVerificationPayloadSha256: semanticJsonSha256("runner-release"),
  releaseVerificationAttestationSha256: semanticJsonSha256("runner-attestation"),
});

const executionBackend = Object.freeze({
  kind: "semantscript-node",
  runtime: "onnxruntime-node",
  device: "cpu",
});

const warmupInputs = Object.freeze([
  Object.freeze({
    customer: Object.freeze({ priorRefunds: 9, tier: "standard" }),
    order: Object.freeze({ ageDays: 3, status: "paid", total: 12 }),
  }),
  Object.freeze({
    customer: Object.freeze({ priorRefunds: 10, tier: "enterprise" }),
    order: Object.freeze({ ageDays: 4, status: "paid", total: 13 }),
  }),
]);

const validPrediction = Object.freeze({
  value: "approve",
  distribution: Object.freeze([
    Object.freeze({ value: "approve", probability: 0.8 }),
    Object.freeze({ value: "deny", probability: 0.1 }),
    Object.freeze({ value: "review", probability: 0.1 }),
  ]),
});

function clock(readings) {
  let index = 0;
  return {
    now() {
      const reading = readings[index];
      assert.notEqual(reading, undefined, "test clock was read more often than expected");
      index += 1;
      return reading;
    },
    calls() {
      return index;
    },
  };
}

function sampler(scope = "process-tree", peak = 9876) {
  const events = [];
  return {
    scope,
    events,
    async start(signal) {
      events.push({ event: "start", signal });
    },
    async stop() {
      events.push({ event: "stop" });
      return peak;
    },
  };
}

function adapter(predict, role = "semantscript") {
  return Object.freeze({
    role,
    taskSpecSha256: REFUND_TASK_SPEC_SHA256,
    model,
    adapter: adapterProvenance,
    trainingEvidence,
    async resolveExecutionBackend() {
      return executionBackend;
    },
    predict,
  });
}

function options(overrides = {}) {
  return {
    dataset: makeDataset(),
    adapter: adapter(async () => validPrediction),
    environment,
    warmupIterations: 0,
    warmupInputs: [],
    memorySampler: sampler(),
    clock: clock([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]),
    ...overrides,
  };
}

function runnerError(code) {
  return (error) => {
    assert.ok(error instanceof BenchmarkRunnerError);
    assert.equal(error.code, code);
    return true;
  };
}

test("warms deterministically then measures every frozen input once in dataset order", async () => {
  const dataset = makeDataset();
  const calls = [];
  const runClock = clock([100, 101, 104, 105, 110, 111, 112, 114, 121, 125]);
  const memorySampler = sampler("process-tree", 98_765);
  const predictionAdapter = adapter(async (inputs, signal) => {
    calls.push({ inputs, signal });
    assert.equal(Object.isFrozen(inputs), true);
    assert.equal(Object.isFrozen(inputs.customer), true);
    assert.equal(Object.isFrozen(inputs.order), true);
    return validPrediction;
  });

  const result = await runRefundBenchmark({
    dataset,
    adapter: predictionAdapter,
    environment,
    warmupIterations: 3,
    warmupInputs,
    memorySampler,
    clock: runClock,
  });

  assert.deepEqual(
    calls.map(({ inputs }) => inputs),
    [
      warmupInputs[0],
      warmupInputs[1],
      warmupInputs[0],
      ...dataset.cases.map(({ inputs }) => inputs),
    ],
  );
  assert.equal(new Set(calls.map(({ signal }) => signal)).size, 1);
  assert.equal(calls.every(({ signal }) => signal.aborted === false), true);
  assert.equal(memorySampler.events.length, 2);
  assert.equal(memorySampler.events[0].event, "start");
  assert.equal(memorySampler.events[0].signal, calls[0].signal);
  assert.equal(memorySampler.events[1].event, "stop");
  assert.equal(runClock.calls(), 10);

  assert.equal(result.datasetSha256, dataset.payloadSha256);
  assert.deepEqual(result.system, {
    role: "semantscript",
    taskSpecSha256: REFUND_TASK_SPEC_SHA256,
    model,
    adapter: adapterProvenance,
    trainingEvidence,
    executionBackend,
  });
  assert.deepEqual(result.environment, environment);
  assert.deepEqual(result.protocol, {
    concurrency: 1,
    warmupIterations: 3,
    warmupInputOrderSha256: semanticJsonSha256(warmupInputs),
    measuredDurationMs: 25,
    memoryScope: "process-tree",
    peakMemoryBytes: 98_765,
  });
  assert.deepEqual(
    result.predictions.map(({ caseId, inputSha256, latencyMs }) => ({
      caseId,
      inputSha256,
      latencyMs,
    })),
    dataset.cases.map(({ id, inputSha256 }, index) => ({
      caseId: id,
      inputSha256,
      latencyMs: [3, 5, 1, 7][index],
    })),
  );
  assert.equal(Object.isFrozen(result), true);
  assert.equal(Object.isFrozen(result.predictions), true);
  assert.match(result.payloadSha256, /^[a-f0-9]{64}$/);
});

test("records client-only memory scope without changing execution semantics", async () => {
  const memorySampler = sampler("client-only", 1234);
  const result = await runRefundBenchmark(options({ memorySampler }));
  assert.equal(result.protocol.memoryScope, "client-only");
  assert.equal(result.protocol.peakMemoryBytes, 1234);
});

test("adapter rejection aborts the shared signal, stops sampling, and never skips ahead", async () => {
  const failure = new Error("intentional adapter failure");
  const calls = [];
  const memorySampler = sampler();
  const predictionAdapter = adapter(async (_inputs, signal) => {
    calls.push(signal);
    if (calls.length === 2) {
      throw failure;
    }
    return validPrediction;
  });

  await assert.rejects(
    runRefundBenchmark(
      options({
        adapter: predictionAdapter,
        memorySampler,
        clock: clock([0, 0, 2, 3]),
      }),
    ),
    (error) => error === failure,
  );
  assert.equal(calls.length, 2);
  assert.equal(calls[0], calls[1]);
  assert.equal(calls[0].aborted, true);
  assert.deepEqual(
    memorySampler.events.map(({ event }) => event),
    ["start", "stop"],
  );
});

test("invalid output fails fast instead of being normalized, repaired, or skipped", async () => {
  let calls = 0;
  const memorySampler = sampler();
  const predictionAdapter = adapter(async () => {
    calls += 1;
    return {
      value: "approve",
      distribution: [
        { value: "approve", probability: 8 },
        { value: "deny", probability: 1 },
        { value: "review", probability: 1 },
      ],
    };
  });
  await assert.rejects(
    runRefundBenchmark(
      options({ adapter: predictionAdapter, memorySampler, clock: clock([0, 0]) }),
    ),
    runnerError("invalid-output"),
  );
  assert.equal(calls, 1);
  assert.equal(memorySampler.events.at(-1).event, "stop");
});

test("caller abort stops after the active call even if the adapter returns a value", async () => {
  const caller = new globalThis.AbortController();
  const reason = new Error("caller cancelled");
  const signals = [];
  const memorySampler = sampler();
  const predictionAdapter = adapter(async (_inputs, signal) => {
    signals.push(signal);
    caller.abort(reason);
    return validPrediction;
  });
  await assert.rejects(
    runRefundBenchmark(
      options({
        adapter: predictionAdapter,
        memorySampler,
        signal: caller.signal,
        clock: clock([0, 0]),
      }),
    ),
    runnerError("aborted"),
  );
  assert.equal(signals.length, 1);
  assert.equal(signals[0].aborted, true);
  assert.equal(signals[0].reason, reason);
  assert.equal(memorySampler.events.at(-1).event, "stop");
});

test("an already-aborted caller performs no warmup, measurement, or sampler work", async () => {
  const caller = new globalThis.AbortController();
  caller.abort("already cancelled");
  let calls = 0;
  const memorySampler = sampler();
  await assert.rejects(
    runRefundBenchmark(
      options({
        adapter: adapter(async () => {
          calls += 1;
          return validPrediction;
        }),
        memorySampler,
        signal: caller.signal,
      }),
    ),
    runnerError("aborted"),
  );
  assert.equal(calls, 0);
  assert.deepEqual(memorySampler.events, []);
});

test("warmup failures stop immediately and do not start measured memory sampling", async () => {
  const failure = new Error("warmup failed");
  let calls = 0;
  const memorySampler = sampler();
  await assert.rejects(
    runRefundBenchmark(
      options({
        warmupIterations: 4,
        warmupInputs,
        memorySampler,
        adapter: adapter(async () => {
          calls += 1;
          if (calls === 2) {
            throw failure;
          }
          return validPrediction;
        }),
      }),
    ),
    (error) => error === failure,
  );
  assert.equal(calls, 2);
  assert.deepEqual(memorySampler.events, []);
});

test("invalid warmup output is a failure and cannot be silently discarded", async () => {
  let calls = 0;
  const memorySampler = sampler();
  await assert.rejects(
    runRefundBenchmark(
      options({
        warmupIterations: 2,
        warmupInputs,
        memorySampler,
        adapter: adapter(async () => {
          calls += 1;
          return {
            ...validPrediction,
            unexpected: "must not be ignored",
          };
        }),
      }),
    ),
    runnerError("invalid-output"),
  );
  assert.equal(calls, 1);
  assert.deepEqual(memorySampler.events, []);
});

test("configuration validation rejects invalid bounds and protocol dependencies before calls", async () => {
  let calls = 0;
  const predictionAdapter = adapter(async () => {
    calls += 1;
    return validPrediction;
  });
  const invalidConfigurations = [
    { warmupIterations: -1 },
    { warmupIterations: 0.5 },
    { warmupIterations: MAXIMUM_WARMUP_ITERATIONS + 1 },
    { warmupIterations: 1, warmupInputs: [] },
    { warmupIterations: 0, warmupInputs },
    { memorySampler: { scope: "host", start() {}, stop() { return 1; } } },
    { memorySampler: { scope: "client-only", start() {} } },
    { clock: { now: 1 } },
    { adapter: { ...predictionAdapter, role: "unknown" } },
    { signal: {} },
  ];
  for (const invalid of invalidConfigurations) {
    await assert.rejects(
      runRefundBenchmark(options({ adapter: predictionAdapter, ...invalid })),
      runnerError("invalid-configuration"),
    );
  }
  assert.equal(calls, 0);
});

test("warmup corpus is validated, unique, and disjoint from final evaluation inputs", async () => {
  const dataset = makeDataset();
  const invalidWarmups = [
    [dataset.cases[0].inputs],
    [warmupInputs[0], warmupInputs[0]],
    [{ ...warmupInputs[0], extra: true }],
  ];
  for (const inputs of invalidWarmups) {
    await assert.rejects(
      runRefundBenchmark(
        options({
          dataset,
          warmupIterations: 1,
          warmupInputs: inputs,
        }),
      ),
      runnerError("invalid-configuration"),
    );
  }
});

test("non-monotonic clock and invalid peak samples reject without a partial result", async () => {
  const clockSampler = sampler();
  await assert.rejects(
    runRefundBenchmark(options({ memorySampler: clockSampler, clock: clock([5, 4]) })),
    runnerError("non-monotonic-clock"),
  );
  assert.equal(clockSampler.events.at(-1).event, "stop");

  const memorySampler = sampler("process-tree", -1);
  await assert.rejects(
    runRefundBenchmark(options({ memorySampler })),
    runnerError("invalid-memory-sample"),
  );
  assert.deepEqual(
    memorySampler.events.map(({ event }) => event),
    ["start", "stop"],
  );
});

test("sampler startup failure aborts the run and attempts lifecycle cleanup", async () => {
  const failure = new Error("sampler could not start");
  let stopCalls = 0;
  let capturedSignal;
  const memorySampler = {
    scope: "process-tree",
    async start(signal) {
      capturedSignal = signal;
      throw failure;
    },
    async stop() {
      stopCalls += 1;
      return 0;
    },
  };
  await assert.rejects(
    runRefundBenchmark(options({ memorySampler })),
    (error) => error === failure,
  );
  assert.equal(capturedSignal.aborted, true);
  assert.equal(stopCalls, 1);
});
