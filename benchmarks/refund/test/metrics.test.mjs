import assert from "node:assert/strict";
import test from "node:test";

import { semanticJsonSha256 } from "@semantscript/compiler";

import {
  createBenchmarkResult,
  evaluatePredictionSet,
  validateBenchmarkBundle,
  validateBenchmarkResult,
} from "../dist/index.js";
import {
  clone,
  makeAllPredictionSets,
  makeDataset,
  makeLedger,
  makePredictionSet,
} from "./fixtures.mjs";

test("computes exact overall/human accuracy, 15-bin ECE, nearest-rank latency, and throughput", () => {
  const dataset = makeDataset();
  const metrics = evaluatePredictionSet(dataset, makePredictionSet(dataset));

  assert.deepEqual(metrics.accuracy, {
    caseCount: 4,
    correctCount: 2,
    accuracy: 0.5,
    attestedCaseCount: 2,
    attestedCorrectCount: 2,
    attestedAccuracy: 1,
  });
  assert.equal(metrics.calibration.binCount, 15);
  assert.ok(Math.abs(metrics.calibration.expectedCalibrationError - 0.35) < 1e-12);
  assert.deepEqual(metrics.latency, { p50Ms: 4, p95Ms: 20 });
  assert.deepEqual(metrics.throughput, {
    concurrency: 1,
    measuredDurationMs: 200,
    requestsPerSecond: 20,
  });
  assert.deepEqual(metrics.memory, { scope: "process-tree", peakBytes: 4096 });
});

test("coverage is exact and binds every prediction to dataset input identity", () => {
  const dataset = makeDataset();
  const missing = clone(makePredictionSet(dataset));
  missing.predictions.pop();
  missing.payloadSha256 = semanticJsonSha256(withoutDigest(missing));
  assert.throws(() => evaluatePredictionSet(dataset, missing), /exactly one prediction/);

  const wrongInput = clone(makePredictionSet(dataset));
  wrongInput.predictions[0].inputSha256 = "f".repeat(64);
  wrongInput.payloadSha256 = semanticJsonSha256(withoutDigest(wrongInput));
  assert.throws(() => evaluatePredictionSet(dataset, wrongInput), /input digest in dataset order/);
});

test("go/no-go remains incomplete until every required role is present", () => {
  const dataset = makeDataset();
  const result = createBenchmarkResult(
    dataset,
    makeLedger(),
    [makePredictionSet(dataset)],
    "2026-09-23T13:00:00Z",
  );

  assert.equal(result.goNoGo.status, "incomplete");
  assert.deepEqual(result.goNoGo.missingSystems, [
    "ollama-1b",
    "ollama-7b",
    "structured-api",
    "laya",
  ]);
  assert.equal(result.goNoGo.latencyPassed, null);
  assert.equal(result.goNoGo.accuracyPassed, null);
});

test("go/no-go uses strict sub-10ms p50 and accuracy at least the 7B baseline", () => {
  const dataset = makeDataset();
  const ledger = makeLedger();
  const go = createBenchmarkResult(
    dataset,
    ledger,
    makeAllPredictionSets(dataset),
    "2026-09-23T13:00:00Z",
  );
  assert.equal(go.goNoGo.status, "go");
  assert.equal(go.goNoGo.latencyPassed, true);
  assert.equal(go.goNoGo.accuracyPassed, true);

  const noGo = createBenchmarkResult(
    dataset,
    ledger,
    makeAllPredictionSets(dataset, {
      semantscript: { latencies: [10, 10, 11, 12] },
    }),
    "2026-09-23T13:00:00Z",
  );
  assert.equal(noGo.goNoGo.status, "no-go");
  assert.equal(noGo.goNoGo.semantscriptP50Ms, 10);
  assert.equal(noGo.goNoGo.latencyPassed, false);
});

test("result is role-ordered, provenance-complete, closed, and digest-bound", () => {
  const dataset = makeDataset();
  const predictions = makeAllPredictionSets(dataset).reverse();
  const result = createBenchmarkResult(
    dataset,
    makeLedger(),
    predictions,
    "2026-09-23T13:00:00.123Z",
  );
  assert.deepEqual(
    result.systems.map((system) => system.role),
    ["semantscript", "ollama-1b", "ollama-7b", "structured-api", "laya"],
  );
  assert.ok(result.systems.every((system) => system.model.artifactSha256.length === 64));
  assert.ok(result.systems.every((system) => system.environment.evidenceSha256.length === 64));
  assert.ok(result.systems.every((system) => system.protocol.warmupInputOrderSha256.length === 64));
  assert.equal(result.systems[0].executionBackend.kind, "semantscript-node");
  assert.equal(result.systems[1].executionBackend.kind, "ollama");
  assert.equal(result.systems[3].executionBackend.kind, "anthropic-api");
  assert.equal(result.systems[4].executionBackend.kind, "laya");
  assert.equal(result.systems[0].trainingEvidence.trainingLedgerSha256, result.trainingLedgerSha256);
  assert.equal(result.systems.slice(1).every(({ trainingEvidence }) => trainingEvidence === null), true);
  assert.equal(validateBenchmarkResult(result).payloadSha256, result.payloadSha256);

  const tampered = clone(result);
  tampered.systems[0].metrics.accuracy.correctCount = 4;
  assert.throws(() => validateBenchmarkResult(tampered), /accuracy.*must equal/);

  const rewritten = clone(result);
  rewritten.goNoGo.status = "no-go";
  rewritten.payloadSha256 = semanticJsonSha256(withoutDigest(rewritten));
  assert.throws(() => validateBenchmarkResult(rewritten), /mechanical exit criterion/);

  const backendMutation = clone(result);
  delete backendMutation.systems[1].executionBackend.serverVersion;
  backendMutation.payloadSha256 = semanticJsonSha256(withoutDigest(backendMutation));
  assert.throws(
    () => validateBenchmarkResult(backendMutation),
    /executionBackend.*must contain exactly/,
  );
});

test("publication requires identical hardware and comparable warmup/memory protocol", () => {
  const dataset = makeDataset();
  const ledger = makeLedger();
  const wrongHardware = makeAllPredictionSets(dataset, {
    laya: { environment: { cpu: "different-cpu" } },
  });
  assert.throws(
    () => createBenchmarkResult(dataset, ledger, wrongHardware, "2026-09-23T13:00:00Z"),
    /hardware identity must match/,
  );

  const wrongWarmup = makeAllPredictionSets(dataset, {
    laya: { warmupInputOrderSha256: "f".repeat(64) },
  });
  assert.throws(
    () => createBenchmarkResult(dataset, ledger, wrongWarmup, "2026-09-23T13:00:00Z"),
    /warmup iterations\/input digest/,
  );

  const allowedDifferences = makeAllPredictionSets(dataset, {
    laya: {
      measuredDurationMs: 123,
      peakMemoryBytes: 999,
      environment: {
        capturedAt: "2026-09-23T12:31:00Z",
        runtimeVersions: [{ name: "python", version: "different" }],
        evidenceSha256: semanticJsonSha256("different-runtime-evidence"),
      },
    },
  });
  assert.doesNotThrow(() =>
    createBenchmarkResult(dataset, ledger, allowedDifferences, "2026-09-23T13:00:00Z"),
  );
});

test("SemantScript publication evidence must exactly bind the supplied ledger", () => {
  const dataset = makeDataset();
  const ledger = makeLedger();
  const prediction = makePredictionSet(dataset, "semantscript", { ledger });
  const differentLedger = sealDifferentLedger(ledger);
  assert.throws(
    () =>
      createBenchmarkResult(
        dataset,
        differentLedger,
        [prediction],
        "2026-09-23T13:00:00Z",
      ),
    /exactly bind the supplied training ledger/,
  );
});

test("SemantScript publication binds the artifact training dataset to the ledger base dataset", () => {
  const dataset = makeDataset();
  const ledger = makeLedger();
  const prediction = clone(makePredictionSet(dataset, "semantscript", { ledger }));
  prediction.system.trainingEvidence.artifactTrainingDatasetSha256 = "f".repeat(64);
  prediction.payloadSha256 = semanticJsonSha256(withoutDigest(prediction));

  assert.throws(
    () =>
      createBenchmarkResult(
        dataset,
        ledger,
        [prediction],
        "2026-09-23T13:00:00Z",
      ),
    /artifact training dataset/,
  );
});

test("rehashed ledger and evidence cannot substitute release verification", () => {
  const dataset = makeDataset();
  const originalLedger = makeLedger();
  const prediction = clone(
    makePredictionSet(dataset, "semantscript", { ledger: originalLedger }),
  );
  const substitutedLedger = clone(originalLedger);
  substitutedLedger.sources.releaseVerificationPayloadSha256 = "f".repeat(64);
  substitutedLedger.payloadSha256 = semanticJsonSha256(
    withoutDigest(substitutedLedger),
  );
  prediction.system.trainingEvidence.trainingLedgerSha256 =
    substitutedLedger.payloadSha256;
  prediction.system.trainingEvidence.releaseVerificationPayloadSha256 =
    substitutedLedger.sources.releaseVerificationPayloadSha256;
  prediction.payloadSha256 = semanticJsonSha256(withoutDigest(prediction));

  assert.throws(
    () =>
      createBenchmarkResult(
        dataset,
        substitutedLedger,
        [prediction],
        "2026-09-23T13:00:00Z",
      ),
    /artifact training dataset\/key/,
  );
});

test("bundle validation recomputes the result from exact prediction files", () => {
  const dataset = makeDataset();
  const ledger = makeLedger();
  const predictions = makeAllPredictionSets(dataset);
  const result = createBenchmarkResult(
    dataset,
    ledger,
    predictions,
    "2026-09-23T13:00:00Z",
  );

  assert.equal(
    validateBenchmarkBundle(dataset, ledger, predictions, result).payloadSha256,
    result.payloadSha256,
  );

  const differentPredictions = [...predictions];
  differentPredictions[4] = makePredictionSet(dataset, "laya", {
    latencies: [9, 1, 20, 5],
  });
  assert.throws(
    () => validateBenchmarkBundle(dataset, ledger, differentPredictions, result),
    /recomputed from the supplied benchmark bundle/,
  );
});

test("result builder rejects duplicate roles and training leakage", () => {
  const dataset = makeDataset();
  const prediction = makePredictionSet(dataset);
  assert.throws(
    () =>
      createBenchmarkResult(
        dataset,
        makeLedger(),
        [prediction, prediction],
        "2026-09-23T13:00:00Z",
      ),
    /duplicate system role/,
  );
  const leaking = makeLedger({
    adversarial: { inputSha256s: [dataset.cases[2].inputSha256] },
  });
  assert.throws(
    () => createBenchmarkResult(dataset, leaking, [prediction], "2026-09-23T13:00:00Z"),
    /overlap the training lifecycle ledger/,
  );
});

test("client-only memory remains distinguishable from process-tree memory", () => {
  const dataset = makeDataset();
  const metrics = evaluatePredictionSet(
    dataset,
    makePredictionSet(dataset, "structured-api", {
      memoryScope: "client-only",
      peakMemoryBytes: 1234,
    }),
  );
  assert.deepEqual(metrics.memory, { scope: "client-only", peakBytes: 1234 });
});

function withoutDigest(value) {
  const payload = { ...value };
  delete payload.payloadSha256;
  return payload;
}

function sealDifferentLedger(ledger) {
  const value = clone(ledger);
  value.sources.baseDatasetSha256 = "e".repeat(64);
  value.payloadSha256 = semanticJsonSha256(withoutDigest(value));
  return value;
}
