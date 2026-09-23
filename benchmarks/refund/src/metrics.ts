import { semanticJsonSha256 } from "@semantscript/compiler";

import {
  BenchmarkContractError,
  auditDatasetSeparation,
  contractInternals,
  validatePredictionSet,
  validateRefundDataset,
  validateTrainingLedger,
} from "./contracts.js";
import {
  REFUND_TASK_SPEC_SHA256,
  deriveRefundArtifactTrainingKeySha256,
} from "./policy.js";
import {
  ATTESTED_CASE_ORIGINS,
  REQUIRED_SYSTEM_ROLES,
  type BenchmarkSystemResult,
  type BenchmarkSystemRole,
  type GoNoGoDecision,
  type RefundBenchmarkDatasetV1,
  type RefundBenchmarkResultV1,
  type RefundPredictionSetV1,
  type SystemMetrics,
} from "./types.js";

const ECE_BIN_COUNT = 15 as const;

export function evaluatePredictionSet(
  datasetValue: unknown,
  predictionValue: unknown,
): SystemMetrics {
  const dataset = validateRefundDataset(datasetValue);
  const predictionSet = validatePredictionSet(predictionValue);
  assertPredictionCoverage(dataset, predictionSet);

  const casesById = new Map(dataset.cases.map((entry) => [entry.id, entry]));
  let correctCount = 0;
  let attestedCorrectCount = 0;
  const attestedCaseCount = dataset.cases.filter((entry) =>
    ATTESTED_CASE_ORIGINS.has(entry.origin),
  ).length;
  const binCounts = Array<number>(ECE_BIN_COUNT).fill(0);
  const binCorrect = Array<number>(ECE_BIN_COUNT).fill(0);
  const binConfidence = Array<number>(ECE_BIN_COUNT).fill(0);

  for (const prediction of predictionSet.predictions) {
    const benchmarkCase = casesById.get(prediction.caseId);
    if (!benchmarkCase) {
      throw new BenchmarkContractError(
        "$.predictions",
        `prediction references unknown case ${JSON.stringify(prediction.caseId)}`,
      );
    }
    const correct = prediction.value === benchmarkCase.expected;
    if (correct) {
      correctCount += 1;
      if (ATTESTED_CASE_ORIGINS.has(benchmarkCase.origin)) {
        attestedCorrectCount += 1;
      }
    }
    const confidence = Math.max(...prediction.distribution.map((entry) => entry.probability));
    const bin = Math.min(ECE_BIN_COUNT - 1, Math.floor(confidence * ECE_BIN_COUNT));
    binCounts[bin] = (binCounts[bin] ?? 0) + 1;
    binCorrect[bin] = (binCorrect[bin] ?? 0) + (correct ? 1 : 0);
    binConfidence[bin] = (binConfidence[bin] ?? 0) + confidence;
  }

  let expectedCalibrationError = 0;
  for (let bin = 0; bin < ECE_BIN_COUNT; bin += 1) {
    const count = binCounts[bin] ?? 0;
    if (count === 0) {
      continue;
    }
    const accuracy = (binCorrect[bin] ?? 0) / count;
    const meanConfidence = (binConfidence[bin] ?? 0) / count;
    expectedCalibrationError +=
      (count / predictionSet.predictions.length) * Math.abs(accuracy - meanConfidence);
  }

  const latencies = predictionSet.predictions
    .map((prediction) => prediction.latencyMs)
    .sort((left, right) => left - right);
  const result: SystemMetrics = {
    accuracy: {
      caseCount: dataset.cases.length,
      correctCount,
      accuracy: correctCount / dataset.cases.length,
      attestedCaseCount,
      attestedCorrectCount,
      attestedAccuracy: attestedCorrectCount / attestedCaseCount,
    },
    calibration: {
      binCount: ECE_BIN_COUNT,
      expectedCalibrationError,
    },
    latency: {
      p50Ms: nearestRank(latencies, 0.5),
      p95Ms: nearestRank(latencies, 0.95),
    },
    throughput: {
      concurrency: 1,
      measuredDurationMs: predictionSet.protocol.measuredDurationMs,
      requestsPerSecond:
        (predictionSet.predictions.length * 1_000) /
        predictionSet.protocol.measuredDurationMs,
    },
    memory: {
      scope: predictionSet.protocol.memoryScope,
      peakBytes: predictionSet.protocol.peakMemoryBytes,
    },
  };
  return contractInternals.frozenClone(result);
}

export function createBenchmarkResult(
  datasetValue: unknown,
  ledgerValue: unknown,
  predictionValues: readonly unknown[],
  generatedAt: string,
): RefundBenchmarkResultV1 {
  const dataset = validateRefundDataset(datasetValue);
  const ledger = validateTrainingLedger(ledgerValue);
  const leakageAudit = auditDatasetSeparation(dataset, ledger);
  contractInternals.rfc3339(generatedAt, "$.generatedAt");

  const boundedPredictionValues = contractInternals.denseArray(
    predictionValues,
    "$.predictions",
    REQUIRED_SYSTEM_ROLES.length,
  );
  const seenRoles = new Set<BenchmarkSystemRole>();
  const systems = boundedPredictionValues.map((predictionValue) => {
    const prediction = validatePredictionSet(predictionValue);
    if (seenRoles.has(prediction.system.role)) {
      throw new BenchmarkContractError(
        "$.systems",
        `duplicate system role ${JSON.stringify(prediction.system.role)}`,
      );
    }
    seenRoles.add(prediction.system.role);
    assertTrainingEvidenceMatchesLedger(prediction.system, ledger);
    const metrics = evaluatePredictionSet(dataset, prediction);
    return {
      role: prediction.system.role,
      predictionSha256: prediction.payloadSha256,
      taskSpecSha256: prediction.system.taskSpecSha256,
      model: prediction.system.model,
      adapter: prediction.system.adapter,
      trainingEvidence: prediction.system.trainingEvidence,
      executionBackend: prediction.system.executionBackend,
      environment: prediction.environment,
      protocol: prediction.protocol,
      metrics,
    } satisfies BenchmarkSystemResult;
  });
  systems.sort(
    (left, right) =>
      REQUIRED_SYSTEM_ROLES.indexOf(left.role) - REQUIRED_SYSTEM_ROLES.indexOf(right.role),
  );
  assertComparableSystems(systems);

  const goNoGo = computeGoNoGo(systems);
  const payload = {
    kind: "semantscript.refund-benchmark-result",
    resultVersion: 1,
    benchmark: "refund-decision",
    generatedAt,
    taskSpecSha256: REFUND_TASK_SPEC_SHA256,
    datasetSha256: dataset.payloadSha256,
    trainingLedgerSha256: ledger.payloadSha256,
    leakageAudit,
    systems,
    goNoGo,
  } as const;
  return validateBenchmarkResult({
    ...payload,
    payloadSha256: semanticJsonSha256(payload),
  });
}

/**
 * Validate a complete committed benchmark bundle and recompute its result from
 * the supplied dataset, lifecycle ledger, and exact prediction files.
 *
 * `validateBenchmarkResult` intentionally validates a standalone result
 * document. Publication and review should use this stronger boundary so a
 * self-consistent result cannot point at absent or different predictions.
 */
export function validateBenchmarkBundle(
  datasetValue: unknown,
  ledgerValue: unknown,
  predictionValues: readonly unknown[],
  resultValue: unknown,
): RefundBenchmarkResultV1 {
  if (!Array.isArray(predictionValues)) {
    throw new BenchmarkContractError("$.predictions", "must be an array");
  }
  const supplied = validateBenchmarkResult(resultValue);
  const recomputed = createBenchmarkResult(
    datasetValue,
    ledgerValue,
    predictionValues,
    supplied.generatedAt,
  );
  if (supplied.payloadSha256 !== recomputed.payloadSha256) {
    throw new BenchmarkContractError(
      "$.payloadSha256",
      "does not match the result recomputed from the supplied benchmark bundle",
    );
  }
  return supplied;
}

export function validateBenchmarkResult(value: unknown): RefundBenchmarkResultV1 {
  const {
    exactObject,
    denseArray,
    fail,
    frozenClone,
    nonNegativeFinite,
    nonNegativeSafeInteger,
    positiveFinite,
    rfc3339,
    sha256,
    validateEnvironment,
    validatePayloadDigest,
  } = contractInternals;
  const root = exactObject(value, "$", [
    "kind",
    "resultVersion",
    "benchmark",
    "generatedAt",
    "taskSpecSha256",
    "datasetSha256",
    "trainingLedgerSha256",
    "leakageAudit",
    "systems",
    "goNoGo",
    "payloadSha256",
  ]);
  if (root.kind !== "semantscript.refund-benchmark-result") {
    fail("$.kind", 'must be "semantscript.refund-benchmark-result"');
  }
  if (root.resultVersion !== 1) {
    fail("$.resultVersion", "must be 1");
  }
  if (root.benchmark !== "refund-decision") {
    fail("$.benchmark", 'must be "refund-decision"');
  }
  rfc3339(root.generatedAt, "$.generatedAt");
  if (root.taskSpecSha256 !== REFUND_TASK_SPEC_SHA256) {
    fail(
      "$.taskSpecSha256",
      `must bind the canonical refund task specification ${REFUND_TASK_SPEC_SHA256}`,
    );
  }
  const datasetSha256 = sha256(root.datasetSha256, "$.datasetSha256");
  const trainingLedgerSha256 = sha256(
    root.trainingLedgerSha256,
    "$.trainingLedgerSha256",
  );

  const audit = exactObject(root.leakageAudit, "$.leakageAudit", [
    "datasetSha256",
    "trainingLedgerSha256",
    "evaluationCaseCount",
    "trainingInputCount",
    "overlapCount",
  ]);
  if (sha256(audit.datasetSha256, "$.leakageAudit.datasetSha256") !== datasetSha256) {
    fail("$.leakageAudit.datasetSha256", "must match result datasetSha256");
  }
  if (
    sha256(audit.trainingLedgerSha256, "$.leakageAudit.trainingLedgerSha256") !==
    trainingLedgerSha256
  ) {
    fail("$.leakageAudit.trainingLedgerSha256", "must match result trainingLedgerSha256");
  }
  const evaluationCaseCount = nonNegativeSafeInteger(
    audit.evaluationCaseCount,
    "$.leakageAudit.evaluationCaseCount",
  );
  if (evaluationCaseCount === 0) {
    fail("$.leakageAudit.evaluationCaseCount", "must be greater than zero");
  }
  nonNegativeSafeInteger(audit.trainingInputCount, "$.leakageAudit.trainingInputCount");
  if (audit.overlapCount !== 0) {
    fail("$.leakageAudit.overlapCount", "must be zero");
  }

  const systemValues = denseArray(
    root.systems,
    "$.systems",
    REQUIRED_SYSTEM_ROLES.length,
  );
  const systems: BenchmarkSystemResult[] = [];
  let previousRoleIndex = -1;
  for (const [index, systemValue] of systemValues.entries()) {
    const path = `$.systems[${String(index)}]`;
    const system = exactObject(systemValue, path, [
      "role",
      "predictionSha256",
      "taskSpecSha256",
      "model",
      "adapter",
      "trainingEvidence",
      "executionBackend",
      "environment",
      "protocol",
      "metrics",
    ]);
    const roleIndex = REQUIRED_SYSTEM_ROLES.indexOf(system.role as BenchmarkSystemRole);
    if (roleIndex <= previousRoleIndex) {
      fail(`${path}.role`, "systems must be unique and in required role order");
    }
    previousRoleIndex = roleIndex;
    if (roleIndex < 0) {
      fail(`${path}.role`, "is not a supported benchmark system role");
    }
    sha256(system.predictionSha256, `${path}.predictionSha256`);
    contractInternals.validateSystemProvenance(
      {
        role: system.role,
        taskSpecSha256: system.taskSpecSha256,
        model: system.model,
        adapter: system.adapter,
        trainingEvidence: system.trainingEvidence,
        executionBackend: system.executionBackend,
      },
      path,
    );
    if (
      system.role === "semantscript" &&
      (system.trainingEvidence as { trainingLedgerSha256?: unknown })
        .trainingLedgerSha256 !== trainingLedgerSha256
    ) {
      fail(
        `${path}.trainingEvidence.trainingLedgerSha256`,
        "must match result trainingLedgerSha256",
      );
    }
    validateEnvironment(system.environment, `${path}.environment`);
    contractInternals.validateProtocol(system.protocol, `${path}.protocol`);
    validateMetrics(system.metrics, path, evaluationCaseCount);
    systems.push(system as unknown as BenchmarkSystemResult);
  }

  assertComparableSystems(systems);

  validateGoNoGo(root.goNoGo, systems);
  validatePayloadDigest(root, "payloadSha256", "$", root.payloadSha256);
  return frozenClone(root) as unknown as RefundBenchmarkResultV1;

  function validateMetrics(metricsValue: unknown, systemPath: string, caseCount: number): void {
    const path = `${systemPath}.metrics`;
    const metrics = exactObject(metricsValue, path, [
      "accuracy",
      "calibration",
      "latency",
      "throughput",
      "memory",
    ]);
    const accuracy = exactObject(metrics.accuracy, `${path}.accuracy`, [
      "caseCount",
      "correctCount",
      "accuracy",
      "attestedCaseCount",
      "attestedCorrectCount",
      "attestedAccuracy",
    ]);
    if (accuracy.caseCount !== caseCount) {
      fail(`${path}.accuracy.caseCount`, "must match leakage audit evaluationCaseCount");
    }
    const correctCount = nonNegativeSafeInteger(
      accuracy.correctCount,
      `${path}.accuracy.correctCount`,
    );
    if (correctCount > caseCount) {
      fail(`${path}.accuracy.correctCount`, "must not exceed caseCount");
    }
    if (accuracy.accuracy !== correctCount / caseCount) {
      fail(`${path}.accuracy.accuracy`, "must equal correctCount divided by caseCount");
    }
    const attestedCaseCount = nonNegativeSafeInteger(
      accuracy.attestedCaseCount,
      `${path}.accuracy.attestedCaseCount`,
    );
    if (attestedCaseCount === 0 || attestedCaseCount > caseCount) {
      fail(`${path}.accuracy.attestedCaseCount`, "must be from one through caseCount");
    }
    const attestedCorrectCount = nonNegativeSafeInteger(
      accuracy.attestedCorrectCount,
      `${path}.accuracy.attestedCorrectCount`,
    );
    if (attestedCorrectCount > attestedCaseCount) {
      fail(`${path}.accuracy.attestedCorrectCount`, "must not exceed attestedCaseCount");
    }
    if (accuracy.attestedAccuracy !== attestedCorrectCount / attestedCaseCount) {
      fail(
        `${path}.accuracy.attestedAccuracy`,
        "must equal attestedCorrectCount divided by attestedCaseCount",
      );
    }

    const calibration = exactObject(metrics.calibration, `${path}.calibration`, [
      "binCount",
      "expectedCalibrationError",
    ]);
    if (calibration.binCount !== ECE_BIN_COUNT) {
      fail(`${path}.calibration.binCount`, `must be ${String(ECE_BIN_COUNT)}`);
    }
    const ece = nonNegativeFinite(
      calibration.expectedCalibrationError,
      `${path}.calibration.expectedCalibrationError`,
    );
    if (ece > 1) {
      fail(`${path}.calibration.expectedCalibrationError`, "must not exceed 1");
    }

    const latency = exactObject(metrics.latency, `${path}.latency`, ["p50Ms", "p95Ms"]);
    const p50 = nonNegativeFinite(latency.p50Ms, `${path}.latency.p50Ms`);
    const p95 = nonNegativeFinite(latency.p95Ms, `${path}.latency.p95Ms`);
    if (p50 > p95) {
      fail(`${path}.latency`, "p50Ms must not exceed p95Ms");
    }

    const throughput = exactObject(metrics.throughput, `${path}.throughput`, [
      "concurrency",
      "measuredDurationMs",
      "requestsPerSecond",
    ]);
    if (throughput.concurrency !== 1) {
      fail(`${path}.throughput.concurrency`, "must be 1");
    }
    const duration = positiveFinite(
      throughput.measuredDurationMs,
      `${path}.throughput.measuredDurationMs`,
    );
    if (throughput.requestsPerSecond !== (caseCount * 1_000) / duration) {
      fail(
        `${path}.throughput.requestsPerSecond`,
        "must equal caseCount * 1000 / measuredDurationMs",
      );
    }

    const memory = exactObject(metrics.memory, `${path}.memory`, ["scope", "peakBytes"]);
    if (memory.scope !== "client-only" && memory.scope !== "process-tree") {
      fail(`${path}.memory.scope`, 'must be "client-only" or "process-tree"');
    }
    nonNegativeSafeInteger(memory.peakBytes, `${path}.memory.peakBytes`);
  }
}

function assertTrainingEvidenceMatchesLedger(
  system: RefundPredictionSetV1["system"],
  ledger: ReturnType<typeof validateTrainingLedger>,
): void {
  const evidence = system.trainingEvidence;
  if (system.role !== "semantscript") {
    return;
  }
  const expectedTrainingKeySha256 = deriveRefundArtifactTrainingKeySha256(
    ledger.sources,
  );
  if (
    evidence?.trainingLedgerSha256 !== ledger.payloadSha256 ||
    evidence.artifactTrainingDatasetSha256 !== ledger.sources.baseDatasetSha256 ||
    evidence.artifactTrainingKeySha256 !== expectedTrainingKeySha256 ||
    evidence.releaseVerificationPayloadSha256 !==
      ledger.sources.releaseVerificationPayloadSha256 ||
    evidence.releaseVerificationAttestationSha256 !==
      ledger.sources.releaseVerificationAttestationSha256
  ) {
    throw new BenchmarkContractError(
      "$.system.trainingEvidence",
      "must exactly bind the supplied training ledger, artifact training dataset/key, and release-verification evidence",
    );
  }
}

function assertComparableSystems(systems: readonly BenchmarkSystemResult[]): void {
  const reference = systems[0];
  if (reference === undefined) return;
  const hardwareIdentity = (system: BenchmarkSystemResult): unknown => ({
    operatingSystem: system.environment.operatingSystem,
    architecture: system.environment.architecture,
    cpu: system.environment.cpu,
    accelerator: system.environment.accelerator,
    memoryBytes: system.environment.memoryBytes,
  });
  const protocolIdentity = (system: BenchmarkSystemResult): unknown => ({
    concurrency: system.protocol.concurrency,
    warmupIterations: system.protocol.warmupIterations,
    warmupInputOrderSha256: system.protocol.warmupInputOrderSha256,
    memoryScope: system.protocol.memoryScope,
  });
  const expectedHardware = semanticJsonSha256(hardwareIdentity(reference));
  const expectedProtocol = semanticJsonSha256(protocolIdentity(reference));
  for (const [index, system] of systems.entries()) {
    if (semanticJsonSha256(hardwareIdentity(system)) !== expectedHardware) {
      throw new BenchmarkContractError(
        `$.systems[${String(index)}].environment`,
        "hardware identity must match every other benchmark system",
      );
    }
    if (semanticJsonSha256(protocolIdentity(system)) !== expectedProtocol) {
      throw new BenchmarkContractError(
        `$.systems[${String(index)}].protocol`,
        "concurrency, warmup iterations/input digest, and memory scope must match every other benchmark system",
      );
    }
  }
}

function assertPredictionCoverage(
  dataset: RefundBenchmarkDatasetV1,
  predictions: RefundPredictionSetV1,
): void {
  if (predictions.datasetSha256 !== dataset.payloadSha256) {
    throw new BenchmarkContractError(
      "$.datasetSha256",
      "prediction set does not bind the supplied held-out dataset",
    );
  }
  if (predictions.predictions.length !== dataset.cases.length) {
    throw new BenchmarkContractError(
      "$.predictions",
      "must contain exactly one prediction for every held-out case",
    );
  }
  for (const [index, benchmarkCase] of dataset.cases.entries()) {
    const prediction = predictions.predictions[index];
    if (
      prediction?.caseId !== benchmarkCase.id ||
      prediction.inputSha256 !== benchmarkCase.inputSha256
    ) {
      throw new BenchmarkContractError(
        `$.predictions[${String(index)}]`,
        "must match the held-out case id and input digest in dataset order",
      );
    }
  }
}

function nearestRank(sortedValues: readonly number[], quantile: number): number {
  const index = Math.max(0, Math.ceil(quantile * sortedValues.length) - 1);
  const value = sortedValues[index];
  if (value === undefined) {
    throw new BenchmarkContractError("$.predictions", "cannot rank an empty latency sample");
  }
  return value;
}

function computeGoNoGo(systems: readonly BenchmarkSystemResult[]): GoNoGoDecision {
  const byRole = new Map(systems.map((system) => [system.role, system]));
  const missingSystems = REQUIRED_SYSTEM_ROLES.filter((role) => !byRole.has(role));
  const semantscript = byRole.get("semantscript");
  const comparator = byRole.get("ollama-7b");
  if (missingSystems.length > 0 || !semantscript || !comparator) {
    return Object.freeze({
      status: "incomplete",
      criterion: {
        semantscriptP50MsExclusiveMaximum: 10 as const,
        accuracyComparator: "ollama-7b" as const,
      },
      semantscriptP50Ms: null,
      semantscriptAccuracy: null,
      comparatorAccuracy: null,
      latencyPassed: null,
      accuracyPassed: null,
      missingSystems,
    });
  }

  const semantscriptP50Ms = semantscript.metrics.latency.p50Ms;
  const semantscriptAccuracy = semantscript.metrics.accuracy.accuracy;
  const comparatorAccuracy = comparator.metrics.accuracy.accuracy;
  const latencyPassed = semantscriptP50Ms < 10;
  const accuracyPassed = semantscriptAccuracy >= comparatorAccuracy;
  return Object.freeze({
    status: latencyPassed && accuracyPassed ? "go" : "no-go",
    criterion: {
      semantscriptP50MsExclusiveMaximum: 10 as const,
      accuracyComparator: "ollama-7b" as const,
    },
    semantscriptP50Ms,
    semantscriptAccuracy,
    comparatorAccuracy,
    latencyPassed,
    accuracyPassed,
    missingSystems,
  });
}

function validateGoNoGo(
  value: unknown,
  systems: readonly BenchmarkSystemResult[],
): void {
  const goNoGo = contractInternals.exactObject(value, "$.goNoGo", [
    "status",
    "criterion",
    "semantscriptP50Ms",
    "semantscriptAccuracy",
    "comparatorAccuracy",
    "latencyPassed",
    "accuracyPassed",
    "missingSystems",
  ]);
  contractInternals.exactObject(goNoGo.criterion, "$.goNoGo.criterion", [
    "semantscriptP50MsExclusiveMaximum",
    "accuracyComparator",
  ]);
  const missing = contractInternals.denseArray(
    goNoGo.missingSystems,
    "$.goNoGo.missingSystems",
  );
  for (const [index, role] of missing.entries()) {
    if (!REQUIRED_SYSTEM_ROLES.includes(role as BenchmarkSystemRole)) {
      contractInternals.fail(
        `$.goNoGo.missingSystems[${String(index)}]`,
        "is not a required system role",
      );
    }
  }
  const expected = computeGoNoGo(systems);
  if (semanticJsonSha256(goNoGo) !== semanticJsonSha256(expected)) {
    contractInternals.fail("$.goNoGo", "does not match the mechanical exit criterion");
  }
}
