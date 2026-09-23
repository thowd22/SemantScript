import { semanticJsonSha256 } from "@semantscript/compiler";

import {
  HUMAN_ATTESTATION_DECLARATION,
  JUDGE_ATTESTATION_DECLARATION,
  REFUND_SUPPORT,
  REQUIRED_SYSTEM_ROLES,
  sealPredictionSet,
  sealRefundDataset,
  sealTrainingLedger,
} from "../dist/index.js";
import {
  REFUND_FUNCTION_BINDING,
  REFUND_SYSTEM_PINS,
  REFUND_TASK_SPEC_SHA256,
  deriveRefundArtifactTrainingKeySha256,
} from "../dist/policy.js";

const functionBinding = REFUND_FUNCTION_BINDING;
const warmupInputOrderSha256 = semanticJsonSha256([
  {
    customer: { priorRefunds: 0, tier: "standard" },
    order: { ageDays: 5, status: "paid", total: 10 },
  },
]);

const inputs = [
  {
    customer: { priorRefunds: 0, tier: "enterprise" },
    order: { ageDays: 45, status: "paid", total: 129 },
  },
  {
    customer: { priorRefunds: 1, tier: "standard" },
    order: { ageDays: 95, status: "paid", total: 25 },
  },
  {
    customer: { priorRefunds: 4, tier: "enterprise" },
    order: { ageDays: 12, status: "fraudulent", total: 400 },
  },
  {
    customer: { priorRefunds: 2, tier: "standard" },
    order: { ageDays: 28, status: "paid", total: 80 },
  },
];

export function makeDataset() {
  const expected = ["approve", "deny", "review", "approve"];
  const origin = ["human-authored", "other-held-out", "human-authored", "other-held-out"];
  return sealRefundDataset({
    kind: "semantscript.refund-benchmark-dataset",
    datasetVersion: 1,
    benchmark: "refund-decision",
    split: "evaluation-only",
    createdAt: "2026-09-23T12:00:00Z",
    function: functionBinding,
    support: REFUND_SUPPORT,
    cases: inputs.map((entry, index) => ({
      id: `case-${String(index + 1).padStart(2, "0")}`,
      inputs: entry,
      inputSha256: semanticJsonSha256(entry),
      expected: expected[index],
      origin: origin[index],
    })),
    humanAttestation: {
      attestor: "unit-test-fixture-not-a-real-attestation",
      attestedAt: "2026-09-23T11:00:00Z",
      caseIds: ["case-01", "case-03"],
      declaration: HUMAN_ATTESTATION_DECLARATION,
      evidenceSha256: "3".repeat(64),
    },
    judgeAttestation: null,
  });
}

export function makeJudgeDataset() {
  const expected = ["approve", "deny", "review", "approve"];
  return sealRefundDataset({
    kind: "semantscript.refund-benchmark-dataset",
    datasetVersion: 1,
    benchmark: "refund-decision",
    split: "evaluation-only",
    createdAt: "2026-09-23T12:00:00Z",
    function: functionBinding,
    support: REFUND_SUPPORT,
    cases: inputs.map((entry, index) => ({
      id: `case-${String(index + 1).padStart(2, "0")}`,
      inputs: entry,
      inputSha256: semanticJsonSha256(entry),
      expected: expected[index],
      origin: "independent-judge",
    })),
    humanAttestation: null,
    judgeAttestation: {
      judge: {
        provider: "unit-test",
        model: "fixture-judge-not-a-real-model",
        interface: "unit-test",
        sessionReference: "unit-test-session",
      },
      rubricSha256: "5".repeat(64),
      attestedAt: "2026-09-23T11:00:00Z",
      caseIds: ["case-01", "case-02", "case-03", "case-04"],
      declaration: JUDGE_ATTESTATION_DECLARATION,
      evidenceSha256: "6".repeat(64),
    },
  });
}

export function makeLedger(partitionOverrides = {}) {
  return sealTrainingLedger({
    kind: "semantscript.refund-training-input-ledger",
    ledgerVersion: 1,
    benchmark: "refund-decision",
    createdAt: "2026-09-23T10:00:00Z",
    function: functionBinding,
    sources: {
      baseDatasetSha256: "a".repeat(64),
      adversarialDatasetSha256: "b".repeat(64),
      releaseVerificationPayloadSha256: "c".repeat(64),
      releaseVerificationAttestationSha256: "d".repeat(64),
    },
    partitions: [
      { name: "examples", inputSha256s: ["4".repeat(64)] },
      { name: "synthetic", inputSha256s: ["5".repeat(64)] },
      { name: "adversarial", inputSha256s: ["6".repeat(64)] },
      { name: "calibration", inputSha256s: ["7".repeat(64)] },
      { name: "verification", inputSha256s: ["8".repeat(64)] },
    ].map((partition) => ({
      ...partition,
      ...(partitionOverrides[partition.name] ?? {}),
    })),
  });
}

export function makePredictionSet(dataset, role = "semantscript", options = {}) {
  const pin = REFUND_SYSTEM_PINS[role];
  const ledger = options.ledger ?? makeLedger();
  const artifactTrainingKeySha256 = deriveRefundArtifactTrainingKeySha256(
    ledger.sources,
  );
  const distributions = options.distributions ?? [
    [0.8, 0.1, 0.1],
    [0.6, 0.3, 0.1],
    [0.1, 0.1, 0.8],
    [0.2, 0.4, 0.4],
  ];
  const values = options.values ?? ["approve", "approve", "review", "deny"];
  const latencies = options.latencies ?? [9, 1, 20, 4];
  return sealPredictionSet({
    kind: "semantscript.refund-benchmark-predictions",
    predictionVersion: 1,
    benchmark: "refund-decision",
    datasetSha256: dataset.payloadSha256,
    support: REFUND_SUPPORT,
    system: {
      role,
      taskSpecSha256: REFUND_TASK_SPEC_SHA256,
      model: {
        provider: pin.model.provider,
        name: pin.model.name,
        version: pin.model.version,
        revision:
          pin.model.revision ??
          (role === "semantscript"
            ? artifactTrainingKeySha256
            : semanticJsonSha256(["test-revision", role])),
        artifactSha256:
          pin.model.artifactSha256 ?? semanticJsonSha256(["test-model", role]),
      },
      adapter: {
        name: pin.adapter.name,
        version: pin.adapter.version,
        configurationSha256: semanticJsonSha256(["test-adapter", role]),
      },
      trainingEvidence:
        role === "semantscript"
          ? {
              trainingLedgerSha256: ledger.payloadSha256,
              artifactTrainingDatasetSha256: ledger.sources.baseDatasetSha256,
              artifactTrainingKeySha256,
              releaseVerificationPayloadSha256:
                ledger.sources.releaseVerificationPayloadSha256,
              releaseVerificationAttestationSha256:
                ledger.sources.releaseVerificationAttestationSha256,
            }
          : null,
      executionBackend:
        options.executionBackend ?? executionBackendForRole(role),
    },
    environment: {
      capturedAt: "2026-09-23T12:30:00Z",
      operatingSystem: "unit-test-os",
      architecture: "test-arch",
      cpu: "test-cpu",
      accelerator: null,
      memoryBytes: 1024,
      runtimeVersions: [
        { name: "adapter", version: "test-only" },
        { name: "node", version: "test-only" },
      ],
      evidenceSha256: semanticJsonSha256(["test-environment", role]),
      ...(options.environment ?? {}),
    },
    protocol: {
      concurrency: 1,
      warmupIterations: options.warmupIterations ?? 2,
      warmupInputOrderSha256:
        options.warmupInputOrderSha256 ?? warmupInputOrderSha256,
      measuredDurationMs: options.measuredDurationMs ?? 200,
      memoryScope: options.memoryScope ?? "process-tree",
      peakMemoryBytes: options.peakMemoryBytes ?? 4096,
    },
    predictions: dataset.cases.map((entry, index) => ({
      caseId: entry.id,
      inputSha256: entry.inputSha256,
      value: values[index],
      distribution: REFUND_SUPPORT.map((supportValue, supportIndex) => ({
        value: supportValue,
        probability: distributions[index][supportIndex],
      })),
      latencyMs: latencies[index],
    })),
  });
}

function executionBackendForRole(role) {
  if (role === "semantscript") {
    return {
      kind: "semantscript-node",
      runtime: "onnxruntime-node",
      device: "cpu",
    };
  }
  if (role === "structured-api") {
    return {
      kind: "anthropic-api",
      apiVersion: "2023-06-01",
      endpoint: "https://api.anthropic.com",
      placement: "provider-managed",
    };
  }
  if (role === "laya") {
    return { kind: "laya", runtime: "python", device: "cuda" };
  }
  return {
    kind: "ollama",
    serverVersion: "0.34.3",
    placement: "gpu",
    modelTotalBytes: 1024,
    modelCpuBytes: 0,
    modelGpuBytes: 1024,
  };
}

export function makeAllPredictionSets(dataset, optionsByRole = {}) {
  const ledger = makeLedger();
  return REQUIRED_SYSTEM_ROLES.map((role) =>
    makePredictionSet(dataset, role, {
      ledger,
      ...(optionsByRole[role] ?? {}),
    }),
  );
}

export function clone(value) {
  return structuredClone(value);
}
