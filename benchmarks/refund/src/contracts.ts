import { compareBytes, semanticJsonSha256 } from "@semantscript/compiler";

import {
  REFUND_FUNCTION_ID,
  REFUND_FUNCTION_SEMANTIC_SHA256,
  REFUND_SYSTEM_PINS,
  REFUND_TASK_SPEC_SHA256,
} from "./policy.js";

import {
  HUMAN_ATTESTATION_DECLARATION,
  JUDGE_ATTESTATION_DECLARATION,
  REFUND_SUPPORT,
  TRAINING_PARTITIONS,
  type AdapterProvenance,
  type BenchmarkSystemRole,
  type EnvironmentProvenance,
  type MeasurementProtocol,
  type ModelProvenance,
  type RefundBenchmarkDatasetV1,
  type RefundCaseOrigin,
  type RefundInputs,
  type RefundPrediction,
  type RefundPredictionSetV1,
  type RefundTrainingLedgerV1,
  type RuntimeVersion,
  type SemantScriptTrainingEvidence,
  type SystemProvenance,
  type TrainingPartition,
} from "./types.js";

const SHA256 = /^[a-f0-9]{64}$/;
const FUNCTION_ID = /^nf_[a-f0-9]{64}$/;
const RECORD_ID = /^[a-z0-9](?:[a-z0-9._-]{0,198}[a-z0-9])?$/;
const RFC3339_UTC =
  /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?Z$/;
const SYSTEM_ROLES = new Set<BenchmarkSystemRole>([
  "semantscript",
  "ollama-1b",
  "ollama-7b",
  "structured-api",
  "laya",
]);
const textEncoder = new TextEncoder();
const PLACEHOLDER_SHA256 = "0".repeat(64);
const MAXIMUM_BENCHMARK_CASE_COUNT = 20_000;
const MAXIMUM_TRAINING_PARTITION_INPUT_COUNT = 50_000;
const MAXIMUM_RUNTIME_VERSION_COUNT = 64;
const MAXIMUM_WARMUP_ITERATIONS = 10_000;

type UnsignedDataset = Omit<RefundBenchmarkDatasetV1, "payloadSha256">;
type UnsignedTrainingLedger = Omit<RefundTrainingLedgerV1, "payloadSha256">;
type UnsignedPredictionSet = Omit<RefundPredictionSetV1, "payloadSha256">;

export class BenchmarkContractError extends TypeError {
  readonly path: string;

  constructor(path: string, message: string) {
    super(`${path}: ${message}`);
    this.name = "BenchmarkContractError";
    this.path = path;
  }
}

export function sealRefundDataset(value: UnsignedDataset): RefundBenchmarkDatasetV1 {
  exactObject(value, "$", [
    "kind",
    "datasetVersion",
    "benchmark",
    "split",
    "createdAt",
    "function",
    "support",
    "cases",
    "humanAttestation",
    "judgeAttestation",
  ]);
  validateRefundDatasetRecord({ ...value, payloadSha256: PLACEHOLDER_SHA256 }, false);
  return validateRefundDataset({ ...value, payloadSha256: semanticJsonSha256(value) });
}

export function sealTrainingLedger(value: UnsignedTrainingLedger): RefundTrainingLedgerV1 {
  exactObject(value, "$", [
    "kind",
    "ledgerVersion",
    "benchmark",
    "createdAt",
    "function",
    "sources",
    "partitions",
  ]);
  validateTrainingLedgerRecord({ ...value, payloadSha256: PLACEHOLDER_SHA256 }, false);
  return validateTrainingLedger({ ...value, payloadSha256: semanticJsonSha256(value) });
}

export function sealPredictionSet(value: UnsignedPredictionSet): RefundPredictionSetV1 {
  exactObject(value, "$", [
    "kind",
    "predictionVersion",
    "benchmark",
    "datasetSha256",
    "support",
    "system",
    "environment",
    "protocol",
    "predictions",
  ]);
  validatePredictionSetRecord({ ...value, payloadSha256: PLACEHOLDER_SHA256 }, false);
  return validatePredictionSet({ ...value, payloadSha256: semanticJsonSha256(value) });
}

export function validateRefundDataset(value: unknown): RefundBenchmarkDatasetV1 {
  return validateRefundDatasetRecord(value, true);
}

function validateRefundDatasetRecord(
  value: unknown,
  finalize: boolean,
): RefundBenchmarkDatasetV1 {
  const root = exactObject(value, "$", [
    "kind",
    "datasetVersion",
    "benchmark",
    "split",
    "createdAt",
    "function",
    "support",
    "cases",
    "humanAttestation",
    "judgeAttestation",
    "payloadSha256",
  ]);

  literal(root.kind, "semantscript.refund-benchmark-dataset", "$.kind");
  literal(root.datasetVersion, 1, "$.datasetVersion");
  literal(root.benchmark, "refund-decision", "$.benchmark");
  literal(root.split, "evaluation-only", "$.split");
  rfc3339(root.createdAt, "$.createdAt");
  validateFunctionBinding(root.function, "$.function");
  validateSupport(root.support, "$.support");

  const cases = denseArray(
    root.cases,
    "$.cases",
    MAXIMUM_BENCHMARK_CASE_COUNT,
  );
  if (cases.length === 0) {
    fail("$.cases", "must contain at least one held-out case");
  }

  const caseIds: string[] = [];
  const caseInputDigests = new Set<string>();
  const caseIdsByOrigin: Record<RefundCaseOrigin, string[]> = {
    "human-authored": [],
    "independent-judge": [],
    "other-held-out": [],
  };
  for (const [index, entry] of cases.entries()) {
    const parsed = validateCase(entry, `$.cases[${String(index)}]`);
    caseIds.push(parsed.id);
    if (caseInputDigests.has(parsed.inputSha256)) {
      fail(`$.cases[${String(index)}].inputSha256`, "duplicates another held-out input");
    }
    caseInputDigests.add(parsed.inputSha256);
    caseIdsByOrigin[parsed.origin].push(parsed.id);
  }
  sortedUnique(caseIds, "$.cases", "case ids");
  if (
    caseIdsByOrigin["human-authored"].length + caseIdsByOrigin["independent-judge"].length ===
    0
  ) {
    fail("$.cases", "must contain at least one attested case (human-authored or independent-judge)");
  }
  validateHumanAttestation(
    root.humanAttestation,
    root.createdAt as string,
    caseIdsByOrigin["human-authored"],
  );
  validateJudgeAttestation(
    root.judgeAttestation,
    root.createdAt as string,
    caseIdsByOrigin["independent-judge"],
  );

  if (finalize) {
    validatePayloadDigest(root, "payloadSha256", "$", root.payloadSha256);
    return frozenClone(root) as unknown as RefundBenchmarkDatasetV1;
  }
  return root as unknown as RefundBenchmarkDatasetV1;
}

export function validateTrainingLedger(value: unknown): RefundTrainingLedgerV1 {
  return validateTrainingLedgerRecord(value, true);
}

function validateTrainingLedgerRecord(
  value: unknown,
  finalize: boolean,
): RefundTrainingLedgerV1 {
  const root = exactObject(value, "$", [
    "kind",
    "ledgerVersion",
    "benchmark",
    "createdAt",
    "function",
    "sources",
    "partitions",
    "payloadSha256",
  ]);
  literal(root.kind, "semantscript.refund-training-input-ledger", "$.kind");
  literal(root.ledgerVersion, 1, "$.ledgerVersion");
  literal(root.benchmark, "refund-decision", "$.benchmark");
  rfc3339(root.createdAt, "$.createdAt");
  validateFunctionBinding(root.function, "$.function");
  const sources = exactObject(root.sources, "$.sources", [
    "baseDatasetSha256",
    "adversarialDatasetSha256",
    "releaseVerificationPayloadSha256",
    "releaseVerificationAttestationSha256",
  ]);
  sha256(sources.baseDatasetSha256, "$.sources.baseDatasetSha256");
  if (sources.adversarialDatasetSha256 !== null) {
    sha256(sources.adversarialDatasetSha256, "$.sources.adversarialDatasetSha256");
  }
  sha256(
    sources.releaseVerificationPayloadSha256,
    "$.sources.releaseVerificationPayloadSha256",
  );
  sha256(
    sources.releaseVerificationAttestationSha256,
    "$.sources.releaseVerificationAttestationSha256",
  );

  const partitions = denseArray(root.partitions, "$.partitions", TRAINING_PARTITIONS.length);
  if (partitions.length !== TRAINING_PARTITIONS.length) {
    fail("$.partitions", "must contain every training lifecycle partition exactly once");
  }
  for (const [index, valueAtIndex] of partitions.entries()) {
    const path = `$.partitions[${String(index)}]`;
    const partition = exactObject(valueAtIndex, path, ["name", "inputSha256s"]);
    literal(partition.name, TRAINING_PARTITIONS[index], `${path}.name`);
    const inputSha256s = denseArray(
      partition.inputSha256s,
      `${path}.inputSha256s`,
      MAXIMUM_TRAINING_PARTITION_INPUT_COUNT,
    ).map((digest, digestIndex) =>
      sha256(digest, `${path}.inputSha256s[${String(digestIndex)}]`),
    );
    sortedUnique(inputSha256s, `${path}.inputSha256s`, "input digests");
  }

  if (finalize) {
    validatePayloadDigest(root, "payloadSha256", "$", root.payloadSha256);
    return frozenClone(root) as unknown as RefundTrainingLedgerV1;
  }
  return root as unknown as RefundTrainingLedgerV1;
}

export function validatePredictionSet(value: unknown): RefundPredictionSetV1 {
  return validatePredictionSetRecord(value, true);
}

function validatePredictionSetRecord(
  value: unknown,
  finalize: boolean,
): RefundPredictionSetV1 {
  const root = exactObject(value, "$", [
    "kind",
    "predictionVersion",
    "benchmark",
    "datasetSha256",
    "support",
    "system",
    "environment",
    "protocol",
    "predictions",
    "payloadSha256",
  ]);
  literal(root.kind, "semantscript.refund-benchmark-predictions", "$.kind");
  literal(root.predictionVersion, 1, "$.predictionVersion");
  literal(root.benchmark, "refund-decision", "$.benchmark");
  sha256(root.datasetSha256, "$.datasetSha256");
  validateSupport(root.support, "$.support");
  validateSystemProvenance(root.system, "$.system");
  validateEnvironment(root.environment, "$.environment");
  validateProtocol(root.protocol, "$.protocol");

  const predictions = denseArray(
    root.predictions,
    "$.predictions",
    MAXIMUM_BENCHMARK_CASE_COUNT,
  );
  if (predictions.length === 0) {
    fail("$.predictions", "must contain at least one measured prediction");
  }
  const caseIds: string[] = [];
  const inputDigests = new Set<string>();
  for (const [index, prediction] of predictions.entries()) {
    const parsed = validatePrediction(prediction, `$.predictions[${String(index)}]`);
    caseIds.push(parsed.caseId);
    if (inputDigests.has(parsed.inputSha256)) {
      fail(`$.predictions[${String(index)}].inputSha256`, "duplicates another prediction input");
    }
    inputDigests.add(parsed.inputSha256);
  }
  sortedUnique(caseIds, "$.predictions", "case ids");

  if (finalize) {
    validatePayloadDigest(root, "payloadSha256", "$", root.payloadSha256);
    return frozenClone(root) as unknown as RefundPredictionSetV1;
  }
  return root as unknown as RefundPredictionSetV1;
}

export function auditDatasetSeparation(
  datasetValue: unknown,
  ledgerValue: unknown,
): {
  readonly datasetSha256: string;
  readonly trainingLedgerSha256: string;
  readonly evaluationCaseCount: number;
  readonly trainingInputCount: number;
  readonly overlapCount: 0;
} {
  const dataset = validateRefundDataset(datasetValue);
  const ledger = validateTrainingLedger(ledgerValue);
  if (
    dataset.function.id !== ledger.function.id ||
    dataset.function.semanticSha256 !== ledger.function.semanticSha256
  ) {
    fail("$", "dataset and training ledger bind different semantic functions");
  }

  const trainingDigests = new Set(
    ledger.partitions.flatMap((partition) => partition.inputSha256s),
  );
  const overlaps = dataset.cases
    .filter((entry) => trainingDigests.has(entry.inputSha256))
    .map((entry) => entry.id);
  if (overlaps.length > 0) {
    fail(
      "$.cases",
      `held-out inputs overlap the training lifecycle ledger: ${overlaps.join(", ")}`,
    );
  }

  return Object.freeze({
    datasetSha256: dataset.payloadSha256,
    trainingLedgerSha256: ledger.payloadSha256,
    evaluationCaseCount: dataset.cases.length,
    trainingInputCount: trainingDigests.size,
    overlapCount: 0,
  });
}

function validateHumanAttestation(
  value: unknown,
  createdAt: string,
  humanCaseIds: readonly string[],
): void {
  if (value === null) {
    if (humanCaseIds.length > 0) {
      fail("$.humanAttestation", "must attest the human-authored cases");
    }
    return;
  }
  const attestation = exactObject(value, "$.humanAttestation", [
    "attestor",
    "attestedAt",
    "caseIds",
    "declaration",
    "evidenceSha256",
  ]);
  boundedString(attestation.attestor, "$.humanAttestation.attestor");
  rfc3339(attestation.attestedAt, "$.humanAttestation.attestedAt");
  if (Date.parse(attestation.attestedAt as string) > Date.parse(createdAt)) {
    fail("$.humanAttestation.attestedAt", "must not be later than dataset creation");
  }
  literal(
    attestation.declaration,
    HUMAN_ATTESTATION_DECLARATION,
    "$.humanAttestation.declaration",
  );
  sha256(attestation.evidenceSha256, "$.humanAttestation.evidenceSha256");
  const attestedCaseIds = attestedCaseIdList(attestation.caseIds, "$.humanAttestation.caseIds");
  if (!sameStrings(attestedCaseIds, humanCaseIds)) {
    fail(
      "$.humanAttestation.caseIds",
      "must exactly list every and only human-authored case",
    );
  }
}

function validateJudgeAttestation(
  value: unknown,
  createdAt: string,
  judgeCaseIds: readonly string[],
): void {
  if (value === null) {
    if (judgeCaseIds.length > 0) {
      fail("$.judgeAttestation", "must attest the independent-judge cases");
    }
    return;
  }
  const attestation = exactObject(value, "$.judgeAttestation", [
    "judge",
    "rubricSha256",
    "attestedAt",
    "caseIds",
    "declaration",
    "evidenceSha256",
  ]);
  const judge = exactObject(attestation.judge, "$.judgeAttestation.judge", [
    "provider",
    "model",
    "interface",
    "sessionReference",
  ]);
  boundedString(judge.provider, "$.judgeAttestation.judge.provider");
  boundedString(judge.model, "$.judgeAttestation.judge.model");
  boundedString(judge.interface, "$.judgeAttestation.judge.interface");
  boundedString(judge.sessionReference, "$.judgeAttestation.judge.sessionReference");
  sha256(attestation.rubricSha256, "$.judgeAttestation.rubricSha256");
  rfc3339(attestation.attestedAt, "$.judgeAttestation.attestedAt");
  if (Date.parse(attestation.attestedAt as string) > Date.parse(createdAt)) {
    fail("$.judgeAttestation.attestedAt", "must not be later than dataset creation");
  }
  literal(
    attestation.declaration,
    JUDGE_ATTESTATION_DECLARATION,
    "$.judgeAttestation.declaration",
  );
  sha256(attestation.evidenceSha256, "$.judgeAttestation.evidenceSha256");
  const attestedCaseIds = attestedCaseIdList(attestation.caseIds, "$.judgeAttestation.caseIds");
  if (!sameStrings(attestedCaseIds, judgeCaseIds)) {
    fail(
      "$.judgeAttestation.caseIds",
      "must exactly list every and only independent-judge case",
    );
  }
}

function attestedCaseIdList(value: unknown, path: string): string[] {
  const caseIds = denseArray(value, path, MAXIMUM_BENCHMARK_CASE_COUNT).map((caseId, index) =>
    recordId(caseId, `${path}[${String(index)}]`),
  );
  sortedUnique(caseIds, path, "case ids");
  return caseIds;
}

function validateCase(
  value: unknown,
  path: string,
): { readonly id: string; readonly inputSha256: string; readonly origin: RefundCaseOrigin } {
  const entry = exactObject(value, path, ["id", "inputs", "inputSha256", "expected", "origin"]);
  const id = recordId(entry.id, `${path}.id`);
  validateRefundInputs(entry.inputs, `${path}.inputs`);
  const inputSha256 = sha256(entry.inputSha256, `${path}.inputSha256`);
  const computed = semanticJsonSha256(entry.inputs);
  if (inputSha256 !== computed) {
    fail(`${path}.inputSha256`, "does not match semantic JSON of inputs");
  }
  refundDecision(entry.expected, `${path}.expected`);
  if (
    entry.origin !== "human-authored" &&
    entry.origin !== "independent-judge" &&
    entry.origin !== "other-held-out"
  ) {
    fail(`${path}.origin`, 'must be "human-authored", "independent-judge" or "other-held-out"');
  }
  return { id, inputSha256, origin: entry.origin };
}

function validateRefundInputs(value: unknown, path: string): RefundInputs {
  const inputs = exactObject(value, path, ["customer", "order"]);
  const customer = exactObject(inputs.customer, `${path}.customer`, ["priorRefunds", "tier"]);
  nonNegativeRefundSafeInteger(customer.priorRefunds, `${path}.customer.priorRefunds`);
  if (customer.tier !== "enterprise" && customer.tier !== "standard") {
    fail(`${path}.customer.tier`, 'must be "enterprise" or "standard"');
  }
  const order = exactObject(inputs.order, `${path}.order`, ["ageDays", "status", "total"]);
  nonNegativeRefundFinite(order.ageDays, `${path}.order.ageDays`);
  if (order.status !== "fraudulent" && order.status !== "paid") {
    fail(`${path}.order.status`, 'must be "fraudulent" or "paid"');
  }
  nonNegativeRefundFinite(order.total, `${path}.order.total`);
  return value as RefundInputs;
}

function validatePrediction(value: unknown, path: string): RefundPrediction {
  const prediction = exactObject(value, path, [
    "caseId",
    "inputSha256",
    "value",
    "distribution",
    "latencyMs",
  ]);
  recordId(prediction.caseId, `${path}.caseId`);
  sha256(prediction.inputSha256, `${path}.inputSha256`);
  const selected = refundDecision(prediction.value, `${path}.value`);
  const distribution = denseArray(
    prediction.distribution,
    `${path}.distribution`,
    REFUND_SUPPORT.length,
  );
  if (distribution.length !== REFUND_SUPPORT.length) {
    fail(`${path}.distribution`, "must contain one entry for every support value");
  }

  let total = 0;
  let bestIndex = 0;
  let bestProbability = -1;
  for (const [index, entryValue] of distribution.entries()) {
    const entryPath = `${path}.distribution[${String(index)}]`;
    const entry = exactObject(entryValue, entryPath, ["value", "probability"]);
    literal(entry.value, REFUND_SUPPORT[index], `${entryPath}.value`);
    const probability = boundedProbability(entry.probability, `${entryPath}.probability`);
    total += probability;
    if (probability > bestProbability) {
      bestProbability = probability;
      bestIndex = index;
    }
  }
  if (Math.abs(total - 1) > 1e-12) {
    fail(`${path}.distribution`, "probabilities must sum to 1 within 1e-12");
  }
  if (selected !== REFUND_SUPPORT[bestIndex]) {
    fail(`${path}.value`, "must be the stable support-order argmax of distribution");
  }
  nonNegativeFinite(prediction.latencyMs, `${path}.latencyMs`);
  return prediction as unknown as RefundPrediction;
}

function validateSystemProvenance(value: unknown, path: string): SystemProvenance {
  const system = exactObject(value, path, [
    "role",
    "taskSpecSha256",
    "model",
    "adapter",
    "trainingEvidence",
    "executionBackend",
  ]);
  if (typeof system.role !== "string" || !SYSTEM_ROLES.has(system.role as BenchmarkSystemRole)) {
    fail(`${path}.role`, "is not a supported benchmark system role");
  }
  const role = system.role as BenchmarkSystemRole;
  literal(system.taskSpecSha256, REFUND_TASK_SPEC_SHA256, `${path}.taskSpecSha256`);
  const model = validateModelProvenance(system.model, `${path}.model`);
  const adapter = validateAdapterProvenance(system.adapter, `${path}.adapter`);
  validatePinnedSystemIdentity(role, model, adapter, path);
  const trainingEvidence = validateTrainingEvidence(
    system.trainingEvidence,
    role,
    `${path}.trainingEvidence`,
  );
  if (
    role === "semantscript" &&
    trainingEvidence?.artifactTrainingKeySha256 !== model.revision
  ) {
    fail(
      `${path}.trainingEvidence.artifactTrainingKeySha256`,
      "must equal the SemantScript model revision",
    );
  }
  validateExecutionBackend(system.executionBackend, role, `${path}.executionBackend`);
  return system as unknown as SystemProvenance;
}

function validateExecutionBackend(
  value: unknown,
  role: BenchmarkSystemRole,
  path: string,
): void {
  if (role === "semantscript") {
    const backend = exactObject(value, path, ["kind", "runtime", "device"]);
    literal(backend.kind, "semantscript-node", `${path}.kind`);
    literal(backend.runtime, "onnxruntime-node", `${path}.runtime`);
    literal(backend.device, "cpu", `${path}.device`);
    return;
  }
  if (role === "structured-api") {
    const backend = exactObject(value, path, [
      "kind",
      "apiVersion",
      "endpoint",
      "placement",
    ]);
    literal(backend.kind, "anthropic-api", `${path}.kind`);
    literal(backend.apiVersion, "2023-06-01", `${path}.apiVersion`);
    literal(backend.endpoint, "https://api.anthropic.com", `${path}.endpoint`);
    literal(backend.placement, "provider-managed", `${path}.placement`);
    return;
  }
  if (role === "laya") {
    const backend = exactObject(value, path, ["kind", "runtime", "device"]);
    literal(backend.kind, "laya", `${path}.kind`);
    literal(backend.runtime, "python", `${path}.runtime`);
    if (backend.device !== "cpu" && backend.device !== "cuda") {
      fail(`${path}.device`, 'must be "cpu" or "cuda"');
    }
    return;
  }

  const backend = exactObject(value, path, [
    "kind",
    "serverVersion",
    "placement",
    "modelTotalBytes",
    "modelCpuBytes",
    "modelGpuBytes",
  ]);
  literal(backend.kind, "ollama", `${path}.kind`);
  boundedString(backend.serverVersion, `${path}.serverVersion`);
  const total = nonNegativeSafeInteger(backend.modelTotalBytes, `${path}.modelTotalBytes`);
  const cpu = nonNegativeSafeInteger(backend.modelCpuBytes, `${path}.modelCpuBytes`);
  const gpu = nonNegativeSafeInteger(backend.modelGpuBytes, `${path}.modelGpuBytes`);
  if (total === 0) {
    fail(`${path}.modelTotalBytes`, "must be greater than zero");
  }
  if (cpu + gpu !== total) {
    fail(path, "modelCpuBytes plus modelGpuBytes must equal modelTotalBytes");
  }
  const placement = gpu === 0 ? "cpu" : cpu === 0 ? "gpu" : "hybrid";
  literal(backend.placement, placement, `${path}.placement`);
}

function validatePinnedSystemIdentity(
  role: BenchmarkSystemRole,
  model: ModelProvenance,
  adapter: AdapterProvenance,
  path: string,
): void {
  const pin = REFUND_SYSTEM_PINS[role];
  for (const key of ["provider", "name", "version"] as const) {
    literal(model[key], pin.model[key], `${path}.model.${key}`);
  }
  if (pin.model.revision === null) {
    sha256(model.revision, `${path}.model.revision`);
  } else {
    literal(model.revision, pin.model.revision, `${path}.model.revision`);
  }
  if (pin.model.artifactSha256 !== null) {
    literal(
      model.artifactSha256,
      pin.model.artifactSha256,
      `${path}.model.artifactSha256`,
    );
  }
  literal(adapter.name, pin.adapter.name, `${path}.adapter.name`);
  literal(adapter.version, pin.adapter.version, `${path}.adapter.version`);
}

function validateTrainingEvidence(
  value: unknown,
  role: BenchmarkSystemRole,
  path: string,
): SemantScriptTrainingEvidence | null {
  if (role !== "semantscript") {
    literal(value, null, path);
    return null;
  }
  const evidence = exactObject(value, path, [
    "trainingLedgerSha256",
    "artifactTrainingDatasetSha256",
    "artifactTrainingKeySha256",
    "releaseVerificationPayloadSha256",
    "releaseVerificationAttestationSha256",
  ]);
  sha256(evidence.trainingLedgerSha256, `${path}.trainingLedgerSha256`);
  sha256(
    evidence.artifactTrainingDatasetSha256,
    `${path}.artifactTrainingDatasetSha256`,
  );
  sha256(evidence.artifactTrainingKeySha256, `${path}.artifactTrainingKeySha256`);
  sha256(
    evidence.releaseVerificationPayloadSha256,
    `${path}.releaseVerificationPayloadSha256`,
  );
  sha256(
    evidence.releaseVerificationAttestationSha256,
    `${path}.releaseVerificationAttestationSha256`,
  );
  return evidence as unknown as SemantScriptTrainingEvidence;
}

function validateModelProvenance(value: unknown, path: string): ModelProvenance {
  const model = exactObject(value, path, [
    "provider",
    "name",
    "version",
    "revision",
    "artifactSha256",
  ]);
  boundedString(model.provider, `${path}.provider`);
  boundedString(model.name, `${path}.name`);
  boundedString(model.version, `${path}.version`);
  boundedString(model.revision, `${path}.revision`);
  sha256(model.artifactSha256, `${path}.artifactSha256`);
  return model as unknown as ModelProvenance;
}

function validateAdapterProvenance(value: unknown, path: string): AdapterProvenance {
  const adapter = exactObject(value, path, ["name", "version", "configurationSha256"]);
  boundedString(adapter.name, `${path}.name`);
  boundedString(adapter.version, `${path}.version`);
  sha256(adapter.configurationSha256, `${path}.configurationSha256`);
  return adapter as unknown as AdapterProvenance;
}

function validateEnvironment(value: unknown, path: string): EnvironmentProvenance {
  const environment = exactObject(value, path, [
    "capturedAt",
    "operatingSystem",
    "architecture",
    "cpu",
    "accelerator",
    "memoryBytes",
    "runtimeVersions",
    "evidenceSha256",
  ]);
  rfc3339(environment.capturedAt, `${path}.capturedAt`);
  boundedString(environment.operatingSystem, `${path}.operatingSystem`);
  boundedString(environment.architecture, `${path}.architecture`);
  boundedString(environment.cpu, `${path}.cpu`);
  if (environment.accelerator !== null) {
    boundedString(environment.accelerator, `${path}.accelerator`);
  }
  nonNegativeSafeInteger(environment.memoryBytes, `${path}.memoryBytes`);
  if ((environment.memoryBytes as number) === 0) {
    fail(`${path}.memoryBytes`, "must be greater than zero");
  }
  const runtimeVersions = denseArray(
    environment.runtimeVersions,
    `${path}.runtimeVersions`,
    MAXIMUM_RUNTIME_VERSION_COUNT,
  );
  if (runtimeVersions.length === 0) {
    fail(`${path}.runtimeVersions`, "must contain at least one runtime version");
  }
  const names: string[] = [];
  for (const [index, runtimeValue] of runtimeVersions.entries()) {
    const runtimePath = `${path}.runtimeVersions[${String(index)}]`;
    const runtime = exactObject(runtimeValue, runtimePath, ["name", "version"]);
    names.push(boundedString(runtime.name, `${runtimePath}.name`));
    boundedString(runtime.version, `${runtimePath}.version`);
  }
  sortedUnique(names, `${path}.runtimeVersions`, "runtime names");
  sha256(environment.evidenceSha256, `${path}.evidenceSha256`);
  return environment as unknown as EnvironmentProvenance;
}

function validateProtocol(value: unknown, path: string): MeasurementProtocol {
  const protocol = exactObject(value, path, [
    "concurrency",
    "warmupIterations",
    "warmupInputOrderSha256",
    "measuredDurationMs",
    "memoryScope",
    "peakMemoryBytes",
  ]);
  literal(protocol.concurrency, 1, `${path}.concurrency`);
  const warmupIterations = nonNegativeSafeInteger(
    protocol.warmupIterations,
    `${path}.warmupIterations`,
  );
  if (warmupIterations > MAXIMUM_WARMUP_ITERATIONS) {
    fail(
      `${path}.warmupIterations`,
      `must not exceed ${String(MAXIMUM_WARMUP_ITERATIONS)}`,
    );
  }
  sha256(protocol.warmupInputOrderSha256, `${path}.warmupInputOrderSha256`);
  positiveFinite(protocol.measuredDurationMs, `${path}.measuredDurationMs`);
  if (protocol.memoryScope !== "client-only" && protocol.memoryScope !== "process-tree") {
    fail(`${path}.memoryScope`, 'must be "client-only" or "process-tree"');
  }
  nonNegativeSafeInteger(protocol.peakMemoryBytes, `${path}.peakMemoryBytes`);
  return protocol as unknown as MeasurementProtocol;
}

function validateFunctionBinding(value: unknown, path: string): void {
  const binding = exactObject(value, path, ["id", "semanticSha256"]);
  if (typeof binding.id !== "string" || !FUNCTION_ID.test(binding.id)) {
    fail(`${path}.id`, "must be a SemantScript function id");
  }
  literal(binding.id, REFUND_FUNCTION_ID, `${path}.id`);
  literal(
    binding.semanticSha256,
    REFUND_FUNCTION_SEMANTIC_SHA256,
    `${path}.semanticSha256`,
  );
}

function validateSupport(value: unknown, path: string): void {
  const support = denseArray(value, path, REFUND_SUPPORT.length);
  if (support.length !== REFUND_SUPPORT.length) {
    fail(path, "must be the canonical refund support");
  }
  for (const [index, expected] of REFUND_SUPPORT.entries()) {
    literal(support[index], expected, `${path}[${String(index)}]`);
  }
}

function validatePayloadDigest(
  object: object,
  digestKey: string,
  path: string,
  digestValue: unknown,
): void {
  const digest = sha256(digestValue, `${path}.${digestKey}`);
  const payload = Object.fromEntries(
    Object.entries(object).filter(([key]) => key !== digestKey),
  );
  const expected = semanticJsonSha256(payload);
  if (digest !== expected) {
    fail(`${path}.${digestKey}`, "does not match the semantic JSON payload digest");
  }
}

function exactObject<const Keys extends readonly string[]>(
  value: unknown,
  path: string,
  keys: Keys,
): { [Key in Keys[number]]: unknown } {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(path, "must be an object");
  }
  const prototype = Object.getPrototypeOf(value) as unknown;
  if (prototype !== Object.prototype && prototype !== null) {
    fail(path, "must be a plain object");
  }
  if (Object.getOwnPropertySymbols(value).length > 0) {
    fail(path, "must not contain symbol properties");
  }
  const names = Object.getOwnPropertyNames(value);
  if (names.length !== keys.length || keys.some((key) => !names.includes(key))) {
    fail(path, `must contain exactly: ${keys.join(", ")}`);
  }
  for (const key of keys) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor?.enumerable || !("value" in descriptor)) {
      fail(`${path}.${key}`, "must be an enumerable data property");
    }
  }
  return value as { [Key in Keys[number]]: unknown };
}

function denseArray(
  value: unknown,
  path: string,
  maximumLength = Number.MAX_SAFE_INTEGER,
): readonly unknown[] {
  if (!Array.isArray(value)) {
    fail(path, "must be an array");
  }
  if (value.length > maximumLength) {
    fail(path, `must contain at most ${String(maximumLength)} items`);
  }
  if (Object.getOwnPropertySymbols(value).length > 0) {
    fail(path, "must not contain symbol properties");
  }
  const names = Object.getOwnPropertyNames(value).filter((name) => name !== "length");
  if (
    names.length !== value.length ||
    names.some((name, index) => name !== String(index))
  ) {
    fail(path, "must be dense and contain no non-index properties");
  }
  for (let index = 0; index < value.length; index += 1) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index));
    if (!descriptor?.enumerable || !("value" in descriptor)) {
      fail(`${path}[${String(index)}]`, "must be an enumerable data property");
    }
  }
  return value;
}

function recordId(value: unknown, path: string): string {
  if (typeof value !== "string" || !RECORD_ID.test(value)) {
    fail(path, "must be a lowercase stable record id");
  }
  return value;
}

function boundedString(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0 || value.length > 500) {
    fail(path, "must be a non-empty string of at most 500 UTF-16 code units");
  }
  semanticJsonSha256(value);
  return value;
}

function sha256(value: unknown, path: string): string {
  if (typeof value !== "string" || !SHA256.test(value)) {
    fail(path, "must be 64 lowercase hexadecimal characters");
  }
  return value;
}

function rfc3339(value: unknown, path: string): string {
  if (typeof value !== "string") {
    fail(path, "must be an RFC 3339 UTC timestamp");
  }
  const match = RFC3339_UTC.exec(value);
  if (!match) {
    fail(path, "must be an RFC 3339 UTC timestamp");
  }
  const [, year, month, day, hour, minute, second, fraction = ""] = match;
  const milliseconds = Number(fraction.padEnd(3, "0"));
  const date = new Date(
    Date.UTC(
      Number(year),
      Number(month) - 1,
      Number(day),
      Number(hour),
      Number(minute),
      Number(second),
      milliseconds,
    ),
  );
  if (
    date.getUTCFullYear() !== Number(year) ||
    date.getUTCMonth() !== Number(month) - 1 ||
    date.getUTCDate() !== Number(day) ||
    date.getUTCHours() !== Number(hour) ||
    date.getUTCMinutes() !== Number(minute) ||
    date.getUTCSeconds() !== Number(second) ||
    date.getUTCMilliseconds() !== milliseconds
  ) {
    fail(path, "must be a real calendar timestamp");
  }
  return value;
}

function refundDecision(value: unknown, path: string): "approve" | "deny" | "review" {
  if (value !== "approve" && value !== "deny" && value !== "review") {
    fail(path, "must be a canonical refund decision");
  }
  return value;
}

function nonNegativeSafeInteger(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    fail(path, "must be a non-negative safe integer");
  }
  return value;
}

function nonNegativeRefundSafeInteger(value: unknown, path: string): number {
  const number = nonNegativeSafeInteger(value, path);
  rejectNegativeZero(number, path);
  return number;
}

function nonNegativeFinite(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    fail(path, "must be a non-negative finite number");
  }
  return value;
}

function nonNegativeRefundFinite(value: unknown, path: string): number {
  const number = nonNegativeFinite(value, path);
  rejectNegativeZero(number, path);
  return number;
}

function rejectNegativeZero(value: number, path: string): void {
  if (Object.is(value, -0)) {
    fail(path, "must not be negative zero");
  }
}

function positiveFinite(value: unknown, path: string): number {
  const number = nonNegativeFinite(value, path);
  if (number === 0) {
    fail(path, "must be greater than zero");
  }
  return number;
}

function boundedProbability(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    fail(path, "must be a finite probability from 0 through 1");
  }
  return value;
}

function sortedUnique(values: readonly string[], path: string, description: string): void {
  for (let index = 1; index < values.length; index += 1) {
    if (
      compareBytes(
        textEncoder.encode(values[index - 1]),
        textEncoder.encode(values[index]),
      ) >= 0
    ) {
      fail(path, `${description} must be strictly ascending and unique`);
    }
  }
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function literal(value: unknown, expected: unknown, path: string): void {
  if (value !== expected) {
    fail(path, `must be ${JSON.stringify(expected)}`);
  }
}

function frozenClone<T>(value: T): Readonly<T> {
  return deepFreeze(structuredClone(value));
}

function deepFreeze<T>(value: T): Readonly<T> {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    for (const entry of Object.values(value)) {
      deepFreeze(entry);
    }
    Object.freeze(value);
  }
  return value;
}

function fail(path: string, message: string): never {
  throw new BenchmarkContractError(path, message);
}

export const contractInternals = Object.freeze({
  boundedProbability,
  denseArray,
  exactObject,
  fail,
  frozenClone,
  nonNegativeFinite,
  nonNegativeSafeInteger,
  positiveFinite,
  rfc3339,
  sha256,
  validateAdapterProvenance,
  validateEnvironment,
  validateModelProvenance,
  validatePayloadDigest,
  validateProtocol,
  validateRefundInputs,
  validateSystemProvenance,
  validateTrainingEvidence,
});

export type {
  AdapterProvenance,
  EnvironmentProvenance,
  MeasurementProtocol,
  ModelProvenance,
  RefundPrediction,
  RuntimeVersion,
  SemantScriptTrainingEvidence,
  SystemProvenance,
  TrainingPartition,
};
