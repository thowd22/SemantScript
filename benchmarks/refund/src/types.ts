export const REFUND_SUPPORT = ["approve", "deny", "review"] as const;

export type RefundDecision = (typeof REFUND_SUPPORT)[number];

export const TRAINING_PARTITIONS = [
  "examples",
  "synthetic",
  "adversarial",
  "calibration",
  "verification",
] as const;

export type TrainingPartitionName = (typeof TRAINING_PARTITIONS)[number];

export const REQUIRED_SYSTEM_ROLES = [
  "semantscript",
  "ollama-1b",
  "ollama-7b",
  "structured-api",
  "laya",
] as const;

export type BenchmarkSystemRole = (typeof REQUIRED_SYSTEM_ROLES)[number];
export type MemoryScope = "client-only" | "process-tree";

export interface RefundInputs {
  readonly customer: {
    readonly priorRefunds: number;
    readonly tier: "enterprise" | "standard";
  };
  readonly order: {
    readonly ageDays: number;
    readonly status: "fraudulent" | "paid";
    readonly total: number;
  };
}

export interface RefundBenchmarkCase {
  readonly id: string;
  readonly inputs: RefundInputs;
  readonly inputSha256: string;
  readonly expected: RefundDecision;
  readonly origin: "human-authored" | "other-held-out";
}

export interface HumanAttestation {
  readonly attestor: string;
  readonly attestedAt: string;
  readonly caseIds: readonly string[];
  readonly declaration: typeof HUMAN_ATTESTATION_DECLARATION;
  readonly evidenceSha256: string;
}

export const HUMAN_ATTESTATION_DECLARATION =
  "The listed cases are human-authored, were not generated or rewritten by any model or teacher, and are licensed or de-identified for this benchmark." as const;

export interface RefundBenchmarkDatasetV1 {
  readonly kind: "semantscript.refund-benchmark-dataset";
  readonly datasetVersion: 1;
  readonly benchmark: "refund-decision";
  readonly split: "evaluation-only";
  readonly createdAt: string;
  readonly function: {
    readonly id: string;
    readonly semanticSha256: string;
  };
  readonly support: typeof REFUND_SUPPORT;
  readonly cases: readonly RefundBenchmarkCase[];
  readonly humanAttestation: HumanAttestation;
  readonly payloadSha256: string;
}

export interface TrainingPartition {
  readonly name: TrainingPartitionName;
  readonly inputSha256s: readonly string[];
}

export interface RefundTrainingLedgerV1 {
  readonly kind: "semantscript.refund-training-input-ledger";
  readonly ledgerVersion: 1;
  readonly benchmark: "refund-decision";
  readonly createdAt: string;
  readonly function: {
    readonly id: string;
    readonly semanticSha256: string;
  };
  readonly sources: {
    readonly baseDatasetSha256: string;
    readonly adversarialDatasetSha256: string | null;
    readonly releaseVerificationPayloadSha256: string;
    readonly releaseVerificationAttestationSha256: string;
  };
  readonly partitions: readonly TrainingPartition[];
  readonly payloadSha256: string;
}

export interface ModelProvenance {
  readonly provider: string;
  readonly name: string;
  readonly version: string;
  readonly revision: string;
  readonly artifactSha256: string;
}

export interface AdapterProvenance {
  readonly name: string;
  readonly version: string;
  readonly configurationSha256: string;
}

export interface SemantScriptTrainingEvidence {
  readonly trainingLedgerSha256: string;
  readonly artifactTrainingDatasetSha256: string;
  readonly artifactTrainingKeySha256: string;
  readonly releaseVerificationPayloadSha256: string;
  readonly releaseVerificationAttestationSha256: string;
}

export interface SemantScriptExecutionBackend {
  readonly kind: "semantscript-node";
  readonly runtime: "onnxruntime-node";
  readonly device: "cpu";
}

export interface OllamaExecutionBackend {
  readonly kind: "ollama";
  readonly serverVersion: string;
  readonly placement: "cpu" | "gpu" | "hybrid";
  readonly modelTotalBytes: number;
  readonly modelCpuBytes: number;
  readonly modelGpuBytes: number;
}

export interface AnthropicExecutionBackend {
  readonly kind: "anthropic-api";
  readonly apiVersion: "2023-06-01";
  readonly endpoint: "https://api.anthropic.com";
  readonly placement: "provider-managed";
}

export interface LayaExecutionBackend {
  readonly kind: "laya";
  readonly runtime: "python";
  readonly device: "cpu" | "cuda";
}

export type ExecutionBackendProvenance =
  | SemantScriptExecutionBackend
  | OllamaExecutionBackend
  | AnthropicExecutionBackend
  | LayaExecutionBackend;

export interface SystemProvenance {
  readonly role: BenchmarkSystemRole;
  readonly taskSpecSha256: string;
  readonly model: ModelProvenance;
  readonly adapter: AdapterProvenance;
  readonly trainingEvidence: SemantScriptTrainingEvidence | null;
  readonly executionBackend: ExecutionBackendProvenance;
}

export interface RuntimeVersion {
  readonly name: string;
  readonly version: string;
}

export interface EnvironmentProvenance {
  readonly capturedAt: string;
  readonly operatingSystem: string;
  readonly architecture: string;
  readonly cpu: string;
  readonly accelerator: string | null;
  readonly memoryBytes: number;
  readonly runtimeVersions: readonly RuntimeVersion[];
  readonly evidenceSha256: string;
}

export interface DistributionEntry {
  readonly value: RefundDecision;
  readonly probability: number;
}

export interface RefundPrediction {
  readonly caseId: string;
  readonly inputSha256: string;
  readonly value: RefundDecision;
  readonly distribution: readonly DistributionEntry[];
  readonly latencyMs: number;
}

export interface MeasurementProtocol {
  readonly concurrency: 1;
  readonly warmupIterations: number;
  readonly warmupInputOrderSha256: string;
  readonly measuredDurationMs: number;
  readonly memoryScope: MemoryScope;
  readonly peakMemoryBytes: number;
}

export interface RefundPredictionSetV1 {
  readonly kind: "semantscript.refund-benchmark-predictions";
  readonly predictionVersion: 1;
  readonly benchmark: "refund-decision";
  readonly datasetSha256: string;
  readonly support: typeof REFUND_SUPPORT;
  readonly system: SystemProvenance;
  readonly environment: EnvironmentProvenance;
  readonly protocol: MeasurementProtocol;
  readonly predictions: readonly RefundPrediction[];
  readonly payloadSha256: string;
}

export interface LeakageAudit {
  readonly datasetSha256: string;
  readonly trainingLedgerSha256: string;
  readonly evaluationCaseCount: number;
  readonly trainingInputCount: number;
  readonly overlapCount: 0;
}

export interface AccuracyMetrics {
  readonly caseCount: number;
  readonly correctCount: number;
  readonly accuracy: number;
  readonly humanCaseCount: number;
  readonly humanCorrectCount: number;
  readonly humanAccuracy: number;
}

export interface CalibrationMetrics {
  readonly binCount: 15;
  readonly expectedCalibrationError: number;
}

export interface LatencyMetrics {
  readonly p50Ms: number;
  readonly p95Ms: number;
}

export interface ThroughputMetrics {
  readonly concurrency: 1;
  readonly measuredDurationMs: number;
  readonly requestsPerSecond: number;
}

export interface MemoryMetrics {
  readonly scope: MemoryScope;
  readonly peakBytes: number;
}

export interface SystemMetrics {
  readonly accuracy: AccuracyMetrics;
  readonly calibration: CalibrationMetrics;
  readonly latency: LatencyMetrics;
  readonly throughput: ThroughputMetrics;
  readonly memory: MemoryMetrics;
}

export interface BenchmarkSystemResult {
  readonly role: BenchmarkSystemRole;
  readonly predictionSha256: string;
  readonly taskSpecSha256: string;
  readonly model: ModelProvenance;
  readonly adapter: AdapterProvenance;
  readonly trainingEvidence: SemantScriptTrainingEvidence | null;
  readonly executionBackend: ExecutionBackendProvenance;
  readonly environment: EnvironmentProvenance;
  readonly protocol: MeasurementProtocol;
  readonly metrics: SystemMetrics;
}

export interface GoNoGoDecision {
  readonly status: "go" | "no-go" | "incomplete";
  readonly criterion: {
    readonly semantscriptP50MsExclusiveMaximum: 10;
    readonly accuracyComparator: "ollama-7b";
  };
  readonly semantscriptP50Ms: number | null;
  readonly semantscriptAccuracy: number | null;
  readonly comparatorAccuracy: number | null;
  readonly latencyPassed: boolean | null;
  readonly accuracyPassed: boolean | null;
  readonly missingSystems: readonly BenchmarkSystemRole[];
}

export interface RefundBenchmarkResultV1 {
  readonly kind: "semantscript.refund-benchmark-result";
  readonly resultVersion: 1;
  readonly benchmark: "refund-decision";
  readonly generatedAt: string;
  readonly taskSpecSha256: string;
  readonly datasetSha256: string;
  readonly trainingLedgerSha256: string;
  readonly leakageAudit: LeakageAudit;
  readonly systems: readonly BenchmarkSystemResult[];
  readonly goNoGo: GoNoGoDecision;
  readonly payloadSha256: string;
}
