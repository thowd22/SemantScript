import type {
  LayaExecutionBackend,
  ModelProvenance,
  RefundInputs,
} from "../types.js";
import { REFUND_SUPPORT } from "../types.js";
import { REFUND_SYSTEM_PINS, REFUND_TASK_SPEC_SHA256 } from "../policy.js";
import {
  DEFAULT_BASELINE_TIMEOUT_MS,
  REFUND_PROMPT_VERSION,
  BaselineAdapterError,
  adapterProvenance,
  assertExactRevision,
  buildRefundPrompt,
  invokeWithTimeout,
  pinnedModelProvenance,
  predictionFromLogits,
  responseObject,
  validateTimeout,
  type BaselinePrediction,
  type RefundBaselineAdapter,
} from "./common.js";

export const LAYA_CHECKPOINT = REFUND_SYSTEM_PINS.laya.model.name;
export const LAYA_CHECKPOINT_REVISION = REFUND_SYSTEM_PINS.laya.model.revision;
export const LAYA_CHECKPOINT_SHA256 = REFUND_SYSTEM_PINS.laya.model.artifactSha256;
export const LAYA_CODE_REVISION = "d120d4ba220711b93c171973118753460310e16b" as const;

export interface LayaChoiceRequest {
  readonly checkpoint: typeof LAYA_CHECKPOINT;
  readonly checkpointRevision: typeof LAYA_CHECKPOINT_REVISION;
  readonly codeRevision: typeof LAYA_CODE_REVISION;
  readonly questionType: "choice";
  readonly question: string;
  readonly options: typeof REFUND_SUPPORT;
  readonly applyCheckpointCalibration: true;
}

export interface LayaChoiceResponse {
  readonly checkpointRevision: string;
  readonly selectedIndex: number;
  readonly logits: readonly number[];
}

export interface LayaRunner {
  readonly device: "cpu" | "cuda";
  choose(request: LayaChoiceRequest, signal: AbortSignal): Promise<LayaChoiceResponse>;
}

export interface LayaAdapterOptions {
  readonly model: ModelProvenance;
  readonly runner: LayaRunner;
  readonly timeoutMs?: number;
}

export function createLayaAdapter(
  options: LayaAdapterOptions,
): RefundBaselineAdapter<"laya"> {
  const timeoutMs = validateTimeout(options.timeoutMs ?? DEFAULT_BASELINE_TIMEOUT_MS);
  const model = pinnedModelProvenance(options.model, {
    provider: "huggingface",
    name: LAYA_CHECKPOINT,
    version: "laya-typed-decisions",
    revision: LAYA_CHECKPOINT_REVISION,
  });
  if (model.artifactSha256 !== LAYA_CHECKPOINT_SHA256) {
    throw new BaselineAdapterError(
      "invalid-configuration",
      `model.artifactSha256 must pin the Laya model checkpoint ${LAYA_CHECKPOINT_SHA256}`,
    );
  }
  const configuration = Object.freeze({
    provider: "laya",
    role: "laya",
    model,
    promptVersion: REFUND_PROMPT_VERSION,
    taskSpecSha256: REFUND_TASK_SPEC_SHA256,
    timeoutMs,
    codeRevision: LAYA_CODE_REVISION,
    questionType: "choice",
    support: REFUND_SUPPORT,
    applyCheckpointCalibration: true,
  });
  const provenance = adapterProvenance("refund-laya-typed-decisions", configuration);

  return Object.freeze({
    role: "laya",
    model,
    adapter: provenance,
    taskSpecSha256: REFUND_TASK_SPEC_SHA256,
    trainingEvidence: null,
    resolveExecutionBackend(): Promise<LayaExecutionBackend> {
      return Promise.resolve(Object.freeze({
        kind: "laya",
        runtime: "python",
        device: options.runner.device,
      }));
    },
    async predict(inputs: RefundInputs, signal?: AbortSignal): Promise<BaselinePrediction> {
      const prompt = buildRefundPrompt(inputs);
      const request: LayaChoiceRequest = Object.freeze({
        checkpoint: LAYA_CHECKPOINT,
        checkpointRevision: LAYA_CHECKPOINT_REVISION,
        codeRevision: LAYA_CODE_REVISION,
        questionType: "choice",
        question: `${prompt.system}\n${prompt.user}`,
        options: REFUND_SUPPORT,
        applyCheckpointCalibration: true,
      });
      const response = await invokeWithTimeout(
        "Laya typed-decisions",
        timeoutMs,
        signal,
        async (runnerSignal) => options.runner.choose(request, runnerSignal),
      );
      const responseRecord = responseObject(response, [
        "checkpointRevision",
        "selectedIndex",
        "logits",
      ]);
      assertExactRevision(responseRecord["checkpointRevision"], LAYA_CHECKPOINT_REVISION);
      return predictionFromLogits(
        responseRecord["selectedIndex"],
        responseRecord["logits"],
      );
    },
  });
}
