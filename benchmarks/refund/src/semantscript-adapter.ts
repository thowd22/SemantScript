import { createHash } from "node:crypto";
import { open } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

import { semanticJsonSha256 } from "@semantscript/compiler";
import { loadSemaArtifact } from "@semantscript/core";

import type { BaselinePrediction } from "./adapters/common.js";
import {
  REFUND_FUNCTION_ID,
  REFUND_FUNCTION_SEMANTIC_SHA256,
  REFUND_SYSTEM_PINS,
  REFUND_TASK_SPEC_SHA256,
} from "./policy.js";
import type { RefundPredictionAdapter } from "./runner.js";
import {
  REFUND_SUPPORT,
  type AdapterProvenance,
  type ModelProvenance,
  type RefundDecision,
  type RefundInputs,
  type SemantScriptTrainingEvidence,
} from "./types.js";

const SHA256 = /^[a-f0-9]{64}$/u;
const FUNCTION_ID = /^nf_[a-f0-9]{64}$/u;
const MAXIMUM_BENCHMARK_MANIFEST_BYTES = 8 * 1024 * 1024;

export const SEMANTSCRIPT_BENCHMARK_ADAPTER_VERSION = "1" as const;

export type SemantScriptAdapterErrorCode =
  | "aborted"
  | "closed"
  | "invalid-configuration"
  | "invalid-output"
  | "provenance-mismatch";

export class SemantScriptAdapterError extends Error {
  readonly code: SemantScriptAdapterErrorCode;

  constructor(
    code: SemantScriptAdapterErrorCode,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "SemantScriptAdapterError";
    this.code = code;
  }
}

export interface RuntimeFunctionProvenance {
  readonly id: string;
  readonly semanticSha256: string;
  readonly resultMode: "value" | "diagnostic";
  readonly trainingProvenance: {
    readonly datasetSha256: string;
    readonly trainingKeySha256: string;
    readonly baseModel: string;
  };
}

export interface LoadedSemantScriptArtifact {
  readonly manifestSha256: string;
  readonly functionIds: ReadonlySet<string>;
  readonly functions: readonly RuntimeFunctionProvenance[];
  call(functionId: string, inputs: RefundInputs): unknown;
  close(): Promise<void>;
}

export interface SemantScriptArtifactLoader {
  load(artifactRoot: string): Promise<LoadedSemantScriptArtifact>;
}

export interface ExpectedSemantScriptProvenance {
  readonly taskSpecSha256: string;
  readonly function: RuntimeFunctionProvenance;
  readonly model: ModelProvenance;
  readonly trainingEvidence: SemantScriptTrainingEvidence;
}

export interface SemantScriptRefundAdapterOptions {
  readonly artifactRoot: string;
  readonly expected: ExpectedSemantScriptProvenance;
  readonly loader?: SemantScriptArtifactLoader;
}

export interface SemantScriptRefundAdapter extends RefundPredictionAdapter<"semantscript"> {
  readonly taskSpecSha256: string;
  close(): Promise<void>;
}

export const nodeSemantScriptArtifactLoader: SemantScriptArtifactLoader =
  Object.freeze({
    async load(artifactRoot: string): Promise<LoadedSemantScriptArtifact> {
      const handle = await loadSemaArtifact(artifactRoot);
      try {
        const functions = await readLoadedFunctionProvenance(
          artifactRoot,
          handle.manifestSha256,
        );
        for (const entry of functions) {
          if (!handle.functionIds.has(entry.id)) {
            mismatch(
              `loaded manifest function ${JSON.stringify(entry.id)} is absent from runtime`,
            );
          }
        }
        if (handle.functionIds.size !== functions.length) {
          mismatch(
            "runtime and loaded manifest expose different function sets",
          );
        }
        return Object.freeze({
          manifestSha256: handle.manifestSha256,
          functionIds: new Set(handle.functionIds),
          functions,
          call(functionId: string, inputs: RefundInputs): unknown {
            return handle.call(
              functionId,
              inputs as unknown as Readonly<Record<string, unknown>>,
            );
          },
          close: async (): Promise<void> => handle.close(),
        });
      } catch (error) {
        await handle.close().catch(() => undefined);
        throw error;
      }
    },
  });

export async function createSemantScriptRefundAdapter(
  options: SemantScriptRefundAdapterOptions,
): Promise<SemantScriptRefundAdapter> {
  const configuration = validateConfiguration(options);
  const loaded = await configuration.loader.load(configuration.artifactRoot);
  try {
    validateLoadedArtifact(loaded, configuration.expected);
  } catch (error) {
    await loaded.close().catch(() => undefined);
    throw error;
  }

  let closing = false;
  let activeCalls = 0;
  let idleResolve: (() => void) | undefined;
  let closePromise: Promise<void> | undefined;
  const adapterProvenance: AdapterProvenance = Object.freeze({
    name: "semantscript-node-artifact-runtime",
    version: SEMANTSCRIPT_BENCHMARK_ADAPTER_VERSION,
    configurationSha256: semanticJsonSha256({
      manifestSha256: loaded.manifestSha256,
      taskSpecSha256: configuration.expected.taskSpecSha256,
      function: configuration.expected.function,
      model: configuration.expected.model,
      trainingEvidence: configuration.expected.trainingEvidence,
    }),
  });

  const waitForIdle = (): Promise<void> => {
    if (activeCalls === 0) return Promise.resolve();
    return new Promise((resolveIdle) => {
      idleResolve = resolveIdle;
    });
  };

  return Object.freeze({
    role: "semantscript",
    model: configuration.expected.model,
    adapter: adapterProvenance,
    taskSpecSha256: configuration.expected.taskSpecSha256,
    trainingEvidence: configuration.expected.trainingEvidence,
    resolveExecutionBackend() {
      return Promise.resolve(
        Object.freeze({
          kind: "semantscript-node" as const,
          runtime: "onnxruntime-node" as const,
          device: "cpu" as const,
        }),
      );
    },
    async predict(
      inputs: RefundInputs,
      signal?: AbortSignal,
    ): Promise<BaselinePrediction> {
      if (closing) {
        throw new SemantScriptAdapterError(
          "closed",
          "SemantScript adapter is closed",
        );
      }
      throwIfAborted(signal);
      activeCalls += 1;
      try {
        const diagnostic = await loaded.call(
          configuration.expected.function.id,
          inputs,
        );
        throwIfAborted(signal);
        return validateDiagnostic(diagnostic);
      } finally {
        activeCalls -= 1;
        if (activeCalls === 0) {
          idleResolve?.();
          idleResolve = undefined;
        }
      }
    },
    close(): Promise<void> {
      closing = true;
      closePromise ??= (async () => {
        await waitForIdle();
        await loaded.close();
      })();
      return closePromise;
    },
  });
}

interface ValidatedConfiguration {
  readonly artifactRoot: string;
  readonly expected: Readonly<{
    readonly taskSpecSha256: string;
    readonly function: RuntimeFunctionProvenance;
    readonly model: ModelProvenance;
    readonly trainingEvidence: SemantScriptTrainingEvidence;
  }>;
  readonly loader: SemantScriptArtifactLoader;
}

function validateConfiguration(options: unknown): ValidatedConfiguration {
  if (
    options === null ||
    typeof options !== "object" ||
    Array.isArray(options)
  ) {
    configurationError("options must be an object");
  }
  const raw = options as Record<string, unknown>;
  const artifactRoot = raw["artifactRoot"];
  if (typeof artifactRoot !== "string" || artifactRoot.length === 0) {
    configurationError("artifactRoot must be a nonempty string");
  }
  const expected = requireRecord(raw["expected"], "expected");
  const taskSpecSha256 = requireSha256(
    expected["taskSpecSha256"],
    "expected.taskSpecSha256",
  );
  if (taskSpecSha256 !== REFUND_TASK_SPEC_SHA256) {
    mismatch(
      "expected task specification does not match the canonical refund task",
    );
  }
  const expectedFunction = validateFunctionProvenance(
    expected["function"],
    "expected.function",
  );
  if (
    expectedFunction.id !== REFUND_FUNCTION_ID ||
    expectedFunction.semanticSha256 !== REFUND_FUNCTION_SEMANTIC_SHA256 ||
    expectedFunction.resultMode !== "diagnostic"
  ) {
    mismatch(
      "expected function does not match the canonical diagnostic refund function",
    );
  }
  const model = validateModelProvenance(expected["model"], expectedFunction);
  const trainingEvidence = validateTrainingEvidence(
    expected["trainingEvidence"],
  );
  if (
    trainingEvidence.artifactTrainingDatasetSha256 !==
    expectedFunction.trainingProvenance.datasetSha256
  ) {
    mismatch(
      "expected training evidence dataset must equal the expected function training dataset",
    );
  }
  if (
    trainingEvidence.artifactTrainingKeySha256 !==
    expectedFunction.trainingProvenance.trainingKeySha256
  ) {
    mismatch(
      "expected training evidence key must equal the expected function training key",
    );
  }
  const loader = validateLoader(
    raw["loader"] ?? nodeSemantScriptArtifactLoader,
  );
  return Object.freeze({
    artifactRoot,
    expected: Object.freeze({
      taskSpecSha256,
      function: expectedFunction,
      model,
      trainingEvidence,
    }),
    loader,
  });
}

function validateLoadedArtifact(
  loaded: LoadedSemantScriptArtifact,
  expected: ValidatedConfiguration["expected"],
): void {
  if (loaded.manifestSha256 !== expected.model.artifactSha256) {
    mismatch("loaded artifact manifest does not match model.artifactSha256");
  }
  if (!loaded.functionIds.has(expected.function.id)) {
    mismatch("loaded artifact does not expose the expected refund function ID");
  }
  const matches = loaded.functions.filter(
    ({ id }) => id === expected.function.id,
  );
  if (matches.length !== 1) {
    mismatch(
      "loaded artifact must contain exactly one expected refund function",
    );
  }
  const actual = matches[0];
  if (actual === undefined) {
    mismatch("loaded artifact omitted the expected refund function provenance");
  }
  if (
    actual.trainingProvenance.datasetSha256 !==
    expected.trainingEvidence.artifactTrainingDatasetSha256
  ) {
    mismatch(
      "loaded refund function training dataset differs from training evidence",
    );
  }
  if (
    actual.trainingProvenance.trainingKeySha256 !==
    expected.trainingEvidence.artifactTrainingKeySha256
  ) {
    mismatch(
      "loaded refund function training key differs from training evidence",
    );
  }
  if (!sameFunctionProvenance(actual, expected.function)) {
    mismatch(
      "loaded refund function provenance differs from the expected export",
    );
  }
  if (actual.resultMode !== "diagnostic") {
    mismatch("loaded refund function must use diagnostic result mode");
  }
}

function sameFunctionProvenance(
  actual: RuntimeFunctionProvenance,
  expected: RuntimeFunctionProvenance,
): boolean {
  return (
    actual.id === expected.id &&
    actual.semanticSha256 === expected.semanticSha256 &&
    actual.resultMode === expected.resultMode &&
    actual.trainingProvenance.datasetSha256 ===
      expected.trainingProvenance.datasetSha256 &&
    actual.trainingProvenance.trainingKeySha256 ===
      expected.trainingProvenance.trainingKeySha256 &&
    actual.trainingProvenance.baseModel ===
      expected.trainingProvenance.baseModel
  );
}

function validateFunctionProvenance(
  value: unknown,
  path: string,
): RuntimeFunctionProvenance {
  const record = requireRecord(value, path);
  const id = record["id"];
  if (typeof id !== "string" || !FUNCTION_ID.test(id)) {
    configurationError(`${path}.id must be a neural-function ID`);
  }
  const semanticSha256 = requireSha256(
    record["semanticSha256"],
    `${path}.semanticSha256`,
  );
  const resultMode = record["resultMode"];
  if (resultMode !== "value" && resultMode !== "diagnostic") {
    configurationError(`${path}.resultMode must be value or diagnostic`);
  }
  const training = requireRecord(
    record["trainingProvenance"],
    `${path}.trainingProvenance`,
  );
  const datasetSha256 = requireSha256(
    training["datasetSha256"],
    `${path}.trainingProvenance.datasetSha256`,
  );
  const trainingKeySha256 = requireSha256(
    training["trainingKeySha256"],
    `${path}.trainingProvenance.trainingKeySha256`,
  );
  const baseModel = training["baseModel"];
  if (typeof baseModel !== "string" || baseModel.length === 0) {
    configurationError(`${path}.trainingProvenance.baseModel must be nonempty`);
  }
  return Object.freeze({
    id,
    semanticSha256,
    resultMode,
    trainingProvenance: Object.freeze({
      datasetSha256,
      trainingKeySha256,
      baseModel,
    }),
  });
}

function validateModelProvenance(
  value: unknown,
  expectedFunction: RuntimeFunctionProvenance,
): ModelProvenance {
  const record = requireRecord(value, "expected.model");
  const fields = ["provider", "name", "version"] as const;
  const strings: Record<(typeof fields)[number], string> = {
    provider: "",
    name: "",
    version: "",
  };
  const pin = REFUND_SYSTEM_PINS.semantscript.model;
  for (const field of fields) {
    const entry = record[field];
    if (entry !== pin[field]) {
      configurationError(
        `expected.model.${field} must be ${JSON.stringify(pin[field])}`,
      );
    }
    strings[field] = pin[field];
  }
  const revision = requireSha256(record["revision"], "expected.model.revision");
  if (revision !== expectedFunction.trainingProvenance.trainingKeySha256) {
    mismatch("expected model revision must equal the artifact training key");
  }
  const artifactSha256 = requireSha256(
    record["artifactSha256"],
    "expected.model.artifactSha256",
  );
  return Object.freeze({ ...strings, revision, artifactSha256 });
}

function validateTrainingEvidence(
  value: unknown,
): SemantScriptTrainingEvidence {
  const record = requireRecord(value, "expected.trainingEvidence");
  const keys = [
    "trainingLedgerSha256",
    "artifactTrainingDatasetSha256",
    "artifactTrainingKeySha256",
    "releaseVerificationPayloadSha256",
    "releaseVerificationAttestationSha256",
  ] as const;
  const names = Object.keys(record);
  if (
    names.length !== keys.length ||
    keys.some((key) => !names.includes(key))
  ) {
    configurationError(
      `expected.trainingEvidence must contain exactly ${keys.join(", ")}`,
    );
  }
  return Object.freeze({
    trainingLedgerSha256: requireSha256(
      record["trainingLedgerSha256"],
      "expected.trainingEvidence.trainingLedgerSha256",
    ),
    artifactTrainingDatasetSha256: requireSha256(
      record["artifactTrainingDatasetSha256"],
      "expected.trainingEvidence.artifactTrainingDatasetSha256",
    ),
    artifactTrainingKeySha256: requireSha256(
      record["artifactTrainingKeySha256"],
      "expected.trainingEvidence.artifactTrainingKeySha256",
    ),
    releaseVerificationPayloadSha256: requireSha256(
      record["releaseVerificationPayloadSha256"],
      "expected.trainingEvidence.releaseVerificationPayloadSha256",
    ),
    releaseVerificationAttestationSha256: requireSha256(
      record["releaseVerificationAttestationSha256"],
      "expected.trainingEvidence.releaseVerificationAttestationSha256",
    ),
  });
}

function validateLoader(value: unknown): SemantScriptArtifactLoader {
  if (value === null || typeof value !== "object" || !("load" in value)) {
    configurationError("loader must provide load()");
  }
  if (typeof value.load !== "function")
    configurationError("loader.load must be a function");
  return value as SemantScriptArtifactLoader;
}

function validateDiagnostic(value: unknown): BaselinePrediction {
  const diagnostic = requireExactRecord(value, "diagnostic", [
    "value",
    "confidence",
    "uncertainty",
    "distribution",
    "expectedValue",
  ]);
  const decision = diagnostic["value"];
  if (!isRefundDecision(decision))
    outputError("diagnostic.value is outside refund support");
  const confidence = probability(
    diagnostic["confidence"],
    "diagnostic.confidence",
  );
  probability(diagnostic["uncertainty"], "diagnostic.uncertainty");
  if (diagnostic["expectedValue"] !== null) {
    outputError(
      "diagnostic.expectedValue must be null for nominal refund output",
    );
  }
  const entries = diagnostic["distribution"];
  if (!Array.isArray(entries) || entries.length !== REFUND_SUPPORT.length) {
    outputError("diagnostic.distribution must contain complete refund support");
  }
  const distribution = REFUND_SUPPORT.map((supportValue, index) => {
    const entry = requireExactRecord(
      entries[index],
      `diagnostic.distribution[${String(index)}]`,
      ["value", "probability"],
    );
    if (entry["value"] !== supportValue) {
      outputError("diagnostic.distribution must be in declared support order");
    }
    return Object.freeze({
      value: supportValue,
      probability: probability(
        entry["probability"],
        `diagnostic.distribution[${String(index)}].probability`,
      ),
    });
  });
  const sum = distribution.reduce(
    (total, entry) => total + entry.probability,
    0,
  );
  if (Math.abs(sum - 1) > 1e-12)
    outputError("diagnostic probabilities must sum to one");
  let bestIndex = 0;
  for (let index = 1; index < distribution.length; index += 1) {
    if (
      (distribution[index]?.probability ?? -1) >
      (distribution[bestIndex]?.probability ?? -1)
    ) {
      bestIndex = index;
    }
  }
  const best = distribution[bestIndex];
  if (best === undefined || decision !== best.value) {
    outputError("diagnostic.value must be the stable support-order argmax");
  }
  if (Math.abs(confidence - best.probability) > Number.EPSILON * 8) {
    outputError("diagnostic.confidence must equal the top probability");
  }
  return Object.freeze({
    value: decision,
    distribution: Object.freeze(distribution),
  });
}

async function readLoadedFunctionProvenance(
  artifactRoot: string,
  manifestSha256: string,
): Promise<readonly RuntimeFunctionProvenance[]> {
  if (!SHA256.test(manifestSha256))
    mismatch("runtime returned an invalid manifest digest");
  const absoluteRoot = resolve(artifactRoot);
  const releaseName = `sha256-${manifestSha256}`;
  const releaseDirectory =
    basename(absoluteRoot) === releaseName
      ? absoluteRoot
      : join(absoluteRoot, "releases", releaseName);
  const manifestPath = join(releaseDirectory, "manifest.json");
  const handle = await open(manifestPath, "r");
  let bytes: Buffer;
  try {
    const stats = await handle.stat();
    if (
      !stats.isFile() ||
      stats.size < 1 ||
      stats.size > MAXIMUM_BENCHMARK_MANIFEST_BYTES
    ) {
      mismatch("loaded artifact manifest has an invalid size");
    }
    bytes = await handle.readFile();
  } finally {
    await handle.close();
  }
  const digest = createHash("sha256").update(bytes).digest("hex");
  if (digest !== manifestSha256)
    mismatch("loaded artifact manifest changed after runtime load");
  const parsed = JSON.parse(bytes.toString("utf8")) as unknown;
  const manifest = requireRecord(parsed, "manifest");
  const functions = manifest["functions"];
  if (!Array.isArray(functions))
    mismatch("loaded artifact manifest has no function list");
  return Object.freeze(
    functions.map((entry, index) => {
      const fn = requireRecord(entry, `manifest.functions[${String(index)}]`);
      const runtime = requireRecord(
        fn["runtime"],
        `manifest.functions[${String(index)}].runtime`,
      );
      const training = requireRecord(
        fn["trainingProvenance"],
        `manifest.functions[${String(index)}].trainingProvenance`,
      );
      return validateFunctionProvenance(
        {
          id: fn["id"],
          semanticSha256: fn["semanticSha256"],
          resultMode: runtime["resultMode"],
          trainingProvenance: {
            datasetSha256: training["datasetSha256"],
            trainingKeySha256: training["trainingKeySha256"],
            baseModel: training["baseModel"],
          },
        },
        `manifest.functions[${String(index)}]`,
      );
    }),
  );
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    configurationError(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireExactRecord(
  value: unknown,
  path: string,
  keys: readonly string[],
): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    outputError(`${path} must be an object`);
  }
  const record = value as Record<string, unknown>;
  const actualKeys = Object.keys(record);
  if (
    actualKeys.length !== keys.length ||
    keys.some((key) => !actualKeys.includes(key))
  ) {
    outputError(`${path} must contain exactly ${keys.join(", ")}`);
  }
  return record;
}

function requireSha256(value: unknown, path: string): string {
  if (typeof value !== "string" || !SHA256.test(value)) {
    configurationError(`${path} must be 64 lowercase hexadecimal characters`);
  }
  return value;
}

function probability(value: unknown, path: string): number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < 0 ||
    value > 1
  ) {
    outputError(`${path} must be a finite probability`);
  }
  return value;
}

function isRefundDecision(value: unknown): value is RefundDecision {
  return value === "approve" || value === "deny" || value === "review";
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    throw new SemantScriptAdapterError(
      "aborted",
      "SemantScript inference was aborted",
    );
  }
}

function configurationError(message: string): never {
  throw new SemantScriptAdapterError("invalid-configuration", message);
}

function mismatch(message: string): never {
  throw new SemantScriptAdapterError("provenance-mismatch", message);
}

function outputError(message: string): never {
  throw new SemantScriptAdapterError("invalid-output", message);
}
