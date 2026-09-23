import { types as nodeTypes } from "node:util";

import type {
  ArtifactFunctionV1,
  ArtifactResourceV1,
  HeadBindingV1,
  OnnxResourceV1,
  StagedArtifactDescriptor,
  TokenizerResourceV1,
} from "./artifact-types.js";
import {
  ArtifactLoadError,
  loadArtifact,
  type ArtifactLoadOptions,
} from "./artifact-loader.js";
import {
  type CanonicalInputEntry,
  serializeCanonicalInputs,
} from "./canonical-input.js";
import {
  SemaConfidenceError,
  SemaFallbackError,
  type SemaDiagnosticResult,
  type SemaFallback,
} from "./confidence-policy.js";
import {
  type InferenceSupportValue,
  type StagedHeadPlan,
  type StagedInferencePlan,
  maximumInferenceResponseBytes,
} from "./inference-protocol.js";
import {
  createInferenceRuntime,
  type InferenceRuntime,
  type InferenceRuntimeOptions,
  SemaUnknownFunctionError,
} from "./inference-runtime.js";
import { inspectOnnxContainer } from "./onnx-model.js";

interface ActiveArtifact {
  readonly token: symbol;
  readonly manifestSha256: string;
  readonly runtime: InferenceRuntime;
  readonly functions: ReadonlyMap<string, ActiveFunction>;
  readonly fallbackInvocationStack: Set<string>;
}

interface ActiveFunction {
  readonly artifact: ArtifactFunctionV1;
  readonly fallback: SemaFallback | undefined;
}

const DEFAULT_RESPONSE_BUFFER_BYTES = 65_536;
const MAXIMUM_RESPONSE_BUFFER_BYTES = 1_048_576;
const MAXIMUM_FALLBACK_RESULT_DEPTH = 100;
const MAXIMUM_FALLBACK_RESULT_NODES = 100_000;

export interface LoadSemaArtifactOptions {
  readonly artifact?: Omit<ArtifactLoadOptions, "backend">;
  readonly inference?: InferenceRuntimeOptions;
  readonly fallbacks?: ReadonlyMap<string, SemaFallback>;
}

export interface SemaArtifactHandle {
  readonly manifestSha256: string;
  readonly functionIds: ReadonlySet<string>;
  // The caller supplies the compiled semantic function's static result type.
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-type-parameters
  call<T>(functionId: string, inputs: Readonly<Record<string, unknown>>): T;
  close(): Promise<void>;
}

export class SemaRuntimeNotLoadedError extends Error {
  readonly code = "SEMA_RUNTIME_NOT_LOADED";

  constructor() {
    super("no SemantScript artifact is loaded");
    this.name = "SemaRuntimeNotLoadedError";
  }
}

export class SemaArtifactInactiveError extends Error {
  readonly code = "SEMA_ARTIFACT_INACTIVE";
  readonly manifestSha256: string;

  constructor(manifestSha256: string) {
    super("the SemantScript artifact handle is no longer active");
    this.name = "SemaArtifactInactiveError";
    this.manifestSha256 = manifestSha256;
  }
}

let activeArtifact: ActiveArtifact | undefined;
let lifecycleTail: Promise<void> = Promise.resolve();

/**
 * Stages and initializes a complete immutable artifact, then atomically makes it
 * visible to the compiler ABI. A failed load never replaces the active artifact.
 */
export function loadSemaArtifact(
  artifactPath: string,
  options: LoadSemaArtifactOptions = {},
): Promise<SemaArtifactHandle> {
  const fallbackSnapshot = snapshotFallbacks(options.fallbacks);
  const artifactOptions = options.artifact;
  const inferenceOptions = options.inference;

  return enqueueLifecycle(async () => {
    const staged = await loadArtifact(artifactPath, artifactOptions);
    const functions = bindFunctions(staged.functions, fallbackSnapshot);
    const plan = buildInferencePlan(staged);
    const runtime = await createInferenceRuntime(
      plan,
      inferenceOptionsForPlan(plan, inferenceOptions),
    );
    const token = Symbol("active SemantScript artifact");
    const next: ActiveArtifact = {
      token,
      manifestSha256: staged.manifestSha256,
      runtime,
      functions,
      fallbackInvocationStack: new Set(),
    };
    const previous = activeArtifact;
    activeArtifact = next;

    if (previous !== undefined) {
      await previous.runtime.close().catch(() => undefined);
    }

    return createHandle(next);
  });
}

/** Closes the active artifact, if any. Primarily useful for orderly shutdown and tests. */
export function closeSemaArtifact(): Promise<void> {
  return enqueueLifecycle(async () => {
    const current = activeArtifact;
    activeArtifact = undefined;
    await current?.runtime.close();
  });
}

/** @internal Compiler-generated code calls this through the exported __sema object. */
// The compiler preserves each site's static result type across this internal ABI.
// eslint-disable-next-line @typescript-eslint/no-unnecessary-type-parameters
export function dispatchSemaCall<T>(
  functionId: string,
  inputs: Readonly<Record<string, unknown>>,
): T {
  const current = activeArtifact;
  if (current === undefined) {
    throw new SemaRuntimeNotLoadedError();
  }

  return dispatchArtifactCall(current, functionId, inputs) as T;
}

function dispatchArtifactCall(
  artifact: ActiveArtifact,
  functionId: string,
  inputs: Readonly<Record<string, unknown>>,
): unknown {
  const activeFunction = artifact.functions.get(functionId);
  if (activeFunction === undefined) {
    throw new SemaUnknownFunctionError(functionId);
  }
  const semanticFunction = activeFunction.artifact;

  const canonicalInput = serializeCanonicalInputs(
    semanticFunction.inputs satisfies readonly CanonicalInputEntry[],
    inputs,
    { maximumBytes: artifact.runtime.maximumInputBytes },
  );
  const inferenceResult = artifact.runtime.call(functionId, canonicalInput);
  const { confidenceThreshold, resultMode } = semanticFunction.runtime;

  if (resultMode === "value" && confidenceThreshold === null) {
    return inferenceResult;
  }

  const diagnostic = inferenceResult as SemaDiagnosticResult;
  if (resultMode === "diagnostic") {
    return diagnostic;
  }

  if (confidenceThreshold === null) {
    throw new Error("value-mode diagnostic inference requires a confidence threshold");
  }

  const confidence = isScalarFunction(semanticFunction)
    ? scalarDiagnostic(diagnostic).confidence
    : objectDiagnostic(diagnostic).minimumFieldConfidence;
  if (confidence >= confidenceThreshold) {
    return diagnostic.value;
  }

  if (activeFunction.fallback === undefined) {
    throw new SemaConfidenceError(functionId, confidenceThreshold, diagnostic);
  }

  const fallbackRef = semanticFunction.runtime.fallbackRef;
  if (fallbackRef === null) {
    throw new Error("a bound fallback requires a non-null fallback reference");
  }
  if (artifact.fallbackInvocationStack.has(functionId)) {
    throw new SemaFallbackError(
      "cycle",
      functionId,
      fallbackRef,
      `fallback invocation cycle reached semantic function ${JSON.stringify(functionId)}`,
    );
  }

  artifact.fallbackInvocationStack.add(functionId);
  try {
    const fallbackResult = activeFunction.fallback(inputs, diagnostic, confidenceThreshold);
    validateFallbackResult(semanticFunction, fallbackRef, fallbackResult);
    return fallbackResult;
  } finally {
    artifact.fallbackInvocationStack.delete(functionId);
  }
}

function createHandle(artifact: ActiveArtifact): SemaArtifactHandle {
  let closed = false;
  return Object.freeze({
    manifestSha256: artifact.manifestSha256,
    functionIds: new Set(artifact.runtime.functionIds),
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-type-parameters
    call<T>(functionId: string, inputs: Readonly<Record<string, unknown>>): T {
      if (closed || activeArtifact?.token !== artifact.token) {
        throw new SemaArtifactInactiveError(artifact.manifestSha256);
      }
      return dispatchArtifactCall(artifact, functionId, inputs) as T;
    },
    async close(): Promise<void> {
      if (closed) {
        return;
      }
      closed = true;
      await enqueueLifecycle(async () => {
        if (activeArtifact?.token === artifact.token) {
          activeArtifact = undefined;
        }
        await artifact.runtime.close();
      });
    },
  });
}

function enqueueLifecycle<T>(operation: () => Promise<T>): Promise<T> {
  const result = lifecycleTail.then(operation, operation);
  lifecycleTail = result.then(
    () => undefined,
    () => undefined,
  );
  return result;
}

function buildInferencePlan(staged: StagedArtifactDescriptor<Uint8Array>): StagedInferencePlan {
  const resources = new Map(
    staged.resources.map(({ metadata, prepared }) => [metadata.ref, { metadata, prepared }]),
  );
  const tokenizer = requireResource(resources, staged.manifest.model.tokenizerRef, "tokenizer");
  const encoder = requireOnnxResource(resources, staged.manifest.model.encoderRef, "encoder");
  const adapters = staged.manifest.resources
    .filter((resource) => resource.role === "adapter")
    .map((resource) => {
      const prepared = requireOnnxResource(resources, resource.ref, "adapter");
      return {
        ref: resource.ref,
        model: validatedModel(prepared),
        abi: prepared.metadata.onnx,
      };
    });
  const functions = staged.functions.map((entry) => ({
    id: entry.id,
    adapterRef: entry.adapterRef,
    diagnosticsRequired:
      entry.runtime.resultMode === "diagnostic" || entry.runtime.confidenceThreshold !== null,
    heads: entry.heads.map((head) => buildHeadPlan(resources, head)),
  }));

  return {
    kind: "onnx",
    tokenizerJson: tokenizer.prepared,
    encoderModel: validatedModel(encoder),
    encoderAbi: encoder.metadata.onnx,
    adapters,
    functions,
    maximumSequenceLength: (tokenizer.metadata as TokenizerResourceV1).maximumSequenceLength,
  };
}

function buildHeadPlan(
  resources: ReadonlyMap<
    string,
    { readonly metadata: ArtifactResourceV1; readonly prepared: Uint8Array }
  >,
  head: HeadBindingV1,
): StagedHeadPlan {
  const resource = requireResource(resources, head.headRef, "head");
  if (resource.metadata.format !== "onnx") {
    throw invalidResourceRole(head.headRef, "head");
  }
  return {
    outputPath: head.outputPath,
    parameterization: head.parameterization,
    support: headSupport(head),
    temperature: head.calibration.temperature,
    expectedValueMode: expectedValueMode(head),
    model: validatedModel({ metadata: resource.metadata, prepared: resource.prepared }),
    abi: resource.metadata.onnx,
  };
}

function expectedValueMode(
  head: HeadBindingV1,
): "none" | "zero-based-rank" | "numeric" {
  if (head.type.kind === "ordinal-string") {
    return "zero-based-rank";
  }
  if (head.type.kind === "ordinal-number") {
    return "numeric";
  }
  return "none";
}

function headSupport(head: HeadBindingV1): readonly InferenceSupportValue[] {
  if (head.type.kind === "ordinal-number") {
    return head.type.supportDecimal.map((value) => Number(value));
  }
  return head.type.support;
}

function requireResource(
  resources: ReadonlyMap<
    string,
    { readonly metadata: ArtifactResourceV1; readonly prepared: Uint8Array }
  >,
  reference: string,
  expectedRole: ArtifactResourceV1["role"],
): { readonly metadata: ArtifactResourceV1; readonly prepared: Uint8Array } {
  const resource = resources.get(reference);
  if (resource === undefined || resource.metadata.role !== expectedRole) {
    throw invalidResourceRole(reference, expectedRole);
  }
  return resource;
}

function requireOnnxResource(
  resources: ReadonlyMap<
    string,
    { readonly metadata: ArtifactResourceV1; readonly prepared: Uint8Array }
  >,
  reference: string,
  expectedRole: OnnxResourceV1["role"],
): { readonly metadata: OnnxResourceV1; readonly prepared: Uint8Array } {
  const resource = requireResource(resources, reference, expectedRole);
  if (resource.metadata.format !== "onnx") {
    throw invalidResourceRole(reference, expectedRole);
  }
  return { metadata: resource.metadata, prepared: resource.prepared };
}

function validatedModel(resource: {
  readonly metadata: OnnxResourceV1;
  readonly prepared: Uint8Array;
}): Uint8Array {
  try {
    const inspection = inspectOnnxContainer(resource.prepared);
    if (inspection.defaultOpset !== resource.metadata.onnx.opset) {
      throw new TypeError(
        `protobuf imports opset ${String(inspection.defaultOpset)} but manifest declares ${String(resource.metadata.onnx.opset)}`,
      );
    }
  } catch (error) {
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_INVALID_MANIFEST",
      `ONNX resource ${JSON.stringify(resource.metadata.ref)} failed container validation`,
      resource.metadata.path,
      error,
    );
  }
  return resource.prepared;
}

function invalidResourceRole(
  reference: string,
  expectedRole: ArtifactResourceV1["role"],
): ArtifactLoadError {
  return new ArtifactLoadError(
    "SEMA_ARTIFACT_INVALID_MANIFEST",
    `resource ${JSON.stringify(reference)} must resolve to role ${expectedRole}`,
  );
}

function inferenceOptionsForPlan(
  plan: StagedInferencePlan,
  options: InferenceRuntimeOptions | undefined,
): InferenceRuntimeOptions {
  let requiredBytes = 1;
  for (const semanticFunction of plan.functions) {
    requiredBytes = Math.max(
      requiredBytes,
      maximumInferenceResponseBytes(semanticFunction, MAXIMUM_RESPONSE_BUFFER_BYTES),
    );
  }
  if (requiredBytes > MAXIMUM_RESPONSE_BUFFER_BYTES) {
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_QUOTA",
      `artifact output requires ${String(requiredBytes)} response bytes; maximum is ${String(MAXIMUM_RESPONSE_BUFFER_BYTES)}`,
    );
  }
  if (options?.responseBufferBytes !== undefined && options.responseBufferBytes < requiredBytes) {
    throw new RangeError(
      `responseBufferBytes must be at least ${String(requiredBytes)} for this artifact`,
    );
  }
  return {
    ...options,
    responseBufferBytes:
      options?.responseBufferBytes ?? Math.max(DEFAULT_RESPONSE_BUFFER_BYTES, requiredBytes),
  };
}

function snapshotFallbacks(
  fallbacks: ReadonlyMap<string, SemaFallback> | undefined,
): ReadonlyMap<string, SemaFallback> {
  const snapshot = new Map<string, SemaFallback>();
  if (fallbacks === undefined) {
    return snapshot;
  }
  for (const [reference, callback] of fallbacks) {
    if (typeof reference !== "string" || reference.length === 0) {
      throw new TypeError("fallback references must be non-empty strings");
    }
    if (typeof callback !== "function") {
      throw new TypeError(`fallback ${JSON.stringify(reference)} must be a function`);
    }
    snapshot.set(reference, callback);
  }
  return snapshot;
}

function bindFunctions(
  functions: readonly ArtifactFunctionV1[],
  fallbacks: ReadonlyMap<string, SemaFallback>,
): ReadonlyMap<string, ActiveFunction> {
  return new Map(
    functions.map((artifact): readonly [string, ActiveFunction] => {
      const fallbackRef = artifact.runtime.fallbackRef;
      const fallback = fallbackRef === null ? undefined : fallbacks.get(fallbackRef);
      if (fallbackRef !== null && fallback === undefined) {
        throw new SemaFallbackError(
          "missing",
          artifact.id,
          fallbackRef,
          `semantic function ${JSON.stringify(artifact.id)} requires unregistered fallback ${JSON.stringify(fallbackRef)}`,
        );
      }
      return [artifact.id, Object.freeze({ artifact, fallback })];
    }),
  );
}

function validateFallbackResult(
  semanticFunction: ArtifactFunctionV1,
  fallbackRef: string,
  result: unknown,
): void {
  assertAcyclicFallbackResult(
    result,
    semanticFunction.id,
    fallbackRef,
    new Set(),
    new Set(),
    { remaining: MAXIMUM_FALLBACK_RESULT_NODES },
    0,
  );

  if (isScalarFunction(semanticFunction)) {
    const head = semanticFunction.heads[0];
    if (head === undefined || !supportContains(head, result)) {
      invalidFallbackResult(semanticFunction.id, fallbackRef, "result is outside scalar output support");
    }
    return;
  }

  if (result === null || typeof result !== "object" || Array.isArray(result)) {
    invalidFallbackResult(semanticFunction.id, fallbackRef, "flat result must be a plain object");
  }
  if (nodeTypes.isProxy(result) || nodeTypes.isPromise(result)) {
    invalidFallbackResult(semanticFunction.id, fallbackRef, "flat result cannot be a proxy or promise");
  }
  const prototype = Object.getPrototypeOf(result) as unknown;
  if (prototype !== Object.prototype && prototype !== null) {
    invalidFallbackResult(semanticFunction.id, fallbackRef, "flat result must be a plain object");
  }

  const descriptors = Object.getOwnPropertyDescriptors(result);
  const actualNames = Object.getOwnPropertyNames(result);
  const expectedNames = new Set(
    semanticFunction.heads.map((head) => head.outputPath[0] ?? missingFlatField(semanticFunction.id)),
  );
  for (const name of actualNames) {
    if (!expectedNames.has(name)) {
      invalidFallbackResult(
        semanticFunction.id,
        fallbackRef,
        `flat result has unexpected field ${JSON.stringify(name)}`,
      );
    }
  }
  for (const head of semanticFunction.heads) {
    const field = head.outputPath[0] ?? missingFlatField(semanticFunction.id);
    const descriptor = descriptors[field];
    if (!descriptor?.enumerable || !("value" in descriptor)) {
      invalidFallbackResult(
        semanticFunction.id,
        fallbackRef,
        `flat result is missing enumerable data field ${JSON.stringify(field)}`,
      );
    }
    if (!supportContains(head, descriptor.value)) {
      invalidFallbackResult(
        semanticFunction.id,
        fallbackRef,
        `flat result field ${JSON.stringify(field)} is outside output support`,
      );
    }
  }
}

function assertAcyclicFallbackResult(
  value: unknown,
  functionId: string,
  fallbackRef: string,
  active: Set<object>,
  completed: Set<object>,
  budget: { remaining: number },
  depth: number,
): void {
  if ((typeof value !== "object" || value === null) && typeof value !== "function") {
    return;
  }
  if (depth > MAXIMUM_FALLBACK_RESULT_DEPTH || budget.remaining <= 0) {
    invalidFallbackResult(functionId, fallbackRef, "result exceeds validation limits");
  }
  budget.remaining -= 1;
  if (nodeTypes.isProxy(value)) {
    invalidFallbackResult(functionId, fallbackRef, "result cannot contain proxies");
  }
  if (nodeTypes.isPromise(value)) {
    invalidFallbackResult(functionId, fallbackRef, "result cannot be a promise");
  }
  if (active.has(value)) {
    invalidFallbackResult(functionId, fallbackRef, "result cannot contain cycles");
  }
  if (completed.has(value)) {
    return;
  }

  active.add(value);
  try {
    if (Object.getOwnPropertySymbols(value).length > 0) {
      invalidFallbackResult(functionId, fallbackRef, "result cannot contain symbol-keyed properties");
    }
    for (const descriptor of Object.values(Object.getOwnPropertyDescriptors(value))) {
      if (!("value" in descriptor)) {
        invalidFallbackResult(functionId, fallbackRef, "result cannot contain accessors");
      }
      assertAcyclicFallbackResult(
        descriptor.value,
        functionId,
        fallbackRef,
        active,
        completed,
        budget,
        depth + 1,
      );
    }
  } finally {
    active.delete(value);
  }
  completed.add(value);
}

function supportContains(head: HeadBindingV1, value: unknown): boolean {
  return headSupport(head).some((candidate) => Object.is(candidate, value));
}

function isScalarFunction(semanticFunction: ArtifactFunctionV1): boolean {
  return semanticFunction.heads.length === 1 && semanticFunction.heads[0]?.outputPath.length === 0;
}

function scalarDiagnostic(diagnostic: SemaDiagnosticResult): {
  readonly confidence: number;
} {
  return diagnostic as { readonly confidence: number };
}

function objectDiagnostic(diagnostic: SemaDiagnosticResult): {
  readonly minimumFieldConfidence: number;
} {
  return diagnostic as { readonly minimumFieldConfidence: number };
}

function invalidFallbackResult(functionId: string, fallbackRef: string, detail: string): never {
  throw new SemaFallbackError(
    "invalid-result",
    functionId,
    fallbackRef,
    `fallback ${JSON.stringify(fallbackRef)} returned an invalid result: ${detail}`,
  );
}

function missingFlatField(functionId: string): never {
  throw new Error(`semantic function ${JSON.stringify(functionId)} has an invalid flat output path`);
}
