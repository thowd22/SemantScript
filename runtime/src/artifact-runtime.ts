import { AsyncLocalStorage } from "node:async_hooks";
import { statSync, watch } from "node:fs";
import { dirname, join, resolve } from "node:path";
import process from "node:process";
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
  type CanonicalInputVersion,
  serializeCanonicalInputs,
  canonicalInputVersion as canonicalInputVersionOf,
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
  readonly canonicalInputVersion: CanonicalInputVersion;
  readonly runtime: InferenceRuntime;
  readonly functions: ReadonlyMap<string, ActiveFunction>;
  readonly fallbackInvocationStack: Set<string>;
  /** `diagnostics: "always"`: every function computes its distribution. */
  readonly diagnosticsAlways: boolean;
  readonly observe: SemaCallObserver | undefined;
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
  /**
   * Development hot-swap: watch the artifact root's `current.json` and reload
   * when it changes. A reload that fails to load leaves the previous artifact
   * active, so a release only replaces the running one once it loads whole.
   */
  readonly watch?: boolean;
  /** Called after each successful watched reload with the new handle. */
  readonly onReload?: (handle: SemaArtifactHandle) => void;
  /** Called when a watched reload fails; the previous artifact stays active. */
  readonly onReloadError?: (error: unknown) => void;
  /**
   * `"always"` computes the calibrated distribution for every function, not
   * only for `sema.withConfidence` sites and `@confidence` thresholds. Program
   * code still receives exactly what it would otherwise (the plain value from a
   * value-mode site); the distribution reaches `observe`. Meant for debugging
   * (`semantscript explain`): it costs the distribution's response bytes on
   * every call.
   */
  readonly diagnostics?: "always";
  /**
   * Called synchronously for every dispatched sema call (single calls and
   * stage entries) with the function id, the inputs and, when the function
   * computes one, its diagnostic result, before the confidence policy runs; a
   * call whose id the artifact lacks is reported with `kind: "missing"` just
   * before `SemaUnknownFunctionError` is thrown. An observer that throws makes
   * the call throw.
   */
  readonly observe?: SemaCallObserver;
}

/** What `LoadSemaArtifactOptions.observe` receives for one dispatched call. */
export type SemaCallObservation =
  | {
      readonly kind: "answered";
      readonly functionId: string;
      readonly inputs: Readonly<Record<string, unknown>>;
      readonly resultMode: "value" | "diagnostic";
      readonly confidenceThreshold: number | null;
      /** The calibrated result; undefined when the function computed no distribution. */
      readonly diagnostic: SemaDiagnosticResult | undefined;
    }
  | {
      readonly kind: "missing";
      readonly functionId: string;
      readonly inputs: Readonly<Record<string, unknown>>;
    };

export type SemaCallObserver = (observation: SemaCallObservation) => void;

export interface SemaStageEntry {
  readonly functionId: string;
  readonly inputs: Readonly<Record<string, unknown>>;
}

export interface SemaStagePasses {
  readonly encoder: number;
  readonly adapter: number;
  readonly head: number;
}

export interface SemaStageOutcome {
  /** One result per entry, in entry order, after each function's confidence policy. */
  readonly results: readonly unknown[];
  readonly passes: SemaStagePasses;
}

export interface SemaExecutionPlanStage {
  readonly index: number;
  readonly functionIds: readonly string[];
}

export interface SemaExecutionPlanDependency {
  readonly producerFunctionId: string;
  readonly consumerFunctionId: string;
  readonly consumerInput: string;
}

/** The compiler's execution plan as emitted in the IR bundle. */
export interface SemaExecutionPlan {
  readonly stages: readonly SemaExecutionPlanStage[];
  readonly dependencies?: readonly SemaExecutionPlanDependency[];
}

export type SemaStageInputsProvider = (
  stage: SemaExecutionPlanStage,
  results: ReadonlyMap<string, unknown>,
) => Readonly<Record<string, Readonly<Record<string, unknown>>>>;

export interface SemaPlanOutcome {
  readonly results: ReadonlyMap<string, unknown>;
  readonly stages: readonly SemaStagePasses[];
}

export interface SemaArtifactHandle {
  readonly manifestSha256: string;
  readonly functionIds: ReadonlySet<string>;
  // The caller supplies the compiled semantic function's static result type.
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-type-parameters
  call<T>(functionId: string, inputs: Readonly<Record<string, unknown>>): T;
  callStage(entries: readonly SemaStageEntry[]): SemaStageOutcome;
  close(): Promise<void>;
}

/** A sema call was made before `loadSemaArtifact()` resolved. */
export class SemaRuntimeNotLoadedError extends Error {
  readonly code = "SEMA_RUNTIME_NOT_LOADED";

  constructor() {
    super("no SemantScript artifact is loaded");
    this.name = "SemaRuntimeNotLoadedError";
  }
}

/** A handle was used after it was closed or replaced by a reload. */
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

/** The artifact directory `loadSemaArtifact()` looks for when no path is passed. */
export const DEFAULT_SEMA_ARTIFACT_PATH = ".semantscript/artifact";
/** Environment variable that overrides the default artifact root. */
export const SEMA_ARTIFACT_ENVIRONMENT_VARIABLE = "SEMANTSCRIPT_ARTIFACT";

export interface DefaultSemaArtifactPathOptions {
  /** The application's entry script (default `process.argv[1]`): the compiled output to search from. */
  readonly entry?: string | undefined;
  /** The working directory to search from as well (default `process.cwd()`). */
  readonly cwd?: string;
}

/**
 * Where the runtime looks for the artifact when the application does not say:
 * `SEMANTSCRIPT_ARTIFACT` when set; else the first `.semantscript/artifact`
 * found walking up from the entry script's directory (the compiled output,
 * so a deployed `dist/` finds the artifact shipped beside it) and then from
 * the working directory; else `.semantscript/artifact` under the working
 * directory, which is where `semantscript init` and `train` put it.
 */
export function defaultSemaArtifactPath(
  env: Readonly<Record<string, string | undefined>> = process.env,
  options: DefaultSemaArtifactPathOptions = {},
): string {
  const override = env[SEMA_ARTIFACT_ENVIRONMENT_VARIABLE];
  if (override !== undefined && override.length > 0) {
    return resolve(override);
  }
  const cwd = resolve(options.cwd ?? process.cwd());
  const entry = "entry" in options ? options.entry : process.argv[1];
  const starts = [
    ...(entry === undefined || entry.length === 0
      ? []
      : [dirname(resolve(entry))]),
    cwd,
  ];
  for (const start of starts) {
    for (const directory of ancestors(start)) {
      const candidate = join(directory, DEFAULT_SEMA_ARTIFACT_PATH);
      if (isDirectory(candidate)) {
        return candidate;
      }
    }
  }
  return join(cwd, DEFAULT_SEMA_ARTIFACT_PATH);
}

function ancestors(start: string): string[] {
  const result: string[] = [];
  let current = start;
  for (;;) {
    result.push(current);
    const parent = dirname(current);
    if (parent === current) return result;
    current = parent;
  }
}

function isDirectory(path: string): boolean {
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}

/**
 * Stages and initializes a complete immutable artifact, then atomically makes it
 * visible to the compiler ABI. A failed load never replaces the active artifact.
 */
export async function loadSemaArtifact(
  artifactPath: string = defaultSemaArtifactPath(),
  options: LoadSemaArtifactOptions = {},
): Promise<SemaArtifactHandle> {
  const handle = await activateArtifact(artifactPath, options);
  stopWatching();
  if (options.watch === true) {
    startWatching(artifactPath, options);
  }
  return handle;
}

let watcher: ArtifactWatcher | undefined;

interface ArtifactWatcher {
  readonly close: () => void;
}

const WATCH_DEBOUNCE_MILLISECONDS = 100;

function startWatching(
  artifactPath: string,
  options: LoadSemaArtifactOptions,
): void {
  const root = resolve(artifactPath);
  let timer: NodeJS.Timeout | undefined;
  let reloading = false;
  let pending = false;
  const reload = (): void => {
    if (reloading) {
      pending = true;
      return;
    }
    reloading = true;
    activateArtifact(artifactPath, options).then(
      (handle) => {
        reloading = false;
        options.onReload?.(handle);
        if (pending) {
          pending = false;
          reload();
        }
      },
      (error: unknown) => {
        reloading = false;
        options.onReloadError?.(error);
        if (pending) {
          pending = false;
          reload();
        }
      },
    );
  };
  const fsWatcher = watch(root, { persistent: false }, (_event, fileName) => {
    if (fileName !== null && fileName !== "current.json") return;
    if (timer !== undefined) clearTimeout(timer);
    timer = setTimeout(reload, WATCH_DEBOUNCE_MILLISECONDS);
    timer.unref();
  });
  fsWatcher.on("error", (error: unknown) => {
    options.onReloadError?.(error);
  });
  watcher = {
    close: () => {
      if (timer !== undefined) clearTimeout(timer);
      fsWatcher.close();
    },
  };
}

function stopWatching(): void {
  watcher?.close();
  watcher = undefined;
}

function activateArtifact(
  artifactPath: string,
  options: LoadSemaArtifactOptions,
): Promise<SemaArtifactHandle> {
  const fallbackSnapshot = snapshotFallbacks(options.fallbacks);
  const artifactOptions = options.artifact;
  const inferenceOptions = options.inference;
  const diagnosticsAlways = diagnosticsOption(options.diagnostics);
  const observe = observerOption(options.observe);

  return enqueueLifecycle(async () => {
    const staged = await loadArtifact(artifactPath, artifactOptions);
    const functions = bindFunctions(staged.functions, fallbackSnapshot);
    const plan = buildInferencePlan(staged, diagnosticsAlways);
    const runtime = await createInferenceRuntime(
      plan,
      inferenceOptionsForPlan(plan, inferenceOptions),
    );
    const token = Symbol("active SemantScript artifact");
    const canonicalInputVersion = canonicalInputVersionOfManifest(
      staged.manifest.compatibility.canonicalInput,
    );
    const next: ActiveArtifact = {
      token,
      manifestSha256: staged.manifestSha256,
      canonicalInputVersion,
      runtime,
      functions,
      fallbackInvocationStack: new Set(),
      diagnosticsAlways,
      observe,
    };
    const previous = activeArtifact;
    activeArtifact = next;

    if (previous !== undefined) {
      await previous.runtime.close().catch(() => undefined);
    }

    return createHandle(next);
  });
}

interface RequestScope {
  readonly id: number;
  readonly passes: { encoder: number; adapter: number; head: number };
  readonly runtimes: Set<InferenceRuntime>;
}

const requestScopes = new AsyncLocalStorage<RequestScope>();
let nextScopeId = 1;

/**
 * Runs `fn` inside one request scope (TASK-8.1): sema calls made while it runs,
 * synchronously or across its awaits, share encoder and adapter passes over
 * identical inputs, the way one execution-plan stage does, and the embeddings
 * kept for that are dropped when `fn` settles. Scopes nest by replacement.
 */
export function withSemaScope<T>(fn: () => T): T {
  const scope: RequestScope = {
    id: nextScopeId++,
    passes: { encoder: 0, adapter: 0, head: 0 },
    runtimes: new Set(),
  };
  const release = (): void => {
    for (const runtime of scope.runtimes) runtime.endScope(scope.id);
    scope.runtimes.clear();
  };
  let result: T;
  try {
    result = requestScopes.run(scope, fn);
  } catch (error) {
    release();
    throw error;
  }
  if (nodeTypes.isPromise(result)) {
    return (result as Promise<unknown>).then(
      (value) => {
        release();
        return value;
      },
      (error: unknown) => {
        release();
        throw error;
      },
    ) as T;
  }
  release();
  return result;
}

/** Model passes performed so far by the calls of the current request scope, or undefined outside one. */
export function semaScopePasses(): SemaStagePasses | undefined {
  const scope = requestScopes.getStore();
  return scope === undefined ? undefined : { ...scope.passes };
}

/** Closes the active artifact, if any. Primarily useful for orderly shutdown and tests. */
export function closeSemaArtifact(): Promise<void> {
  stopWatching();
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

function canonicalInputVersionOfManifest(
  encoding: string,
): CanonicalInputVersion {
  const version = canonicalInputVersionOf(encoding);
  if (version === undefined) {
    throw new TypeError(
      `artifact declares unimplemented canonical input encoding ${JSON.stringify(encoding)}`,
    );
  }
  return version;
}

/** @internal Compiler-generated code may call this through the exported __sema object. */
export function dispatchSemaStage(
  entries: readonly SemaStageEntry[],
): SemaStageOutcome {
  const current = activeArtifact;
  if (current === undefined) {
    throw new SemaRuntimeNotLoadedError();
  }
  return dispatchArtifactStage(current, entries);
}

/**
 * Runs the compiler's execution plan stage by stage. `provide` returns the
 * inputs of every function in a stage and receives the results so far, which
 * is how one stage's outputs feed the next.
 */
export function executeSemaPlan(
  plan: SemaExecutionPlan,
  provide: SemaStageInputsProvider,
): SemaPlanOutcome {
  const current = activeArtifact;
  if (current === undefined) {
    throw new SemaRuntimeNotLoadedError();
  }
  return executeArtifactPlan(current, plan, provide);
}

function executeArtifactPlan(
  artifact: ActiveArtifact,
  plan: SemaExecutionPlan,
  provide: SemaStageInputsProvider,
): SemaPlanOutcome {
  validateExecutionPlan(plan);
  const results = new Map<string, unknown>();
  const stagePasses: SemaStagePasses[] = [];
  for (const stage of plan.stages) {
    const inputsByFunction: Readonly<
      Record<string, Readonly<Record<string, unknown>>>
    > = provide(stage, results);
    const provided = Object.keys(inputsByFunction);
    if (
      provided.length !== stage.functionIds.length ||
      stage.functionIds.some((id) => !(id in inputsByFunction))
    ) {
      throw new TypeError(
        `stage ${String(stage.index)} inputs must cover exactly its functions`,
      );
    }
    const outcome = dispatchArtifactStage(
      artifact,
      stage.functionIds.map((functionId) => ({
        functionId,
        inputs: inputsByFunction[functionId] ?? {},
      })),
    );
    for (const [position, functionId] of stage.functionIds.entries()) {
      results.set(functionId, outcome.results[position]);
    }
    stagePasses.push(outcome.passes);
  }
  return { results, stages: stagePasses };
}

function isList(value: readonly unknown[]): boolean {
  // Callers from plain JavaScript can pass anything; the parameter type keeps
  // element types intact for TypeScript while the runtime check stays.
  return Array.isArray(value);
}

function validateExecutionPlan(plan: SemaExecutionPlan): void {
  if (!isList(plan.stages) || plan.stages.length === 0) {
    throw new TypeError("execution plan must list at least one stage");
  }
  const seen = new Set<string>();
  const stageOf = new Map<string, number>();
  for (const [position, stage] of plan.stages.entries()) {
    if (stage.index !== position) {
      throw new TypeError(
        "execution plan stages must be indexed consecutively from zero",
      );
    }
    if (!isList(stage.functionIds) || stage.functionIds.length === 0) {
      throw new TypeError(
        `execution plan stage ${String(position)} must name at least one function`,
      );
    }
    for (const functionId of stage.functionIds) {
      if (typeof functionId !== "string" || seen.has(functionId)) {
        throw new TypeError(
          `execution plan function ${JSON.stringify(functionId)} is not unique`,
        );
      }
      seen.add(functionId);
      stageOf.set(functionId, position);
    }
  }
  for (const dependency of plan.dependencies ?? []) {
    const producer = stageOf.get(dependency.producerFunctionId);
    const consumer = stageOf.get(dependency.consumerFunctionId);
    if (
      producer === undefined ||
      consumer === undefined ||
      producer >= consumer
    ) {
      throw new TypeError(
        `execution plan dependency ${dependency.producerFunctionId} -> ${dependency.consumerFunctionId} is not satisfiable in stage order`,
      );
    }
  }
}

function dispatchArtifactStage(
  artifact: ActiveArtifact,
  entries: readonly SemaStageEntry[],
): SemaStageOutcome {
  if (!isList(entries) || entries.length === 0) {
    throw new TypeError("a stage must carry at least one entry");
  }
  const resolved = entries.map((entry) => {
    const activeFunction = artifact.functions.get(entry.functionId);
    if (activeFunction === undefined) {
      artifact.observe?.({
        kind: "missing",
        functionId: entry.functionId,
        inputs: entry.inputs,
      });
      throw new SemaUnknownFunctionError(entry.functionId);
    }
    return {
      activeFunction,
      inputs: entry.inputs,
      canonicalInput: serializeCanonicalInputs(
        activeFunction.artifact.inputs satisfies readonly CanonicalInputEntry[],
        entry.inputs,
        {
          maximumBytes: artifact.runtime.maximumInputBytes,
          version: artifact.canonicalInputVersion,
        },
      ),
    };
  });
  const stage = artifact.runtime.callStage(
    resolved.map((entry, index) => ({
      functionId: entries[index]?.functionId ?? "",
      canonicalInput: entry.canonicalInput,
    })),
  );
  return {
    results: resolved.map((entry, index) =>
      applyConfidencePolicy(
        artifact,
        entry.activeFunction,
        entry.inputs,
        stage.results[index],
      ),
    ),
    passes: stage.passes,
  };
}

function dispatchArtifactCall(
  artifact: ActiveArtifact,
  functionId: string,
  inputs: Readonly<Record<string, unknown>>,
): unknown {
  const activeFunction = artifact.functions.get(functionId);
  if (activeFunction === undefined) {
    artifact.observe?.({ kind: "missing", functionId, inputs });
    throw new SemaUnknownFunctionError(functionId);
  }
  const semanticFunction = activeFunction.artifact;

  const canonicalInput = serializeCanonicalInputs(
    semanticFunction.inputs satisfies readonly CanonicalInputEntry[],
    inputs,
    {
      maximumBytes: artifact.runtime.maximumInputBytes,
      version: artifact.canonicalInputVersion,
    },
  );
  const scope = requestScopes.getStore();
  const inferenceResult = artifact.runtime.call(
    functionId,
    canonicalInput,
    scope?.id,
  );
  if (scope !== undefined) {
    scope.runtimes.add(artifact.runtime);
    const passes = artifact.runtime.lastPasses;
    if (passes !== undefined) {
      scope.passes.encoder += passes.encoder;
      scope.passes.adapter += passes.adapter;
      scope.passes.head += passes.head;
    }
  }
  return applyConfidencePolicy(
    artifact,
    activeFunction,
    inputs,
    inferenceResult,
  );
}

function applyConfidencePolicy(
  artifact: ActiveArtifact,
  activeFunction: ActiveFunction,
  inputs: Readonly<Record<string, unknown>>,
  inferenceResult: unknown,
): unknown {
  const semanticFunction = activeFunction.artifact;
  const functionId = semanticFunction.id;
  const { confidenceThreshold, resultMode } = semanticFunction.runtime;
  const computesDiagnostic =
    artifact.diagnosticsAlways ||
    resultMode === "diagnostic" ||
    confidenceThreshold !== null;
  artifact.observe?.({
    kind: "answered",
    functionId,
    inputs,
    resultMode,
    confidenceThreshold,
    diagnostic: computesDiagnostic
      ? (inferenceResult as SemaDiagnosticResult)
      : undefined,
  });

  if (resultMode === "value" && confidenceThreshold === null) {
    return artifact.diagnosticsAlways
      ? (inferenceResult as SemaDiagnosticResult).value
      : inferenceResult;
  }

  const diagnostic = inferenceResult as SemaDiagnosticResult;
  if (resultMode === "diagnostic") {
    return diagnostic;
  }

  if (confidenceThreshold === null) {
    throw new Error(
      "value-mode diagnostic inference requires a confidence threshold",
    );
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
    const fallbackResult = activeFunction.fallback(
      inputs,
      diagnostic,
      confidenceThreshold,
    );
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
    callStage(entries: readonly SemaStageEntry[]): SemaStageOutcome {
      if (closed || activeArtifact?.token !== artifact.token) {
        throw new SemaArtifactInactiveError(artifact.manifestSha256);
      }
      return dispatchArtifactStage(artifact, entries);
    },
    async close(): Promise<void> {
      if (closed) {
        return;
      }
      closed = true;
      if (activeArtifact?.token === artifact.token) {
        stopWatching();
      }
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

function diagnosticsOption(value: unknown): boolean {
  if (value === undefined) return false;
  if (value === "always") return true;
  throw new TypeError('diagnostics must be "always" when set');
}

function observerOption(value: unknown): SemaCallObserver | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "function") {
    throw new TypeError("observe must be a function when set");
  }
  return value as SemaCallObserver;
}

function buildInferencePlan(
  staged: StagedArtifactDescriptor<Uint8Array>,
  diagnosticsAlways: boolean,
): StagedInferencePlan {
  const resources = new Map(
    staged.resources.map(({ metadata, prepared }) => [
      metadata.ref,
      { metadata, prepared },
    ]),
  );
  const tokenizer = requireResource(
    resources,
    staged.manifest.model.tokenizerRef,
    "tokenizer",
  );
  const encoder = requireOnnxResource(
    resources,
    staged.manifest.model.encoderRef,
    "encoder",
  );
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
  const encoders = staged.manifest.resources
    .filter(
      (resource) =>
        resource.role === "encoder" &&
        resource.ref !== staged.manifest.model.encoderRef,
    )
    .map((resource) => {
      const prepared = requireOnnxResource(resources, resource.ref, "encoder");
      return {
        ref: resource.ref,
        model: validatedModel(prepared),
        abi: prepared.metadata.onnx,
      };
    });
  const functions = staged.functions.map((entry) => ({
    id: entry.id,
    adapterRef: entry.adapterRef,
    ...(entry.encoderRef === undefined ||
    entry.encoderRef === staged.manifest.model.encoderRef
      ? {}
      : { encoderRef: entry.encoderRef }),
    diagnosticsRequired:
      diagnosticsAlways ||
      entry.runtime.resultMode === "diagnostic" ||
      entry.runtime.confidenceThreshold !== null,
    heads: entry.heads.map((head) => buildHeadPlan(resources, head)),
  }));

  return {
    kind: "onnx",
    tokenizerJson: tokenizer.prepared,
    encoderModel: validatedModel(encoder),
    encoderAbi: encoder.metadata.onnx,
    ...(encoders.length === 0 ? {} : { encoders }),
    adapters,
    functions,
    maximumSequenceLength: (tokenizer.metadata as TokenizerResourceV1)
      .maximumSequenceLength,
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
    model: validatedModel({
      metadata: resource.metadata,
      prepared: resource.prepared,
    }),
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
      maximumInferenceResponseBytes(
        semanticFunction,
        MAXIMUM_RESPONSE_BUFFER_BYTES,
      ),
    );
  }
  if (requiredBytes > MAXIMUM_RESPONSE_BUFFER_BYTES) {
    throw new ArtifactLoadError(
      "SEMA_ARTIFACT_QUOTA",
      `artifact output requires ${String(requiredBytes)} response bytes; maximum is ${String(MAXIMUM_RESPONSE_BUFFER_BYTES)}`,
    );
  }
  if (
    options?.responseBufferBytes !== undefined &&
    options.responseBufferBytes < requiredBytes
  ) {
    throw new RangeError(
      `responseBufferBytes must be at least ${String(requiredBytes)} for this artifact`,
    );
  }
  return {
    ...options,
    responseBufferBytes:
      options?.responseBufferBytes ??
      Math.max(DEFAULT_RESPONSE_BUFFER_BYTES, requiredBytes),
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
      throw new TypeError(
        `fallback ${JSON.stringify(reference)} must be a function`,
      );
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
      const fallback =
        fallbackRef === null ? undefined : fallbacks.get(fallbackRef);
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
      invalidFallbackResult(
        semanticFunction.id,
        fallbackRef,
        "result is outside scalar output support",
      );
    }
    return;
  }

  if (result === null || typeof result !== "object" || Array.isArray(result)) {
    invalidFallbackResult(
      semanticFunction.id,
      fallbackRef,
      "flat result must be a plain object",
    );
  }
  if (nodeTypes.isProxy(result) || nodeTypes.isPromise(result)) {
    invalidFallbackResult(
      semanticFunction.id,
      fallbackRef,
      "flat result cannot be a proxy or promise",
    );
  }
  const prototype = Object.getPrototypeOf(result) as unknown;
  if (prototype !== Object.prototype && prototype !== null) {
    invalidFallbackResult(
      semanticFunction.id,
      fallbackRef,
      "flat result must be a plain object",
    );
  }

  const descriptors = Object.getOwnPropertyDescriptors(result);
  const actualNames = Object.getOwnPropertyNames(result);
  const expectedNames = new Set(
    semanticFunction.heads.map(
      (head) => head.outputPath[0] ?? missingFlatField(semanticFunction.id),
    ),
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
  if (
    (typeof value !== "object" || value === null) &&
    typeof value !== "function"
  ) {
    return;
  }
  if (depth > MAXIMUM_FALLBACK_RESULT_DEPTH || budget.remaining <= 0) {
    invalidFallbackResult(
      functionId,
      fallbackRef,
      "result exceeds validation limits",
    );
  }
  budget.remaining -= 1;
  if (nodeTypes.isProxy(value)) {
    invalidFallbackResult(
      functionId,
      fallbackRef,
      "result cannot contain proxies",
    );
  }
  if (nodeTypes.isPromise(value)) {
    invalidFallbackResult(
      functionId,
      fallbackRef,
      "result cannot be a promise",
    );
  }
  if (active.has(value)) {
    invalidFallbackResult(
      functionId,
      fallbackRef,
      "result cannot contain cycles",
    );
  }
  if (completed.has(value)) {
    return;
  }

  active.add(value);
  try {
    if (Object.getOwnPropertySymbols(value).length > 0) {
      invalidFallbackResult(
        functionId,
        fallbackRef,
        "result cannot contain symbol-keyed properties",
      );
    }
    for (const descriptor of Object.values(
      Object.getOwnPropertyDescriptors(value),
    )) {
      if (!("value" in descriptor)) {
        invalidFallbackResult(
          functionId,
          fallbackRef,
          "result cannot contain accessors",
        );
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
  return (
    semanticFunction.heads.length === 1 &&
    semanticFunction.heads[0]?.outputPath.length === 0
  );
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

function invalidFallbackResult(
  functionId: string,
  fallbackRef: string,
  detail: string,
): never {
  throw new SemaFallbackError(
    "invalid-result",
    functionId,
    fallbackRef,
    `fallback ${JSON.stringify(fallbackRef)} returned an invalid result: ${detail}`,
  );
}

function missingFlatField(functionId: string): never {
  throw new Error(
    `semantic function ${JSON.stringify(functionId)} has an invalid flat output path`,
  );
}
