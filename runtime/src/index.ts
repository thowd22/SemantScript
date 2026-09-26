import { dispatchSemaCall, dispatchSemaStage } from "./artifact-runtime.js";
import type { SemaStageEntry, SemaStageOutcome } from "./artifact-runtime.js";

export {
  checkSemaArtifact,
  closeSemaArtifact,
  executeSemaPlan,
  DEFAULT_SEMA_ARTIFACT_PATH,
  defaultSemaArtifactPath,
  type DefaultSemaArtifactPathOptions,
  loadSemaArtifact,
  SEMA_ARTIFACT_ENVIRONMENT_VARIABLE,
  semaScopePasses,
  withSemaScope,
  SemaArtifactInactiveError,
  SemaRuntimeNotLoadedError,
  type LoadSemaArtifactOptions,
  type SemaArtifactCheck,
  type SemaArtifactHandle,
  type SemaExecutionPlan,
  type SemaExecutionPlanDependency,
  type SemaExecutionPlanStage,
  type SemaPlanOutcome,
  type SemaStageEntry,
  type SemaStageInputsProvider,
  type SemaStageOutcome,
  type SemaStagePasses,
} from "./artifact-runtime.js";
export {
  ArtifactLoadError,
  type ArtifactLoadErrorCode,
  type ArtifactLoadOptions,
} from "./artifact-loader.js";
export {
  remedy,
  remedyFamily,
  type RemedyId,
  type RemedyParams,
} from "./remedies.js";
export {
  SemaInputError,
  type SemaInputErrorReason,
  type SemaInputPath,
} from "./canonical-input.js";
export {
  SemaInferenceError,
  SemaInferenceInitializationError,
  SemaInferenceInputError,
  SemaInferenceTimeoutError,
  SemaUnknownFunctionError,
  type SemaInferenceErrorCode,
} from "./inference-runtime.js";
export {
  SemaConfidenceError,
  SemaFallbackError,
  type DistributionEntry,
  type ObjectSemaResult,
  type ScalarSemaResult,
  type SemaDiagnosticResult,
  type SemaFallback,
  type SemaFallbackErrorReason,
  type SemaResult,
  type SemaRuntimeScalarValue,
} from "./confidence-policy.js";

import type { SemaResult } from "./confidence-policy.js";

declare const ordinalKind: unique symbol;
declare const boundedIntKind: unique symbol;
declare const boundedNumberKind: unique symbol;

export type Ordinal<Values extends readonly [string, string, ...string[]]> =
  Values[number] & {
    readonly [ordinalKind]?: Values;
  };

export type BoundedInt<
  Minimum extends number,
  Maximum extends number,
> = number & {
  readonly [boundedIntKind]?: readonly [Minimum, Maximum];
};

export type BoundedNumber<
  Minimum extends number,
  Maximum extends number,
  Step extends number,
> = number & {
  readonly [boundedNumberKind]?: readonly [Minimum, Maximum, Step];
};

const semaConstraintKind = Symbol("semantscript.constraint");

export interface SemaConstraint<T> {
  readonly [semaConstraintKind]: T;
}

export interface SemaExample<T> {
  readonly inputs: Readonly<Record<string, unknown>>;
  readonly output: T;
}

export interface SemaOptions<T> {
  readonly examples?: readonly SemaExample<T>[];
  // The compiler validates each constraint output against T. Keeping the
  // container erased avoids a TypeScript contextual-inference runaway when
  // generic always/never calls appear inside sema<T> options.
  readonly constraints?: readonly SemaConstraint<unknown>[];
}

export interface ConfiguredSemaTag<T> {
  (strings: TemplateStringsArray, ...inputs: unknown[]): T;
}

export interface ConfiguredDiagnosticSemaTag<T> {
  (strings: TemplateStringsArray, ...inputs: unknown[]): SemaResult<T>;
}

export interface DiagnosticSemaTag {
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): SemaResult<T>;
  <T>(options: SemaOptions<T>): ConfiguredDiagnosticSemaTag<T>;
}

export interface SemaTag {
  // Explicit T is the source-language output declaration consumed by the compiler.
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-type-parameters
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): T;
  <T>(options: SemaOptions<T>): ConfiguredSemaTag<T>;
  readonly withConfidence: DiagnosticSemaTag;
}

function uncompiledPrimitive(): never {
  throw new Error("sema expressions must be compiled before they can execute");
}

const diagnosticSema = (() => uncompiledPrimitive()) as DiagnosticSemaTag;
const semaImplementation = (() => uncompiledPrimitive()) as unknown as SemaTag;

Object.defineProperty(semaImplementation, "withConfidence", {
  configurable: false,
  enumerable: true,
  value: diagnosticSema,
  writable: false,
});

export const sema = semaImplementation;

export interface SemaRuntimeDispatcher {
  // The compiler preserves each site's static result type across this internal ABI.
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-type-parameters
  call<T>(functionId: string, inputs: Readonly<Record<string, unknown>>): T;
  /** One execution-plan stage: identical inputs share one encoder pass. */
  callStage(entries: readonly SemaStageEntry[]): SemaStageOutcome;
}

/** Compiler-internal call ABI populated by the artifact runtime in TASK-5.9. */
export const __sema: SemaRuntimeDispatcher = {
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-type-parameters
  call<T>(functionId: string, inputs: Readonly<Record<string, unknown>>): T {
    return dispatchSemaCall<T>(functionId, inputs);
  },
  callStage(entries: readonly SemaStageEntry[]): SemaStageOutcome {
    return dispatchSemaStage(entries);
  },
};

/** Declare that the output must be `output` whenever `predicate` holds; compiled into the IR and never executed at runtime. */
export function always<T>(
  predicate: () => boolean,
  requiredOutput: T,
): SemaConstraint<T> {
  if (typeof predicate !== "function") {
    throw new TypeError("always predicate must be a function");
  }

  return { [semaConstraintKind]: requiredOutput };
}

/** Declare that the output must never be `output` whenever `predicate` holds; compiled into the IR and never executed at runtime. */
export function never<T>(
  predicate: () => boolean,
  forbiddenOutput: T,
): SemaConstraint<T> {
  if (typeof predicate !== "function") {
    throw new TypeError("never predicate must be a function");
  }

  return { [semaConstraintKind]: forbiddenOutput };
}
