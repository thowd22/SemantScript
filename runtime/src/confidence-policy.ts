export interface DistributionEntry<T> {
  readonly value: T;
  readonly probability: number;
}

export interface ScalarSemaResult<T> {
  readonly value: T;
  readonly confidence: number;
  readonly uncertainty: number;
  readonly distribution: readonly DistributionEntry<T>[];
  readonly expectedValue: number | null;
}

export interface ObjectSemaResult<T extends object> {
  readonly value: T;
  readonly minimumFieldConfidence: number;
  readonly maximumFieldUncertainty: number;
  readonly fields: {
    readonly [Key in keyof T]: ScalarSemaResult<T[Key]>;
  };
}

export type SemaResult<T> = [T] extends [boolean | string | number]
  ? ScalarSemaResult<T>
  : [T] extends [object]
    ? ObjectSemaResult<T>
    : never;

export type SemaRuntimeScalarValue = boolean | string | number;
export type SemaDiagnosticResult =
  | ScalarSemaResult<SemaRuntimeScalarValue>
  | ObjectSemaResult<Readonly<Record<string, SemaRuntimeScalarValue>>>;

/** Synchronous application fallback invoked only for a below-threshold value call. */
export type SemaFallback = (
  inputs: Readonly<Record<string, unknown>>,
  diagnostic: SemaDiagnosticResult,
  requiredConfidence: number,
) => unknown;

export class SemaConfidenceError extends Error {
  readonly code = "SEMA_CONFIDENCE_BELOW_THRESHOLD";
  readonly functionId: string;
  readonly threshold: number;
  readonly diagnostic: SemaDiagnosticResult;

  constructor(functionId: string, threshold: number, diagnostic: SemaDiagnosticResult) {
    super(
      `semantic function ${JSON.stringify(functionId)} returned confidence below ${String(threshold)}`,
    );
    this.name = "SemaConfidenceError";
    this.functionId = functionId;
    this.threshold = threshold;
    this.diagnostic = diagnostic;
  }
}

export type SemaFallbackErrorReason = "missing" | "invalid-result" | "cycle";

export class SemaFallbackError extends TypeError {
  readonly code = "SEMA_FALLBACK_INVALID";
  readonly reason: SemaFallbackErrorReason;
  readonly functionId: string;
  readonly fallbackRef: string;

  constructor(
    reason: SemaFallbackErrorReason,
    functionId: string,
    fallbackRef: string,
    message: string,
  ) {
    super(message);
    this.name = "SemaFallbackError";
    this.reason = reason;
    this.functionId = functionId;
    this.fallbackRef = fallbackRef;
  }
}
