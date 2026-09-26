import type { InferenceHeadPlan } from "./inference-protocol.js";

/**
 * Pure head arithmetic shared by the inference worker and the testing stub, so
 * a stubbed diagnostic's uncertainty and expected value are computed exactly
 * as a real head's are. `invalid` builds the error each caller throws.
 */
export type HeadMathError = (message: string) => Error;

export function normalizedEntropy(
  probabilities: readonly number[],
  invalid: HeadMathError,
): number {
  if (probabilities.length <= 1) {
    return 0;
  }
  const entropy = compensatedSum(probabilities.length, (index) => {
    const probability = probabilities[index] ?? 0;
    return probability === 0 ? 0 : -probability * Math.log(probability);
  });
  if (!Number.isFinite(entropy)) {
    throw invalid("head entropy is invalid");
  }
  if (entropy === 0) {
    return 0;
  }
  return clamp(entropy / Math.log(probabilities.length), 0, 1);
}

export function expectedValue(
  head: Pick<InferenceHeadPlan, "expectedValueMode" | "support">,
  probabilities: readonly number[],
  invalid: HeadMathError,
): number | null {
  if (head.expectedValueMode === "none") {
    return null;
  }
  if (head.expectedValueMode === "zero-based-rank") {
    return compensatedSum(
      probabilities.length,
      (index) => (probabilities[index] ?? 0) * index,
    );
  }

  let scale = 1;
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (const supportValue of head.support) {
    if (typeof supportValue !== "number") {
      throw invalid("numeric ordinal support is invalid");
    }
    scale = Math.max(scale, Math.abs(supportValue));
    minimum = Math.min(minimum, supportValue);
    maximum = Math.max(maximum, supportValue);
  }
  const normalized = compensatedSum(probabilities.length, (index) => {
    const supportValue = head.support[index];
    if (typeof supportValue !== "number") {
      throw invalid("numeric ordinal support is invalid");
    }
    return (probabilities[index] ?? 0) * (supportValue / scale);
  });
  const result = clamp(normalized, minimum / scale, maximum / scale) * scale;
  if (!Number.isFinite(result)) {
    throw invalid("ordinal expected value is invalid");
  }
  return result;
}

export function compensatedSum(
  length: number,
  valueAt: (index: number) => number,
): number {
  let sum = 0;
  let correction = 0;
  for (let index = 0; index < length; index += 1) {
    const value = valueAt(index);
    const next = sum + value;
    correction +=
      Math.abs(sum) >= Math.abs(value)
        ? sum - next + value
        : value - next + sum;
    sum = next;
  }
  return sum + correction;
}

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
