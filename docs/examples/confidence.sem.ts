// @confidence thresholds and sema.withConfidence diagnostics.
import { sema, type Ordinal } from "@semantscript/core";

type Severity = Ordinal<["low", "medium", "high", "critical"]>;

interface Incident {
  service: string;
  errorRate: number;
  customersAffected: number;
}

// Below 0.9 the runtime calls the fallback registered for this expression, or
// throws SemaConfidenceError when none is registered.
export function severityOrFallback(incident: Incident): Severity {
  return sema<Severity>`
    @confidence(0.9)
    Assess the incident severity.
    Incident: ${incident}
  `;
}

// The diagnostic form always returns the calibrated distribution, so the caller
// decides what a low confidence means.
export function severityWithDiagnostics(incident: Incident) {
  const result = sema.withConfidence<Severity>`
    Assess the incident severity.
    Incident: ${incident}
  `;
  if (result.confidence < 0.9) {
    return {
      value: "medium" as Severity,
      escalate: true,
      expectedRank: result.expectedValue,
    };
  }
  return {
    value: result.value,
    escalate: false,
    expectedRank: result.expectedValue,
  };
}
