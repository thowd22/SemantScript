import {
  always,
  never,
  sema,
  type ScalarSemaResult,
} from "@semantscript/core";

export type RefundDecision = "approve" | "deny" | "review";

export interface Customer {
  readonly priorRefunds: number;
  readonly tier: "enterprise" | "standard";
}

export interface Order {
  readonly ageDays: number;
  readonly status: "fraudulent" | "paid";
  readonly total: number;
}

/**
 * Canonical diagnostic form of the Phase 1 refund task.
 *
 * Examples intentionally live outside this source file: benchmark evaluation
 * cases must never become compiler examples or training inputs by accident.
 */
export function decideRefund(
  customer: Customer,
  order: Order,
): ScalarSemaResult<RefundDecision> {
  return sema.withConfidence<RefundDecision>({
    constraints: [
      never(() => order.status === "fraudulent", "approve"),
      always(() => order.ageDays > 90, "deny"),
    ],
  })`Apply our refund policy. Enterprise customers get 60 days; everyone else gets 30. Suspicious circumstances go to review.
Customer: ${customer}
Order: ${order}
`;
}
