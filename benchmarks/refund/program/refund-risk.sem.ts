import {
  always,
  sema,
  type ScalarSemaResult,
} from "@semantscript/core";

export type RefundRisk = "low" | "medium" | "high";

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
 * Companion function for the shared-encoder experiment (TASK-6.1).
 *
 * It reads the same customer and order records as the refund decision but
 * answers a different question, so a shared encoder must serve two heads with
 * different label spaces. The policy is stated completely and every rule is a
 * constraint, which is what lets the training corpus be labeled from the
 * compiled constraints on real inputs without a language-model teacher. The
 * top rule depends on a single field so that every input has a one-field
 * counterfactual twin, which the constraint-based adversarial teacher needs.
 */
export function assessRefundRisk(
  customer: Customer,
  order: Order,
): ScalarSemaResult<RefundRisk> {
  return sema.withConfidence<RefundRisk>({
    constraints: [
      always(() => order.status === "fraudulent", "high"),
      always(
        () =>
          order.status === "paid" &&
          (customer.priorRefunds >= 3 || order.total >= 1500 || order.ageDays > 60),
        "medium",
      ),
      always(
        () =>
          order.status === "paid" &&
          !(customer.priorRefunds >= 3 || order.total >= 1500 || order.ageDays > 60),
        "low",
      ),
    ],
  })`Rate the refund risk of this request. The risk is high when the order is fraudulent. Otherwise the risk is medium when the customer has three or more prior refunds, when the order total is at least 1500, or when the order is older than 60 days. Otherwise the risk is low.
Customer: ${customer}
Order: ${order}
`;
}
