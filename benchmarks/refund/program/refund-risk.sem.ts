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
 * compiled constraints on real inputs without a language-model teacher.
 */
export function assessRefundRisk(
  customer: Customer,
  order: Order,
): ScalarSemaResult<RefundRisk> {
  return sema.withConfidence<RefundRisk>({
    constraints: [
      always(
        () =>
          order.status === "fraudulent" ||
          customer.priorRefunds >= 5 ||
          order.total >= 5000,
        "high",
      ),
      always(
        () =>
          !(
            order.status === "fraudulent" ||
            customer.priorRefunds >= 5 ||
            order.total >= 5000
          ) &&
          ((customer.priorRefunds >= 2 && order.total >= 500) ||
            order.ageDays > 60 ||
            order.total >= 1500),
        "medium",
      ),
      always(
        () =>
          !(
            order.status === "fraudulent" ||
            customer.priorRefunds >= 5 ||
            order.total >= 5000
          ) &&
          !(
            (customer.priorRefunds >= 2 && order.total >= 500) ||
            order.ageDays > 60 ||
            order.total >= 1500
          ),
        "low",
      ),
    ],
  })`Rate the refund risk of this request. The risk is high when the order is fraudulent, when the customer has five or more prior refunds, or when the order total is at least 5000. Otherwise the risk is medium when the customer has two or more prior refunds on an order of at least 500, when the order is older than 60 days, or when the order total is at least 1500. Otherwise the risk is low.
Customer: ${customer}
Order: ${order}
`;
}
