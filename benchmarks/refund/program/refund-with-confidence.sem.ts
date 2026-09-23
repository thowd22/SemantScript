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
 * The policy is stated completely: the first Phase 1 run showed that the short
 * form ("60 days, 30 days, suspicious goes to review") was read three different
 * ways by the teacher, the judge and the baselines. Every rule below is also a
 * deterministic constraint so generated cases are rule-checked at build time.
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
      always(() => order.ageDays <= 90 && order.status === "fraudulent", "review"),
      always(
        () =>
          order.ageDays <= 90 &&
          order.status === "paid" &&
          ((customer.tier === "standard" && order.ageDays > 30) ||
            (customer.tier === "enterprise" && order.ageDays > 60)),
        "deny",
      ),
      always(
        () =>
          order.status === "paid" &&
          ((customer.tier === "standard" && order.ageDays <= 30) ||
            (customer.tier === "enterprise" && order.ageDays <= 60)) &&
          (customer.priorRefunds >= 5 ||
            (customer.priorRefunds >= 3 && order.total >= 1000) ||
            order.total >= 5000 ||
            (order.ageDays <= 1 && order.total >= 2000 && customer.priorRefunds >= 2)),
        "review",
      ),
      always(
        () =>
          order.status === "paid" &&
          ((customer.tier === "standard" && order.ageDays <= 30) ||
            (customer.tier === "enterprise" && order.ageDays <= 60)) &&
          !(
            customer.priorRefunds >= 5 ||
            (customer.priorRefunds >= 3 && order.total >= 1000) ||
            order.total >= 5000 ||
            (order.ageDays <= 1 && order.total >= 2000 && customer.priorRefunds >= 2)
          ),
        "approve",
      ),
    ],
  })`Apply our refund policy. An order older than 90 days is always denied. A fraudulent order that is 90 days old or younger goes to review. A paid order outside its refund window is denied: enterprise customers get 60 days and everyone else gets 30, counting the window as inclusive. A paid order inside its window goes to review when the circumstances are suspicious, meaning five or more prior refunds, three or more prior refunds on an order of at least 1000, an order total of at least 5000, or a same-day cancellation (age at most 1 day) of an order of at least 2000 by a customer with two or more prior refunds. Otherwise a paid order inside its window is approved.
Customer: ${customer}
Order: ${order}
`;
}
