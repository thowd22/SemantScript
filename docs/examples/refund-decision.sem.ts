// Plain sema<T>: a nominal string-union output with two interpolated inputs.
import { sema } from "@semantscript/core";

type RefundDecision = "approve" | "deny" | "review";

interface Customer {
  tier: "standard" | "enterprise";
  priorRefunds: number;
}

interface Order {
  ageDays: number;
  status: "paid" | "fraudulent";
  total: number;
}

export function decideRefund(customer: Customer, order: Order): RefundDecision {
  return sema<RefundDecision>`
    Apply our refund policy. Enterprise customers get 60 days; everyone else
    gets 30. Suspicious circumstances go to review.
    Customer: ${customer}
    Order: ${order}
  `;
}
