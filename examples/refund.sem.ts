import { always, never, sema } from "@semantscript/core";

type RefundDecision = "approve" | "deny" | "review";

interface Customer {
  priorRefunds: number;
  tier: "enterprise" | "standard";
}

interface Order {
  ageDays: number;
  status: "fraudulent" | "paid";
  total: number;
}

export function decideRefund(customer: Customer, order: Order): RefundDecision {
  return sema<RefundDecision>({
    examples: [
      {
        inputs: {
          customer: { priorRefunds: 0, tier: "enterprise" },
          order: { ageDays: 45, status: "paid", total: 129 },
        },
        output: "approve",
      },
    ],
    constraints: [
      never(() => order.status === "fraudulent", "approve"),
      always(() => order.ageDays > 90, "deny"),
    ],
  })`Apply our refund policy. Enterprise customers get 60 days; everyone else gets 30. Suspicious circumstances go to review.
Customer: ${customer}
Order: ${order}
`;
}
