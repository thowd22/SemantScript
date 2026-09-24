// The configured form: gold examples and always/never constraints.
import { always, never, sema } from "@semantscript/core";

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

const enterpriseExample: Customer = { tier: "enterprise", priorRefunds: 0 };
const orderAt45Days: Order = { ageDays: 45, status: "paid", total: 129 };

export function decideRefund(customer: Customer, order: Order): RefundDecision {
  return sema<RefundDecision>({
    examples: [
      {
        inputs: { customer: enterpriseExample, order: orderAt45Days },
        output: "approve",
      },
    ],
    constraints: [
      never(() => order.status === "fraudulent", "approve"),
      always(() => order.ageDays > 90, "deny"),
    ],
  })`
    Apply our refund policy. Enterprise customers get 60 days; everyone else
    gets 30. Suspicious circumstances go to review.
    Customer: ${customer}
    Order: ${order}
  `;
}
