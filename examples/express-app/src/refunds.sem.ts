import { sema } from "@semantscript/core";

export interface Customer {
  tier: "standard" | "enterprise";
  priorRefunds: number;
}

export interface Order {
  total: number;
  ageDays: number;
  status: "paid" | "fraudulent" | "cancelled";
}

export type RefundDecision = "approve" | "deny" | "review";

/**
 * The refund policy as a decision over plain values read from the database.
 * The expression sees the customer and order records and nothing else.
 */
export function decideRefund(customer: Customer, order: Order): RefundDecision {
  return sema<RefundDecision>({
    // Attested cases the verifier must reproduce before a release is published.
    examples: [
      {
        inputs: {
          customer: { tier: "standard", priorRefunds: 1 },
          order: { total: 88.5, ageDays: 12, status: "paid" },
        },
        output: "approve",
      },
      {
        inputs: {
          customer: { tier: "enterprise", priorRefunds: 4 },
          order: { total: 900, ageDays: 70, status: "paid" },
        },
        output: "deny",
      },
      {
        inputs: {
          customer: { tier: "standard", priorRefunds: 0 },
          order: { total: 40, ageDays: 3, status: "fraudulent" },
        },
        output: "review",
      },
    ],
  })`
    Decide a refund request. Deny orders older than 90 days. Never approve a
    fraudulent order; review it unless it is stale. Deny paid orders outside
    the tier window (60 days enterprise, 30 days standard). Inside the window,
    review when the customer has more than two prior refunds, else approve.
    Customer: ${customer}
    Order: ${order}
  `;
}
