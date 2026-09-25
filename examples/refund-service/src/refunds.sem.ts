import { always, never, sema } from "@semantscript/core";

// Domain "refunds": the refund policy as three decisions over plain records.
// Every rule is also a constraint, so the trainer labels inputs from the
// constraints (decision-7) and the verifier checks them; the model learns the
// policy, the constraints keep it honest.

export interface Customer {
  tier: "enterprise" | "standard";
  priorRefunds: number;
}

export interface Order {
  total: number;
  ageDays: number;
  status: "paid" | "fraudulent";
}

export interface Payment {
  method: "card" | "bank" | "voucher";
}

export type RefundDecision = "approve" | "deny" | "review";
export type RefundMethod = "original-payment" | "store-credit" | "manual";
export type RefundRisk = "high" | "low" | "medium";

export function decideRefund(customer: Customer, order: Order): RefundDecision {
  return sema<RefundDecision>({
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
    constraints: [
      always(() => order.ageDays > 90, "deny"),
      always(
        () => order.ageDays <= 90 && order.status === "fraudulent",
        "review",
      ),
      // The tier window: 60 days for enterprise customers, 30 for everyone else.
      always(
        () =>
          order.ageDays <= 90 &&
          order.status === "paid" &&
          ((customer.tier === "enterprise" && order.ageDays > 60) ||
            (customer.tier === "standard" && order.ageDays > 30)),
        "deny",
      ),
      always(
        () =>
          order.status === "paid" &&
          ((customer.tier === "enterprise" && order.ageDays <= 60) ||
            (customer.tier === "standard" && order.ageDays <= 30)) &&
          customer.priorRefunds > 2,
        "review",
      ),
      always(
        () =>
          order.status === "paid" &&
          ((customer.tier === "enterprise" && order.ageDays <= 60) ||
            (customer.tier === "standard" && order.ageDays <= 30)) &&
          customer.priorRefunds <= 2,
        "approve",
      ),
      never(() => order.status === "fraudulent", "approve"),
    ],
  })`
    Apply the refund policy. Orders older than 90 days are denied. A fraudulent
    order is never approved and is reviewed unless it is stale. Paid orders
    outside the tier window (60 days enterprise, 30 days standard) are denied;
    inside the window they are reviewed when the customer has more than two
    prior refunds and approved otherwise.
    Customer: ${customer}
    Order: ${order}
  `;
}

export function refundMethod(order: Order, payment: Payment): RefundMethod {
  return sema<RefundMethod>({
    examples: [
      {
        inputs: {
          order: { total: 88.5, ageDays: 12, status: "paid" },
          payment: { method: "card" },
        },
        output: "original-payment",
      },
      {
        inputs: {
          order: { total: 250, ageDays: 75, status: "paid" },
          payment: { method: "card" },
        },
        output: "store-credit",
      },
      {
        inputs: {
          order: { total: 900, ageDays: 20, status: "paid" },
          payment: { method: "bank" },
        },
        output: "manual",
      },
    ],
    constraints: [
      always(
        () => payment.method === "card" && order.ageDays <= 60,
        "original-payment",
      ),
      always(
        () => payment.method === "card" && order.ageDays > 60,
        "store-credit",
      ),
      always(() => payment.method === "voucher", "store-credit"),
      always(() => payment.method === "bank", "manual"),
    ],
  })`
    How an approved refund is paid out. Card payments go back to the card while
    the chargeback window is open (60 days), and to store credit after it.
    Vouchers always become store credit. Bank transfers are refunded manually.
    Order: ${order}
    Payment: ${payment}
  `;
}

export function refundRisk(customer: Customer, order: Order): RefundRisk {
  return sema<RefundRisk>({
    examples: [
      {
        inputs: {
          customer: { tier: "standard", priorRefunds: 0 },
          order: { total: 60, ageDays: 5, status: "paid" },
        },
        output: "low",
      },
      {
        inputs: {
          customer: { tier: "standard", priorRefunds: 1 },
          order: { total: 400, ageDays: 5, status: "paid" },
        },
        output: "medium",
      },
      {
        inputs: {
          customer: { tier: "enterprise", priorRefunds: 0 },
          order: { total: 2600, ageDays: 5, status: "paid" },
        },
        output: "high",
      },
    ],
    constraints: [
      always(
        () =>
          order.status === "fraudulent" ||
          customer.priorRefunds > 2 ||
          order.total > 1000,
        "high",
      ),
      always(
        () =>
          order.status === "paid" &&
          customer.priorRefunds === 0 &&
          order.total < 100,
        "low",
      ),
      always(
        () =>
          order.status === "paid" &&
          !(customer.priorRefunds > 2 || order.total > 1000) &&
          !(customer.priorRefunds === 0 && order.total < 100),
        "medium",
      ),
    ],
  })`
    The risk of a refund request for the fraud team's queue. High when the
    order is fraudulent, the customer has more than two prior refunds or the
    order is above 1000. Low for a paid order under 100 from a customer with no
    prior refunds. Medium otherwise.
    Customer: ${customer}
    Order: ${order}
  `;
}
