import {
  Controller,
  Post,
  RequestContext,
  require,
  gate,
  rollback,
  transactional,
} from "@semantscript/framework";

import { db } from "./db.js";
import { decideRefund, refundMethod, refundRisk } from "./refunds.sem.js";

interface RefundRow {
  id: string;
  total: string;
  age_days: number;
  status: "paid" | "fraudulent";
  payment_method: "card" | "bank" | "voucher";
  tier: "standard" | "enterprise";
  prior_refunds: number;
}

/**
 * POST /refunds/:orderId reads the order and its customer inside one
 * transaction, decides with the three refund expressions and commits a refund
 * row only when the decision approves. Everything else rolls back with the
 * decision as the reason, so nothing is persisted for a denied or reviewed
 * request.
 */
@Controller("/refunds")
export class RefundController {
  @Post("/:orderId")
  async request(context: RequestContext) {
    const orderId = context.params["orderId"];
    require(typeof orderId === "string" &&
      orderId.length > 0, "orderId is required", 400);
    return transactional(db, async (tx) => {
      const { rows } = await tx.query<RefundRow>(
        `SELECT o.id, o.total, o.age_days, o.status, o.payment_method, c.tier, c.prior_refunds
           FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.id = $1`,
        [orderId],
      );
      const row = rows[0];
      if (row === undefined) return rollback("no such order");
      const customer = { tier: row.tier, priorRefunds: row.prior_refunds };
      const order = {
        total: Number(row.total),
        ageDays: row.age_days,
        status: row.status,
      };
      const decision = decideRefund(customer, order);
      const risk = refundRisk(customer, order);
      return gate(
        decision,
        (value) => value === "approve",
        async () => {
          const method = refundMethod(order, { method: row.payment_method });
          await tx.query(
            "INSERT INTO refunds (order_id, decision, method, risk) VALUES ($1, $2, $3, $4)",
            [orderId, decision, method, risk],
          );
          return { orderId, decision, method, risk, refunded: true };
        },
        `decision ${decision} (risk ${risk}): no refund written`,
      );
    });
  }
}
