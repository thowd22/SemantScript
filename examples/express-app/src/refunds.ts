import { PGlite } from "@electric-sql/pglite";
import {
  Controller,
  Post,
  RequestContext,
  require,
  gate,
  rollback,
  transactional,
} from "@semantscript/framework";

import { decideRefund } from "./refunds.sem.js";

/**
 * Data stays out of the weights: the handler reads the customer and order rows,
 * hands the decision plain values, and the decision gates the transaction. A
 * refund row is written only when the model approves; otherwise the
 * transaction rolls back and nothing is persisted.
 *
 * Without DATABASE_URL the app runs Postgres in process (PGlite, Postgres 17
 * in WebAssembly) with a seeded schema; with it, pass a pg Pool instead.
 */
export const db = new PGlite();

export async function migrate(): Promise<void> {
  await db.exec(`
    CREATE TABLE IF NOT EXISTS customers (id text PRIMARY KEY, tier text NOT NULL, prior_refunds int NOT NULL);
    CREATE TABLE IF NOT EXISTS orders (id text PRIMARY KEY, customer_id text NOT NULL REFERENCES customers(id), total numeric NOT NULL, age_days int NOT NULL, status text NOT NULL);
    CREATE TABLE IF NOT EXISTS refunds (order_id text PRIMARY KEY REFERENCES orders(id), decision text NOT NULL, decided_at timestamptz NOT NULL DEFAULT now());
    INSERT INTO customers VALUES ('c1', 'standard', 1), ('c2', 'enterprise', 4) ON CONFLICT DO NOTHING;
    INSERT INTO orders VALUES ('o1', 'c1', 88.5, 12, 'paid'), ('o2', 'c2', 900, 70, 'paid') ON CONFLICT DO NOTHING;
  `);
}

@Controller("/refunds")
export class RefundController {
  @Post("/:orderId")
  async request(context: RequestContext) {
    const orderId = context.params["orderId"];
    require(typeof orderId === "string" &&
      orderId.length > 0, "orderId is required", 400);
    return transactional(db, async (tx) => {
      const { rows } = await tx.query<{
        id: string;
        total: string;
        age_days: number;
        status: "paid" | "fraudulent" | "cancelled";
        tier: "standard" | "enterprise";
        prior_refunds: number;
      }>(
        `SELECT o.id, o.total, o.age_days, o.status, c.tier, c.prior_refunds
           FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.id = $1`,
        [orderId],
      );
      const row = rows[0];
      if (row === undefined) return rollback("no such order");
      const decision = decideRefund(
        { tier: row.tier, priorRefunds: row.prior_refunds },
        { total: Number(row.total), ageDays: row.age_days, status: row.status },
      );
      return gate(
        decision,
        (value) => value === "approve",
        async () => {
          await tx.query(
            "INSERT INTO refunds (order_id, decision) VALUES ($1, $2)",
            [orderId, decision],
          );
          return { orderId, decision, refunded: true };
        },
        `decision ${decision}: no refund written`,
      );
    });
  }
}
