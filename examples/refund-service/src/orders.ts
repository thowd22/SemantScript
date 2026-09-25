import {
  Controller,
  Post,
  RequestContext,
  require,
  transactional,
} from "@semantscript/framework";

import { db } from "./db.js";
import { screenOrder } from "./orders.sem.js";

interface ScreeningRow {
  id: string;
  total: string;
  mismatched_address: boolean;
  orders_last_hour: number;
  chargebacks: number;
  prior_refunds: number;
}

/**
 * POST /orders/:orderId/screen runs the fraud chain (the flag first, then the
 * hold and the escalation that depend on it). The shipment hold is written to
 * the order and the review row records who looks at it next.
 */
@Controller("/orders")
export class OrderController {
  @Post("/:orderId/screen")
  async screen(context: RequestContext) {
    const orderId = context.params["orderId"];
    require(typeof orderId === "string" &&
      orderId.length > 0, "orderId is required", 400);
    return transactional(db, async (tx) => {
      const { rows } = await tx.query<ScreeningRow>(
        `SELECT o.id, o.total, o.mismatched_address, o.orders_last_hour, o.chargebacks, c.prior_refunds
           FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.id = $1`,
        [orderId],
      );
      const row = rows[0];
      require(row !== undefined, "no such order", 404);
      const {
        flag,
        hold,
        escalation: reviewer,
      } = screenOrder(
        { total: Number(row.total) },
        {
          mismatchedAddress: row.mismatched_address,
          ordersLastHour: row.orders_last_hour,
          chargebacks: row.chargebacks,
        },
        { priorRefunds: row.prior_refunds },
      );
      await tx.query("UPDATE orders SET shipment_held = $2 WHERE id = $1", [
        orderId,
        hold,
      ]);
      await tx.query(
        `INSERT INTO fraud_reviews (order_id, flag, escalation) VALUES ($1, $2, $3)
           ON CONFLICT (order_id) DO UPDATE SET flag = EXCLUDED.flag, escalation = EXCLUDED.escalation`,
        [orderId, flag, reviewer],
      );
      return { orderId, flag, shipmentHeld: hold, escalation: reviewer };
    });
  }
}
