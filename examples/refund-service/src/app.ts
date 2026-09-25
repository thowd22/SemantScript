import express from "express";
import { mountControllers } from "@semantscript/framework";

import { migrate } from "./db.js";
import { OrderController } from "./orders.js";
import { RefundController } from "./refunds.js";
import { TicketController } from "./tickets.js";

export { decideRefund, refundMethod, refundRisk } from "./refunds.sem.js";
export { needsHuman, ticketPriority, ticketQueue } from "./tickets.sem.js";
export { screenOrder } from "./orders.sem.js";
export { db, migrate } from "./db.js";

/**
 * The application without a listener: `createApp` is what the tests, the
 * server and `semantscript run` share. Exported sema functions are callable
 * directly with `semantscript run dist/app.js --call decideRefund --input
 * '[{...}, {...}]'` once the artifact is loaded.
 */
export async function createApp(): Promise<express.Express> {
  await migrate();
  const app = express();
  app.use(express.json());
  mountControllers(app, [
    new RefundController(),
    new TicketController(),
    new OrderController(),
  ]);
  return app;
}
