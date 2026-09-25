import express from "express";
import { mountControllers } from "@semantscript/framework";

import { migrate, RefundController } from "./refunds.js";
import { triage } from "./triage.sem.js";

export { db, migrate } from "./refunds.js";
export { decideRefund } from "./refunds.sem.js";
export { triage } from "./triage.sem.js";

/**
 * The application without a listener, shared by the server and the tests.
 * The caller loads the artifact first (`loadSemaArtifact`); the routes only
 * evaluate sema functions per request.
 */
export async function createApp(): Promise<express.Express> {
  await migrate();
  const app = express();
  app.use(express.json());

  app.post("/tickets", (request, response) => {
    const { subject, body } = (request.body ?? {}) as {
      subject?: string;
      body?: string;
    };
    if (typeof subject !== "string" || typeof body !== "string") {
      response.status(400).json({ error: "subject and body are required" });
      return;
    }
    const ticket = {
      id: crypto.randomUUID(),
      subject,
      body,
      priority: triage(subject, body),
    };
    response.status(201).json(ticket);
  });

  // POST /refunds/:orderId reads Postgres, decides with sema and commits only on approve.
  mountControllers(app, [new RefundController()]);
  return app;
}
