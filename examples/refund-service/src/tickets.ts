import {
  Controller,
  Post,
  RequestContext,
  require,
  transactional,
} from "@semantscript/framework";

import { db } from "./db.js";
import { needsHuman, ticketPriority, ticketQueue } from "./tickets.sem.js";

interface TicketRow {
  id: string;
  category: "outage" | "billing" | "question" | "feature";
  impacted_users: number;
  customer_tier: "enterprise" | "standard";
}

/**
 * POST /tickets/:ticketId/triage stamps a stored ticket with its priority,
 * queue and whether a person must look first. The three expressions share one
 * input and one domain, so they run over one encoding of the ticket.
 */
@Controller("/tickets")
export class TicketController {
  @Post("/:ticketId/triage")
  async triage(context: RequestContext) {
    const ticketId = context.params["ticketId"];
    require(typeof ticketId === "string" &&
      ticketId.length > 0, "ticketId is required", 400);
    return transactional(db, async (tx) => {
      const { rows } = await tx.query<TicketRow>(
        "SELECT id, category, impacted_users, customer_tier FROM tickets WHERE id = $1",
        [ticketId],
      );
      const row = rows[0];
      require(row !== undefined, "no such ticket", 404);
      const ticket = {
        category: row.category,
        impactedUsers: row.impacted_users,
        customerTier: row.customer_tier,
      };
      const triaged = {
        priority: ticketPriority(ticket),
        queue: ticketQueue(ticket),
        needsHuman: needsHuman(ticket),
      };
      await tx.query(
        "UPDATE tickets SET priority = $2, queue = $3, needs_human = $4 WHERE id = $1",
        [ticketId, triaged.priority, triaged.queue, triaged.needsHuman],
      );
      return { ticketId, ...triaged };
    });
  }
}
