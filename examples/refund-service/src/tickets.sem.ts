import { always, sema } from "@semantscript/core";

// Domain "tickets": support triage as three decisions over one ticket record.

export interface Ticket {
  category: "outage" | "billing" | "question" | "feature";
  impactedUsers: number;
  customerTier: "enterprise" | "standard";
}

export type Priority = "urgent" | "normal" | "low";
export type Queue = "engineering" | "billing" | "support";

export function ticketPriority(ticket: Ticket): Priority {
  return sema<Priority>({
    examples: [
      {
        inputs: {
          ticket: {
            category: "outage",
            impactedUsers: 400,
            customerTier: "enterprise",
          },
        },
        output: "urgent",
      },
      {
        inputs: {
          ticket: {
            category: "billing",
            impactedUsers: 3,
            customerTier: "standard",
          },
        },
        output: "normal",
      },
      {
        inputs: {
          ticket: {
            category: "feature",
            impactedUsers: 1,
            customerTier: "standard",
          },
        },
        output: "low",
      },
    ],
    constraints: [
      always(() => ticket.category === "outage", "urgent"),
      always(
        () =>
          ticket.category === "billing" && ticket.customerTier === "enterprise",
        "urgent",
      ),
      always(
        () =>
          ticket.category === "billing" && ticket.customerTier === "standard",
        "normal",
      ),
      always(
        () => ticket.category === "question" || ticket.category === "feature",
        "low",
      ),
    ],
  })`
    The priority of a support ticket. Outages are urgent. Billing problems are
    urgent for enterprise customers and normal for everyone else. Questions
    and feature ideas are low.
    Ticket: ${ticket}
  `;
}

export function ticketQueue(ticket: Ticket): Queue {
  return sema<Queue>({
    examples: [
      {
        inputs: {
          ticket: {
            category: "outage",
            impactedUsers: 400,
            customerTier: "enterprise",
          },
        },
        output: "engineering",
      },
      {
        inputs: {
          ticket: {
            category: "billing",
            impactedUsers: 3,
            customerTier: "standard",
          },
        },
        output: "billing",
      },
      {
        inputs: {
          ticket: {
            category: "question",
            impactedUsers: 1,
            customerTier: "standard",
          },
        },
        output: "support",
      },
    ],
    constraints: [
      always(() => ticket.category === "outage", "engineering"),
      always(() => ticket.category === "billing", "billing"),
      always(
        () => ticket.category === "question" || ticket.category === "feature",
        "support",
      ),
    ],
  })`
    Which team owns a ticket: outages go to engineering, billing problems to
    billing, questions and feature ideas to support.
    Ticket: ${ticket}
  `;
}

export function needsHuman(ticket: Ticket): boolean {
  return sema<boolean>({
    examples: [
      {
        inputs: {
          ticket: {
            category: "outage",
            impactedUsers: 400,
            customerTier: "enterprise",
          },
        },
        output: true,
      },
      {
        inputs: {
          ticket: {
            category: "billing",
            impactedUsers: 30,
            customerTier: "standard",
          },
        },
        output: true,
      },
      {
        inputs: {
          ticket: {
            category: "billing",
            impactedUsers: 3,
            customerTier: "standard",
          },
        },
        output: false,
      },
    ],
    constraints: [
      always(() => ticket.category === "outage", true),
      always(
        () => ticket.category === "billing" && ticket.impactedUsers > 10,
        true,
      ),
      always(
        () => ticket.category === "billing" && ticket.impactedUsers <= 10,
        false,
      ),
      always(
        () => ticket.category === "question" || ticket.category === "feature",
        false,
      ),
    ],
  })`
    Whether a person must look at the ticket before any automated reply: every
    outage, and a billing problem that affects more than ten users. Smaller
    billing problems, questions and feature ideas can be answered automatically.
    Ticket: ${ticket}
  `;
}
