import { sema } from "@semantscript/core";

// One support request, four decisions. `route` and `priority` read only the
// request, so they are independent and share stage 0 with `abusive`; the
// reply template depends on the route, so it sits in stage 1.

interface SupportRequest {
  subject: string;
  body: string;
  customerTier: "enterprise" | "standard";
}

export function handle(request: SupportRequest) {
  const route = sema<"billing" | "engineering" | "support">`
    Which team should own this request?
    Request: ${request}
  `;
  const priority = sema<"urgent" | "normal" | "low">`
    How urgently should we answer?
    Request: ${request}
  `;
  const abusive = sema<boolean>`
    Is the message abusive toward the agent?
    Request: ${request}
  `;
  const template = sema<"acknowledge" | "ask-for-details" | "escalate">`
    Which reply template fits a request owned by ${route}?
    Request: ${request}
  `;
  return { route, priority, abusive, template };
}
