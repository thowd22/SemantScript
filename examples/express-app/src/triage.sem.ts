import { sema } from "@semantscript/core";

export type Priority = "urgent" | "normal" | "low";

/** Priority of a support ticket, decided by the trained artifact at request time. */
export function triage(subject: string, body: string): Priority {
  return sema<Priority>`
    Priority of a customer support ticket. "urgent" when service is down,
    data is lost or a deadline is today; "low" for questions and feature
    ideas; "normal" otherwise.
    Subject: ${subject}
    Body: ${body}
  `;
}
