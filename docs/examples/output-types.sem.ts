// Every v1 output kind: boolean, string union, enums, Ordinal, BoundedInt,
// BoundedNumber and a flat interface of scalar fields.
import {
  sema,
  type BoundedInt,
  type BoundedNumber,
  type Ordinal,
} from "@semantscript/core";

enum Channel {
  Email = "email",
  Chat = "chat",
}

enum Priority {
  Low = 1,
  High = 2,
}

type Severity = Ordinal<["low", "medium", "high", "critical"]>;
type Stars = BoundedInt<1, 5>;
type Percent = BoundedNumber<0, 100, 0.5>;

interface Triage {
  urgency: Ordinal<["low", "medium", "high"]>;
  route: "self-service" | "agent";
  abusive: boolean;
  score: BoundedInt<0, 10>;
}

export function classify(message: string) {
  const spam = sema<boolean>`Is this message spam? Message: ${message}`;
  const tone = sema<
    "positive" | "negative" | "neutral"
  >`Classify the tone. Message: ${message}`;
  const channel = sema<Channel>`Which support channel fits best? Message: ${message}`;
  const priority = sema<Priority>`How urgently should we answer? Message: ${message}`;
  const severity = sema<Severity>`Rate the incident severity. Message: ${message}`;
  const stars = sema<Stars>`How many stars would this reviewer give? Message: ${message}`;
  const percent = sema<Percent>`How likely is a refund request, as a percentage? Message: ${message}`;
  const triage = sema<Triage>`Triage this support message. Message: ${message}`;
  return { spam, tone, channel, priority, severity, stars, percent, triage };
}
