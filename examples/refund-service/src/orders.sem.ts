import { always, sema } from "@semantscript/core";

// Domain "orders": fraud screening as a chain of three decisions inside one
// function. The flag feeds the hold and the escalation, so the compiler records
// the two dependencies and places the consumers in a second stage of the
// execution plan; the runtime encodes each stage's inputs once.

export interface OrderFacts {
  total: number;
}

export interface Signals {
  mismatchedAddress: boolean;
  ordersLastHour: number;
  chargebacks: number;
}

export interface Account {
  priorRefunds: number;
}

export type FraudFlag = "flag" | "watch" | "clear";
export type Escalation = "none" | "analyst" | "legal";

export interface Screening {
  flag: FraudFlag;
  hold: boolean;
  escalation: Escalation;
}

export function screenOrder(
  order: OrderFacts,
  signals: Signals,
  account: Account,
): Screening {
  const flag = sema<FraudFlag>({
    examples: [
      {
        inputs: {
          order: { total: 6200 },
          signals: {
            mismatchedAddress: true,
            ordersLastHour: 5,
            chargebacks: 1,
          },
        },
        output: "flag",
      },
      {
        inputs: {
          order: { total: 2600 },
          signals: {
            mismatchedAddress: true,
            ordersLastHour: 2,
            chargebacks: 0,
          },
        },
        output: "watch",
      },
      {
        inputs: {
          order: { total: 88.5 },
          signals: {
            mismatchedAddress: false,
            ordersLastHour: 1,
            chargebacks: 0,
          },
        },
        output: "clear",
      },
    ],
    constraints: [
      always(() => signals.chargebacks > 0, "flag"),
      always(
        () =>
          signals.chargebacks === 0 &&
          signals.mismatchedAddress &&
          signals.ordersLastHour > 3,
        "flag",
      ),
      always(
        () =>
          signals.chargebacks === 0 &&
          signals.mismatchedAddress !== signals.ordersLastHour > 3,
        "watch",
      ),
      always(
        () =>
          signals.chargebacks === 0 &&
          !signals.mismatchedAddress &&
          signals.ordersLastHour <= 3,
        "clear",
      ),
    ],
  })`
    Screen an order for fraud. Any prior chargeback flags it. A mismatched
    billing address together with more than three orders in the last hour flags
    it; either signal alone puts it on watch; neither clears it.
    Order: ${order}
    Signals: ${signals}
  `;

  const hold = sema<boolean>({
    examples: [
      { inputs: { order: { total: 6200 }, flag: "flag" }, output: true },
      { inputs: { order: { total: 2600 }, flag: "watch" }, output: true },
      { inputs: { order: { total: 88.5 }, flag: "clear" }, output: false },
    ],
    constraints: [
      always(() => flag === "flag", true),
      always(() => flag === "watch" && order.total > 2000, true),
      always(() => flag === "watch" && order.total <= 2000, false),
      always(() => flag === "clear", false),
    ],
  })`
    Whether to hold the shipment: always for a flagged order, for a watched
    order only when it is worth more than 2000, never for a cleared order.
    Order: ${order}
    Flag: ${flag}
  `;

  const escalation = sema<Escalation>({
    examples: [
      {
        inputs: {
          order: { total: 6200 },
          flag: "flag",
          account: { priorRefunds: 1 },
        },
        output: "legal",
      },
      {
        inputs: {
          order: { total: 900 },
          flag: "flag",
          account: { priorRefunds: 1 },
        },
        output: "analyst",
      },
      {
        inputs: {
          order: { total: 2600 },
          flag: "watch",
          account: { priorRefunds: 0 },
        },
        output: "none",
      },
    ],
    constraints: [
      always(() => flag === "flag" && order.total > 5000, "legal"),
      always(() => flag === "flag" && order.total <= 5000, "analyst"),
      always(() => flag === "watch" && account.priorRefunds > 2, "analyst"),
      always(() => flag === "watch" && account.priorRefunds <= 2, "none"),
      always(() => flag === "clear", "none"),
    ],
  })`
    Who reviews the order: legal for a flagged order above 5000, an analyst for
    any other flagged order and for a watched order from an account with more
    than two prior refunds; nobody otherwise.
    Order: ${order}
    Flag: ${flag}
    Account: ${account}
  `;

  return { flag, hold, escalation };
}
