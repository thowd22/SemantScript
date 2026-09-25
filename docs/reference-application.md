# The reference application

[`examples/refund-service`](../examples/refund-service/README.md) is what a
SemantScript application is meant to look like: an Express API over Postgres
whose business policy is nine `sema` expressions in three domains, with the
TypeScript around them limited to persistence, routing, validation,
transactions and the constraints that bound each decision. Its README is the
full walkthrough with the measured tables; this page follows one expression
through the four things a build produces, side by side.

```text
refunds.sem.ts   decideRefund, refundMethod, refundRisk      POST /refunds/:orderId
tickets.sem.ts   ticketPriority, ticketQueue, needsHuman     POST /tickets/:ticketId/triage
orders.sem.ts    screenOrder: flag -> hold, escalation        POST /orders/:orderId/screen
```

## One expression, four views

**Source** (`src/refunds.sem.ts`): the policy in prose, three gold examples
and a complete constraint set.

```ts
export function refundRisk(customer: Customer, order: Order): RefundRisk {
  return sema<RefundRisk>({
    examples: [
      {
        inputs: {
          customer: { tier: "standard", priorRefunds: 0 },
          order: { total: 60, ageDays: 5, status: "paid" },
        },
        output: "low",
      },
      /* two more */
    ],
    constraints: [
      always(
        () =>
          order.status === "fraudulent" ||
          customer.priorRefunds > 2 ||
          order.total > 1000,
        "high",
      ),
      always(
        () =>
          order.status === "paid" &&
          customer.priorRefunds === 0 &&
          order.total < 100,
        "low",
      ),
      always(
        () =>
          order.status === "paid" &&
          !(customer.priorRefunds > 2 || order.total > 1000) &&
          !(customer.priorRefunds === 0 && order.total < 100),
        "medium",
      ),
    ],
  })`
    The risk of a refund request for the fraud team's queue. High when the
    order is fraudulent, the customer has more than two prior refunds or the
    order is above 1000. Low for a paid order under 100 from a customer with no
    prior refunds. Medium otherwise.
    Customer: ${customer}
    Order: ${order}
  `;
}
```

**Compiled JavaScript** (`dist/refunds.sem.js`): the text, examples and
constraints are gone; the site is a call by id.

```js
export function refundRisk(customer, order) {
  return __sema.call("nf_f5a00c9b…", { customer, order });
}
```

**IR record** (`dist/semantscript.ir.v1.json`, abridged): what the trainer
consumes.

```json
{
  "id": "nf_f5a00c9b…",
  "inputs": [
    {
      "name": "customer",
      "type": {
        "kind": "object",
        "fields": [
          { "name": "priorRefunds", "type": { "kind": "number" } },
          {
            "name": "tier",
            "type": {
              "kind": "union",
              "variants": [
                { "kind": "literal", "value": "enterprise" },
                { "kind": "literal", "value": "standard" }
              ]
            }
          }
        ]
      }
    },
    { "name": "order", "type": { "kind": "object", "fields": ["…"] } }
  ],
  "output": {
    "kind": "scalar",
    "tsType": "RefundRisk",
    "head": {
      "kind": "nominal",
      "sourceKind": "string-union",
      "support": ["high", "low", "medium"]
    }
  },
  "definition": {
    "template": ["…"],
    "examples": ["…"],
    "constraints": [
      {
        "kind": "always",
        "predicate": { "node": "binary", "operator": "||", "…": "…" },
        "output": "high"
      }
    ]
  },
  "model": {
    "encoder": "encoder.refund-service.depth-006",
    "encoderDepth": 6,
    "adapter": "adapter.refund-service.refunds",
    "heads": [{ "outputPath": "", "ref": "head.f5a00c9b….000" }]
  }
}
```

**Artifact manifest entry** (`.semantscript/artifact/releases/…/manifest.json`,
abridged): what the runtime loads.

```json
{
  "id": "nf_f5a00c9b…",
  "adapterRef": "adapter.refund-service.refunds",
  "heads": [
    {
      "outputPath": [],
      "headRef": "head.f5a00c9b….000",
      "type": {
        "kind": "nominal-string",
        "support": ["high", "low", "medium"]
      },
      "parameterization": "categorical-softmax",
      "calibration": {
        "method": "temperature-scaling",
        "temperature": 0.05,
        "ece": 0.0
      },
      "verification": { "accuracy": 1.0, "pairConsistency": 1.0 }
    }
  ],
  "runtime": {
    "resultMode": "value",
    "confidenceThreshold": null,
    "policy": "none",
    "fallbackRef": null
  },
  "verification": {
    "status": "passed",
    "accuracy": 1.0,
    "ece": 0.0,
    "attestedCases": 3,
    "exampleFailures": 0,
    "constraintViolations": 0
  },
  "trainingProvenance": {
    "teacher": "constraints/compiled-constraints-v2@sha256:954871d5…",
    "baseModel": "answerdotai/ModernBERT-base@…"
  }
}
```

The manifest's model block names `encoder.refund-service.depth-006` (a
six-layer prefix of the encoder, one ONNX graph) and the resources list one
adapter per domain and one head per expression. The release is 266 MB, most
of it the prefix.

## Domains and stages

Each `.sem.ts` file is a domain (the file stem is the default name), and the
transformer entry in `tsconfig.json` gives every domain depth 6. The bundle's
plan lists three domains on the one prefix and two stages: seven expressions
in stage 0 and, in stage 1, the shipment hold and the escalation, which take
the fraud flag `screenOrder` decides first. That chain lives in one function
in `orders.sem.ts` so the compiler can see it through the local.

## What is deterministic and why

The line tally in the README (from the TypeScript AST) puts 63% of the source
in sema expressions (46% excluding gold examples) and the rest in
persistence, routing, transactions, validation and glue. The deterministic
code is deterministic for a reason each time: SQL and row mapping because
data stays out of the weights; `transactional` and `gate` because the
decision only says whether a write happens; `require` because request
contracts are not policy; constraints because a rule the model must never
break belongs where the trainer labels from it and the verifier gates on it.
There is no hand-written policy branch outside a constraint.

## Training and results

The application trains from its own constraints: every expression's
constraint set is complete, so `npm run train` is `semantscript train
--teacher constraints`, the built-in constraints teacher labelling sampled
inputs on the ordinary `train_bundle` path, with no language model and no
API key. On 200 held-out inputs per expression through the Node runtime,
five of nine expressions score 200/200 and four 199/200 (each miss an input
next to a threshold), the verifier's ECE is at most 0.0026, and p50
latency per call is about 3.9 to 5.4 ms on the CPU at depth 6. The README has the per-expression table, the
verification story (four strict-gate failures on near-threshold amounts
before a recorded 0.5% tolerance, and a release that then recorded zero raw
violations) and the tests: bundle shape, the framework path over the runtime's
fixture artifact, and every controller over the trained artifact.

Run it:

```sh
npm install && npm run build   # at the repository root, once
cd examples/refund-service
npm install && npm run build && npm run train && npm test
npx semantscript run --artifact .semantscript/artifact dist/app.js \
  --call decideRefund --input '[{"tier":"standard","priorRefunds":1},{"total":88.5,"ageDays":12,"status":"paid"}]'
```
