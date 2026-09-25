# Refund service: a reference application

A refunds, tickets and orders service where the business policy lives in sema
expressions and the TypeScript around them is limited to persistence,
routing, validation, transactions and the constraints that bound each
decision. It is what a SemantScript application is meant to look like, not a
demo with one neural call: nine decisions in three domains, an Express API over
Postgres, and an artifact trained from the code itself.

```text
src/refunds.sem.ts   domain "refunds":  decideRefund, refundMethod, refundRisk
src/tickets.sem.ts   domain "tickets":  ticketPriority, ticketQueue, needsHuman
src/orders.sem.ts    domain "orders":   screenOrder = flag -> hold, escalation (two stages)
src/refunds.ts       POST /refunds/:orderId          reads rows, decides, commits only on approve
src/tickets.ts       POST /tickets/:ticketId/triage  stamps priority, queue, needs-human
src/orders.ts        POST /orders/:orderId/screen    runs the fraud chain, holds the shipment
src/db.ts            PGlite (Postgres 17 in process), schema and seed rows
src/app.ts           createApp(): Express + controllers; re-exports the sema functions
src/server.ts        loadSemaArtifact(); listen
scripts/heldout.py   the held-out set measure scores, labelled by the constraints
scripts/measure.mjs  held-out accuracy and latency per expression through the runtime
scripts/tally.mjs    lines of logic per category (the table below)
test/app.test.mjs    the bundle's domains and plan; handlers over PGlite
```

## Build, train, run

The linked packages resolve their own dependencies from the repository's
`node_modules`, so install and build the repository root first. `npm test`
passes before any training: the trained-artifact test skips with
`train first: npm run train` until the artifact exists.

```sh
npm install && npm run build  # at the repository root, once
cd examples/refund-service
npm install                   # links ../../compiler, ../../runtime, ../../framework, ../../cli
npm run build                 # tspc: dist/*.js and dist/semantscript.ir.v1.json
npm test                      # bundle shape and fixture-artifact handlers, no training needed
npm run train                 # semantscript train --teacher constraints on the GPU (the Python training extra in the active python3; activate .venv first), ~11 min: .semantscript/artifact, train-report.json, then heldout.json
npm test                      # again: now the trained-artifact handlers run too
npm run measure               # held-out accuracy, ECE and latency per expression
npm start                     # http://localhost:3000
```

Any exported sema function runs directly through the CLI once the artifact
exists (`--input` spreads a JSON array as positional arguments):

```sh
npx semantscript run --artifact .semantscript/artifact dist/app.js \
  --call decideRefund --input '[{"tier":"standard","priorRefunds":1},{"total":88.5,"ageDays":12,"status":"paid"}]'
"approve"

npx semantscript run --artifact .semantscript/artifact dist/app.js \
  --call screenOrder --input '[{"total":6200},{"mismatchedAddress":true,"ordersLastHour":5,"chargebacks":1},{"priorRefunds":1}]'
{
  "flag": "flag",
  "hold": true,
  "escalation": "legal"
}
```

Or over HTTP against the seeded rows:

```sh
curl -X POST localhost:3000/refunds/o1          # {"committed":true,"value":{"orderId":"o1","decision":"approve","method":"original-payment","risk":"medium","refunded":true}}
curl -X POST localhost:3000/refunds/o2          # {"committed":false,"reason":"decision deny (risk high): no refund written"}
curl -X POST localhost:3000/tickets/t1/triage   # {"committed":true,"value":{"ticketId":"t1","priority":"urgent","queue":"engineering","needsHuman":true}}
curl -X POST localhost:3000/orders/o4/screen    # {"committed":true,"value":{"orderId":"o4","flag":"flag","shipmentHeld":true,"escalation":"legal"}}
```

## The source

One expression, in full, from `src/refunds.sem.ts`:

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
      {
        inputs: {
          customer: { tier: "standard", priorRefunds: 1 },
          order: { total: 400, ageDays: 5, status: "paid" },
        },
        output: "medium",
      },
      {
        inputs: {
          customer: { tier: "enterprise", priorRefunds: 0 },
          order: { total: 2600, ageDays: 5, status: "paid" },
        },
        output: "high",
      },
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

Three parts, and each has a job in the build:

- **The policy text** is the specification a teacher reads. This service has
  no language-model teacher (see below), so here it is documentation for the
  next developer and the text the encoder sees is the canonical input, not
  the template.
- **The constraints** bound the model: the verifier refuses to publish a
  release whose raw predictions violate an active `always` or `never` on more
  than a configured share of the corpus. This service trains with a tolerance
  of 0.5% (`--max-constraint-violation-rate 0.005`, recorded in the report).
  Under the strict zero gate, four earlier runs each failed one to three of
  the nine expressions on a handful of amounts within a few units of a
  threshold (4,976 against a 5,000 rule, 1,045.5 against 1,000; raw rates of
  0.04% to 0.25%), the limit of a six-layer encoder reading numbers as text
  rather than a policy error; the published release happened to record zero
  raw violations on every expression. Enforcement is at release time, not per
  call: the runtime returns the calibrated prediction. In this service every expression's
  constraints are _complete_: for any input exactly one output satisfies them
  all. That is a design choice, not a language requirement, and it is what
  lets the built-in constraints teacher label the corpus with no language
  model.
- **The gold examples** are attested cases the verifier must reproduce
  exactly. They are also the seed rows in `src/db.ts`, so the HTTP examples
  above are verified behavior.

Every rule the policy states is a constraint. The interesting question is
then what the model adds: it generalizes the constraints to a smooth decision
over the whole input space, the constraints guarantee the corners, and the
handler code never re-implements the policy.

The domains follow the files. `src/refunds.sem.ts` compiles to domain
`refunds`, and so on; no `@domain` header is needed because the default
domain is the file stem (or the enclosing controller class). The three
depths are set in `tsconfig.json`, on the transformer plugin entry:

```json
{
  "transform": "@semantscript/compiler/transformer",
  "application": "refund-service",
  "domainDepths": { "refunds": 6, "tickets": 6, "orders": 6 }
}
```

Every domain runs the first six of the encoder's twenty-two layers before its
own adapter, the depth the refund benchmark measured as the best latency point
with no accuracy loss on a policy of this shape (see the depth sweep in
[`benchmarks/refund`](../../benchmarks/refund/README.md)).

`src/orders.sem.ts` is the one place with a dependency between expressions:
`screenOrder` decides the flag, then the hold and the escalation take the flag
as an input. The compiler follows the local through the function and records
two labeled dependencies, so the plan has two stages.

## The compiled output

`tspc` emits the same JavaScript `tsc` would, with each sema expression
replaced by a runtime call carrying the function's id. The text, the
constraints and the examples are gone from the program; they live in the IR
bundle the trainer consumes. `dist/orders.sem.js`:

```js
import { __sema as __sema } from "@semantscript/core";
export function screenOrder(order, signals, account) {
  const flag = __sema.call("nf_6e6a18ee…", { order, signals });
  const hold = __sema.call("nf_eb1c3a8a…", { order, flag });
  const escalation = __sema.call("nf_b03d7f79…", { order, flag, account });
  return { flag, hold, escalation };
}
```

The controllers compile unchanged (standard decorators, `tsc` output). The
source map keeps every runtime call on its `.sem.ts` line, so a stack trace
under `--enable-source-maps` names `src/orders.sem.ts:32`, not the emitted
line.

`dist/semantscript.ir.v1.json` is the bundle: nine `NeuralFunction` IR
records (inputs typed from TypeScript, output support, template, constraints
lowered to a predicate tree, examples, the model binding) and the execution
plan. The plan is where the domains and the chain show up:

```json
"executionPlan": {
  "domains": [
    { "name": "orders",  "adapterRef": "adapter.refund-service.orders",  "encoderRef": "encoder.refund-service.depth-006", "encoderDepth": 6, "functionIds": [3 ids] },
    { "name": "refunds", "adapterRef": "adapter.refund-service.refunds", "encoderRef": "encoder.refund-service.depth-006", "encoderDepth": 6, "functionIds": [3 ids] },
    { "name": "tickets", "adapterRef": "adapter.refund-service.tickets", "encoderRef": "encoder.refund-service.depth-006", "encoderDepth": 6, "functionIds": [3 ids] }
  ],
  "dependencies": [
    { "producerFunctionId": "nf_6e6a18ee…", "consumerFunctionId": "nf_eb1c3a8a…", "consumerInput": "flag" },
    { "producerFunctionId": "nf_6e6a18ee…", "consumerFunctionId": "nf_b03d7f79…", "consumerInput": "flag" }
  ],
  "stages": [
    { "index": 0, "functionIds": [7 ids], "adapterRefs": ["adapter.refund-service.orders", "adapter.refund-service.refunds", "adapter.refund-service.tickets"] },
    { "index": 1, "functionIds": [2 ids], "adapterRefs": ["adapter.refund-service.orders"] }
  ]
}
```

## Training without a language model

`npm run train` is plain `semantscript train --teacher constraints` with
this service's recipe (800 cases per expression, 12 epochs keeping the best
held-out epoch, counterfactual ratio 0.5, the 0.5% violation tolerance, on
`cuda`), then `npm run heldout`. The built-in constraints teacher is the
ordinary `train_bundle` path (synthetic corpus, adversarial cases, joint
training over the shared encoder, calibration, verification, build cache,
artifact export) with labels from the constraints instead of a language
model. Because every expression's constraints are complete, it samples
structured inputs from the IR's input types (numbers over ranges inferred
from the thresholds the predicates mention and the gold examples, with three
draws in ten on or beside a threshold; counts that are often zero; amounts on
a log scale) and labels each input with the one output that violates no
constraint. Boundary pairs are two labelled inputs one field apart on either
side of a predicate; counterfactual twins are single-field edits that change
the label. The corpus keeps only inputs that one edit can move across the
policy, since any training row may be chosen as a counterfactual anchor. No
range is configured here: the inferred ones train every expression (a
`semantscript.teacher.toml` with `backend = "constraints"` and
`[teacher.ranges]` would override them; see
[teachers](../../docs/teachers.md)).

The teacher's identity (provider `constraints`, model
`compiled-constraints-v2`, and its sampling configuration digest
`954871d5…`) is recorded in every function's training provenance like any
other teacher, so the artifact says where its labels came from. Gold examples
are the attested cases; teacher labels are never recorded as human-authored.
`scripts/heldout.py` draws 200 inputs per expression from the teacher's
separate `heldout` stream and drops any input found in a cached training or
adversarial dataset, so the held-out set is disjoint from what was trained.

This is the real-input path decision-7 describes: when the policy is fully
stated as constraints, no API key and no language model are needed to train
it. A policy with judgment in it (the text says more than the constraints do)
needs a teacher that can read; `semantscript train` with an Anthropic or
Ollama teacher takes the same bundle, or the constraints teacher with that
teacher as its `[teacher.fallback]` for the inputs the constraints leave open.

## The artifact

`npm run train` publishes `.semantscript/artifact`, the directory
`loadSemaArtifact()` finds with no path from `dist/server.js`. It is one
immutable release with a manifest, and the routed layout is visible in its
resources:

```text
.semantscript/artifact/  (release 1bf655e6e92d…, 279 MB)
  tokenizer/tokenizer.json                                   tokenizer     1.6 MB   ref tokenizer.main
  models/encoder/depth-006.onnx                              encoder     275.3 MB   ref encoder.refund-service.depth-006
  models/adapters/adapter-refund-service-orders.onnx         adapter       0.4 MB   ref adapter.refund-service.orders
  models/adapters/adapter-refund-service-refunds.onnx        adapter       0.4 MB   ref adapter.refund-service.refunds
  models/adapters/adapter-refund-service-tickets.onnx        adapter       0.4 MB   ref adapter.refund-service.tickets
  models/heads/<function id>/head-000.onnx  (nine)          head          0.03 MB  ref head.<function id prefix>.value
```

The manifest's model block names the depth-6 prefix as the application's
encoder and each function's entry binds its domain adapter, so the runtime
encodes a request once and runs the domain's adapter and the function's head
on top; the two stages of `screenOrder` are two rounds inside one request
scope.

## What moved, what stayed, and why

The service's business policy is entirely in sema expressions. What the
TypeScript around them does, by design:

- **Persistence** (`src/db.ts`, the SQL in each controller): rows in, rows
  out. A sema expression cannot take a database client as an input (the
  compiler rejects it, diagnostic 9112), so handlers read rows and hand the
  decisions plain values.
- **Transactions**: `transactional` opens one, the decision gates it
  (`gate`), and a non-approval rolls back with the decision as the reason.
  Nothing neural touches the transaction; it only decides whether the write
  happens.
- **Validation**: `require` guards on request input (`orderId` present) and
  lookups (404 for a missing ticket). These are not policy, they are
  contract.
- **Routing**: decorated controllers, the Express mount and startup.
- **Constraints**: the one kind of business rule that stays deterministic,
  and it stays deterministic _inside_ the sema expression so that the
  trainer labels from it and the verifier gates the release on it.

The line tally (`npm run tally`, which walks the TypeScript AST and gives
each non-blank, non-comment line the category of the innermost node that
claims it):

| Category                                                                        | Lines | Share | By file                                                                                                                        |
| ------------------------------------------------------------------------------- | ----: | ----: | ------------------------------------------------------------------------------------------------------------------------------ |
| sema: policy text: the natural-language policy inside a sema template           |    62 |    8% | orders.sem.ts 21, refunds.sem.ts 24, tickets.sem.ts 17                                                                         |
| sema: constraints: always/never predicates that bound the policy                |   143 |   20% | orders.sem.ts 36, refunds.sem.ts 67, tickets.sem.ts 40                                                                         |
| sema: gold examples: attested input/output pairs the verifier must reproduce    |   231 |   32% | orders.sem.ts 66, refunds.sem.ts 69, tickets.sem.ts 96                                                                         |
| sema: declaration: the sema call, its output type and the function around it    |    28 |    4% | orders.sem.ts 10, refunds.sem.ts 9, tickets.sem.ts 9                                                                           |
| decision glue: calling a sema function with plain values and reading its result |    21 |    3% | orders.ts 13, refunds.ts 3, tickets.ts 5                                                                                       |
| types: interfaces, type aliases and row shapes shared by both sides             |    63 |    9% | orders.sem.ts 18, orders.ts 8, refunds.sem.ts 15, refunds.ts 9, tickets.sem.ts 7, tickets.ts 6                                 |
| persistence: SQL, schema, seed rows and row-to-value mapping                    |    78 |   11% | db.ts 33, orders.ts 15, refunds.ts 16, tickets.ts 14                                                                           |
| transactions: transactional, gate and rollback                                  |    14 |    2% | orders.ts 2, refunds.ts 10, tickets.ts 2                                                                                       |
| validation: require guards on request input and lookups                         |     8 |    1% | orders.ts 3, refunds.ts 2, tickets.ts 3                                                                                        |
| routing: controllers, decorators, responses, Express wiring and startup         |    40 |    5% | app.ts 11, orders.ts 8, refunds.ts 8, server.ts 5, tickets.ts 8                                                                |
| imports and exports: module plumbing                                            |    45 |    6% | app.ts 10, db.ts 1, orders.sem.ts 1, orders.ts 9, refunds.sem.ts 1, refunds.ts 11, server.ts 2, tickets.sem.ts 1, tickets.ts 9 |
| total                                                                           |   733 |  100% | sema expressions 464 (63%)                                                                                                     |

Sema expressions are 63% of the source, or 46% (233 of 502 lines) with the
gold examples left out, since Prettier puts one field of each example on its
own line. The rest is persistence, routing and glue with no policy in it.
There is no hand-written `if (order.ageDays > 90)` anywhere outside a
constraint.

## Accuracy, calibration and latency per expression

Measured on the development machine (AMD Ryzen 9 9900X, WSL2, Node
v22.22.0, CPU inference through ONNX Runtime) with `npm run measure`:
200 held-out inputs per expression drawn by `scripts/heldout.py` from the
constraints teacher's `heldout` stream (a different stream than the training
corpus, with every cached training input excluded) and labeled by the
constraints, scored through the Node runtime; the ECE is the verifier's
15-bin calibration error from `.semantscript/train-report.json`. Latency is
per call, after a 20-call warm-up, one expression at a time.

| Expression               | Domain  | Held-out accuracy | Verification ECE | p50 ms | p95 ms |
| ------------------------ | ------- | ----------------- | ---------------- | ------ | ------ |
| `src/orders.sem.ts:36`   | orders  | 100.0% (200/200)  | 0.0000           | 5.63   | 8.01   |
| `src/orders.sem.ts:103`  | orders  | 99.5% (199/200)   | 0.0026           | 3.69   | 4.89   |
| `src/orders.sem.ts:122`  | orders  | 99.5% (199/200)   | 0.0000           | 4.43   | 6.08   |
| `src/refunds.sem.ts:28`  | refunds | 100.0% (200/200)  | 0.0000           | 5.39   | 7.33   |
| `src/refunds.sem.ts:97`  | refunds | 99.5% (199/200)   | 0.0000           | 4.75   | 5.98   |
| `src/refunds.sem.ts:143` | refunds | 99.5% (199/200)   | 0.0000           | 5.25   | 6.31   |
| `src/tickets.sem.ts:15`  | tickets | 100.0% (200/200)  | 0.0000           | 4.72   | 6.57   |
| `src/tickets.sem.ts:74`  | tickets | 100.0% (200/200)  | 0.0000           | 4.67   | 6.74   |
| `src/tickets.sem.ts:123` | tickets | 100.0% (200/200)  | 0.0000           | 4.52   | 6.07   |

Held-out misses (each one input next to a threshold):

- `src/orders.sem.ts:103`: 1 of 200, inputs `{"order":{"total":2011},"flag":"watch"}` expected `true` and got `false` (a 2,000 rule).
- `src/orders.sem.ts:122`: 1 of 200, inputs `{"order":{"total":5621},"flag":"flag","account":{"priorRefunds":2}}` expected `"legal"` and got `"analyst"` (a 5,000 rule).
- `src/refunds.sem.ts:97`: 1 of 200, inputs `{"order":{"ageDays":0,"status":"fraudulent","total":739},"payment":{"method":"card"}}` expected `"original-payment"` and got `"store-credit"`.
- `src/refunds.sem.ts:143`: 1 of 200, inputs `{"customer":{"priorRefunds":0,"tier":"standard"},"order":{"ageDays":14,"status":"paid","total":2169}}` expected `"high"` and got `"medium"` (a 2,000 rule).

The p50 across expressions spans 3.69 to 5.63 ms per call on the CPU; the two-stage `screenOrder` chain costs three calls in a request. The verifier's ECE is on the calibration split; accuracy here is on fresh inputs the trainer never saw.

Training: 2026-09-25, `npm run train` (the built-in constraints teacher, `semantscript train --teacher constraints`), 9 expressions jointly over one ModernBERT-base encoder cut to 6 layers and three adapters, 14456 rows in all (800 sampled cases plus boundary pairs and counterfactual twins per expression, three gold examples each), 12 epochs with the best held-out epoch kept (selected epoch 4), on an AMD Radeon RX 9070 XT, 11 minutes wall clock (11:02 by `/usr/bin/time`, with 42 minutes of user CPU time across cores; the split between phases was not measured, and the former driver's stated ~4 minutes was not re-timed) including generation, export and the held-out draw. Every expression passed verification with accuracy 1.0000 and ECE at most 0.0026; the configured raw-violation tolerance was 0.5% and the published release (`1bf655e6e92d…`) recorded 0 constraint violations across 14,456 verification records. Earlier releases scored on their own held-out draws: the first built-in-teacher release (`2f9eb3e890d1…`, algorithm v1) 200/200 on seven expressions and 199/200 on two, and the example's former custom driver (`55efd40c3dca…`) 200/200 on eight and 199/200 on one. The report is `.semantscript/train-report.json`; the build cache under `.semantscript/cache` makes an unchanged rebuild a no-op.

## Tests

`npm test` runs three tests. The first reads the bundle and checks the three
domains, the depth binding, the two flag dependencies and the two stages. The
second drives the refund controller over PGlite with the runtime's fixture
artifact re-keyed to this app's refund functions: the fixture answers
`review`, so the gate rolls back and no row is written, which exercises the
framework path without a trained model. The third needs `npm run train` first
and drives all three controllers against the seeded rows through the trained
artifact; it is skipped with a message when the artifact is absent.
