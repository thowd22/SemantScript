# Execution plans

Several `sema` expressions in one program are not several models. They are
several small heads over one shared encoder, and the compiler works out, at
build time, which of them can run together and which must wait for another's
answer. That schedule is the neural execution plan. This page explains it with
one worked example: the source, the dependency graph the compiler derives, the
stages it emits, and what the runtime does with them.

## The worked example

Four decisions about one support request
([`execution-plan.sem.ts`](examples/execution-plan.sem.ts), compiled by the
docs test suite):

```ts
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
```

Three of the four read only `request`. The fourth interpolates `route`, a
value another sema expression produced. Nothing else about the four is
special: each is an ordinary expression with its own id, output support and
head.

## The dependency graph

The compiler follows every interpolated input back to what could have produced
it: variable initializers, aliases, destructuring and direct property writes,
across the file and across files. Whenever that chain reaches another sema
expression, it records a labeled dependency: producer, consumer and the name
of the consumer's input that carries the value. Here there is exactly one:

```text
route ──(template.route)──▶ template
priority
abusive
```

The analysis is a may-analysis. It says a value _could_ flow from `route` to
`template`, not that the two always execute together; branches, loops and
call frames are not modeled, and an overwritten binding can leave a harmless
false-positive edge. What it never does is miss a flow it can see, so a
consumer is never scheduled before a producer it might read. Values that reach
a sema expression through a function call or an accessor are outside the v1
analysis; if you want a chain the plan can see, keep it in locals (the refund
service's `screenOrder` does this).

## The stages

Stages are the minimum topological depth of that graph: independent
expressions share stage 0, and a consumer sits one stage after its deepest
producer. The bundle for the example (`semantscript.ir.v1.json`, ids
shortened) is:

```json
"executionPlan": {
  "stages": [
    { "index": 0, "functionIds": ["nf_7b1bf0ba…", "nf_d65ad302…", "nf_361ebd1f…"] },
    { "index": 1, "functionIds": ["nf_2690f615…"] }
  ],
  "dependencies": [
    { "producerFunctionId": "nf_7b1bf0ba…", "consumerFunctionId": "nf_2690f615…", "consumerInput": "route" }
  ]
}
```

`nf_7b1bf0ba…` is `route`, `nf_d65ad302…` is `priority`, `nf_361ebd1f…` is
`abusive` and `nf_2690f615…` is `template`. The plan is build metadata: it is
not part of any function's semantic identity, so adding an unrelated
expression to the file does not retrain the others.

A routed bundle (a program that names domain depths or uses `@domain`
headers) also lists `domains`, one per compile-time domain with its adapter,
encoder prefix, depth and members, and each stage lists the `adapterRefs` it
touches. The [language reference](language-reference.md#domains) covers the
headers and the [IR reference](ir-and-artifact-reference.md#ir-bundle-schemasir-bundlev1schemajson)
the fields.

## What the runtime does with a stage

Within a stage the runtime fuses by input, not by function. Entries whose
canonical inputs are byte-identical share one encoder pass and one adapter
pass per adapter, and then every function's own head runs. For the example,
stage 0 is one encoder pass, one adapter pass and three head passes; stage 1
is one more encoder pass over the request plus the route, and one head. The
pass counts are observable: `handle.callStage(entries)` returns
`{ results, passes: { encoder, adapter, head } }`, and `executeSemaPlan(plan,
provide)` runs a bundle's plan stage by stage, calling `provide(stage,
resultsSoFar)` for each stage's inputs so a later stage can be built from
earlier answers.

Two things follow from fusing by input. A stage of `N` functions over the same
record costs one encoder pass plus `N` small heads, which is where the
[scaling results](scaling-results.md) come from. And a stage of `B` distinct
inputs costs `B` encoder passes: the worker does not pad distinct inputs into
one batched run, so throughput per core is set by the encoder, whatever the
plan says.

Compiled code does not call `callStage` itself. Each `sema` expression becomes
`__sema.call(id, inputs)` in source order, and fusion happens through request
scopes: `withSemaScope(fn)` (opened per request by the framework) keeps the
embeddings a scope produces, so the calls made while it runs share encoder and
adapter passes over identical inputs exactly as a stage would, without changing
call order or the synchronous ABI. In the example, calling `handle(request)`
inside a scope encodes the request once for the first three expressions and
once more, with the route, for the fourth.

## Reading a plan for performance

- **Count distinct inputs per stage, not expressions.** One record and ten
  expressions is one encoder pass; ten records and one expression is ten.
- **A chain adds a full stage.** `template` waits for `route` and encodes a
  different text (the route is part of its input), so the example costs two
  encoder passes end to end. Chains are worth it when the earlier answer
  changes what the later expression should read; otherwise interpolate the
  same inputs and let the two share a stage.
- **Depth routing changes the cost of a pass, not the plan.** A domain at
  depth 6 runs six of the encoder's layers before its adapter; functions of
  one domain share their prefix pass, and functions at different depths each
  run their own. The refund service's three domains at depth 6 answer at 3 to
  5 ms per call on the CPU.
- **Pass counts are the test.** The framework returns them as the
  `x-sema-passes` header and `semaScopePasses()` reports them in a scope, so
  an unexpected extra encoder pass is visible in a test, not a profile.
