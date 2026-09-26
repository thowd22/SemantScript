# Framework guide

`@semantscript/framework` is the thin layer between an HTTP stack an
application already has and the sema expressions its handlers call. It owns
no server and no ORM. It gives a handler three things: a place to put
deterministic guards around a neural result, one request scope so the
expressions in a request share encoder passes, and a transaction shape in
which a decision gates a write. This guide covers all of it with the
[refund service](reference-application.md) as the running example.

## Controllers

A controller is a class with standard TC39 decorators (TypeScript 5 syntax,
no `experimentalDecorators`):

```ts
import {
  Controller,
  Post,
  RequestContext,
  reply,
  require,
} from "@semantscript/framework";
import { triage } from "./triage.sem.js";

@Controller("/tickets")
export class TicketController {
  @Post("/triage")
  triage({ body }: RequestContext) {
    const { subject, text } = body as { subject?: string; text?: string };
    require(typeof subject === "string" &&
      subject.length > 0, "subject is required", 400);
    const priority = triage(subject, text ?? "");
    require(priority !== "urgent" ||
      text !== undefined, "urgent tickets need a body", 409);
    return { subject, priority };
  }
}
```

- `@Controller(prefix)` on the class and `@Get`, `@Post`, `@Put`, `@Patch`,
  `@Delete` on named methods declare routes; `routesOf(instance)` lists them
  in declaration order with bound handlers. Method decorators queue their
  routes and the class decorator binds them, so decorate the class last (as
  TypeScript does).
- A handler receives one `RequestContext`: `method`, `path`, `params` (route
  parameters as strings), `query`, `headers` and the parsed `body`. It may be
  synchronous or async.
- A returned value is answered as JSON 200. `reply(status, body, headers?)`
  returns a `Reply` for any other status. A thrown `RequirementError` is
  answered as its status with `{ error, code: "SEMA_REQUIREMENT_FAILED" }`;
  any other exception is 500 with the message.
- `handle(route, context)` is the one function every integration calls; it
  is public so a custom server or a test can drive a route without HTTP.

## Guards before and after the model

`require(condition, message, status = 422)` is a plain assertion. Because
sema calls are synchronous, order in the handler is order of evaluation: a
guard written above an expression runs before the model is consulted (a
failing pre-guard performs no encoder pass), and a guard written below sees
the model's value. Use pre-guards for request contracts (a field is present,
an id is well formed, a row exists) and post-guards for business invariants
the model must not be allowed to violate on its own (an urgent ticket needs
a body; a large denied refund needs a human). Constraints in the expression
itself are for rules the trainer should learn and the verifier should check;
guards are for rules the handler enforces regardless of what was learned.

## Request scopes and pass counts

Every handler runs inside the runtime's `withSemaScope`, so the sema
expressions it evaluates share encoder and adapter passes over identical
inputs, the way one [execution plan](execution-plans.md) stage fuses them,
and the embeddings kept for that are dropped when the response is produced.
Nothing in the handler changes: calls stay synchronous and in source order.

The response carries `x-sema-passes: <encoder>/<adapter>/<head>`. For the
refund service's `POST /refunds/:orderId`, which evaluates three expressions
over the same customer and order, that is `1/1/3` on the approve path. A test
that asserts the header catches an accidental extra encoding (an input object
rebuilt between calls with a different key order, say) without a profiler.
`semaScopePasses()` reports the same counts from inside a handler.

## Mounting

**Express** (and Nest's Express adapter):

```ts
import express from "express";
import { loadSemaArtifact } from "@semantscript/core";
import { mountControllers } from "@semantscript/framework";

const app = express();
app.use(express.json());
await loadSemaArtifact();
mountControllers(app, [new TicketController(), new RefundController()]);
app.listen(3000);
```

`mountControllers` registers every route on any object with `get`, `post`,
… registration methods (an app or a router) and returns the list it mounted.
Handlers written without decorators can still share passes: `app.use(semaRequestMiddleware())`
opens a scope for the rest of the request and closes it when the response
finishes. Nest users add the middleware globally.

**Next.js App Router**:

```ts
// app/api/tickets/[...path]/route.ts
export const { GET, POST } = nextRouteHandlers(new TicketController());
```

Handlers take the web `Request`, read `params` from the route context and
answer with `Response.json` and the same header. Load the artifact once per
server process (see the [Next.js build tool page](build-tools/next.md)).

## Persistence and transactions

The rule the framework enforces is that data stays out of the weights: a
handler reads rows, hands the decision plain values, and the decision gates
the write. A sema expression cannot take a database client as an input (the
compiler rejects it, diagnostic 9112) and the runtime serializes only plain
data, so an expression can neither read nor write beyond what it was handed.

```ts
import { gate, rollback, transactional } from "@semantscript/framework";

@Controller("/refunds")
export class RefundController {
  @Post("/:orderId")
  async request(context: RequestContext) {
    const orderId = context.params["orderId"];
    require(typeof orderId === "string" &&
      orderId.length > 0, "orderId is required", 400);
    return transactional(db, async (tx) => {
      const { rows } = await tx.query<RefundRow>(
        `SELECT o.id, o.total, o.age_days, o.status, o.payment_method, c.tier, c.prior_refunds
           FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.id = $1`,
        [orderId],
      );
      const row = rows[0];
      if (row === undefined) return rollback("no such order");
      const customer = { tier: row.tier, priorRefunds: row.prior_refunds };
      const order = {
        total: Number(row.total),
        ageDays: row.age_days,
        status: row.status,
      };
      const decision = decideRefund(customer, order);
      const risk = refundRisk(customer, order);
      return gate(
        decision,
        (value) => value === "approve",
        async () => {
          const method = refundMethod(order, { method: row.payment_method });
          await tx.query(
            "INSERT INTO refunds (order_id, decision, method, risk) VALUES ($1, $2, $3, $4)",
            [orderId, decision, method, risk],
          );
          return { orderId, decision, method, risk, refunded: true };
        },
        `decision ${decision} (risk ${risk}): no refund written`,
      );
    });
  }
}
```

- `transactional(client, body)` runs `BEGIN`, the body, then `COMMIT`, or
  `ROLLBACK` when the body returns `rollback(reason)` or throws. It takes any
  client with the node-postgres query shape: a `pg` Client, a `pg` Pool
  (checked out and released around the transaction) or PGlite (Postgres in
  WebAssembly, which is what the examples and tests use so no server is
  needed).
- The outcome is `{ committed: true, value }` or `{ committed: false, reason, value? }`;
  returned from a handler it is the JSON body, so a client sees whether the
  write happened and why not.
- `gate(decision, accept, commit, reason?)` is the usual tail: run `commit`
  when the decision satisfies `accept`, else roll back naming the decision.
  The default reason quotes the decision.
- Expressions that only inform the write (here `refundRisk`) run inside the
  same transaction and scope; the risk is recorded with the refund, and on a
  rollback nothing is recorded at all.

## Testing handlers

Handlers are ordinary functions of a `RequestContext`, so they test without a
server: call `handle(route, context)` for one of `routesOf(controller)`, or
mount on Express and use `fetch` against an ephemeral port. For the model,
load either the trained artifact or a stub artifact from
`@semantscript/core/testing`, which needs no training and no model files: it
is built from the IR bundle `semantscript build` writes, and each expression
answers what the test says.

```js
import test from "node:test";
import { loadSemaStubArtifact } from "@semantscript/core/testing";
import { handle, routesOf } from "@semantscript/framework";
import { decideRefund, migrate, refundRisk } from "../dist/app.js";
import { RefundController } from "../dist/refunds.js";
const bundle = new URL("../dist/semantscript.ir.v1.json", import.meta.url);

test("a reviewed refund rolls back", async (t) => {
  const answers = new Map().set(decideRefund, "review").set(refundRisk, "high");
  const stub = await loadSemaStubArtifact(bundle, { answers });
  t.after(() => stub.close());
  await migrate();
  const [route] = routesOf(new RefundController());
  const response = await handle(route, { params: { orderId: "o1" } });
  t.assert.match(response.body.reason, /^decision review \(risk high\)/);
  t.assert.deepEqual(response.passes, { encoder: 1, adapter: 1, head: 2 });
});
```

- The context carries what the handler reads (here the route parameter).
- An answer is keyed by the compiled function (or by its `nf_...` id) and is
  a fixed value, `{ value, confidence }`, `{ byInput: [{ inputs, value }],
otherwise }` matched on the canonical input, or `{ compute: (inputs) =>
value }` (synchronous). A flat object output answers
  `{ value: { ...fields } }`, with one confidence or one per field.
  Confidence defaults to 1 and must stay above one over the number of
  possible values, so the answer is the top value.
- A compiled function that calls several expressions (the staged
  `screenOrder` asks three) has no single id, so answer each expression by
  its `nf_...` id. Keying by that function throws `SemaStubError`
  (`unresolved-function`) listing every expression's source position and
  id; the IR bundle has the same pairs under `functions[].id` and
  `functions[].source`.
- The stub loads through `loadSemaArtifact`, so the real worker runs every
  call: request scopes, `x-sema-passes`, input validation, `@confidence`
  thresholds and the `fallbacks` you pass behave as with a trained artifact.
  A low confidence answer takes the fallback path.
- The bundle is the one `semantscript build` writes; a `URL` resolves it
  against the test file, a plain path against the working directory.
- A call to an expression without an answer throws `SemaStubError`
  (`unanswered`), whose message names the expression's source position and
  id; `close()` removes the stub. A loaded stub is the process's active
  artifact, like a trained one, so load one stub at a time (node:test runs a
  file's tests in order) and do not nest them: closing a later stub leaves
  no artifact active. The
  [reference application](reference-application.md) tests its refund
  controller this way and its other controllers over the trained artifact.
