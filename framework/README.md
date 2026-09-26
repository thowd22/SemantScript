# Framework

`@semantscript/framework` is the thin Phase 4 layer: controller-style handlers
with sema expressions inside, mounted on the HTTP stack an application already
has. It owns no server. Three things come from it:

- **Decorated controllers.** `@Controller(prefix)` on a class and `@Get`,
  `@Post`, `@Put`, `@Patch`, `@Delete` on its methods declare routes;
  `routesOf(instance)` lists them with bound handlers. A handler receives a
  `RequestContext` (`method`, `path`, `params`, `query`, `headers`, `body`)
  and returns a value (JSON 200) or `reply(status, body)`.
- **One request scope per handler.** Every handler runs inside the runtime's
  `withSemaScope`, so the sema expressions it evaluates share encoder and
  adapter passes over identical inputs, the way one execution-plan stage
  fuses them, and the embeddings kept for that are dropped when the response
  is produced. The response carries `x-sema-passes` (`encoder/adapter/head`)
  so the sharing is observable per request.
- **Deterministic guards.** `require(condition, message, status = 422)` is a
  plain assertion that throws a `RequirementError`, answered as its status
  with `{ error, code }`. Because sema calls are synchronous, a guard written
  above an expression runs before the model is consulted and a guard written
  below sees the model's value: guards run before and after neural results
  exactly as written.

```ts
import { Controller, Post, require } from "@semantscript/framework";
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

## Integrations

- **Express** (and Nest's Express adapter): `mountControllers(app, [new TicketController()])`
  registers every route on any object with `get`, `post`, ... registration
  methods, an app or a router. `semaRequestMiddleware()` opens a scope for
  plain handlers written without decorators; the scope closes when the
  response finishes. Nest users add it as a global middleware.
- **Next.js App Router**: `export const { GET, POST } = nextRouteHandlers(new TicketController())`
  in a `route.ts` file; handlers take the web `Request`, read `params` from the
  route context and answer with `Response.json`.

The runtime's `semaScopePasses()` reports the passes performed so far in the
current request. `test/framework.test.mjs` covers the decorators, guard
ordering (a failing pre-guard performs no pass; a post-guard reads the neural
value), the Express round trip over HTTP, the middleware and the Next handlers,
and a controller driven through `handle()` over a stub artifact from
`@semantscript/core/testing`. To test your own handlers without a trained
model, see "Testing handlers" in the
[framework guide](../docs/framework-guide.md#testing-handlers).

## Persistence and transactions

Data stays out of the weights: handlers read rows, hand the decision plain
values, and the decision gates the write. `transactional(client, body)` runs
`BEGIN`, the body, then `COMMIT`, or `ROLLBACK` when the body returns
`rollback(reason)` or throws; it takes any client with the node-postgres query
shape (`pg` Client, a `pg` Pool, which is checked out and released around the
transaction, or PGlite, Postgres compiled to WebAssembly). `gate(decision,
accept, commit, reason)` is the usual body tail: commit when the neural
decision satisfies `accept`, else roll back naming the decision.

```ts
return transactional(pool, async (tx) => {
  const { rows } = await tx.query("SELECT ... WHERE o.id = $1", [orderId]);
  if (rows[0] === undefined) return rollback("no such order");
  const decision = decideRefund(customerOf(rows[0]), orderOf(rows[0]));
  return gate(
    decision,
    (d) => d === "approve",
    async () => {
      await tx.query(
        "INSERT INTO refunds (order_id, decision) VALUES ($1, $2)",
        [orderId, decision],
      );
      return { orderId, decision };
    },
  );
});
```

No sema expression has ambient database access: an expression's inputs are
its interpolated values, the compiler rejects a client object as an input
(diagnostic 9112, unsupported input type; `compiler/test/compile.test.mjs`),
and the runtime serializes only plain data, so the expression can neither
read nor write beyond what the handler passed it. `test/persistence.test.mjs`
runs the commit, rollback, exception and pool paths against PGlite, and
`examples/express-app` has the handler end to end (`POST /refunds/:orderId`).
