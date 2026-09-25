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
value), the Express round trip over HTTP, the middleware and the Next handlers.
