import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { setTimeout } from "node:timers";

import express from "express";

import {
  createFixtureArtifact,
  fixtureFunctionId,
} from "../../runtime/test/fixtures/artifact.mjs";
import {
  __sema,
  closeSemaArtifact,
  loadSemaArtifact,
  semaScopePasses,
} from "@semantscript/core";
import { loadSemaStubArtifact } from "@semantscript/core/testing";
import {
  Controller,
  Get,
  handle,
  mountControllers,
  nextRouteHandlers,
  Post,
  reply,
  require,
  RequirementError,
  routesOf,
  semaRequestMiddleware,
} from "../dist/index.js";

const siblingId = `nf_${"8".repeat(64)}`;
const trace = [];

// A controller in the style the task describes: deterministic guards and sema
// expressions mixed in one handler, in source order. The fixture artifact's
// functions stand in for compiled sema sites (they answer "review"). Plain
// JavaScript has no decorator syntax, so the decorators are applied by hand
// exactly as TypeScript would: methods first, then the class.
class TicketController {
  triage(context) {
    const { subject, amount } = context.body ?? {};
    trace.push("pre-guard");
    require(typeof subject === "string" &&
      subject.length > 0, "subject is required", 400);
    require(typeof amount === "number" &&
      amount >= 0, "amount must be a non-negative number", 400);
    const facts = { a: Math.min(amount, 9), b: subject.length % 10 };
    trace.push("neural-1");
    const priority = __sema.call(fixtureFunctionId, { facts });
    trace.push("neural-2");
    const route = __sema.call(siblingId, { facts });
    trace.push("post-guard");
    require(priority !== "deny" ||
      amount < 1000, "large denied amounts need a human", 409);
    return { subject, priority, route, passes: semaScopePasses() };
  }

  show(context) {
    return reply(200, {
      id: context.params.id,
      query: context.query,
      agent: context.headers["user-agent"] ?? null,
    });
  }
}
Post("/triage")(TicketController.prototype.triage, {
  kind: "method",
  name: "triage",
});
Get("/:id")(TicketController.prototype.show, { kind: "method", name: "show" });
Controller("/tickets")(TicketController);

async function withArtifact(t) {
  const root = await mkdtemp(join(tmpdir(), "semantscript-framework-"));
  await createFixtureArtifact(root, {
    extraFunctions: [{ id: siblingId, headRef: "head.sibling.value" }],
  });
  await loadSemaArtifact(root);
  t.after(async () => {
    await closeSemaArtifact();
    await rm(root, { recursive: true, force: true });
  });
}

test("decorators declare routes with their prefix, method and bound handler", () => {
  const routes = routesOf(new TicketController());
  assert.deepEqual(
    routes.map(({ method, path, name }) => ({ method, path, name })),
    [
      { method: "POST", path: "/tickets/triage", name: "triage" },
      { method: "GET", path: "/tickets/:id", name: "show" },
    ],
  );
  assert.throws(() => routesOf({}), /declares no routes/u);
});

test("a handler's sema sites run inside one request scope and its guards run in source order", async (t) => {
  await withArtifact(t);
  const [triage] = routesOf(new TicketController());
  trace.length = 0;
  const ok = await handle(triage, {
    method: "POST",
    path: "/tickets/triage",
    params: {},
    query: {},
    headers: {},
    body: { subject: "site down", amount: 120 },
  });
  assert.equal(ok.status, 200);
  assert.deepEqual(
    ok.body.passes,
    { encoder: 1, adapter: 1, head: 2 },
    "two sites over one input share the encoder and adapter passes",
  );
  assert.deepEqual(ok.passes, { encoder: 1, adapter: 1, head: 2 });
  assert.deepEqual([ok.body.priority, ok.body.route], ["review", "review"]);
  assert.deepEqual(trace, ["pre-guard", "neural-1", "neural-2", "post-guard"]);

  // A failing pre-guard answers before any neural pass.
  trace.length = 0;
  const rejected = await handle(triage, {
    method: "POST",
    path: "/tickets/triage",
    params: {},
    query: {},
    headers: {},
    body: { subject: "", amount: 5 },
  });
  assert.equal(rejected.status, 400);
  assert.deepEqual(rejected.body, {
    error: "subject is required",
    code: "SEMA_REQUIREMENT_FAILED",
  });
  assert.deepEqual(rejected.passes, { encoder: 0, adapter: 0, head: 0 });
  assert.deepEqual(trace, ["pre-guard"]);

  // A post-guard sees the neural value: "review" is not "deny", so a large amount passes it.
  const large = await handle(triage, {
    method: "POST",
    path: "/tickets/triage",
    params: {},
    query: {},
    headers: {},
    body: { subject: "refund", amount: 5000 },
  });
  assert.equal(large.status, 200);

  // Other errors are 500 with the message, and the scope still closes.
  const broken = await handle(
    {
      ...triage,
      handler: () => {
        throw new Error("boom");
      },
    },
    {
      method: "POST",
      path: "/x",
      params: {},
      query: {},
      headers: {},
      body: {},
    },
  );
  assert.equal(broken.status, 500);
  assert.equal(broken.body.code, "SEMA_HANDLER_FAILED");
  assert.equal(semaScopePasses(), undefined);
  assert.ok(new RequirementError("x", 418) instanceof Error);
});

test("controllers mount on an Express app and answer over HTTP", async (t) => {
  await withArtifact(t);
  const app = express();
  app.use(express.json());
  const mounted = mountControllers(app, [new TicketController()]);
  assert.deepEqual(mounted, ["POST /tickets/triage", "GET /tickets/:id"]);
  const server = createServer(app);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => server.close());
  const base = `http://127.0.0.1:${server.address().port}`;

  const triaged = await fetch(`${base}/tickets/triage`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ subject: "checkout broken", amount: 42 }),
  });
  assert.equal(triaged.status, 200);
  assert.equal(triaged.headers.get("x-sema-passes"), "1/1/2");
  const body = await triaged.json();
  assert.equal(body.priority, "review");

  const shown = await fetch(`${base}/tickets/17?expand=1`, {
    headers: { "user-agent": "test-agent" },
  });
  assert.equal(shown.status, 200);
  assert.deepEqual(await shown.json(), {
    id: "17",
    query: { expand: "1" },
    agent: "test-agent",
  });

  const bad = await fetch(`${base}/tickets/triage`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ subject: "x", amount: -1 }),
  });
  assert.equal(bad.status, 400);
  assert.equal(
    (await bad.json()).error,
    "amount must be a non-negative number",
  );
});

test("the request middleware scopes plain Express handlers too", async (t) => {
  await withArtifact(t);
  const app = express();
  app.use(semaRequestMiddleware());
  app.get("/plain", (_request, response) => {
    const facts = { a: 1, b: 2 };
    __sema.call(fixtureFunctionId, { facts });
    __sema.call(siblingId, { facts });
    response.json({ passes: semaScopePasses() });
  });
  const server = createServer(app);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => server.close());
  const response = await fetch(
    `http://127.0.0.1:${server.address().port}/plain`,
  );
  assert.deepEqual(await response.json(), {
    passes: { encoder: 1, adapter: 1, head: 2 },
  });
  await new Promise((resolve) => setTimeout(resolve, 10));
});

test("a controller becomes Next.js App Router handlers over web requests", async (t) => {
  await withArtifact(t);
  const handlers = nextRouteHandlers(new TicketController());
  assert.deepEqual(Object.keys(handlers).sort(), ["GET", "POST"]);
  const posted = await handlers.POST(
    new Request("http://localhost/api/tickets/triage", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ subject: "invoice mismatch", amount: 12 }),
    }),
  );
  assert.equal(posted.status, 200);
  assert.equal(posted.headers.get("x-sema-passes"), "1/1/2");
  assert.equal((await posted.json()).route, "review");
  const got = await handlers.GET(
    new Request("http://localhost/api/tickets/9?expand=yes"),
    { params: Promise.resolve({ id: "9" }) },
  );
  assert.deepEqual(await got.json(), {
    id: "9",
    query: { expand: "yes" },
    agent: null,
  });
});

// The same controller over a stub artifact built from an IR bundle: the two
// sites answer what the test says, and the second is thresholded with a
// fallback, so a low-confidence answer takes the fallback path.
function stubBundle() {
  const facts = {
    name: "facts",
    index: 0,
    tsType: "Facts",
    type: {
      kind: "object",
      name: "Facts",
      fields: [
        { name: "a", optional: false, type: { kind: "number" } },
        { name: "b", optional: false, type: { kind: "number" } },
      ],
    },
  };
  const site = (id, threshold, fallbackRef) => ({
    id,
    semanticSha256: id.slice(3),
    inputs: [facts],
    output: {
      kind: "scalar",
      tsType: "Decision",
      head: {
        kind: "nominal",
        sourceKind: "string-union",
        support: ["approve", "deny", "review"],
      },
    },
    model: {
      adapter: "adapter.tickets",
      encoder: "encoder.tickets",
      heads: [{ outputPath: "", ref: `head.${id.slice(3, 9)}` }],
    },
    runtime: {
      resultMode: "value",
      confidenceThreshold: threshold,
      fallbackRef,
      synchronous: true,
    },
    trainingProvenance: { status: "pending" },
  });
  return {
    kind: "semantscript.ir-bundle",
    bundleVersion: 1,
    functions: [
      site(fixtureFunctionId, null, null),
      site(siblingId, 0.8, "route-by-hand"),
    ],
  };
}

test("a handler tested over a stub artifact keeps its scope, passes and fallbacks", async (t) => {
  const stub = await loadSemaStubArtifact(stubBundle(), {
    answers: {
      [fixtureFunctionId]: {
        compute: ({ facts }) => (facts.a > 5 ? "deny" : "approve"),
      },
      [siblingId]: {
        compute: ({ facts }) => ({
          value: "approve",
          confidence: facts.b === 0 ? 0.5 : 0.9,
        }),
      },
    },
    fallbacks: new Map([["route-by-hand", () => "review"]]),
  });
  t.after(() => stub.close());
  const [triage] = routesOf(new TicketController());
  const request = (body) => ({
    method: "POST",
    path: "/tickets/triage",
    params: {},
    query: {},
    headers: {},
    body,
  });

  const ok = await handle(triage, request({ subject: "site down", amount: 3 }));
  assert.equal(ok.status, 200);
  assert.deepEqual([ok.body.priority, ok.body.route], ["approve", "approve"]);
  assert.deepEqual(ok.passes, { encoder: 1, adapter: 1, head: 2 });

  // Confidence 0.5 is below the site's 0.8: the registered fallback answers.
  const fallback = await handle(
    triage,
    request({ subject: "0123456789", amount: 3 }),
  );
  assert.deepEqual(
    [fallback.body.priority, fallback.body.route],
    ["approve", "review"],
  );

  // The post-guard sees the stubbed "deny".
  const denied = await handle(
    triage,
    request({ subject: "refund", amount: 5000 }),
  );
  assert.equal(denied.status, 409);
  assert.equal(denied.body.error, "large denied amounts need a human");
  assert.deepEqual(denied.passes, { encoder: 1, adapter: 1, head: 2 });

  const handlers = nextRouteHandlers(new TicketController());
  const posted = await handlers.POST(
    new Request("http://localhost/api/tickets/triage", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ subject: "invoice", amount: 1 }),
    }),
  );
  assert.equal(posted.status, 200);
  assert.equal(posted.headers.get("x-sema-passes"), "1/1/2");
});
