import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { closeSemaArtifact, loadSemaArtifact } from "@semantscript/core";
import { loadSemaStubArtifact } from "@semantscript/core/testing";

import { decideRefund, refundMethod, refundRisk } from "../dist/refunds.sem.js";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..");
const bundlePath = join(root, "dist", "semantscript.ir.v1.json");
const artifactRoot = join(root, ".semantscript", "artifact");
const reportPath = join(root, ".semantscript", "train-report.json");

async function bundle() {
  return JSON.parse(await readFile(bundlePath, "utf8"));
}

function functionsBySite(ir) {
  // Name every function by the export or local it is assigned to in its file.
  const names = new Map();
  for (const fn of ir.functions) {
    const key = `${fn.source.path}:${fn.output.tsType}`;
    names.set(key, fn);
  }
  return names;
}

async function serve(t) {
  // The app module owns the PGlite instance; import it lazily so the artifact
  // is loaded before any handler runs.
  const { createApp } = await import("../dist/app.js");
  const app = await createApp();
  const server = createServer(app);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  t.after(() => new Promise((resolve) => server.close(resolve)));
  return async (path) => {
    const response = await fetch(`http://127.0.0.1:${port}${path}`, {
      method: "POST",
    });
    return {
      status: response.status,
      passes: response.headers.get("x-sema-passes"),
      body: await response.json(),
    };
  };
}

test("the bundle routes three domains at depth 6 and stages the fraud chain", async () => {
  const ir = await bundle();
  assert.equal(ir.functions.length, 9);
  const domains = ir.executionPlan.domains.map((domain) => ({
    name: domain.name,
    adapterRef: domain.adapterRef,
    encoderRef: domain.encoderRef,
    encoderDepth: domain.encoderDepth,
    functions: domain.functionIds.length,
  }));
  assert.deepEqual(domains, [
    {
      name: "orders",
      adapterRef: "adapter.refund-service.orders",
      encoderRef: "encoder.refund-service.depth-006",
      encoderDepth: 6,
      functions: 3,
    },
    {
      name: "refunds",
      adapterRef: "adapter.refund-service.refunds",
      encoderRef: "encoder.refund-service.depth-006",
      encoderDepth: 6,
      functions: 3,
    },
    {
      name: "tickets",
      adapterRef: "adapter.refund-service.tickets",
      encoderRef: "encoder.refund-service.depth-006",
      encoderDepth: 6,
      functions: 3,
    },
  ]);
  // The flag feeds the hold and the escalation: two labeled dependencies, two stages.
  const bySite = functionsBySite(ir);
  const flag = bySite.get("src/orders.sem.ts:FraudFlag");
  const hold = bySite.get("src/orders.sem.ts:boolean");
  const escalation = bySite.get("src/orders.sem.ts:Escalation");
  assert.deepEqual(ir.executionPlan.dependencies, [
    {
      producerFunctionId: flag.id,
      consumerFunctionId: hold.id,
      consumerInput: "flag",
    },
    {
      producerFunctionId: flag.id,
      consumerFunctionId: escalation.id,
      consumerInput: "flag",
    },
  ]);
  assert.deepEqual(
    ir.executionPlan.stages.map((stage) => stage.functionIds.length),
    [7, 2],
  );
  assert.deepEqual(
    ir.executionPlan.stages[1].functionIds.sort(),
    [hold.id, escalation.id].sort(),
  );
  assert.deepEqual(ir.executionPlan.stages[1].adapterRefs, [
    "adapter.refund-service.orders",
  ]);
  // Every expression carries gold examples and a constraint set the trainer labels from.
  for (const fn of ir.functions) {
    assert.ok(
      fn.definition.examples.length >= 3,
      `${fn.source.path}:${fn.source.line} has gold examples`,
    );
    assert.ok(
      fn.definition.constraints.length >= 3,
      `${fn.source.path}:${fn.source.line} has constraints`,
    );
    assert.equal(fn.model.encoderDepth, 6);
  }
});

test("the refund handler rolls back or writes a refund over PGlite with a stub artifact", async (t) => {
  // A stub artifact built from the app's IR bundle answers what the test
  // says, keyed by the compiled functions: a per-input decision (o1 is
  // reviewed, every other order approved), a fixed risk, and a payout method
  // computed from the inputs. The framework path, request scopes and pass
  // counts run for real; no model is trained.
  const o1 = {
    customer: { tier: "standard", priorRefunds: 1 },
    order: { total: 88.5, ageDays: 12, status: "paid" },
  };
  const stub = await loadSemaStubArtifact(bundlePath, {
    answers: [
      [
        decideRefund,
        { byInput: [{ inputs: o1, value: "review" }], otherwise: "approve" },
      ],
      [refundRisk, { value: "medium", confidence: 0.9 }],
      [
        refundMethod,
        {
          compute: ({ payment }) =>
            payment.method === "card" ? "original-payment" : "manual",
        },
      ],
    ],
  });
  t.after(() => stub.close());
  const post = await serve(t);
  const { db } = await import("../dist/app.js");
  const count = async (orderId) =>
    (
      await db.query(
        "SELECT count(*)::int AS n FROM refunds WHERE order_id = $1",
        [orderId],
      )
    ).rows[0].n;

  // Review: the gate rolls back and nothing is written. The decision and the
  // risk read the same inputs through one adapter: one encoder pass, two heads.
  const reviewed = await post("/refunds/o1");
  assert.equal(reviewed.status, 200);
  assert.equal(reviewed.body.committed, false);
  assert.match(reviewed.body.reason, /^decision review \(risk medium\)/);
  assert.equal(reviewed.passes, "1/1/2");
  assert.equal(await count("o1"), 0);

  // Approve: the payout method runs over its own inputs and a row is written.
  const approved = await post("/refunds/o3");
  assert.deepEqual(approved.body, {
    committed: true,
    value: {
      orderId: "o3",
      decision: "approve",
      method: "original-payment",
      risk: "medium",
      refunded: true,
    },
  });
  assert.equal(approved.passes, "2/2/3");
  assert.equal(await count("o3"), 1);
  await db.query("DELETE FROM refunds WHERE order_id = 'o3'");

  const missing = await post("/refunds/nope");
  assert.equal(missing.status, 200);
  assert.deepEqual(missing.body, { committed: false, reason: "no such order" });
  assert.equal(missing.passes, "0/0/0");
});

test(
  "the trained artifact drives all three controllers",
  { skip: !trained() && "train first: npm run train" },
  async (t) => {
    await loadSemaArtifact(artifactRoot);
    t.after(() => closeSemaArtifact());
    const post = await serve(t);
    const approved = await post("/refunds/o1");
    assert.equal(approved.status, 200);
    assert.deepEqual(approved.body, {
      committed: true,
      value: {
        orderId: "o1",
        decision: "approve",
        method: "original-payment",
        risk: "medium",
        refunded: true,
      },
    });
    const denied = await post("/refunds/o2");
    assert.equal(denied.body.committed, false);
    assert.match(denied.body.reason, /^decision deny/);
    const fraudulent = await post("/refunds/o4");
    assert.match(fraudulent.body.reason, /^decision review \(risk high\)/);

    const outage = await post("/tickets/t1/triage");
    assert.deepEqual(outage.body, {
      committed: true,
      value: {
        ticketId: "t1",
        priority: "urgent",
        queue: "engineering",
        needsHuman: true,
      },
    });
    const billing = await post("/tickets/t2/triage");
    assert.deepEqual(billing.body.value, {
      ticketId: "t2",
      priority: "normal",
      queue: "billing",
      needsHuman: false,
    });
    assert.equal((await post("/tickets/t9/triage")).status, 404);

    const flagged = await post("/orders/o4/screen");
    assert.deepEqual(flagged.body.value, {
      orderId: "o4",
      flag: "flag",
      shipmentHeld: true,
      escalation: "legal",
    });
    const watched = await post("/orders/o3/screen");
    assert.deepEqual(watched.body.value, {
      orderId: "o3",
      flag: "watch",
      shipmentHeld: true,
      escalation: "none",
    });
    const clear = await post("/orders/o1/screen");
    assert.deepEqual(clear.body.value, {
      orderId: "o1",
      flag: "clear",
      shipmentHeld: false,
      escalation: "none",
    });
  },
);

function trained() {
  if (!existsSync(reportPath)) return false;
  try {
    const report = JSON.parse(readFileSync(reportPath, "utf8"));
    return ["passed", "reused"].includes(report.status);
  } catch {
    return false;
  }
}
