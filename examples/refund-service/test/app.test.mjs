import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { closeSemaArtifact, loadSemaArtifact } from "@semantscript/core";

import {
  createFixtureArtifact,
  semanticSha,
} from "../../../runtime/test/fixtures/artifact.mjs";

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
    return { status: response.status, body: await response.json() };
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

test("the refund handler rolls back a non-approval over PGlite with the fixture artifact", async (t) => {
  // The fixture artifact answers the third support value for every function.
  // Re-keyed to this app's refund functions and input shapes, that is "review"
  // for the decision and "medium" for the risk, so the gate rolls back and
  // nothing is written. The framework path runs end to end without a trained
  // model.
  const ir = await bundle();
  const refunds = ir.functions.filter(
    (fn) => fn.source.path === "src/refunds.sem.ts",
  );
  assert.equal(refunds.length, 3);
  const fixtureRoot = await mkdtemp(join(tmpdir(), "refund-service-fixture-"));
  await createFixtureArtifact(fixtureRoot, {
    extraFunctions: refunds
      .slice(1)
      .map((fn) => ({
        id: fn.id,
        headRef: `head.${fn.id.slice(3, 11)}.value`,
      })),
    transformManifest: (manifest) => {
      manifest.functions = manifest.functions.map((entry, index) => {
        const fn = refunds[index];
        return {
          ...entry,
          id: fn.id,
          inputs: fn.inputs,
          inputSchemaSha256: semanticSha(fn.inputs),
          heads: [
            {
              ...entry.heads[0],
              type: { kind: "nominal-string", support: fn.output.head.support },
            },
          ],
        };
      });
    },
  });
  await loadSemaArtifact(fixtureRoot);
  t.after(async () => {
    await closeSemaArtifact();
    await rm(fixtureRoot, { recursive: true, force: true });
  });
  const post = await serve(t);
  const { status, body } = await post("/refunds/o1");
  assert.equal(status, 200);
  assert.equal(body.committed, false);
  assert.match(body.reason, /^decision review \(risk medium\)/);
  const { db } = await import("../dist/app.js");
  const { rows } = await db.query("SELECT count(*)::int AS n FROM refunds");
  assert.equal(rows[0].n, 0);
  const missing = await post("/refunds/nope");
  assert.equal(missing.status, 200);
  assert.deepEqual(missing.body, { committed: false, reason: "no such order" });
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
