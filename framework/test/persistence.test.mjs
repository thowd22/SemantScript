import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { PGlite } from "@electric-sql/pglite";

import {
  createFixtureArtifact,
  fixtureFunctionId,
} from "../../runtime/test/fixtures/artifact.mjs";
import {
  __sema,
  closeSemaArtifact,
  loadSemaArtifact,
} from "@semantscript/core";
import { gate, rollback, Rollback, transactional } from "../dist/index.js";

async function database() {
  // Postgres 17 compiled to WebAssembly, in process: a real Postgres, no server.
  const db = new PGlite();
  await db.exec(`
    CREATE TABLE customers (id text PRIMARY KEY, tier text NOT NULL, prior_refunds int NOT NULL);
    CREATE TABLE orders (id text PRIMARY KEY, customer_id text NOT NULL REFERENCES customers(id), total numeric NOT NULL, age_days int NOT NULL, status text NOT NULL);
    CREATE TABLE refunds (order_id text PRIMARY KEY REFERENCES orders(id), decision text NOT NULL, decided_at timestamptz NOT NULL DEFAULT now());
    INSERT INTO customers VALUES ('c1', 'standard', 1), ('c2', 'enterprise', 4);
    INSERT INTO orders VALUES ('o1', 'c1', 88.5, 12, 'paid'), ('o2', 'c2', 900, 70, 'paid');
  `);
  return db;
}

/**
 * The handler shape the task asks for: read rows, hand plain values to the
 * decision, let the decision gate the write. `decide` stands for a compiled
 * sema function; it receives only the values the handler interpolates.
 */
async function refundHandler(db, orderId, decide) {
  return transactional(db, async (tx) => {
    const { rows } = await tx.query(
      `SELECT o.id, o.total, o.age_days, o.status, c.tier, c.prior_refunds
         FROM orders o JOIN customers c ON c.id = o.customer_id WHERE o.id = $1`,
      [orderId],
    );
    const order = rows[0];
    if (order === undefined) return rollback("no such order");
    const customer = {
      tier: order.tier,
      priorRefunds: Number(order.prior_refunds),
    };
    const facts = {
      total: Number(order.total),
      ageDays: Number(order.age_days),
      status: order.status,
    };
    const decision = decide(customer, facts);
    return gate(
      decision,
      (value) => value === "approve",
      async () => {
        await tx.query(
          "INSERT INTO refunds (order_id, decision) VALUES ($1, $2)",
          [orderId, decision],
        );
        return { orderId, decision };
      },
    );
  });
}

test("a neural decision gates the transaction: approve commits the refund row, anything else rolls back", async (t) => {
  const db = await database();
  t.after(() => db.close());
  const refunds = async () =>
    (await db.query("SELECT order_id, decision FROM refunds ORDER BY order_id"))
      .rows;

  const approved = await refundHandler(db, "o1", () => "approve");
  assert.deepEqual(approved, {
    committed: true,
    value: { orderId: "o1", decision: "approve" },
  });
  assert.deepEqual(await refunds(), [{ order_id: "o1", decision: "approve" }]);

  const denied = await refundHandler(db, "o2", () => "deny");
  assert.equal(denied.committed, false);
  assert.match(denied.reason, /"deny" does not permit the write/u);
  assert.deepEqual(
    await refunds(),
    [{ order_id: "o1", decision: "approve" }],
    "a rolled-back decision writes nothing",
  );

  const missing = await refundHandler(db, "o9", () => "approve");
  assert.deepEqual(missing, {
    committed: false,
    reason: "no such order",
    value: undefined,
  });

  // An exception inside the body rolls back and propagates; the write it made is gone.
  await assert.rejects(
    transactional(db, async (tx) => {
      await tx.query(
        "INSERT INTO refunds (order_id, decision) VALUES ('o2', 'approve')",
      );
      throw new Error("downstream failed");
    }),
    /downstream failed/u,
  );
  assert.deepEqual(await refunds(), [{ order_id: "o1", decision: "approve" }]);
  assert.ok(rollback("x") instanceof Rollback);
});

test("the real runtime's decision drives the gate and never sees the database", async (t) => {
  const db = await database();
  t.after(() => db.close());
  const root = await mkdtemp(join(tmpdir(), "semantscript-persistence-"));
  await createFixtureArtifact(root);
  await loadSemaArtifact(root);
  t.after(async () => {
    await closeSemaArtifact();
    await rm(root, { recursive: true, force: true });
  });
  // The fixture function answers "review" for these facts: not an approval, so no row.
  const outcome = await refundHandler(db, "o1", (customer, facts) =>
    __sema.call(fixtureFunctionId, {
      facts: { a: customer.priorRefunds, b: facts.ageDays % 10 },
    }),
  );
  assert.equal(outcome.committed, false);
  assert.match(outcome.reason, /"review"/u);
  assert.deepEqual(
    (await db.query("SELECT count(*)::int AS n FROM refunds")).rows,
    [{ n: 0 }],
  );
  // The runtime accepts only plain data as inputs: a client object is rejected before any pass.
  assert.throws(
    () => __sema.call(fixtureFunctionId, { facts: db }),
    /input|facts/u,
  );
});

test("a pool is checked out and released around the transaction", async (t) => {
  const db = await database();
  t.after(() => db.close());
  let released = 0;
  const pool = {
    connect: async () => ({
      query: (text, params) => db.query(text, params),
      release: () => {
        released += 1;
      },
    }),
  };
  const outcome = await transactional(
    pool,
    async (tx) => (await tx.query("SELECT 1 AS one")).rows[0].one,
  );
  assert.deepEqual(outcome, { committed: true, value: 1 });
  assert.equal(released, 1);
});
