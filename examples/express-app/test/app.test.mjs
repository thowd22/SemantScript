import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { closeSemaArtifact, loadSemaArtifact } from "@semantscript/core";

import {
  readBundle,
  writeFixtureArtifact,
} from "../scripts/fixture-artifact.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");

async function fixture(t) {
  const fixtureRoot = await mkdtemp(join(tmpdir(), "express-app-fixture-"));
  t.after(() => rm(fixtureRoot, { recursive: true, force: true }));
  await writeFixtureArtifact(fixtureRoot, await readBundle());
  return fixtureRoot;
}

async function freePort() {
  const probe = createServer();
  await new Promise((resolve) => probe.listen(0, "127.0.0.1", resolve));
  const { port } = probe.address();
  await new Promise((resolve) => probe.close(resolve));
  return port;
}

function client(port) {
  return async (path, body) => {
    const response = await fetch(`http://127.0.0.1:${port}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    return { status: response.status, body: await response.json() };
  };
}

test("the bundle holds the two sema functions with their supports and gold examples", async () => {
  const ir = await readBundle();
  const bySource = Object.fromEntries(
    ir.functions.map((fn) => [fn.source.path, fn]),
  );
  assert.deepEqual(Object.keys(bySource).sort(), [
    "src/refunds.sem.ts",
    "src/triage.sem.ts",
  ]);
  assert.deepEqual(bySource["src/triage.sem.ts"].output.head.support, [
    "low",
    "normal",
    "urgent",
  ]);
  assert.deepEqual(
    bySource["src/triage.sem.ts"].inputs.map((input) => input.name),
    ["subject", "body"],
  );
  assert.deepEqual(bySource["src/refunds.sem.ts"].output.head.support, [
    "approve",
    "deny",
    "review",
  ]);
  assert.equal(bySource["src/refunds.sem.ts"].definition.constraints.length, 6);
  for (const fn of ir.functions) {
    assert.equal(fn.definition.examples.length, 3, fn.source.path);
  }
  // Prompt text never reaches the emitted JavaScript.
  const emitted = await readFile(join(root, "dist", "triage.sem.js"), "utf8");
  assert.doesNotMatch(emitted, /Priority of a customer support ticket/);
});

test("both routes answer over the fixture artifact and a non-approval rolls back", async (t) => {
  await loadSemaArtifact(await fixture(t));
  t.after(() => closeSemaArtifact());
  const { createApp, db } = await import("../dist/app.js");
  const server = createServer(await createApp());
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise((resolve) => server.close(resolve)));
  const post = client(server.address().port);

  const ticket = await post("/tickets", {
    subject: "Checkout is down",
    body: "Every customer sees a 500.",
  });
  assert.equal(ticket.status, 201);
  assert.equal(ticket.body.subject, "Checkout is down");
  // The fixture answers the third support value: "urgent" for triage.
  assert.equal(ticket.body.priority, "urgent");
  const invalid = await post("/tickets", { subject: "no body" });
  assert.equal(invalid.status, 400);

  // ... and "review" for decideRefund, so the gate rolls the transaction back.
  const refund = await post("/refunds/o1");
  assert.equal(refund.status, 200);
  assert.equal(refund.body.committed, false);
  assert.match(refund.body.reason, /^decision review/);
  const { rows } = await db.query("SELECT count(*)::int AS n FROM refunds");
  assert.equal(rows[0].n, 0);
  const missing = await post("/refunds/nope");
  assert.deepEqual(missing.body, { committed: false, reason: "no such order" });
});

test("dist/server.js loads SEMANTSCRIPT_ARTIFACT and listens on PORT", async (t) => {
  assert.ok(existsSync(join(root, "dist", "server.js")));
  const artifact = await fixture(t);
  const port = await freePort();
  const child = spawn(
    process.execPath,
    ["--enable-source-maps", join(root, "dist", "server.js")],
    {
      cwd: root,
      env: {
        ...process.env,
        PORT: String(port),
        SEMANTSCRIPT_ARTIFACT: artifact,
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  let output = "";
  child.stdout.on("data", (chunk) => (output += chunk));
  child.stderr.on("data", (chunk) => (output += chunk));
  const exited = new Promise((resolve) => child.on("exit", resolve));
  t.after(async () => {
    child.kill();
    await exited;
  });
  const deadline = Date.now() + 30_000;
  while (!output.includes("listening")) {
    if (child.exitCode !== null) {
      assert.fail(`server exited with ${child.exitCode}:\n${output}`);
    }
    if (Date.now() > deadline) assert.fail(`server never listened:\n${output}`);
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  assert.match(output, new RegExp(`localhost:${port}`));
  const ticket = await client(port)("/tickets", {
    subject: "Feature idea",
    body: "CSV export please",
  });
  assert.equal(ticket.status, 201);
  assert.equal(ticket.body.priority, "urgent");
});
