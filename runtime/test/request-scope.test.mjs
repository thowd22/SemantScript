import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { setTimeout } from "node:timers";

import {
  createFixtureArtifact,
  fixtureFunctionId,
} from "./fixtures/artifact.mjs";

const siblingId = `nf_${"8".repeat(64)}`;

test("a request scope shares encoder and adapter passes across sibling calls over one input", async () => {
  const runtime = await import("../dist/index.js");
  const root = await mkdtemp(join(tmpdir(), "semantscript-runtime-scope-"));
  try {
    await createFixtureArtifact(root, {
      extraFunctions: [{ id: siblingId, headRef: "head.sibling.value" }],
    });
    await runtime.loadSemaArtifact(root);
    const inputs = { facts: { a: 1, b: 2 } };
    assert.equal(
      runtime.semaScopePasses(),
      undefined,
      "no scope outside withSemaScope",
    );

    // Two sites over the same inputs in one request: one encoder pass, one adapter pass, two heads.
    const scoped = runtime.withSemaScope(() => {
      const first = runtime.__sema.call(fixtureFunctionId, inputs);
      const second = runtime.__sema.call(siblingId, { facts: { b: 2, a: 1 } });
      const again = runtime.__sema.call(fixtureFunctionId, inputs);
      return { first, second, again, passes: runtime.semaScopePasses() };
    });
    assert.deepEqual(
      [scoped.first, scoped.second, scoped.again],
      ["review", "review", "review"],
    );
    assert.deepEqual(scoped.passes, { encoder: 1, adapter: 1, head: 3 });

    // A different input in the same scope costs its own passes.
    const mixed = runtime.withSemaScope(() => {
      runtime.__sema.call(fixtureFunctionId, inputs);
      runtime.__sema.call(siblingId, { facts: { a: 3, b: 4 } });
      return runtime.semaScopePasses();
    });
    assert.deepEqual(mixed, { encoder: 2, adapter: 2, head: 2 });

    // Scopes do not share with each other: a new scope recomputes.
    const next = runtime.withSemaScope(() => {
      runtime.__sema.call(siblingId, inputs);
      return runtime.semaScopePasses();
    });
    assert.deepEqual(next, { encoder: 1, adapter: 1, head: 1 });

    // Async scopes span awaits and settle before releasing.
    const asynchronous = await runtime.withSemaScope(async () => {
      runtime.__sema.call(fixtureFunctionId, inputs);
      await new Promise((resolve) => setTimeout(resolve, 5));
      runtime.__sema.call(siblingId, inputs);
      return runtime.semaScopePasses();
    });
    assert.deepEqual(asynchronous, { encoder: 1, adapter: 1, head: 2 });

    // Errors release the scope and propagate.
    await assert.rejects(
      runtime.withSemaScope(async () => {
        runtime.__sema.call(fixtureFunctionId, inputs);
        throw new Error("handler failed");
      }),
      /handler failed/u,
    );
    // Values outside any scope are unchanged.
    assert.equal(runtime.__sema.call(siblingId, inputs), "review");
    await runtime.closeSemaArtifact();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
