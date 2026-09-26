import assert from "node:assert/strict";
import test from "node:test";

test("runtime package loads", async () => {
  const runtime = await import("../dist/index.js");

  assert.equal(typeof runtime.loadSemaArtifact, "function");
  assert.equal(typeof runtime.closeSemaArtifact, "function");
  assert.equal(typeof runtime.SemaArtifactInactiveError, "function");
  assert.equal(typeof runtime.__sema.call, "function");
  assert.equal(typeof runtime.sema, "function");
  assert.equal(typeof runtime.sema.withConfidence, "function");
  assert.throws(
    () => runtime.__sema.call(`nf_${"0".repeat(64)}`, {}),
    runtime.SemaRuntimeNotLoadedError,
  );
  assert.throws(() => runtime.sema`uncompiled`, /must be compiled/);
});

test("runtime testing subpath loads", async () => {
  const testing = await import("../dist/testing.js");

  assert.equal(typeof testing.createSemaStubArtifact, "function");
  assert.equal(typeof testing.loadSemaStubArtifact, "function");
  assert.equal(typeof testing.semaFunctionId, "function");
  assert.equal(
    new testing.SemaStubError("unanswered", "x").code,
    "SEMA_STUB_INVALID",
  );
});
