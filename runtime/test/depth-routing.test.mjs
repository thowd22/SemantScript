import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  createFixtureArtifact,
  fixtureFunctionId,
} from "./fixtures/artifact.mjs";

const routedId = `nf_${"9".repeat(64)}`;

test("a depth-routed function runs its own encoder prefix and adapter while sharing a stage", async () => {
  const runtime = await import("../dist/index.js");
  const root = await mkdtemp(join(tmpdir(), "semantscript-runtime-routed-"));
  try {
    await createFixtureArtifact(root, {
      extraEncoders: [
        { ref: "encoder.fixture.depth-006", path: "depth-006.onnx" },
      ],
      extraAdapters: [{ ref: "adapter.fixture.refund" }],
      extraFunctions: [
        {
          id: routedId,
          headRef: "head.routed.value",
          adapterRef: "adapter.fixture.refund",
          encoderRef: "encoder.fixture.depth-006",
        },
      ],
    });
    const handle = await runtime.loadSemaArtifact(root);
    assert.deepEqual(
      [...handle.functionIds].sort(),
      [fixtureFunctionId, routedId].sort(),
    );
    const inputs = { facts: { a: 1, b: 2 } };
    assert.equal(handle.call(routedId, inputs), "review");
    assert.equal(handle.call(fixtureFunctionId, inputs), "review");

    // One stage, one input, two functions on different encoders and adapters:
    // two encoder passes (the full stack and the prefix), two adapter passes.
    const outcome = handle.callStage([
      { functionId: fixtureFunctionId, inputs },
      { functionId: routedId, inputs },
    ]);
    assert.deepEqual(outcome.results, ["review", "review"]);
    assert.deepEqual(outcome.passes, { encoder: 2, adapter: 2, head: 2 });
    // Two functions of one domain share the prefix pass.
    const shared = handle.callStage([
      { functionId: routedId, inputs },
      { functionId: routedId, inputs: { facts: { a: 1, b: 2 } } },
    ]);
    assert.deepEqual(shared.passes, { encoder: 1, adapter: 1, head: 2 });
    await handle.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("a function naming an encoder that is not in the manifest is rejected at load", async () => {
  const runtime = await import("../dist/index.js");
  const root = await mkdtemp(
    join(tmpdir(), "semantscript-runtime-routed-bad-"),
  );
  try {
    await createFixtureArtifact(root, {
      extraFunctions: [
        {
          id: routedId,
          headRef: "head.routed.value",
          encoderRef: "encoder.fixture.missing",
        },
      ],
    });
    await assert.rejects(
      runtime.loadSemaArtifact(root),
      runtime.ArtifactLoadError,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
