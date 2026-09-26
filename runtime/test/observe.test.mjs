import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  createFixtureArtifact,
  fixtureFunctionId,
} from "./fixtures/artifact.mjs";

const inputs = { facts: { a: 1, b: 2 } };
const missingId = `nf_${"f".repeat(64)}`;

test('diagnostics: "always" hands program code the plain value and the observer the distribution', async (t) => {
  const runtime = await import("../dist/index.js");
  const root = await mkdtemp(join(tmpdir(), "semantscript-runtime-observe-"));
  t.after(async () => {
    await runtime.closeSemaArtifact();
    await rm(root, { recursive: true, force: true });
  });
  await createFixtureArtifact(root);

  // Without the options nothing is observed and the value is plain.
  await runtime.loadSemaArtifact(root);
  const plain = runtime.__sema.call(fixtureFunctionId, inputs);
  assert.equal(plain, "review");

  const observed = [];
  await runtime.loadSemaArtifact(root, {
    diagnostics: "always",
    observe: (observation) => observed.push(observation),
  });
  assert.equal(runtime.__sema.call(fixtureFunctionId, inputs), plain);
  assert.equal(observed.length, 1);
  const [answered] = observed;
  assert.equal(answered.kind, "answered");
  assert.equal(answered.functionId, fixtureFunctionId);
  assert.deepEqual(answered.inputs, inputs);
  assert.equal(answered.resultMode, "value");
  assert.equal(answered.confidenceThreshold, null);
  assert.equal(answered.diagnostic.value, plain);
  assert.deepEqual(
    answered.diagnostic.distribution.map(({ value }) => value),
    ["approve", "deny", "review"],
  );
  const total = answered.diagnostic.distribution.reduce(
    (sum, { probability }) => sum + probability,
    0,
  );
  assert.ok(Math.abs(total - 1) < 1e-12);

  // Stage entries are observed one by one, and program code still gets values.
  const stage = runtime.__sema.callStage([
    { functionId: fixtureFunctionId, inputs },
  ]);
  assert.deepEqual(stage.results, [plain]);
  assert.equal(observed.length, 2);

  // A missing id is reported before SemaUnknownFunctionError reaches the caller.
  assert.throws(
    () => runtime.__sema.call(missingId, inputs),
    runtime.SemaUnknownFunctionError,
  );
  assert.deepEqual(observed[2], {
    kind: "missing",
    functionId: missingId,
    inputs,
  });
  assert.throws(
    () => runtime.__sema.callStage([{ functionId: missingId, inputs }]),
    runtime.SemaUnknownFunctionError,
  );
  assert.equal(observed[3].kind, "missing");

  // An observer without "always" sees value-mode calls without a distribution.
  const bare = [];
  await runtime.loadSemaArtifact(root, {
    observe: (observation) => bare.push(observation),
  });
  assert.equal(runtime.__sema.call(fixtureFunctionId, inputs), plain);
  assert.equal(bare[0].kind, "answered");
  assert.equal(bare[0].diagnostic, undefined);

  await assert.rejects(
    runtime.loadSemaArtifact(root, { diagnostics: "sometimes" }),
    TypeError,
  );
  await assert.rejects(
    runtime.loadSemaArtifact(root, { observe: 42 }),
    TypeError,
  );
});

test("observed diagnostic and below-threshold calls keep their result contract", async (t) => {
  const runtime = await import("../dist/index.js");
  const diagnosticRoot = await mkdtemp(
    join(tmpdir(), "semantscript-runtime-observe-diagnostic-"),
  );
  const thresholdRoot = await mkdtemp(
    join(tmpdir(), "semantscript-runtime-observe-threshold-"),
  );
  t.after(async () => {
    await runtime.closeSemaArtifact();
    await rm(diagnosticRoot, { recursive: true, force: true });
    await rm(thresholdRoot, { recursive: true, force: true });
  });
  await createFixtureArtifact(diagnosticRoot, {
    transformManifest(manifest) {
      Object.assign(manifest.functions[0].runtime, {
        resultMode: "diagnostic",
      });
    },
  });
  await createFixtureArtifact(thresholdRoot, {
    transformManifest(manifest) {
      Object.assign(manifest.functions[0].runtime, {
        confidenceThreshold: 1,
        policy: "scalar-top1",
      });
    },
  });

  const observed = [];
  const observe = (observation) => observed.push(observation);
  await runtime.loadSemaArtifact(diagnosticRoot, {
    diagnostics: "always",
    observe,
  });
  const diagnostic = runtime.__sema.call(fixtureFunctionId, inputs);
  assert.equal(typeof diagnostic, "object");
  assert.equal(diagnostic.value, "review");
  assert.equal(observed[0].resultMode, "diagnostic");
  assert.deepEqual(observed[0].diagnostic, diagnostic);

  await runtime.loadSemaArtifact(thresholdRoot, {
    diagnostics: "always",
    observe,
  });
  assert.throws(
    () => runtime.__sema.call(fixtureFunctionId, inputs),
    runtime.SemaConfidenceError,
  );
  assert.equal(observed[1].kind, "answered");
  assert.equal(observed[1].confidenceThreshold, 1);
  assert.ok(observed[1].diagnostic.confidence < 1);
});
