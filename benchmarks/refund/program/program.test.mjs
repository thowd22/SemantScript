import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  REFUND_FUNCTION_ID,
  REFUND_FUNCTION_SEMANTIC_SHA256,
  REFUND_TASK_SPEC,
  REFUND_TASK_SPEC_SHA256,
} from "../dist/policy.js";
import { compileRefundProgram } from "./compile-program.mjs";
import { validateRefundDiagnostic } from "./run-runtime.mjs";

test("compiles the canonical refund task to stable diagnostic source IR", async (t) => {
  const firstRoot = await mkdtemp(
    join(tmpdir(), "semantscript-refund-program-"),
  );
  const secondRoot = await mkdtemp(
    join(tmpdir(), "semantscript-refund-program-"),
  );
  t.after(() =>
    Promise.all([
      rm(firstRoot, { recursive: true }),
      rm(secondRoot, { recursive: true }),
    ]),
  );

  const first = await compileRefundProgram(firstRoot);
  const second = await compileRefundProgram(secondRoot);
  assert.equal(first.bundleText, second.bundleText);

  const bundle = JSON.parse(await readFile(first.bundlePath, "utf8"));
  assert.equal(bundle.kind, "semantscript.ir-bundle");
  assert.equal(bundle.functions.length, 1);
  const [record] = bundle.functions;
  assert.equal(record.id, REFUND_FUNCTION_ID);
  assert.equal(record.semanticSha256, REFUND_FUNCTION_SEMANTIC_SHA256);
  assert.equal(record.stage, "source");
  assert.equal(record.runtime.resultMode, "diagnostic");
  assert.deepEqual(record.output.head.support, REFUND_TASK_SPEC.outputSupport);
  assert.deepEqual(record.definition.examples, []);
  assert.deepEqual(record.definition.template, REFUND_TASK_SPEC.template);
  assert.deepEqual(
    record.definition.constraints,
    REFUND_TASK_SPEC.hardConstraints,
  );
  assert.deepEqual(
    record.inputs.map(({ name }) => name),
    ["customer", "order"],
  );
  const orderInput = record.inputs.find(({ name }) => name === "order");
  const statusField = orderInput.type.fields.find(
    ({ name }) => name === "status",
  );
  assert.deepEqual(
    statusField.type.variants.map(({ value }) => value),
    REFUND_TASK_SPEC.inputDomain.order.status,
  );
  assert.match(REFUND_TASK_SPEC_SHA256, /^[a-f0-9]{64}$/u);
  const javascript = await readFile(
    join(firstRoot, "refund-with-confidence.sem.js"),
    "utf8",
  );
  assert.equal(javascript.includes("Apply our refund policy"), false);
  assert.match(
    javascript,
    /__sema(?:_\d+)?\.call\("nf_[a-f0-9]{64}", \{ customer, order \}\)/u,
  );
});

test("diagnostic guard requires full canonical probability support", () => {
  const valid = {
    value: "review",
    confidence: 0.5,
    uncertainty: 0.7,
    distribution: [
      { value: "approve", probability: 0.2 },
      { value: "deny", probability: 0.3 },
      { value: "review", probability: 0.5 },
    ],
    expectedValue: null,
  };
  assert.doesNotThrow(() => validateRefundDiagnostic(valid));
  assert.throws(
    () =>
      validateRefundDiagnostic({
        ...valid,
        distribution: [...valid.distribution].reverse(),
      }),
    /support-ordered/u,
  );
  assert.throws(
    () =>
      validateRefundDiagnostic({
        ...valid,
        distribution: valid.distribution.map((entry) => ({
          ...entry,
          probability: entry.probability / 2,
        })),
      }),
    /sum to one/u,
  );
});

test("runtime diagnostic dispatch stays scoped to the loaded artifact handle", async () => {
  const source = await readFile(
    new URL("./run-runtime.mjs", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(source, /\b__sema\.call\s*\(/u);
  assert.match(source, /\bhandle\.call\(functionId, inputs\)/u);
});

test("stage scaling measurement times fused heads and batches on a fixture artifact", async (t) => {
  const { createFixtureArtifact, fixtureFunctionId } =
    await import("../../../runtime/test/fixtures/artifact.mjs");
  const { loadSemaArtifact } = await import("@semantscript/core");
  const { measureStageScaling, percentile, renderMarkdown, summarize } =
    await import("./run-stage-scaling.mjs");

  assert.deepEqual(summarize([3, 1, 2, 10]), {
    count: 4,
    meanMs: 4,
    p50Ms: 2,
    p95Ms: 10,
    minMs: 1,
    maxMs: 10,
  });
  assert.equal(percentile([1, 2, 3, 4, 5], 0.5), 3);
  assert.equal(percentile([1, 2, 3, 4, 5], 0.95), 5);

  const root = await mkdtemp(join(tmpdir(), "semantscript-stage-scaling-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const extraIds = [`nf_${"2".repeat(64)}`, `nf_${"3".repeat(64)}`];
  await createFixtureArtifact(root, {
    extraFunctions: extraIds.map((id, index) => ({
      id,
      headRef: `head.fixture.extra${String(index)}`,
    })),
  });
  const handle = await loadSemaArtifact(root);
  try {
    const inputs = [{ facts: { a: 1, b: 2 } }, { facts: { a: 3, b: 4 } }];
    let tick = 0;
    const record = measureStageScaling(handle, {
      inputs,
      headCounts: [1, 3],
      batchSizes: [1, 2],
      iterations: 2,
      warmupIterations: 1,
      now: () => (tick += 1),
    });

    assert.equal(record.functionCount, 3);
    assert.equal(record.distinctInputs, 2);
    assert.deepEqual(
      record.headsPerStage.map(({ heads, passes }) => [heads, passes]),
      [
        [1, { encoder: 1, adapter: 1, head: 1 }],
        [3, { encoder: 1, adapter: 1, head: 3 }],
      ],
    );
    assert.deepEqual(
      record.batchScaling.map(({ batch, passes }) => [batch, passes]),
      [
        [1, { encoder: 1, adapter: 1, head: 1 }],
        [2, { encoder: 2, adapter: 2, head: 2 }],
      ],
    );
    // The injected clock advances one unit per reading, so every timed call measures 1.
    for (const row of record.headsPerStage) {
      assert.equal(row.fused.p50Ms, 1);
      assert.equal(row.sequential.p50Ms, 1);
      assert.equal(row.fusedDecisionsPerSecond, row.heads * 1000);
      assert.equal(row.sequentialOverFusedP50, 1);
    }
    assert.equal(record.batchScaling[1].perDecisionP50Ms, 0.5);
    const markdown = renderMarkdown(record);
    assert.match(
      markdown,
      /\| 3 \| 1\.00 \| 1\.00 \| 1\.00 \| 1\.00x \| 3000\.0 \| 0\.33 \|/u,
    );
    assert.match(markdown, /\| 2 \| 1\.00 \| 1\.00 \| 0\.50 \| 2000\.0 \|/u);

    assert.throws(
      () =>
        measureStageScaling(handle, {
          inputs,
          headCounts: [4],
          batchSizes: [1],
          iterations: 1,
        }),
      /exposes 3 functions; 4 heads requested/u,
    );
    assert.throws(
      () =>
        measureStageScaling(handle, {
          inputs,
          headCounts: [1],
          batchSizes: [3],
          iterations: 1,
        }),
      /2 distinct inputs supplied; batch of 3 requested/u,
    );
  } finally {
    await handle.close();
  }
});
