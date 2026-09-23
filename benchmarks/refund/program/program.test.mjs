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
  const firstRoot = await mkdtemp(join(tmpdir(), "semantscript-refund-program-"));
  const secondRoot = await mkdtemp(join(tmpdir(), "semantscript-refund-program-"));
  t.after(() => Promise.all([rm(firstRoot, { recursive: true }), rm(secondRoot, { recursive: true })]));

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
  assert.deepEqual(record.definition.constraints, REFUND_TASK_SPEC.hardConstraints);
  assert.deepEqual(
    record.inputs.map(({ name }) => name),
    ["customer", "order"],
  );
  const orderInput = record.inputs.find(({ name }) => name === "order");
  const statusField = orderInput.type.fields.find(({ name }) => name === "status");
  assert.deepEqual(
    statusField.type.variants.map(({ value }) => value),
    REFUND_TASK_SPEC.inputDomain.order.status,
  );
  assert.match(REFUND_TASK_SPEC_SHA256, /^[a-f0-9]{64}$/u);
  const javascript = await readFile(join(firstRoot, "refund-with-confidence.sem.js"), "utf8");
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
  const source = await readFile(new URL("./run-runtime.mjs", import.meta.url), "utf8");
  assert.doesNotMatch(source, /\b__sema\.call\s*\(/u);
  assert.match(source, /\bhandle\.call\(functionId, inputs\)/u);
});
