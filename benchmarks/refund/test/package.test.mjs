import assert from "node:assert/strict";
import test from "node:test";

test("refund benchmark package exposes the complete benchmark harness", async () => {
  const benchmark = await import("../dist/index.js");

  assert.equal(typeof benchmark.validateRefundDataset, "function");
  assert.equal(typeof benchmark.auditDatasetSeparation, "function");
  assert.equal(typeof benchmark.evaluatePredictionSet, "function");
  assert.equal(typeof benchmark.createBenchmarkResult, "function");
  assert.equal(typeof benchmark.validateBenchmarkBundle, "function");
  assert.equal(typeof benchmark.createOllamaQwenAdapter, "function");
  assert.equal(typeof benchmark.createAnthropicSonnetAdapter, "function");
  assert.equal(typeof benchmark.createLayaAdapter, "function");
  assert.equal(typeof benchmark.createLiveOllamaTransport, "function");
  assert.equal(typeof benchmark.createLiveAnthropicTransport, "function");
  assert.equal(typeof benchmark.createLiveLayaRunner, "function");
  assert.equal(typeof benchmark.createSemantScriptRefundAdapter, "function");
  assert.equal(typeof benchmark.createProcessRssSampler, "function");
  assert.equal(typeof benchmark.runRefundBenchmark, "function");
  assert.match(benchmark.REFUND_TASK_SPEC_SHA256, /^[a-f0-9]{64}$/u);
  assert.equal(
    benchmark.REFUND_FUNCTION_ID,
    "nf_65e347f7dd8736c55d82e539be7ad005cedb3ca396dfa2ab89de619f77adfc9c",
  );
  assert.equal(
    benchmark.REFUND_FUNCTION_SEMANTIC_SHA256,
    "f7efe891ae5e62b2f0dcec517e118482c995468aaafd813f263c4179a99e545f",
  );
  assert.equal(benchmark.REFUND_SUPPORT.join(","), "approve,deny,review");
});
