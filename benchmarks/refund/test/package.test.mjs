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
    "nf_955824a910df4df5cc32a079555fe109919c41492697d9a1cc507decc5afba20",
  );
  assert.equal(
    benchmark.REFUND_FUNCTION_SEMANTIC_SHA256,
    "29b03f7d9ec695eb4178e6c4320b6094f7d1c37bc6bfca3516e983e93a0dc1f1",
  );
  assert.equal(benchmark.REFUND_SUPPORT.join(","), "approve,deny,review");
});
