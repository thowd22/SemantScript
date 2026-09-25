import assert from "node:assert/strict";
import test from "node:test";

import { semanticJsonSha256 } from "@semantscript/compiler";

import {
  ANTHROPIC_SONNET_MODEL,
  BaselineAdapterError,
  LAYA_CHECKPOINT,
  LAYA_CHECKPOINT_REVISION,
  LAYA_CHECKPOINT_SHA256,
  LAYA_CODE_REVISION,
  OLLAMA_QWEN_MANIFEST_SHA256,
  OLLAMA_QWEN_MODELS,
  OLLAMA_QWEN_WEIGHT_SHA256,
  REFUND_BASELINE_POLICY,
  REFUND_OUTPUT_SCHEMA,
  REFUND_TASK_SPEC,
  REFUND_TASK_SPEC_SHA256,
  buildRefundPrompt,
  canonicalRefundInput,
  createAnthropicSonnetAdapter,
  createLayaAdapter,
  createOllamaQwenAdapter,
  parseStructuredPrediction,
} from "../dist/adapters/index.js";

const INPUTS = Object.freeze({
  customer: Object.freeze({ priorRefunds: 1, tier: "standard" }),
  order: Object.freeze({ ageDays: 8, status: "paid", total: 49.95 }),
});

const SHA_B = "b".repeat(64);

function ollamaModel(role = "ollama-1b") {
  return {
    provider: "ollama",
    name: OLLAMA_QWEN_MODELS[role],
    version: "qwen2.5",
    revision: OLLAMA_QWEN_MANIFEST_SHA256[role],
    artifactSha256: OLLAMA_QWEN_WEIGHT_SHA256[role],
  };
}

function anthropicModel() {
  return {
    provider: "anthropic",
    name: ANTHROPIC_SONNET_MODEL,
    version: ANTHROPIC_SONNET_MODEL,
    revision: ANTHROPIC_SONNET_MODEL,
    artifactSha256: SHA_B,
  };
}

function layaModel() {
  return {
    provider: "huggingface",
    name: LAYA_CHECKPOINT,
    version: "laya-typed-decisions",
    revision: LAYA_CHECKPOINT_REVISION,
    artifactSha256: LAYA_CHECKPOINT_SHA256,
  };
}

function structuredOutput(decision = "approve") {
  return {
    decision,
    probabilities: { approve: 0.6, deny: 0.1, review: 0.3 },
  };
}

function assertAdapterError(code) {
  return (error) => {
    assert.ok(error instanceof BaselineAdapterError);
    assert.equal(error.code, code);
    return true;
  };
}

test("canonical input ignores object insertion order and rejects widened inputs", () => {
  const reordered = {
    order: { total: 49.95, status: "paid", ageDays: 8 },
    customer: { tier: "standard", priorRefunds: 1 },
  };
  assert.equal(canonicalRefundInput(reordered), canonicalRefundInput(INPUTS));
  assert.equal(
    canonicalRefundInput(reordered),
    '{"customer":{"priorRefunds":1,"tier":"standard"},"order":{"ageDays":8,"status":"paid","total":49.95}}',
  );
  assert.throws(
    () => canonicalRefundInput({ ...INPUTS, untracked: true }),
    assertAdapterError("invalid-response"),
  );
  for (const inputs of [
    { ...INPUTS, customer: { ...INPUTS.customer, priorRefunds: -0 } },
    { ...INPUTS, order: { ...INPUTS.order, ageDays: -0 } },
    { ...INPUTS, order: { ...INPUTS.order, total: -0 } },
  ]) {
    assert.throws(
      () => canonicalRefundInput(inputs),
      assertAdapterError("invalid-configuration"),
    );
  }
});

test("baseline prompt is a readable field-ordered snapshot of the complete task", () => {
  assert.deepEqual(buildRefundPrompt(INPUTS), {
    system:
      "Apply the complete refund policy and hard constraints supplied by the user. Return only the required structured decision and a complete probability distribution. The decision must be the first option in support order among tied maximum probabilities.",
    user: [
      "Prompt protocol: refund-decision.v2",
      `Task specification SHA-256: ${REFUND_TASK_SPEC_SHA256}`,
      "Policy: An order older than 90 days is always denied. A fraudulent order that is 90 days old or younger goes to review. A paid order outside its refund window is denied: enterprise customers get 60 days and everyone else gets 30, counting the window as inclusive. A paid order inside its window goes to review when the circumstances are suspicious, meaning five or more prior refunds, three or more prior refunds on an order of at least 1000, an order total of at least 5000, or a same-day cancellation (age at most 1 day) of an order of at least 2000 by a customer with two or more prior refunds. Otherwise a paid order inside its window is approved.",
      "Hard constraints:",
      '1. An order whose status is "fraudulent" must never be classified as "approve".',
      '2. An order whose ageDays is greater than 90 must always be classified as "deny".',
      '3. An order whose ageDays is at most 90 and whose status is "fraudulent" must always be classified as "review".',
      '4. A "paid" order whose ageDays is at most 90 and outside its window (standard tier: ageDays greater than 30; enterprise tier: ageDays greater than 60) must always be classified as "deny".',
      '5. A "paid" order inside its window (standard tier: ageDays at most 30; enterprise tier: ageDays at most 60) must always be classified as "review" when priorRefunds is at least 5, or priorRefunds is at least 3 and total is at least 1000, or total is at least 5000, or ageDays is at most 1 and total is at least 2000 and priorRefunds is at least 2.',
      '6. A "paid" order inside its window that meets none of those suspicious conditions must always be classified as "approve".',
      'Input domain: order.status is exactly "paid" or "fraudulent".',
      "Support order: approve, deny, review",
      'Refund input (canonical refund JSON): {"customer":{"priorRefunds":1,"tier":"standard"},"order":{"ageDays":8,"status":"paid","total":49.95}}',
    ].join("\n"),
  });
});

test("canonical task specification is deeply immutable and digest bound", () => {
  const visit = (value) => {
    if (value !== null && typeof value === "object") {
      assert.equal(Object.isFrozen(value), true);
      for (const child of Object.values(value)) visit(child);
    }
  };
  visit(REFUND_TASK_SPEC);
  visit(REFUND_BASELINE_POLICY);
  assert.match(REFUND_TASK_SPEC_SHA256, /^[a-f0-9]{64}$/u);
  assert.deepEqual(REFUND_TASK_SPEC.inputDomain.order.status, [
    "fraudulent",
    "paid",
  ]);
  assert.equal(
    REFUND_TASK_SPEC.policy,
    "An order older than 90 days is always denied. A fraudulent order that is 90 days old or younger goes to review. A paid order outside its refund window is denied: enterprise customers get 60 days and everyone else gets 30, counting the window as inclusive. A paid order inside its window goes to review when the circumstances are suspicious, meaning five or more prior refunds, three or more prior refunds on an order of at least 1000, an order total of at least 5000, or a same-day cancellation (age at most 1 day) of an order of at least 2000 by a customer with two or more prior refunds. Otherwise a paid order inside its window is approved",
  );
  assert.equal(REFUND_TASK_SPEC.hardConstraints.length, 6);
});

test("Ollama Qwen adapter fixes decoding, schema, input, and provenance", async () => {
  const calls = [];
  const transport = {
    async generate(request, signal) {
      calls.push({ request, signal });
      return {
        model: OLLAMA_QWEN_MODELS["ollama-1b"],
        response: JSON.stringify(structuredOutput()),
        done: true,
        done_reason: "stop",
      };
    },
  };
  const adapter = createOllamaQwenAdapter({
    role: "ollama-1b",
    model: ollamaModel(),
    transport,
    timeoutMs: 5_000,
  });
  const prediction = await adapter.predict(INPUTS);

  assert.equal(adapter.role, "ollama-1b");
  assert.deepEqual(adapter.model, ollamaModel());
  assert.notEqual(adapter.model.revision, adapter.model.artifactSha256);
  assert.match(adapter.adapter.configurationSha256, /^[a-f0-9]{64}$/);
  assert.equal(adapter.taskSpecSha256, REFUND_TASK_SPEC_SHA256);
  assert.equal(
    adapter.adapter.configurationSha256,
    semanticJsonSha256({
      provider: "ollama",
      role: "ollama-1b",
      model: ollamaModel(),
      promptVersion: "refund-decision.v2",
      taskSpecSha256: REFUND_TASK_SPEC_SHA256,
      timeoutMs: 5_000,
      structuredOutputSchema: REFUND_OUTPUT_SCHEMA,
      decoding: { temperature: 0, seed: 0, numPredict: 256 },
    }),
  );
  assert.equal(prediction.value, "approve");
  assert.deepEqual(
    prediction.distribution.map(({ value }) => value),
    ["approve", "deny", "review"],
  );
  assert.equal(
    prediction.distribution.reduce((sum, entry) => sum + entry.probability, 0),
    1,
  );
  assert.ok(Object.isFrozen(prediction));
  assert.equal(calls.length, 1);
  const [{ request, signal }] = calls;
  assert.equal(request.model, OLLAMA_QWEN_MODELS["ollama-1b"]);
  assert.equal(request.stream, false);
  assert.equal(request.think, false);
  assert.deepEqual(request.options, {
    temperature: 0,
    seed: 0,
    num_predict: 256,
  });
  assert.strictEqual(request.format, REFUND_OUTPUT_SCHEMA);
  assert.ok(request.prompt.includes(canonicalRefundInput(INPUTS)));
  assert.match(
    request.prompt,
    /enterprise customers get 60 days and everyone else gets 30/u,
  );
  assert.match(request.prompt, /fraudulent.+never.+approve/u);
  assert.match(request.prompt, /ageDays.+greater than 90.+always.+deny/u);
  assert.ok(request.prompt.includes(REFUND_TASK_SPEC_SHA256));
  assert.equal(signal.aborted, false);

  const sameConfiguration = createOllamaQwenAdapter({
    role: "ollama-1b",
    model: ollamaModel(),
    transport: {
      async generate() {
        throw new Error("unused");
      },
    },
    timeoutMs: 5_000,
  });
  assert.equal(
    adapter.adapter.configurationSha256,
    sameConfiguration.adapter.configurationSha256,
  );
});

test("Ollama provenance pins distinct full manifests and weight blobs", () => {
  assert.equal(Object.isFrozen(OLLAMA_QWEN_MANIFEST_SHA256), true);
  assert.equal(Object.isFrozen(OLLAMA_QWEN_WEIGHT_SHA256), true);
  for (const role of ["ollama-1b", "ollama-7b"]) {
    assert.match(OLLAMA_QWEN_MANIFEST_SHA256[role], /^[a-f0-9]{64}$/u);
    assert.match(OLLAMA_QWEN_WEIGHT_SHA256[role], /^[a-f0-9]{64}$/u);
    assert.notEqual(
      OLLAMA_QWEN_MANIFEST_SHA256[role],
      OLLAMA_QWEN_WEIGHT_SHA256[role],
    );
    const adapter = createOllamaQwenAdapter({
      role,
      model: ollamaModel(role),
      transport: {
        async generate() {
          throw new Error("unused");
        },
      },
    });
    assert.equal(adapter.model.revision, OLLAMA_QWEN_MANIFEST_SHA256[role]);
    assert.equal(adapter.model.artifactSha256, OLLAMA_QWEN_WEIGHT_SHA256[role]);
  }
});

test("Ollama adapter rejects unpinned models, refusals, and invalid distributions", async () => {
  assert.throws(
    () =>
      createOllamaQwenAdapter({
        role: "ollama-1b",
        model: {
          ...ollamaModel(),
          revision: OLLAMA_QWEN_MANIFEST_SHA256["ollama-7b"],
        },
        transport: {
          async generate() {
            throw new Error("unused");
          },
        },
      }),
    assertAdapterError("invalid-configuration"),
  );
  assert.throws(
    () =>
      createOllamaQwenAdapter({
        role: "ollama-1b",
        model: {
          ...ollamaModel(),
          artifactSha256: OLLAMA_QWEN_WEIGHT_SHA256["ollama-7b"],
        },
        transport: {
          async generate() {
            throw new Error("unused");
          },
        },
      }),
    assertAdapterError("invalid-configuration"),
  );

  const response = {
    model: OLLAMA_QWEN_MODELS["ollama-1b"],
    response: JSON.stringify(structuredOutput()),
    done: true,
    done_reason: "refusal",
  };
  const adapter = createOllamaQwenAdapter({
    role: "ollama-1b",
    model: ollamaModel(),
    transport: {
      async generate() {
        return response;
      },
    },
  });
  await assert.rejects(
    adapter.predict(INPUTS),
    assertAdapterError("refused-output"),
  );

  response.done_reason = "stop";
  response.response = JSON.stringify({
    decision: "deny",
    probabilities: { approve: 0.6, deny: 0.1, review: 0.3 },
  });
  await assert.rejects(
    adapter.predict(INPUTS),
    assertAdapterError("invalid-response"),
  );

  response.response = JSON.stringify({
    decision: "approve",
    probabilities: { approve: 0.6, deny: 0.4 },
  });
  await assert.rejects(
    adapter.predict(INPUTS),
    assertAdapterError("invalid-response"),
  );
});

test("structured responses reject symbols, accessors, and non-plain objects", async () => {
  const outputWithSymbol = structuredOutput();
  outputWithSymbol[Symbol("hidden")] = true;
  const response = {
    model: ANTHROPIC_SONNET_MODEL,
    stopReason: "end_turn",
    output: outputWithSymbol,
  };
  const adapter = createAnthropicSonnetAdapter({
    model: anthropicModel(),
    transport: {
      async generate() {
        return response;
      },
    },
  });
  await assert.rejects(
    adapter.predict(INPUTS),
    assertAdapterError("invalid-response"),
  );

  response.output = {
    decision: "approve",
    get probabilities() {
      return { approve: 0.6, deny: 0.1, review: 0.3 };
    },
  };
  await assert.rejects(
    adapter.predict(INPUTS),
    assertAdapterError("invalid-response"),
  );

  response.output = Object.assign(
    Object.create({ inherited: true }),
    structuredOutput(),
  );
  await assert.rejects(
    adapter.predict(INPUTS),
    assertAdapterError("invalid-response"),
  );

  const envelope = Object.create(null);
  Object.defineProperties(envelope, {
    model: { enumerable: true, value: ANTHROPIC_SONNET_MODEL },
    stopReason: { enumerable: true, value: "end_turn" },
    output: { enumerable: true, get: () => structuredOutput() },
  });
  const envelopeAdapter = createAnthropicSonnetAdapter({
    model: anthropicModel(),
    transport: {
      async generate() {
        return envelope;
      },
    },
  });
  await assert.rejects(
    envelopeAdapter.predict(INPUTS),
    assertAdapterError("invalid-response"),
  );
});

test("adapter-enforced timeout aborts an unresponsive transport", async () => {
  let observedAbort = false;
  const adapter = createOllamaQwenAdapter({
    role: "ollama-1b",
    model: ollamaModel(),
    transport: {
      async generate(_request, signal) {
        return await new Promise((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => {
              observedAbort = true;
              reject(new Error("aborted by adapter"));
            },
            { once: true },
          );
        });
      },
    },
    timeoutMs: 5,
  });
  await assert.rejects(adapter.predict(INPUTS), assertAdapterError("timeout"));
  assert.equal(observedAbort, true);
});

test("Sonnet 5 adapter uses fixed structured output without sampling overrides", async () => {
  let captured;
  const adapter = createAnthropicSonnetAdapter({
    model: anthropicModel(),
    transport: {
      async generate(request) {
        captured = request;
        return {
          model: ANTHROPIC_SONNET_MODEL,
          stopReason: "end_turn",
          output: structuredOutput(),
        };
      },
    },
  });
  const prediction = await adapter.predict(INPUTS);

  assert.equal(adapter.role, "structured-api");
  assert.equal(adapter.taskSpecSha256, REFUND_TASK_SPEC_SHA256);
  assert.equal(prediction.value, "approve");
  assert.equal(captured.model, ANTHROPIC_SONNET_MODEL);
  assert.equal(captured.maxTokens, 256);
  assert.equal(Object.hasOwn(captured, "temperature"), false);
  assert.equal(Object.hasOwn(captured, "topP"), false);
  assert.strictEqual(captured.outputConfig.format.schema, REFUND_OUTPUT_SCHEMA);
  assert.match(
    captured.messages[0].content,
    /Support order: approve, deny, review/,
  );
  assert.match(captured.messages[0].content, /everyone else gets 30/u);
  assert.ok(captured.messages[0].content.includes(REFUND_TASK_SPEC_SHA256));
});

test("Sonnet 5 adapter rejects mutable aliases, model drift, and refusals", async () => {
  assert.throws(
    () =>
      createAnthropicSonnetAdapter({
        model: { ...anthropicModel(), revision: "claude-sonnet-5-latest" },
        transport: {
          async generate() {
            throw new Error("unused");
          },
        },
      }),
    assertAdapterError("invalid-configuration"),
  );

  const response = {
    model: "claude-sonnet-5-other",
    stopReason: "end_turn",
    output: structuredOutput(),
  };
  const adapter = createAnthropicSonnetAdapter({
    model: anthropicModel(),
    transport: {
      async generate() {
        return response;
      },
    },
  });
  await assert.rejects(
    adapter.predict(INPUTS),
    assertAdapterError("invalid-response"),
  );
  response.model = ANTHROPIC_SONNET_MODEL;
  response.stopReason = "refusal";
  await assert.rejects(
    adapter.predict(INPUTS),
    assertAdapterError("refused-output"),
  );
});

test("Laya adapter fixes checkpoint, code, input, option order, and calibrated logits", async () => {
  let captured;
  const adapter = createLayaAdapter({
    model: layaModel(),
    runner: {
      async choose(request) {
        captured = request;
        return {
          checkpointRevision: LAYA_CHECKPOINT_REVISION,
          selectedIndex: 1,
          logits: [0, 2, 1],
        };
      },
    },
  });
  const prediction = await adapter.predict(INPUTS);

  assert.equal(adapter.role, "laya");
  assert.equal(adapter.taskSpecSha256, REFUND_TASK_SPEC_SHA256);
  assert.equal(prediction.value, "deny");
  assert.equal(captured.checkpoint, LAYA_CHECKPOINT);
  assert.equal(captured.checkpointRevision, LAYA_CHECKPOINT_REVISION);
  assert.equal(captured.codeRevision, LAYA_CODE_REVISION);
  assert.equal(captured.questionType, "choice");
  assert.deepEqual(captured.options, ["approve", "deny", "review"]);
  assert.equal(captured.applyCheckpointCalibration, true);
  assert.match(captured.question, /Support order: approve, deny, review/);
  assert.match(captured.question, /everyone else gets 30/u);
  assert.ok(captured.question.includes(REFUND_TASK_SPEC_SHA256));
  assert.equal(
    Math.abs(
      prediction.distribution.reduce(
        (sum, entry) => sum + entry.probability,
        0,
      ) - 1,
    ) < 1e-12,
    true,
  );
  assert.ok(
    prediction.distribution[1].probability >
      prediction.distribution[2].probability,
  );
});

test("Laya adapter rejects checkpoint drift, non-finite logits, and argmax mismatch", async () => {
  const response = {
    checkpointRevision: "0".repeat(40),
    selectedIndex: 1,
    logits: [0, 2, 1],
  };
  const adapter = createLayaAdapter({
    model: layaModel(),
    runner: {
      async choose() {
        return response;
      },
    },
  });
  await assert.rejects(
    adapter.predict(INPUTS),
    assertAdapterError("invalid-response"),
  );
  response.checkpointRevision = LAYA_CHECKPOINT_REVISION;
  response.selectedIndex = 0;
  await assert.rejects(
    adapter.predict(INPUTS),
    assertAdapterError("invalid-response"),
  );
  response.selectedIndex = 1;
  response.logits = [0, Number.NaN, 1];
  await assert.rejects(
    adapter.predict(INPUTS),
    assertAdapterError("invalid-response"),
  );

  const accessorLogits = [0, 2, 1];
  Object.defineProperty(accessorLogits, "1", {
    enumerable: true,
    get: () => 2,
  });
  response.logits = accessorLogits;
  await assert.rejects(
    adapter.predict(INPUTS),
    assertAdapterError("invalid-response"),
  );

  const extendedLogits = [0, 2, 1];
  extendedLogits.extra = 0;
  response.logits = extendedLogits;
  await assert.rejects(
    adapter.predict(INPUTS),
    assertAdapterError("invalid-response"),
  );
});

test("structured probabilities are renormalized within tolerance and rejected beyond it", () => {
  const rounded = parseStructuredPrediction({
    decision: "approve",
    probabilities: { approve: 0.9, deny: 0.05, review: 0.04 },
  });
  const total = rounded.distribution.reduce(
    (sum, entry) => sum + entry.probability,
    0,
  );
  assert.ok(Math.abs(total - 1) <= 1e-12);
  assert.equal(rounded.value, "approve");
  assert.ok(Math.abs(rounded.distribution[0].probability - 0.9 / 0.99) < 1e-12);

  assert.throws(
    () =>
      parseStructuredPrediction({
        decision: "approve",
        probabilities: { approve: 0.6, deny: 0.1, review: 0.1 },
      }),
    /must sum to 1 within 0\.1/,
  );
});
