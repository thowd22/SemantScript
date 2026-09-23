import assert from "node:assert/strict";
import test from "node:test";

import {
  REFUND_FUNCTION_ID,
  REFUND_FUNCTION_SEMANTIC_SHA256,
  REFUND_TASK_SPEC_SHA256,
} from "../dist/policy.js";
import {
  SEMANTSCRIPT_BENCHMARK_ADAPTER_VERSION,
  SemantScriptAdapterError,
  createSemantScriptRefundAdapter,
} from "../dist/semantscript-adapter.js";

const FUNCTION_ID = REFUND_FUNCTION_ID;
const MANIFEST_SHA256 = "2".repeat(64);
const DATASET_SHA256 = "3".repeat(64);
const TRAINING_KEY_SHA256 = "4".repeat(64);
const SEMANTIC_SHA256 = REFUND_FUNCTION_SEMANTIC_SHA256;

const INPUTS = Object.freeze({
  customer: Object.freeze({ priorRefunds: 1, tier: "standard" }),
  order: Object.freeze({ ageDays: 8, status: "paid", total: 49.95 }),
});

const FUNCTION_PROVENANCE = Object.freeze({
  id: FUNCTION_ID,
  semanticSha256: SEMANTIC_SHA256,
  resultMode: "diagnostic",
  trainingProvenance: Object.freeze({
    datasetSha256: DATASET_SHA256,
    trainingKeySha256: TRAINING_KEY_SHA256,
    baseModel: "answerdotai/ModernBERT-base@immutable-test-revision",
  }),
});

const MODEL = Object.freeze({
  provider: "semantscript",
  name: "refund-decision",
  version: "semantscript-artifact-v1",
  revision: TRAINING_KEY_SHA256,
  artifactSha256: MANIFEST_SHA256,
});

const DIAGNOSTIC = Object.freeze({
  value: "review",
  confidence: 0.6,
  uncertainty: 0.4,
  distribution: Object.freeze([
    Object.freeze({ value: "approve", probability: 0.1 }),
    Object.freeze({ value: "deny", probability: 0.3 }),
    Object.freeze({ value: "review", probability: 0.6 }),
  ]),
  expectedValue: null,
});

const TRAINING_EVIDENCE = Object.freeze({
  trainingLedgerSha256: "6".repeat(64),
  artifactTrainingDatasetSha256: DATASET_SHA256,
  artifactTrainingKeySha256: TRAINING_KEY_SHA256,
  releaseVerificationPayloadSha256: "7".repeat(64),
  releaseVerificationAttestationSha256: "8".repeat(64),
});

function expected(overrides = {}) {
  return {
    taskSpecSha256: REFUND_TASK_SPEC_SHA256,
    function: FUNCTION_PROVENANCE,
    model: MODEL,
    trainingEvidence: TRAINING_EVIDENCE,
    ...overrides,
  };
}

function fixtureLoader(overrides = {}) {
  const state = { loads: [], closes: 0, calls: [] };
  const loaded = {
    manifestSha256: MANIFEST_SHA256,
    functionIds: new Set([FUNCTION_ID]),
    functions: [FUNCTION_PROVENANCE],
    call(functionId, inputs) {
      state.calls.push({ functionId, inputs });
      return DIAGNOSTIC;
    },
    async close() {
      state.closes += 1;
    },
    ...overrides,
  };
  return {
    state,
    loader: {
      async load(artifactRoot) {
        state.loads.push(artifactRoot);
        return loaded;
      },
    },
  };
}

function adapterError(code) {
  return (error) => {
    assert.ok(error instanceof SemantScriptAdapterError);
    assert.equal(error.code, code);
    return true;
  };
}

test("loads one artifact and reuses one persistent runtime for every prediction", async () => {
  const { loader, state } = fixtureLoader();
  const adapter = await createSemantScriptRefundAdapter({
    artifactRoot: "/test/refund-artifact",
    expected: expected(),
    loader,
  });

  const first = await adapter.predict(INPUTS);
  const second = await adapter.predict(INPUTS);
  assert.deepEqual(state.loads, ["/test/refund-artifact"]);
  assert.equal(state.calls.length, 2);
  assert.equal(state.calls.every(({ functionId }) => functionId === FUNCTION_ID), true);
  assert.equal(state.calls.every(({ inputs }) => inputs === INPUTS), true);
  assert.deepEqual(first, {
    value: "review",
    distribution: DIAGNOSTIC.distribution,
  });
  assert.deepEqual(second, first);
  assert.equal(first.distribution[2].probability, 0.6);
  assert.equal(Object.isFrozen(first), true);
  assert.equal(Object.isFrozen(first.distribution), true);
  assert.equal(adapter.role, "semantscript");
  assert.deepEqual(adapter.model, MODEL);
  assert.equal(adapter.taskSpecSha256, REFUND_TASK_SPEC_SHA256);
  assert.equal(adapter.adapter.version, SEMANTSCRIPT_BENCHMARK_ADAPTER_VERSION);
  assert.match(adapter.adapter.configurationSha256, /^[a-f0-9]{64}$/u);

  await adapter.close();
  await adapter.close();
  assert.equal(state.closes, 1);
  await assert.rejects(adapter.predict(INPUTS), adapterError("closed"));
});

test("rejects task, artifact, function, training, and result-mode provenance drift", async () => {
  const preLoad = fixtureLoader();
  await assert.rejects(
    createSemantScriptRefundAdapter({
      artifactRoot: "/test/refund-artifact",
      expected: expected({ taskSpecSha256: "a".repeat(64) }),
      loader: preLoad.loader,
    }),
    adapterError("provenance-mismatch"),
  );
  assert.equal(preLoad.state.loads.length, 0);

  for (const expectedOverride of [
    { model: { ...MODEL, provider: "not-semantscript" } },
    { model: { ...MODEL, revision: "9".repeat(64) } },
    {
      trainingEvidence: {
        ...TRAINING_EVIDENCE,
        trainingLedgerSha256: "not-a-digest",
      },
    },
    {
      trainingEvidence: {
        ...TRAINING_EVIDENCE,
        artifactTrainingDatasetSha256: "9".repeat(64),
      },
    },
    {
      trainingEvidence: {
        ...TRAINING_EVIDENCE,
        artifactTrainingKeySha256: "9".repeat(64),
      },
    },
  ]) {
    const invalid = fixtureLoader();
    await assert.rejects(
      createSemantScriptRefundAdapter({
        artifactRoot: "/test/refund-artifact",
        expected: expected(expectedOverride),
        loader: invalid.loader,
      }),
      (error) =>
        error instanceof SemantScriptAdapterError &&
        ["invalid-configuration", "provenance-mismatch"].includes(error.code),
    );
    assert.equal(invalid.state.loads.length, 0);
  }

  const cases = [
    { manifestSha256: "a".repeat(64) },
    { functionIds: new Set() },
    {
      functions: [{ ...FUNCTION_PROVENANCE, semanticSha256: "a".repeat(64) }],
    },
    {
      functions: [
        {
          ...FUNCTION_PROVENANCE,
          trainingProvenance: {
            ...FUNCTION_PROVENANCE.trainingProvenance,
            datasetSha256: "a".repeat(64),
          },
        },
      ],
    },
    {
      functions: [
        {
          ...FUNCTION_PROVENANCE,
          trainingProvenance: {
            ...FUNCTION_PROVENANCE.trainingProvenance,
            trainingKeySha256: "a".repeat(64),
          },
        },
      ],
    },
    { functions: [{ ...FUNCTION_PROVENANCE, resultMode: "value" }] },
  ];
  for (const loadedOverrides of cases) {
    const fixture = fixtureLoader(loadedOverrides);
    await assert.rejects(
      createSemantScriptRefundAdapter({
        artifactRoot: "/test/refund-artifact",
        expected: expected(),
        loader: fixture.loader,
      }),
      adapterError("provenance-mismatch"),
    );
    assert.equal(fixture.state.loads.length, 1);
    assert.equal(fixture.state.closes, 1);
  }
});

test("returns exact runtime probabilities and rejects malformed diagnostics", async () => {
  const responses = [
    {
      ...DIAGNOSTIC,
      distribution: DIAGNOSTIC.distribution.map((entry) => ({
        ...entry,
        probability: entry.probability / 2,
      })),
    },
    {
      ...DIAGNOSTIC,
      value: "deny",
    },
    {
      ...DIAGNOSTIC,
      distribution: [...DIAGNOSTIC.distribution].reverse(),
    },
    {
      ...DIAGNOSTIC,
      confidence: 0.61,
    },
    {
      ...DIAGNOSTIC,
      extra: true,
    },
  ];
  for (const response of responses) {
    const fixture = fixtureLoader({ call() { return response; } });
    const adapter = await createSemantScriptRefundAdapter({
      artifactRoot: "/test/refund-artifact",
      expected: expected(),
      loader: fixture.loader,
    });
    await assert.rejects(adapter.predict(INPUTS), adapterError("invalid-output"));
    await adapter.close();
  }
});

test("close rejects new work and waits for an active runtime call", async () => {
  const fixture = fixtureLoader({ async call() { return await pendingResponse; } });
  let resolveCall;
  const pendingResponse = new Promise((resolve) => {
    resolveCall = resolve;
  });
  const adapter = await createSemantScriptRefundAdapter({
    artifactRoot: "/test/refund-artifact",
    expected: expected(),
    loader: fixture.loader,
  });
  const prediction = adapter.predict(INPUTS);
  const closing = adapter.close();
  await assert.rejects(adapter.predict(INPUTS), adapterError("closed"));
  await Promise.resolve();
  assert.equal(fixture.state.closes, 0);
  resolveCall(DIAGNOSTIC);
  assert.equal((await prediction).value, "review");
  await closing;
  assert.equal(fixture.state.closes, 1);
});

test("an already-aborted prediction never calls the persistent runtime", async () => {
  let calls = 0;
  const fixture = fixtureLoader({ call() { calls += 1; return DIAGNOSTIC; } });
  const controller = new globalThis.AbortController();
  controller.abort(new Error("cancelled"));
  const adapter = await createSemantScriptRefundAdapter({
    artifactRoot: "/test/refund-artifact",
    expected: expected(),
    loader: fixture.loader,
  });
  await assert.rejects(adapter.predict(INPUTS, controller.signal), adapterError("aborted"));
  assert.equal(calls, 0);
  await adapter.close();
});
