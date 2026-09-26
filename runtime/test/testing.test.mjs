import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  createFixtureArtifact,
  fixtureFunctionId,
} from "./fixtures/artifact.mjs";

const fixtures = new URL("./fixtures/", import.meta.url);

// A synthetic IR bundle covering every head family, a flat object, a
// thresholded function with a fallback, two adapters and a depth-routed
// encoder, in the shape the compiler emits.
const ids = {
  label: `nf_${"a".repeat(64)}`,
  flag: `nf_${"b".repeat(64)}`,
  score: `nf_${"c".repeat(64)}`,
  triage: `nf_${"d".repeat(64)}`,
  routed: `nf_${"e".repeat(64)}`,
};

const factsInputs = [
  {
    name: "facts",
    index: 0,
    tsType: "FixtureFacts",
    type: {
      kind: "object",
      name: "FixtureFacts",
      fields: [
        { name: "a", optional: false, type: { kind: "number" } },
        { name: "b", optional: false, type: { kind: "number" } },
      ],
    },
  },
];

function irFunction(id, output, options = {}) {
  const fields =
    output.kind === "scalar" ? [""] : output.fields.map((f) => `/${f.name}`);
  return {
    kind: "semantscript.neural-function",
    irVersion: 1,
    stage: "source",
    id,
    semanticSha256: id.slice(3),
    source: {
      path: "src/app.sem.ts",
      line: 1,
      column: 1,
      sourceSha256: "0".repeat(64),
    },
    inputs: options.inputs ?? factsInputs,
    output,
    model: {
      adapter: options.adapter ?? "adapter.app.core",
      encoder: options.encoder ?? "encoder.app.full",
      ...(options.encoderDepth === undefined
        ? {}
        : { encoderDepth: options.encoderDepth }),
      heads: fields.map((outputPath, index) => ({
        outputPath,
        ref: `head.${id.slice(3, 11)}.${String(index).padStart(3, "0")}`,
      })),
    },
    runtime: {
      resultMode: options.resultMode ?? "value",
      confidenceThreshold: options.threshold ?? null,
      fallbackRef: options.fallbackRef ?? null,
      synchronous: true,
    },
    trainingProvenance: { status: "pending" },
    verification: { status: "pending" },
  };
}

const labelHead = {
  kind: "nominal",
  sourceKind: "string-union",
  support: ["approve", "deny", "review"],
};
const booleanHead = {
  kind: "nominal",
  sourceKind: "boolean",
  support: [false, true],
};
const priorityHead = {
  kind: "ordinal",
  sourceKind: "ordinal-string",
  support: ["low", "high", "urgent"],
  expectedValue: "zero-based-rank",
};

const bundle = {
  kind: "semantscript.ir-bundle",
  bundleVersion: 1,
  functions: [
    irFunction(ids.label, { kind: "scalar", tsType: "Label", head: labelHead }),
    irFunction(ids.flag, {
      kind: "scalar",
      tsType: "boolean",
      head: booleanHead,
    }),
    irFunction(
      ids.score,
      {
        kind: "scalar",
        tsType: "BoundedInt<1, 3>",
        head: {
          kind: "ordinal",
          sourceKind: "bounded-int",
          minimum: "1",
          maximum: "3",
          step: "1",
          supportDecimal: ["1", "2", "3"],
          expectedValue: "numeric",
        },
      },
      { resultMode: "diagnostic" },
    ),
    irFunction(
      ids.triage,
      {
        kind: "object",
        tsType: "Triage",
        fields: [
          { name: "priority", head: priorityHead },
          { name: "needsHuman", head: booleanHead },
        ],
      },
      { threshold: 0.7, fallbackRef: "triage-by-hand" },
    ),
    irFunction(
      ids.routed,
      { kind: "scalar", tsType: "Label", head: labelHead },
      {
        adapter: "adapter.app.refunds",
        encoder: "encoder.app.depth-006",
        encoderDepth: 6,
        threshold: 0.9,
      },
    ),
  ],
};

const facts = { a: 1, b: 2 };
const byHand = { priority: "high", needsHuman: true };

async function runtimeModules() {
  return {
    runtime: await import("../dist/index.js"),
    testing: await import("../dist/testing.js"),
  };
}

function fallbacks(calls = []) {
  return new Map([
    [
      "triage-by-hand",
      (inputs, diagnostic, threshold) => {
        calls.push({ inputs, diagnostic, threshold });
        return byHand;
      },
    ],
  ]);
}

test("the testing subpath resolves through the package exports map", async () => {
  const testing = await import("@semantscript/core/testing");
  assert.equal(typeof testing.createSemaStubArtifact, "function");
  assert.equal(typeof testing.loadSemaStubArtifact, "function");
  assert.equal(typeof testing.SemaStubError, "function");
  const manifest = JSON.parse(
    await readFile(new URL("../package.json", import.meta.url), "utf8"),
  );
  assert.deepEqual(manifest.exports["./testing"], {
    types: "./dist/testing.d.ts",
    default: "./dist/testing.js",
  });
});

test("the stub writes the fixtures' ONNX graphs byte for byte and a manifest derived from the IR", async () => {
  const onnx = await import("../dist/stub-onnx.js");
  const names = {
    producer: "semantscript-test-fixture",
    graphPrefix: "fixture",
  };
  const fixture = async (name) =>
    new Uint8Array(await readFile(new URL(name, fixtures)));
  assert.deepEqual(onnx.stubEncoderOnnx(names), await fixture("encoder.onnx"));
  assert.deepEqual(onnx.stubAdapterOnnx(names), await fixture("adapter.onnx"));
  assert.deepEqual(
    onnx.stubHeadOnnx(3, names, [0, 0.01, 0.02], [-1, 0, 1]),
    await fixture("head.onnx"),
  );

  const { testing } = await runtimeModules();
  const stub = await testing.createSemaStubArtifact(bundle, {
    answers: { [ids.label]: "deny" },
  });
  try {
    const pointer = JSON.parse(
      await readFile(join(stub.root, "current.json"), "utf8"),
    );
    assert.equal(pointer.manifestSha256, stub.manifestSha256);
    const manifest = JSON.parse(
      await readFile(join(stub.root, pointer.release, "manifest.json"), "utf8"),
    );
    // The full encoder is the model's; the depth-6 prefix is routed by ref.
    assert.deepEqual(manifest.model, {
      tokenizerRef: "tokenizer.stub",
      encoderRef: "encoder.app.full",
      adapterRef: "adapter.app.core",
    });
    const byId = new Map(manifest.functions.map((fn) => [fn.id, fn]));
    assert.equal(byId.get(ids.routed).encoderRef, "encoder.app.depth-006");
    assert.equal(byId.get(ids.routed).adapterRef, "adapter.app.refunds");
    assert.equal(byId.get(ids.label).encoderRef, undefined);
    assert.deepEqual([...stub.functionIds].sort(), Object.values(ids).sort());
    assert.deepEqual(byId.get(ids.flag).heads[0].type, {
      kind: "boolean",
      support: [false, true],
    });
    assert.equal(
      byId.get(ids.flag).heads[0].parameterization,
      "binary-sigmoid",
    );
    assert.equal(byId.get(ids.score).heads[0].type.kind, "ordinal-number");
    assert.deepEqual(
      byId
        .get(ids.triage)
        .heads.map((head) => [head.outputPath, head.type.kind]),
      [
        [["priority"], "ordinal-string"],
        [["needsHuman"], "boolean"],
      ],
    );
    assert.deepEqual(byId.get(ids.triage).runtime, {
      resultMode: "value",
      confidenceThreshold: 0.7,
      policy: "all-fields",
      fallbackRef: "triage-by-hand",
    });
    assert.equal(
      byId.get(ids.label).trainingProvenance.teacher,
      "semantscript-stub",
    );
    // Every creation is its own release, so two stubs of one bundle stay apart.
    const other = await testing.createSemaStubArtifact(bundle, { answers: {} });
    assert.notEqual(other.manifestSha256, stub.manifestSha256);
    await other.dispose();
  } finally {
    await stub.dispose();
  }
  assert.equal(
    existsSync(stub.root),
    false,
    "dispose removes a temporary root",
  );
});

test("functions answer a fixed value, a value per input or a function of their inputs", async () => {
  const { runtime, testing } = await runtimeModules();
  const compiledFlag = function flag(facts) {
    return runtime.__sema.call(
      "nf_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      { facts },
    );
  };
  const handle = await testing.loadSemaStubArtifact(bundle, {
    answers: new Map([
      [ids.label, "deny"],
      // Keyed by the compiled function itself.
      [compiledFlag, { compute: ({ facts }) => facts.a > facts.b }],
      [
        ids.score,
        {
          byInput: [{ inputs: { facts }, value: 2, confidence: 0.8 }],
          otherwise: 3,
        },
      ],
      [ids.triage, { value: { priority: "urgent", needsHuman: false } }],
      [
        ids.routed,
        {
          byInput: [
            { inputs: { facts: { b: 2, a: 1 } }, value: "approve" },
            {
              inputs: { facts: { a: 5, b: 5 } },
              value: "review",
              confidence: 0.95,
            },
          ],
        },
      ],
    ]),
    fallbacks: fallbacks(),
  });
  try {
    assert.equal(runtime.__sema.call(ids.label, { facts }), "deny");
    assert.equal(compiledFlag(facts), false);
    assert.equal(compiledFlag({ a: 3, b: 2 }), true);
    assert.deepEqual(runtime.__sema.call(ids.triage, { facts }), {
      priority: "urgent",
      needsHuman: false,
    });
    // Matched on canonical input, so key order does not matter.
    assert.equal(runtime.__sema.call(ids.routed, { facts }), "approve");
    assert.equal(
      runtime.__sema.call(ids.routed, { facts: { a: 5, b: 5 } }),
      "review",
    );
    assert.throws(
      () => runtime.__sema.call(ids.routed, { facts: { a: 9, b: 9 } }),
      (error) =>
        error instanceof testing.SemaStubError &&
        error.reason === "unmatched-input" &&
        error.functionId === ids.routed,
    );

    // A diagnostic function returns the full distribution the answer implies.
    const score = runtime.__sema.call(ids.score, { facts });
    assert.equal(score.value, 2);
    assert.equal(score.confidence, 0.8);
    assert.deepEqual(
      score.distribution.map((entry) => entry.value),
      [1, 2, 3],
    );
    assert.ok(Math.abs(score.distribution[0].probability - 0.1) < 1e-12);
    assert.ok(Math.abs(score.expectedValue - 2) < 1e-12);
    assert.ok(score.uncertainty > 0 && score.uncertainty < 1);
    const other = runtime.__sema.call(ids.score, { facts: { a: 0, b: 0 } });
    assert.deepEqual(
      [other.value, other.confidence, other.uncertainty, other.expectedValue],
      [3, 1, 0, 3],
    );

    // Inputs are validated against the IR exactly as with a trained artifact.
    assert.throws(
      () => runtime.__sema.call(ids.label, { facts: { a: 1 } }),
      runtime.SemaInputError,
    );
  } finally {
    await handle.close();
  }
  assert.equal(existsSync(handle.root), false);
  assert.throws(
    () => handle.call(ids.label, { facts }),
    runtime.SemaArtifactInactiveError,
  );
});

test("the confidence policy and fallbacks see the stubbed confidence", async () => {
  const { runtime, testing } = await runtimeModules();
  const calls = [];
  const handle = await testing.loadSemaStubArtifact(bundle, {
    answers: {
      [ids.triage]: {
        byInput: [
          {
            inputs: { facts },
            value: { priority: "low", needsHuman: false },
            confidence: { priority: 0.6 },
          },
        ],
        otherwise: {
          value: { priority: "urgent", needsHuman: true },
          confidence: 0.75,
        },
      },
      [ids.routed]: {
        compute: ({ facts }) => ({
          value: "approve",
          confidence: facts.a === 1 ? 0.8 : 0.95,
        }),
      },
    },
    fallbacks: fallbacks(calls),
  });
  try {
    // Below the 0.7 threshold on one field: the fallback answers, with the diagnostic.
    assert.deepEqual(runtime.__sema.call(ids.triage, { facts }), byHand);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].threshold, 0.7);
    assert.equal(calls[0].diagnostic.minimumFieldConfidence, 0.6);
    assert.equal(calls[0].diagnostic.fields.needsHuman.confidence, 1);
    assert.deepEqual(calls[0].inputs, { facts });
    // Above it: the stubbed value.
    assert.deepEqual(
      runtime.__sema.call(ids.triage, { facts: { a: 2, b: 2 } }),
      {
        priority: "urgent",
        needsHuman: true,
      },
    );
    assert.equal(calls.length, 1);

    // No fallback: below the threshold throws SemaConfidenceError.
    assert.throws(
      () => runtime.__sema.call(ids.routed, { facts }),
      (error) =>
        error instanceof runtime.SemaConfidenceError &&
        error.threshold === 0.9 &&
        error.diagnostic.value === "approve" &&
        error.diagnostic.confidence === 0.8,
    );
    assert.equal(
      runtime.__sema.call(ids.routed, { facts: { a: 2, b: 2 } }),
      "approve",
    );
  } finally {
    await handle.close();
  }

  // A fallback the IR names must be registered, as with a trained artifact.
  await assert.rejects(
    testing.loadSemaStubArtifact(bundle, { answers: {} }),
    (error) =>
      error instanceof runtime.SemaFallbackError && error.reason === "missing",
  );
});

test("request scopes, stages and execution plans count the same passes as a real artifact", async (t) => {
  const { runtime, testing } = await runtimeModules();
  const routedId = `nf_${"9".repeat(64)}`;
  const siblingId = `nf_${"8".repeat(64)}`;

  // The fixture artifact with the same routing: two functions on the full
  // encoder and core adapter, one on the depth-6 prefix and its own adapter.
  const root = await mkdtemp(join(tmpdir(), "semantscript-stub-compare-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  await createFixtureArtifact(root, {
    extraEncoders: [
      { ref: "encoder.fixture.depth-006", path: "depth-006.onnx" },
    ],
    extraAdapters: [{ ref: "adapter.fixture.refund" }],
    extraFunctions: [
      { id: siblingId, headRef: "head.sibling.value" },
      {
        id: routedId,
        headRef: "head.routed.value",
        adapterRef: "adapter.fixture.refund",
        encoderRef: "encoder.fixture.depth-006",
      },
    ],
  });

  const scenario = (handle, [first, sibling, routed]) => {
    const scoped = runtime.withSemaScope(() => {
      runtime.__sema.call(first, { facts });
      runtime.__sema.call(sibling, { facts: { b: 2, a: 1 } });
      runtime.__sema.call(routed, { facts });
      runtime.__sema.call(first, { facts: { a: 3, b: 4 } });
      return runtime.semaScopePasses();
    });
    const stage = handle.callStage([
      { functionId: first, inputs: { facts } },
      { functionId: routed, inputs: { facts } },
      { functionId: sibling, inputs: { facts } },
    ]).passes;
    const plan = runtime.executeSemaPlan(
      {
        stages: [
          { index: 0, functionIds: [first, sibling] },
          { index: 1, functionIds: [routed] },
        ],
        dependencies: [
          {
            producerFunctionId: first,
            consumerFunctionId: routed,
            consumerInput: "facts",
          },
        ],
      },
      (stageInfo, results) =>
        stageInfo.index === 0
          ? { [first]: { facts }, [sibling]: { facts } }
          : { [routed]: { facts: { a: results.size, b: 0 } } },
    );
    return {
      scoped,
      stage,
      plan: plan.stages,
      results: [...plan.results.values()],
    };
  };

  const fixtureHandle = await runtime.loadSemaArtifact(root);
  const real = scenario(fixtureHandle, [
    fixtureFunctionId,
    siblingId,
    routedId,
  ]);
  await runtime.closeSemaArtifact();

  const handle = await testing.loadSemaStubArtifact(bundle, {
    answers: {
      [ids.label]: "approve",
      [ids.flag]: true,
      [ids.routed]: { value: "deny", confidence: 0.95 },
    },
    fallbacks: fallbacks(),
  });
  t.after(() => handle.close());
  const stubbed = scenario(handle, [ids.label, ids.flag, ids.routed]);

  assert.deepEqual(real.scoped, { encoder: 3, adapter: 3, head: 4 });
  assert.deepEqual(stubbed.scoped, real.scoped);
  assert.deepEqual(stubbed.stage, real.stage);
  assert.deepEqual(stubbed.plan, real.plan);
  assert.deepEqual(stubbed.results, ["approve", true, "deny"]);
  assert.deepEqual(real.results, ["review", "review", "review"]);
});

test("answers are checked against the IR when the stub is created", async () => {
  const { testing } = await runtimeModules();
  const rejects = (answers, reason) =>
    assert.rejects(
      testing.createSemaStubArtifact(bundle, { answers }),
      (error) =>
        error instanceof testing.SemaStubError &&
        error.code === "SEMA_STUB_INVALID" &&
        error.reason === reason,
      reason,
    );
  await rejects({ [`nf_${"f".repeat(64)}`]: "deny" }, "unknown-function");
  await rejects({ facts: "deny" }, "unknown-function");
  await rejects([[() => "no sema here", "deny"]], "unresolved-function");
  await rejects({ [ids.label]: "maybe" }, "invalid-value");
  await rejects({ [ids.score]: "2" }, "invalid-value");
  await rejects(
    { [ids.flag]: { value: true, confidence: 0.5 } },
    "invalid-confidence",
  );
  await rejects(
    { [ids.label]: { value: "deny", confidence: 1 / 3 } },
    "invalid-confidence",
  );
  await rejects(
    { [ids.label]: { value: "deny", confidence: 1.5 } },
    "invalid-confidence",
  );
  await rejects(
    { [ids.triage]: { value: { priority: "low" } } },
    "invalid-value",
  );
  await rejects(
    {
      [ids.triage]: { value: { priority: "low", needsHuman: true, extra: 1 } },
    },
    "invalid-value",
  );
  await rejects(
    { [ids.triage]: { value: byHand, confidence: { other: 0.9 } } },
    "invalid-confidence",
  );
  await rejects(
    {
      [ids.label]: {
        byInput: [{ inputs: { facts: { a: "x", b: 1 } }, value: "deny" }],
      },
    },
    "invalid-value",
  );
  await rejects({ [ids.label]: { compute: "deny" } }, "invalid-value");
  await rejects({ [ids.label]: { value: "deny", note: "x" } }, "invalid-value");
  await assert.rejects(
    testing.createSemaStubArtifact({ kind: "other" }, { answers: {} }),
    (error) => error.reason === "invalid-bundle",
  );
});

test("calls without an answer, bad computed answers and unregistered stubs fail with typed errors", async (t) => {
  const { runtime, testing } = await runtimeModules();
  const handle = await testing.loadSemaStubArtifact(bundle, {
    answers: { [ids.label]: { compute: () => "sometimes" } },
    fallbacks: fallbacks(),
  });
  try {
    assert.throws(
      () => runtime.__sema.call(ids.flag, { facts }),
      (error) =>
        error instanceof testing.SemaStubError && error.reason === "unanswered",
    );
    assert.throws(
      () => runtime.__sema.call(ids.label, { facts }),
      (error) =>
        error instanceof testing.SemaStubError &&
        error.reason === "invalid-value",
    );
  } finally {
    await handle.close();
  }

  // A stub release whose answers are not registered in this process never loads.
  const directory = await mkdtemp(join(tmpdir(), "semantscript-stub-dir-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const stub = await testing.createSemaStubArtifact(bundle, {
    answers: {},
    directory,
  });
  await stub.dispose();
  assert.ok(
    existsSync(join(directory, "current.json")),
    "a given directory is kept",
  );
  await assert.rejects(
    runtime.loadSemaArtifact(directory, { fallbacks: fallbacks() }),
    (error) =>
      error instanceof testing.SemaStubError && error.reason === "unregistered",
  );
});

test("a bundle path loads like the bundle object", async (t) => {
  const { runtime, testing } = await runtimeModules();
  const directory = await mkdtemp(join(tmpdir(), "semantscript-stub-path-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const path = join(directory, "semantscript.ir.v1.json");
  await writeFile(path, JSON.stringify(bundle));
  const handle = await testing.loadSemaStubArtifact(path, {
    answers: { [ids.label]: "review" },
    fallbacks: fallbacks(),
  });
  t.after(() => handle.close());
  assert.equal(handle.call(ids.label, { facts }), "review");
  assert.equal(runtime.__sema.call(ids.label, { facts }), "review");
  await assert.rejects(
    testing.createSemaStubArtifact(join(directory, "missing.json"), {
      answers: {},
    }),
    (error) => error.reason === "invalid-bundle",
  );
});
