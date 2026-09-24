import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { URL } from "node:url";
import { TextEncoder } from "node:util";

import {
  SemaInferenceError,
  SemaInferenceInitializationError,
  SemaInferenceInputError,
  SemaInferenceTimeoutError,
  SemaUnknownFunctionError,
  createInferenceRuntime,
  parseInferenceResultPayload,
} from "../dist/inference-runtime.js";
import {
  MAXIMUM_SERIALIZED_FINITE_NUMBER_BYTES,
  maximumInferenceResponseBytes,
  stringifyInferenceResult,
} from "../dist/inference-protocol.js";

const canonicalInput = new TextEncoder().encode(
  '["semantscript-input",1,[["message",["string","hello"]]]]',
);

const encoderAbi = {
  inputs: [
    { name: "input_ids", dtype: "int64", shape: ["BATCH", "SEQUENCE"] },
    { name: "attention_mask", dtype: "int64", shape: ["BATCH", "SEQUENCE"] },
  ],
  outputs: [{ name: "sentence_embedding", dtype: "float32", shape: ["BATCH", 1] }],
};
const adapterAbi = {
  inputs: [{ name: "sentence_embedding", dtype: "float32", shape: ["BATCH", 1] }],
  outputs: [{ name: "function_embedding", dtype: "float32", shape: ["BATCH", 1] }],
};
const headAbi = {
  inputs: [{ name: "function_embedding", dtype: "float32", shape: ["BATCH", 1] }],
  outputs: [{ name: "logits", dtype: "float32", shape: ["BATCH", 3] }],
};
const fixedBatchEncoderAbi = {
  inputs: [
    { name: "input_ids", dtype: "int64", shape: [1, "SEQUENCE"] },
    { name: "attention_mask", dtype: "int64", shape: [1, "SEQUENCE"] },
  ],
  outputs: [{ name: "sentence_embedding", dtype: "float32", shape: [1, 1] }],
};
const fixedBatchAdapterAbi = {
  inputs: [{ name: "sentence_embedding", dtype: "float32", shape: [1, 1] }],
  outputs: [{ name: "function_embedding", dtype: "float32", shape: [1, 1] }],
};
const fixedBatchHeadAbi = {
  inputs: [{ name: "function_embedding", dtype: "float32", shape: [1, 1] }],
  outputs: [{ name: "logits", dtype: "float32", shape: [1, 3] }],
};
const symbolicHiddenAdapterAbi = {
  inputs: [{ name: "sentence_embedding", dtype: "float32", shape: ["BATCH", "HIDDEN"] }],
  outputs: [{ name: "function_embedding", dtype: "float32", shape: ["BATCH", "HIDDEN"] }],
};
const wideAdapterAbi = {
  inputs: [{ name: "sentence_embedding", dtype: "float32", shape: ["BATCH", 2] }],
  outputs: [{ name: "function_embedding", dtype: "float32", shape: ["BATCH", 2] }],
};
const fixedSequenceEncoderAbi = {
  inputs: [
    { name: "input_ids", dtype: "int64", shape: ["BATCH", 1] },
    { name: "attention_mask", dtype: "int64", shape: ["BATCH", 1] },
  ],
  outputs: [{ name: "sentence_embedding", dtype: "float32", shape: ["BATCH", 1] }],
};

function head(
  outputPath,
  parameterization,
  support,
  logits,
  temperature = 1,
  expectedValueMode = "none",
) {
  return { outputPath, parameterization, support, temperature, expectedValueMode, logits };
}

async function fixtureBytes(name) {
  return new Uint8Array(await readFile(new URL(`fixtures/${name}`, import.meta.url)));
}

function approximately(actual, expected, tolerance = 1e-12) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${String(actual)} is not within ${String(tolerance)} of ${String(expected)}`,
  );
}

async function fixturePlan({
  id,
  encoderSource = "encoder.onnx",
  encoderDescriptor = encoderAbi,
  adapterSource = "adapter.onnx",
  adapterDescriptor = adapterAbi,
  headSource = "head.onnx",
  headDescriptor = headAbi,
  diagnosticsRequired = false,
}) {
  return {
    kind: "onnx",
    tokenizerJson: await fixtureBytes("tokenizer.json"),
    encoderModel: await fixtureBytes(encoderSource),
    encoderAbi: encoderDescriptor,
    adapters: [
      { ref: "fixture-adapter", model: await fixtureBytes(adapterSource), abi: adapterDescriptor },
    ],
    functions: [
      {
        id,
        adapterRef: "fixture-adapter",
        diagnosticsRequired,
        heads: [
          {
            outputPath: [],
            parameterization: "categorical-softmax",
            support: ["approve", "deny", "review"],
            temperature: 1,
            expectedValueMode: "none",
            model: await fixtureBytes(headSource),
            abi: headDescriptor,
          },
        ],
      },
    ],
    maximumSequenceLength: 128,
  };
}

test("production worker runs the deterministic tokenizer and ONNX fixture chain", async (t) => {
  const runtime = await createInferenceRuntime({
    kind: "onnx",
    tokenizerJson: await fixtureBytes("tokenizer.json"),
    encoderModel: await fixtureBytes("encoder.onnx"),
    encoderAbi,
    adapters: [
      { ref: "fixture-adapter", model: await fixtureBytes("adapter.onnx"), abi: adapterAbi },
    ],
    functions: [
      {
        id: "fixture-function",
        adapterRef: "fixture-adapter",
        diagnosticsRequired: false,
        heads: [
          {
            outputPath: [],
            parameterization: "categorical-softmax",
            support: ["approve", "deny", "review"],
            temperature: 1,
            expectedValueMode: "none",
            model: await fixtureBytes("head.onnx"),
            abi: headAbi,
          },
        ],
      },
    ],
    maximumSequenceLength: 128,
  });
  t.after(async () => runtime.close());

  assert.equal(runtime.call("fixture-function", canonicalInput), "review");
});

test("production worker returns calibrated diagnostics for real ONNX logits", async (t) => {
  const runtime = await createInferenceRuntime(
    await fixturePlan({ id: "fixture-diagnostic", diagnosticsRequired: true }),
  );
  t.after(async () => runtime.close());

  const result = runtime.call("fixture-diagnostic", canonicalInput);
  assert.equal(result.value, "review");
  assert.equal(result.distribution.length, 3);
  approximately(
    result.distribution.reduce((sum, entry) => sum + entry.probability, 0),
    1,
  );
  assert.equal(result.confidence, result.distribution[2].probability);
  assert.equal(result.expectedValue, null);
});

test("response schemas snapshot support before staged model buffers transfer", async (t) => {
  const plan = await fixturePlan({ id: "response-schema-snapshot" });
  const support = plan.functions[0].heads[0].support;
  const pendingRuntime = createInferenceRuntime(plan);

  assert.equal(plan.encoderModel.byteLength, 0, "model ownership transfers during initialization");
  support[2] = "mutated-after-transfer";

  const runtime = await pendingRuntime;
  t.after(async () => runtime.close());
  assert.equal(runtime.call("response-schema-snapshot", canonicalInput), "review");
});

test("worker bridge maps scalar and flat-object v1 heads synchronously", async (t) => {
  const runtime = await createInferenceRuntime({
    kind: "test",
    functions: [
      {
        id: "categorical",
        diagnosticsRequired: false,
        heads: [head([], "categorical-softmax", ["deny", "review", "approve"], [-3, 0, 4])],
      },
      {
        id: "boolean",
        diagnosticsRequired: false,
        heads: [head([], "binary-sigmoid", [false, true], [-0.25], 2)],
      },
      {
        id: "object",
        diagnosticsRequired: false,
        heads: [
          head(["decision"], "categorical-softmax", ["deny", "approve"], [-2, 2]),
          head(["retry"], "binary-sigmoid", [false, true], [3]),
          head(["score"], "categorical-softmax", [0, 0.5, 1], [0, 5, 1]),
        ],
      },
      {
        id: "calibrated-tie-compact",
        diagnosticsRequired: false,
        heads: [
          head(
            [],
            "categorical-softmax",
            ["earlier", "raw-logit-later"],
            [0, Number.MIN_VALUE],
            Number.MAX_VALUE,
          ),
        ],
      },
    ],
  });
  t.after(async () => runtime.close());

  assert.deepEqual([...runtime.functionIds], [
    "categorical",
    "boolean",
    "object",
    "calibrated-tie-compact",
  ]);
  assert.equal(runtime.call("categorical", canonicalInput), "approve");
  assert.equal(runtime.call("boolean", canonicalInput), false);
  assert.deepEqual(runtime.call("object", canonicalInput), {
    decision: "approve",
    retry: true,
    score: 0.5,
  });
  assert.equal(runtime.call("calibrated-tie-compact", canonicalInput), "earlier");

  // A second request exercises sequence advancement and buffer reuse.
  assert.equal(runtime.call("categorical", canonicalInput), "approve");
});

test("worker bridge preserves exact finite numeric support values", async (t) => {
  const runtime = await createInferenceRuntime({
    kind: "test",
    functions: [
      {
        id: "scalar-negative-zero",
        diagnosticsRequired: false,
        heads: [head([], "categorical-softmax", [-0, 1], [2, -2])],
      },
      {
        id: "numeric-object",
        diagnosticsRequired: false,
        heads: [
          head(["negativeZero"], "categorical-softmax", [-0, 1], [2, -2]),
          head(["smallest"], "categorical-softmax", [Number.MIN_VALUE, 1], [2, -2]),
          head(["fraction"], "categorical-softmax", [0.30000000000000004, 1], [2, -2]),
        ],
      },
    ],
  });
  t.after(async () => runtime.close());

  assert.equal(Object.is(runtime.call("scalar-negative-zero", canonicalInput), -0), true);
  const object = runtime.call("numeric-object", canonicalInput);
  assert.equal(Object.is(object.negativeZero, -0), true);
  assert.equal(Object.is(object.smallest, Number.MIN_VALUE), true);
  assert.equal(Object.is(object.fraction, 0.30000000000000004), true);
});

test("diagnostics use stable calibrated probabilities, ties, and normalized entropy", async (t) => {
  const runtime = await createInferenceRuntime({
    kind: "test",
    functions: [
      {
        id: "binary-tie",
        diagnosticsRequired: true,
        heads: [head([], "binary-sigmoid", [false, true], [0])],
      },
      {
        id: "extreme-tie",
        diagnosticsRequired: true,
        heads: [
          head(
            [],
            "categorical-softmax",
            ["first", "second", "last"],
            [Number.MAX_VALUE, Number.MAX_VALUE, -Number.MAX_VALUE],
            Number.MIN_VALUE,
          ),
        ],
      },
      {
        id: "uniform",
        diagnosticsRequired: true,
        heads: [head([], "categorical-softmax", ["a", "b", "c"], [7, 7, 7])],
      },
      {
        id: "cold",
        diagnosticsRequired: true,
        heads: [head([], "categorical-softmax", ["a", "b"], [2, 0], 0.5)],
      },
      {
        id: "hot",
        diagnosticsRequired: true,
        heads: [head([], "categorical-softmax", ["a", "b"], [2, 0], 2)],
      },
      {
        id: "calibrated-tie-diagnostic",
        diagnosticsRequired: true,
        heads: [
          head(
            [],
            "categorical-softmax",
            ["earlier", "raw-logit-later"],
            [0, Number.MIN_VALUE],
            Number.MAX_VALUE,
          ),
        ],
      },
    ],
  });
  t.after(async () => runtime.close());

  const binary = runtime.call("binary-tie", canonicalInput);
  assert.equal(binary.value, false);
  assert.equal(binary.confidence, 0.5);
  assert.equal(binary.uncertainty, 1);
  assert.deepEqual(binary.distribution, [
    { value: false, probability: 0.5 },
    { value: true, probability: 0.5 },
  ]);
  assert.equal(binary.expectedValue, null);

  const extreme = runtime.call("extreme-tie", canonicalInput);
  assert.equal(extreme.value, "first");
  assert.equal(extreme.confidence, 0.5);
  assert.deepEqual(
    extreme.distribution.map(({ probability }) => probability),
    [0.5, 0.5, 0],
  );
  approximately(extreme.uncertainty, Math.log(2) / Math.log(3));

  const uniform = runtime.call("uniform", canonicalInput);
  assert.equal(uniform.value, "a");
  approximately(uniform.confidence, 1 / 3);
  assert.equal(uniform.uncertainty, 1);

  const cold = runtime.call("cold", canonicalInput);
  const hot = runtime.call("hot", canonicalInput);
  assert.equal(cold.value, "a");
  assert.equal(hot.value, "a");
  assert.ok(cold.confidence > hot.confidence);

  const calibratedTie = runtime.call("calibrated-tie-diagnostic", canonicalInput);
  assert.equal(calibratedTie.value, "earlier");
  assert.deepEqual(
    calibratedTie.distribution.map(({ probability }) => probability),
    [0.5, 0.5],
  );
});

test("diagnostics distinguish nominal and ordinal expected values", async (t) => {
  const runtime = await createInferenceRuntime({
    kind: "test",
    functions: [
      {
        id: "nominal-number",
        diagnosticsRequired: true,
        heads: [head([], "categorical-softmax", [10, 20], [0, 0])],
      },
      {
        id: "ordinal-string",
        diagnosticsRequired: true,
        heads: [
          head(
            [],
            "categorical-softmax",
            ["low", "medium", "high"],
            [0, 0, 0],
            1,
            "zero-based-rank",
          ),
        ],
      },
      {
        id: "ordinal-number",
        diagnosticsRequired: true,
        heads: [head([], "categorical-softmax", [-2, 0, 2], [0, 0, 0], 1, "numeric")],
      },
    ],
  });
  t.after(async () => runtime.close());

  assert.equal(runtime.call("nominal-number", canonicalInput).expectedValue, null);
  approximately(runtime.call("ordinal-string", canonicalInput).expectedValue, 1);
  approximately(runtime.call("ordinal-number", canonicalInput).expectedValue, 0);
});

test("flat diagnostics preserve odd fields, aggregates, and nested signed zero", async (t) => {
  const runtime = await createInferenceRuntime({
    kind: "test",
    functions: [
      {
        id: "odd-object",
        diagnosticsRequired: true,
        heads: [
          head([""], "categorical-softmax", [-0, 1], [2, -2]),
          head(["__proto__"], "binary-sigmoid", [false, true], [0]),
          head(
            ["fields"],
            "categorical-softmax",
            ["low", "high"],
            [0, 0],
            1,
            "zero-based-rank",
          ),
        ],
      },
    ],
  });
  t.after(async () => runtime.close());

  const result = runtime.call("odd-object", canonicalInput);
  assert.equal(Object.is(result.value[""], -0), true);
  assert.equal(Object.hasOwn(result.value, "__proto__"), true);
  assert.equal(result.value.__proto__, false);
  assert.equal(Object.is(result.fields[""].value, -0), true);
  assert.equal(Object.is(result.fields[""].distribution[0].value, -0), true);
  assert.equal(result.fields.fields.expectedValue, 0.5);
  assert.equal(result.minimumFieldConfidence, 0.5);
  assert.equal(result.maximumFieldUncertainty, 1);
});

test("diagnostic wire preserves nested negative zero and rejects malformed payloads", () => {
  const signedZeroPlan = {
    diagnosticsRequired: true,
    heads: [head([], "categorical-softmax", [-0, 1], [], 1, "numeric")],
  };
  const wire = stringifyInferenceResult({
    kind: "scalar",
    result: {
      value: -0,
      confidence: 1,
      uncertainty: 0,
      distribution: [
        { value: -0, probability: 1 },
        { value: 1, probability: 0 },
      ],
      expectedValue: -0,
    },
  });
  const decoded = parseInferenceResultPayload(wire, signedZeroPlan);
  assert.equal(Object.is(decoded.value, -0), true);
  assert.equal(Object.is(decoded.distribution[0].value, -0), true);
  assert.equal(Object.is(decoded.expectedValue, -0), true);

  const nominalPlan = {
    diagnosticsRequired: true,
    heads: [head([], "categorical-softmax", ["a", "b"], [])],
  };
  const plainPlan = {
    diagnosticsRequired: false,
    heads: [head([], "categorical-softmax", ["a", "b"], [])],
  };
  const flatPlan = {
    diagnosticsRequired: false,
    heads: [
      head(["x"], "categorical-softmax", ["a", "b"], []),
      head(["y"], "categorical-softmax", [false, true], []),
    ],
  };

  assert.throws(
    () => parseInferenceResultPayload('"old-unwrapped-value"', plainPlan),
    /must be an object/,
  );
  assert.throws(
    () =>
      parseInferenceResultPayload(
        '{"kind":"value","result":"a","extra":true}',
        plainPlan,
      ),
    /invalid fields/,
  );
  assert.throws(
    () =>
      parseInferenceResultPayload(
        '{"kind":"scalar","result":{"value":"a","confidence":0.6,"uncertainty":1,' +
          '"distribution":[{"value":"a","probability":0.6},{"value":"b","probability":0.6}],' +
          '"expectedValue":null}}',
        nominalPlan,
      ),
    /sum to one/,
  );
  assert.throws(
    () =>
      parseInferenceResultPayload(
        '{"kind":"scalar","result":{"value":"b","confidence":0.5,"uncertainty":1,' +
          '"distribution":[{"value":"a","probability":0.5},{"value":"b","probability":0.5}],' +
          '"expectedValue":null}}',
        nominalPlan,
      ),
    /top-1/,
  );
  assert.throws(
    () =>
      parseInferenceResultPayload(
        '{"kind":"scalar","result":{"value":"a","confidence":-0,"uncertainty":1,' +
          '"distribution":[{"value":"a","probability":-0},{"value":"b","probability":1}],' +
          '"expectedValue":null}}',
        nominalPlan,
      ),
    /number in \[0,1\]/,
  );
  assert.throws(
    () =>
      parseInferenceResultPayload(
        '{"kind":"object","result":{"value":{"x":"a"},"minimumFieldConfidence":0.5,' +
          '"maximumFieldUncertainty":0,"fields":{"x":{"value":"a","confidence":1,' +
          '"uncertainty":0,"distribution":[{"value":"a","probability":1},' +
          '{"value":"b","probability":0}],"expectedValue":null}}}}',
        {
          diagnosticsRequired: true,
          heads: [head(["x"], "categorical-softmax", ["a", "b"], [])],
        },
      ),
    /aggregates/,
  );

  assert.throws(
    () =>
      parseInferenceResultPayload(
        '{"kind":"value","kind":"scalar","result":"a"}',
        plainPlan,
      ),
    /duplicate object property "kind"/,
  );
  assert.throws(
    () => parseInferenceResultPayload('{"kind":"value","result":"outside"}', plainPlan),
    /outside the planned support/,
  );
  assert.throws(
    () => parseInferenceResultPayload('{"kind":"scalar","result":"a"}', plainPlan),
    /kind must be "value"/,
  );
  assert.throws(
    () => parseInferenceResultPayload('{"kind":"value","result":{"x":"a","z":true}}', flatPlan),
    /invalid fields/,
  );
  for (const distribution of [
    '[{"value":"a","probability":0.5},{"value":"a","probability":0.5}]',
    '[{"value":"b","probability":0.5},{"value":"a","probability":0.5}]',
  ]) {
    assert.throws(
      () =>
        parseInferenceResultPayload(
          '{"kind":"scalar","result":{"value":"a","confidence":0.5,"uncertainty":1,' +
            `"distribution":${distribution},"expectedValue":null}}`,
          nominalPlan,
        ),
      /does not match planned support order/,
    );
  }
  assert.throws(
    () =>
      parseInferenceResultPayload(
        '{"kind":"scalar","result":{"value":"a","confidence":0.5,"uncertainty":1,' +
          '"distribution":[{"value":"a","probability":0.5},{"value":"b","probability":0.5}],' +
          '"expectedValue":0}}',
        {
          diagnosticsRequired: true,
          heads: [
            head([], "categorical-softmax", ["a", "b"], [], 1, "zero-based-rank"),
          ],
        },
      ),
    /does not match its distribution/,
  );
  assert.throws(
    () =>
      parseInferenceResultPayload('{"kind":"value","result":"a"}', {
        diagnosticsRequired: false,
        heads: [head([], "categorical-softmax", ["a", "a"], [])],
      }),
    /support values must be unique/,
  );

  const signedZeros = parseInferenceResultPayload('{"kind":"value","result":-0}', {
    diagnosticsRequired: false,
    heads: [head([], "categorical-softmax", [-0, 0], [])],
  });
  assert.equal(Object.is(signedZeros, -0), true);
});

test("response schema support validation remains linear for large finite supports", () => {
  const support = Array.from({ length: 50_000 }, (_, index) => `value-${String(index)}`);
  const selected = support.at(-1);
  const decoded = parseInferenceResultPayload(
    `{"kind":"value","result":${JSON.stringify(selected)}}`,
    {
      diagnosticsRequired: false,
      heads: [head([], "categorical-softmax", support, [])],
    },
  );
  assert.equal(decoded, selected);
});

test("diagnostic response sizing is conservative and saturates at the caller cap", () => {
  const plan = {
    diagnosticsRequired: true,
    heads: [head(["emoji-💡"], "categorical-softmax", [-0, "line\nbreak"], [1, 0])],
  };
  const bound = maximumInferenceResponseBytes(plan, 10_000);
  const payload = stringifyInferenceResult({
    kind: "object",
    result: {
      value: { "emoji-💡": -0 },
      minimumFieldConfidence: 0.75,
      maximumFieldUncertainty: 0.5,
      fields: {
        "emoji-💡": {
          value: -0,
          confidence: 0.75,
          uncertainty: 0.5,
          distribution: [
            { value: -0, probability: 0.75 },
            { value: "line\nbreak", probability: 0.25 },
          ],
          expectedValue: null,
        },
      },
    },
  });
  assert.ok(new TextEncoder().encode(payload).byteLength <= bound);
  assert.equal(maximumInferenceResponseBytes(plan, 64), 65);

  const longestFiniteNumber = -0.0000063354006458703095;
  assert.equal(JSON.stringify(longestFiniteNumber).length, 25);
  assert.ok(MAXIMUM_SERIALIZED_FINITE_NUMBER_BYTES >= 25);

  const boundaryHead = head(
    [],
    "categorical-softmax",
    [longestFiniteNumber, "x"],
    [],
  );
  const compactBoundaryPlan = { diagnosticsRequired: false, heads: [boundaryHead] };
  const compactBoundaryPayload = stringifyInferenceResult({
    kind: "value",
    result: longestFiniteNumber,
  });
  assert.equal(
    new TextEncoder().encode(compactBoundaryPayload).byteLength,
    maximumInferenceResponseBytes(compactBoundaryPlan, 10_000),
  );

  const diagnosticBoundaryPlan = { diagnosticsRequired: true, heads: [boundaryHead] };
  const diagnosticBoundaryPayload = stringifyInferenceResult({
    kind: "scalar",
    result: {
      value: longestFiniteNumber,
      confidence: longestFiniteNumber,
      uncertainty: longestFiniteNumber,
      distribution: [
        { value: longestFiniteNumber, probability: longestFiniteNumber },
        { value: "x", probability: longestFiniteNumber },
      ],
      expectedValue: null,
    },
  });
  assert.equal(
    new TextEncoder().encode(diagnosticBoundaryPayload).byteLength,
    maximumInferenceResponseBytes(diagnosticBoundaryPlan, 10_000),
  );
});

test("production worker accepts mixed fixed and symbolic batch edges", async (t) => {
  const symbolicToFixed = await createInferenceRuntime({
    kind: "onnx",
    tokenizerJson: await fixtureBytes("tokenizer.json"),
    encoderModel: await fixtureBytes("encoder.onnx"),
    encoderAbi,
    adapters: [
      {
        ref: "fixed-adapter",
        model: await fixtureBytes("fixed-batch-adapter.onnx"),
        abi: fixedBatchAdapterAbi,
      },
    ],
    functions: [
      {
        id: "symbolic-to-fixed",
        adapterRef: "fixed-adapter",
        diagnosticsRequired: false,
        heads: [
          {
            outputPath: [],
            parameterization: "categorical-softmax",
            support: ["approve", "deny", "review"],
            temperature: 1,
            expectedValueMode: "none",
            model: await fixtureBytes("head.onnx"),
            abi: headAbi,
          },
        ],
      },
    ],
    maximumSequenceLength: 128,
  });
  t.after(async () => symbolicToFixed.close());

  const fixedToSymbolic = await createInferenceRuntime({
    kind: "onnx",
    tokenizerJson: await fixtureBytes("tokenizer.json"),
    encoderModel: await fixtureBytes("fixed-batch-encoder.onnx"),
    encoderAbi: fixedBatchEncoderAbi,
    adapters: [
      { ref: "symbolic-adapter", model: await fixtureBytes("adapter.onnx"), abi: adapterAbi },
    ],
    functions: [
      {
        id: "fixed-to-symbolic",
        adapterRef: "symbolic-adapter",
        diagnosticsRequired: false,
        heads: [
          {
            outputPath: [],
            parameterization: "categorical-softmax",
            support: ["approve", "deny", "review"],
            temperature: 1,
            expectedValueMode: "none",
            model: await fixtureBytes("fixed-batch-head.onnx"),
            abi: fixedBatchHeadAbi,
          },
        ],
      },
    ],
    maximumSequenceLength: 128,
  });
  t.after(async () => fixedToSymbolic.close());

  assert.equal(symbolicToFixed.call("symbolic-to-fixed", canonicalInput), "review");
  assert.equal(fixedToSymbolic.call("fixed-to-symbolic", canonicalInput), "review");
});

test("production worker accepts symbolic hidden edges adjacent to fixed widths", async (t) => {
  const runtime = await createInferenceRuntime(
    await fixturePlan({
      id: "mixed-hidden",
      adapterSource: "symbolic-hidden-adapter.onnx",
      adapterDescriptor: symbolicHiddenAdapterAbi,
    }),
  );
  t.after(async () => runtime.close());

  assert.equal(runtime.call("mixed-hidden", canonicalInput), "review");
});

test("production worker rejects unequal fixed hidden widths and fixed sequence dimensions", async () => {
  await assert.rejects(
    createInferenceRuntime(
      await fixturePlan({
        id: "unequal-hidden",
        adapterSource: "wide-adapter.onnx",
        adapterDescriptor: wideAdapterAbi,
      }),
    ),
    (error) =>
      error instanceof SemaInferenceInitializationError &&
      /encoder to adapter.*incompatible/.test(error.message),
  );

  await assert.rejects(
    createInferenceRuntime(
      await fixturePlan({
        id: "fixed-sequence",
        encoderDescriptor: fixedSequenceEncoderAbi,
      }),
    ),
    (error) =>
      error instanceof SemaInferenceInitializationError &&
      /encoder tensor metadata does not match/.test(error.message),
  );
});

test("worker bridge reports typed input, id, and backend failures", async (t) => {
  const runtime = await createInferenceRuntime({
    kind: "test",
    functions: [
      {
        id: "healthy",
        diagnosticsRequired: false,
        heads: [head([], "categorical-softmax", ["no", "yes"], [0, 1])],
      },
      {
        id: "failure",
        diagnosticsRequired: false,
        error: "deterministic backend failure",
        heads: [head([], "categorical-softmax", ["no", "yes"], [0, 1])],
      },
    ],
  });
  t.after(async () => runtime.close());

  assert.throws(
    () => runtime.call("missing", canonicalInput),
    (error) => error instanceof SemaUnknownFunctionError && error.functionId === "missing",
  );
  assert.throws(
    () => runtime.call("healthy", new Uint8Array()),
    (error) => error instanceof SemaInferenceInputError && error.code === "invalid-input",
  );
  assert.throws(
    () => runtime.call("healthy", Uint8Array.of(0xff)),
    (error) => error instanceof SemaInferenceError && error.code === "invalid-input",
  );
  assert.throws(
    () => runtime.call("failure", canonicalInput),
    (error) =>
      error instanceof SemaInferenceError &&
      error.code === "backend" &&
      /deterministic backend failure/.test(error.message),
  );

  // Ordinary invocation failures do not poison the worker.
  assert.equal(runtime.call("healthy", canonicalInput), "yes");
});

test("inference timeout poisons the synchronous facade instead of accepting a stale response", async (t) => {
  const runtime = await createInferenceRuntime(
    {
      kind: "test",
      functions: [
        {
          id: "slow",
          diagnosticsRequired: false,
          delayMilliseconds: 100,
          heads: [head([], "binary-sigmoid", [false, true], [1])],
        },
      ],
    },
    { inferenceTimeoutMilliseconds: 10 },
  );
  t.after(async () => runtime.close());

  assert.throws(
    () => runtime.call("slow", canonicalInput),
    (error) => error instanceof SemaInferenceTimeoutError && error.code === "timeout",
  );
  assert.throws(
    () => runtime.call("slow", canonicalInput),
    (error) => error instanceof SemaInferenceError && error.code === "worker-failed",
  );
});

test("worker initialization errors and timeouts reject create", async () => {
  await assert.rejects(
    createInferenceRuntime({
      kind: "test",
      initializationError: "fixture initialization failed",
      functions: [],
    }),
    (error) =>
      error instanceof SemaInferenceInitializationError &&
      /fixture initialization failed/.test(error.message),
  );

  await assert.rejects(
    createInferenceRuntime(
      {
        kind: "test",
        initializationDelayMilliseconds: 100,
        functions: [],
      },
      { initializationTimeoutMilliseconds: 10 },
    ),
    (error) => error instanceof SemaInferenceTimeoutError && error.code === "timeout",
  );
});

test("a stage shares one encoder pass across functions with identical inputs", async (t) => {
  const plan = await fixturePlan({ id: "stage-a" });
  plan.functions.push({ ...plan.functions[0], id: "stage-b", model: undefined });
  const runtime = await createInferenceRuntime({
    kind: "onnx",
    tokenizerJson: await fixtureBytes("tokenizer.json"),
    encoderModel: await fixtureBytes("encoder.onnx"),
    encoderAbi,
    adapters: [
      { ref: "fixture-adapter", model: await fixtureBytes("adapter.onnx"), abi: adapterAbi },
    ],
    functions: [
      {
        id: "stage-a",
        adapterRef: "fixture-adapter",
        diagnosticsRequired: true,
        heads: [
          {
            outputPath: [],
            parameterization: "categorical-softmax",
            support: ["approve", "deny", "review"],
            temperature: 1,
            expectedValueMode: "none",
            model: await fixtureBytes("head.onnx"),
            abi: headAbi,
          },
        ],
      },
      {
        id: "stage-b",
        adapterRef: "fixture-adapter",
        diagnosticsRequired: false,
        heads: [
          {
            outputPath: [],
            parameterization: "categorical-softmax",
            support: ["approve", "deny", "review"],
            temperature: 1,
            expectedValueMode: "none",
            model: await fixtureBytes("head.onnx"),
            abi: headAbi,
          },
        ],
      },
    ],
    maximumSequenceLength: 128,
  });
  t.after(async () => runtime.close());
  const otherInput = new TextEncoder().encode('["semantscript-input",1,[["message",["string","other"]]]]');

  const single = {
    a: runtime.call("stage-a", canonicalInput),
    b: runtime.call("stage-b", canonicalInput),
    bOther: runtime.call("stage-b", otherInput),
  };
  const fused = runtime.callStage([
    { functionId: "stage-a", canonicalInput },
    { functionId: "stage-b", canonicalInput },
  ]);
  assert.deepEqual(fused.passes, { encoder: 1, adapter: 1, head: 2 });
  assert.deepEqual(fused.results, [single.a, single.b]);

  const split = runtime.callStage([
    { functionId: "stage-a", canonicalInput },
    { functionId: "stage-b", canonicalInput: otherInput },
    { functionId: "stage-b", canonicalInput },
  ]);
  assert.deepEqual(split.passes, { encoder: 2, adapter: 2, head: 3 });
  assert.deepEqual(split.results, [single.a, single.bOther, single.b]);

  assert.throws(() => runtime.callStage([]), SemaInferenceInputError);
  assert.throws(
    () => runtime.callStage([{ functionId: "missing", canonicalInput }]),
    (error) => error instanceof SemaUnknownFunctionError && error.functionId === "missing",
  );
  assert.equal(runtime.call("stage-a", canonicalInput).value, single.a.value, "the facade stays usable after a rejected stage");
});

test("test backend stages report distinct inputs and keep per-function results", async (t) => {
  const runtime = await createInferenceRuntime({
    kind: "test",
    functions: [
      {
        id: "left",
        diagnosticsRequired: false,
        heads: [{ outputPath: [], parameterization: "categorical-softmax", support: ["x", "y"], temperature: 1, expectedValueMode: "none", logits: [0.2, 0.9] }],
      },
      {
        id: "right",
        diagnosticsRequired: false,
        heads: [{ outputPath: [], parameterization: "categorical-softmax", support: ["p", "q"], temperature: 1, expectedValueMode: "none", logits: [1.5, 0.1] }],
      },
    ],
  });
  t.after(async () => runtime.close());
  const outcome = runtime.callStage([
    { functionId: "left", canonicalInput },
    { functionId: "right", canonicalInput },
    { functionId: "right", canonicalInput: new TextEncoder().encode("other") },
  ]);
  assert.deepEqual(outcome.results, ["y", "p", "p"]);
  assert.deepEqual(outcome.passes, { encoder: 2, adapter: 2, head: 3 });
});
