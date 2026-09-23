import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";

import { createSourceIrBundle } from "../dist/index.js";

const repositoryRoot = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const bundleSchema = JSON.parse(
  await readFile(join(repositoryRoot, "schemas", "ir-bundle.v1.schema.json"), "utf8"),
);
const neuralFunctionSchema = JSON.parse(
  await readFile(join(repositoryRoot, "schemas", "neural-function.v1.schema.json"), "utf8"),
);
const ajv = new Ajv2020({ allErrors: true, allowUnionTypes: true, strict: true });
addFormats(ajv);
ajv.addSchema(neuralFunctionSchema);
const validateBundleSchema = ajv.compile(bundleSchema);

test("creates valid empty and nonempty source IR bundle envelopes", () => {
  const empty = createSourceIrBundle([], { stages: [], dependencies: [] });
  assert.deepEqual(empty, {
    kind: "semantscript.ir-bundle",
    bundleVersion: 1,
    functions: [],
    executionPlan: { stages: [], dependencies: [] },
  });

  const first = sourceRecord("a", "src/a.sem.ts");
  const second = sourceRecord("b", "src/b.sem.ts");
  const consumer = sourceRecord("c", "src/c.sem.ts", ["left", "right"]);
  const bundle = createSourceIrBundle(
    [first, second, consumer],
    {
      stages: [
        { index: 0, functionIds: [first.id, second.id] },
        { index: 1, functionIds: [consumer.id] },
      ],
      dependencies: [
        dependency(first, consumer, "left"),
        dependency(second, consumer, "right"),
      ],
    },
  );

  assert.equal(bundle.kind, "semantscript.ir-bundle");
  assert.equal(bundle.bundleVersion, 1);
  assert.deepEqual(bundle.functions, [first, second, consumer]);
  assert.deepEqual(bundle.executionPlan.stages, [
    { index: 0, functionIds: [first.id, second.id] },
    { index: 1, functionIds: [consumer.id] },
  ]);
});

test("defensively copies bundle and execution-plan containers", () => {
  const first = sourceRecord("a", "src/a.sem.ts");
  const second = sourceRecord("b", "src/b.sem.ts");
  const consumer = sourceRecord("c", "src/c.sem.ts", ["choice"]);
  const functions = [first, second, consumer];
  const firstStageIds = [first.id, second.id];
  const secondStageIds = [consumer.id];
  const stages = [
    { index: 0, functionIds: firstStageIds },
    { index: 1, functionIds: secondStageIds },
  ];
  const edge = dependency(first, consumer, "choice");
  const dependencies = [edge];
  const bundle = createSourceIrBundle(functions, { stages, dependencies });

  functions.length = 0;
  firstStageIds.reverse();
  secondStageIds.push(first.id);
  stages[0].index = 9;
  stages.length = 0;
  edge.consumerInput = "tampered";
  dependencies.length = 0;

  assert.deepEqual(bundle.functions, [first, second, consumer]);
  assert.strictEqual(bundle.functions[0], first, "records themselves remain shared immutable inputs");
  assert.deepEqual(bundle.executionPlan.stages, [
    { index: 0, functionIds: [first.id, second.id] },
    { index: 1, functionIds: [consumer.id] },
  ]);
  assert.deepEqual(bundle.executionPlan.dependencies, [
    dependency(first, consumer, "choice"),
  ]);
});

test("rejects noncanonical functions, dependencies, and stages", () => {
  const first = sourceRecord("a", "src/a.sem.ts");
  const second = sourceRecord("b", "src/b.sem.ts");
  const consumer = sourceRecord("c", "src/c.sem.ts", ["left", "right"]);
  const functions = [first, second, consumer];
  const stages = [
    { index: 0, functionIds: [first.id, second.id] },
    { index: 1, functionIds: [consumer.id] },
  ];
  const canonicalDependencies = [
    dependency(first, consumer, "left"),
    dependency(second, consumer, "right"),
  ];

  assert.throws(
    () => createSourceIrBundle([second, first, consumer], { stages, dependencies: canonicalDependencies }),
    /canonical source order/,
  );
  assert.throws(
    () =>
      createSourceIrBundle(functions, {
        stages,
        dependencies: [...canonicalDependencies].reverse(),
      }),
    /dependencies must be in canonical order/,
  );
  assert.throws(
    () =>
      createSourceIrBundle(functions, {
        stages,
        dependencies: [canonicalDependencies[0], canonicalDependencies[0]],
      }),
    /duplicate labeled edges/,
  );
  assert.throws(
    () =>
      createSourceIrBundle(functions, {
        stages,
        dependencies: [dependency(first, consumer, "missing")],
      }),
    /unknown input/,
  );
  assert.throws(
    () =>
      createSourceIrBundle(functions, {
        stages: [
          { index: 0, functionIds: [second.id, first.id] },
          { index: 1, functionIds: [consumer.id] },
        ],
        dependencies: canonicalDependencies,
      }),
    /within a stage must be in canonical source order/,
  );
  assert.throws(
    () =>
      createSourceIrBundle(functions, {
        stages: [
          { index: 1, functionIds: [first.id, second.id] },
          { index: 2, functionIds: [consumer.id] },
        ],
        dependencies: canonicalDependencies,
      }),
    /stage indexes must be dense and start at zero/,
  );
});

test("rejects non-minimum and tampered dependency stage assignments", () => {
  const first = sourceRecord("a", "src/a.sem.ts");
  const second = sourceRecord("b", "src/b.sem.ts", ["first"]);
  const third = sourceRecord("c", "src/c.sem.ts", ["first"]);
  const independent = sourceRecord("d", "src/d.sem.ts");
  const functions = [first, second, third, independent];
  const dependencies = [
    dependency(first, second, "first"),
    dependency(first, third, "first"),
  ];
  const valid = createSourceIrBundle(functions, {
    stages: [
      { index: 0, functionIds: [first.id, independent.id] },
      { index: 1, functionIds: [second.id, third.id] },
    ],
    dependencies,
  });

  const late = copyExecutionPlan(valid.executionPlan);
  late.stages = [
    { index: 0, functionIds: [first.id, independent.id] },
    { index: 1, functionIds: [second.id] },
    { index: 2, functionIds: [third.id] },
  ];
  assert.throws(
    () => createSourceIrBundle(functions, late),
    /not assigned to its minimum dependency stage/,
  );

  const removedEdge = copyExecutionPlan(valid.executionPlan);
  removedEdge.dependencies = [dependency(first, third, "first")];
  assert.throws(
    () => createSourceIrBundle(functions, removedEdge),
    /not assigned to its minimum dependency stage/,
  );

  const sameStage = copyExecutionPlan(valid.executionPlan);
  sameStage.stages = [
    { index: 0, functionIds: [first.id, independent.id] },
    { index: 1, functionIds: [second.id, third.id] },
  ];
  sameStage.dependencies = [
    dependency(first, second, "first"),
    dependency(second, third, "first"),
  ];
  assert.throws(
    () => createSourceIrBundle(functions, sameStage),
    /must point from earlier to later stages/,
  );
});

test("Draft 2020-12 bundle schema accepts envelopes and rejects legacy root arrays", () => {
  const first = sourceRecord("a", "src/a.sem.ts");
  const bundle = createSourceIrBundle(
    [first],
    { stages: [{ index: 0, functionIds: [first.id] }], dependencies: [] },
  );
  const empty = createSourceIrBundle([], { stages: [], dependencies: [] });

  assert.equal(validateBundleSchema(empty), true, ajv.errorsText(validateBundleSchema.errors));
  assert.equal(validateBundleSchema(bundle), true, ajv.errorsText(validateBundleSchema.errors));
  assert.equal(validateBundleSchema([first]), false, "the v1 envelope is no longer a root array");
});

function dependency(producer, consumer, consumerInput) {
  return {
    producerFunctionId: producer.id,
    consumerFunctionId: consumer.id,
    consumerInput,
  };
}

function copyExecutionPlan(plan) {
  return {
    stages: plan.stages.map(({ index, functionIds }) => ({
      index,
      functionIds: [...functionIds],
    })),
    dependencies: plan.dependencies.map((edge) => ({ ...edge })),
  };
}

function sourceRecord(idDigit, sourcePath, inputNames = []) {
  const id = `nf_${idDigit.repeat(64)}`;
  return {
    kind: "semantscript.neural-function",
    irVersion: 1,
    stage: "source",
    id,
    semanticSha256: idDigit.repeat(64),
    source: {
      path: sourcePath,
      line: 1,
      column: 1,
      sourceSha256: "0".repeat(64),
    },
    definition: {
      template: [{ kind: "text", text: `fixture ${idDigit}` }],
      examples: [],
      constraints: [],
    },
    inputs: inputNames.map((name, index) => ({
      name,
      index,
      tsType: "string",
      type: { kind: "string" },
    })),
    output: {
      kind: "scalar",
      tsType: "boolean",
      head: { kind: "nominal", sourceKind: "boolean", support: [false, true] },
    },
    model: {
      encoder: "encoder.main",
      adapter: "adapter.application",
      heads: [{ outputPath: "", ref: `head.${idDigit}` }],
    },
    runtime: {
      resultMode: "value",
      confidenceThreshold: null,
      fallbackRef: null,
      synchronous: true,
    },
    trainingProvenance: { status: "pending" },
    verification: { status: "pending" },
  };
}
