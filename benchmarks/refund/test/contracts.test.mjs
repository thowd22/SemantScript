import assert from "node:assert/strict";
import test from "node:test";

import { semanticJsonSha256 } from "@semantscript/compiler";

import {
  BenchmarkContractError,
  HUMAN_ATTESTATION_DECLARATION,
  REFUND_FUNCTION_ID,
  REFUND_FUNCTION_SEMANTIC_SHA256,
  REFUND_TASK_SPEC_SHA256,
  auditDatasetSeparation,
  deriveRefundArtifactTrainingKeySha256,
  sealPredictionSet,
  sealRefundDataset,
  sealTrainingLedger,
  validatePredictionSet,
  validateRefundDataset,
  validateTrainingLedger,
} from "../dist/index.js";
import { clone, makeDataset, makeLedger, makePredictionSet } from "./fixtures.mjs";

test("seals closed dataset, ledger, and prediction contracts with semantic digests", () => {
  const dataset = makeDataset();
  const ledger = makeLedger();
  const predictions = makePredictionSet(dataset);

  assert.equal(validateRefundDataset(dataset).payloadSha256, dataset.payloadSha256);
  assert.equal(validateTrainingLedger(ledger).payloadSha256, ledger.payloadSha256);
  assert.equal(validatePredictionSet(predictions).payloadSha256, predictions.payloadSha256);
  assert.ok(Object.isFrozen(dataset));
  assert.ok(Object.isFrozen(dataset.cases[0].inputs));
  assert.ok(Object.isFrozen(predictions.system.model));
  assert.equal(dataset.function.id, REFUND_FUNCTION_ID);
  assert.equal(dataset.function.semanticSha256, REFUND_FUNCTION_SEMANTIC_SHA256);
  assert.equal(predictions.system.taskSpecSha256, REFUND_TASK_SPEC_SHA256);
  assert.deepEqual(auditDatasetSeparation(dataset, ledger), {
    datasetSha256: dataset.payloadSha256,
    trainingLedgerSha256: ledger.payloadSha256,
    evaluationCaseCount: 4,
    trainingInputCount: 5,
    overlapCount: 0,
  });
});

test("artifact training key is closed and release-evidence sensitive", () => {
  const sources = makeLedger().sources;
  const first = deriveRefundArtifactTrainingKeySha256(sources);
  assert.match(first, /^[a-f0-9]{64}$/u);
  assert.equal(first, deriveRefundArtifactTrainingKeySha256({ ...sources }));
  assert.notEqual(
    first,
    deriveRefundArtifactTrainingKeySha256({
      ...sources,
      releaseVerificationPayloadSha256: "f".repeat(64),
    }),
  );
  assert.notEqual(
    first,
    deriveRefundArtifactTrainingKeySha256({
      ...sources,
      adversarialDatasetSha256: null,
    }),
  );
  assert.throws(
    () => deriveRefundArtifactTrainingKeySha256({ ...sources, extra: true }),
    /must contain exactly/,
  );
  assert.equal(
    deriveRefundArtifactTrainingKeySha256({
      baseDatasetSha256: "a".repeat(64),
      adversarialDatasetSha256: null,
      releaseVerificationPayloadSha256: "c".repeat(64),
      releaseVerificationAttestationSha256: "d".repeat(64),
    }),
    "04f065ef6bdd717115ec6179da818f42c73d17ca37b24ad30e33ad885ac212b0",
  );
});

test("dataset rejects unknown fields, stale digests, malformed dates, and unsorted cases", () => {
  const dataset = makeDataset();
  const unknown = clone(dataset);
  unknown.note = "not closed";
  assert.throws(() => validateRefundDataset(unknown), /must contain exactly/);

  const stale = clone(dataset);
  stale.cases[0].inputs.order.total = 130;
  assert.throws(() => validateRefundDataset(stale), /inputSha256.*does not match/);

  const invalidDate = clone(dataset);
  invalidDate.createdAt = "2026-02-30T12:00:00Z";
  assert.throws(() => validateRefundDataset(invalidDate), /real calendar timestamp/);

  const unsorted = clone(dataset);
  unsorted.cases.reverse();
  assert.throws(() => validateRefundDataset(unsorted), /strictly ascending and unique/);
});

test("dataset requires a complete explicit non-teacher human attestation", () => {
  const dataset = makeDataset();
  const incomplete = clone(dataset);
  incomplete.humanAttestation.caseIds = ["case-01"];
  incomplete.payloadSha256 = semanticJsonSha256(withoutDigest(incomplete));
  assert.throws(() => validateRefundDataset(incomplete), /every and only human-authored/);

  const alteredDeclaration = clone(dataset);
  alteredDeclaration.humanAttestation.declaration = "AI generated";
  alteredDeclaration.payloadSha256 = semanticJsonSha256(withoutDigest(alteredDeclaration));
  assert.throws(() => validateRefundDataset(alteredDeclaration), /must be.*listed cases/);

  const noHuman = clone(dataset);
  for (const entry of noHuman.cases) entry.origin = "other-held-out";
  noHuman.humanAttestation.caseIds = [];
  noHuman.humanAttestation.declaration = HUMAN_ATTESTATION_DECLARATION;
  noHuman.payloadSha256 = semanticJsonSha256(withoutDigest(noHuman));
  assert.throws(() => validateRefundDataset(noHuman), /at least one human-authored/);
});

test("ledger partitions are fixed and sorted while allowing real lifecycle reuse", () => {
  const ledger = makeLedger();
  const reordered = clone(ledger);
  reordered.partitions.reverse();
  reordered.payloadSha256 = semanticJsonSha256(withoutDigest(reordered));
  assert.throws(() => validateTrainingLedger(reordered), /must be "examples"/);

  const crossDuplicate = clone(ledger);
  crossDuplicate.partitions[3].inputSha256s = [crossDuplicate.partitions[1].inputSha256s[0]];
  crossDuplicate.payloadSha256 = semanticJsonSha256(withoutDigest(crossDuplicate));
  assert.doesNotThrow(() => validateTrainingLedger(crossDuplicate));
  assert.equal(auditDatasetSeparation(makeDataset(), crossDuplicate).trainingInputCount, 4);

  assert.throws(
    () =>
      sealTrainingLedger({
        ...withoutDigest(ledger),
        partitions: ledger.partitions.map((partition, index) =>
          index === 0
            ? { ...partition, inputSha256s: ["f".repeat(64), "0".repeat(64)] }
            : partition,
        ),
      }),
    /strictly ascending/,
  );
});

test("leakage audit rejects overlap with every lifecycle partition", () => {
  const dataset = makeDataset();
  for (const partition of ["examples", "synthetic", "adversarial", "calibration", "verification"]) {
    const ledger = makeLedger({
      [partition]: { inputSha256s: [dataset.cases[0].inputSha256] },
    });
    assert.throws(
      () => auditDatasetSeparation(dataset, ledger),
      /overlap the training lifecycle ledger/,
      partition,
    );
  }

  const wrongFunction = clone(makeLedger());
  wrongFunction.function.semanticSha256 = "9".repeat(64);
  wrongFunction.payloadSha256 = semanticJsonSha256(withoutDigest(wrongFunction));
  assert.throws(
    () => auditDatasetSeparation(dataset, wrongFunction),
    /semanticSha256.*must be/,
  );
});

test("dataset and ledger reject every non-canonical compiled refund binding", () => {
  for (const record of [clone(makeDataset()), clone(makeLedger())]) {
    record.function.id = `nf_${"f".repeat(64)}`;
    record.payloadSha256 = semanticJsonSha256(withoutDigest(record));
    assert.throws(() =>
      record.kind.endsWith("dataset")
        ? validateRefundDataset(record)
        : validateTrainingLedger(record),
    /function\.id.*must be/);
  }
});

test("prediction contract requires normalized support order and stable argmax", () => {
  const dataset = makeDataset();
  const prediction = makePredictionSet(dataset);

  const wrongOrder = clone(prediction);
  wrongOrder.predictions[0].distribution.reverse();
  wrongOrder.payloadSha256 = semanticJsonSha256(withoutDigest(wrongOrder));
  assert.throws(() => validatePredictionSet(wrongOrder), /must be "approve"/);

  const notNormalized = clone(prediction);
  notNormalized.predictions[0].distribution[0].probability = 0.7;
  notNormalized.payloadSha256 = semanticJsonSha256(withoutDigest(notNormalized));
  assert.throws(() => validatePredictionSet(notNormalized), /sum to 1/);

  const wrongArgmax = clone(prediction);
  wrongArgmax.predictions[3].value = "review";
  wrongArgmax.payloadSha256 = semanticJsonSha256(withoutDigest(wrongArgmax));
  assert.throws(() => validatePredictionSet(wrongArgmax), /stable support-order argmax/);

  const tieUsesFirst = sealPredictionSet({
    ...withoutDigest(prediction),
    predictions: prediction.predictions.map((entry, index) =>
      index === 3 ? { ...entry, value: "deny" } : entry,
    ),
  });
  assert.equal(tieUsesFirst.predictions[3].value, "deny");
});

test("prediction publication boundary fixes task, model, adapter, and training identities", () => {
  const dataset = makeDataset();
  const semantscript = clone(makePredictionSet(dataset));
  semantscript.system.model.version = "unreviewed-model";
  semantscript.payloadSha256 = semanticJsonSha256(withoutDigest(semantscript));
  assert.throws(() => validatePredictionSet(semantscript), /model\.version.*must be/);

  const baseline = clone(makePredictionSet(dataset, "ollama-1b"));
  baseline.system.trainingEvidence = {
    trainingLedgerSha256: "a".repeat(64),
    artifactTrainingDatasetSha256: "d".repeat(64),
    artifactTrainingKeySha256: "e".repeat(64),
    releaseVerificationPayloadSha256: "b".repeat(64),
    releaseVerificationAttestationSha256: "c".repeat(64),
  };
  baseline.payloadSha256 = semanticJsonSha256(withoutDigest(baseline));
  assert.throws(() => validatePredictionSet(baseline), /trainingEvidence.*must be null/);

  const wrongTask = clone(makePredictionSet(dataset, "structured-api"));
  wrongTask.system.taskSpecSha256 = "f".repeat(64);
  wrongTask.payloadSha256 = semanticJsonSha256(withoutDigest(wrongTask));
  assert.throws(() => validatePredictionSet(wrongTask), /taskSpecSha256.*must be/);

  const laya = clone(makePredictionSet(dataset, "laya"));
  laya.system.model.artifactSha256 = "f".repeat(64);
  laya.payloadSha256 = semanticJsonSha256(withoutDigest(laya));
  assert.throws(() => validatePredictionSet(laya), /artifactSha256.*must be/);
});

test("prediction publication requires closed role-specific execution backend evidence", () => {
  const dataset = makeDataset();

  const omitted = clone(makePredictionSet(dataset, "semantscript"));
  delete omitted.system.executionBackend;
  omitted.payloadSha256 = semanticJsonSha256(withoutDigest(omitted));
  assert.throws(() => validatePredictionSet(omitted), /must contain exactly/);

  const wrongRole = clone(makePredictionSet(dataset, "laya"));
  wrongRole.system.executionBackend = {
    kind: "semantscript-node",
    runtime: "onnxruntime-node",
    device: "cpu",
  };
  wrongRole.payloadSha256 = semanticJsonSha256(withoutDigest(wrongRole));
  assert.throws(() => validatePredictionSet(wrongRole), /executionBackend\.kind.*must be "laya"/);

  const inconsistentOffload = clone(makePredictionSet(dataset, "ollama-7b"));
  inconsistentOffload.system.executionBackend.modelGpuBytes = 512;
  inconsistentOffload.payloadSha256 = semanticJsonSha256(
    withoutDigest(inconsistentOffload),
  );
  assert.throws(
    () => validatePredictionSet(inconsistentOffload),
    /modelCpuBytes plus modelGpuBytes must equal modelTotalBytes/,
  );

  const falsePlacement = clone(makePredictionSet(dataset, "ollama-1b"));
  falsePlacement.system.executionBackend.placement = "cpu";
  falsePlacement.payloadSha256 = semanticJsonSha256(withoutDigest(falsePlacement));
  assert.throws(
    () => validatePredictionSet(falsePlacement),
    /executionBackend\.placement.*must be "gpu"/,
  );
});

test("refund numeric inputs reject negative zero before digest identity can collapse", () => {
  const positiveZero = clone(makeDataset());
  positiveZero.cases[0].inputs.order.total = 0;
  positiveZero.cases[0].inputSha256 = semanticJsonSha256(positiveZero.cases[0].inputs);
  positiveZero.payloadSha256 = semanticJsonSha256(withoutDigest(positiveZero));
  assert.doesNotThrow(() => validateRefundDataset(positiveZero));

  for (const field of ["priorRefunds", "ageDays", "total"]) {
    const negativeZero = clone(positiveZero);
    if (field === "priorRefunds") {
      negativeZero.cases[0].inputs.customer.priorRefunds = -0;
    } else {
      negativeZero.cases[0].inputs.order[field] = -0;
    }
    negativeZero.cases[0].inputSha256 = semanticJsonSha256(
      negativeZero.cases[0].inputs,
    );
    negativeZero.payloadSha256 = semanticJsonSha256(withoutDigest(negativeZero));
    assert.throws(
      () => validateRefundDataset(negativeZero),
      new RegExp(`${field}.*negative zero`),
    );
  }
});

test("contract cardinality limits accept their boundary", { timeout: 30_000 }, () => {
  const dataset = makeMaximumDataset();
  assert.equal(dataset.cases.length, 20_000);
  assert.equal(dataset.humanAttestation.caseIds.length, 20_000);

  const ledger = makeLedger();
  const ledgerPayload = withoutDigest(ledger);
  ledgerPayload.partitions = ledgerPayload.partitions.map((partition, index) =>
    index === 0
      ? {
          ...partition,
          inputSha256s: Array.from({ length: 50_000 }, (_, digestIndex) =>
            digestIndex.toString(16).padStart(64, "0"),
          ),
        }
      : partition,
  );
  assert.equal(sealTrainingLedger(ledgerPayload).partitions[0].inputSha256s.length, 50_000);

  const predictions = withoutDigest(clone(makePredictionSet(makeDataset())));
  predictions.environment.runtimeVersions = Array.from({ length: 64 }, (_, index) => ({
    name: `runtime-${String(index).padStart(2, "0")}`,
    version: "test-only",
  }));
  assert.equal(sealPredictionSet(predictions).environment.runtimeVersions.length, 64);
});

test("contract cardinality limits reject over-limit arrays before entries are read", () => {
  let invoked = false;
  const oversizedCases = oversizedSparseArray(20_001, () => {
    invoked = true;
  });
  const datasetPayload = withoutDigest(clone(makeDataset()));
  datasetPayload.cases = oversizedCases;
  assert.throws(() => sealRefundDataset(datasetPayload), /cases.*at most 20000/);
  assert.equal(invoked, false);

  const oversizedAttestation = withoutDigest(clone(makeDataset()));
  oversizedAttestation.humanAttestation.caseIds = oversizedSparseArray(20_001);
  assert.throws(
    () => sealRefundDataset(oversizedAttestation),
    /humanAttestation\.caseIds.*at most 20000/,
  );

  const oversizedLedger = withoutDigest(clone(makeLedger()));
  oversizedLedger.partitions[0].inputSha256s = oversizedSparseArray(50_001);
  assert.throws(
    () => sealTrainingLedger(oversizedLedger),
    /inputSha256s.*at most 50000/,
  );

  const oversizedPredictions = withoutDigest(clone(makePredictionSet(makeDataset())));
  oversizedPredictions.predictions = oversizedSparseArray(20_001);
  assert.throws(
    () => sealPredictionSet(oversizedPredictions),
    /predictions.*at most 20000/,
  );

  const oversizedRuntimeVersions = withoutDigest(clone(makePredictionSet(makeDataset())));
  oversizedRuntimeVersions.environment.runtimeVersions = oversizedSparseArray(65);
  assert.throws(
    () => sealPredictionSet(oversizedRuntimeVersions),
    /runtimeVersions.*at most 64/,
  );
});

test("nested string and safe-integer bounds accept boundary and reject excess", () => {
  const dataset = withoutDigest(clone(makeDataset()));
  dataset.humanAttestation.attestor = "a".repeat(500);
  dataset.cases[0].inputs.customer.priorRefunds = Number.MAX_SAFE_INTEGER;
  dataset.cases[0].inputSha256 = semanticJsonSha256(dataset.cases[0].inputs);
  assert.doesNotThrow(() => sealRefundDataset(dataset));

  const longAttestor = clone(dataset);
  longAttestor.humanAttestation.attestor = "a".repeat(501);
  assert.throws(() => sealRefundDataset(longAttestor), /at most 500/);

  const unsafeCount = clone(dataset);
  unsafeCount.cases[0].inputs.customer.priorRefunds = Number.MAX_SAFE_INTEGER + 1;
  unsafeCount.cases[0].inputSha256 = semanticJsonSha256(unsafeCount.cases[0].inputs);
  assert.throws(() => sealRefundDataset(unsafeCount), /non-negative safe integer/);
});

test("sealers reject accessors rather than invoking them", () => {
  const dataset = clone(makeDataset());
  let invoked = false;
  Object.defineProperty(dataset.cases[0], "expected", {
    enumerable: true,
    get() {
      invoked = true;
      return "approve";
    },
  });
  assert.throws(() => validateRefundDataset(dataset), BenchmarkContractError);
  assert.equal(invoked, false);
});

test("test fixtures remain explicitly non-production", () => {
  const dataset = makeDataset();
  assert.match(dataset.humanAttestation.attestor, /not-a-real-attestation/);
  assert.ok(dataset.cases.every((entry) => entry.id.startsWith("case-")));
});

function withoutDigest(value) {
  const payload = { ...value };
  delete payload.payloadSha256;
  return payload;
}

function makeMaximumDataset() {
  const base = withoutDigest(clone(makeDataset()));
  const cases = Array.from({ length: 20_000 }, (_, index) => {
    const inputs = {
      customer: { priorRefunds: index, tier: "standard" },
      order: { ageDays: 1, status: "paid", total: 1 },
    };
    return {
      id: `case-${String(index).padStart(5, "0")}`,
      inputs,
      inputSha256: semanticJsonSha256(inputs),
      expected: "approve",
      origin: "human-authored",
    };
  });
  base.cases = cases;
  base.humanAttestation.caseIds = cases.map((entry) => entry.id);
  return sealRefundDataset(base);
}

function oversizedSparseArray(length, onRead = () => {}) {
  const values = [];
  values.length = length;
  Object.defineProperty(values, "0", {
    enumerable: true,
    get: onRead,
  });
  return values;
}
