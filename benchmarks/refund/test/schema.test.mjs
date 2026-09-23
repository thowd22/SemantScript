import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { semanticJsonSha256 } from "@semantscript/compiler";
import Ajv2020 from "ajv/dist/2020.js";

import { createBenchmarkResult } from "../dist/index.js";
import { REFUND_SYSTEM_PINS } from "../dist/policy.js";
import {
  makeAllPredictionSets,
  makeDataset,
  makeLedger,
  makePredictionSet,
} from "./fixtures.mjs";

const repositoryRoot = join(dirname(fileURLToPath(import.meta.url)), "../../..");
const schemaNames = [
  "refund-benchmark-dataset.v1.schema.json",
  "refund-training-input-ledger.v1.schema.json",
  "refund-benchmark-predictions.v1.schema.json",
  "refund-benchmark-result.v1.schema.json",
];

test("all refund benchmark schemas compile strictly and accept runtime-validated records", async () => {
  const ajv = new Ajv2020({ allErrors: true, allowUnionTypes: true, strict: true });
  const common = await readSchema("refund-benchmark-common.v1.schema.json");
  ajv.addSchema(common);
  const validators = new Map();
  for (const schemaName of schemaNames) {
    validators.set(schemaName, ajv.compile(await readSchema(schemaName)));
  }

  const dataset = makeDataset();
  const ledger = makeLedger();
  const predictions = makePredictionSet(dataset);
  const result = createBenchmarkResult(
    dataset,
    ledger,
    makeAllPredictionSets(dataset),
    "2026-09-23T13:00:00Z",
  );
  const records = [dataset, ledger, predictions, result];
  for (const [index, schemaName] of schemaNames.entries()) {
    const validate = validators.get(schemaName);
    assert.equal(validate(records[index]), true, JSON.stringify(validate.errors));
  }
});

test("schemas are closed at every provenance boundary", async () => {
  const ajv = new Ajv2020({ allErrors: true, allowUnionTypes: true, strict: true });
  ajv.addSchema(await readSchema("refund-benchmark-common.v1.schema.json"));
  const validate = ajv.compile(await readSchema("refund-benchmark-predictions.v1.schema.json"));
  const dataset = makeDataset();
  const predictions = structuredClone(makePredictionSet(dataset));
  predictions.system.model.unpinned = true;
  assert.equal(validate(predictions), false);
  assert.ok(validate.errors.some((error) => error.keyword === "additionalProperties"));
});

test("prediction/result schemas expose canonical task and role evidence boundaries", async () => {
  const ajv = new Ajv2020({ allErrors: true, allowUnionTypes: true, strict: true });
  ajv.addSchema(await readSchema("refund-benchmark-common.v1.schema.json"));
  const validatePredictions = ajv.compile(
    await readSchema("refund-benchmark-predictions.v1.schema.json"),
  );
  const validateResult = ajv.compile(
    await readSchema("refund-benchmark-result.v1.schema.json"),
  );
  const dataset = makeDataset();

  const wrongModel = structuredClone(makePredictionSet(dataset));
  wrongModel.system.model.version = "not-canonical";
  assert.equal(validatePredictions(wrongModel), false);

  const baselineEvidence = structuredClone(makePredictionSet(dataset, "ollama-1b"));
  baselineEvidence.system.trainingEvidence = {
    trainingLedgerSha256: "a".repeat(64),
    artifactTrainingDatasetSha256: "d".repeat(64),
    artifactTrainingKeySha256: "e".repeat(64),
    releaseVerificationPayloadSha256: "b".repeat(64),
    releaseVerificationAttestationSha256: "c".repeat(64),
  };
  assert.equal(validatePredictions(baselineEvidence), false);

  const result = structuredClone(
    createBenchmarkResult(
      dataset,
      makeLedger(),
      makeAllPredictionSets(dataset),
      "2026-09-23T13:00:00Z",
    ),
  );
  result.taskSpecSha256 = "f".repeat(64);
  assert.equal(validateResult(result), false);
});

test("prediction and result schemas reject rehashed role-identity substitutions", async () => {
  const ajv = new Ajv2020({ allErrors: true, allowUnionTypes: true, strict: true });
  ajv.addSchema(await readSchema("refund-benchmark-common.v1.schema.json"));
  const validatePredictions = ajv.compile(
    await readSchema("refund-benchmark-predictions.v1.schema.json"),
  );
  const validateResult = ajv.compile(
    await readSchema("refund-benchmark-result.v1.schema.json"),
  );
  const dataset = makeDataset();
  const roles = Object.keys(REFUND_SYSTEM_PINS);

  for (const role of roles) {
    const pin = REFUND_SYSTEM_PINS[role];
    const fixedIdentityMutations = [
      ["model", "provider", `${pin.model.provider}-attacker`],
      ["model", "name", `${pin.model.name}-attacker`],
      ["model", "version", `${pin.model.version}-attacker`],
      ["adapter", "name", `${pin.adapter.name}-attacker`],
      ["adapter", "version", `${pin.adapter.version}-attacker`],
    ];
    if (pin.model.revision !== null) {
      fixedIdentityMutations.push(["model", "revision", attackerDigest(pin.model.revision)]);
    }
    if (pin.model.artifactSha256 !== null) {
      fixedIdentityMutations.push([
        "model",
        "artifactSha256",
        attackerDigest(pin.model.artifactSha256),
      ]);
    }

    for (const [section, field, attackerValue] of fixedIdentityMutations) {
      const predictions = structuredClone(makePredictionSet(dataset, role));
      predictions.system[section][field] = attackerValue;
      rehash(predictions);
      assert.equal(
        validatePredictions(predictions),
        false,
        `prediction schema accepted rehashed ${role} ${section}.${field}`,
      );

      const result = structuredClone(
        createBenchmarkResult(
          dataset,
          makeLedger(),
          makeAllPredictionSets(dataset),
          "2026-09-23T13:00:00Z",
        ),
      );
      const system = result.systems.find((entry) => entry.role === role);
      system[section][field] = attackerValue;
      rehash(result);
      assert.equal(
        validateResult(result),
        false,
        `result schema accepted rehashed ${role} ${section}.${field}`,
      );
    }
  }
});

test("schemas retain SHA-shaped runtime identities for fields without publication pins", async () => {
  const ajv = new Ajv2020({ allErrors: true, allowUnionTypes: true, strict: true });
  ajv.addSchema(await readSchema("refund-benchmark-common.v1.schema.json"));
  const validatePredictions = ajv.compile(
    await readSchema("refund-benchmark-predictions.v1.schema.json"),
  );
  const dataset = makeDataset();

  const semantscript = structuredClone(makePredictionSet(dataset, "semantscript"));
  semantscript.system.model.revision = "e".repeat(64);
  semantscript.system.model.artifactSha256 = "f".repeat(64);
  rehash(semantscript);
  assert.equal(validatePredictions(semantscript), true, JSON.stringify(validatePredictions.errors));

  const structuredApi = structuredClone(makePredictionSet(dataset, "structured-api"));
  structuredApi.system.model.artifactSha256 = "e".repeat(64);
  rehash(structuredApi);
  assert.equal(
    validatePredictions(structuredApi),
    true,
    JSON.stringify(validatePredictions.errors),
  );
});

test("schema cardinality ceilings match runtime contract ceilings", async () => {
  const common = await readSchema("refund-benchmark-common.v1.schema.json");
  const datasetSchema = await readSchema("refund-benchmark-dataset.v1.schema.json");
  const ledgerSchema = await readSchema("refund-training-input-ledger.v1.schema.json");
  const predictionsSchema = await readSchema(
    "refund-benchmark-predictions.v1.schema.json",
  );
  const resultSchema = await readSchema("refund-benchmark-result.v1.schema.json");

  assert.equal(datasetSchema.properties.cases.maxItems, 20_000);
  assert.equal(common.$defs.humanAttestation.properties.caseIds.maxItems, 20_000);
  assert.equal(predictionsSchema.properties.predictions.maxItems, 20_000);
  assert.equal(ledgerSchema.$defs.inputDigests.maxItems, 50_000);
  assert.equal(common.$defs.environmentProvenance.properties.runtimeVersions.maxItems, 64);
  assert.equal(common.$defs.measurementProtocol.properties.warmupIterations.maximum, 10_000);
  assert.equal(resultSchema.properties.systems.maxItems, 5);
  assert.equal(common.$defs.goNoGo.properties.missingSystems.maxItems, 5);

  const ajv = new Ajv2020({ allErrors: true, allowUnionTypes: true, strict: true });
  ajv.addSchema(common);
  const validatePredictions = ajv.compile(predictionsSchema);
  const predictions = structuredClone(makePredictionSet(makeDataset()));
  predictions.environment.runtimeVersions = Array.from({ length: 64 }, (_, index) => ({
    name: `runtime-${String(index).padStart(2, "0")}`,
    version: "test-only",
  }));
  assert.equal(validatePredictions(predictions), true, JSON.stringify(validatePredictions.errors));
  predictions.environment.runtimeVersions.push({ name: "runtime-64", version: "test-only" });
  assert.equal(validatePredictions(predictions), false);
  assert.ok(
    validatePredictions.errors.some(
      (error) =>
        error.keyword === "maxItems" &&
        error.instancePath === "/environment/runtimeVersions",
    ),
  );
});

async function readSchema(name) {
  return JSON.parse(await readFile(join(repositoryRoot, "schemas", name), "utf8"));
}

function rehash(record) {
  const payload = { ...record };
  delete payload.payloadSha256;
  record.payloadSha256 = semanticJsonSha256(payload);
}

function attackerDigest(current) {
  return current === "f".repeat(64) ? "e".repeat(64) : "f".repeat(64);
}
