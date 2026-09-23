import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const fixturesDirectory = dirname(fileURLToPath(import.meta.url));
const zeroSha = "0".repeat(64);

export const fixtureFunctionId = `nf_${"1".repeat(64)}`;

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function binary64(value) {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setFloat64(0, value, false);
  return Buffer.from(bytes).toString("hex");
}

function semanticNode(value) {
  if (value === null) return ["null"];
  if (typeof value === "boolean") return ["boolean", value];
  if (typeof value === "number") return ["number", binary64(value)];
  if (typeof value === "string") return ["string", value];
  if (Array.isArray(value)) return ["array", value.map(semanticNode)];

  const entries = Object.entries(value)
    .map(([key, entry]) => [key, semanticNode(entry)])
    .sort(([left], [right]) => Buffer.from(left).compare(Buffer.from(right)));
  return ["object", entries];
}

function semanticSha(value) {
  return sha256(Buffer.from(JSON.stringify(["semantscript-semantic-json", 1, semanticNode(value)])));
}

async function resource(release, source, path, ref, role, onnx) {
  const destination = join(release, path);
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(join(fixturesDirectory, source), destination);
  const bytes = await readFile(destination);
  const descriptor = {
    ref,
    role,
    format: onnx === undefined ? "tokenizer-json" : "onnx",
    formatVersion: 1,
    path,
    byteLength: (await stat(destination)).size,
    sha256: sha256(bytes),
  };
  return onnx === undefined
    ? { ...descriptor, maximumSequenceLength: 128 }
    : { ...descriptor, onnx };
}

export async function createFixtureArtifact(root, options = {}) {
  const staging = join(root, "staging");
  await mkdir(staging, { recursive: true });

  const resources = [
    await resource(
      staging,
      "tokenizer.json",
      "tokenizer/tokenizer.json",
      "tokenizer.main",
      "tokenizer",
    ),
    await resource(
      staging,
      options.encoderSource ?? "encoder.onnx",
      "models/encoder/model.onnx",
      "encoder.main",
      "encoder",
      {
        opset: 17,
        inputs: [
          { name: "input_ids", dtype: "int64", shape: ["BATCH", "SEQUENCE"] },
          { name: "attention_mask", dtype: "int64", shape: ["BATCH", "SEQUENCE"] },
        ],
        outputs: [
          { name: "sentence_embedding", dtype: "float32", shape: ["BATCH", 1] },
        ],
        externalData: false,
      },
    ),
    await resource(
      staging,
      options.adapterSource ?? "adapter.onnx",
      "models/adapters/application.onnx",
      "adapter.application",
      "adapter",
      {
        opset: 17,
        inputs: [
          { name: "sentence_embedding", dtype: "float32", shape: ["BATCH", 1] },
        ],
        outputs: [
          { name: "function_embedding", dtype: "float32", shape: ["BATCH", 1] },
        ],
        externalData: false,
      },
    ),
    await resource(
      staging,
      options.headSource ?? "head.onnx",
      `models/heads/${fixtureFunctionId}/head-000.onnx`,
      "head.fixture.value",
      "head",
      {
        opset: 17,
        inputs: [
          { name: "function_embedding", dtype: "float32", shape: ["BATCH", 1] },
        ],
        outputs: [{ name: "logits", dtype: "float32", shape: ["BATCH", 3] }],
        externalData: false,
      },
    ),
  ];

  const inputs = [
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
  const manifest = {
    kind: "semantscript.application-artifact",
    artifactVersion: 1,
    irVersion: 1,
    compatibility: {
      runtimeAbiVersion: 1,
      modelAbiVersion: 1,
      canonicalInput: "semantscript.canonical-input/v1",
      minimumRuntimeVersion: "0.0.0",
      requiredCapabilities: [],
    },
    application: { id: "runtime-fixture", version: "0.0.0" },
    build: {
      createdAt: "2026-09-22T00:00:00Z",
      compilerVersion: "0.0.0",
      trainerVersion: "0.0.0",
      sourceIrSha256: zeroSha,
    },
    resources,
    model: {
      tokenizerRef: "tokenizer.main",
      encoderRef: "encoder.main",
      adapterRef: "adapter.application",
    },
    functions: [
      {
        id: fixtureFunctionId,
        semanticSha256: "2".repeat(64),
        inputs,
        inputSchemaSha256: semanticSha(inputs),
        outputSchemaSha256: "3".repeat(64),
        adapterRef: "adapter.application",
        heads: [
          {
            outputPath: [],
            headRef: "head.fixture.value",
            type: { kind: "nominal-string", support: ["approve", "deny", "review"] },
            parameterization: "categorical-softmax",
            calibration: {
              method: "temperature-scaling",
              temperature: 1,
              ece: 0,
              brier: 0,
              sampleCount: 1,
              splitSha256: "4".repeat(64),
              eceBins: 2,
            },
            verification: { accuracy: 1, pairConsistency: 1 },
          },
        ],
        runtime: {
          resultMode: "value",
          confidenceThreshold: null,
          policy: "none",
          fallbackRef: null,
        },
        verification: {
          status: "passed",
          accuracy: 1,
          ece: 0,
          brier: 0,
          pairConsistency: 1,
          humanAuthoredCases: 1,
          exampleFailures: 0,
          constraintViolations: 0,
          typeErrors: 0,
        },
        trainingProvenance: {
          datasetSha256: "5".repeat(64),
          trainingKeySha256: "6".repeat(64),
          teacher: "fixture",
          baseModel: "fixture",
        },
      },
    ],
  };
  if (typeof options.transformManifest === "function") {
    await options.transformManifest(manifest);
  }
  const manifestBytes = Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`);
  const manifestSha256 = sha256(manifestBytes);
  await writeFile(join(staging, "manifest.json"), manifestBytes);

  const release = join(root, "releases", `sha256-${manifestSha256}`);
  await mkdir(dirname(release), { recursive: true });
  await rename(staging, release);
  const pointer = {
    kind: "semantscript.artifact-pointer",
    pointerVersion: 1,
    release: `releases/sha256-${manifestSha256}`,
    manifestSha256,
  };
  await writeFile(join(root, "current.json"), `${JSON.stringify(pointer, null, 2)}\n`);

  if (typeof options.mutate === "function") {
    await options.mutate({ manifest, manifestSha256, release, root });
  }
  return { manifest, manifestSha256, release, root };
}
