import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { URL } from "node:url";

import { inspectOnnxContainer } from "../dist/onnx-model.js";

const encoder = new Uint8Array(
  await readFile(new URL("fixtures/encoder.onnx", import.meta.url)),
);

test("inspects the committed ONNX fixture container", () => {
  const inspection = inspectOnnxContainer(encoder);
  assert.equal(inspection.defaultOpset, 17);
  assert.deepEqual([...inspection.operatorDomains], [""]);
});

test("rejects custom operator domains and external tensor data", () => {
  const customDomainImport = Uint8Array.of(
    0x42,
    0x0a,
    0x0a,
    0x06,
    ...Buffer.from("custom"),
    0x10,
    0x11,
  );
  const custom = new Uint8Array(encoder.length + customDomainImport.length);
  custom.set(encoder);
  custom.set(customDomainImport, encoder.length);
  assert.throws(() => inspectOnnxContainer(custom), /custom domain/);

  // ModelProto(opset_import={version:17}, graph={initializer:{external_data:{}}})
  const external = Uint8Array.of(0x42, 0x02, 0x10, 0x11, 0x3a, 0x04, 0x2a, 0x02, 0x6a, 0x00);
  assert.throws(() => inspectOnnxContainer(external), /external tensor data/);
});

test("rejects truncated and malformed protobuf fields", () => {
  assert.throws(() => inspectOnnxContainer(Uint8Array.of(0x42, 0x80)), /varint|length/);
  assert.throws(() => inspectOnnxContainer(Uint8Array.of(0x3a, 0x04, 0x08)), /exceeds|ends/);
});

test("rejects deployment-irrelevant training graphs and local functions", () => {
  const trainingInfo = new Uint8Array(encoder.length + 3);
  trainingInfo.set(encoder);
  trainingInfo.set(Uint8Array.of(0xa2, 0x01, 0x00), encoder.length);
  assert.throws(() => inspectOnnxContainer(trainingInfo), /training graphs/);

  const localFunction = new Uint8Array(encoder.length + 3);
  localFunction.set(encoder);
  localFunction.set(Uint8Array.of(0xca, 0x01, 0x00), encoder.length);
  assert.throws(() => inspectOnnxContainer(localFunction), /local function/);
});
