import process from "node:process";
import { pathToFileURL } from "node:url";

import { loadSemaArtifact } from "@semantscript/core";

const MAXIMUM_STDIN_BYTES = 1024 * 1024;
const REFUND_SUPPORT = ["approve", "deny", "review"];

export async function runRefundDiagnostic(artifactRoot, functionId, inputs) {
  if (typeof artifactRoot !== "string" || artifactRoot.length === 0) {
    throw new TypeError("artifactRoot must be a nonempty string");
  }
  if (typeof functionId !== "string" || !/^nf_[a-f0-9]{64}$/u.test(functionId)) {
    throw new TypeError("functionId must be a SemantScript neural-function ID");
  }
  if (inputs === null || typeof inputs !== "object" || Array.isArray(inputs)) {
    throw new TypeError("runtime inputs must be an object");
  }

  const handle = await loadSemaArtifact(artifactRoot);
  try {
    if (!handle.functionIds.has(functionId)) {
      throw new Error("compiled refund function is absent from the exported artifact");
    }
    const result = handle.call(functionId, inputs);
    validateRefundDiagnostic(result);
    return result;
  } finally {
    await handle.close();
  }
}

export function validateRefundDiagnostic(result) {
  if (result === null || typeof result !== "object" || Array.isArray(result)) {
    throw new TypeError("refund runtime must return a scalar diagnostic object");
  }
  const keys = Object.keys(result);
  const expectedKeys = ["value", "confidence", "uncertainty", "distribution", "expectedValue"];
  if (keys.length !== expectedKeys.length || keys.some((key, index) => key !== expectedKeys[index])) {
    throw new TypeError("refund runtime returned an unexpected diagnostic shape");
  }
  if (!REFUND_SUPPORT.includes(result.value)) {
    throw new TypeError("refund runtime returned a value outside the declared support");
  }
  if (!Number.isFinite(result.confidence) || result.confidence < 0 || result.confidence > 1) {
    throw new TypeError("refund runtime returned invalid confidence");
  }
  if (!Number.isFinite(result.uncertainty) || result.uncertainty < 0 || result.uncertainty > 1) {
    throw new TypeError("refund runtime returned invalid uncertainty");
  }
  if (result.expectedValue !== null) {
    throw new TypeError("nominal refund output must have a null expected value");
  }
  if (!Array.isArray(result.distribution) || result.distribution.length !== REFUND_SUPPORT.length) {
    throw new TypeError("refund runtime must return the complete declared distribution");
  }

  let probabilitySum = 0;
  let maximumProbability = 0;
  for (const [index, entry] of result.distribution.entries()) {
    if (
      entry === null ||
      typeof entry !== "object" ||
      Array.isArray(entry) ||
      Object.keys(entry).length !== 2 ||
      !Object.hasOwn(entry, "value") ||
      !Object.hasOwn(entry, "probability") ||
      entry.value !== REFUND_SUPPORT[index] ||
      !Number.isFinite(entry.probability) ||
      entry.probability < 0 ||
      entry.probability > 1
    ) {
      throw new TypeError("refund runtime returned an invalid support-ordered distribution");
    }
    probabilitySum += entry.probability;
    maximumProbability = Math.max(maximumProbability, entry.probability);
  }
  if (Math.abs(probabilitySum - 1) > 1e-12) {
    throw new TypeError("refund runtime probabilities do not sum to one");
  }
  if (Math.abs(maximumProbability - result.confidence) > Number.EPSILON * 8) {
    throw new TypeError("refund runtime confidence is not the top probability");
  }
  const expectedValue = result.distribution.find(
    ({ probability }) => probability === maximumProbability,
  )?.value;
  if (result.value !== expectedValue) {
    throw new TypeError("refund runtime value is not the stable top-probability decision");
  }
}

async function readBoundedStandardInput() {
  const chunks = [];
  let byteLength = 0;
  for await (const chunk of process.stdin) {
    byteLength += chunk.byteLength;
    if (byteLength > MAXIMUM_STDIN_BYTES) {
      throw new RangeError(`runtime input exceeds ${String(MAXIMUM_STDIN_BYTES)} bytes`);
    }
    chunks.push(chunk);
  }
  return Buffer.concat(chunks, byteLength).toString("utf8");
}

async function main() {
  const [artifactRoot, functionId] = process.argv.slice(2);
  if (artifactRoot === undefined || functionId === undefined) {
    throw new Error("usage: node run-runtime.mjs <artifact-root> <function-id> < input.json");
  }
  const text = await readBoundedStandardInput();
  const inputs = JSON.parse(text);
  const result = await runRefundDiagnostic(artifactRoot, functionId, inputs);
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

const invokedPath = process.argv[1];
if (invokedPath !== undefined && import.meta.url === pathToFileURL(invokedPath).href) {
  await main();
}
