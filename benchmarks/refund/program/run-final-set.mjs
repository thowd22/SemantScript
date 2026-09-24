// Score one exported refund artifact on the frozen final set through the Node
// runtime and time every call: accuracy against the attested decision, p50 and
// p95 latency after a warm-up prefix. Prints a JSON summary.
//
//   node benchmarks/refund/program/run-final-set.mjs <artifact-root> <function-id> \
//     <final-benchmark-dataset.json> [warmup]
import { readFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import process from "node:process";

import { loadSemaArtifact } from "@semantscript/core";

const [artifactRoot, functionId, datasetPath, warmupText] =
  process.argv.slice(2);
if (!artifactRoot || !functionId || !datasetPath) {
  process.stderr.write(
    "usage: run-final-set.mjs <artifact-root> <function-id> <dataset.json> [warmup]\n",
  );
  process.exit(2);
}
const warmup = Number(warmupText ?? "20");
const dataset = JSON.parse(await readFile(datasetPath, "utf8"));
const cases = dataset.cases;
const handle = await loadSemaArtifact(artifactRoot);
try {
  if (!handle.functionIds.has(functionId)) {
    throw new Error(`function ${functionId} is absent from the artifact`);
  }
  // Warm the session on the first cases, then time every case once in order.
  for (const entry of cases.slice(0, warmup))
    handle.call(functionId, entry.inputs);
  const samples = [];
  const misses = [];
  let correct = 0;
  for (const entry of cases) {
    const started = performance.now();
    const result = handle.call(functionId, entry.inputs);
    samples.push(performance.now() - started);
    const value =
      result !== null && typeof result === "object" && "value" in result
        ? result.value
        : result;
    if (value === entry.expected) correct += 1;
    else
      misses.push({ id: entry.id, expected: entry.expected, predicted: value });
  }
  const sorted = [...samples].sort((a, b) => a - b);
  const pick = (q) =>
    sorted[
      Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1))
    ];
  process.stdout.write(
    `${JSON.stringify({
      cases: cases.length,
      correct,
      accuracy: Number((correct / cases.length).toFixed(4)),
      misses,
      latencyMs: {
        p50: Number(pick(0.5).toFixed(3)),
        p95: Number(pick(0.95).toFixed(3)),
        mean: Number(
          (samples.reduce((a, b) => a + b, 0) / samples.length).toFixed(3),
        ),
        min: Number(sorted[0].toFixed(3)),
      },
      warmup,
      manifestSha256: handle.manifestSha256,
    })}\n`,
  );
} finally {
  await handle.close();
}
