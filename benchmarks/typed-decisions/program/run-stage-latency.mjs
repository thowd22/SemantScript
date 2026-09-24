// Time one fused execution-plan stage per test case (all of a workflow's
// questions over one state) through the Node runtime. Prints a JSON summary.
import { readFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import process from "node:process";

import { loadSemaArtifact } from "@semantscript/core";

const [artifactRoot, statesPath, functionIdsJson, warmupText] =
  process.argv.slice(2);
const functionIds = JSON.parse(functionIdsJson);
const warmup = Number(warmupText ?? "10");
const states = JSON.parse(await readFile(statesPath, "utf8"));
const handle = await loadSemaArtifact(artifactRoot);
try {
  const samples = [];
  let passes;
  states.forEach((state, index) => {
    const started = performance.now();
    const outcome = handle.callStage(
      functionIds.map((functionId) => ({ functionId, inputs: { state } })),
    );
    const elapsed = performance.now() - started;
    passes = outcome.passes;
    if (index >= warmup) samples.push(elapsed);
  });
  const sorted = [...samples].sort((a, b) => a - b);
  const pick = (q) =>
    sorted[
      Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1))
    ];
  process.stdout.write(
    JSON.stringify({
      p50Ms: pick(0.5),
      p95Ms: pick(0.95),
      meanMs: samples.reduce((a, b) => a + b, 0) / samples.length,
      count: samples.length,
      passes,
      questionsPerCase: functionIds.length,
    }),
  );
} finally {
  await handle.close();
}
