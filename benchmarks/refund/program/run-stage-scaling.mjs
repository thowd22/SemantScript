// Measure the shared-encoder thesis on the CPU runtime (TASK-6.5): how stage
// latency and throughput move with the number of heads per stage, fused versus
// one call per head, and how one head scales with the number of distinct inputs
// in a stage (the batch curve).
//
//   node benchmarks/refund/program/run-stage-scaling.mjs \
//     --artifact <artifact-root> --inputs-source <dataset.json> --output-dir <dir> \
//     [--head-counts 1,10,50] [--batch-sizes 1,2,4,8,16,32,64] [--iterations 30]
//
// The artifact is expected to expose at least max(head-counts) functions over one
// encoder and adapter (see derive_multihead_artifact.py). Inputs come from a
// synthetic or warmup corpus, never from an evaluation set.

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import { dirname, join, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

export const DEFAULT_HEAD_COUNTS = [1, 10, 50];
export const DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64];
export const DEFAULT_ITERATIONS = 30;
export const DEFAULT_WARMUP_ITERATIONS = 5;

const REPOSITORY_ROOT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../..",
);

function sha256(text) {
  return createHash("sha256").update(text).digest("hex");
}

function utcNow() {
  return new Date().toISOString().replace(/\.\d{3}Z$/u, "Z");
}

function dump(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function positiveIntegerList(text, name) {
  const values = text.split(",").map((item) => Number(item.trim()));
  if (
    values.length === 0 ||
    values.some((value) => !Number.isInteger(value) || value < 1)
  ) {
    throw new TypeError(
      `${name} must be a comma-separated list of positive integers`,
    );
  }
  return values;
}

/** Nearest-rank percentile of ascending samples; q in [0, 1]. */
export function percentile(sorted, q) {
  if (sorted.length === 0) throw new RangeError("percentile of no samples");
  const rank = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(q * sorted.length) - 1),
  );
  return sorted[rank];
}

export function summarize(samples) {
  const sorted = [...samples].sort((left, right) => left - right);
  const total = sorted.reduce((sum, value) => sum + value, 0);
  return {
    count: sorted.length,
    meanMs: total / sorted.length,
    p50Ms: percentile(sorted, 0.5),
    p95Ms: percentile(sorted, 0.95),
    minMs: sorted[0],
    maxMs: sorted[sorted.length - 1],
  };
}

function timed(now, run) {
  const started = now();
  const result = run();
  return { elapsedMs: now() - started, result };
}

function assertPasses(actual, expected, context) {
  for (const key of ["encoder", "adapter", "head"]) {
    if (actual[key] !== expected[key]) {
      throw new Error(
        `${context}: expected ${key} passes ${expected[key]}, saw ${actual[key]}`,
      );
    }
  }
}

/**
 * Time head-count and batch sweeps on a loaded artifact handle. Pure apart from
 * the clock: returns the measurement record without touching the filesystem.
 */
export function measureStageScaling(handle, options) {
  const {
    inputs,
    headCounts = DEFAULT_HEAD_COUNTS,
    batchSizes = DEFAULT_BATCH_SIZES,
    iterations = DEFAULT_ITERATIONS,
    warmupIterations = DEFAULT_WARMUP_ITERATIONS,
    now = () => performance.now(),
  } = options;
  if (!Array.isArray(inputs) || inputs.length === 0)
    throw new TypeError("inputs must be a nonempty array");
  if (!Number.isInteger(iterations) || iterations < 1)
    throw new TypeError("iterations must be a positive integer");
  if (!Number.isInteger(warmupIterations) || warmupIterations < 0) {
    throw new TypeError("warmupIterations must be a non-negative integer");
  }
  const functionIds = [...handle.functionIds].sort();
  const maximumHeads = Math.max(...headCounts);
  const maximumBatch = Math.max(...batchSizes);
  if (functionIds.length < maximumHeads) {
    throw new RangeError(
      `artifact exposes ${functionIds.length} functions; ${maximumHeads} heads requested`,
    );
  }
  if (inputs.length < maximumBatch) {
    throw new RangeError(
      `${inputs.length} distinct inputs supplied; batch of ${maximumBatch} requested`,
    );
  }

  const headRequests = (count) =>
    functionIds
      .slice(0, count)
      .map((functionId) => ({ functionId, inputs: inputs[0] }));
  const batchRequests = (count) =>
    inputs
      .slice(0, count)
      .map((entry) => ({ functionId: functionIds[0], inputs: entry }));

  for (let index = 0; index < warmupIterations; index += 1) {
    handle.callStage(headRequests(maximumHeads));
    handle.callStage(batchRequests(maximumBatch));
    for (const functionId of functionIds.slice(0, Math.min(maximumHeads, 3)))
      handle.call(functionId, inputs[0]);
  }

  const headsPerStage = [];
  for (const heads of headCounts) {
    const requests = headRequests(heads);
    const fusedSamples = [];
    const sequentialSamples = [];
    let passes = null;
    for (let index = 0; index < iterations; index += 1) {
      const fused = timed(now, () => handle.callStage(requests));
      assertPasses(
        fused.result.passes,
        { encoder: 1, adapter: 1, head: heads },
        `${heads} heads fused`,
      );
      passes = fused.result.passes;
      fusedSamples.push(fused.elapsedMs);

      const sequential = timed(now, () =>
        requests.map(({ functionId }) => handle.call(functionId, inputs[0])),
      );
      sequentialSamples.push(sequential.elapsedMs);
      if (
        JSON.stringify(sequential.result) !==
        JSON.stringify(fused.result.results)
      ) {
        throw new Error(
          `${heads} heads: fused stage results differ from one call per head`,
        );
      }
    }
    const fused = summarize(fusedSamples);
    const sequential = summarize(sequentialSamples);
    headsPerStage.push({
      heads,
      passes,
      fused,
      sequential,
      fusedDecisionsPerSecond: (heads * 1000) / fused.meanMs,
      sequentialDecisionsPerSecond: (heads * 1000) / sequential.meanMs,
      fusedPerDecisionP50Ms: fused.p50Ms / heads,
      sequentialOverFusedP50: sequential.p50Ms / fused.p50Ms,
    });
  }

  const batchScaling = [];
  for (const batch of batchSizes) {
    const requests = batchRequests(batch);
    const samples = [];
    let passes = null;
    for (let index = 0; index < iterations; index += 1) {
      const stage = timed(now, () => handle.callStage(requests));
      assertPasses(
        stage.result.passes,
        { encoder: batch, adapter: batch, head: batch },
        `batch of ${batch}`,
      );
      passes = stage.result.passes;
      samples.push(stage.elapsedMs);
    }
    const stage = summarize(samples);
    batchScaling.push({
      batch,
      passes,
      stage,
      perDecisionP50Ms: stage.p50Ms / batch,
      decisionsPerSecond: (batch * 1000) / stage.meanMs,
    });
  }

  return {
    functionCount: functionIds.length,
    distinctInputs: inputs.length,
    iterations,
    warmupIterations,
    headsPerStage,
    batchScaling,
  };
}

function formatMs(value) {
  return value.toFixed(2);
}

export function renderMarkdown(record) {
  const lines = [
    "| Heads per stage | Fused p50 ms | Fused p95 ms | One call per head p50 ms | Speedup (p50) | Fused decisions/s | Per-decision p50 ms |",
    "| --- | --- | --- | --- | --- | --- | --- |",
  ];
  for (const row of record.headsPerStage) {
    lines.push(
      `| ${row.heads} | ${formatMs(row.fused.p50Ms)} | ${formatMs(row.fused.p95Ms)} | ${formatMs(row.sequential.p50Ms)} | ${row.sequentialOverFusedP50.toFixed(2)}x | ${row.fusedDecisionsPerSecond.toFixed(1)} | ${formatMs(row.fusedPerDecisionP50Ms)} |`,
    );
  }
  lines.push(
    "",
    "| Distinct inputs per stage | Stage p50 ms | Stage p95 ms | Per-decision p50 ms | Decisions/s |",
    "| --- | --- | --- | --- | --- |",
  );
  for (const row of record.batchScaling) {
    lines.push(
      `| ${row.batch} | ${formatMs(row.stage.p50Ms)} | ${formatMs(row.stage.p95Ms)} | ${formatMs(row.perDecisionP50Ms)} | ${row.decisionsPerSecond.toFixed(1)} |`,
    );
  }
  return `${lines.join("\n")}\n`;
}

async function readJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

async function inputsFrom(source, count) {
  const document = await readJson(source);
  const cases = document.payload?.cases ?? document.cases;
  if (!Array.isArray(cases) || cases.length < count) {
    throw new Error(`inputs source must contain at least ${count} cases`);
  }
  const seen = new Set();
  const inputs = [];
  for (const entry of cases) {
    const key = JSON.stringify(entry.inputs);
    if (seen.has(key)) continue;
    seen.add(key);
    inputs.push(structuredClone(entry.inputs));
    if (inputs.length === count) break;
  }
  if (inputs.length < count)
    throw new Error(
      `inputs source holds only ${inputs.length} distinct inputs`,
    );
  return inputs;
}

async function captureEnvironment(accelerator) {
  const runtimeVersions = [{ name: "node", version: process.versions.node }];
  try {
    const onnx = await readJson(
      join(REPOSITORY_ROOT, "node_modules/onnxruntime-node/package.json"),
    );
    runtimeVersions.push({ name: "onnxruntime-node", version: onnx.version });
  } catch {
    // runtime package absent: recorded by omission
  }
  const cpus = os.cpus();
  const capturedAt = utcNow();
  const evidence = {
    capturedAt,
    os: {
      type: os.type(),
      release: os.release(),
      version: os.version(),
      platform: os.platform(),
    },
    architecture: os.arch(),
    cpu: { model: cpus[0]?.model ?? "unknown", count: cpus.length },
    totalMemoryBytes: os.totalmem(),
    accelerator,
    runtimeVersions,
    onnxSessionOptions: {
      executionProviders: ["cpu"],
      threads: "onnxruntime default",
    },
    uvThreadpoolSize: process.env.UV_THREADPOOL_SIZE ?? null,
  };
  const environment = {
    capturedAt,
    operatingSystem: `${os.type()} ${os.release()}`,
    architecture: os.arch(),
    cpu: `${cpus[0]?.model ?? "unknown"} x${cpus.length}`,
    accelerator,
    memoryBytes: os.totalmem(),
    runtimeVersions,
    evidenceSha256: sha256(JSON.stringify(evidence)),
  };
  return { environment, evidence };
}

async function main() {
  const { values: options } = parseArgs({
    options: {
      artifact: { type: "string" },
      "inputs-source": { type: "string" },
      "output-dir": { type: "string" },
      "head-counts": { type: "string", default: DEFAULT_HEAD_COUNTS.join(",") },
      "batch-sizes": { type: "string", default: DEFAULT_BATCH_SIZES.join(",") },
      iterations: { type: "string", default: String(DEFAULT_ITERATIONS) },
      "warmup-iterations": {
        type: "string",
        default: String(DEFAULT_WARMUP_ITERATIONS),
      },
      accelerator: { type: "string" },
    },
  });
  for (const name of ["artifact", "inputs-source", "output-dir"]) {
    if (typeof options[name] !== "string" || options[name].length === 0)
      throw new Error(`--${name} is required`);
  }
  const headCounts = positiveIntegerList(
    options["head-counts"],
    "--head-counts",
  );
  const batchSizes = positiveIntegerList(
    options["batch-sizes"],
    "--batch-sizes",
  );
  const iterations = Number(options.iterations);
  const warmupIterations = Number(options["warmup-iterations"]);
  const artifactRoot = resolve(options.artifact);
  const inputs = await inputsFrom(
    resolve(options["inputs-source"]),
    Math.max(...batchSizes),
  );
  const outputDir = resolve(options["output-dir"]);
  await mkdir(outputDir, { recursive: true });

  const { loadSemaArtifact } = await import("@semantscript/core");
  const pointer = await readJson(join(artifactRoot, "current.json"));
  const handle = await loadSemaArtifact(artifactRoot);
  let measurement;
  const startedAt = utcNow();
  try {
    measurement = measureStageScaling(handle, {
      inputs,
      headCounts,
      batchSizes,
      iterations,
      warmupIterations,
    });
  } finally {
    await handle.close();
  }
  const { environment, evidence } = await captureEnvironment(
    options.accelerator ?? null,
  );
  const record = {
    kind: "semantscript.stage-scaling-result",
    resultVersion: 1,
    startedAt,
    completedAt: utcNow(),
    artifact: {
      root: options.artifact,
      manifestSha256: pointer.manifestSha256,
    },
    inputsSource: options["inputs-source"],
    inputsSha256: sha256(JSON.stringify(inputs)),
    environment,
    ...measurement,
  };
  await writeFile(join(outputDir, "results.json"), dump(record));
  await writeFile(
    join(outputDir, "environment.json"),
    dump({ environment, evidence }),
  );
  process.stdout.write(renderMarkdown(record));
}

if (
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
