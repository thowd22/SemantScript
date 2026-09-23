// Drive the refund benchmark: run each required system on the frozen final
// dataset with one shared environment and warmup protocol, persist every
// sealed prediction set, and assemble the machine-checked result and go/no-go.
//
// usage:
//   node benchmarks/refund/program/run-benchmark.mjs run \
//     --dataset <final-benchmark-dataset.json> --output-dir <dir> \
//     --warmup-source <synthetic-dataset.json> --systems ollama-1b,ollama-7b,laya \
//     [--pipeline-manifest <pipeline-manifest.json>] [--laya-source <repo>] \
//     [--laya-checkpoint <snapshot>] [--accelerator "<name>"]
//   node benchmarks/refund/program/run-benchmark.mjs assemble \
//     --dataset <final-benchmark-dataset.json> --ledger <training-input-ledger.json> \
//     --output-dir <dir>

import { createHash } from "node:crypto";
import { readFile, readdir, writeFile, mkdir } from "node:fs/promises";
import os from "node:os";
import { dirname, join, resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { REQUIRED_SYSTEM_ROLES, validateRefundDataset } from "../dist/index.js";
import {
  createAnthropicSonnetAdapter,
  createLayaAdapter,
  createOllamaQwenAdapter,
} from "../dist/adapters/index.js";
import {
  createLiveAnthropicTransport,
  createLiveLayaRunner,
  createLiveOllamaTransport,
} from "../dist/live/index.js";
import { createProcessRssSampler } from "../dist/measurement.js";
import { createBenchmarkResult } from "../dist/metrics.js";
import { REFUND_SYSTEM_PINS, REFUND_TASK_SPEC_SHA256 } from "../dist/policy.js";
import { runRefundBenchmark } from "../dist/runner.js";
import { createSemantScriptRefundAdapter } from "../dist/semantscript-adapter.js";

const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const DEFAULT_WARMUP_ITERATIONS = 10;
const DEFAULT_WARMUP_INPUT_COUNT = 8;

function sha256(text) {
  return createHash("sha256").update(text).digest("hex");
}

function utcNow() {
  return new Date().toISOString().replace(/\.\d{3}Z$/u, "Z");
}

function dump(value) {
  return `${JSON.stringify(value, null, 1)}\n`;
}

async function readJson(file) {
  return JSON.parse(await readFile(file, "utf8"));
}

async function warmupInputsFrom(source, count) {
  const document = await readJson(source);
  const cases = document.payload?.cases ?? document.cases;
  if (!Array.isArray(cases) || cases.length < count) {
    throw new Error("warmup source must contain at least the requested number of cases");
  }
  return cases.slice(0, count).map((entry) => structuredClone(entry.inputs));
}

async function loadOrCaptureEnvironment(outputDir, accelerator) {
  const evidencePath = join(outputDir, "environment.json");
  try {
    const existing = await readJson(evidencePath);
    return existing.environment;
  } catch {
    // first capture below
  }
  const runtimeVersions = [{ name: "node", version: process.versions.node }];
  try {
    const onnx = await readJson(join(REPOSITORY_ROOT, "node_modules/onnxruntime-node/package.json"));
    runtimeVersions.push({ name: "onnxruntime-node", version: onnx.version });
  } catch {
    // runtime package absent: recorded by omission
  }
  try {
    const response = await fetch("http://127.0.0.1:11434/api/version");
    const body = await response.json();
    runtimeVersions.push({ name: "ollama", version: String(body.version) });
  } catch {
    // Ollama not running: recorded by omission
  }
  runtimeVersions.sort((left, right) => (left.name < right.name ? -1 : left.name > right.name ? 1 : 0));
  const cpus = os.cpus();
  const capturedAt = utcNow();
  const evidence = {
    capturedAt,
    os: { type: os.type(), release: os.release(), version: os.version(), platform: os.platform() },
    architecture: os.arch(),
    cpu: { model: cpus[0]?.model ?? "unknown", count: cpus.length },
    totalMemoryBytes: os.totalmem(),
    accelerator,
    runtimeVersions,
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
  await writeFile(evidencePath, dump({ environment, evidence }));
  return environment;
}

async function semantscriptAdapter(pipelineManifestPath) {
  const manifest = await readJson(pipelineManifestPath);
  const releaseManifest = await readJson(join(manifest.artifact.releaseDirectory, "manifest.json"));
  const fn = releaseManifest.functions[0];
  const trainingKeySha256 = manifest.ledger.trainingKeySha256;
  return createSemantScriptRefundAdapter({
    artifactRoot: manifest.artifact.root,
    expected: {
      taskSpecSha256: REFUND_TASK_SPEC_SHA256,
      function: {
        id: manifest.function.id,
        semanticSha256: manifest.function.semanticSha256,
        resultMode: fn.runtime.resultMode,
        trainingProvenance: {
          datasetSha256: manifest.ledger.sources.baseDatasetSha256,
          trainingKeySha256,
          baseModel: fn.trainingProvenance.baseModel,
        },
      },
      model: {
        provider: "semantscript",
        name: "refund-decision",
        version: "semantscript-artifact-v1",
        revision: trainingKeySha256,
        artifactSha256: manifest.artifact.manifestSha256,
      },
      trainingEvidence: {
        trainingLedgerSha256: manifest.ledger.payloadSha256,
        artifactTrainingDatasetSha256: manifest.ledger.sources.baseDatasetSha256,
        artifactTrainingKeySha256: trainingKeySha256,
        releaseVerificationPayloadSha256: manifest.ledger.sources.releaseVerificationPayloadSha256,
        releaseVerificationAttestationSha256:
          manifest.ledger.sources.releaseVerificationAttestationSha256,
      },
    },
  });
}

async function buildAdapter(role, options) {
  if (role === "semantscript") {
    if (!options["pipeline-manifest"]) throw new Error("semantscript needs --pipeline-manifest");
    return semantscriptAdapter(resolve(options["pipeline-manifest"]));
  }
  if (role === "ollama-1b" || role === "ollama-7b") {
    return createOllamaQwenAdapter({
      role,
      model: REFUND_SYSTEM_PINS[role].model,
      transport: createLiveOllamaTransport(),
    });
  }
  if (role === "laya") {
    if (!options["laya-source"] || !options["laya-checkpoint"]) {
      throw new Error("laya needs --laya-source and --laya-checkpoint");
    }
    const runner = await createLiveLayaRunner({
      sourcePath: resolve(options["laya-source"]),
      checkpointPath: resolve(options["laya-checkpoint"]),
      pythonExecutable: options["python"] ?? "python3",
      device: "cuda",
      // The user site-packages carries a NumPy-2 SciPy that breaks the project's
      // NumPy 1.26 through transformers' optional imports; isolate the worker from it.
      environment: { PYTHONPATH: join(REPOSITORY_ROOT, ".python-packages"), PYTHONNOUSERSITE: "1" },
    });
    const adapter = createLayaAdapter({ model: REFUND_SYSTEM_PINS.laya.model, runner });
    return { ...adapter, close: async () => runner.close() };
  }
  if (role === "structured-api") {
    if (!process.env.ANTHROPIC_API_KEY) throw new Error("structured-api needs ANTHROPIC_API_KEY");
    return createAnthropicSonnetAdapter({
      model: { ...REFUND_SYSTEM_PINS["structured-api"].model, artifactSha256: sha256("claude-sonnet-5") },
      transport: createLiveAnthropicTransport(),
    });
  }
  throw new Error(`unknown system role ${role}`);
}

async function runSystems(options) {
  const outputDir = resolve(options["output-dir"]);
  await mkdir(outputDir, { recursive: true });
  const dataset = validateRefundDataset(await readJson(resolve(options.dataset)));
  const warmupInputs = await warmupInputsFrom(
    resolve(options["warmup-source"]),
    DEFAULT_WARMUP_INPUT_COUNT,
  );
  const environment = await loadOrCaptureEnvironment(outputDir, options.accelerator ?? null);
  const systems = options.systems.split(",").map((value) => value.trim()).filter(Boolean);
  for (const role of systems) {
    if (!REQUIRED_SYSTEM_ROLES.includes(role)) throw new Error(`unknown system role ${role}`);
  }
  const failures = {};
  for (const role of systems) {
    process.stdout.write(`${utcNow()} ${role}: preparing adapter\n`);
    let adapter;
    try {
      adapter = await buildAdapter(role, options);
    } catch (error) {
      failures[role] = error instanceof Error ? error.message : String(error);
      process.stdout.write(`${utcNow()} ${role}: adapter unavailable: ${failures[role]}\n`);
      continue;
    }
    const startedAt = Date.now();
    try {
      const predictionSet = await runRefundBenchmark({
        dataset,
        adapter,
        environment,
        warmupIterations: DEFAULT_WARMUP_ITERATIONS,
        warmupInputs,
        memorySampler: createProcessRssSampler(),
      });
      await writeFile(join(outputDir, `predictions-${role}.json`), dump(predictionSet));
      const seconds = ((Date.now() - startedAt) / 1000).toFixed(0);
      process.stdout.write(
        `${utcNow()} ${role}: ${predictionSet.predictions.length} predictions in ${seconds}s, ` +
          `measured ${predictionSet.protocol.measuredDurationMs.toFixed(0)} ms\n`,
      );
    } catch (error) {
      failures[role] = error instanceof Error ? error.message : String(error);
      process.stdout.write(`${utcNow()} ${role}: failed: ${failures[role]}\n`);
    } finally {
      if (typeof adapter.close === "function") await adapter.close();
    }
  }
  await writeFile(join(outputDir, "failures.json"), dump(failures));
  if (Object.keys(failures).length > 0) {
    process.exitCode = 2;
  }
}

async function assemble(options) {
  const outputDir = resolve(options["output-dir"]);
  const dataset = await readJson(resolve(options.dataset));
  const ledger = await readJson(resolve(options.ledger));
  const files = (await readdir(outputDir)).filter((name) => /^predictions-.*\.json$/u.test(name));
  const predictions = [];
  for (const name of files.sort()) predictions.push(await readJson(join(outputDir, name)));
  const result = createBenchmarkResult(dataset, ledger, predictions, utcNow());
  await writeFile(join(outputDir, "result.json"), dump(result));
  const rows = result.systems.map((system) => ({
    role: system.role,
    accuracy: system.metrics.accuracy.accuracy.toFixed(4),
    attestedAccuracy: system.metrics.accuracy.attestedAccuracy.toFixed(4),
    ece: system.metrics.calibration.expectedCalibrationError.toFixed(4),
    p50Ms: system.metrics.latency.p50Ms.toFixed(2),
    p95Ms: system.metrics.latency.p95Ms.toFixed(2),
    rps: system.metrics.throughput.requestsPerSecond.toFixed(2),
    peakMB: (system.metrics.memory.peakBytes / 1048576).toFixed(0),
  }));
  process.stdout.write(`${JSON.stringify({ goNoGo: result.goNoGo, systems: rows }, null, 1)}\n`);
}

async function main() {
  const { positionals, values } = parseArgs({
    allowPositionals: true,
    options: {
      dataset: { type: "string" },
      ledger: { type: "string" },
      "output-dir": { type: "string" },
      "warmup-source": { type: "string" },
      systems: { type: "string" },
      "pipeline-manifest": { type: "string" },
      "laya-source": { type: "string" },
      "laya-checkpoint": { type: "string" },
      python: { type: "string" },
      accelerator: { type: "string" },
    },
  });
  const command = positionals[0];
  if (command === "run") {
    for (const name of ["dataset", "output-dir", "warmup-source", "systems"]) {
      if (!values[name]) throw new Error(`run needs --${name}`);
    }
    await runSystems(values);
    return;
  }
  if (command === "assemble") {
    for (const name of ["dataset", "ledger", "output-dir"]) {
      if (!values[name]) throw new Error(`assemble needs --${name}`);
    }
    await assemble(values);
    return;
  }
  throw new Error("usage: run-benchmark.mjs <run|assemble> ...");
}

main().catch((error) => {
  process.stderr.write(`benchmark failed: ${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  process.exit(1);
});
