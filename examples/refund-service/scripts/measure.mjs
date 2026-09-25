// Score the trained artifact on the held-out set scripts/heldout.py wrote and
// time every call through the Node runtime: accuracy per expression against
// the constraint labels, p50 and p95 latency after a warm-up, and the
// verification ECE from the trainer's report. Prints a Markdown table and
// writes .semantscript/measurement.json.
//
//   node scripts/measure.mjs [artifact-root] [heldout.json] [train-report.json]
import { readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { loadSemaArtifact } from "@semantscript/core";

const here = dirname(fileURLToPath(import.meta.url));
const stateDirectory = join(here, "..", ".semantscript");
const [artifactRoot, heldoutPath, reportPath] = [
  process.argv[2] ?? join(stateDirectory, "artifact"),
  process.argv[3] ?? join(stateDirectory, "heldout.json"),
  process.argv[4] ?? join(stateDirectory, "train-report.json"),
].map((path) => resolve(path));

const heldout = JSON.parse(await readFile(heldoutPath, "utf8"));
const report = JSON.parse(await readFile(reportPath, "utf8"));
const bundle = JSON.parse(
  await readFile(join(here, "..", "dist", "semantscript.ir.v1.json"), "utf8"),
);
const names = new Map(
  bundle.functions.map((fn) => [fn.id, `${fn.source.path}:${fn.source.line}`]),
);
const verification = new Map(
  report.functions.map((fn) => [fn.id, fn.verification.metrics]),
);
const pick = (sorted, q) =>
  sorted[
    Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1))
  ];

const handle = await loadSemaArtifact(artifactRoot);
const rows = [];
try {
  for (const fn of heldout.functions) {
    if (!handle.functionIds.has(fn.id))
      throw new Error(`${fn.id} is not in the artifact`);
    for (const entry of fn.cases.slice(0, 20)) handle.call(fn.id, entry.inputs);
    const samples = [];
    let correct = 0;
    const misses = [];
    for (const entry of fn.cases) {
      const started = performance.now();
      const value = handle.call(fn.id, entry.inputs);
      samples.push(performance.now() - started);
      if (value === entry.expected) correct += 1;
      else
        misses.push({
          inputs: entry.inputs,
          expected: entry.expected,
          predicted: value,
        });
    }
    const sorted = [...samples].sort((a, b) => a - b);
    const metrics = verification.get(fn.id) ?? {};
    rows.push({
      id: fn.id,
      site: names.get(fn.id) ?? fn.id,
      domain: fn.adapterRef.split(".").at(-1),
      cases: fn.cases.length,
      accuracy: Number((correct / fn.cases.length).toFixed(4)),
      misses: misses.slice(0, 5),
      missCount: misses.length,
      verificationAccuracy: metrics.accuracy ?? null,
      ece: metrics.ece ?? null,
      p50Ms: Number(pick(sorted, 0.5).toFixed(2)),
      p95Ms: Number(pick(sorted, 0.95).toFixed(2)),
    });
  }
} finally {
  await handle.close();
}

const measurement = {
  kind: "refund-service-measurement",
  measuredAt: new Date().toISOString(),
  artifactManifestSha256: report.artifact?.manifestSha256 ?? null,
  node: process.version,
  rows,
};
await writeFile(
  join(stateDirectory, "measurement.json"),
  `${JSON.stringify(measurement, null, 1)}\n`,
);
const lines = [
  "| Expression | Domain | Held-out accuracy | Verification ECE | p50 ms | p95 ms |",
  "| --- | --- | --- | --- | --- | --- |",
  ...rows.map(
    (row) =>
      `| \`${row.site}\` | ${row.domain} | ${(row.accuracy * 100).toFixed(1)}% (${row.cases - row.missCount}/${row.cases}) | ${row.ece === null ? "n/a" : row.ece.toFixed(4)} | ${row.p50Ms.toFixed(2)} | ${row.p95Ms.toFixed(2)} |`,
  ),
];
process.stdout.write(`${lines.join("\n")}\n`);
