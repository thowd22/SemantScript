// Measures a cold start of the built app: a fresh Node process, the artifact
// load, and the first request. Reports wall-clock milliseconds and the
// process's resident set size after the first response.
//
//   node scripts/cold-start.mjs            # the Express server on port 3000
//   node scripts/cold-start.mjs --lambda   # one invocation of deploy/lambda.mjs
//
// Set SEMANTSCRIPT_ARTIFACT to measure against another artifact root.
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const lambda = process.argv.includes("--lambda");
const runs = Number(
  process.argv.find((a) => a.startsWith("--runs="))?.slice(7) ?? 3,
);
const request = {
  subject: "site down",
  body: "checkout returns 500 for everyone",
};

function rssMegabytes(pid) {
  try {
    const status = readFileSync(`/proc/${pid}/status`, "utf8");
    const match = /VmRSS:\s+(\d+) kB/u.exec(status);
    return match ? Math.round(Number(match[1]) / 1024) : NaN;
  } catch {
    return NaN; // not Linux, or the process already exited
  }
}

async function measureServer() {
  const started = performance.now();
  const child = spawn(
    process.execPath,
    ["--enable-source-maps", join(root, "dist", "server.js")],
    {
      cwd: root,
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  let listening;
  let exited;
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    if (String(chunk).includes("listening")) listening = performance.now();
  });
  child.stderr.on("data", (chunk) => (stderr += chunk));
  child.on("exit", (code) => (exited = code ?? -1));
  try {
    let response;
    for (let attempt = 0; attempt < 600; attempt += 1) {
      if (exited !== undefined) {
        throw new Error(
          `the server exited with status ${exited} before answering:\n${stderr}`,
        );
      }
      try {
        response = await fetch("http://localhost:3000/tickets", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(request),
        });
        break;
      } catch {
        await new Promise((resolve) => setTimeout(resolve, 25));
      }
    }
    const firstResponse = performance.now();
    const status = response?.status;
    await response?.text();
    return {
      listeningMs: Math.round((listening ?? firstResponse) - started),
      firstResponseMs: Math.round(firstResponse - started),
      status,
      rssMb: rssMegabytes(child.pid),
    };
  } finally {
    child.kill();
  }
}

async function measureLambda() {
  const started = performance.now();
  const script = `
    import { closeSemaArtifact } from "@semantscript/core";
    import { handler } from ${JSON.stringify(join(root, "deploy", "lambda.mjs"))};
    const result = await handler({ body: ${JSON.stringify(JSON.stringify(request))} });
    process.stdout.write(JSON.stringify({ status: result.statusCode, rssMb: Math.round(process.memoryUsage().rss / 1048576) }));
    // A function platform freezes the environment between invocations; here the
    // inference worker must be closed for the process to exit.
    await closeSemaArtifact();
  `;
  const child = spawn(process.execPath, ["--input-type=module", "-e", script], {
    cwd: root,
    stdio: ["ignore", "pipe", "inherit"],
  });
  let output = "";
  child.stdout.on("data", (chunk) => (output += chunk));
  await new Promise((resolve) => child.on("close", resolve));
  const firstResponse = performance.now();
  const parsed = JSON.parse(output || "{}");
  return {
    firstResponseMs: Math.round(firstResponse - started),
    status: parsed.status,
    rssMb: parsed.rssMb,
  };
}

const results = [];
for (let run = 0; run < runs; run += 1) {
  results.push(await (lambda ? measureLambda() : measureServer()));
}
console.log(
  JSON.stringify(
    {
      mode: lambda ? "lambda" : "server",
      node: process.version,
      runs: results,
    },
    null,
    2,
  ),
);
