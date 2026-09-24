import assert from "node:assert/strict";
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import process from "node:process";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  createFixtureArtifact,
  fixtureFunctionId,
} from "../../runtime/test/fixtures/artifact.mjs";
import {
  canonical,
  pythonPath,
  renderTrainReport,
  runCli,
  USAGE,
} from "../dist/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixtures = join(here, "fixtures");

function capture(cwd, env = {}) {
  const out = [];
  const err = [];
  return {
    io: {
      cwd,
      env: { ...process.env, ...env },
      stdout: (text) => out.push(text),
      stderr: (text) => err.push(text),
    },
    stdout: () => out.join(""),
    stderr: () => err.join(""),
  };
}

async function scratch(t, prefix) {
  const root = await mkdtemp(join(tmpdir(), prefix));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

async function createProject(root, sources) {
  const coreRoot = join(root, "node_modules", "@semantscript", "core");
  await mkdir(coreRoot, { recursive: true });
  await writeFile(
    join(coreRoot, "package.json"),
    JSON.stringify({
      name: "@semantscript/core",
      type: "module",
      types: "index.d.ts",
    }),
  );
  await copyFile(join(fixtures, "core.d.ts"), join(coreRoot, "index.d.ts"));
  await writeFile(
    join(root, "package.json"),
    JSON.stringify({ type: "module" }),
  );
  await mkdir(join(root, "src"), { recursive: true });
  for (const [name, text] of Object.entries(sources)) {
    await writeFile(join(root, "src", name), text);
  }
  await writeFile(
    join(root, "tsconfig.json"),
    JSON.stringify({
      compilerOptions: {
        module: "NodeNext",
        moduleResolution: "NodeNext",
        target: "ES2023",
        strict: true,
        outDir: "dist",
        rootDir: "src",
        skipLibCheck: true,
      },
      include: ["src/**/*.ts"],
    }),
  );
  return join(root, "tsconfig.json");
}

const program = `import { sema } from "@semantscript/core";
declare const message: string;
export const verdict = sema<"yes" | "no">\`Is this positive? \${message}\`;
`;

test("usage and unknown commands exit with status 2", async () => {
  const empty = capture(process.cwd());
  assert.equal(await runCli([], empty.io), 2);
  assert.equal(empty.stdout(), USAGE);
  const help = capture(process.cwd());
  assert.equal(await runCli(["--help"], help.io), 0);
  const bogus = capture(process.cwd());
  assert.equal(await runCli(["bogus"], bogus.io), 2);
  assert.match(bogus.stderr(), /unknown command bogus/u);
  const missing = capture(process.cwd());
  assert.equal(await runCli(["test"], missing.io), 2);
  assert.match(missing.stderr(), /--artifact is required/u);
  const unknownOption = capture(process.cwd());
  assert.equal(await runCli(["build", "--nope"], unknownOption.io), 2);
  assert.match(unknownOption.stderr(), /Unknown option/u);
});

test("build compiles a project to runtime calls plus an IR bundle under the application refs", async (t) => {
  const root = await scratch(t, "semantscript-cli-build-");
  const configPath = await createProject(root, { "app.sem.ts": program });
  const run = capture(root);

  const status = await runCli(
    ["build", "--project", configPath, "--application", "demo-app"],
    run.io,
  );

  assert.equal(status, 0, run.stderr());
  assert.match(
    run.stdout(),
    /compiled 1 neural function\(s\) for application demo-app/u,
  );
  assert.match(
    run.stdout(),
    /src\/app\.sem\.ts:3:24 {2}"yes" \| "no" {2}\(1 head\)/u,
  );
  assert.match(run.stdout(), /bundle: dist\/semantscript\.ir\.v1\.json/u);
  const bundle = JSON.parse(
    await readFile(join(root, "dist", "semantscript.ir.v1.json"), "utf8"),
  );
  assert.equal(bundle.kind, "semantscript.ir-bundle");
  assert.equal(bundle.functions.length, 1);
  assert.equal(bundle.functions[0].stage, "source");
  assert.equal(bundle.functions[0].model.encoder, "encoder.demo-app");
  assert.equal(bundle.functions[0].model.adapter, "adapter.demo-app");
  const emitted = await readFile(join(root, "dist", "app.sem.js"), "utf8");
  assert.match(emitted, /__sema\.call\("nf_[a-f0-9]{64}"/u);
  assert.doesNotMatch(
    emitted,
    /Is this positive/u,
    "prompt text never reaches the emitted JavaScript",
  );

  const invalid = capture(root);
  assert.equal(
    await runCli(
      ["build", "--project", configPath, "--application", "Demo App"],
      invalid.io,
    ),
    2,
  );
  assert.match(invalid.stderr(), /--application must be lowercase/u);
});

test("build reports analysis and type diagnostics without emitting", async (t) => {
  const root = await scratch(t, "semantscript-cli-build-diag-");
  const configPath = await createProject(root, {
    "app.sem.ts": `import { sema } from "@semantscript/core";
declare const message: string;
export const freeText = sema<string>\`free text \${message}\`;
`,
  });
  const analysis = capture(root);
  assert.equal(
    await runCli(["build", "--project", configPath], analysis.io),
    1,
  );
  assert.match(analysis.stderr(), /not a supported finite scalar output/u);
  assert.equal(analysis.stdout(), "");

  const typed = await createProject(join(root, "typed"), {
    "app.sem.ts": `${program}const count: number = "one";\nexport { count };\n`,
  });
  const typeErrors = capture(root);
  assert.equal(await runCli(["build", "--project", typed], typeErrors.io), 1);
  assert.match(typeErrors.stderr(), /TS2322/u);

  const missingOutDir = capture(root);
  await writeFile(
    join(root, "no-outdir.json"),
    JSON.stringify({ compilerOptions: {}, include: ["src/**/*.ts"] }),
  );
  assert.equal(
    await runCli(
      ["build", "--project", join(root, "no-outdir.json")],
      missingOutDir.io,
    ),
    1,
  );
  assert.match(missingOutDir.stderr(), /must set compilerOptions\.outDir/u);
});

test("test reports shipped verification and replays bundle examples through the runtime", async (t) => {
  const root = await scratch(t, "semantscript-cli-test-");
  const artifactRoot = join(root, "artifact");
  await createFixtureArtifact(artifactRoot);
  const { loadSemaArtifact } = await import("@semantscript/core");
  const handle = await loadSemaArtifact(artifactRoot);
  const inputs = { facts: { a: 1, b: 2 } };
  const expected = handle.call(fixtureFunctionId, inputs);
  await handle.close();
  const wrong = expected === "approve" ? "deny" : "approve";
  const bundleFor = (functions) =>
    JSON.stringify({
      kind: "semantscript.ir-bundle",
      bundleVersion: 1,
      functions,
      executionPlan: { stages: [], dependencies: [] },
    });
  const passing = join(root, "passing.json");
  await writeFile(
    passing,
    bundleFor([
      {
        id: fixtureFunctionId,
        definition: { examples: [{ inputs, output: expected }] },
      },
    ]),
  );
  const failing = join(root, "failing.json");
  await writeFile(
    failing,
    bundleFor([
      {
        id: fixtureFunctionId,
        definition: {
          examples: [
            { inputs, output: expected },
            { inputs: { facts: { b: 2, a: 1 } }, output: wrong },
          ],
        },
      },
      { id: `nf_${"9".repeat(64)}`, definition: { examples: [] } },
    ]),
  );

  const stats = capture(root);
  assert.equal(await runCli(["test", "--artifact", artifactRoot], stats.io), 0);
  assert.match(
    stats.stdout(),
    /nf_11111111…\s+passed\s+1\.0000\s+0\.0000\s+0\.0000\s+1\.0000\s+1\s+0\s+1\.0000\s+-/u,
  );
  assert.match(stats.stdout(), /test passed\n$/u);

  const replay = capture(root);
  assert.equal(
    await runCli(
      ["test", "--artifact", artifactRoot, "--bundle", passing],
      replay.io,
    ),
    0,
  );
  assert.match(replay.stdout(), /1\/1/u);

  const failed = capture(root);
  assert.equal(
    await runCli(
      ["test", "--artifact", artifactRoot, "--bundle", failing],
      failed.io,
    ),
    1,
  );
  assert.match(failed.stdout(), /1\/2/u);
  assert.match(
    failed.stdout(),
    new RegExp(`example 1: expected "${wrong}", got "${expected}"`, "u"),
  );
  assert.match(failed.stdout(), /nf_99999999…: absent from the artifact/u);
  assert.match(failed.stdout(), /test failed\n$/u);

  const json = capture(root);
  assert.equal(
    await runCli(
      ["test", "--artifact", artifactRoot, "--bundle", failing, "--json"],
      json.io,
    ),
    1,
  );
  const document = JSON.parse(json.stdout());
  assert.equal(document.ok, false);
  assert.equal(document.functions[0].status, "passed");
  assert.equal(document.functions[0].examples.passed, 1);
  assert.deepEqual(document.missingFunctions, [`nf_${"9".repeat(64)}`]);

  const unverified = join(root, "unverified");
  await createFixtureArtifact(unverified, {
    transformManifest: (manifest) => {
      manifest.functions[0].verification.status = "failed";
    },
  });
  const refused = capture(root);
  assert.equal(await runCli(["test", "--artifact", unverified], refused.io), 1);
  assert.match(refused.stdout(), /failed/u);
});

test("run loads the artifact, imports the module and calls an export with JSON input", async (t) => {
  const root = await scratch(t, "semantscript-cli-run-");
  const artifactRoot = join(root, "artifact");
  await createFixtureArtifact(artifactRoot);
  const modulePath = join(fixtures, "run-module.mjs");
  const { loadSemaArtifact } = await import("@semantscript/core");
  const handle = await loadSemaArtifact(artifactRoot);
  const expected = handle.call(fixtureFunctionId, { facts: { a: 1, b: 2 } });
  await handle.close();

  const positional = capture(root);
  assert.equal(
    await runCli(
      [
        "run",
        "--artifact",
        artifactRoot,
        modulePath,
        "--call",
        "decide",
        "--input",
        "[1, 2]",
      ],
      positional.io,
    ),
    0,
  );
  assert.equal(positional.stdout(), `${JSON.stringify(expected, null, 2)}\n`);

  const inputFile = join(root, "input.json");
  await writeFile(inputFile, JSON.stringify({ facts: { a: 1, b: 2 } }));
  const promised = capture(root);
  assert.equal(
    await runCli(
      [
        "run",
        "--artifact",
        artifactRoot,
        modulePath,
        "--call",
        "decideLater",
        "--input-file",
        inputFile,
      ],
      promised.io,
    ),
    0,
  );
  assert.equal(promised.stdout(), `${JSON.stringify(expected, null, 2)}\n`);

  const sideEffectsOnly = capture(root);
  assert.equal(
    await runCli(
      ["run", "--artifact", artifactRoot, modulePath],
      sideEffectsOnly.io,
    ),
    0,
  );
  assert.equal(sideEffectsOnly.stdout(), "");

  const notCallable = capture(root);
  assert.equal(
    await runCli(
      ["run", "--artifact", artifactRoot, modulePath, "--call", "notCallable"],
      notCallable.io,
    ),
    1,
  );
  assert.match(notCallable.stderr(), /no function export named notCallable/u);

  const badInput = capture(root);
  assert.equal(
    await runCli(
      [
        "run",
        "--artifact",
        artifactRoot,
        modulePath,
        "--call",
        "decide",
        "--input",
        "{",
      ],
      badInput.io,
    ),
    2,
  );
  assert.match(badInput.stderr(), /input must be JSON/u);

  // Nothing stays loaded once run returns: a bare call must fail closed.
  const { decide } = await import(modulePath);
  assert.throws(() => decide(1, 2), /no SemantScript artifact is loaded/u);
});

test("train spawns the Python driver with resolved paths and renders its report", async (t) => {
  const root = await scratch(t, "semantscript-cli-train-");
  await writeFile(join(root, "bundle.json"), "{}");
  await writeFile(join(root, "teacher.toml"), "[teacher]\n");
  const argvPath = join(root, "argv.json");
  const env = {
    PYTHONPATH: fixtures,
    FAKE_TRAINER_ARGV_PATH: argvPath,
    FAKE_TRAINER_EXIT: "0",
    FAKE_TRAINER_SKIP_REPORT: "",
  };
  const args = [
    "train",
    "--bundle",
    "bundle.json",
    "--artifact",
    "out/artifact",
    "--teacher",
    "teacher.toml",
    "--python",
    process.platform === "win32" ? "python" : "python3",
    "--trainer-module",
    "fake_trainer",
    "--cases",
    "16",
    "--epochs",
    "2",
    "--select-best-epoch",
    "--application-id",
    "demo",
    "--full",
  ];

  const passed = capture(root, env);
  assert.equal(await runCli(args, passed.io), 0, passed.stderr());
  const recorded = JSON.parse(await readFile(argvPath, "utf8"));
  assert.deepEqual(recorded.argv.slice(0, 3), [
    "train",
    "--bundle",
    join(root, "bundle.json"),
  ]);
  assert.ok(
    recorded.argv.includes("--report") &&
      recorded.argv.includes(join(root, "out/artifact.report.json")),
  );
  assert.ok(
    recorded.argv.includes("--cache-dir") &&
      recorded.argv.includes(join(root, ".semantscript/cache")),
  );
  for (const expectedPair of [
    ["--cases", "16"],
    ["--epochs", "2"],
    ["--application-id", "demo"],
  ]) {
    const index = recorded.argv.indexOf(expectedPair[0]);
    assert.equal(recorded.argv[index + 1], expectedPair[1]);
  }
  assert.ok(recorded.argv.includes("--select-best-epoch"));
  assert.ok(recorded.argv.includes("--full"));
  assert.ok(!recorded.argv.includes("--local-files-only"));
  assert.ok(!recorded.argv.includes("--no-cache"));
  assert.match(recorded.pythonpath, /trainer[\\/]src/u);
  assert.ok(
    recorded.pythonpath.endsWith(fixtures),
    "the caller's PYTHONPATH is kept last",
  );
  assert.match(
    passed.stdout(),
    /nf_33333333…\s+src\/app\.sem\.ts\s+trained\s+16\s+4\s+0\.8750\s+passed\s+0\.9000\s+0\.0500\s+1\s+0/u,
  );
  assert.match(
    passed.stdout(),
    /build cache: 1 reused, 1 trained \(.*\.semantscript\/cache\)/u,
  );

  const uncached = capture(root, env);
  assert.equal(
    await runCli([...args, "--no-cache"], uncached.io),
    0,
    uncached.stderr(),
  );
  assert.match(uncached.stdout(), /build cache: 0 reused, 1 trained \(off\)/u);
  assert.ok(
    JSON.parse(await readFile(argvPath, "utf8")).argv.includes("--no-cache"),
  );
  assert.match(passed.stdout(), /artifact: .*out\/artifact \(release 2{64}\)/u);
  assert.match(passed.stdout(), /train passed\n$/u);

  const failed = capture(root, { ...env, FAKE_TRAINER_EXIT: "3" });
  assert.equal(await runCli(args, failed.io), 3);
  assert.match(failed.stdout(), /failed\s+0\.9000/u);
  assert.match(
    failed.stdout(),
    /verification failures:\n {2}injected failure\n/u,
  );
  assert.match(failed.stdout(), /train failed\n$/u);

  const silent = capture(root, { ...env, FAKE_TRAINER_SKIP_REPORT: "1" });
  await rm(join(root, "out"), { recursive: true, force: true });
  assert.equal(await runCli(args, silent.io), 1);
  assert.match(silent.stderr(), /wrote no report/u);

  const absent = capture(root, env);
  assert.equal(
    await runCli(
      [...args.slice(0, 7), "--python", join(root, "no-such-python")],
      absent.io,
    ),
    1,
  );
  assert.match(absent.stderr(), /unable to run/u);
});

test("helpers canonicalize JSON, extend PYTHONPATH and render reports", () => {
  assert.equal(
    canonical({ b: 1, a: [{ d: -0, c: "x" }] }),
    '{"a":[{"c":"x","d":0}],"b":1}',
  );
  assert.equal(canonical(undefined), "undefined");
  const path = pythonPath("extra");
  assert.ok(path.endsWith("extra"));
  assert.match(path, /model[\\/]src/u);
  assert.throws(
    () => renderTrainReport({ status: "passed" }),
    /report\.functions must be an array/u,
  );
});
