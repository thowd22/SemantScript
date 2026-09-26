import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import {
  chmod,
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import process from "node:process";
import test from "node:test";
import { setTimeout } from "node:timers";
import { fileURLToPath, pathToFileURL } from "node:url";

import {
  createFixtureArtifact,
  fixtureFunctionId,
} from "../../runtime/test/fixtures/artifact.mjs";
import {
  canonical,
  checkNode,
  classifyTrainerFailure,
  checkRuntimeBindings,
  pythonPath,
  renderChecks,
  renderTrainReport,
  runCli,
  TrainerStderr,
  trainerDoctorCommand,
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
  const commandHelp = capture(process.cwd());
  assert.equal(await runCli(["doctor", "--help"], commandHelp.io), 0);
  assert.equal(commandHelp.stdout(), USAGE);
  assert.equal(await runCli(["train", "-h"], capture(process.cwd()).io), 0);
  const rootPackage = JSON.parse(
    readFileSync(new URL("../../package.json", import.meta.url), "utf8"),
  );
  for (const flag of ["--version", "-v", "version"]) {
    const version = capture(process.cwd());
    assert.equal(await runCli([flag], version.io), 0);
    assert.equal(version.stdout(), `semantscript ${rootPackage.version}\n`);
  }
  assert.match(
    renderChecks([
      { id: "device", status: "warn", summary: "cpu", fix: null },
      { id: "node", status: "pass", summary: "ok", fix: null },
    ]),
    /1 passed, 1 warning, 0 failed, 0 skipped\n$/u,
  );
  const bogus = capture(process.cwd());
  assert.equal(await runCli(["bogus"], bogus.io), 2);
  assert.match(bogus.stderr(), /unknown command bogus/u);
  const missing = capture(process.cwd());
  assert.equal(await runCli(["train"], missing.io), 2);
  assert.match(
    missing.stderr(),
    /--bundle is required: no semantscript\.ir\.v1\.json under/u,
  );
  const noArtifact = capture(process.cwd());
  assert.equal(await runCli(["test"], noArtifact.io), 1);
  assert.match(noArtifact.stderr(), /\.semantscript\/artifact/u);
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
  assert.match(
    missingOutDir.stderr(),
    /outDir; next: set compilerOptions\.outDir in .*no-outdir\.json \(for example dist\)\n$/u,
  );

  const noProject = capture(root);
  assert.equal(
    await runCli(
      ["build", "--project", join(root, "missing", "tsconfig.json")],
      noProject.io,
    ),
    1,
  );
  assert.match(
    noProject.stderr(),
    /next: create \S*missing[\\/]tsconfig\.json with npx -p typescript tsc --init --rootDir . --outDir dist, then run semantscript init to add the SemantScript plugins, or pass --project <tsconfig\.json>\n$/u,
  );
});

test("build rewrites sema sites when ts-patch has patched typescript and tsconfig lists the transformer", async (t) => {
  // The tsc setup `init` writes: a `plugins` transform entry plus `ts-patch
  // install`. The patched emit must not apply the transformer a second time,
  // or build fails with "planned 1 sema rewrites ... but matched 0". The
  // loader hook stands in for the patched install: it hands every importer of
  // `typescript` the ts-patch live compiler, which honors `plugins`.
  const root = await scratch(t, "semantscript-cli-build-patched-");
  const configPath = await createProject(root, { "app.sem.ts": program });
  const config = JSON.parse(await readFile(configPath, "utf8"));
  config.compilerOptions.plugins = [
    { transform: "@semantscript/compiler/transformer" },
  ];
  await writeFile(configPath, JSON.stringify(config));
  await symlink(
    resolve(here, "..", "..", "compiler"),
    join(root, "node_modules", "@semantscript", "compiler"),
    "junction",
  );
  const hook = join(root, "patched-typescript.mjs");
  await writeFile(
    hook,
    [
      'import { register } from "node:module";',
      "const source = `export async function resolve(specifier, context, next) {",
      '  return next(specifier === "typescript" ? "ts-patch/compiler" : specifier, context);',
      "}`;",
      "register(`data:text/javascript,${encodeURIComponent(source)}`, import.meta.url);",
      "",
    ].join("\n"),
  );
  const result = spawnSync(
    process.execPath,
    [
      "--import",
      pathToFileURL(hook).href,
      join(here, "..", "bin", "semantscript.js"),
      "build",
      "--project",
      configPath,
    ],
    { cwd: root, encoding: "utf8" },
  );
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /compiled 1 neural function\(s\)/u);
  const emitted = await readFile(join(root, "dist", "app.sem.js"), "utf8");
  assert.match(emitted, /__sema\.call\("nf_[a-f0-9]{64}"/u);
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
  assert.equal(
    await runCli(["test", "--artifact", artifactRoot, "--no-bundle"], stats.io),
    0,
  );
  assert.match(
    stats.stdout(),
    /nf_11111111…\s+passed\s+1\.0000\s+0\.0000\s+0\.0000\s+1\.0000\s+1\s+0\s+-\s+-\s+1\.0000\s+-/u,
  );
  assert.match(stats.stdout(), /\s+violations\s+held-out\s+seed\s+/u);
  assert.match(stats.stdout(), /test passed\n$/u);

  const seededRoot = join(root, "seeded");
  await createFixtureArtifact(seededRoot, {
    transformManifest: (manifest) => {
      manifest.functions[0].trainingProvenance.seed = 4;
      manifest.functions[0].verification.heldOutConstraints = {
        sampleSize: 512,
        violations: 3,
        violationRate: 3 / 512,
        seed: 4,
      };
    },
  });
  const seeded = capture(root);
  assert.equal(
    await runCli(["test", "--artifact", seededRoot, "--no-bundle"], seeded.io),
    0,
  );
  assert.match(seeded.stdout(), /\s+seed\s+/u);
  // The held-out constraint figure: 3 of 512 sampled inputs broke a constraint.
  assert.match(seeded.stdout(), /\s1\s+0\s+3\/512\s+4\s+1\.0000\s+-/u);
  const seededJson = capture(root);
  assert.equal(
    await runCli(
      ["test", "--artifact", seededRoot, "--no-bundle", "--json"],
      seededJson.io,
    ),
    0,
  );
  assert.equal(JSON.parse(seededJson.stdout()).functions[0].seed, 4);

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
  assert.match(
    failed.stdout(),
    /\nnext: run semantscript train on this bundle \(semantscript build first if the source changed\)\nnext: run semantscript build, then semantscript train, and rerun semantscript test/u,
  );
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
  assert.equal(document.functions[0].seed, null);
  assert.deepEqual(document.missingFunctions, [`nf_${"9".repeat(64)}`]);
  assert.equal(document.next.length, 2);
  assert.match(
    document.next[1],
    /^run semantscript build, then semantscript train/u,
  );

  const nothing = capture(root);
  const empty = join(root, "empty-artifact");
  await mkdir(empty);
  assert.equal(await runCli(["test", "--artifact", empty], nothing.io), 1);
  assert.equal(
    nothing.stderr(),
    `no artifact at ${empty} (current.json is missing); next: run semantscript train to publish an artifact at ${empty}, or pass --artifact for one published elsewhere\n`,
  );

  const unverified = join(root, "unverified");
  await createFixtureArtifact(unverified, {
    transformManifest: (manifest) => {
      manifest.functions[0].verification.status = "failed";
    },
  });
  const refused = capture(root);
  assert.equal(
    await runCli(["test", "--artifact", unverified, "--no-bundle"], refused.io),
    1,
  );
  assert.match(
    refused.stdout(),
    /\nnext: retrain with semantscript train, following the next: lines of its report, or switch to a passing release with semantscript releases rollback <release> \(semantscript releases list shows which releases pass\)\ntest failed\n$/u,
  );

  // A pointer or release that does not read names the same fix run prints.
  const corrupt = join(root, "corrupt");
  await createFixtureArtifact(corrupt);
  await writeFile(join(corrupt, "current.json"), '{"x":1}\n');
  const broken = capture(root);
  assert.equal(await runCli(["test", "--artifact", corrupt], broken.io), 1);
  assert.match(
    broken.stderr(),
    /; next: run semantscript releases list to find an intact release, then switch to it with semantscript releases rollback <release>, [^\n]*; or publish a new one with semantscript train\n$/u,
  );
  const dangling = join(root, "dangling");
  await createFixtureArtifact(dangling);
  await rm(join(dangling, "releases"), { recursive: true, force: true });
  const gone = capture(root);
  assert.equal(await runCli(["test", "--artifact", dangling], gone.io), 1);
  assert.match(
    gone.stderr(),
    /fails the runtime's load checks: SEMA_ARTIFACT_PATH: [^\n]*; next: run semantscript releases list to find an intact release/u,
  );
});

test("test checks the artifact against the build's bundle and the release's digests by default", async (t) => {
  const root = await scratch(t, "semantscript-cli-test-default-");
  const artifactRoot = join(root, ".semantscript", "artifact");
  await createFixtureArtifact(artifactRoot);
  const { loadSemaArtifact } = await import("@semantscript/core");
  const handle = await loadSemaArtifact(artifactRoot);
  const inputs = { facts: { a: 1, b: 2 } };
  const expected = handle.call(fixtureFunctionId, inputs);
  await handle.close();
  const bundleFor = (id) =>
    JSON.stringify({
      kind: "semantscript.ir-bundle",
      bundleVersion: 1,
      functions: [
        { id, definition: { examples: [{ inputs, output: expected }] } },
      ],
      executionPlan: { stages: [], dependencies: [] },
    });

  // No build output yet: test names semantscript build.
  const noBuild = capture(root);
  assert.equal(await runCli(["test"], noBuild.io), 1);
  assert.match(
    noBuild.stderr(),
    /^no semantscript\.ir\.v1\.json under [^\n]*; next: run semantscript build, then rerun semantscript test; or pass --bundle <path> for a bundle elsewhere, or --no-bundle to check the artifact alone\n$/u,
  );
  const artifactOnly = capture(root);
  assert.equal(await runCli(["test", "--no-bundle"], artifactOnly.io), 0);
  assert.match(artifactOnly.stdout(), /\s-\ntest passed\n$/u);
  assert.doesNotMatch(artifactOnly.stdout(), /\nbundle /u);
  const both = capture(root);
  assert.equal(
    await runCli(["test", "--bundle", "x.json", "--no-bundle"], both.io),
    2,
  );
  assert.match(both.stderr(), /--bundle and --no-bundle cannot be combined/u);

  // The build's bundle comes from the tsconfig outDir, as train and explain find it.
  await writeFile(
    join(root, "tsconfig.json"),
    JSON.stringify({ compilerOptions: { outDir: "build-output" } }),
  );
  const bundlePath = join(root, "build-output", "semantscript.ir.v1.json");
  await mkdir(dirname(bundlePath));
  await writeFile(bundlePath, bundleFor(fixtureFunctionId));
  const current = capture(root);
  assert.equal(await runCli(["test"], current.io), 0, current.stderr());
  assert.match(current.stdout(), new RegExp(`\\nbundle ${bundlePath}\\n`, "u"));
  assert.match(current.stdout(), /1\/1\ntest passed\n$/u);

  // The program changed since training: the ids no longer match either way.
  const changedId = `nf_${"8".repeat(64)}`;
  await writeFile(bundlePath, bundleFor(changedId));
  const stale = capture(root);
  assert.equal(await runCli(["test"], stale.io), 1);
  assert.match(stale.stdout(), /nf_88888888…: absent from the artifact/u);
  assert.match(
    stale.stdout(),
    /nf_11111111…: in the artifact but not in the bundle/u,
  );
  assert.match(stale.stdout(), /\ntest failed\n$/u);
  // Missing both ways is one fix, rebuild and retrain: one next: line.
  assert.match(
    stale.stdout(),
    /\nnext: the program changed since training: run semantscript build, then semantscript train, and rerun semantscript test; or pass --bundle for the bundle this artifact was trained from\ntest failed\n$/u,
  );
  const staleJson = capture(root);
  assert.equal(await runCli(["test", "--json"], staleJson.io), 1);
  const document = JSON.parse(staleJson.stdout());
  assert.equal(document.ok, false);
  assert.equal(document.bundle, bundlePath);
  assert.deepEqual(document.missingFunctions, [changedId]);
  assert.deepEqual(document.unbundledFunctions, [fixtureFunctionId]);
  assert.equal(document.next.length, 1);
  const skipped = capture(root);
  assert.equal(await runCli(["test", "--no-bundle", "--json"], skipped.io), 0);
  const skippedDocument = JSON.parse(skipped.stdout());
  assert.equal(skippedDocument.bundle, null);
  assert.deepEqual(skippedDocument.unbundledFunctions, []);

  // A release that fails the runtime's digest or symlink checks fails test
  // with the runtime's ArtifactLoadError code and remedy, bundle or not.
  const corruptFix =
    "run semantscript releases list to find an intact release, then switch to it with semantscript releases rollback <release>";
  const tampered = join(root, "tampered");
  const { release } = await createFixtureArtifact(tampered);
  const tokenizer = join(release, "tokenizer", "tokenizer.json");
  const bytes = await readFile(tokenizer);
  bytes[bytes.length - 1] = bytes[bytes.length - 1] === 0x20 ? 0x0a : 0x20;
  await writeFile(tokenizer, bytes);
  for (const flag of [[], ["--no-bundle"]]) {
    const run = capture(root);
    assert.equal(
      await runCli(["test", "--artifact", tampered, ...flag], run.io),
      1,
    );
    assert.equal(run.stdout(), "");
    assert.match(
      run.stderr(),
      new RegExp(
        `^artifact at ${tampered} fails the runtime's load checks: SEMA_ARTIFACT_INTEGRITY: [^\\n]*; next: ${corruptFix}`,
        "u",
      ),
    );
  }

  const edited = join(root, "edited");
  const editedRelease = (await createFixtureArtifact(edited)).release;
  const manifestPath = join(editedRelease, "manifest.json");
  await writeFile(manifestPath, `${await readFile(manifestPath, "utf8")} \n`);
  const manifestRun = capture(root);
  assert.equal(await runCli(["test", "--artifact", edited], manifestRun.io), 1);
  assert.match(
    manifestRun.stderr(),
    new RegExp(
      `fails the runtime's load checks: SEMA_ARTIFACT_INTEGRITY: [^\\n]*manifest[^\\n]*; next: ${corruptFix}`,
      "u",
    ),
  );

  const linked = join(root, "linked");
  await createFixtureArtifact(linked);
  await copyFile(
    join(linked, "current.json"),
    join(root, "elsewhere-current.json"),
  );
  await rm(join(linked, "current.json"));
  await symlink(
    join(root, "elsewhere-current.json"),
    join(linked, "current.json"),
  );
  const linkRun = capture(root);
  assert.equal(
    await runCli(["test", "--artifact", linked, "--no-bundle"], linkRun.io),
    1,
  );
  assert.match(
    linkRun.stderr(),
    /fails the runtime's load checks: SEMA_ARTIFACT_PATH: [^\n]*; next: point at the artifact root semantscript train published/u,
  );

  // A passing release whose manifest was hand-edited to read "failed" is
  // tampering, not an honest unverified release: the manifest digest fails.
  const flipped = join(root, "flipped");
  const flippedManifest = join(
    (await createFixtureArtifact(flipped)).release,
    "manifest.json",
  );
  const text = await readFile(flippedManifest, "utf8");
  assert.match(text, /"status": "passed"/u);
  await writeFile(
    flippedManifest,
    text.replace('"status": "passed"', '"status": "failed"'),
  );
  const flippedRun = capture(root);
  assert.equal(
    await runCli(["test", "--artifact", flipped, "--no-bundle"], flippedRun.io),
    1,
  );
  assert.equal(flippedRun.stdout(), "");
  assert.match(
    flippedRun.stderr(),
    new RegExp(
      `fails the runtime's load checks: SEMA_ARTIFACT_INTEGRITY: manifest digest [^\\n]*; next: ${corruptFix}`,
      "u",
    ),
  );

  // A manifest that no longer parses, and a pointer symlinked to nothing,
  // report the runtime's code rather than a bare read error.
  const garbled = join(root, "garbled");
  await writeFile(
    join((await createFixtureArtifact(garbled)).release, "manifest.json"),
    "garbage",
  );
  const garbledRun = capture(root);
  assert.equal(
    await runCli(["test", "--artifact", garbled, "--no-bundle"], garbledRun.io),
    1,
  );
  assert.match(
    garbledRun.stderr(),
    /^artifact at [^\n]* fails the runtime's load checks: SEMA_ARTIFACT_[A-Z_]+: /u,
  );
  const nowhere = join(root, "nowhere");
  await createFixtureArtifact(nowhere);
  await rm(join(nowhere, "current.json"));
  await symlink(
    join(root, "missing-current.json"),
    join(nowhere, "current.json"),
  );
  const nowhereRun = capture(root);
  assert.equal(
    await runCli(["test", "--artifact", nowhere, "--no-bundle"], nowhereRun.io),
    1,
  );
  assert.match(
    nowhereRun.stderr(),
    /fails the runtime's load checks: SEMA_ARTIFACT_PATH: /u,
  );

  // A release manifest, or a release directory, symlinked to nothing is a
  // path problem, not a missing file: it reports the runtime's code too.
  const deadManifest = join(root, "dead-manifest");
  const deadManifestRelease = (await createFixtureArtifact(deadManifest))
    .release;
  await rm(join(deadManifestRelease, "manifest.json"));
  await symlink(
    join(root, "missing-manifest.json"),
    join(deadManifestRelease, "manifest.json"),
  );
  const deadManifestRun = capture(root);
  assert.equal(
    await runCli(
      ["test", "--artifact", deadManifest, "--no-bundle"],
      deadManifestRun.io,
    ),
    1,
  );
  assert.match(
    deadManifestRun.stderr(),
    /fails the runtime's load checks: SEMA_ARTIFACT_PATH: /u,
  );
  const deadRelease = join(root, "dead-release");
  const deadReleaseDir = (await createFixtureArtifact(deadRelease)).release;
  await rm(deadReleaseDir, { recursive: true, force: true });
  await symlink(join(root, "missing-release"), deadReleaseDir);
  const deadReleaseRun = capture(root);
  assert.equal(
    await runCli(
      ["test", "--artifact", deadRelease, "--no-bundle"],
      deadReleaseRun.io,
    ),
    1,
  );
  assert.match(
    deadReleaseRun.stderr(),
    /fails the runtime's load checks: SEMA_ARTIFACT_PATH: /u,
  );

  // A pointer that names a release that does not exist, with digests that
  // disagree, a consistent pointer to a deleted release, and a deleted
  // manifest all report the runtime's code; the missing files keep the
  // rollback fix.
  const rollbackFix =
    /; next: run semantscript releases list to find an intact release, then switch to it with semantscript releases rollback <release>, /u;
  const ghost = join(root, "ghost");
  await createFixtureArtifact(ghost);
  const ghostPointer = JSON.parse(
    await readFile(join(ghost, "current.json"), "utf8"),
  );
  ghostPointer.release = `releases/sha256-${"b".repeat(64)}`;
  await writeFile(join(ghost, "current.json"), JSON.stringify(ghostPointer));
  const gone = join(root, "gone");
  await rm((await createFixtureArtifact(gone)).release, {
    recursive: true,
    force: true,
  });
  const bare = join(root, "bare");
  await rm(join((await createFixtureArtifact(bare)).release, "manifest.json"));
  for (const [artifact, code] of [
    [ghost, "SEMA_ARTIFACT_INVALID_POINTER: pointer digests differ"],
    [gone, "SEMA_ARTIFACT_PATH: cannot inspect release directory"],
    [bare, "SEMA_ARTIFACT_PATH: cannot inspect manifest"],
  ]) {
    const run = capture(root);
    assert.equal(
      await runCli(["test", "--artifact", artifact, "--no-bundle"], run.io),
      1,
    );
    assert.match(
      run.stderr(),
      new RegExp(
        `^artifact at [^\\n]* fails the runtime's load checks: ${code}; `,
        "u",
      ),
    );
    assert.match(run.stderr(), rollbackFix);
  }

  // A bundle path that does not exist, or a file that is not an IR bundle,
  // fails with a next: line instead of a bare ENOENT or a wrong comparison.
  const missingBundle = capture(root);
  assert.equal(
    await runCli(["test", "--bundle", "nope.json"], missingBundle.io),
    1,
  );
  assert.match(
    missingBundle.stderr(),
    /^cannot read the bundle [^\n]*nope\.json: [^\n]*; next: run semantscript build/u,
  );
  await writeFile(
    join(root, "other.json"),
    JSON.stringify({ kind: "other", functions: [] }),
  );
  const otherBundle = capture(root);
  assert.equal(
    await runCli(["test", "--bundle", "other.json"], otherBundle.io),
    1,
  );
  assert.match(
    otherBundle.stderr(),
    /other\.json is not a semantscript\.ir-bundle; next: run semantscript build/u,
  );
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
  assert.match(
    notCallable.stderr(),
    /; next: pass --call one of the module's function exports: decide, decideLater\n$/u,
  );

  const noArtifact = capture(root);
  assert.equal(
    await runCli(
      ["run", "--artifact", join(root, "absent"), modulePath],
      noArtifact.io,
    ),
    1,
  );
  assert.match(
    noArtifact.stderr(),
    /cannot inspect artifact directory; next: run semantscript train to publish an artifact at /u,
  );

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
    "--held-out-samples",
    "256",
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
    ["--held-out-samples", "256"],
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
    /nf_33333333…\s+src\/app\.sem\.ts\s+trained\s+16\s+4\s+0\.8750\s+passed\s+0\.9000\s+0\.0500\s+1\s+0\s+0\/512\n/u,
  );
  assert.match(passed.stdout(), /\s+violations\s+held-out constraints\n/u);
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
  assert.match(failed.stdout(), /\s+1\s+0\s+3\/512 \(0\.59%\)\n/u);
  assert.match(
    failed.stdout(),
    /verification failures:\n {2}injected failure\n {4}next: rerun with --epochs 5 \(now 3\)\n/u,
  );
  assert.match(failed.stdout(), /train failed\n$/u);

  // A report the trainer did not rewrite is an earlier run's: never rendered.
  const stale = capture(root, {
    ...env,
    FAKE_TRAINER_SKIP_REPORT: "1",
    FAKE_TRAINER_EXIT: "1",
  });
  assert.equal(await runCli(args, stale.io), 1);
  assert.equal(stale.stdout(), "", "the earlier run's report is not rendered");
  const staleZero = capture(root, { ...env, FAKE_TRAINER_SKIP_REPORT: "1" });
  assert.equal(await runCli(args, staleZero.io), 1);
  assert.match(staleZero.stderr(), /wrote no report/u);
  assert.equal(staleZero.stdout(), "");

  // A traceback the trainer went on from is forwarded, not treated as fatal.
  for (const exit of ["0", "3"]) {
    const noisy = capture(root, {
      ...env,
      FAKE_TRAINER_NOISE: "1",
      FAKE_TRAINER_EXIT: exit,
    });
    assert.equal(await runCli(args, noisy.io), Number(exit), exit);
    assert.match(
      noisy.stderr(),
      /Traceback \(most recent call last\):\n {2}File "lib\.py", line 3, in load\nValueError: optional backend unavailable\nprogress: continuing\n/u,
    );
    assert.match(
      noisy.stderr(),
      /Exception ignored in: <function Handle\.__del__ at 0x1>\nTraceback \(most recent call last\):\n {2}File "h\.py", line 9, in __del__\nOSError: handle closed\n/u,
    );
    assert.doesNotMatch(noisy.stderr(), /the trainer stopped/u);
    assert.match(
      noisy.stdout(),
      exit === "0" ? /train passed\n$/u : /next: rerun with --epochs 5/u,
    );
  }

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

  const noInterpreter = capture(root, env);
  assert.equal(
    await runCli(
      [
        ...args.slice(0, 7),
        "--no-preflight",
        "--python",
        join(root, "no-such-python"),
      ],
      noInterpreter.io,
    ),
    1,
  );
  assert.match(
    noInterpreter.stderr(),
    /unable to run .*no-such-python: .*ENOENT; next: run semantscript doctor --python \S*no-such-python --teacher teacher\.toml and fix its python check: install Python 3\.12 or later/u,
  );
});

test("train wraps a trainer traceback into one line with the doctor check to run", async (t) => {
  const root = await scratch(t, "semantscript-cli-train-crash-");
  await writeFile(join(root, "bundle.json"), "{}");
  await writeFile(join(root, "teacher.toml"), "[teacher]\n");
  const python = process.platform === "win32" ? "python" : "python3";
  const args = [
    "train",
    "--bundle",
    "bundle.json",
    "--artifact",
    "out/artifact",
    "--teacher",
    "teacher.toml",
    "--python",
    python,
    "--trainer-module",
    "fake_trainer",
    "--no-preflight",
  ];
  const traceback = join(root, "out", "artifact.report.traceback.txt");
  const doctor = `semantscript doctor --python ${python} --trainer-module fake_trainer --teacher teacher.toml`;
  const cases = [
    [
      "module:torch",
      "ModuleNotFoundError: No module named 'torch'",
      `run ${doctor} and fix its torch check: ${python} cannot import torch`,
    ],
    [
      "module:semantscript_model.heads",
      "ModuleNotFoundError: No module named 'semantscript_model.heads'",
      `run ${doctor} and fix its model check: ${python} cannot import semantscript_model.heads`,
    ],
    [
      "module:onnx",
      "ModuleNotFoundError: No module named 'onnx'",
      `run ${doctor} and fix its onnxruntime check: ${python} cannot import onnx`,
    ],
    [
      // This interpreter is 3.12 or later, so the trainer's own file is broken.
      "syntax",
      "SyntaxError: invalid syntax",
      `run ${doctor} to check the environment; if every check passes, rerun with --no-cache`,
    ],
    [
      "crash",
      "RuntimeError: the fake trainer crashed",
      `run ${doctor} to check the environment; if every check passes, rerun with --no-cache`,
    ],
  ];
  for (const [mode, exception, fix] of cases) {
    await rm(join(root, "out"), { recursive: true, force: true });
    const run = capture(root, {
      PYTHONPATH: fixtures,
      FAKE_TRAINER_RAISE: mode,
    });
    assert.equal(await runCli(args, run.io), 1, mode);
    const stderr = run.stderr();
    assert.match(stderr, /generating 64 cases \(1 gold\)\n/u, mode);
    assert.doesNotMatch(stderr, /Traceback|File "/u, mode);
    const line = stderr.split("\n").at(-2);
    assert.ok(
      line.startsWith(
        `semantscript train: the trainer stopped: ${exception}; next: ${fix}`,
      ),
      `${mode}: ${line}`,
    );
    assert.ok(line.endsWith(`; full traceback in ${traceback}`), line);
    assert.match(
      await readFile(traceback, "utf8"),
      /^Traceback \(most recent call last\):\n[\s\S]*fake_trainer\.py/u,
    );
    assert.equal(run.stdout(), "", "no report is rendered");
  }

  const launch = capture(root, {
    PYTHONPATH: fixtures,
    FAKE_TRAINER_RAISE: "launch",
  });
  assert.equal(await runCli(args, launch.io), 1);
  assert.match(
    launch.stderr(),
    new RegExp(
      `the trainer stopped: ModuleNotFoundError: No module named 'semantscript_trainer\\.cli'; next: run semantscript doctor --python ${python} --trainer-module fake_trainer --teacher teacher\\.toml and fix its trainer check: ${python} cannot import semantscript_trainer\\.cli`,
      "u",
    ),
  );

  const estimate = capture(root, {
    PYTHONPATH: fixtures,
    FAKE_TRAINER_RAISE: "module:onnxruntime",
  });
  assert.equal(await runCli([...args, "--estimate"], estimate.io), 1);
  assert.match(estimate.stderr(), /generating 64 cases/u);
  assert.doesNotMatch(estimate.stderr(), /Traceback/u);
  assert.match(
    estimate.stderr(),
    /the trainer stopped: ModuleNotFoundError: No module named 'onnxruntime'; next: run semantscript doctor --python \S+ --trainer-module fake_trainer --teacher teacher\.toml and fix its onnxruntime check/u,
  );
});

test("the trainer's stderr is forwarded line by line until a traceback starts", () => {
  const forwarded = [];
  const filter = new TrainerStderr((text) => forwarded.push(text));
  filter.push("first li");
  assert.deepEqual(forwarded, []);
  filter.push("ne\nsecond line\nTraceback (most recent call last):\n  File");
  filter.push(' "x.py", line 1\nValueError: bad');
  filter.end();
  assert.deepEqual(forwarded, ["first line\n", "second line\n"]);
  assert.equal(
    filter.traceback,
    'Traceback (most recent call last):\n  File "x.py", line 1\nValueError: bad\n',
  );
  assert.deepEqual(classifyTrainerFailure(filter.traceback), {
    exception: "ValueError: bad",
    remedy: "trainer-crash",
    check: "trainer",
  });
  assert.equal(
    classifyTrainerFailure(
      "Error while finding module specification for 'semantscript_trainer.cli' (ModuleNotFoundError: No module named 'semantscript_trainer')\n",
    ).check,
    "trainer",
  );
  assert.equal(
    classifyTrainerFailure(
      "Traceback (most recent call last):\nModuleNotFoundError: No module named 'transformers.models'\n",
    ).check,
    "torch",
  );
});

test("the trainer's stderr filter releases a traceback the trainer went on from", () => {
  const forwarded = [];
  const filter = new TrainerStderr((text) => forwarded.push(text));
  // A progress bar redraws with carriage returns: each redraw goes out at once.
  filter.push("Loading weights:   0%|\rLoading weights: 100%|");
  assert.deepEqual(forwarded, ["Loading weights:   0%|\r"]);
  filter.push("\r\n");
  assert.deepEqual(forwarded.slice(1), ["Loading weights: 100%|\r\n"]);
  forwarded.length = 0;
  filter.push(
    'Traceback (most recent call last):\n  File "a.py", line 1\nValueError: first\n\nDuring handling of the above exception, another exception occurred:\n\nTraceback (most recent call last):\n  File "b.py", line 2\nKeyError: second\n',
  );
  assert.deepEqual(forwarded, [], "a traceback is held");
  filter.push("progress: step 2\n");
  assert.equal(forwarded.length, 10, "an ordinary line releases it in place");
  assert.equal(forwarded.at(-1), "progress: step 2\n");
  assert.equal(filter.traceback, undefined);
  // Released and no handled error followed: at a nonzero exit it still names the fix.
  assert.match(filter.fatalTraceback(), /KeyError: second\n$/u);
  filter.push(
    "error: bundle must be an object; next: run semantscript build\n",
  );
  assert.equal(
    filter.fatalTraceback(),
    undefined,
    "a handled error line after it is the failure",
  );
  assert.equal(
    forwarded.at(-1),
    "error: bundle must be an object; next: run semantscript build\n",
  );
  // The trainer's own error line naming a missing module is not a launch failure.
  filter.push("error: tokenizers: No module named 'tokenizers'; next: x\n");
  assert.equal(filter.traceback, undefined);
  filter.push("/usr/bin/python3: No module named semantscript_trainer.cli\n");
  assert.match(filter.traceback, /No module named semantscript_trainer\.cli/u);
  filter.release();
  assert.equal(
    forwarded.at(-1),
    "/usr/bin/python3: No module named semantscript_trainer.cli\n",
  );
});

test("the doctor command carries the interpreter and module train used", () => {
  assert.equal(
    trainerDoctorCommand({}, "semantscript_trainer.cli"),
    "semantscript doctor",
  );
  assert.equal(
    trainerDoctorCommand({ python: "/opt/py 3/bin/python" }, "my_trainer.cli"),
    'semantscript doctor --python "/opt/py 3/bin/python" --trainer-module my_trainer.cli',
  );
  assert.equal(
    trainerDoctorCommand({ teacher: "bad.toml" }, "semantscript_trainer.cli"),
    "semantscript doctor --teacher bad.toml",
  );
});

test("train names the signal and the fix when the trainer is killed", async (t) => {
  if (process.platform === "win32") {
    t.skip("POSIX signals");
    return;
  }
  const root = await scratch(t, "semantscript-cli-train-signal-");
  await writeFile(join(root, "bundle.json"), "{}");
  await writeFile(join(root, "teacher.toml"), "[teacher]\n");
  const args = [
    "train",
    "--bundle",
    "bundle.json",
    "--artifact",
    "out/artifact",
    "--teacher",
    "teacher.toml",
    "--python",
    "python3",
    "--trainer-module",
    "fake_trainer",
    "--no-preflight",
  ];
  const doctor =
    "semantscript doctor --python python3 --trainer-module fake_trainer --teacher teacher.toml";
  for (const [mode, signal, number] of [
    ["signal:SIGSEGV", "SIGSEGV", 11],
    ["signal:SIGKILL", "SIGKILL", 9],
    ["signal:SIGKILL:after-warning", "SIGKILL", 9],
  ]) {
    const run = capture(root, {
      PYTHONPATH: fixtures,
      FAKE_TRAINER_RAISE: mode,
    });
    assert.equal(await runCli(args, run.io), 128 + number, mode);
    const stderr = run.stderr();
    const line = stderr.split("\n").at(-2);
    assert.equal(
      line,
      `semantscript train: the trainer was killed by ${signal}; next: run ${doctor} and fix its torch and device checks, then rerun with a smaller --batch-size or with --device cpu: the datasets that finished stay cached, so the rerun asks the teacher only for the rest`,
      mode,
    );
    assert.doesNotMatch(stderr, /the trainer stopped/u, mode);
    if (mode.endsWith("after-warning")) {
      // The traceback it went on from is shown where it occurred, not blamed.
      assert.match(
        stderr,
        /ValueError: optional backend unavailable\ncontinuing without it\n/u,
      );
    }
  }
  // --estimate exits the same way.
  const estimate = capture(root, {
    PYTHONPATH: fixtures,
    FAKE_TRAINER_RAISE: "signal:SIGKILL",
  });
  assert.equal(await runCli([...args, "--estimate"], estimate.io), 128 + 9);
  // Its fix names no training-only flag: --estimate trains nothing.
  assert.equal(
    estimate.stderr().split("\n").at(-2),
    `semantscript train: the trainer was killed by SIGKILL; next: run ${doctor} and fix its checks, then rerun semantscript train --estimate (it trains nothing and sends no teacher request); if it is killed again, report it as a bug with the signal`,
  );
  assert.doesNotMatch(estimate.stderr(), /--batch-size|--device/u);
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

test("init wires a tsc project through ts-patch, keeps tsconfig comments and is idempotent", async (t) => {
  const root = await scratch(t, "semantscript-cli-init-tsc-");
  await writeFile(
    join(root, "package.json"),
    '{\n  "name": "app",\n  "type": "module",\n  "scripts": { "build": "tsc -p tsconfig.json", "prepare": "husky" }\n}\n',
  );
  await writeFile(
    join(root, "tsconfig.json"),
    '{\n  // strict project\n  "compilerOptions": {\n    "outDir": "dist",\n    "strict": true\n  },\n  "include": ["src"]\n}\n',
  );
  await mkdir(join(root, "src"));

  const first = capture(root);
  assert.equal(
    await runCli(["init", "--no-doctor"], first.io),
    0,
    first.stderr(),
  );
  assert.match(first.stdout(), /detected tsc/u);
  const tsconfig = await readFile(join(root, "tsconfig.json"), "utf8");
  assert.match(tsconfig, /\/\/ strict project/u);
  assert.match(
    tsconfig,
    /"compilerOptions": \{\n {4}"plugins": \[\{ "name": "@semantscript\/compiler\/ts-plugin" \}, \{ "transform": "@semantscript\/compiler\/transformer" \}\],\n {4}"outDir": "dist"/u,
  );
  const pkg = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
  assert.equal(pkg.scripts.build, "tsc -p tsconfig.json");
  assert.equal(pkg.scripts.prepare, "husky && ts-patch install");
  assert.ok("@semantscript/core" in pkg.dependencies);
  assert.ok("@semantscript/compiler" in pkg.devDependencies);
  assert.ok("ts-patch" in pkg.devDependencies);
  assert.match(
    await readFile(join(root, "src", "hello.sem.ts"), "utf8"),
    /sema<boolean>\(\{\s+examples: \[/u,
  );
  assert.match(
    await readFile(join(root, ".semantscript", ".gitignore"), "utf8"),
    /artifact\/\ncache\/\npackage\/\n\.package-staging-\*\/\n\*\.traceback\.txt\n/u,
  );

  const second = capture(root);
  assert.equal(
    await runCli(["init", "--no-doctor"], second.io),
    0,
    second.stderr(),
  );
  assert.equal((second.stdout().match(/^ {2}unchanged/gmu) ?? []).length, 5);
  assert.equal(await readFile(join(root, "tsconfig.json"), "utf8"), tsconfig);

  // A project initialised before `package` existed gets the new entries
  // appended (the train traceback included), and its own lines kept.
  await writeFile(
    join(root, ".semantscript", ".gitignore"),
    "# mine\nartifact/\ncache/",
  );
  const third = capture(root);
  assert.equal(
    await runCli(["init", "--no-doctor"], third.io),
    0,
    third.stderr(),
  );
  assert.equal(
    await readFile(join(root, ".semantscript", ".gitignore"), "utf8"),
    "# mine\nartifact/\ncache/\npackage/\n.package-staging-*/\n*.traceback.txt\n",
  );
  assert.match(third.stdout(), /also reserves \.semantscript\/package\//u);
  assert.match(third.stdout(), /\.semantscript\/\*\.traceback\.txt/u);
});

test("init wires Vite, Next.js and esbuild projects and leaves conflicting configs to the user", async (t) => {
  const vite = await scratch(t, "semantscript-cli-init-vite-");
  await writeFile(
    join(vite, "package.json"),
    '{ "name": "v", "devDependencies": { "vite": "8.3.1" } }\n',
  );
  await writeFile(
    join(vite, "vite.config.ts"),
    'import { defineConfig } from "vite";\nimport react from "@vitejs/plugin-react";\n\nexport default defineConfig({\n  plugins: [react()],\n});\n',
  );
  const viteRun = capture(vite);
  assert.equal(
    await runCli(["init", "--no-example", "--no-doctor"], viteRun.io),
    0,
    viteRun.stderr(),
  );
  assert.match(viteRun.stdout(), /detected vite/u);
  assert.match(
    viteRun.stdout(),
    /manual {5}tsconfig\.json is missing; create one with/u,
  );
  const viteConfig = await readFile(join(vite, "vite.config.ts"), "utf8");
  assert.match(
    viteConfig,
    /plugin-react";\nimport semantscript from "@semantscript\/compiler\/vite";\n/u,
  );
  assert.match(viteConfig, /plugins: \[semantscript\(\), react\(\)\]/u);
  assert.equal(existsSync(join(vite, "hello.sem.ts")), false);

  const next = await scratch(t, "semantscript-cli-init-next-");
  await writeFile(
    join(next, "package.json"),
    '{ "name": "n", "dependencies": { "next": "16.3.6" } }\n',
  );
  await writeFile(
    join(next, "next.config.ts"),
    'import type { NextConfig } from "next";\n\nconst nextConfig: NextConfig = {\n  reactStrictMode: true,\n};\n\nexport default nextConfig;\n',
  );
  const nextRun = capture(next);
  assert.equal(
    await runCli(["init", "--no-doctor"], nextRun.io),
    0,
    nextRun.stderr(),
  );
  assert.match(nextRun.stdout(), /detected next/u);
  const nextConfig = await readFile(join(next, "next.config.ts"), "utf8");
  assert.match(
    nextConfig,
    /const nextConfig: NextConfig = \{\n {2}\/\/ SemantScript/u,
  );
  assert.match(
    nextConfig,
    /"\*\.sem\.ts": \{ loaders: \["@semantscript\/compiler\/loader"\] \}/u,
  );
  assert.match(
    nextConfig,
    /serverExternalPackages: \["@semantscript\/core"\]/u,
  );
  assert.match(
    nextConfig,
    /outputFileTracingIncludes: \{ "\/\*\*": \["\.\/\.semantscript\/artifact\/\*\*"\] \}/u,
  );
  assert.match(nextConfig, /reactStrictMode: true,\n\};/u);
  assert.ok(existsSync(join(next, "lib", "hello.sem.ts")));

  const conflicting = await scratch(t, "semantscript-cli-init-next-conflict-");
  await writeFile(join(conflicting, "package.json"), '{ "name": "c" }\n');
  await writeFile(
    join(conflicting, "next.config.mjs"),
    "export default {\n  turbopack: { rules: {} },\n};\n",
  );
  const conflictRun = capture(conflicting);
  assert.equal(
    await runCli(
      ["init", "--tool", "next", "--no-example", "--no-doctor"],
      conflictRun.io,
    ),
    0,
  );
  assert.match(
    conflictRun.stdout(),
    /manual {5}next\.config\.mjs already sets turbopack/u,
  );
  assert.equal(
    await readFile(join(conflicting, "next.config.mjs"), "utf8"),
    "export default {\n  turbopack: { rules: {} },\n};\n",
  );

  const esbuild = await scratch(t, "semantscript-cli-init-esbuild-");
  await writeFile(
    join(esbuild, "package.json"),
    '{ "name": "e", "scripts": { "build": "node build.mjs" }, "devDependencies": { "esbuild": "0.28.2" } }\n',
  );
  await writeFile(
    join(esbuild, "build.mjs"),
    'import { build } from "esbuild";\n\nawait build({\n  entryPoints: ["src/main.ts"],\n  bundle: true,\n});\n',
  );
  const esbuildRun = capture(esbuild);
  assert.equal(
    await runCli(["init", "--no-example", "--no-doctor"], esbuildRun.io),
    0,
    esbuildRun.stderr(),
  );
  assert.match(esbuildRun.stdout(), /detected esbuild/u);
  const script = await readFile(join(esbuild, "build.mjs"), "utf8");
  assert.match(
    script,
    /import semantscript from "@semantscript\/compiler\/esbuild";/u,
  );
  assert.match(
    script,
    /await build\(\{\n {2}plugins: \[semantscript\(\)\],\n {2}entryPoints/u,
  );
});

test("init starts a TypeScript project in a directory that has only the package.json npm install wrote", async (t) => {
  const bare = await scratch(t, "semantscript-cli-init-bare-");
  await writeFile(
    join(bare, "package.json"),
    '{\n  "dependencies": {\n    "semantscript": "^0.1.0"\n  }\n}\n',
  );
  const bareRun = capture(bare);
  assert.equal(
    await runCli(["init", "--no-doctor"], bareRun.io),
    0,
    bareRun.stderr(),
  );
  assert.match(bareRun.stdout(), /started a TypeScript project built by tspc/u);
  assert.doesNotMatch(bareRun.stdout(), /manual/u);
  const tsconfig = JSON.parse(
    await readFile(join(bare, "tsconfig.json"), "utf8"),
  );
  assert.equal(tsconfig.compilerOptions.module, "NodeNext");
  assert.equal(tsconfig.compilerOptions.outDir, "dist");
  assert.deepEqual(tsconfig.include, ["src"]);
  assert.deepEqual(tsconfig.compilerOptions.plugins, [
    { name: "@semantscript/compiler/ts-plugin" },
    { transform: "@semantscript/compiler/transformer" },
  ]);
  const pkg = JSON.parse(await readFile(join(bare, "package.json"), "utf8"));
  assert.equal(pkg.type, "module");
  assert.equal(pkg.scripts.build, "tspc -p tsconfig.json");
  assert.equal(pkg.scripts.prepare, "ts-patch install");
  assert.equal(pkg.dependencies.semantscript, "^0.1.0");
  assert.ok("@semantscript/core" in pkg.dependencies);
  assert.ok("typescript" in pkg.devDependencies);
  assert.ok("ts-patch" in pkg.devDependencies);
  assert.ok(existsSync(join(bare, "src", "hello.sem.ts")));

  // A second run changes nothing.
  const again = capture(bare);
  assert.equal(await runCli(["init", "--no-doctor"], again.io), 0);
  assert.match(again.stdout(), /detected tsc/u);
  assert.doesNotMatch(again.stdout(), /^ {2}changed/mu);

  // npm init -y's package.json counts as new; an existing CommonJS entry keeps its module type.
  const npmInit = await scratch(t, "semantscript-cli-init-npm-init-");
  await writeFile(
    join(npmInit, "package.json"),
    JSON.stringify({
      name: "x",
      main: "index.js",
      scripts: { test: 'echo "Error: no test specified" && exit 1' },
    }),
  );
  assert.equal(await runCli(["init", "--no-doctor"], capture(npmInit).io), 0);
  assert.equal(
    JSON.parse(await readFile(join(npmInit, "package.json"), "utf8")).type,
    "module",
  );
  const commonjs = await scratch(t, "semantscript-cli-init-commonjs-");
  await writeFile(
    join(commonjs, "package.json"),
    JSON.stringify({ name: "y", main: "index.js" }),
  );
  await writeFile(join(commonjs, "index.js"), "module.exports = {};\n");
  assert.equal(
    await runCli(
      ["init", "--tool", "tsc", "--no-doctor"],
      capture(commonjs).io,
    ),
    0,
  );
  const kept = JSON.parse(
    await readFile(join(commonjs, "package.json"), "utf8"),
  );
  assert.equal(kept.type, undefined);
  assert.equal(kept.scripts.build, "tspc -p tsconfig.json");

  // npm 11's npm init -y writes "type": "commonjs"; with no code yet that is a placeholder too.
  const npm11 = await scratch(t, "semantscript-cli-init-npm11-");
  await writeFile(
    join(npm11, "package.json"),
    JSON.stringify({
      name: "z",
      main: "index.js",
      type: "commonjs",
      scripts: { test: 'echo "Error: no test specified" && exit 1' },
    }),
  );
  const npm11Run = capture(npm11);
  assert.equal(await runCli(["init", "--no-doctor"], npm11Run.io), 0);
  assert.match(
    npm11Run.stdout(),
    /type: module \(was npm init's commonjs default\)/u,
  );
  assert.equal(
    JSON.parse(await readFile(join(npm11, "package.json"), "utf8")).type,
    "module",
  );

  // A plain CommonJS script with no main or scripts (npm install wrote the package.json) keeps working.
  const script = await scratch(t, "semantscript-cli-init-cjs-script-");
  await writeFile(
    join(script, "package.json"),
    '{ "dependencies": { "semantscript": "^0.1.0" } }\n',
  );
  await writeFile(
    join(script, "server.js"),
    'const path = require("node:path");\n',
  );
  assert.equal(await runCli(["init", "--no-doctor"], capture(script).io), 0);
  assert.equal(
    JSON.parse(await readFile(join(script, "package.json"), "utf8")).type,
    undefined,
  );
});

test("init in a directory without package.json names npm init -y", async (t) => {
  const empty = await scratch(t, "semantscript-cli-init-empty-");
  const run = capture(empty);
  assert.equal(await runCli(["init", "--no-doctor"], run.io), 1);
  assert.match(run.stderr(), /no package\.json in .*`npm init -y`/u);
  assert.doesNotMatch(run.stderr(), /usage/iu);
  assert.equal(existsSync(join(empty, "tsconfig.json")), false);
});

test("train, test and run resolve the bundle, artifact and teacher from documented defaults", async (t) => {
  const root = await scratch(t, "semantscript-cli-defaults-");
  await writeFile(
    join(root, "tsconfig.json"),
    JSON.stringify({ compilerOptions: { outDir: "build-output" }, files: [] }),
  );
  await mkdir(join(root, "build-output"));
  await writeFile(join(root, "build-output", "semantscript.ir.v1.json"), "{}");
  const argvPath = join(root, "argv.json");
  const env = {
    PYTHONPATH: fixtures,
    FAKE_TRAINER_ARGV_PATH: argvPath,
    FAKE_TRAINER_EXIT: "0",
    FAKE_TRAINER_SKIP_REPORT: "",
    ANTHROPIC_API_KEY: "",
  };
  const python = process.platform === "win32" ? "python" : "python3";

  const noTeacher = capture(root, env);
  assert.equal(
    await runCli(
      ["train", "--python", python, "--trainer-module", "fake_trainer"],
      noTeacher.io,
    ),
    2,
  );
  assert.match(
    noTeacher.stderr(),
    /--teacher is required: no semantscript\.teacher\.toml, teacher\.toml/u,
  );

  const generated = capture(root, {
    ...env,
    ANTHROPIC_API_KEY: "not-a-real-key",
  });
  assert.equal(
    await runCli(
      ["train", "--python", python, "--trainer-module", "fake_trainer"],
      generated.io,
    ),
    0,
    generated.stderr(),
  );
  const recorded = JSON.parse(await readFile(argvPath, "utf8"));
  const after = (flag) => recorded.argv[recorded.argv.indexOf(flag) + 1];
  assert.equal(
    after("--bundle"),
    join(root, "build-output", "semantscript.ir.v1.json"),
  );
  assert.equal(after("--artifact"), join(root, ".semantscript", "artifact"));
  assert.equal(after("--teacher"), join(root, ".semantscript", "teacher.toml"));
  // The manifest's build.compilerVersion comes from the installed compiler.
  const compilerPackage = JSON.parse(
    await readFile(
      new URL("../../compiler/package.json", import.meta.url),
      "utf8",
    ),
  );
  assert.equal(after("--compiler-version"), compilerPackage.version);
  assert.match(
    await readFile(join(root, ".semantscript", "teacher.toml"), "utf8"),
    /backend = "anthropic"/u,
  );
  assert.match(
    generated.stderr(),
    /wrote .*\.semantscript\/teacher\.toml \(Anthropic backend, claude-sonnet-5\)/u,
  );

  await writeFile(join(root, "teacher.toml"), "[teacher]\n");
  const explicit = capture(root, {
    ...env,
    SEMANTSCRIPT_ARTIFACT: "elsewhere/artifact",
  });
  assert.equal(
    await runCli(
      [
        "train",
        "--python",
        python,
        "--trainer-module",
        "fake_trainer",
        "--compiler-version",
        "9.8.7",
      ],
      explicit.io,
    ),
    0,
    explicit.stderr(),
  );
  const second = JSON.parse(await readFile(argvPath, "utf8"));
  const secondAfter = (flag) => second.argv[second.argv.indexOf(flag) + 1];
  assert.equal(secondAfter("--teacher"), join(root, "teacher.toml"));
  assert.equal(secondAfter("--compiler-version"), "9.8.7");
  assert.equal(
    second.argv.filter((value) => value === "--compiler-version").length,
    1,
  );
  assert.equal(secondAfter("--artifact"), join(root, "elsewhere", "artifact"));

  const testRun = capture(root, {
    SEMANTSCRIPT_ARTIFACT: "elsewhere/artifact",
  });
  assert.equal(await runCli(["test"], testRun.io), 1);
  assert.match(testRun.stderr(), /elsewhere\/artifact/u);
});

test("--teacher constraints reaches train, its preflight and doctor as the built-in keyword", async (t) => {
  const root = await scratch(t, "semantscript-cli-constraints-");
  await mkdir(join(root, "dist"));
  await writeFile(join(root, "dist", "semantscript.ir.v1.json"), "{}");
  const argvPath = join(root, "argv.json");
  const doctorArgvPath = join(root, "doctor-argv.json");
  const env = {
    PYTHONPATH: fixtures,
    FAKE_TRAINER_ARGV_PATH: argvPath,
    FAKE_DOCTOR_ARGV_PATH: doctorArgvPath,
    FAKE_TRAINER_EXIT: "0",
    FAKE_TRAINER_SKIP_REPORT: "",
    ANTHROPIC_API_KEY: "",
  };
  const python = process.platform === "win32" ? "python" : "python3";
  const base = ["--python", python, "--trainer-module", "fake_trainer"];
  const teacherOf = async (path) => {
    const argv = JSON.parse(await readFile(path, "utf8")).argv;
    return argv[argv.indexOf("--teacher") + 1];
  };

  const train = capture(root, env);
  assert.equal(
    await runCli(["train", ...base, "--teacher", "constraints"], train.io),
    0,
    train.stderr(),
  );
  assert.equal(await teacherOf(argvPath), "constraints");
  assert.equal(await teacherOf(doctorArgvPath), "constraints");
  assert.ok(!existsSync(join(root, ".semantscript", "teacher.toml")));

  const doctor = capture(root, env);
  assert.equal(
    await runCli(["doctor", ...base, "--teacher", "constraints"], doctor.io),
    0,
    doctor.stderr(),
  );
  assert.equal(await teacherOf(doctorArgvPath), "constraints");

  // A directory of that name does not shadow the keyword.
  await mkdir(join(root, "constraints"));
  const directory = capture(root, env);
  assert.equal(
    await runCli(["train", ...base, "--teacher", "constraints"], directory.io),
    0,
    directory.stderr(),
  );
  assert.equal(await teacherOf(argvPath), "constraints");
  await rm(join(root, "constraints"), { recursive: true });

  // A file of that name is a teacher file like any other.
  await writeFile(join(root, "constraints"), "[teacher]\n");
  const file = capture(root, env);
  assert.equal(
    await runCli(["train", ...base, "--teacher", "constraints"], file.io),
    0,
    file.stderr(),
  );
  assert.equal(await teacherOf(argvPath), join(root, "constraints"));
});

test("dev builds and trains, reruns on a saved source change with the cache, and stops on abort", async (t) => {
  const root = await scratch(t, "semantscript-cli-dev-");
  const configPath = await createProject(root, { "app.sem.ts": program });
  await writeFile(join(root, "teacher.toml"), "[teacher]\n");
  const argvPath = join(root, "argv.json");
  const env = {
    PYTHONPATH: fixtures,
    FAKE_TRAINER_ARGV_PATH: argvPath,
    FAKE_TRAINER_EXIT: "0",
    FAKE_TRAINER_SKIP_REPORT: "",
  };
  const python = process.platform === "win32" ? "python" : "python3";
  const controller = new AbortController();
  const run = capture(root, env);
  const io = { ...run.io, signal: controller.signal };
  const finished = runCli(
    [
      "dev",
      "--project",
      configPath,
      "--application",
      "demo",
      "--python",
      python,
      "--trainer-module",
      "fake_trainer",
      "--debounce",
      "50",
    ],
    io,
  );

  const waitFor = async (predicate) => {
    for (let attempt = 0; attempt < 400; attempt += 1) {
      if (predicate()) return;
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    throw new Error(`timed out; stderr: ${run.stderr()}`);
  };
  await waitFor(
    () =>
      /cycle 1 \(initial build\)/u.test(run.stderr()) &&
      /release published/u.test(run.stderr()),
  );
  const first = JSON.parse(await readFile(argvPath, "utf8"));
  assert.ok(first.argv.includes(join(root, "dist", "semantscript.ir.v1.json")));
  assert.ok(first.argv.includes(join(root, ".semantscript", "cache")));
  assert.ok(
    !first.argv.includes("--no-cache") && !first.argv.includes("--full"),
  );
  assert.match(run.stdout(), /nf_33333333…\s+src\/app\.sem\.ts\s+trained/u);

  // A save of a sema source triggers one more build and train.
  await writeFile(
    join(root, "src", "app.sem.ts"),
    program.replace("Is this positive?", "Is this message positive?"),
  );
  await waitFor(() =>
    /cycle 2 \(src\/app\.sem\.ts changed\)/u.test(run.stderr()),
  );
  await waitFor(
    () => (run.stderr().match(/release published/gu) ?? []).length === 2,
  );
  assert.equal(
    (run.stdout().match(/compiled 1 neural function/gu) ?? []).length,
    2,
  );

  // Emitted JavaScript under dist/ does not retrigger the loop.
  await writeFile(join(root, "dist", "note.js"), "// not a source\n");
  await new Promise((resolve) => setTimeout(resolve, 200));
  assert.equal((run.stderr().match(/cycle \d/gu) ?? []).length, 2);

  controller.abort();
  assert.equal(await finished, 0);

  // --once runs one cycle and returns the train status.
  const once = capture(root, { ...env, FAKE_TRAINER_EXIT: "1" });
  assert.equal(
    await runCli(
      [
        "dev",
        "--once",
        "--project",
        configPath,
        "--python",
        python,
        "--trainer-module",
        "fake_trainer",
      ],
      once.io,
    ),
    1,
  );
  assert.match(
    once.stderr(),
    /training failed; the previous artifact stays in service/u,
  );
});

test("doctor prints one line per check with the fix, and exits 1 on a failure", async (t) => {
  const root = await scratch(t, "semantscript-cli-doctor-");
  await writeFile(join(root, "teacher.toml"), "[teacher]\n");
  const argvPath = join(root, "doctor-argv.json");
  const python = process.platform === "win32" ? "python" : "python3";
  const base = [
    "doctor",
    "--python",
    python,
    "--trainer-module",
    "fake_trainer",
  ];
  const env = { PYTHONPATH: fixtures, FAKE_DOCTOR_ARGV_PATH: argvPath };

  const passed = capture(root, env);
  assert.equal(await runCli(base, passed.io), 0, passed.stderr());
  const lines = passed.stdout().split("\n");
  assert.match(lines[0], /^semantscript doctor: /u);
  assert.match(lines[1], /^ {2}pass {2}node {14}Node \d+\.\d+\.\d+ /u);
  assert.match(
    lines[2],
    /^ {2}pass {2}runtime-bindings {2}onnxruntime-node \S+ and tokenizers \S+ loaded for /u,
  );
  for (const id of ["python", "trainer", "torch", "device", "teacher-probe"]) {
    assert.match(
      passed.stdout(),
      new RegExp(`\\n {2}pass {2}${id} +fake ${id}\\n`, "u"),
    );
  }
  assert.match(
    passed.stdout(),
    /12 passed, 0 warnings, 0 failed, 0 skipped\n$/u,
  );
  const forwarded = JSON.parse(await readFile(argvPath, "utf8")).argv;
  assert.deepEqual(forwarded.slice(0, 3), ["--json", "--probe", "request"]);
  assert.equal(
    forwarded[forwarded.indexOf("--teacher") + 1],
    join(root, "teacher.toml"),
  );
  assert.ok(!forwarded.includes("--quick"));

  const failed = capture(root, { ...env, FAKE_DOCTOR_FAIL: "teacher-key" });
  assert.equal(
    await runCli(
      [...base, "--probe", "none", "--no-teacher", "--device", "cpu"],
      failed.io,
    ),
    1,
  );
  assert.match(
    failed.stdout(),
    / {2}fail {2}teacher-key {7}fake teacher-key\n {8}fix: fake fix for teacher-key\n/u,
  );
  const limited = JSON.parse(await readFile(argvPath, "utf8")).argv;
  assert.ok(limited.includes("--no-teacher"));
  assert.equal(limited[limited.indexOf("--probe") + 1], "none");
  assert.equal(limited[limited.indexOf("--device") + 1], "cpu");

  const json = capture(root, env);
  assert.equal(await runCli([...base, "--json"], json.io), 0);
  const report = JSON.parse(json.stdout());
  assert.equal(report.kind, "semantscript.doctor-report");
  assert.equal(report.checks.length, 12);
  assert.deepEqual(report.checks[0].id, "node");

  const runtime = capture(root, env);
  assert.equal(
    await runCli(["doctor", "--runtime"], runtime.io),
    0,
    runtime.stdout(),
  );
  assert.match(
    runtime.stdout(),
    /2 passed, 0 warnings, 0 failed, 0 skipped\n$/u,
  );

  const malformed = capture(root, { ...env, FAKE_DOCTOR_MALFORMED: "1" });
  assert.equal(await runCli(base, malformed.io), 1);
  assert.match(
    malformed.stderr(),
    /doctor printed no JSON report: this is not a report/u,
  );

  // A piped Python on Windows writes the locale code page (simulated here
  // with PYTHONIOENCODING=cp1252): the CLI must ask for UTF-8 so a
  // non-ASCII checkout or user name neither crashes nor garbles the report.
  const unicodeDirectory = join(root, "项目 café");
  await mkdir(unicodeDirectory);
  await writeFile(join(unicodeDirectory, "teacher.toml"), "[teacher]\n");
  const unicode = capture(unicodeDirectory, {
    ...env,
    FAKE_DOCTOR_RAW_TEACHER: "1",
    PYTHONIOENCODING: "cp1252",
  });
  assert.equal(await runCli(base, unicode.io), 0, unicode.stdout());
  assert.ok(
    unicode
      .stdout()
      .includes(`teacher ${join(unicodeDirectory, "teacher.toml")}\n`),
    unicode.stdout(),
  );

  const badId = capture(root, { ...env, FAKE_DOCTOR_BAD_ID: "1" });
  assert.equal(await runCli(base, badId.io), 1);
  assert.match(badId.stderr(), /checks\[0\]\.id "gpu" is not a known check/u);

  const noPython = capture(root, env);
  assert.equal(
    await runCli(
      ["doctor", "--python", join(root, "no-such-python")],
      noPython.io,
    ),
    1,
  );
  assert.match(
    noPython.stdout(),
    / {2}fail {2}python {12}unable to run .*no-such-python/u,
  );
  assert.match(noPython.stdout(), /fix: install Python 3\.12 or later/u);
  assert.match(
    noPython.stdout(),
    / {2}skip {2}teacher-probe {5}not checked: the interpreter does not start/u,
  );

  const noModule = capture(root, env);
  assert.equal(
    await runCli(
      [
        "doctor",
        "--python",
        python,
        "--trainer-module",
        "no_such_trainer_module",
      ],
      noModule.io,
    ),
    1,
  );
  assert.match(noModule.stdout(), / {2}pass {2}python {12}Python 3\./u);
  assert.match(
    noModule.stdout(),
    / {2}fail {2}trainer {11}.* cannot import no_such_trainer_module/u,
  );
  assert.match(
    noModule.stdout(),
    /fix: install the trainer into .*: pip install "semantscript-trainer\[training\]" \(in a SemantScript checkout: pip install -e "\.\[training\]"/u,
  );

  const badProbe = capture(root, env);
  assert.equal(await runCli([...base, "--probe", "maybe"], badProbe.io), 2);
  assert.match(
    badProbe.stderr(),
    /--probe must be one of request, free, none/u,
  );

  assert.equal(checkNode("20.11.0").status, "fail");
  assert.match(checkNode("20.11.0").fix, /Node 22\.13\.0 or later/u);
  assert.equal(checkNode("22.13.0").status, "pass");
  const unresolved = await checkRuntimeBindings(
    pathToFileURL(join(root, "index.js")).href,
  );
  assert.equal(unresolved.status, "fail");
  assert.match(unresolved.summary, /onnxruntime-node does not load on /u);
  assert.match(unresolved.fix, /npm rebuild onnxruntime-node tokenizers/u);
});

test("train runs the environment preflight first and stops before the trainer on a failure", async (t) => {
  const root = await scratch(t, "semantscript-cli-preflight-");
  await writeFile(join(root, "bundle.json"), "{}");
  await writeFile(join(root, "teacher.toml"), "[teacher]\n");
  const argvPath = join(root, "argv.json");
  const doctorPath = join(root, "doctor-argv.json");
  const env = {
    PYTHONPATH: fixtures,
    FAKE_TRAINER_ARGV_PATH: argvPath,
    FAKE_DOCTOR_ARGV_PATH: doctorPath,
    FAKE_TRAINER_EXIT: "0",
    FAKE_TRAINER_SKIP_REPORT: "",
  };
  const python = process.platform === "win32" ? "python" : "python3";
  const args = [
    "train",
    "--bundle",
    "bundle.json",
    "--python",
    python,
    "--trainer-module",
    "fake_trainer",
    "--device",
    "cpu",
  ];

  const stopped = capture(root, { ...env, FAKE_DOCTOR_FAIL: "teacher-key" });
  assert.equal(await runCli(args, stopped.io), 1);
  assert.ok(!existsSync(argvPath), "the trainer never started");
  assert.match(
    stopped.stderr(),
    /semantscript train: environment preflight \(\d+\.\d s\)/u,
  );
  assert.match(
    stopped.stderr(),
    /fail {2}teacher-key .*\n {8}fix: fake fix for teacher-key/u,
  );
  assert.match(
    stopped.stderr(),
    /stopped before training; fix the failed checks above/u,
  );
  const checked = JSON.parse(await readFile(doctorPath, "utf8")).argv;
  assert.equal(checked[checked.indexOf("--probe") + 1], "free");
  assert.ok(checked.includes("--quick"));
  assert.equal(
    checked[checked.indexOf("--teacher") + 1],
    join(root, "teacher.toml"),
  );
  assert.equal(checked[checked.indexOf("--device") + 1], "cpu");

  await rm(doctorPath);
  const skipped = capture(root, { ...env, FAKE_DOCTOR_FAIL: "teacher-key" });
  assert.equal(
    await runCli([...args, "--no-preflight"], skipped.io),
    0,
    skipped.stderr(),
  );
  assert.ok(!existsSync(doctorPath), "--no-preflight runs no checks");
  assert.ok(
    !JSON.parse(await readFile(argvPath, "utf8")).argv.includes(
      "--no-preflight",
    ),
  );

  const passed = capture(root, env);
  assert.equal(await runCli(args, passed.io), 0, passed.stderr());
  assert.match(
    passed.stderr(),
    /10 passed, 0 warnings, 0 failed, 0 skipped\n/u,
  );
  assert.match(passed.stdout(), /train passed\n$/u);
});

test("init ends with the environment checks and keeps exit status 0", async (t) => {
  const root = await scratch(t, "semantscript-cli-init-doctor-");
  await writeFile(
    join(root, "package.json"),
    '{ "name": "app", "type": "module", "scripts": { "build": "tsc" } }\n',
  );
  await writeFile(join(root, "tsconfig.json"), '{ "compilerOptions": {} }\n');
  const python = process.platform === "win32" ? "python" : "python3";
  const doctorPath = join(root, "doctor-argv.json");
  const run = capture(root, {
    PYTHONPATH: fixtures,
    FAKE_DOCTOR_FAIL: "teacher-config",
    FAKE_DOCTOR_ARGV_PATH: doctorPath,
  });
  assert.equal(
    await runCli(
      [
        "init",
        "--no-example",
        "--python",
        python,
        "--trainer-module",
        "fake_trainer",
      ],
      run.io,
    ),
    0,
    run.stderr(),
  );
  assert.match(
    run.stdout(),
    /next steps:[\s\S]*environment \(semantscript doctor\):\n {2}pass {2}node /u,
  );
  assert.match(
    run.stdout(),
    /fail {2}teacher-config .*\n {8}fix: fake fix for teacher-config\n/u,
  );
  assert.match(
    run.stdout(),
    /fix the 1 failed check before semantscript train/u,
  );
  const checked = JSON.parse(await readFile(doctorPath, "utf8")).argv;
  assert.equal(checked[checked.indexOf("--probe") + 1], "free");
});

test(
  "doctor fails an interpreter older than 3.12 whose trainer import breaks",
  {
    skip:
      process.platform === "win32"
        ? "the shim interpreter is a shell script"
        : false,
  },
  async (t) => {
    const root = await scratch(t, "semantscript-cli-doctor-old-python-");
    const shim = join(root, "python3.11");
    // Answers the version query as 3.11 and fails the trainer import the way a
    // PEP 695 `type` statement does before 3.12.
    await writeFile(
      shim,
      '#!/bin/sh\nif [ "$1" = "-c" ]; then echo 3.11.9; exit 0; fi\n' +
        'echo "SyntaxError: invalid syntax" >&2; exit 1\n',
    );
    await chmod(shim, 0o755);
    const run = capture(root, {});
    assert.equal(
      await runCli(["doctor", "--python", shim, "--no-teacher"], run.io),
      1,
    );
    assert.match(
      run.stdout(),
      / {2}fail {2}python {12}Python 3\.11\.9 \(.*\) is older than 3\.12, which the trainer needs\n {8}fix: install Python 3\.12 or later/u,
    );
    assert.match(run.stdout(), / {2}skip {2}trainer {11}not checked/u);
  },
);

test("doctor names a nearby .venv when the default interpreter lacks a package", async (t) => {
  const root = await scratch(t, "semantscript-cli-doctor-venv-");
  const venvPython =
    process.platform === "win32"
      ? join(root, ".venv", "Scripts", "python.exe")
      : join(root, ".venv", "bin", "python");
  await mkdir(dirname(venvPython), { recursive: true });
  await writeFile(venvPython, "");
  const project = join(root, "app");
  await mkdir(project);
  const env = {
    PYTHONPATH: fixtures,
    SEMANTSCRIPT_PYTHON: "",
    FAKE_DOCTOR_FAIL: "torch",
  };
  const base = ["doctor", "--trainer-module", "fake_trainer", "--no-teacher"];

  const defaulted = capture(project, env);
  assert.equal(await runCli(base, defaulted.io), 1, defaulted.stderr());
  assert.match(
    defaulted.stdout(),
    / {2}fail {2}torch .*\n {8}fix: fake fix for torch; or if the packages are in \.\.[/\\]\.venv\S+ rather than python3? \(the default interpreter\), pass --python \.\.[/\\]\.venv/u,
  );

  const python = process.platform === "win32" ? "python" : "python3";
  const chosen = capture(project, env);
  assert.equal(await runCli([...base, "--python", python], chosen.io), 1);
  assert.match(chosen.stdout(), /fix: fake fix for torch\n/u);
});

test("init keeps exit status 0 when the doctor breaks its report contract", async (t) => {
  const root = await scratch(t, "semantscript-cli-init-doctor-broken-");
  await writeFile(
    join(root, "package.json"),
    '{ "name": "app", "type": "module", "scripts": { "build": "tsc" } }\n',
  );
  await writeFile(join(root, "tsconfig.json"), '{ "compilerOptions": {} }\n');
  const python = process.platform === "win32" ? "python" : "python3";
  const run = capture(root, {
    PYTHONPATH: fixtures,
    FAKE_DOCTOR_MALFORMED: "1",
  });
  assert.equal(
    await runCli(
      [
        "init",
        "--no-example",
        "--python",
        python,
        "--trainer-module",
        "fake_trainer",
      ],
      run.io,
    ),
    0,
    run.stderr(),
  );
  assert.match(
    run.stdout(),
    /environment \(semantscript doctor\): the checks did not run: .*no JSON report/u,
  );
});

test("train --estimate renders the teacher's cost without the preflight or a run", async (t) => {
  const root = await scratch(t, "semantscript-cli-estimate-");
  await writeFile(join(root, "bundle.json"), "{}");
  await writeFile(join(root, "teacher.toml"), "[teacher]\n");
  const argvPath = join(root, "argv.json");
  const env = {
    PYTHONPATH: fixtures,
    FAKE_TRAINER_ARGV_PATH: argvPath,
    // A failing preflight would stop a real run; the estimate never runs it.
    FAKE_DOCTOR_FAIL: "teacher-config",
  };
  const python = process.platform === "win32" ? "python" : "python3";
  const args = [
    "train",
    "--bundle",
    "bundle.json",
    "--teacher",
    "teacher.toml",
    "--python",
    python,
    "--trainer-module",
    "fake_trainer",
    "--cases",
    "32",
    "--estimate",
  ];
  const run = capture(root, env);
  assert.equal(await runCli(args, run.io), 0, run.stderr());
  const recorded = JSON.parse(await readFile(argvPath, "utf8")).argv;
  assert.ok(recorded.includes("--estimate"));
  assert.doesNotMatch(run.stderr(), /environment preflight/u);
  assert.match(
    run.stdout(),
    /^teacher: anthropic claude-sonnet-5; price: pinned Anthropic list price 2026-09-25 \(USD 2 in \/ 10 out per million tokens\)\n/u,
  );
  assert.match(
    run.stdout(),
    /nf_33333333…\s+src\/app\.sem\.ts\s+90 \(max 300\)\s+180,000\s+150,000\s+9,000\s+0\.20 \(max 0\.66\)\s+6 min/u,
  );
  assert.match(
    run.stdout(),
    /nf_44444444…\s+src\/other\.sem\.ts \(cached\)\s+0\s/u,
  );
  assert.match(run.stdout(), /\ntotal\s+90 \(max 300\)/u);
  assert.match(run.stdout(), /no teacher request was sent\n$/u);
  assert.doesNotMatch(run.stdout(), /Message Batches/u);

  // A Message Batches run shows the batch's expected time and its maximum.
  const batched = capture(root, { ...env, FAKE_ESTIMATE_BATCH: "1" });
  assert.equal(await runCli(args, batched.io), 0, batched.stderr());
  assert.match(
    batched.stdout(),
    /nf_33333333…\s+src\/app\.sem\.ts\s+90 \(max 300\).*62 min \(max 97\.2 h\)/u,
  );
  assert.match(
    batched.stdout(),
    /60 requests go through the Message Batches API at half price; the time counts about 1 h per batch/u,
  );

  const capped = capture(root, env);
  assert.equal(
    await runCli([...args, "--max-cost-usd", "0.1"], capped.io),
    0,
    capped.stderr(),
  );
  assert.match(
    capped.stdout(),
    /--max-cost-usd 0\.1 is below the expected USD 0\.20: the run will stop before it finishes/u,
  );
  const covered = capture(root, env);
  assert.equal(await runCli([...args, "--max-cost-usd", "5"], covered.io), 0);
  assert.match(covered.stdout(), /--max-cost-usd 5 covers the maximum cost/u);

  const invalid = capture(root, env);
  assert.equal(
    await runCli([...args, "--max-cost-usd", "free"], invalid.io),
    2,
  );
  assert.match(invalid.stderr(), /--max-cost-usd must be a positive number/u);
});

test("train passes --max-cost-usd through and prints the teacher's running cost", async (t) => {
  const root = await scratch(t, "semantscript-cli-spend-");
  await writeFile(join(root, "bundle.json"), "{}");
  await writeFile(join(root, "teacher.toml"), "[teacher]\n");
  const argvPath = join(root, "argv.json");
  const run = capture(root, {
    PYTHONPATH: fixtures,
    FAKE_TRAINER_ARGV_PATH: argvPath,
    FAKE_TRAINER_SPEND: "1",
  });
  assert.equal(
    await runCli(
      [
        "train",
        "--bundle",
        "bundle.json",
        "--teacher",
        "teacher.toml",
        "--python",
        process.platform === "win32" ? "python" : "python3",
        "--trainer-module",
        "fake_trainer",
        "--no-preflight",
        "--max-cost-usd",
        "2.5",
      ],
      run.io,
    ),
    0,
    run.stderr(),
  );
  const recorded = JSON.parse(await readFile(argvPath, "utf8")).argv;
  assert.equal(recorded[recorded.indexOf("--max-cost-usd") + 1], "2.5");
  assert.match(
    run.stdout(),
    /teacher: 40 requests \(12 replayed\), USD 0\.0931 of the USD 2\.5 cap\n/u,
  );
});

test("train passes the seed retry flags through and prints every attempt", async (t) => {
  const root = await scratch(t, "semantscript-cli-retry-");
  await writeFile(join(root, "bundle.json"), "{}");
  await writeFile(join(root, "teacher.toml"), "[teacher]\n");
  const argvPath = join(root, "argv.json");
  const env = {
    PYTHONPATH: fixtures,
    FAKE_TRAINER_ARGV_PATH: argvPath,
    FAKE_TRAINER_RETRY: "1",
  };
  const args = [
    "train",
    "--bundle",
    "bundle.json",
    "--teacher",
    "teacher.toml",
    "--python",
    process.platform === "win32" ? "python" : "python3",
    "--trainer-module",
    "fake_trainer",
    "--no-preflight",
    "--seed",
    "3",
    "--seed-attempts",
    "5",
    "--seed-retry-margin",
    "2.5",
  ];
  const run = capture(root, env);
  assert.equal(await runCli(args, run.io), 0, run.stderr());
  const recorded = JSON.parse(await readFile(argvPath, "utf8")).argv;
  assert.equal(recorded[recorded.indexOf("--seed-attempts") + 1], "5");
  assert.equal(recorded[recorded.indexOf("--seed-retry-margin") + 1], "2.5");
  const out = run.stdout();
  assert.match(out, /training attempts:\n/u);
  assert.match(
    out,
    /attempt\s+seed\s+function\s+status\s+accuracy\s+violations\s+held-out\s+ece\n/u,
  );
  assert.match(
    out,
    /1\s+3\s+nf_33333333…\s+failed\s+0\.9700\s+4\/394 \(1\.02%\)\s+1\/512 \(0\.20%\)\s+0\.0400\n/u,
  );
  assert.match(
    out,
    /3\s+5\s+nf_33333333…\s+passed\s+0\.9700\s+2\/394 \(0\.51%\)\s+0\/512\s+0\.0400/u,
  );
  assert.match(out, /seed retry: published seed 5 after 3 attempts\n/u);
  assert.doesNotMatch(out, /seed retry stopped/u);

  const stopped = capture(root, { ...env, FAKE_TRAINER_EXIT: "1" });
  assert.equal(await runCli(args, stopped.io), 1);
  assert.match(
    stopped.stdout(),
    /2\s+4\s+nf_33333333…\s+failed\s+0\.9700\s+8\/394 \(2\.03%\)/u,
  );
  assert.doesNotMatch(stopped.stdout(), /published seed/u);
  assert.match(
    stopped.stdout(),
    /seed retry stopped: nf_3{64}: violation rate 2\.0305% \(8 of 394\) is outside the retry margin 2 x 0\.01 = 2\.0000%\n/u,
  );
  assert.match(stopped.stdout(), /train failed\n$/u);
});

test("teacher probe reports the model, latency, tokens and cost of one request", async (t) => {
  const root = await scratch(t, "semantscript-cli-probe-");
  await writeFile(join(root, "teacher.toml"), "[teacher]\n");
  const argvPath = join(root, "probe-argv.json");
  const python = process.platform === "win32" ? "python" : "python3";
  const args = [
    "teacher",
    "probe",
    "--python",
    python,
    "--trainer-module",
    "fake_trainer",
  ];
  const run = capture(root, {
    PYTHONPATH: fixtures,
    FAKE_DOCTOR_ARGV_PATH: argvPath,
  });
  assert.equal(await runCli(args, run.io), 0, run.stderr());
  const recorded = JSON.parse(await readFile(argvPath, "utf8")).argv;
  assert.equal(
    recorded[recorded.indexOf("--teacher") + 1],
    join(root, "teacher.toml"),
  );
  assert.ok(recorded.includes("--json"));
  assert.match(
    run.stdout(),
    /model {4}anthropic\/claude-sonnet-5 via openrouter\.ai\n {2}latency {2}1\.25 s\n {2}tokens {3}16 in \/ 4 out\n {2}cost {5}USD 0\.000072 \(OpenRouter price list \(2026-09-25\)\)\n {2}ok {7}one request/u,
  );

  const json = capture(root, { PYTHONPATH: fixtures });
  assert.equal(await runCli([...args, "--json"], json.io), 0);
  assert.equal(JSON.parse(json.stdout()).costUsd, 0.000072);

  const failed = capture(root, { PYTHONPATH: fixtures, FAKE_PROBE_FAIL: "1" });
  assert.equal(await runCli(args, failed.io), 1);
  assert.match(
    failed.stdout(),
    /failed {3}one-request probe failed\n {2}fix {6}check the network\n/u,
  );

  const empty = await scratch(t, "semantscript-cli-probe-empty-");
  const missing = capture(empty, { PYTHONPATH: fixtures });
  assert.equal(await runCli(args, missing.io), 2);
  assert.match(missing.stderr(), /no teacher to probe/u);
  const bare = capture(root, { PYTHONPATH: fixtures });
  assert.equal(await runCli(["teacher"], bare.io), 2);
});

test("init writes the chosen teacher without a key, asks only on a terminal and never replaces one", async (t) => {
  const python = process.platform === "win32" ? "python" : "python3";
  const expected = {
    anthropic: [/backend = "anthropic"/u, /model = "claude-sonnet-5"/u],
    openrouter: [
      /model = "anthropic\/claude-sonnet-5"/u,
      /base_url = "https:\/\/openrouter\.ai\/api"/u,
      /mode = "direct"/u,
      /ANTHROPIC_API_KEY="\$OPENROUTER_API_KEY"/u,
    ],
    ollama: [/backend = "ollama"/u, /model = "qwen3:14b"/u],
    constraints: [/backend = "constraints"/u],
  };
  for (const [choice, patterns] of Object.entries(expected)) {
    const root = await scratch(t, `semantscript-cli-init-${choice}-`);
    await writeFile(
      join(root, "package.json"),
      '{ "name": "app", "type": "module" }\n',
    );
    await writeFile(join(root, "tsconfig.json"), '{ "compilerOptions": {} }\n');
    const doctorPath = join(root, "doctor-argv.json");
    const run = capture(root, {
      PYTHONPATH: fixtures,
      FAKE_DOCTOR_ARGV_PATH: doctorPath,
      ANTHROPIC_API_KEY: "sk-test-secret",
    });
    const args = [
      "init",
      "--no-example",
      "--teacher",
      choice,
      "--python",
      python,
      "--trainer-module",
      "fake_trainer",
    ];
    assert.equal(await runCli(args, run.io), 0, run.stderr());
    const written = await readFile(
      join(root, ".semantscript", "teacher.toml"),
      "utf8",
    );
    for (const pattern of patterns) assert.match(written, pattern, choice);
    assert.doesNotMatch(written, /sk-test-secret|api_key/u);
    assert.match(run.stdout(), /changed {4}\.semantscript\/teacher\.toml: /u);
    const checked = JSON.parse(await readFile(doctorPath, "utf8")).argv;
    assert.equal(
      checked[checked.indexOf("--teacher") + 1],
      join(root, ".semantscript", "teacher.toml"),
    );

    const again = capture(root, { PYTHONPATH: fixtures });
    assert.equal(
      await runCli([...args.slice(0, 3), "ollama", ...args.slice(4)], again.io),
      0,
    );
    assert.match(
      again.stdout(),
      /unchanged {2}\.semantscript\/teacher\.toml: a teacher file already exists/u,
    );
    assert.equal(
      await readFile(join(root, ".semantscript", "teacher.toml"), "utf8"),
      written,
    );
  }

  const root = await scratch(t, "semantscript-cli-init-ask-");
  await writeFile(
    join(root, "package.json"),
    '{ "name": "app", "type": "module" }\n',
  );
  await writeFile(join(root, "tsconfig.json"), '{ "compilerOptions": {} }\n');
  const answers = ["gpt", "OpenRouter"];
  const questions = [];
  const asked = capture(root, { PYTHONPATH: fixtures });
  const io = {
    ...asked.io,
    ask: async (question) => {
      questions.push(question);
      return answers.shift() ?? "";
    },
  };
  assert.equal(
    await runCli(["init", "--no-example", "--no-doctor"], io),
    0,
    asked.stderr(),
  );
  assert.equal(questions.length, 2);
  assert.match(questions[0], /anthropic, openrouter, ollama, constraints/u);
  assert.match(asked.stdout(), /gpt is not one of/u);
  assert.match(
    await readFile(join(root, ".semantscript", "teacher.toml"), "utf8"),
    /openrouter\.ai/u,
  );

  const quiet = await scratch(t, "semantscript-cli-init-quiet-");
  await writeFile(
    join(quiet, "package.json"),
    '{ "name": "app", "type": "module" }\n',
  );
  await writeFile(join(quiet, "tsconfig.json"), '{ "compilerOptions": {} }\n');
  const plain = capture(quiet, { PYTHONPATH: fixtures });
  assert.equal(
    await runCli(["init", "--no-example", "--no-doctor"], plain.io),
    0,
  );
  assert.ok(!existsSync(join(quiet, ".semantscript", "teacher.toml")));
  const wrong = capture(quiet, { PYTHONPATH: fixtures });
  assert.equal(
    await runCli(["init", "--no-doctor", "--teacher", "gpt"], wrong.io),
    2,
  );
  assert.match(
    wrong.stderr(),
    /--teacher must be one of anthropic, openrouter, ollama, constraints/u,
  );
});
