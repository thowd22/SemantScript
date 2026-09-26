import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import { cp, mkdir, mkdtemp, rm, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import process from "node:process";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  createFixtureArtifact,
  fixtureFunctionId,
} from "../../runtime/test/fixtures/artifact.mjs";
import { runCli } from "../dist/index.js";

const fixtures = join(dirname(fileURLToPath(import.meta.url)), "fixtures");
const modulePath = join(fixtures, "explain-module.mjs");
const changedFunctionId = `nf_${"9".repeat(64)}`;

function capture(cwd) {
  const out = [];
  const err = [];
  return {
    io: {
      cwd,
      env: { ...process.env },
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

const facts = (a, b) => ({ facts: { a, b } });
const input = (name) => ({ node: "input", name });
const property = (object, name) => ({
  node: "property",
  object,
  property: name,
});
const literal = (value) => ({ node: "literal", value });
const binary = (operator, left, right) => ({
  node: "binary",
  operator,
  left,
  right,
});
const fact = (name) => property(input("facts"), name);

/** The fixture answers "review" for facts {a: 1, b: 2}. */
function definition() {
  return {
    template: [],
    examples: [
      { inputs: facts(50, 50), output: "deny" },
      { inputs: facts(1, 2), output: "review" },
      { inputs: facts(2, 2), output: "approve" },
    ],
    constraints: [
      {
        kind: "always",
        source: "facts.a > 0",
        predicate: binary(">", fact("a"), literal(0)),
        output: "review",
      },
      {
        kind: "never",
        source: "facts.b === 2",
        predicate: binary("===", fact("b"), literal(2)),
        output: "review",
      },
      {
        kind: "always",
        source: "facts.a > 100",
        predicate: binary(">", fact("a"), literal(100)),
        output: "deny",
      },
      {
        kind: "always",
        source: "facts.missing.x === 1",
        predicate: binary("===", property(fact("missing"), "x"), literal(1)),
        output: "approve",
      },
    ],
  };
}

async function writeBundle(path, ids) {
  await writeFile(
    path,
    JSON.stringify({
      kind: "semantscript.ir-bundle",
      bundleVersion: 1,
      functions: ids.map((id, index) => ({
        id,
        source: { path: "src/decide.sem.ts", line: 10 + index, column: 3 },
        output: { kind: "scalar" },
        definition: definition(),
      })),
      executionPlan: { stages: [], dependencies: [] },
    }),
  );
}

async function writeCached(cacheDir, family, document) {
  const bytes = Buffer.from(JSON.stringify(document));
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  const path = join(
    cacheDir,
    family,
    "v1",
    sha256.slice(0, 2),
    `${sha256}.json`,
  );
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, bytes);
  return { path, sha256 };
}

function dataset(cases) {
  return {
    kind: "semantscript.training-dataset",
    datasetVersion: 1,
    payload: {
      function: { id: fixtureFunctionId, semanticSha256: "2".repeat(64) },
      teacher: {
        provider: "anthropic",
        model: "claude-sonnet-5",
        configurationSha256: "8".repeat(64),
      },
      cases,
    },
  };
}

function sidecar(baseSha256, cases) {
  return {
    kind: "semantscript.adversarial-dataset",
    datasetVersion: 1,
    payload: {
      baseDataset: { datasetSha256: baseSha256, payloadSha256: "0".repeat(64) },
      function: { id: fixtureFunctionId, semanticSha256: "2".repeat(64) },
      cases,
    },
  };
}

/** A training cache holding the release's dataset, its sidecar, and a newer decoy dataset. */
async function createCache(cacheDir) {
  const release = await writeCached(
    cacheDir,
    "datasets",
    dataset([
      { origin: "gold", inputs: facts(1, 2), output: "review" },
      { origin: "synthetic", inputs: facts(40, 40), output: "deny" },
      { origin: "synthetic", inputs: facts(1, 4), output: "approve" },
    ]),
  );
  await writeCached(
    cacheDir,
    "adversarial-datasets",
    sidecar(release.sha256, [
      {
        caseId: "ac_1",
        tag: "constraint-boundary",
        inputs: facts(1, 3),
        output: "approve",
      },
    ]),
  );
  const decoy = await writeCached(
    cacheDir,
    "datasets",
    dataset([{ origin: "synthetic", inputs: facts(1, 2), output: "decoy" }]),
  );
  const later = new Date(Date.now() + 60_000);
  await utimes(decoy.path, later, later);
  await writeCached(
    cacheDir,
    "adversarial-datasets",
    sidecar(decoy.sha256, [
      {
        caseId: "ac_2",
        tag: "counterfactual",
        inputs: facts(1, 2),
        output: "decoy",
      },
    ]),
  );
  return { release, decoy };
}

function explainArgs(artifactRoot, bundle, cacheDir, ...rest) {
  return [
    "explain",
    "--artifact",
    artifactRoot,
    "--bundle",
    bundle,
    "--cache-dir",
    cacheDir,
    modulePath,
    ...rest,
  ];
}

test("explain reports the distribution, active constraints, nearest cases and the release for one call", async (t) => {
  const root = await scratch(t, "semantscript-cli-explain-");
  const cacheDir = join(root, "cache");
  const { release } = await createCache(cacheDir);
  const artifactRoot = join(root, "artifact");
  const artifact = await createFixtureArtifact(artifactRoot, {
    transformManifest(manifest) {
      manifest.functions[0].trainingProvenance.datasetSha256 = release.sha256;
      manifest.functions[0].trainingProvenance.teacher =
        "anthropic/claude-sonnet-5";
    },
  });
  const bundle = join(root, "bundle.json");
  await writeBundle(bundle, [fixtureFunctionId]);

  const text = capture(root);
  assert.equal(
    await runCli(
      explainArgs(
        artifactRoot,
        bundle,
        cacheDir,
        "--call",
        "decide",
        "--input",
        "[1, 2]",
        "--neighbors",
        "3",
      ),
      text.io,
    ),
    0,
    text.stderr(),
  );
  const out = text.stdout();
  assert.match(out, /^explain decide\(1, 2\)\n/u);
  assert.match(out, new RegExp(`manifest ${artifact.manifestSha256}`, "u"));
  assert.match(out, /runtime-fixture@0\.0\.0, built 2026-09-22T00:00:00Z/u);
  assert.match(
    out,
    new RegExp(
      `call 1: ${fixtureFunctionId} \\(src/decide\\.sem\\.ts:10\\)`,
      "u",
    ),
  );
  assert.match(out, /value {8}"review"/u);
  assert.match(out, /distribution "review" \d\.\d{4} \| "(approve|deny)"/u);
  assert.match(
    out,
    /constraints {2}2 active of 4 \(1 inactive, 1 could not be evaluated\)/u,
  );
  assert.match(out, /always "review" when facts\.a > 0: satisfied/u);
  assert.match(
    out,
    /never "review" when facts\.b === 2: VIOLATED by the answer/u,
  );
  assert.match(
    out,
    /constraint 3 \(facts\.missing\.x === 1\) could not be evaluated: cannot read property "x"/u,
  );
  assert.doesNotMatch(out, /facts\.a > 100:/u);
  assert.match(
    out,
    /gold examples nearest the input \(3 of 3\)\n {4}0\.000 {2}"review"/u,
  );
  assert.match(
    out,
    /training cases nearest the input \(3 of 4; the release's dataset [0-9a-f]{12}… plus its adversarial sidecar, 1 other cached dataset for this id\)/u,
  );
  assert.match(out, /0\.000 {2}"review" {2}gold/u);
  assert.match(out, /"approve" {2}adversarial constraint-boundary/u);
  assert.match(out, /"approve" {2}synthetic \(anthropic\/claude-sonnet-5\)/u);
  assert.doesNotMatch(out, /decoy/u);
  assert.match(
    out,
    /verification passed: accuracy 1\.0000, ECE 0\.0000, Brier 0\.0000, pair consistency 1\.0000, 1 attested cases, 0 constraint violations/u,
  );
  assert.match(
    out,
    new RegExp(
      `trained {6}teacher anthropic/claude-sonnet-5, base model fixture, dataset ${release.sha256}`,
      "u",
    ),
  );
  assert.match(out, /result {3}"review"/u);
  assert.match(out, /distance mean over the inputs' leaf fields/u);

  const json = capture(root);
  assert.equal(
    await runCli(
      explainArgs(
        artifactRoot,
        bundle,
        cacheDir,
        "--call",
        "decide",
        "--input",
        "[1, 2]",
        "--json",
      ),
      json.io,
    ),
    0,
  );
  const document = JSON.parse(json.stdout());
  assert.equal(document.ok, true);
  assert.equal(document.result, "review");
  assert.equal(document.artifact.manifestSha256, artifact.manifestSha256);
  assert.equal(document.calls.length, 1);
  const [call] = document.calls;
  assert.equal(call.functionId, fixtureFunctionId);
  assert.equal(call.inArtifact, true);
  assert.deepEqual(call.inputs, facts(1, 2));
  const total = call.answer.distribution.reduce(
    (sum, entry) => sum + entry.probability,
    0,
  );
  assert.ok(Math.abs(total - 1) < 1e-9);
  assert.equal(call.answer.value, "review");
  assert.deepEqual(
    call.constraints.active.map((entry) => [entry.index, entry.satisfied]),
    [
      [0, true],
      [1, false],
    ],
  );
  assert.equal(call.constraints.inactive, 1);
  assert.deepEqual(
    call.constraints.errors.map((entry) => entry.index),
    [3],
  );
  assert.deepEqual(
    call.examples.nearest.map((entry) => entry.output),
    ["review", "approve", "deny"],
  );
  assert.equal(call.training.release, true);
  assert.equal(call.training.datasetSha256, release.sha256);
  assert.equal(call.training.total, 4);
  assert.deepEqual(
    call.training.nearest.map((entry) => [entry.origin, entry.output]),
    [
      ["gold", "review"],
      ["adversarial", "approve"],
      ["synthetic", "approve"],
      ["synthetic", "deny"],
    ],
  );
  assert.ok(
    call.training.nearest.every(
      (entry, index, all) =>
        index === 0 || all[index - 1].distance <= entry.distance,
    ),
  );
  assert.equal(call.verification.status, "passed");
  assert.equal(call.provenance.datasetSha256, release.sha256);
  assert.deepEqual(document.staleArtifactFunctions, []);

  // Without the release's dataset in the cache, the newest other one is shown and flagged.
  const noRelease = join(root, "artifact-other-dataset");
  await createFixtureArtifact(noRelease);
  const flagged = capture(root);
  assert.equal(
    await runCli(
      explainArgs(
        noRelease,
        bundle,
        cacheDir,
        "--call",
        "decide",
        "--input",
        "[1, 2]",
      ),
      flagged.io,
    ),
    0,
  );
  assert.match(
    flagged.stdout(),
    /NOT the release's dataset \(it is not cached\)/u,
  );
  assert.match(flagged.stdout(), /"decoy" {2}adversarial counterfactual/u);

  const emptyCache = capture(root);
  assert.equal(
    await runCli(
      explainArgs(
        artifactRoot,
        bundle,
        join(root, "no-cache"),
        "--call",
        "decide",
        "--input",
        "[1, 2]",
      ),
      emptyCache.io,
    ),
    0,
  );
  assert.match(
    emptyCache.stdout(),
    /training {5}no cached dataset for this function/u,
  );
});

test("explain says when the compiled function id is missing from the loaded artifact", async (t) => {
  const root = await scratch(t, "semantscript-cli-explain-missing-");
  const artifactRoot = join(root, "artifact");
  await createFixtureArtifact(artifactRoot);
  const bundle = join(root, "bundle.json");
  // The rebuilt bundle has the changed expression and no longer the trained one.
  await writeBundle(bundle, [changedFunctionId]);
  const cacheDir = join(root, "cache");

  const text = capture(root);
  assert.equal(
    await runCli(
      explainArgs(
        artifactRoot,
        bundle,
        cacheDir,
        "--call",
        "decideChanged",
        "--input",
        "[1, 2]",
      ),
      text.io,
    ),
    1,
  );
  const out = text.stdout();
  assert.match(out, new RegExp(`call 1: ${changedFunctionId}`, "u"));
  assert.match(
    out,
    /missing {6}the loaded artifact has no function nf_99999999…: the expression changed since the artifact was trained; run semantscript build and semantscript train, then explain again/u,
  );
  // The bundle's constraints still show which apply, with no answer to check.
  assert.match(out, /always "review" when facts\.a > 0: no answer to check/u);
  assert.match(out, /gold examples nearest the input \(3 of 3\)/u);
  assert.match(
    out,
    /artifact functions the bundle no longer has \(earlier versions of changed expressions\): nf_11111111…/u,
  );
  assert.match(out, /error {4}SemaUnknownFunctionError/u);

  const json = capture(root);
  assert.equal(
    await runCli(
      explainArgs(
        artifactRoot,
        bundle,
        cacheDir,
        "--call",
        "decideChanged",
        "--input",
        "[1, 2]",
        "--json",
      ),
      json.io,
    ),
    1,
  );
  const document = JSON.parse(json.stdout());
  assert.equal(document.ok, false);
  assert.equal(document.calls[0].inArtifact, false);
  assert.equal(document.calls[0].inBundle, true);
  assert.equal(document.calls[0].answer, null);
  assert.deepEqual(document.staleArtifactFunctions, [fixtureFunctionId]);
  assert.equal(document.error.name, "SemaUnknownFunctionError");

  // An id that neither the artifact nor the bundle knows points at a stale build.
  const stale = capture(root);
  const other = join(root, "other-bundle.json");
  await writeBundle(other, [fixtureFunctionId]);
  assert.equal(
    await runCli(
      explainArgs(
        artifactRoot,
        other,
        cacheDir,
        "--call",
        "decideChanged",
        "--input",
        "[1, 2]",
      ),
      stale.io,
    ),
    1,
  );
  assert.match(stale.stdout(), /neither the artifact nor the bundle knows it/u);
  assert.match(
    stale.stdout(),
    /constraints {2}not shown: the bundle does not describe this function/u,
  );
});

test("explain reads diagnostic-mode and below-threshold answers without changing what the program receives", async (t) => {
  const root = await scratch(t, "semantscript-cli-explain-modes-");
  const bundle = join(root, "bundle.json");
  await writeBundle(bundle, [fixtureFunctionId]);
  const cacheDir = join(root, "cache");

  const diagnosticRoot = join(root, "diagnostic");
  await createFixtureArtifact(diagnosticRoot, {
    transformManifest(manifest) {
      manifest.functions[0].runtime.resultMode = "diagnostic";
    },
  });
  const diagnostic = capture(root);
  assert.equal(
    await runCli(
      explainArgs(
        diagnosticRoot,
        bundle,
        cacheDir,
        "--call",
        "decide",
        "--input",
        "[1, 2]",
        "--json",
      ),
      diagnostic.io,
    ),
    0,
  );
  const document = JSON.parse(diagnostic.stdout());
  assert.equal(document.calls[0].resultMode, "diagnostic");
  assert.equal(document.calls[0].answer.value, "review");
  // The export returned the whole diagnostic result, as sema.withConfidence does.
  assert.deepEqual(document.result, document.calls[0].answer);
  const diagnosticText = capture(root);
  await runCli(
    explainArgs(
      diagnosticRoot,
      bundle,
      cacheDir,
      "--call",
      "decide",
      "--input",
      "[1, 2]",
    ),
    diagnosticText.io,
  );
  assert.match(
    diagnosticText.stdout(),
    /mode {9}sema\.withConfidence: the program receives the whole diagnostic result/u,
  );

  const thresholdRoot = join(root, "threshold");
  await createFixtureArtifact(thresholdRoot, {
    transformManifest(manifest) {
      Object.assign(manifest.functions[0].runtime, {
        confidenceThreshold: 1,
        policy: "scalar-top1",
      });
    },
  });
  const below = capture(root);
  assert.equal(
    await runCli(
      explainArgs(
        thresholdRoot,
        bundle,
        cacheDir,
        "--call",
        "decide",
        "--input",
        "[1, 2]",
      ),
      below.io,
    ),
    0,
  );
  assert.match(
    below.stdout(),
    /threshold {4}@confidence 1: below, the program receives SemaConfidenceError/u,
  );
  assert.match(below.stdout(), /error {4}SemaConfidenceError/u);

  // A sema.withConfidence site with an @confidence threshold: the runtime
  // returns the diagnostic before it checks the threshold, so no error.
  const bothRoot = join(root, "diagnostic-threshold");
  await createFixtureArtifact(bothRoot, {
    transformManifest(manifest) {
      Object.assign(manifest.functions[0].runtime, {
        resultMode: "diagnostic",
        confidenceThreshold: 0.99,
        policy: "scalar-top1",
      });
    },
  });
  const both = capture(root);
  assert.equal(
    await runCli(
      explainArgs(
        bothRoot,
        bundle,
        cacheDir,
        "--call",
        "decide",
        "--input",
        "[1, 2]",
      ),
      both.io,
    ),
    0,
  );
  assert.match(
    both.stdout(),
    /threshold {4}@confidence 0\.99: below, but sema\.withConfidence returns the diagnostic result either way/u,
  );
  assert.doesNotMatch(both.stdout(), /SemaConfidenceError/u);
  const bothJson = capture(root);
  await runCli(
    explainArgs(
      bothRoot,
      bundle,
      cacheDir,
      "--call",
      "decide",
      "--input",
      "[1, 2]",
      "--json",
    ),
    bothJson.io,
  );
  const bothDocument = JSON.parse(bothJson.stdout());
  assert.equal(bothDocument.error, null);
  assert.deepEqual(bothDocument.result, bothDocument.calls[0].answer);

  const fallbackRoot = join(root, "fallback");
  await createFixtureArtifact(fallbackRoot, {
    transformManifest(manifest) {
      Object.assign(manifest.functions[0].runtime, {
        confidenceThreshold: 1,
        policy: "scalar-top1",
        fallbackRef: "fallback.review-desk",
      });
    },
  });
  const fallback = capture(root);
  assert.equal(
    await runCli(
      explainArgs(
        fallbackRoot,
        bundle,
        cacheDir,
        "--call",
        "decide",
        "--input",
        "[1, 2]",
      ),
      fallback.io,
    ),
    0,
  );
  assert.match(
    fallback.stdout(),
    /below, the application's fallback fallback\.review-desk decides \(explain shows the model's answer\)/u,
  );
});

test("explain handles an export without a sema call, a missing bundle and bad arguments", async (t) => {
  const root = await scratch(t, "semantscript-cli-explain-edges-");
  const artifactRoot = join(root, "artifact");
  await createFixtureArtifact(artifactRoot);

  const plain = capture(root);
  assert.equal(
    await runCli(
      [
        "explain",
        "--artifact",
        artifactRoot,
        modulePath,
        "--call",
        "noSema",
        "--input",
        '"hi"',
      ],
      plain.io,
    ),
    0,
  );
  assert.match(plain.stdout(), /bundle {3}none found/u);
  assert.match(plain.stdout(), /the export made no sema call for this input/u);
  assert.match(plain.stdout(), /result {3}"HI"/u);

  const noBundle = capture(root);
  assert.equal(
    await runCli(
      [
        "explain",
        "--artifact",
        artifactRoot,
        modulePath,
        "--call",
        "decide",
        "--input",
        "[1, 2]",
      ],
      noBundle.io,
    ),
    0,
  );
  assert.match(noBundle.stdout(), /constraints {2}not shown: no bundle/u);
  assert.match(noBundle.stdout(), /value {8}"review"/u);

  const missingExport = capture(root);
  assert.equal(
    await runCli(
      ["explain", "--artifact", artifactRoot, modulePath, "--call", "absent"],
      missingExport.io,
    ),
    1,
  );
  assert.match(missingExport.stderr(), /no function export named absent/u);

  const noCall = capture(root);
  assert.equal(
    await runCli(
      ["explain", "--artifact", artifactRoot, modulePath],
      noCall.io,
    ),
    2,
  );
  assert.match(noCall.stderr(), /explain needs --call <export>/u);

  const badCount = capture(root);
  assert.equal(
    await runCli(
      ["explain", modulePath, "--call", "decide", "--neighbors", "two"],
      badCount.io,
    ),
    2,
  );
  assert.match(
    badCount.stderr(),
    /--neighbors must be a non-negative integer/u,
  );
});

test("explain says to run the application's CLI when the module reached another copy of the runtime", async (t) => {
  const root = await scratch(t, "semantscript-cli-explain-other-runtime-");
  const artifactRoot = join(root, "artifact");
  await createFixtureArtifact(artifactRoot);

  // A second, separately installed @semantscript/core: its classes and its
  // loaded-artifact state are different objects from the CLI's copy.
  const runtimeRoot = join(fixtures, "..", "..", "..", "runtime");
  const copy = join(root, "app", "node_modules", "@semantscript", "core");
  await mkdir(copy, { recursive: true });
  await cp(join(runtimeRoot, "package.json"), join(copy, "package.json"));
  await cp(join(runtimeRoot, "dist"), join(copy, "dist"), { recursive: true });
  const appModule = join(root, "app", "explain-module.mjs");
  await cp(modulePath, appModule);

  const other = capture(root);
  assert.equal(
    await runCli(
      [
        "explain",
        "--artifact",
        artifactRoot,
        appModule,
        "--call",
        "decide",
        "--input",
        "[1, 2]",
      ],
      other.io,
    ),
    1,
  );
  assert.match(
    other.stdout(),
    /error {4}SemaRuntimeNotLoadedError: no SemantScript artifact is loaded/u,
  );
  assert.match(
    other.stdout(),
    /reached a different @semantscript\/core than the CLI loaded the artifact into; run the CLI installed in the application/u,
  );
});
