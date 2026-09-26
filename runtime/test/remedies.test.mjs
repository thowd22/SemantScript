import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import process from "node:process";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  ArtifactLoadError,
  loadSemaArtifact,
  remedy,
  remedyFamily,
  SemaRuntimeNotLoadedError,
  SemaUnknownFunctionError,
} from "../dist/index.js";
import { createFixtureArtifact } from "./fixtures/artifact.mjs";

const repository = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

async function scratch(t) {
  const root = await mkdtemp(join(tmpdir(), "semantscript-remedies-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return root;
}

test("the generated remedy tables and the catalogue match diagnostics/remedies.json", () => {
  const result = spawnSync(
    process.execPath,
    [join(repository, "scripts", "generate-remedies.mjs"), "--check"],
    { encoding: "utf8" },
  );
  assert.equal(result.status, 0, result.stderr);
});

test("the remedies whose next step is explain or the stub artifact name it", async () => {
  const source = JSON.parse(
    await readFile(join(repository, "diagnostics", "remedies.json"), "utf8"),
  );
  const fixes = new Map(source.remedies.map((entry) => [entry.id, entry.fix]));
  for (const id of ["test-example-mismatch", "unknown-function"]) {
    assert.match(fixes.get(id), /semantscript explain/, id);
  }
  for (const id of [
    "runtime-not-loaded",
    "artifact-missing",
    "editor-no-artifact",
  ]) {
    assert.match(
      fixes.get(id),
      /loadSemaStubArtifact\(\) from @semantscript\/core\/testing/,
      id,
    );
  }
  // A failed retrain still caches its dataset, and explain lists the gold
  // example beside the nearest teacher-labelled cases from it when a release
  // is already published.
  assert.match(
    fixes.get("gold-check-example"),
    /when a release is already published, semantscript explain/,
  );
  assert.doesNotMatch(fixes.get("estimate-killed"), /--batch-size|--device/);
});

test("remedy fills a template and rejects unknown ids and parameters", () => {
  assert.equal(
    remedy("run-no-export", { exports: "decideRefund, verdict" }),
    "pass --call one of the module's function exports: decideRefund, verdict",
  );
  assert.equal(remedyFamily("type-no-cache"), "verifier");
  assert.match(remedy("runtime-not-loaded"), /loadSemaArtifact\(\)/u);
  assert.throws(() => remedy("no-such-remedy"), /unknown remedy/u);
  assert.throws(() => remedy("run-no-export", {}), /needs parameter exports/u);
  assert.throws(
    () => remedy("run-no-export", { exports: "a", extra: 1 }),
    /has no parameter extra/u,
  );
});

test("a missing or empty artifact root names semantscript train", async (t) => {
  const root = await scratch(t);
  const missing = join(root, "absent");
  await assert.rejects(loadSemaArtifact(missing), (error) => {
    assert.ok(error instanceof ArtifactLoadError);
    assert.equal(error.code, "SEMA_ARTIFACT_PATH");
    assert.equal(error.detail, "cannot inspect artifact directory");
    assert.equal(error.remedy, remedy("artifact-missing", { path: missing }));
    assert.equal(error.message, `${error.detail}; next: ${error.remedy}`);
    assert.match(
      error.message,
      /next: run semantscript train to publish an artifact at /u,
    );
    return true;
  });
  const empty = join(root, "empty");
  await mkdir(empty);
  await assert.rejects(loadSemaArtifact(empty), (error) => {
    assert.ok(error instanceof ArtifactLoadError);
    assert.match(error.remedy, /^run semantscript train/u);
    return true;
  });
});

test("a corrupt pointer names releases rollback, and a stale call names build then train", async (t) => {
  const root = await scratch(t);
  await createFixtureArtifact(root);
  await writeFile(join(root, "current.json"), "{not json");
  await assert.rejects(loadSemaArtifact(root), (error) => {
    assert.ok(error instanceof ArtifactLoadError);
    assert.equal(error.code, "SEMA_ARTIFACT_INVALID_JSON");
    assert.equal(error.remedy, remedy("artifact-corrupt"));
    assert.match(
      error.message,
      /; next: run semantscript releases list to find an intact release, then switch to it with semantscript releases rollback <release>/u,
    );
    return true;
  });

  const unknown = new SemaUnknownFunctionError(`nf_${"f".repeat(64)}`);
  assert.match(
    unknown.message,
    /run semantscript build, then semantscript train/u,
  );
  assert.equal(unknown.remedy, remedy("unknown-function"));
  const notLoaded = new SemaRuntimeNotLoadedError();
  assert.match(
    notLoaded.message,
    /^no SemantScript artifact is loaded; next: await loadSemaArtifact\(\)/u,
  );
  assert.match(notLoaded.remedy, /run semantscript train/u);
});
