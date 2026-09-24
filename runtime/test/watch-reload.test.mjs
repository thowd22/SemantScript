import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { setTimeout } from "node:timers";

import {
  createFixtureArtifact,
  fixtureFunctionId,
} from "./fixtures/artifact.mjs";

function waitFor(predicate, timeoutMilliseconds = 5000) {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const poll = () => {
      if (predicate()) return resolve();
      if (Date.now() - started > timeoutMilliseconds)
        return reject(new Error("timed out"));
      setTimeout(poll, 25);
    };
    poll();
  });
}

test("a watched artifact swaps in a new release when current.json changes and keeps the old one when the new one fails", async () => {
  const runtime = await import("../dist/index.js");
  const root = await mkdtemp(join(tmpdir(), "semantscript-runtime-watch-"));
  try {
    const first = await createFixtureArtifact(root);
    const reloads = [];
    const failures = [];
    const handle = await runtime.loadSemaArtifact(root, {
      watch: true,
      onReload: (next) => reloads.push(next.manifestSha256),
      onReloadError: (error) => failures.push(error),
    });
    assert.equal(handle.manifestSha256, first.manifestSha256);
    assert.equal(
      runtime.__sema.call(fixtureFunctionId, { facts: { a: 1, b: 2 } }),
      "review",
    );

    // A second release with a different manifest digest, published by rewriting the pointer.
    const second = await createFixtureArtifact(root, {
      transformManifest: (manifest) => {
        manifest.application.version = "0.0.1";
      },
    });
    assert.notEqual(second.manifestSha256, first.manifestSha256);
    await waitFor(() => reloads.length === 1);
    assert.deepEqual(reloads, [second.manifestSha256]);
    assert.equal(
      runtime.__sema.call(fixtureFunctionId, { facts: { a: 1, b: 2 } }),
      "review",
    );
    assert.throws(
      () => handle.call(fixtureFunctionId, { facts: { a: 1, b: 2 } }),
      runtime.SemaArtifactInactiveError,
      "the first handle is retired by the swap; compiled code goes through __sema",
    );

    // A broken pointer: the reload fails and the second release stays in service.
    await writeFile(join(root, "current.json"), "{ not json\n");
    await waitFor(() => failures.length === 1);
    assert.equal(reloads.length, 1);
    assert.equal(
      runtime.__sema.call(fixtureFunctionId, { facts: { a: 3, b: 4 } }),
      "review",
    );

    await runtime.closeSemaArtifact();
    assert.throws(
      () => runtime.__sema.call(fixtureFunctionId, { facts: { a: 1, b: 2 } }),
      runtime.SemaRuntimeNotLoadedError,
    );
    // Closed: pointer changes no longer reload anything.
    await createFixtureArtifact(root, {
      transformManifest: (manifest) => {
        manifest.application.version = "0.0.2";
      },
    });
    await new Promise((resolve) => setTimeout(resolve, 400));
    assert.equal(reloads.length, 1);
    assert.equal(failures.length, 1);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
