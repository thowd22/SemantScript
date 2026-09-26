import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { existsSync } from "node:fs";
import {
  mkdir,
  mkdtemp,
  readdir,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import process from "node:process";
import test from "node:test";
import { setTimeout } from "node:timers";

import {
  createFixtureArtifact,
  fixtureFunctionId,
} from "../../runtime/test/fixtures/artifact.mjs";
import { pointerBytes, runCli } from "../dist/index.js";

function capture(cwd) {
  const out = [];
  const err = [];
  return {
    io: {
      cwd,
      env: { ...process.env, SEMANTSCRIPT_ARTIFACT: undefined },
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

function waitFor(predicate, timeoutMilliseconds = 10_000) {
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

const recent = () => new Date().toISOString().replace(/\.[0-9]{3}Z$/u, "Z");

/** Three releases, oldest to newest; the newest is current. */
async function threeReleases(root, options = {}) {
  const make = (version, createdAt, extra = () => {}) =>
    createFixtureArtifact(root, {
      transformManifest: (manifest) => {
        manifest.application.version = version;
        manifest.build.createdAt = createdAt;
        extra(manifest);
      },
    });
  const oldest = await make("0.0.1", "2020-01-01T00:00:00Z", options.oldest);
  const middle = await make("0.0.2", "2020-01-02T00:00:00Z", options.middle);
  const newestAt = recent();
  const newest = await make("0.0.3", newestAt, options.newest);
  return {
    oldest: oldest.manifestSha256,
    middle: middle.manifestSha256,
    newest: newest.manifestSha256,
    newestAt,
  };
}

async function run(root, args) {
  const output = capture(root);
  const code = await runCli(
    ["releases", ...args, "--artifact", root],
    output.io,
  );
  return { code, stdout: output.stdout(), stderr: output.stderr() };
}

async function currentDigest(root) {
  return JSON.parse(await readFile(join(root, "current.json"), "utf8"))
    .manifestSha256;
}

test("releases lists every release newest first with its date, digest, verification and the current marker", async (t) => {
  const root = await scratch(t, "semantscript-releases-list-");
  const { oldest, middle, newest, newestAt } = await threeReleases(root);
  // Not releases: an in-progress export, a pruning leftover, a stray file.
  await mkdir(join(root, "releases", ".staging-abc"), { recursive: true });
  await writeFile(join(root, "releases", "notes.txt"), "x");
  // A release whose manifest does not hash to its name.
  const forged = `sha256-${"f".repeat(64)}`;
  await mkdir(join(root, "releases", forged));
  await writeFile(join(root, "releases", forged, "manifest.json"), "{}\n");

  const listed = await run(root, []);
  assert.equal(listed.code, 0, listed.stderr);
  const text = listed.stdout;
  assert.match(text, new RegExp(`current  releases/sha256-${newest}`, "u"));
  const rows = text
    .split("\n")
    .filter(
      (line) => /[a-f0-9]{12}/u.test(line) && !line.startsWith("current"),
    );
  assert.match(
    rows[0],
    new RegExp(
      `^\\* +${newestAt} +${newest.slice(0, 12)} +runtime-fixture@0\\.0\\.3 +1/1 passed +1\\.0000 +0\\.0000 +0$`,
      "u",
    ),
  );
  assert.match(
    rows[1],
    new RegExp(
      `^ +2020-01-02T00:00:00Z +${middle.slice(0, 12)} +runtime-fixture@0\\.0\\.2 +1/1 passed`,
      "u",
    ),
  );
  assert.match(
    rows[2],
    new RegExp(
      `^ +2020-01-01T00:00:00Z +${oldest.slice(0, 12)} +runtime-fixture@0\\.0\\.1`,
      "u",
    ),
  );
  assert.match(rows[3], /^ +- +ffffffffffff +- +invalid/u);
  assert.match(
    text,
    new RegExp(`${forged}: invalid: manifest.json hashes to`, "u"),
  );
  assert.doesNotMatch(text, /staging|notes\.txt/u);

  // `list` is the default subcommand, and --json carries the full summaries.
  const json = await run(root, ["list", "--json"]);
  assert.equal(json.code, 0, json.stderr);
  const document = JSON.parse(json.stdout);
  assert.equal(document.kind, "semantscript.releases");
  assert.equal(document.version, 1);
  assert.deepEqual(document.current, {
    release: `releases/sha256-${newest}`,
    manifestSha256: newest,
  });
  assert.deepEqual(
    document.releases.map((release) => [
      release.manifestSha256,
      release.current,
      release.valid,
    ]),
    [
      [newest, true, true],
      [middle, false, true],
      [oldest, false, true],
      ["f".repeat(64), false, false],
    ],
  );
  assert.equal(document.releases[1].createdAt, "2020-01-02T00:00:00Z");
  assert.deepEqual(document.releases[1].verification, {
    passed: 1,
    total: 1,
    minAccuracy: 1,
    maxEce: 0,
    constraintViolations: 0,
  });
  assert.equal(document.releases[1].functions[0].id, fixtureFunctionId);
  assert.equal(document.releases[1].functions[0].status, "passed");

  // No artifact root yet: an empty listing, not an error.
  const empty = await run(join(root, "missing"), []);
  assert.equal(empty.code, 0);
  assert.match(
    empty.stdout,
    /current {2}none \(no current\.json\)\nno releases under/u,
  );
});

test("releases show prints one release's per-function verification table", async (t) => {
  const root = await scratch(t, "semantscript-releases-show-");
  const { middle, newest } = await threeReleases(root);

  const shown = await run(root, ["show", middle.slice(0, 10)]);
  assert.equal(shown.code, 0, shown.stderr);
  assert.match(
    shown.stdout,
    new RegExp(
      `^release {6}releases/sha256-${middle}\ncreated {6}2020-01-02T00:00:00Z\napplication  runtime-fixture@0\\.0\\.2\nfunction +verification +accuracy +ece +brier +pairs +attested +violations +heads\n`,
      "u",
    ),
  );
  assert.match(
    shown.stdout,
    /nf_11111111… +passed +1\.0000 +0\.0000 +0\.0000 +1\.0000 +1 +0 +1\.0000/u,
  );

  const current = await run(root, [
    "show",
    `releases/sha256-${newest}`,
    "--json",
  ]);
  assert.equal(current.code, 0, current.stderr);
  const document = JSON.parse(current.stdout);
  assert.equal(document.current, true);
  assert.equal(document.application.version, "0.0.3");

  const unknown = await run(root, ["show", "0123456789"]);
  assert.equal(unknown.code, 1);
  assert.match(
    unknown.stderr,
    /^RELEASE_NOT_FOUND: no release under releases\/ matches 0123456789/u,
  );
  assert.equal((await run(root, ["show"])).code, 2);
  assert.equal(
    (await run(root, ["show", "abc"])).code,
    2,
    "too short a prefix",
  );
  assert.equal((await run(root, ["frobnicate"])).code, 2);
});

test("releases rollback rewrites the pointer atomically to the previous or a named release and never removes one", async (t) => {
  const root = await scratch(t, "semantscript-releases-rollback-");
  const { oldest, middle, newest } = await threeReleases(root);
  const releasesBefore = (await readdir(join(root, "releases"))).sort();

  const dry = await run(root, ["rollback", "--dry-run"]);
  assert.equal(dry.code, 0, dry.stderr);
  assert.match(dry.stdout, /would point .*current\.json/u);
  assert.equal(await currentDigest(root), newest);

  const first = await run(root, ["rollback"]);
  assert.equal(first.code, 0, first.stderr);
  assert.match(
    first.stdout,
    new RegExp(
      `from releases/sha256-${newest}\n {2}to {3}releases/sha256-${middle} \\(runtime-fixture@0\\.0\\.2, created 2020-01-02T00:00:00Z\\)`,
      "u",
    ),
  );
  // The exact bytes the trainer's exporter writes (sorted keys, indent 2, newline).
  const bytes = await readFile(join(root, "current.json"), "utf8");
  assert.equal(bytes, pointerBytes(middle));
  assert.deepEqual(Object.keys(JSON.parse(bytes)), [
    "kind",
    "manifestSha256",
    "pointerVersion",
    "release",
  ]);
  assert.deepEqual(JSON.parse(bytes), {
    kind: "semantscript.artifact-pointer",
    manifestSha256: middle,
    pointerVersion: 1,
    release: `releases/sha256-${middle}`,
  });

  const second = await run(root, ["rollback", "--json"]);
  assert.equal(second.code, 0, second.stderr);
  assert.deepEqual(
    (({ from, to, dryRun }) => ({ from, to, dryRun }))(
      JSON.parse(second.stdout),
    ),
    { from: middle, to: oldest, dryRun: false },
  );
  const none = await run(root, ["rollback"]);
  assert.equal(none.code, 1);
  assert.match(
    none.stderr,
    /^RELEASE_NO_PREVIOUS: no valid release was created before the current one/u,
  );

  // Rolling forward is the same operation with a name: a unique prefix works.
  const promoted = await run(root, ["promote", newest.slice(0, 7)]);
  assert.equal(promoted.code, 0, promoted.stderr);
  assert.equal(await currentDigest(root), newest);
  const named = await run(root, ["rollback", `sha256-${oldest}`]);
  assert.equal(named.code, 0, named.stderr);
  assert.equal(await currentDigest(root), oldest);

  const again = await run(root, ["rollback", oldest]);
  assert.equal(again.code, 1);
  assert.match(again.stderr, /^RELEASE_ALREADY_CURRENT/u);
  assert.equal((await run(root, ["promote"])).code, 2);
  assert.equal((await run(root, ["rollback", oldest, middle])).code, 2);

  assert.deepEqual(
    (await readdir(join(root, "releases"))).sort(),
    releasesBefore,
  );
  assert.deepEqual(
    (await readdir(root)).filter((name) => name.startsWith(".current-")),
    [],
    "no temporary pointer left behind",
  );
});

test("releases rollback refuses ambiguous, tampered, symlinked and unverified targets and repairs a broken pointer by name", async (t) => {
  const root = await scratch(t, "semantscript-releases-refuse-");
  const { oldest, middle, newest } = await threeReleases(root, {
    oldest: (manifest) => {
      manifest.functions[0].verification.status = "failed";
    },
  });

  // Two directories sharing a 7-character prefix.
  for (const suffix of ["0", "1"]) {
    const name = `sha256-abcdef0${suffix}${"0".repeat(56)}`;
    await mkdir(join(root, "releases", name));
  }
  const ambiguous = await run(root, ["rollback", "abcdef0"]);
  assert.equal(ambiguous.code, 1);
  assert.match(
    ambiguous.stderr,
    /^RELEASE_AMBIGUOUS: abcdef0 matches sha256-abcdef00/u,
  );

  // A tampered model file in the middle release: the runtime would refuse it.
  const tokenizer = join(
    root,
    "releases",
    `sha256-${middle}`,
    "tokenizer",
    "tokenizer.json",
  );
  const original = await readFile(tokenizer);
  const tampered = Buffer.from(original);
  tampered[tampered.length - 2] = 0x20;
  await writeFile(tokenizer, tampered);
  const integrity = await run(root, ["rollback"]);
  assert.equal(integrity.code, 1);
  assert.match(
    integrity.stderr,
    /^RELEASE_INTEGRITY: sha256-[a-f0-9]{64}: tokenizer\/tokenizer\.json hashes to/u,
  );
  assert.equal(await currentDigest(root), newest);
  await writeFile(tokenizer, original);

  const unverified = await run(root, ["rollback", oldest]);
  assert.equal(unverified.code, 1);
  assert.match(unverified.stderr, /^RELEASE_UNVERIFIED: .* is failed/u);
  assert.equal(await currentDigest(root), newest);

  // A symlinked release directory is not a release the runtime will load.
  const linked = `sha256-${"e".repeat(64)}`;
  await symlink(
    join(root, "releases", `sha256-${middle}`),
    join(root, "releases", linked),
  );
  const symlinked = await run(root, ["rollback", "e".repeat(64)]);
  assert.equal(symlinked.code, 1);
  assert.match(
    symlinked.stderr,
    /^RELEASE_INTEGRITY: .*not a non-symlink directory/u,
  );

  // A corrupt pointer: rollback needs a name, and a named one repairs it.
  await writeFile(join(root, "current.json"), "{ not json\n");
  const guess = await run(root, ["rollback"]);
  assert.equal(guess.code, 1);
  assert.match(
    guess.stderr,
    /^POINTER_INVALID: (?!POINTER_INVALID).*name the release to switch to/u,
  );
  const repaired = await run(root, ["rollback", middle]);
  assert.equal(repaired.code, 0, repaired.stderr);
  assert.match(repaired.stdout, /from \(no valid pointer\)/u);
  assert.equal(await currentDigest(root), middle);
});

test("releases rollback and promote refuse a release the runtime's loader would refuse", async (t) => {
  const root = await scratch(t, "semantscript-releases-incompatible-");
  const { oldest, middle, newest } = await threeReleases(root, {
    // Hashes and verification are fine; only the runtime compatibility check fails.
    middle: (manifest) => {
      manifest.compatibility.minimumRuntimeVersion = "99.0.0";
    },
  });

  const rollback = await run(root, ["rollback"]);
  assert.equal(rollback.code, 1);
  assert.match(
    rollback.stderr,
    new RegExp(
      `^RELEASE_REJECTED: sha256-${middle}: the runtime refuses to load it \\(SEMA_ARTIFACT_INCOMPATIBLE: artifact requires runtime 99\\.0\\.0`,
      "u",
    ),
  );
  assert.equal(await currentDigest(root), newest);

  const promote = await run(root, ["promote", middle]);
  assert.equal(promote.code, 1);
  assert.match(promote.stderr, /^RELEASE_REJECTED: /u);
  assert.equal(await currentDigest(root), newest);

  // A compatible release still switches.
  const named = await run(root, ["rollback", oldest]);
  assert.equal(named.code, 0, named.stderr);
  assert.equal(await currentDigest(root), oldest);
});

test("releases prune removes by count or age and never the current release, invalid ones or staging directories", async (t) => {
  const root = await scratch(t, "semantscript-releases-prune-");
  const { oldest, middle, newest } = await threeReleases(root);
  const invalid = `sha256-${"f".repeat(64)}`;
  await mkdir(join(root, "releases", invalid));
  await mkdir(join(root, "releases", ".staging-inprogress"));
  await mkdir(join(root, "releases", ".pruning-leftover"));
  await writeFile(join(root, "releases", ".pruning-leftover", "x"), "12345");
  const exists = (digest) =>
    existsSync(join(root, "releases", `sha256-${digest}`));

  // Roll back to the oldest: it is current, so it stays even though it is oldest.
  assert.equal((await run(root, ["rollback", oldest])).code, 0);

  const dry = await run(root, ["prune", "--keep", "1", "--dry-run"]);
  assert.equal(dry.code, 0, dry.stderr);
  assert.match(
    dry.stdout,
    new RegExp(
      `would remove +created +size\n.*\n${middle.slice(0, 12)} +2020-01-02T00:00:00Z +[0-9.]+ KiB`,
      "u",
    ),
  );
  assert.ok(
    exists(middle) && existsSync(join(root, "releases", ".pruning-leftover")),
  );

  const pruned = await run(root, ["prune", "--keep", "1", "--json"]);
  assert.equal(pruned.code, 0, pruned.stderr);
  const report = JSON.parse(pruned.stdout);
  assert.deepEqual(
    report.removed.map((item) => item.manifestSha256),
    [middle],
  );
  assert.deepEqual(report.kept, [newest, oldest]);
  assert.deepEqual(
    report.invalid.map((item) => item.manifestSha256),
    ["f".repeat(64)],
  );
  assert.equal(report.leftoversRemoved, 1);
  assert.ok(report.bytesFreed > report.removed[0].bytes);
  assert.ok(!exists(middle));
  assert.ok(exists(oldest) && exists(newest));
  assert.ok(existsSync(join(root, "releases", invalid)));
  assert.ok(existsSync(join(root, "releases", ".staging-inprogress")));
  assert.equal(
    (await readdir(join(root, "releases"))).filter((name) =>
      name.startsWith(".pruning-"),
    ).length,
    0,
  );

  // By age: the newest is recent, the current (oldest) is protected, so nothing goes.
  const byAge = await run(root, ["prune", "--older-than", "30d"]);
  assert.equal(byAge.code, 0, byAge.stderr);
  assert.match(byAge.stdout, /nothing removed/u);
  assert.ok(exists(oldest) && exists(newest));
  // Current back on the newest: the old one is now older than 30 days.
  assert.equal((await run(root, ["promote", newest])).code, 0);
  const aged = await run(root, ["prune", "--older-than", "30d", "--keep", "0"]);
  assert.equal(aged.code, 0, aged.stderr);
  assert.match(aged.stdout, /removed +created +size/u);
  assert.match(aged.stdout, /kept 1 release; freed/u);
  assert.ok(!exists(oldest) && exists(newest));
  const keepAll = await run(root, ["prune", "--keep", "0"]);
  assert.match(
    keepAll.stdout,
    /nothing removed/u,
    "the only release is current",
  );
  assert.ok(exists(newest));

  for (const args of [
    ["prune"],
    ["prune", "--keep", "x"],
    ["prune", "--older-than", "5y"],
    ["prune", "--keep", "1", "extra"],
  ]) {
    assert.equal((await run(root, args)).code, 2, args.join(" "));
  }
  // A well-formed pointer naming a release that is not on disk.
  const pointerFile = join(root, "current.json");
  const saved = await readFile(pointerFile);
  await writeFile(pointerFile, pointerBytes("d".repeat(64)));
  const dangling = await run(root, ["prune", "--keep", "0"]);
  assert.equal(dangling.code, 1);
  assert.match(
    dangling.stderr,
    /^POINTER_INVALID: .*names releases\/sha256-d{64}, which is missing or invalid/u,
  );
  assert.ok(exists(newest));
  await writeFile(pointerFile, saved);

  await rm(join(root, "current.json"));
  const noPointer = await run(root, ["prune", "--keep", "0"]);
  assert.equal(noPointer.code, 1);
  assert.match(noPointer.stderr, /^POINTER_INVALID: .*prune never guesses/u);
  assert.ok(exists(newest));
});

test("releases finds the subcommand after leading flags", async (t) => {
  const root = await scratch(t, "semantscript-releases-flags-");
  const { middle } = await threeReleases(root);
  for (const args of [
    ["--artifact", root, "show", middle.slice(0, 12)],
    ["--json", "list", "--artifact", root],
    ["--artifact", root, "rollback", "--dry-run"],
  ]) {
    const output = capture(root);
    const code = await runCli(["releases", ...args], output.io);
    assert.equal(code, 0, `${args.join(" ")}: ${output.stderr()}`);
  }
});

test("a running process with watch enabled swaps to the release a rollback points at", async (t) => {
  const root = await scratch(t, "semantscript-releases-watch-");
  const { middle, newest } = await threeReleases(root);
  const runtime = await import("@semantscript/core");
  const reloads = [];
  const failures = [];
  const handle = await runtime.loadSemaArtifact(root, {
    watch: true,
    onReload: (next) => reloads.push(next.manifestSha256),
    onReloadError: (error) => failures.push(error),
  });
  try {
    assert.equal(handle.manifestSha256, newest);
    assert.equal(
      runtime.__sema.call(fixtureFunctionId, { facts: { a: 1, b: 2 } }),
      "review",
    );

    const rolled = await run(root, ["rollback"]);
    assert.equal(rolled.code, 0, rolled.stderr);
    await waitFor(() => reloads.length > 0 || failures.length > 0);
    assert.deepEqual(failures, []);
    assert.deepEqual(reloads, [middle]);
    assert.equal(
      runtime.__sema.call(fixtureFunctionId, { facts: { a: 1, b: 2 } }),
      "review",
    );
    assert.throws(
      () => handle.call(fixtureFunctionId, { facts: { a: 1, b: 2 } }),
      runtime.SemaArtifactInactiveError,
      "the handle of the rolled-back-from release is retired",
    );

    // And forward again.
    assert.equal((await run(root, ["promote", newest])).code, 0);
    await waitFor(() => reloads.length > 1 || failures.length > 0);
    assert.deepEqual(reloads, [middle, newest]);
    assert.equal(
      runtime.__sema.call(fixtureFunctionId, { facts: { a: 3, b: 4 } }),
      "review",
    );
  } finally {
    await runtime.closeSemaArtifact();
  }
});
