// scripts/version.mjs keeps one release version across the npm packages, the
// lockfile and the Python trainer; CI runs these tests on every push, so a
// version edited in one place only fails here.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  findMismatches,
  PUBLISHED_WORKSPACES,
  pythonVersion,
  setVersion,
} from "../../scripts/version.mjs";

const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));
const script = join(repositoryRoot, "scripts", "version.mjs");
const VERSIONED_FILES = [
  "package.json",
  "package-lock.json",
  ...PUBLISHED_WORKSPACES.map(({ directory }) => `${directory}/package.json`),
  "benchmarks/refund/package.json",
  "trainer/src/semantscript_trainer/_version.py",
  "runtime/src/version.ts",
  "compiler/src/version.ts",
];

async function copyOfVersionedFiles(t) {
  const root = await mkdtemp(join(tmpdir(), "semantscript-version-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (const file of VERSIONED_FILES) {
    await mkdir(dirname(join(root, file)), { recursive: true });
    await cp(join(repositoryRoot, file), join(root, file));
  }
  return root;
}

function runScript(args) {
  return spawnSync(process.execPath, [script, ...args], { encoding: "utf8" });
}

test("the repository's version is applied everywhere", async () => {
  const result = runScript(["check"]);
  assert.equal(result.status, 0, result.stderr);
  const version = runScript(["print"]).stdout.trim();
  assert.equal(
    version,
    JSON.parse(await readFile(join(repositoryRoot, "package.json"), "utf8"))
      .version,
  );
  assert.match(result.stdout, new RegExp(`version ${version} `, "u"));
});

test("set writes one version into every package, pin, lockfile entry and the trainer", async (t) => {
  const root = await copyOfVersionedFiles(t);
  setVersion(root, "3.4.5-rc.2");
  assert.deepEqual(findMismatches(root, "3.4.5-rc.2"), []);
  const cli = JSON.parse(
    await readFile(join(root, "cli/package.json"), "utf8"),
  );
  assert.equal(cli.version, "3.4.5-rc.2");
  assert.equal(cli.dependencies["@semantscript/core"], "3.4.5-rc.2");
  assert.equal(cli.dependencies["@semantscript/compiler"], "3.4.5-rc.2");
  const lock = JSON.parse(
    await readFile(join(root, "package-lock.json"), "utf8"),
  );
  assert.equal(lock.packages.framework.version, "3.4.5-rc.2");
  assert.equal(
    lock.packages["benchmarks/refund"].dependencies["@semantscript/core"],
    "3.4.5-rc.2",
  );
  assert.match(
    await readFile(
      join(root, "trainer/src/semantscript_trainer/_version.py"),
      "utf8",
    ),
    /__version__ = "3\.4\.5rc2"/u,
  );
  assert.match(
    await readFile(join(root, "runtime/src/version.ts"), "utf8"),
    /export const VERSION = "3\.4\.5-rc\.2";/u,
  );

  const check = runScript(["check", "--root", root, "--tag", "v3.4.5-rc.2"]);
  assert.equal(check.status, 0, check.stderr);
  const wrongTag = runScript(["check", "--root", root, "--tag", "v3.4.5"]);
  assert.equal(wrongTag.status, 1);
  assert.match(
    wrongTag.stderr,
    /tag v3\.4\.5 does not match the package\.json version 3\.4\.5-rc\.2: tag the release as v3\.4\.5-rc\.2/u,
  );
});

test("check names each place a hand edit left behind", async (t) => {
  const root = await copyOfVersionedFiles(t);
  setVersion(root, "1.0.0");
  const framework = JSON.parse(
    await readFile(join(root, "framework/package.json"), "utf8"),
  );
  framework.dependencies["@semantscript/core"] = "^0.9.0";
  await writeFile(
    join(root, "framework/package.json"),
    `${JSON.stringify(framework, null, 2)}\n`,
  );
  await writeFile(
    join(root, "trainer/src/semantscript_trainer/_version.py"),
    '__version__ = "0.9.0"\n',
  );
  const result = runScript(["check", "--root", root]);
  assert.equal(result.status, 1);
  assert.match(
    result.stderr,
    /framework\/package\.json dependencies\.@semantscript\/core: found "\^0\.9\.0", expected "1\.0\.0"/u,
  );
  assert.match(
    result.stderr,
    /trainer\/src\/semantscript_trainer\/_version\.py/u,
  );
  assert.match(result.stderr, /node scripts\/version\.mjs set 1\.0\.0/u);
});

test("the Python spelling follows PEP 440 and odd versions are refused", () => {
  assert.equal(pythonVersion("0.1.0"), "0.1.0");
  assert.equal(pythonVersion("1.2.3-alpha.4"), "1.2.3a4");
  assert.equal(pythonVersion("1.2.3-beta.0"), "1.2.3b0");
  assert.equal(pythonVersion("1.2.3-rc.1"), "1.2.3rc1");
  assert.throws(() => pythonVersion("1.2.3-dev.1"), /not a release version/u);
  assert.throws(() => pythonVersion("01.2.3"), /not a release version/u);
});
