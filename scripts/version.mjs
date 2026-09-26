// One version for every published package. The root package.json "version"
// is the source; `set` writes it everywhere else and `check` fails naming each
// place that disagrees. docs/releasing.md describes the release steps.
//
//   node scripts/version.mjs check [--tag v1.2.3] [--root <dir>]
//   node scripts/version.mjs set <version> [--root <dir>]
//   node scripts/version.mjs print [python]
//
// Places kept in step:
//   - the four published workspaces' package.json "version"
//   - every @semantscript/* and semantscript dependency pin inside the
//     repository's workspaces (exact versions, so one release installs together)
//   - package-lock.json's entries for those workspaces
//   - trainer/src/semantscript_trainer/_version.py (pyproject.toml reads it)
//   - runtime/src/version.ts and compiler/src/version.ts (the runtime's
//     default runtimeVersion and the compiler version the CLI hands the trainer)
import { readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

/** The workspaces published to npm, in dependency order (publish order). */
export const PUBLISHED_WORKSPACES = [
  { directory: "runtime", name: "@semantscript/core" },
  { directory: "compiler", name: "@semantscript/compiler" },
  { directory: "framework", name: "@semantscript/framework" },
  { directory: "cli", name: "semantscript" },
];
/** Private workspaces whose internal dependency pins still follow the version. */
const PINNING_WORKSPACES = ["benchmarks/refund"];
const INTERNAL_NAMES = new Set(PUBLISHED_WORKSPACES.map(({ name }) => name));
const PYTHON_VERSION_FILE = "trainer/src/semantscript_trainer/_version.py";
const TYPESCRIPT_VERSION_FILES = [
  "runtime/src/version.ts",
  "compiler/src/version.ts",
];
const DEPENDENCY_FIELDS = [
  "dependencies",
  "devDependencies",
  "peerDependencies",
  "optionalDependencies",
];
const SEMVER =
  /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-(alpha|beta|rc)\.(0|[1-9]\d*))?$/u;

/**
 * The PEP 440 spelling of a release version: 1.2.3 stays, 1.2.3-rc.1 becomes
 * 1.2.3rc1. Only the pre-release tags PyPI and npm both order the same way
 * (alpha, beta, rc) are accepted.
 */
export function pythonVersion(version) {
  const match = SEMVER.exec(version);
  if (match === null) {
    throw new Error(
      `${version} is not a release version (MAJOR.MINOR.PATCH, optionally -alpha.N, -beta.N or -rc.N)`,
    );
  }
  const [, major, minor, patch, tag, number] = match;
  const base = `${major}.${minor}.${patch}`;
  if (tag === undefined) return base;
  const short = { alpha: "a", beta: "b", rc: "rc" }[tag];
  return `${base}${short}${number}`;
}

function readJson(root, path) {
  return JSON.parse(readFileSync(join(root, path), "utf8"));
}

function writeJson(root, path, value) {
  writeFileSync(join(root, path), `${JSON.stringify(value, null, 2)}\n`);
}

function pythonSource(version) {
  return [
    '"""The release version, shared with the npm packages (scripts/version.mjs writes it)."""',
    "",
    `__version__ = "${pythonVersion(version)}"`,
    "",
  ].join("\n");
}

function typescriptSource(version) {
  return [
    "// The release version, shared by every SemantScript package.",
    "// scripts/version.mjs writes this file; do not edit it by hand.",
    `export const VERSION = "${version}";`,
    "",
  ].join("\n");
}

function readSourceVersion(root) {
  const version = readJson(root, "package.json").version;
  if (typeof version !== "string" || !SEMVER.test(version)) {
    throw new Error(
      `package.json version ${String(version)} is not a release version`,
    );
  }
  return version;
}

/** Every place that disagrees with `version`, as "file: found X, expected Y". */
export function findMismatches(root, version) {
  const problems = [];
  const expectEqual = (where, found, expected) => {
    if (found !== expected) {
      problems.push(
        `${where}: found ${JSON.stringify(found)}, expected ${JSON.stringify(expected)}`,
      );
    }
  };
  for (const { directory, name } of PUBLISHED_WORKSPACES) {
    const pkg = readJson(root, join(directory, "package.json"));
    expectEqual(`${directory}/package.json name`, pkg.name, name);
    expectEqual(`${directory}/package.json version`, pkg.version, version);
  }
  for (const directory of [
    ...PUBLISHED_WORKSPACES.map(({ directory }) => directory),
    ...PINNING_WORKSPACES,
  ]) {
    const pkg = readJson(root, join(directory, "package.json"));
    for (const field of DEPENDENCY_FIELDS) {
      for (const [dependency, range] of Object.entries(pkg[field] ?? {})) {
        if (INTERNAL_NAMES.has(dependency)) {
          expectEqual(
            `${directory}/package.json ${field}.${dependency}`,
            range,
            version,
          );
        }
      }
    }
  }
  const lock = readJson(root, "package-lock.json");
  expectEqual("package-lock.json version", lock.version, version);
  expectEqual(
    'package-lock.json packages[""].version',
    lock.packages?.[""]?.version,
    version,
  );
  for (const directory of [
    ...PUBLISHED_WORKSPACES.map(({ directory }) => directory),
    ...PINNING_WORKSPACES,
  ]) {
    const entry = lock.packages?.[directory];
    if (entry === undefined) {
      problems.push(`package-lock.json: no packages["${directory}"] entry`);
      continue;
    }
    if (PUBLISHED_WORKSPACES.some((w) => w.directory === directory)) {
      expectEqual(
        `package-lock.json packages["${directory}"].version`,
        entry.version,
        version,
      );
    }
    for (const field of DEPENDENCY_FIELDS) {
      for (const [dependency, range] of Object.entries(entry[field] ?? {})) {
        if (INTERNAL_NAMES.has(dependency)) {
          expectEqual(
            `package-lock.json packages["${directory}"].${field}.${dependency}`,
            range,
            version,
          );
        }
      }
    }
  }
  expectEqual(
    PYTHON_VERSION_FILE,
    readFileSync(join(root, PYTHON_VERSION_FILE), "utf8"),
    pythonSource(version),
  );
  for (const file of TYPESCRIPT_VERSION_FILES) {
    expectEqual(
      file,
      readFileSync(join(root, file), "utf8"),
      typescriptSource(version),
    );
  }
  return problems;
}

/** Writes `version` into the root package.json and every place it is kept. */
export function setVersion(root, version) {
  pythonVersion(version);
  const rootPackage = readJson(root, "package.json");
  rootPackage.version = version;
  writeJson(root, "package.json", rootPackage);
  const pinDependencies = (record) => {
    for (const field of DEPENDENCY_FIELDS) {
      for (const dependency of Object.keys(record[field] ?? {})) {
        if (INTERNAL_NAMES.has(dependency)) record[field][dependency] = version;
      }
    }
  };
  const published = new Set(
    PUBLISHED_WORKSPACES.map(({ directory }) => directory),
  );
  for (const directory of [...published, ...PINNING_WORKSPACES]) {
    const path = join(directory, "package.json");
    const pkg = readJson(root, path);
    if (published.has(directory)) pkg.version = version;
    pinDependencies(pkg);
    writeJson(root, path, pkg);
  }
  const lock = readJson(root, "package-lock.json");
  lock.version = version;
  if (lock.packages?.[""] !== undefined) lock.packages[""].version = version;
  for (const directory of [...published, ...PINNING_WORKSPACES]) {
    const entry = lock.packages?.[directory];
    if (entry === undefined) continue;
    if (published.has(directory)) entry.version = version;
    pinDependencies(entry);
  }
  writeJson(root, "package-lock.json", lock);
  writeFileSync(join(root, PYTHON_VERSION_FILE), pythonSource(version));
  for (const file of TYPESCRIPT_VERSION_FILES) {
    writeFileSync(join(root, file), typescriptSource(version));
  }
}

function main(argv) {
  const [command, ...rest] = argv;
  let root = resolve(fileURLToPath(new URL("..", import.meta.url)));
  let tag;
  const positionals = [];
  for (let index = 0; index < rest.length; index += 1) {
    const argument = rest[index];
    if (argument === "--root") root = resolve(rest[(index += 1)] ?? "");
    else if (argument === "--tag") tag = rest[(index += 1)];
    else positionals.push(argument);
  }
  if (command === "set" && positionals.length === 1) {
    setVersion(root, positionals[0].replace(/^v/u, ""));
    const problems = findMismatches(root, positionals[0].replace(/^v/u, ""));
    if (problems.length > 0) {
      console.error(problems.join("\n"));
      return 1;
    }
    console.log(
      `version ${positionals[0].replace(/^v/u, "")} written; run npm install to refresh node_modules`,
    );
    return 0;
  }
  if (command === "check" && positionals.length === 0) {
    const version = readSourceVersion(root);
    const problems = findMismatches(root, version);
    if (
      tag !== undefined &&
      tag.replace(/^refs\/tags\//u, "") !== `v${version}`
    ) {
      problems.push(
        `tag ${tag} does not match package.json version ${version} (expected v${version})`,
      );
    }
    if (problems.length > 0) {
      console.error(
        `version ${version} is not applied everywhere (fix with: node scripts/version.mjs set ${version}):\n${problems.join("\n")}`,
      );
      return 1;
    }
    console.log(
      `version ${version} (Python ${pythonVersion(version)}) is consistent`,
    );
    return 0;
  }
  if (command === "print" && positionals.length <= 1) {
    const version = readSourceVersion(root);
    console.log(positionals[0] === "python" ? pythonVersion(version) : version);
    return 0;
  }
  console.error(
    "usage: node scripts/version.mjs check [--tag vX.Y.Z] [--root dir] | set <version> [--root dir] | print [python]",
  );
  return 2;
}

if (
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  process.exitCode = main(process.argv.slice(2));
}
