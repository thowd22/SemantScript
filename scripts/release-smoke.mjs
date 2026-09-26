// Installs the release packages into a fresh directory with no repository
// checkout and drives the documented flow there: init, build, train, test and
// run. The release workflow (.github/workflows/release.yml) runs it on the
// packed tarballs before anything is published; docs/releasing.md runs it
// against a local registry. It never reads the repository's sources: every
// package comes from the tarballs or the registry, and the trainer from the
// interpreter given with --python.
//
//   node scripts/release-smoke.mjs --tarballs <dir> [options]
//   node scripts/release-smoke.mjs --registry <url> [--version <v>] [options]
//
// Options:
//   --python <exe>        interpreter with semantscript-trainer[training]
//                         installed (default: SEMANTSCRIPT_PYTHON or python3)
//   --device <name>       training device (default cpu)
//   --cases <n>           training cases (default 96)
//   --epochs <n>          training epochs (default 3)
//   --train-arg <arg>     one more argument for semantscript train (repeatable)
//   --no-train            stop after init, build and the package imports
//   --workdir <dir>       use this empty directory instead of a new temp one
//   --keep                keep the directory afterwards (always kept on failure)
//   --help, -h            print this usage
//
// Prints a JSON summary on the last line, with the versions the artifact
// manifest records; "directoryKept" says whether "directory" still exists.
import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
  mkdirSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import process from "node:process";
import { parseArgs } from "node:util";

import { PUBLISHED_WORKSPACES } from "./version.mjs";

function usage() {
  // The usage is the header comment of this file, minus the comment markers.
  const source = readFileSync(new URL(import.meta.url), "utf8").split("\n");
  const header = [];
  for (const line of source) {
    if (!line.startsWith("//")) break;
    header.push(line.replace(/^\/\/ ?/u, ""));
  }
  return header.join("\n");
}

let values;
try {
  ({ values } = parseArgs({
    options: {
      help: { type: "boolean", short: "h", default: false },
      tarballs: { type: "string" },
      registry: { type: "string" },
      version: { type: "string" },
      python: { type: "string" },
      device: { type: "string", default: "cpu" },
      cases: { type: "string", default: "96" },
      epochs: { type: "string", default: "3" },
      "train-arg": { type: "string", multiple: true, default: [] },
      "no-train": { type: "boolean", default: false },
      workdir: { type: "string" },
      keep: { type: "boolean", default: false },
    },
    allowPositionals: false,
  }));
} catch (error) {
  console.error(`release-smoke: ${error.message}\n\n${usage()}`);
  process.exit(2);
}

if (values.help) {
  console.log(usage());
  process.exit(0);
}

if ((values.tarballs === undefined) === (values.registry === undefined)) {
  console.error(
    "release-smoke: pass exactly one of --tarballs <dir> or --registry <url>",
  );
  process.exit(2);
}

const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const npx = process.platform === "win32" ? "npx.cmd" : "npx";
const python =
  values.python ??
  process.env.SEMANTSCRIPT_PYTHON ??
  (process.platform === "win32" ? "python" : "python3");

const directory =
  values.workdir === undefined
    ? mkdtempSync(join(tmpdir(), "semantscript-smoke-"))
    : resolve(values.workdir);
mkdirSync(directory, { recursive: true });
if (readdirSync(directory).length > 0) {
  console.error(`release-smoke: ${directory} is not empty`);
  process.exit(2);
}
// The point of the smoke run is that no checkout is involved.
const insideGit = spawnSync("git", ["rev-parse", "--show-toplevel"], {
  cwd: directory,
  encoding: "utf8",
});
if (insideGit.status === 0) {
  console.error(
    `release-smoke: ${directory} is inside the git checkout ${insideGit.stdout.trim()}; use a directory outside any repository`,
  );
  process.exit(2);
}

const env = {
  ...process.env,
  SEMANTSCRIPT_PYTHON: python,
  npm_config_fund: "false",
  npm_config_audit: "false",
  npm_config_update_notifier: "false",
};

function step(title, command, args, options = {}) {
  console.log(`\n$ ${[command, ...args].join(" ")}`);
  const result = spawnSync(command, args, {
    cwd: directory,
    env,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
    shell: process.platform === "win32" && command.endsWith(".cmd"),
    ...options,
  });
  if (result.stdout) process.stdout.write(result.stdout);
  if (result.error !== undefined || result.status !== 0) {
    console.error(
      `release-smoke: ${title} failed (${result.error?.message ?? `exit ${result.status}`}); the directory is kept at ${directory}`,
    );
    process.exit(1);
  }
  return result.stdout ?? "";
}

function expect(condition, message) {
  if (!condition) {
    console.error(
      `release-smoke: ${message}; the directory is kept at ${directory}`,
    );
    process.exit(1);
  }
}

console.log(`release-smoke: working in ${directory} with ${python}`);
// The directory starts empty: `npm install` writes package.json and `init`
// starts the TypeScript project (tsconfig.json, type module, build script).
// 1. The four packages, the way the getting-started page installs them.
let specs;
if (values.tarballs !== undefined) {
  const tarballs = readdirSync(resolve(values.tarballs)).filter((name) =>
    name.endsWith(".tgz"),
  );
  specs = PUBLISHED_WORKSPACES.map(({ name }) => {
    const prefix = `${name.replace(/^@/u, "").replace("/", "-")}-`;
    const found = tarballs.filter(
      (file) =>
        file.startsWith(prefix) && /^\d/u.test(file.slice(prefix.length)),
    );
    expect(
      found.length === 1,
      `expected one ${prefix}<version>.tgz in ${values.tarballs}, found ${found.length}`,
    );
    return join(resolve(values.tarballs), found[0]);
  });
} else {
  writeFileSync(join(directory, ".npmrc"), `registry=${values.registry}\n`);
  specs = PUBLISHED_WORKSPACES.map(({ name }) =>
    values.version === undefined ? name : `${name}@${values.version}`,
  );
}
step("npm install", npm, ["install", ...specs]);
const installed = Object.fromEntries(
  PUBLISHED_WORKSPACES.map(({ name }) => [
    name,
    JSON.parse(
      readFileSync(
        join(directory, "node_modules", ...name.split("/"), "package.json"),
        "utf8",
      ),
    ).version,
  ]),
);
const version = installed["semantscript"];
for (const [name, found] of Object.entries(installed)) {
  expect(
    found === version,
    `${name} installed at ${found}, semantscript at ${version}`,
  );
}
expect(
  values.version === undefined || version === values.version,
  `installed ${version}, asked for ${values.version}`,
);

// 2. init wires the compiler; --no-example because the starter needs a
// language-model teacher and this run labels its cases from constraints.
step("semantscript --help", npx, ["--no-install", "semantscript", "--help"]);
step("semantscript init", npx, [
  "--no-install",
  "semantscript",
  "init",
  "--no-example",
  "--teacher",
  "constraints",
  "--python",
  python,
]);
const tsconfig = JSON.parse(
  readFileSync(join(directory, "tsconfig.json"), "utf8"),
);
expect(
  JSON.stringify(tsconfig.compilerOptions.plugins ?? []).includes(
    "@semantscript/compiler/transformer",
  ),
  "init did not add the transformer to tsconfig.json",
);
const scaffolded = JSON.parse(
  readFileSync(join(directory, "package.json"), "utf8"),
);
expect(
  scaffolded.type === "module" &&
    scaffolded.scripts?.build === "tspc -p tsconfig.json",
  "init did not set type module and the tspc build script in package.json",
);
step("npm install (after init)", npm, ["install"]);

mkdirSync(join(directory, "src"), { recursive: true });
writeFileSync(
  join(directory, "src", "hold.sem.ts"),
  `import { always, sema } from "@semantscript/core";

export interface Order {
  total: number;
}

export type Flag = "flag" | "watch" | "clear";

/** Whether to hold a shipment; the flag alone decides, through the constraints. */
export function hold(order: Order, flag: Flag): boolean {
  return sema<boolean>({
    examples: [
      { inputs: { order: { total: 6200 }, flag: "flag" }, output: true },
      { inputs: { order: { total: 2600 }, flag: "watch" }, output: true },
      { inputs: { order: { total: 88.5 }, flag: "clear" }, output: false },
    ],
    constraints: [
      always(() => flag === "flag", true),
      always(() => flag === "watch", true),
      always(() => flag === "clear", false),
    ],
  })\`
    Whether to hold the shipment: for a flagged or watched order, never for a
    cleared order, whatever its total.
    Order: \${order}
    Flag: \${flag}
  \`;
}
`,
);
writeFileSync(
  join(directory, "src", "imports.ts"),
  `import { loadSemaArtifact } from "@semantscript/core";
import { createSemaStubArtifact } from "@semantscript/core/testing";
import { Controller } from "@semantscript/framework";

export const loaded = [typeof loadSemaArtifact, typeof createSemaStubArtifact, typeof Controller];
`,
);

// 3. build: the compiler rewrites the site and writes the IR bundle.
step("semantscript build", npx, ["--no-install", "semantscript", "build"]);
expect(
  existsSync(join(directory, "dist", "semantscript.ir.v1.json")),
  "build wrote no dist/semantscript.ir.v1.json",
);
expect(
  existsSync(join(directory, "dist", "hold.sem.js")),
  "build wrote no dist/hold.sem.js",
);
step("package imports", process.execPath, [
  "--input-type=module",
  "-e",
  'const m = await import("./dist/imports.js"); if (m.loaded.join() !== "function,function,function") throw new Error(m.loaded.join());',
]);

const summary = { directory, version, installed, trained: false };
if (!values["no-train"]) {
  // 4. train with the constraints teacher: no key, no network teacher.
  step("semantscript train", npx, [
    "--no-install",
    "semantscript",
    "train",
    "--teacher",
    "constraints",
    "--cases",
    values.cases,
    "--epochs",
    values.epochs,
    "--device",
    values.device,
    ...values["train-arg"],
  ]);
  // 5. test replays the IR examples through the runtime; run calls the function.
  step("semantscript test", npx, [
    "--no-install",
    "semantscript",
    "test",
    "--bundle",
    "dist/semantscript.ir.v1.json",
  ]);
  const answers = {};
  for (const [input, expected] of [
    ['[{"total":6200},"flag"]', "true"],
    ['[{"total":88.5},"clear"]', "false"],
  ]) {
    const printed = step("semantscript run", npx, [
      "--no-install",
      "semantscript",
      "run",
      "dist/hold.sem.js",
      "--call",
      "hold",
      "--input",
      input,
    ]).trim();
    answers[input] = printed;
    expect(
      printed === expected,
      `run ${input} printed ${printed}, expected ${expected}`,
    );
  }
  const root = join(directory, ".semantscript", "artifact");
  const pointer = JSON.parse(readFileSync(join(root, "current.json"), "utf8"));
  const manifest = JSON.parse(
    readFileSync(join(root, pointer.release, "manifest.json"), "utf8"),
  );
  summary.trained = true;
  summary.answers = answers;
  summary.manifestBuild = manifest.build;
  expect(
    manifest.build.compilerVersion === version,
    `manifest build.compilerVersion is ${manifest.build.compilerVersion}, expected ${version}`,
  );
  expect(
    manifest.build.trainerVersion === version,
    `manifest build.trainerVersion is ${manifest.build.trainerVersion}, expected ${version} (install the matching semantscript-trainer)`,
  );
}

const removeDirectory = !values.keep && values.workdir === undefined;
summary.directoryKept = !removeDirectory;
console.log(`\nrelease-smoke: passed\n${JSON.stringify(summary)}`);
if (removeDirectory) rmSync(directory, { recursive: true, force: true });
