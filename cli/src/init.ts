import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, relative } from "node:path";
import { createRequire } from "node:module";
import { parseArgs } from "node:util";

import ts from "typescript";

import {
  DEFAULT_ARTIFACT_PATH,
  ARTIFACT_ENVIRONMENT_VARIABLE,
  TEACHER_CHOICES,
  TEACHER_CONFIG_CANDIDATES,
  isTeacherChoice,
  teacherToml,
  type TeacherChoice,
} from "./defaults.js";
import { collectChecks, renderChecks } from "./doctor.js";
import { CliUsageError, type CliIo } from "./io.js";

export type BuildTool = "next" | "vite" | "esbuild" | "tsc";

const BUILD_TOOLS: readonly BuildTool[] = ["next", "vite", "esbuild", "tsc"];
const CORE_PACKAGE = "@semantscript/core";
const COMPILER_PACKAGE = "@semantscript/compiler";
const TS_PATCH_PACKAGE = "ts-patch";
const TRANSFORMER_MODULE = `${COMPILER_PACKAGE}/transformer`;
const VITE_MODULE = `${COMPILER_PACKAGE}/vite`;
const ESBUILD_MODULE = `${COMPILER_PACKAGE}/esbuild`;
const LOADER_MODULE = `${COMPILER_PACKAGE}/loader`;
const EDITOR_PLUGIN = `${COMPILER_PACKAGE}/ts-plugin`;
const STARTER_FILE = "hello.sem.ts";
const TEACHER_FILE = ".semantscript/teacher.toml";
const STARTER_SOURCE = `import { sema } from "${CORE_PACKAGE}";

/**
 * A first decision. Replace the text with what your application needs to know.
 * Verification needs at least one gold example per expression.
 */
export function needsAttentionToday(subject: string): boolean {
  return sema<boolean>({
    examples: [
      { inputs: { subject: "Checkout is down for every customer" }, output: true },
      { inputs: { subject: "Idea: a dark mode for the dashboard" }, output: false },
    ],
  })\`
    Whether a support ticket with this subject needs attention today rather
    than in the normal queue.
    Subject: \${subject}
  \`;
}
`;
const VITE_CONFIG_FILES = [
  "vite.config.ts",
  "vite.config.mts",
  "vite.config.js",
  "vite.config.mjs",
];
const NEXT_CONFIG_FILES = [
  "next.config.ts",
  "next.config.mts",
  "next.config.js",
  "next.config.mjs",
];
const ESBUILD_SCRIPT_FILES = [
  "esbuild.config.mjs",
  "esbuild.config.js",
  "esbuild.config.ts",
  "esbuild.mjs",
  "esbuild.js",
  "build.mjs",
  "build.js",
  "build.ts",
  "scripts/build.mjs",
  "scripts/build.js",
  "scripts/build.ts",
];
const NEXT_CONFIG_SNIPPET = `  // SemantScript: compile .sem.ts modules and keep the runtime's native bindings external.
  turbopack: {
    rules: { "*.sem.ts": { loaders: ["${LOADER_MODULE}"] } },
  },
  serverExternalPackages: ["${CORE_PACKAGE}"],
  outputFileTracingIncludes: { "/**": ["./${DEFAULT_ARTIFACT_PATH}/**"] },
`;

type Outcome =
  | { readonly kind: "changed"; readonly file: string; readonly what: string }
  | { readonly kind: "unchanged"; readonly file: string; readonly what: string }
  | {
      readonly kind: "manual";
      readonly what: string;
      readonly snippet: string;
    };

interface TeacherOutcome {
  readonly kind: "changed" | "unchanged";
  readonly file: string;
  readonly what: string;
  /** The teacher file the doctor checks. */
  readonly path: string;
}

interface PackageJson {
  version?: unknown;
  scripts?: Record<string, unknown>;
  dependencies?: Record<string, unknown>;
  devDependencies?: Record<string, unknown>;
  [key: string]: unknown;
}

/**
 * `semantscript init`: detect the project's build tool, wire the matching
 * compiler adapter, add the packages, reserve `.semantscript/` and write one
 * starter expression, so the next build compiles a sema site. It ends with the
 * doctor's checks (no billed teacher request) so a missing piece shows now,
 * not minutes into the first `semantscript train`; they never change its exit
 * status. `--teacher anthropic|openrouter|ollama|constraints` (asked for on an
 * interactive terminal when no teacher file exists) writes
 * `.semantscript/teacher.toml` without any key; an existing teacher file is
 * never replaced.
 */
export async function initCommand(
  args: readonly string[],
  io: CliIo,
): Promise<number> {
  const { values } = parseArgs({
    args: [...args],
    options: {
      tool: { type: "string" },
      "no-example": { type: "boolean" },
      "no-doctor": { type: "boolean" },
      python: { type: "string" },
      "trainer-module": { type: "string" },
      teacher: { type: "string" },
      "teacher-model": { type: "string" },
    },
    allowPositionals: false,
  });
  const requestedTeacher =
    values.teacher === undefined ? undefined : parseTeacher(values.teacher);
  const root = io.cwd;
  const packagePath = join(root, "package.json");
  if (!existsSync(packagePath)) {
    throw new Error(
      `no package.json in ${root}. Run init at the project root; in a new directory, create one first with \`npm init -y\` (npm install run here, with no package.json, installs into the nearest parent project instead).`,
    );
  }
  const pkg = readPackageJson(packagePath);
  const outcomes: Outcome[] = [];
  let tool: BuildTool;
  let scaffolded = false;
  if (values.tool === undefined) {
    const detected = detectBuildTool(root, pkg, { allowNone: true });
    if (detected === undefined) {
      // A new project (for example the package.json `npm install` just wrote):
      // start a plain TypeScript project compiled by tspc.
      tool = "tsc";
      scaffolded = true;
    } else {
      tool = detected;
    }
  } else {
    tool = parseTool(values.tool);
    scaffolded = tool === "tsc" && !existsSync(join(root, "tsconfig.json"));
  }
  if (scaffolded) {
    outcomes.push(scaffoldTsconfig(root));
    outcomes.push(scaffoldPackageJson(root, pkg));
  }

  outcomes.push(wireBuildTool(tool, root));
  outcomes.push(wireEditorPlugin(root));
  outcomes.push(addPackages(packagePath, pkg, tool, scaffolded));
  outcomes.push(reserveArtifactDirectory(root));
  if (values["no-example"] !== true) {
    outcomes.push(writeStarter(root, tool));
  }
  const teacher = await chooseTeacher(root, requestedTeacher, io);
  const teacherOutcome =
    teacher === undefined
      ? undefined
      : writeTeacher(root, teacher, values["teacher-model"]);
  if (teacherOutcome !== undefined) outcomes.push(teacherOutcome);

  io.stdout(render(tool, root, outcomes, scaffolded));
  if (values["no-doctor"] !== true) {
    let checks: Awaited<ReturnType<typeof collectChecks>>;
    // init's --teacher is a choice, not a path: the doctor checks the file it wrote.
    const doctorValues = {
      ...values,
      teacher: teacherOutcome === undefined ? undefined : teacherOutcome.path,
    };
    try {
      checks = await collectChecks(doctorValues, io, {
        probe: "free",
        quick: true,
      });
    } catch (error: unknown) {
      // The wiring succeeded; a doctor that breaks its contract is reported, not fatal.
      io.stdout(
        `environment (semantscript doctor): the checks did not run: ${error instanceof Error ? error.message : String(error)}\nrun semantscript doctor to see them\n`,
      );
      return 0;
    }
    const failed = checks.filter((check) => check.status === "fail").length;
    io.stdout(
      `environment (semantscript doctor):\n${renderChecks(checks)}${
        failed === 0
          ? "the environment is ready for semantscript train\n"
          : `fix the ${String(failed)} failed check${failed === 1 ? "" : "s"} before semantscript train; semantscript doctor re-runs them (and sends one teacher request)\n`
      }`,
    );
  }
  return 0;
}

/**
 * Detect the project's build tool from its config files and dependencies:
 * Next.js, Vite, esbuild, else tsc when a tsconfig.json exists. With
 * `allowNone`, a project with none of them returns undefined instead of
 * throwing, so init can start a new TypeScript project there.
 */
export function detectBuildTool(root: string, pkg: PackageJson): BuildTool;
export function detectBuildTool(
  root: string,
  pkg: PackageJson,
  options: { readonly allowNone: true },
): BuildTool | undefined;
export function detectBuildTool(
  root: string,
  pkg: PackageJson,
  options?: { readonly allowNone?: boolean },
): BuildTool | undefined {
  const dependencies = { ...pkg.dependencies, ...pkg.devDependencies };
  if (
    NEXT_CONFIG_FILES.some((file) => existsSync(join(root, file))) ||
    "next" in dependencies
  ) {
    return "next";
  }
  if (
    VITE_CONFIG_FILES.some((file) => existsSync(join(root, file))) ||
    "vite" in dependencies
  ) {
    return "vite";
  }
  const scripts = Object.values(pkg.scripts ?? {});
  if (
    "esbuild" in dependencies ||
    scripts.some(
      (script) => typeof script === "string" && /\besbuild\b/u.test(script),
    )
  ) {
    return "esbuild";
  }
  if (existsSync(join(root, "tsconfig.json"))) {
    return "tsc";
  }
  if (options?.allowNone === true) return undefined;
  throw new CliUsageError(
    "no build tool detected (next.config, vite.config, esbuild in package.json or a tsconfig.json); pass --tool next|vite|esbuild|tsc",
  );
}

function parseTeacher(value: string): TeacherChoice {
  if (isTeacherChoice(value)) return value;
  throw new CliUsageError(
    `--teacher must be one of ${TEACHER_CHOICES.join(", ")}`,
  );
}

/**
 * The teacher to write: `--teacher`, else the answer to one question when stdin is
 * an interactive terminal and no teacher file exists yet, else none.
 */
async function chooseTeacher(
  root: string,
  requested: TeacherChoice | undefined,
  io: CliIo,
): Promise<TeacherChoice | undefined> {
  if (requested !== undefined) return requested;
  if (io.ask === undefined || existingTeacher(root) !== undefined) {
    return undefined;
  }
  for (;;) {
    const answer = (
      await io.ask(
        `teacher for semantscript train (${TEACHER_CHOICES.join(", ")}; empty to skip): `,
      )
    )
      .trim()
      .toLowerCase();
    if (answer === "") return undefined;
    if (isTeacherChoice(answer)) return answer;
    io.stdout(`  ${answer} is not one of ${TEACHER_CHOICES.join(", ")}\n`);
  }
}

function existingTeacher(root: string): string | undefined {
  return TEACHER_CONFIG_CANDIDATES.find((candidate) =>
    existsSync(join(root, candidate)),
  );
}

function writeTeacher(
  root: string,
  choice: TeacherChoice,
  model: string | undefined,
): TeacherOutcome {
  const existing = existingTeacher(root);
  if (existing !== undefined) {
    return {
      kind: "unchanged",
      file: existing,
      what: `a teacher file already exists; left as is (init --teacher ${choice} writes ${TEACHER_FILE} only when there is none)`,
      path: join(root, existing),
    };
  }
  if (choice === "constraints" && model !== undefined) {
    throw new CliUsageError(
      "--teacher-model does not apply to the constraints teacher",
    );
  }
  const path = join(root, TEACHER_FILE);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, teacherToml(choice, model));
  return {
    kind: "changed",
    file: TEACHER_FILE,
    what: `${choice} teacher${choice === "constraints" ? "" : ` (${model ?? "default model"})`}, no key in the file`,
    path,
  };
}

/** The tsconfig.json init writes for a new project: ES modules from src/ to dist/. */
const SCAFFOLD_TSCONFIG = `{
  "compilerOptions": {
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "target": "ES2022",
    "strict": true,
    "skipLibCheck": true,
    "rootDir": "src",
    "outDir": "dist"
  },
  "include": ["src"]
}
`;

/** A new project gets a tsconfig.json and a src/ directory; the plugin entries are added next. */
function scaffoldTsconfig(root: string): Outcome {
  writeFileSync(join(root, "tsconfig.json"), SCAFFOLD_TSCONFIG);
  mkdirSync(join(root, "src"), { recursive: true });
  return {
    kind: "changed",
    file: "tsconfig.json",
    what: "new TypeScript project: NodeNext modules, src/ compiled to dist/",
  };
}

/** The test script `npm init -y` writes; a package.json with only this has no code yet. */
const NPM_INIT_TEST_SCRIPT = 'echo "Error: no test specified" && exit 1';

/**
 * A new project's package.json gets a `build` script that runs tspc and
 * TypeScript itself, each only when it is not there already, and `type:
 * module` when nothing in it points at existing code (no `type`, no `main`
 * file on disk, no scripts beyond `npm init`'s placeholder) so the change
 * cannot turn existing CommonJS files into ES modules. It is written with the
 * packages by addPackages.
 */
function scaffoldPackageJson(root: string, pkg: PackageJson): Outcome {
  const added: string[] = [];
  const main = pkg["main"];
  // No code relies on the module type yet: no entry point on disk, no
  // scripts beyond npm init's placeholder and no JavaScript files in the root
  // or src/. Only then does init choose ES modules.
  const untouched =
    (main === undefined ||
      (typeof main === "string" && !existsSync(join(root, main)))) &&
    Object.values(pkg.scripts ?? {}).every(
      (script) => script === NPM_INIT_TEST_SCRIPT,
    ) &&
    !hasJavaScriptFiles(root) &&
    !hasJavaScriptFiles(join(root, "src"));
  const type = pkg["type"];
  if (untouched && type === undefined) {
    pkg["type"] = "module";
    added.push("type: module");
  } else if (untouched && type === "commonjs") {
    // npm 11's `npm init -y` writes "type": "commonjs" into its placeholder.
    pkg["type"] = "module";
    added.push("type: module (was npm init's commonjs default)");
  }
  const scripts = pkg.scripts ?? {};
  if (typeof scripts["build"] !== "string" || scripts["build"].length === 0) {
    scripts["build"] = "tspc -p tsconfig.json";
    added.push("scripts.build (tspc -p tsconfig.json)");
  }
  pkg.scripts = scripts;
  const dependencies = { ...pkg.dependencies, ...pkg.devDependencies };
  if (!("typescript" in dependencies)) {
    pkg.devDependencies = {
      ...pkg.devDependencies,
      typescript: typescriptRange(),
    };
    added.push("typescript");
  }
  return added.length === 0
    ? {
        kind: "unchanged",
        file: "package.json",
        what: "type, build script and typescript already set",
      }
    : {
        kind: "changed",
        file: "package.json",
        what: `new project: ${added.join(", ")}`,
      };
}

/** Whether `directory` holds .js or .cjs files, whose meaning depends on the module type. */
function hasJavaScriptFiles(directory: string): boolean {
  if (!existsSync(directory)) return false;
  return readdirSync(directory, { withFileTypes: true }).some(
    (entry) =>
      entry.isFile() &&
      (entry.name.endsWith(".js") || entry.name.endsWith(".cjs")),
  );
}

function parseTool(value: string): BuildTool {
  if ((BUILD_TOOLS as readonly string[]).includes(value))
    return value as BuildTool;
  throw new CliUsageError(`--tool must be one of ${BUILD_TOOLS.join(", ")}`);
}

function wireBuildTool(tool: BuildTool, root: string): Outcome {
  switch (tool) {
    case "tsc":
      return wireTsc(root);
    case "vite":
      return wireVite(root);
    case "next":
      return wireNext(root);
    case "esbuild":
      return wireEsbuild(root);
  }
}

function wireTsc(root: string): Outcome {
  return insertTsconfigPlugin(
    root,
    `{ "transform": "${TRANSFORMER_MODULE}" }`,
    TRANSFORMER_MODULE,
    "plugins entry for the ts-patch transformer",
    "transformer already listed in plugins",
  );
}

/** The editor plugin goes into every project's tsconfig: hover and diagnostics at sema sites. */
function wireEditorPlugin(root: string): Outcome {
  return insertTsconfigPlugin(
    root,
    `{ "name": "${EDITOR_PLUGIN}" }`,
    EDITOR_PLUGIN,
    "plugins entry for the editor plugin (hover and diagnostics at sema sites)",
    "editor plugin already listed in plugins",
  );
}

function insertTsconfigPlugin(
  root: string,
  entry: string,
  marker: string,
  changedWhat: string,
  unchangedWhat: string,
): Outcome {
  const file = "tsconfig.json";
  const path = join(root, file);
  const snippet = `"compilerOptions": { "plugins": [${entry}] }`;
  if (!existsSync(path)) {
    return {
      kind: "manual",
      what: "tsconfig.json is missing; create one with",
      snippet,
    };
  }
  const text = readFileSync(path, "utf8");
  if (text.includes(marker)) {
    return { kind: "unchanged", file, what: unchangedWhat };
  }
  const parsed = ts.parseConfigFileTextToJson(path, text);
  if (parsed.error !== undefined) {
    return { kind: "manual", what: `${file} does not parse; add`, snippet };
  }
  let updated: string | undefined;
  const plugins = /"plugins"\s*:\s*\[/u.exec(text);
  const compilerOptions = /"compilerOptions"\s*:\s*\{/u.exec(text);
  if (plugins !== null) {
    const insertAt = plugins.index + plugins[0].length;
    const rest = text.slice(insertAt);
    const empty = /^\s*\]/u.test(rest);
    updated = `${text.slice(0, insertAt)}${entry}${empty ? "" : ", "}${rest}`;
  } else if (compilerOptions !== null) {
    const insertAt = compilerOptions.index + compilerOptions[0].length;
    const indent = /\n([ \t]+)"/u.exec(text.slice(insertAt))?.[1] ?? "    ";
    updated = `${text.slice(0, insertAt)}\n${indent}"plugins": [${entry}],${text.slice(insertAt)}`;
  } else {
    const brace = text.indexOf("{");
    if (brace >= 0) {
      updated = `${text.slice(0, brace + 1)}\n  "compilerOptions": { "plugins": [${entry}] },${text.slice(brace + 1)}`;
    }
  }
  if (
    updated === undefined ||
    ts.parseConfigFileTextToJson(path, updated).error !== undefined
  ) {
    return {
      kind: "manual",
      what: `could not edit ${file}; add to compilerOptions`,
      snippet,
    };
  }
  writeFileSync(path, updated);
  return { kind: "changed", file, what: changedWhat };
}

function wireVite(root: string): Outcome {
  const file = VITE_CONFIG_FILES.find((candidate) =>
    existsSync(join(root, candidate)),
  );
  const snippet = `import semantscript from "${VITE_MODULE}";\nexport default defineConfig({ plugins: [semantscript()] });`;
  if (file === undefined) {
    return {
      kind: "manual",
      what: "no vite.config found; create one with",
      snippet,
    };
  }
  return wireEsmConfig(
    root,
    file,
    VITE_MODULE,
    [/plugins\s*:\s*\[/u],
    [/defineConfig\(\s*\{/u, /export\s+default\s+\{/u],
    snippet,
  );
}

function wireEsbuild(root: string): Outcome {
  const file = ESBUILD_SCRIPT_FILES.find(
    (candidate) =>
      existsSync(join(root, candidate)) &&
      /\besbuild\b/u.test(readFileSync(join(root, candidate), "utf8")),
  );
  const snippet = `import semantscript from "${ESBUILD_MODULE}";\nawait build({ /* your options */ plugins: [semantscript()] });`;
  if (file === undefined) {
    return {
      kind: "manual",
      what: "no esbuild build script found (esbuild plugins need the JS API, not the CLI); in your build script add",
      snippet,
    };
  }
  return wireEsmConfig(
    root,
    file,
    ESBUILD_MODULE,
    [/plugins\s*:\s*\[/u],
    [/\bbuild\(\s*\{/u, /\bcontext\(\s*\{/u],
    snippet,
  );
}

/** Adds a default import of `module` and `semantscript()` to the first plugins array (or creates one). */
function wireEsmConfig(
  root: string,
  file: string,
  module: string,
  pluginsPatterns: readonly RegExp[],
  objectPatterns: readonly RegExp[],
  snippet: string,
): Outcome {
  const path = join(root, file);
  const text = readFileSync(path, "utf8");
  if (text.includes(module)) {
    return { kind: "unchanged", file, what: "plugin already configured" };
  }
  if (/\brequire\(/u.test(text) && !/^\s*import\b/mu.test(text)) {
    return { kind: "manual", what: `${file} uses require(); add`, snippet };
  }
  let updated: string | undefined;
  const plugins = firstMatch(text, pluginsPatterns);
  const object = firstMatch(text, objectPatterns);
  if (plugins !== undefined) {
    const insertAt = plugins.index + plugins[0].length;
    const empty = /^\s*\]/u.test(text.slice(insertAt));
    updated = `${text.slice(0, insertAt)}semantscript()${empty ? "" : ", "}${text.slice(insertAt)}`;
  } else if (object !== undefined) {
    const insertAt = object.index + object[0].length;
    updated = `${text.slice(0, insertAt)}\n  plugins: [semantscript()],${text.slice(insertAt)}`;
  }
  if (updated === undefined) {
    return {
      kind: "manual",
      what: `could not find a plugins array or config object in ${file}; add`,
      snippet,
    };
  }
  writeFileSync(
    path,
    insertImport(updated, `import semantscript from "${module}";`),
  );
  return {
    kind: "changed",
    file,
    what: `import and plugins entry for ${module}`,
  };
}

function wireNext(root: string): Outcome {
  const file = NEXT_CONFIG_FILES.find((candidate) =>
    existsSync(join(root, candidate)),
  );
  const snippet = NEXT_CONFIG_SNIPPET.trimEnd();
  if (file === undefined) {
    return {
      kind: "manual",
      what: "no next.config found; create one exporting",
      snippet,
    };
  }
  const path = join(root, file);
  const text = readFileSync(path, "utf8");
  if (text.includes(LOADER_MODULE)) {
    return { kind: "unchanged", file, what: "loader rule already configured" };
  }
  if (
    /\b(turbopack|serverExternalPackages|outputFileTracingIncludes)\s*:/u.test(
      text,
    )
  ) {
    return {
      kind: "manual",
      what: `${file} already sets turbopack, serverExternalPackages or outputFileTracingIncludes; merge`,
      snippet,
    };
  }
  const object = firstMatch(text, [
    /const\s+\w+\s*(?::\s*NextConfig)?\s*=\s*\{/u,
    /module\.exports\s*=\s*\{/u,
    /export\s+default\s+\{/u,
  ]);
  if (object === undefined) {
    return {
      kind: "manual",
      what: `could not find the config object in ${file}; add`,
      snippet,
    };
  }
  const insertAt = object.index + object[0].length;
  writeFileSync(
    path,
    `${text.slice(0, insertAt)}\n${NEXT_CONFIG_SNIPPET.trimEnd()}${text.slice(insertAt)}`,
  );
  return {
    kind: "changed",
    file,
    what: "turbopack loader rule, external runtime and artifact tracing",
  };
}

function firstMatch(
  text: string,
  patterns: readonly RegExp[],
): RegExpExecArray | undefined {
  for (const pattern of patterns) {
    const match = pattern.exec(text);
    if (match !== null) return match;
  }
  return undefined;
}

function insertImport(text: string, importLine: string): string {
  const imports = [...text.matchAll(/^import\b[^\n]*\n/gmu)];
  const last = imports.at(-1);
  if (last === undefined) return `${importLine}\n${text}`;
  const insertAt = last.index + last[0].length;
  return `${text.slice(0, insertAt)}${importLine}\n${text.slice(insertAt)}`;
}

function addPackages(
  packagePath: string,
  pkg: PackageJson,
  tool: BuildTool,
  alreadyChanged = false,
): Outcome {
  const version = packageVersion();
  const added: string[] = [];
  const dependencies = pkg.dependencies ?? {};
  const devDependencies = pkg.devDependencies ?? {};
  const has = (name: string): boolean =>
    name in dependencies || name in devDependencies;
  if (!has(CORE_PACKAGE)) {
    dependencies[CORE_PACKAGE] = version;
    added.push(CORE_PACKAGE);
  }
  if (!has(COMPILER_PACKAGE)) {
    devDependencies[COMPILER_PACKAGE] = version;
    added.push(COMPILER_PACKAGE);
  }
  if (tool === "tsc") {
    if (!has(TS_PATCH_PACKAGE)) {
      devDependencies[TS_PATCH_PACKAGE] = "^4.0.1";
      added.push(TS_PATCH_PACKAGE);
    }
    const scripts = pkg.scripts ?? {};
    const prepare = scripts["prepare"];
    if (typeof prepare !== "string" || prepare.length === 0) {
      scripts["prepare"] = "ts-patch install";
      added.push("scripts.prepare");
    } else if (!prepare.includes("ts-patch install")) {
      scripts["prepare"] = `${prepare} && ts-patch install`;
      added.push("scripts.prepare");
    }
    pkg.scripts = scripts;
  }
  pkg.dependencies = dependencies;
  pkg.devDependencies = { ...pkg.devDependencies, ...devDependencies };
  if (added.length === 0 && !alreadyChanged) {
    return {
      kind: "unchanged",
      file: "package.json",
      what: "packages already present",
    };
  }
  const original = readFileSync(packagePath, "utf8");
  const indent = /^([ \t]+)"/mu.exec(original)?.[1] ?? "  ";
  writeFileSync(packagePath, `${JSON.stringify(pkg, null, indent)}\n`);
  return added.length === 0
    ? {
        kind: "unchanged",
        file: "package.json",
        what: "packages already present",
      }
    : {
        kind: "changed",
        file: "package.json",
        what: `added ${added.join(", ")}`,
      };
}

function reserveArtifactDirectory(root: string): Outcome {
  const directory = join(root, ".semantscript");
  const ignorePath = join(directory, ".gitignore");
  const file = relative(root, ignorePath);
  if (existsSync(ignorePath)) {
    return {
      kind: "unchanged",
      file,
      what: "artifact and cache directories already reserved",
    };
  }
  mkdirSync(directory, { recursive: true });
  writeFileSync(
    ignorePath,
    "# SemantScript build outputs: the trained artifact and the build cache\nartifact/\ncache/\n",
  );
  return {
    kind: "changed",
    file,
    what: "reserves .semantscript/artifact and .semantscript/cache",
  };
}

function writeStarter(root: string, tool: BuildTool): Outcome {
  const existing = findSemaSource(root, root, 0);
  if (existing !== undefined) {
    return {
      kind: "unchanged",
      file: relative(root, existing),
      what: "a .sem.ts file already exists",
    };
  }
  const directory =
    tool === "next" && existsSync(join(root, "lib"))
      ? "lib"
      : existsSync(join(root, "src"))
        ? "src"
        : tool === "next"
          ? "lib"
          : ".";
  const path = join(root, directory, STARTER_FILE);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, STARTER_SOURCE);
  return {
    kind: "changed",
    file: relative(root, path),
    what: "one starter sema expression",
  };
}

function findSemaSource(
  root: string,
  directory: string,
  depth: number,
): string | undefined {
  if (depth > 4) return undefined;
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    if (
      entry.name.startsWith(".") ||
      entry.name === "node_modules" ||
      entry.name === "dist"
    )
      continue;
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      const found = findSemaSource(root, path, depth + 1);
      if (found !== undefined) return found;
    } else if (entry.name.endsWith(".sem.ts")) {
      return path;
    }
  }
  return undefined;
}

function readPackageJson(path: string): PackageJson {
  const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new CliUsageError(`${path} must contain a JSON object`);
  }
  return parsed as PackageJson;
}

/** The TypeScript range the CLI accepts (its peer dependency), for a new project's devDependencies. */
function typescriptRange(): string {
  try {
    const own = createRequire(import.meta.url)("../package.json") as {
      peerDependencies?: Record<string, unknown>;
    };
    const range = own.peerDependencies?.["typescript"];
    return typeof range === "string" ? range : "^6.0.0";
  } catch {
    return "^6.0.0";
  }
}

function packageVersion(): string {
  try {
    const own = createRequire(import.meta.url)("../package.json") as {
      version?: unknown;
    };
    return typeof own.version === "string" ? own.version : "0.0.0";
  } catch {
    return "0.0.0";
  }
}

function buildInstructions(tool: BuildTool): string {
  switch (tool) {
    case "tsc":
      return "npm run build (tsc is patched by `ts-patch install` from the prepare script; `npx tspc` also works)";
    case "vite":
      return "npx vite build";
    case "next":
      return "npx next build";
    case "esbuild":
      return "run your esbuild build script";
  }
}

function render(
  tool: BuildTool,
  root: string,
  outcomes: readonly Outcome[],
  scaffolded: boolean,
): string {
  const lines = [
    scaffolded
      ? `semantscript init: no tsconfig.json in ${root}; started a TypeScript project built by tspc`
      : `semantscript init: detected ${tool} in ${root}`,
  ];
  for (const outcome of outcomes) {
    if (outcome.kind === "manual") {
      lines.push(`  manual     ${outcome.what}:`);
      for (const line of outcome.snippet.split("\n"))
        lines.push(`               ${line}`);
    } else {
      lines.push(
        `  ${outcome.kind.padEnd(10)} ${outcome.file}: ${outcome.what}`,
      );
    }
  }
  lines.push(
    "next steps:",
    "  1. npm install",
    `  2. ${buildInstructions(tool)}   (writes the IR bundle next to the build output)`,
    "  3. npx semantscript teacher probe, then npx semantscript train --estimate   (one small request to the teacher; then the cost and time of the run, without calling it)",
    "  4. npx semantscript train   (the teacher file init wrote, ANTHROPIC_API_KEY with the default teacher, --teacher <toml>, or --teacher constraints when the constraints decide every input; --max-cost-usd <x> caps the spend)",
    `  5. call loadSemaArtifact() once at startup; it reads ${DEFAULT_ARTIFACT_PATH} unless ${ARTIFACT_ENVIRONMENT_VARIABLE} is set`,
    "  editor: after npm install, hover a sema expression for its verified accuracy (VS Code loads the plugin from node_modules; no extension needed)",
    "",
  );
  return lines.join("\n");
}
