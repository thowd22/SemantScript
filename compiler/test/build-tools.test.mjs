import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import {
  mkdtemp,
  mkdir,
  readFile,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import test from "node:test";

import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import { build as esbuildBuild } from "esbuild";
import ts from "typescript";
import { build as viteBuild } from "vite";

import { planSemaCompilation, planSemaCompilationSync } from "../dist/index.js";
import esbuildPlugin from "../dist/esbuild.js";
import semantscriptLoader from "../dist/loader.js";
import semantscriptTransformer from "../dist/transformer.js";
import vitePlugin from "../dist/vite.js";

const run = promisify(execFile);
const compilerRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = join(compilerRoot, "..");
const tspc = join(repositoryRoot, "node_modules", "ts-patch", "bin", "tspc.js");
const ajv = new Ajv2020({
  allErrors: true,
  allowUnionTypes: true,
  strict: true,
});
addFormats(ajv);
ajv.addSchema(
  JSON.parse(
    await readFile(
      join(repositoryRoot, "schemas", "neural-function.v1.schema.json"),
      "utf8",
    ),
  ),
);
const validateBundle = ajv.compile(
  JSON.parse(
    await readFile(
      join(repositoryRoot, "schemas", "ir-bundle.v1.schema.json"),
      "utf8",
    ),
  ),
);

const coreDeclarations = `
interface ConfiguredSemaTag<T> { (strings: TemplateStringsArray, ...inputs: unknown[]): T }
interface SemaTag {
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): T;
  <T>(options: unknown): ConfiguredSemaTag<T>;
}
export declare const sema: SemaTag;
export declare const __sema: { call<T>(id: string, inputs: Readonly<Record<string, unknown>>): T };
`;
const coreImplementation = `
export const sema = () => { throw new Error("sema must be compiled"); };
export const __sema = { call() { throw new Error("no SemantScript artifact is loaded"); } };
`;
// The sema expression sits on line 4 of this file; every stack-trace check
// expects the original .sem.ts line, not the emitted one.
const mainSource = `import { sema } from "@semantscript/core";

export function classify(message: string): "urgent" | "routine" {
  return sema<"urgent" | "routine">\`PROMPT_CLASSIFY \${message}\`;
}

console.log(classify("server is down"));
`;
const malformedSource = `import { sema } from "@semantscript/core";

export const broken = sema\`PROMPT_WITHOUT_TYPE\`;
`;

test("ts-patch transformer compiles a plain tsc project from one tsconfig entry", async (t) => {
  const fixture = await createFixture({ "src/main.sem.ts": mainSource });
  t.after(fixture.dispose);

  const { stdout, stderr } = await run(
    process.execPath,
    [tspc, "-p", "tsconfig.json"],
    {
      cwd: fixture.root,
    },
  );
  assert.equal(stderr, "");
  assert.equal(stdout, "");

  const javascript = await readFile(
    join(fixture.root, "dist", "main.sem.js"),
    "utf8",
  );
  assert.equal(javascript.includes("PROMPT_CLASSIFY"), false);
  assert.match(javascript, /__sema\.call\("nf_[a-f0-9]{64}", \{ message \}\)/u);
  const sourceMap = JSON.parse(
    await readFile(join(fixture.root, "dist", "main.sem.js.map"), "utf8"),
  );
  assert.deepEqual(sourceMap.sources, ["../src/main.sem.ts"]);
  assert.equal(Object.hasOwn(sourceMap, "sourcesContent"), false);
  await assertBundle(join(fixture.root, "dist", "semantscript.ir.v1.json"));
  await assertStackTracePointsAtSource(
    join(fixture.root, "dist", "main.sem.js"),
  );
});

test("ts-patch transformer reports sema diagnostics through tsc", async (t) => {
  const fixture = await createFixture({ "src/broken.sem.ts": malformedSource });
  t.after(fixture.dispose);

  const outcome = await run(process.execPath, [tspc, "-p", "tsconfig.json"], {
    cwd: fixture.root,
  }).then(
    () => undefined,
    (error) => error,
  );
  assert.ok(outcome, "tspc must exit non-zero on a malformed site");
  assert.notEqual(outcome.code, 0);
  assert.match(outcome.stdout, /src\/broken\.sem\.ts\(3,23\): error TS9100:/u);
});

test("transformer factory adds diagnostics through ts-patch extras and leaves files untouched", async (t) => {
  const fixture = await createFixture({ "src/broken.sem.ts": malformedSource });
  t.after(fixture.dispose);
  const program = createFixtureProgram(fixture.root);
  const added = [];

  const transformer = semantscriptTransformer(
    program,
    {},
    { addDiagnostic: (d) => added.push(d) },
  );
  assert.equal(added.length, 1);
  assert.equal(added[0].code, 9100);
  const sourceFile = program.getSourceFile(
    join(fixture.root, "src", "broken.sem.ts"),
  );
  const result = ts.transform(sourceFile, [transformer]);
  assert.equal(result.transformed[0], sourceFile);
  result.dispose();
});

test("esbuild plugin compiles .sem.ts modules with one line of build config", async (t) => {
  const fixture = await createFixture({ "src/main.sem.ts": mainSource });
  t.after(fixture.dispose);
  const outdir = join(fixture.root, "out");

  const result = await esbuildBuild({
    absWorkingDir: fixture.root,
    entryPoints: ["src/main.sem.ts"],
    bundle: true,
    platform: "node",
    format: "esm",
    outdir,
    sourcemap: true,
    packages: "external",
    logLevel: "silent",
    plugins: [esbuildPlugin()],
  });
  assert.deepEqual(result.errors, []);
  assert.deepEqual(result.warnings, []);

  const javascript = await readFile(join(outdir, "main.sem.js"), "utf8");
  assert.equal(javascript.includes("PROMPT_CLASSIFY"), false);
  assert.match(javascript, /__sema\.call\("nf_[a-f0-9]{64}", \{ message \}\)/u);
  const sourceMap = JSON.parse(
    await readFile(join(outdir, "main.sem.js.map"), "utf8"),
  );
  assert.deepEqual(sourceMap.sources, ["../src/main.sem.ts"]);
  await assertBundle(join(outdir, "semantscript.ir.v1.json"));
  await assertStackTracePointsAtSource(join(outdir, "main.sem.js"));
});

test("esbuild plugin passes domain depths and routing through to the plan", async (t) => {
  const fixture = await createFixture({ "src/main.sem.ts": mainSource });
  t.after(fixture.dispose);
  const outdir = join(fixture.root, "out");

  const result = await esbuildBuild({
    absWorkingDir: fixture.root,
    entryPoints: ["src/main.sem.ts"],
    bundle: true,
    platform: "node",
    format: "esm",
    outdir,
    packages: "external",
    logLevel: "silent",
    plugins: [
      esbuildPlugin({ application: "routed", domainDepths: { main: 4 } }),
    ],
  });
  assert.deepEqual(result.errors, []);

  const bundle = JSON.parse(
    await readFile(join(outdir, "semantscript.ir.v1.json"), "utf8"),
  );
  assert.deepEqual(
    bundle.executionPlan.domains.map((domain) => ({
      name: domain.name,
      adapterRef: domain.adapterRef,
      encoderRef: domain.encoderRef,
      encoderDepth: domain.encoderDepth,
    })),
    [
      {
        name: "main",
        adapterRef: "adapter.routed.main",
        encoderRef: "encoder.routed.depth-004",
        encoderDepth: 4,
      },
    ],
  );
  assert.equal(bundle.functions[0].model.encoderDepth, 4);
});

test("esbuild plugin fails the build with the site's location on a malformed site", async (t) => {
  const fixture = await createFixture({ "src/broken.sem.ts": malformedSource });
  t.after(fixture.dispose);

  const failure = await esbuildBuild({
    absWorkingDir: fixture.root,
    entryPoints: ["src/broken.sem.ts"],
    bundle: true,
    platform: "node",
    format: "esm",
    outdir: join(fixture.root, "out"),
    packages: "external",
    logLevel: "silent",
    plugins: [esbuildPlugin()],
  }).then(
    () => undefined,
    (error) => error,
  );
  assert.ok(failure, "esbuild must reject the build");
  assert.equal(failure.errors.length, 1);
  assert.equal(failure.errors[0].pluginName, "semantscript");
  assert.equal(failure.errors[0].location.file, "src/broken.sem.ts");
  assert.equal(failure.errors[0].location.line, 3);
  assert.match(failure.errors[0].text, /sema/u);
});

test("Vite plugin compiles .sem.ts modules and keeps the bundle after outDir is emptied", async (t) => {
  const fixture = await createFixture({ "src/main.sem.ts": mainSource });
  t.after(fixture.dispose);
  const outDir = join(fixture.root, "out");

  await viteBuild({
    root: fixture.root,
    configFile: false,
    logLevel: "silent",
    build: {
      outDir,
      ssr: "src/main.sem.ts",
      sourcemap: true,
      rollupOptions: {
        external: ["@semantscript/core"],
        output: { entryFileNames: "main.js" },
      },
    },
    plugins: [vitePlugin()],
  });

  const javascript = await readFile(join(outDir, "main.js"), "utf8");
  assert.equal(javascript.includes("PROMPT_CLASSIFY"), false);
  assert.match(javascript, /__sema\.call\("nf_[a-f0-9]{64}", \{ message \}\)/u);
  const sourceMap = JSON.parse(
    await readFile(join(outDir, "main.js.map"), "utf8"),
  );
  assert.deepEqual(sourceMap.sources, ["../src/main.sem.ts"]);
  await assertBundle(join(outDir, "semantscript.ir.v1.json"));
  await assertStackTracePointsAtSource(join(outDir, "main.js"));
});

test("Vite plugin fails the build on a malformed site", async (t) => {
  const fixture = await createFixture({ "src/broken.sem.ts": malformedSource });
  t.after(fixture.dispose);

  await assert.rejects(
    viteBuild({
      root: fixture.root,
      configFile: false,
      logLevel: "silent",
      build: {
        outDir: join(fixture.root, "out"),
        ssr: "src/broken.sem.ts",
        rollupOptions: { external: ["@semantscript/core"] },
      },
      plugins: [vitePlugin()],
    }),
    /src\/broken\.sem\.ts\(3,23\): error TS9100/u,
  );
});

test("webpack-style loader emits code and a map for a .sem.ts module", async (t) => {
  const fixture = await createFixture({ "src/main.sem.ts": mainSource });
  t.after(fixture.dispose);
  const resourcePath = join(fixture.root, "src", "main.sem.ts");
  const dependencies = [];
  const outcome = await new Promise((resolve) => {
    semantscriptLoader.call(
      {
        resourcePath,
        rootContext: fixture.root,
        getOptions: () => ({ bundlePath: "out/semantscript.ir.v1.json" }),
        addDependency: (fileName) => dependencies.push(fileName),
        callback: (error, code, map) => resolve({ error, code, map }),
      },
      mainSource,
    );
  });

  assert.equal(outcome.error, null);
  assert.equal(outcome.code.includes("PROMPT_CLASSIFY"), false);
  assert.match(
    outcome.code,
    /__sema\.call\("nf_[a-f0-9]{64}", \{ message \}\)/u,
  );
  assert.equal(outcome.code.includes("sourceMappingURL"), false);
  const sourceMap = outcome.map;
  assert.equal(typeof sourceMap, "object");
  assert.deepEqual(sourceMap.sources, ["../src/main.sem.ts"]);
  assert.equal(Object.hasOwn(sourceMap, "sourcesContent"), false);
  assert.deepEqual(dependencies, [join(fixture.root, "tsconfig.json")]);
  await assertBundle(join(fixture.root, "out", "semantscript.ir.v1.json"));

  // The loader plans from the files on disk, as webpack hands it what it read.
  const brokenPath = join(fixture.root, "src", "broken.sem.ts");
  await writeFile(brokenPath, malformedSource);
  const failure = await new Promise((resolve) => {
    semantscriptLoader.call(
      {
        resourcePath: brokenPath,
        rootContext: fixture.root,
        callback: (error) => resolve(error),
      },
      malformedSource,
    );
  });
  assert.match(failure.message, /src\/broken\.sem\.ts\(3,23\): error TS9100/u);
});

test("the package exposes the adapters as subpath exports for import and require", async (t) => {
  const fixture = await createFixture({ "src/main.sem.ts": mainSource });
  t.after(fixture.dispose);
  const require = createRequire(join(fixture.root, "package.json"));

  for (const name of ["transformer", "esbuild", "vite", "loader"]) {
    const specifier = `@semantscript/compiler/${name}`;
    assert.equal(typeof require(specifier).default, "function", specifier);
    assert.equal(
      typeof (await import(require.resolve(specifier))).default,
      "function",
      specifier,
    );
  }
});

test("planSemaCompilationSync produces the same plan as planSemaCompilation", async (t) => {
  const fixture = await createFixture({ "src/main.sem.ts": mainSource });
  t.after(fixture.dispose);
  const program = createFixtureProgram(fixture.root);

  const asynchronous = await planSemaCompilation(program, {
    projectRoot: fixture.root,
  });
  const synchronous = planSemaCompilationSync(program, {
    projectRoot: fixture.root,
  });
  assert.ok(asynchronous.ok && synchronous.ok);
  assert.equal(synchronous.value.bundleText, asynchronous.value.bundleText);
});

async function assertBundle(bundlePath) {
  const bundle = JSON.parse(await readFile(bundlePath, "utf8"));
  assert.equal(
    validateBundle(bundle),
    true,
    ajv.errorsText(validateBundle.errors),
  );
  assert.equal(bundle.functions.length, 1);
  assert.equal(bundle.functions[0].source.path, "src/main.sem.ts");
  assert.equal(bundle.functions[0].source.line, 4);
  assert.equal(bundle.functions[0].model.encoder, "encoder.application");
}

async function assertStackTracePointsAtSource(entryPath) {
  const failure = await run(process.execPath, [
    "--enable-source-maps",
    entryPath,
  ]).then(
    () => undefined,
    (error) => error,
  );
  assert.ok(failure, "the uncompiled runtime stub must throw");
  assert.match(failure.stderr, /no SemantScript artifact is loaded/u);
  assert.match(failure.stderr, /at classify \(.*src\/main\.sem\.ts:4:10\)/u);
}

function createFixtureProgram(root) {
  const configPath = join(root, "tsconfig.json");
  const parsed = ts.getParsedCommandLineOfConfigFile(
    configPath,
    {},
    {
      ...ts.sys,
      onUnRecoverableConfigFileDiagnostic: (diagnostic) => {
        throw new Error(
          ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n"),
        );
      },
    },
  );
  assert.deepEqual(parsed.errors, []);
  return ts.createProgram({
    rootNames: parsed.fileNames,
    options: parsed.options,
  });
}

async function createFixture(files) {
  const root = await mkdtemp(join(tmpdir(), "semantscript-build-tools-"));
  const coreRoot = join(root, "node_modules", "@semantscript", "core");
  await mkdir(coreRoot, { recursive: true });
  await writeFile(
    join(coreRoot, "package.json"),
    JSON.stringify({
      name: "@semantscript/core",
      type: "module",
      main: "index.js",
      types: "index.d.ts",
    }),
  );
  await writeFile(join(coreRoot, "index.d.ts"), coreDeclarations);
  await writeFile(join(coreRoot, "index.js"), coreImplementation);
  await symlink(
    compilerRoot,
    join(root, "node_modules", "@semantscript", "compiler"),
    "dir",
  );
  await writeFile(
    join(root, "package.json"),
    JSON.stringify({ type: "module", private: true }),
  );
  await writeFile(
    join(root, "tsconfig.json"),
    JSON.stringify(
      {
        compilerOptions: {
          module: "NodeNext",
          moduleResolution: "NodeNext",
          target: "ES2022",
          strict: true,
          sourceMap: true,
          outDir: "dist",
          rootDir: "src",
          skipLibCheck: true,
          plugins: [{ transform: "@semantscript/compiler/transformer" }],
        },
        include: ["src"],
      },
      null,
      2,
    ),
  );

  for (const [fileName, contents] of Object.entries(files)) {
    const target = join(root, fileName);
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, contents);
  }

  return {
    root,
    async dispose() {
      await rm(root, { force: true, recursive: true });
    },
  };
}
