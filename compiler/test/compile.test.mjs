import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import Ajv2020 from "ajv/dist/2020.js";
import addFormats from "ajv-formats";
import ts from "typescript";

import {
  compileSemantScriptProgram,
  createFirstBeforeSemaRewriteTransformer,
  emitSemaCompilation,
  planSemaCompilation,
  serializeIrBundle,
} from "../dist/index.js";

const repositoryRoot = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const schema = JSON.parse(
  await readFile(join(repositoryRoot, "schemas", "neural-function.v1.schema.json"), "utf8"),
);
const bundleSchema = JSON.parse(
  await readFile(join(repositoryRoot, "schemas", "ir-bundle.v1.schema.json"), "utf8"),
);
const ajv = new Ajv2020({ allErrors: true, allowUnionTypes: true, strict: true });
addFormats(ajv);
ajv.addSchema(schema);
const validateIr = ajv.compile(schema);
const validateBundle = ajv.compile(bundleSchema);

const coreDeclarations = `
declare const boundedIntKind: unique symbol;
export type BoundedInt<Min extends number, Max extends number> =
  number & { readonly [boundedIntKind]?: readonly [Min, Max] };
interface SemaExample<T> { readonly inputs: Readonly<Record<string, unknown>>; readonly output: T }
declare const semaConstraintKind: unique symbol;
interface SemaConstraint<T> { readonly [semaConstraintKind]: T }
interface SemaOptions<T> {
  readonly examples?: readonly SemaExample<T>[];
  readonly constraints?: readonly SemaConstraint<unknown>[];
}
interface ConfiguredSemaTag<T> { (strings: TemplateStringsArray, ...inputs: unknown[]): T }
interface ConfiguredDiagnosticSemaTag<T> {
  (strings: TemplateStringsArray, ...inputs: unknown[]): { value: T };
}
interface DiagnosticSemaTag {
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): { value: T };
  <T>(options: SemaOptions<T>): ConfiguredDiagnosticSemaTag<T>;
}
interface SemaTag {
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): T;
  <T>(options: SemaOptions<T>): ConfiguredSemaTag<T>;
  readonly withConfidence: DiagnosticSemaTag;
}
export declare const sema: SemaTag;
export declare function always<T>(predicate: () => boolean, output: T): SemaConstraint<T>;
export declare function never<T>(predicate: () => boolean, output: T): SemaConstraint<T>;
export declare const __sema: { call<T>(id: string, inputs: Readonly<Record<string, unknown>>): T };
`;

test("emits runtime calls and a schema-valid deterministic IR bundle without prompt leakage", async (t) => {
  const fixture = await createProject({
    "src/compile.sem.ts": `"use client";
import { sema, always, type BoundedInt } from "@semantscript/core";

const __sema = "user binding";
declare const message: string;
declare const score: number;

export const direct = sema<"yes" | "no">\`PROMPT_DIRECT \${message}\`;
export const configured = sema.withConfidence<BoundedInt<0, 2>>({
  examples: [{ inputs: { score: 1 }, output: 2 }],
  constraints: [always(() => score >= 0, 2)],
})\`
  @confidence(0.75)
  PROMPT_CONFIGURED \${score}
\`;
export const configuredValue = sema<boolean>({
  examples: [{ inputs: { message: "known" }, output: true }],
})\`PROMPT_CONFIGURED_VALUE \${message}\`;
export const directDiagnostic = sema.withConfidence<"yes" | "no">\`
  @confidence(0.5)
  PROMPT_DIRECT_DIAGNOSTIC \${message}
\`;

void __sema;
`,
  });
  t.after(fixture.dispose);

  const result = await compileSemantScriptProgram(fixture.program, {
    projectRoot: fixture.root,
  });
  assert.equal(result.ok, true, formatDiagnostics(result));
  assert.ok(result.ok);

  const javascriptPath = join(fixture.outDir, "compile.sem.js");
  const mapPath = `${javascriptPath}.map`;
  const javascript = await readFile(javascriptPath, "utf8");
  const sourceMap = JSON.parse(await readFile(mapPath, "utf8"));
  const bundleText = await readFile(result.value.bundlePath, "utf8");
  const bundle = JSON.parse(bundleText);
  const records = bundle.functions;

  assert.equal(javascript.includes("PROMPT_DIRECT"), false);
  assert.equal(javascript.includes("PROMPT_CONFIGURED"), false);
  assert.equal(javascript.includes("PROMPT_CONFIGURED_VALUE"), false);
  assert.equal(javascript.includes("PROMPT_DIRECT_DIAGNOSTIC"), false);
  assert.equal(javascript.includes("@confidence"), false);
  assert.equal(javascript.includes("examples:"), false);
  assert.match(javascript, /import \{ __sema as __sema_\d+ \}/);
  assert.match(javascript, /__sema_\d+\.call\("nf_[a-f0-9]{64}", \{ message \}\)/);
  assert.match(javascript, /__sema_\d+\.call\("nf_[a-f0-9]{64}", \{ score \}\)/);
  assert.equal(Object.hasOwn(sourceMap, "sourcesContent"), false);
  assert.equal(JSON.stringify(sourceMap).includes("PROMPT_"), false);

  assert.equal(validateBundle(bundle), true, ajv.errorsText(validateBundle.errors));
  assert.equal(bundle.kind, "semantscript.ir-bundle");
  assert.equal(bundle.bundleVersion, 1);
  assert.equal(records.length, 4);
  for (const record of records) {
    assert.equal(validateIr(record), true, ajv.errorsText(validateIr.errors));
  }

  assert.equal(records[0].source.path, "src/compile.sem.ts");
  assert.equal(records[0].runtime.confidenceThreshold, null);
  assert.equal(records[1].runtime.resultMode, "diagnostic");
  assert.equal(records[1].runtime.confidenceThreshold, 0.75);
  assert.equal(records[1].definition.examples[0].inputs.score, 1);
  assert.equal(records[1].definition.constraints[0].kind, "always");
  assert.equal(records[2].runtime.resultMode, "value");
  assert.equal(records[2].definition.examples[0].inputs.message, "known");
  assert.equal(records[3].runtime.resultMode, "diagnostic");
  assert.equal(records[3].runtime.confidenceThreshold, 0.5);
  assert.equal(
    records[1].definition.template.some(
      (part) => part.kind === "text" && part.text.includes("@confidence"),
    ),
    false,
  );
  assert.deepEqual(bundle.executionPlan.dependencies, []);
  assert.deepEqual(bundle.executionPlan.stages, [
    { index: 0, functionIds: records.map(({ id }) => id) },
  ]);
});

test("emits labeled dependencies and minimum-depth stages in the bundle", async (t) => {
  const fixture = await createProject({
    "src/graph.sem.ts": `import { sema } from "@semantscript/core";
declare const message: string;
const root = sema<boolean>\`ROOT \${message}\`;
const alias = root;
const child = sema<boolean>\`CHILD \${alias}\`;
const other = sema<boolean>\`OTHER \${message}\`;
const joined = child ? "yes" : "no";
export const final = sema<boolean>\`FINAL \${joined}\`;
void other;
`,
  });
  t.after(fixture.dispose);

  const result = await compileSemantScriptProgram(fixture.program, {
    projectRoot: fixture.root,
  });
  assert.equal(result.ok, true, formatDiagnostics(result));
  assert.ok(result.ok);
  const { executionPlan, functions } = JSON.parse(
    await readFile(result.value.bundlePath, "utf8"),
  );
  const ids = Object.fromEntries(
    functions.map((record) => [templateText(record).trim().split(" ")[0], record.id]),
  );

  assert.deepEqual(executionPlan.dependencies, [
    {
      producerFunctionId: ids.ROOT,
      consumerFunctionId: ids.CHILD,
      consumerInput: "alias",
    },
    {
      producerFunctionId: ids.CHILD,
      consumerFunctionId: ids.FINAL,
      consumerInput: "joined",
    },
  ]);
  assert.deepEqual(executionPlan.stages, [
    { index: 0, functionIds: [ids.ROOT, ids.OTHER] },
    { index: 1, functionIds: [ids.CHILD] },
    { index: 2, functionIds: [ids.FINAL] },
  ]);
});

test("emits a schema-valid empty plan when the program has no sema sites", async (t) => {
  const fixture = await createProject({
    "src/empty.sem.ts": "export const ordinary = 1;\n",
  });
  t.after(fixture.dispose);

  const result = await compileSemantScriptProgram(fixture.program, {
    projectRoot: fixture.root,
  });
  assert.equal(result.ok, true, formatDiagnostics(result));
  assert.ok(result.ok);
  const bundle = JSON.parse(await readFile(result.value.bundlePath, "utf8"));
  assert.equal(validateBundle(bundle), true, ajv.errorsText(validateBundle.errors));
  assert.deepEqual(bundle.functions, []);
  assert.deepEqual(bundle.executionPlan, { dependencies: [], stages: [] });
});

test("removes prompt-bearing sourcesContent from inline source maps", async (t) => {
  const fixture = await createProject(
    {
      "src/inline-map.sem.ts": 'import { sema } from "@semantscript/core";\nexport const value = sema<boolean>`INLINE_SECRET`;\n',
    },
    { inlineSourceMap: true, sourceMap: false },
  );
  t.after(fixture.dispose);

  const result = await compileSemantScriptProgram(fixture.program, {
    projectRoot: fixture.root,
  });
  assert.equal(result.ok, true, formatDiagnostics(result));
  assert.ok(result.ok);
  const javascript = await readFile(join(fixture.outDir, "inline-map.sem.js"), "utf8");
  assert.equal(javascript.includes("INLINE_SECRET"), false);
  const match = /sourceMappingURL=data:application\/json(?:;charset=[^;,]+)?;base64,([A-Za-z0-9+/=]+)/u.exec(
    javascript,
  );
  assert.ok(match?.[1]);
  const sourceMap = JSON.parse(Buffer.from(match[1], "base64").toString("utf8"));
  assert.equal(Object.hasOwn(sourceMap, "sourcesContent"), false);
});

test("function ids are stable for unrelated edits and sensitive to text and type changes", async () => {
  const variants = {
    base: sourceWithTarget("TARGET", '"yes" | "no"'),
    same: sourceWithTarget("TARGET", '"yes" | "no"'),
    unrelated: sourceWithTarget("TARGET", '"yes" | "no"', {
      prefix: 'const unrelated = sema<boolean>`UNRELATED`;',
    }),
    changedText: sourceWithTarget("TARGET CHANGED", '"yes" | "no"'),
    changedType: sourceWithTarget("TARGET", '"yes" | "maybe"'),
    changedExample: configuredTargetSource(
      'examples: [{ inputs: { input: "fixture" }, output: "yes" }]',
    ),
    changedConstraint: configuredTargetSource(
      'constraints: [always(() => input === "fixture", "yes")]',
      ", always",
    ),
    changedConfidence: sourceWithTarget("@confidence(0.9)\nTARGET", '"yes" | "no"'),
    duplicate: sourceWithTarget("TARGET", '"yes" | "no"', {
      prefix: 'const duplicate = sema<"yes" | "no">`TARGET ${input}`;',
    }),
  };
  const plans = {};
  for (const [name, source] of Object.entries(variants)) {
    const fixture = await createProject({ "src/identity.sem.ts": source });

    try {
      const result = await planSemaCompilation(fixture.program, { projectRoot: fixture.root });
      assert.equal(result.ok, true, `${name}: ${formatDiagnostics(result)}`);
      assert.ok(result.ok);
      plans[name] = {
        records: result.value.bundle.functions,
        duplicateOrdinals: result.value.sites.map(({ duplicateOrdinal }) => duplicateOrdinal),
      };
    } finally {
      await fixture.dispose();
    }
  }

  const base = recordForText(plans.base.records, "TARGET");
  const same = recordForText(plans.same.records, "TARGET");
  const unrelated = recordForText(plans.unrelated.records, "TARGET");
  const changedText = recordForText(plans.changedText.records, "TARGET CHANGED");
  const changedType = recordForText(plans.changedType.records, "TARGET");
  const changedExample = recordForText(plans.changedExample.records, "TARGET");
  const changedConstraint = recordForText(plans.changedConstraint.records, "TARGET");
  const changedConfidence = recordForText(plans.changedConfidence.records, "TARGET");
  const duplicateRecords = plans.duplicate.records.filter(
    (record) => templateText(record) === "TARGET ",
  );

  assert.equal(base.id, same.id);
  assert.equal(base.id, unrelated.id);
  assert.equal(base.semanticSha256, unrelated.semanticSha256);
  assert.notEqual(base.id, changedText.id);
  assert.notEqual(base.semanticSha256, changedText.semanticSha256);
  assert.notEqual(base.id, changedType.id);
  assert.notEqual(base.semanticSha256, changedType.semanticSha256);
  assert.notEqual(base.semanticSha256, changedExample.semanticSha256);
  assert.notEqual(base.semanticSha256, changedConstraint.semanticSha256);
  assert.notEqual(base.semanticSha256, changedConfidence.semanticSha256);
  assert.equal(duplicateRecords.length, 2);
  assert.notEqual(duplicateRecords[0].id, duplicateRecords[1].id);
  assert.equal(duplicateRecords[0].semanticSha256, duplicateRecords[1].semanticSha256);
  assert.equal(plans.duplicate.duplicateOrdinals.at(-1), 1);
});

test("orders source records by UTF-8 path bytes", async (t) => {
  const fixture = await createProject({
    "src/\uE000.sem.ts": 'import { sema } from "@semantscript/core";\nexport const bmp = sema<boolean>`BMP`;\n',
    "src/\u{10000}.sem.ts": 'import { sema } from "@semantscript/core";\nexport const astral = sema<boolean>`ASTRAL`;\n',
  });
  t.after(fixture.dispose);

  const result = await planSemaCompilation(fixture.program, { projectRoot: fixture.root });
  assert.equal(result.ok, true, formatDiagnostics(result));
  assert.ok(result.ok);
  assert.deepEqual(
    result.value.bundle.functions.map(({ source }) => source.path),
    ["src/\uE000.sem.ts", "src/\u{10000}.sem.ts"],
  );
});

test("plans the committed refund example with inline nested example values", async (t) => {
  const source = await readFile(join(repositoryRoot, "examples", "refund.sem.ts"), "utf8");
  const fixture = await createProject({ "src/refund.sem.ts": source });
  t.after(fixture.dispose);

  const result = await planSemaCompilation(fixture.program, { projectRoot: fixture.root });
  assert.equal(result.ok, true, formatDiagnostics(result));
  assert.ok(result.ok);
  assert.equal(result.value.bundle.functions.length, 1);
  assert.deepEqual(result.value.bundle.functions[0].definition.examples[0].inputs.customer, {
    priorRefunds: 0,
    tier: "enterprise",
  });
});

test("rejects malformed and stale rewrite plans before prompt-bearing output is written", async (t) => {
  const fixture = await createProject({
    "src/stale.sem.ts": 'import { sema } from "@semantscript/core";\nexport const value = sema<boolean>`SECRET_PROMPT`;\n',
  });
  t.after(fixture.dispose);

  const result = await planSemaCompilation(fixture.program, { projectRoot: fixture.root });
  assert.equal(result.ok, true, formatDiagnostics(result));
  assert.ok(result.ok);
  const [site] = result.value.sites;
  assert.ok(site);

  assert.throws(
    () =>
      createFirstBeforeSemaRewriteTransformer([
        { ...site, functionId: "not-a-function-id" },
      ]),
    /function id must match/,
  );
  assert.throws(
    () =>
      createFirstBeforeSemaRewriteTransformer([
        { ...site, inputNames: ["input", "input"] },
      ]),
    /duplicate planned sema input name/,
  );

  const stalePlan = {
    ...result.value,
    sites: [{ ...site, start: site.start + 1 }],
  };
  await assert.rejects(
    emitSemaCompilation(fixture.program, stalePlan),
    /does not match the Program's discovered sema sites/,
  );
  await assert.rejects(readFile(join(fixture.outDir, "stale.sem.js")), /ENOENT/);
});

test("rejects an incomplete plan before any unplanned prompt can be emitted", async (t) => {
  const fixture = await createProject({
    "src/incomplete.sem.ts": `import { sema } from "@semantscript/core";
export const first = sema<boolean>\`FIRST_SECRET\`;
export const second = sema<boolean>\`SECOND_SECRET\`;
`,
  });
  t.after(fixture.dispose);

  const result = await planSemaCompilation(fixture.program, { projectRoot: fixture.root });
  assert.equal(result.ok, true, formatDiagnostics(result));
  assert.ok(result.ok);

  await assert.rejects(
    emitSemaCompilation(fixture.program, {
      ...result.value,
      sites: result.value.sites.slice(0, 1),
    }),
    /1 rewrites for 2 discovered sema sites/,
  );
  await assert.rejects(readFile(join(fixture.outDir, "incomplete.sem.js")), /ENOENT/);
});

test("rejects swapping internally consistent identities between planned sites", async (t) => {
  const fixture = await createProject({
    "src/swapped.sem.ts": `import { sema } from "@semantscript/core";
export const first = sema<boolean>\`FIRST_SITE\`;
export const second = sema<boolean>\`SECOND_SITE\`;
`,
  });
  t.after(fixture.dispose);

  const result = await planSemaCompilation(fixture.program, { projectRoot: fixture.root });
  assert.equal(result.ok, true, formatDiagnostics(result));
  assert.ok(result.ok);
  const [first, second] = result.value.sites;
  assert.ok(first && second);
  const firstIdentity = {
    functionId: first.functionId,
    semanticSha256: first.semanticSha256,
    record: first.record,
  };
  Object.assign(first, {
    functionId: second.functionId,
    semanticSha256: second.semanticSha256,
    record: second.record,
  });
  Object.assign(second, firstIdentity);
  result.value.bundle.functions.splice(0, 2, first.record, second.record);
  result.value.bundleText = serializeIrBundle(result.value.bundle);

  await assert.rejects(
    emitSemaCompilation(fixture.program, result.value),
    /IR bundle functions must be in canonical source order|compilation plan (?:bundle and bundle text are inconsistent|was modified after planning)/,
  );
  await assert.rejects(readFile(join(fixture.outDir, "swapped.sem.js")), /ENOENT/);
});

test("rejects a structurally valid execution plan modified after planning", async (t) => {
  const fixture = await createProject({
    "src/graph-seal.sem.ts": `import { sema } from "@semantscript/core";
const producer = sema<boolean>\`PRODUCER\`;
export const consumer = sema<boolean>\`CONSUMER \${producer}\`;
`,
  });
  t.after(fixture.dispose);

  const result = await planSemaCompilation(fixture.program, { projectRoot: fixture.root });
  assert.equal(result.ok, true, formatDiagnostics(result));
  assert.ok(result.ok);
  const ids = result.value.bundle.functions.map(({ id }) => id);
  result.value.bundle.executionPlan.dependencies.splice(0);
  result.value.bundle.executionPlan.stages.splice(
    0,
    result.value.bundle.executionPlan.stages.length,
    { index: 0, functionIds: ids },
  );
  result.value.bundleText = serializeIrBundle(result.value.bundle);

  await assert.rejects(
    emitSemaCompilation(fixture.program, result.value),
    /compilation plan was modified after planning/,
  );
  await assert.rejects(readFile(join(fixture.outDir, "graph-seal.sem.js")), /ENOENT/);
});

test("rejects a plan mutated during asynchronous filesystem preflight", async (t) => {
  const fixture = await createProject({
    "src/snapshot.sem.ts": `import { sema } from "@semantscript/core";
export const first = sema<boolean>\`FIRST_SNAPSHOT_SECRET\`;
export const second = sema<boolean>\`SECOND_SNAPSHOT_SECRET\`;
`,
  });
  t.after(fixture.dispose);

  const result = await planSemaCompilation(fixture.program, { projectRoot: fixture.root });
  assert.equal(result.ok, true, formatDiagnostics(result));
  assert.ok(result.ok);
  const emitting = emitSemaCompilation(fixture.program, result.value);
  const mutation = Promise.resolve().then(() => result.value.sites.splice(0));
  await assert.rejects(emitting, /0 rewrites for 2 discovered sema sites/);
  await mutation;
  await assert.rejects(readFile(join(fixture.outDir, "snapshot.sem.js")), /ENOENT/);
});

test("rejects a bundle path that would overwrite a TypeScript source", async (t) => {
  const sourcePath = "src/source-path.sem.ts";
  const source = 'import { sema } from "@semantscript/core";\nexport const value = sema<boolean>`KEEP_SOURCE`;\n';
  const fixture = await createProject({ [sourcePath]: source });
  t.after(fixture.dispose);

  await assert.rejects(
    compileSemantScriptProgram(fixture.program, {
      projectRoot: fixture.root,
      bundlePath: join(fixture.root, sourcePath),
    }),
    /IR bundle path conflicts with a TypeScript source file/,
  );
  assert.equal(await readFile(join(fixture.root, sourcePath), "utf8"), source);
  await assert.rejects(readFile(join(fixture.outDir, "source-path.sem.js")), /ENOENT/);
});

test("rejects a directory bundle target before committing TypeScript outputs", async (t) => {
  const fixture = await createProject({
    "src/directory-target.sem.ts": 'import { sema } from "@semantscript/core";\nexport const value = sema<boolean>`DIRECTORY_TARGET`;\n',
  });
  t.after(fixture.dispose);
  const bundleDirectory = join(fixture.root, "bundle-directory");
  await mkdir(bundleDirectory);

  await assert.rejects(
    compileSemantScriptProgram(fixture.program, {
      projectRoot: fixture.root,
      bundlePath: bundleDirectory,
    }),
    /IR bundle path must not be an existing directory/,
  );
  await assert.rejects(readFile(join(fixture.outDir, "directory-target.sem.js")), /ENOENT/);
});

test("preflights every emitted target before committing any output", async (t) => {
  const fixture = await createProject({
    "src/output-directory.sem.ts": 'import { sema } from "@semantscript/core";\nexport const value = sema<boolean>`OUTPUT_DIRECTORY`;\n',
  });
  t.after(fixture.dispose);
  await mkdir(join(fixture.outDir, "output-directory.sem.js"), { recursive: true });

  await assert.rejects(
    compileSemantScriptProgram(fixture.program, { projectRoot: fixture.root }),
    /TypeScript output path must not be an existing directory/,
  );
  await assert.rejects(readFile(join(fixture.outDir, "output-directory.sem.d.ts")), /ENOENT/);
  await assert.rejects(readFile(join(fixture.outDir, "semantscript.ir.v1.json")), /ENOENT/);
});

test("resolves symlinked parents when checking bundle path collisions", async (t) => {
  if (process.platform === "win32") {
    t.skip("directory symlink creation requires additional Windows privileges");
    return;
  }

  const sourcePath = "src/symlink-path.sem.ts";
  const source = 'import { sema } from "@semantscript/core";\nexport const value = sema<boolean>`KEEP_SYMLINK_SOURCE`;\n';
  const fixture = await createProject({ [sourcePath]: source });
  t.after(fixture.dispose);
  const sourceAlias = join(fixture.root, "source-alias");
  await symlink(join(fixture.root, "src"), sourceAlias, "dir");

  await assert.rejects(
    compileSemantScriptProgram(fixture.program, {
      projectRoot: fixture.root,
      bundlePath: join(sourceAlias, "symlink-path.sem.ts"),
    }),
    /IR bundle path conflicts with a TypeScript source file/,
  );
  assert.equal(await readFile(join(fixture.root, sourcePath), "utf8"), source);

  const danglingOutputAlias = join(fixture.root, "dangling-output-alias");
  await symlink("dist", danglingOutputAlias, "dir");
  await assert.rejects(
    compileSemantScriptProgram(fixture.program, {
      projectRoot: fixture.root,
      bundlePath: join(danglingOutputAlias, "symlink-path.sem.js"),
    }),
    /IR bundle path conflicts with a TypeScript output path/,
  );
  await assert.rejects(readFile(join(fixture.outDir, "symlink-path.sem.js")), /ENOENT/);

  await mkdir(fixture.outDir, { recursive: true });
  const outputAlias = join(fixture.root, "output-alias");
  await symlink(fixture.outDir, outputAlias, "dir");
  await assert.rejects(
    compileSemantScriptProgram(fixture.program, {
      projectRoot: fixture.root,
      bundlePath: join(outputAlias, "symlink-path.sem.js"),
    }),
    /IR bundle path conflicts with a TypeScript output path/,
  );
  await assert.rejects(readFile(join(fixture.outDir, "symlink-path.sem.js")), /ENOENT/);
});

test("diagnostics prevent JavaScript and bundle writes", async (t) => {
  const fixture = await createProject({
    "src/invalid.sem.ts": `import { sema } from "@semantscript/core";
export const invalid = sema<string>\`unsupported free text\`;
`,
  });
  t.after(fixture.dispose);

  const result = await compileSemantScriptProgram(fixture.program, {
    projectRoot: fixture.root,
  });
  assert.equal(result.ok, false);
  assert.match(formatDiagnostics(result), /not a supported finite scalar output/);
  await assert.rejects(readFile(join(fixture.outDir, "invalid.sem.js")), /ENOENT/);
  await assert.rejects(readFile(join(fixture.outDir, "semantscript.ir.v1.json")), /ENOENT/);
});

function sourceWithTarget(text, outputType, options = {}) {
  return `import { sema } from "@semantscript/core";
declare const input: string;
${options.prefix ?? ""}
export const target = sema<${outputType}>\`${text} \${input}\`;
`;
}

function configuredTargetSource(optionText, extraImport = "") {
  return `import { sema${extraImport} } from "@semantscript/core";
declare const input: string;
export const target = sema<"yes" | "no">({ ${optionText} })\`TARGET \${input}\`;
`;
}

function recordForText(records, expected) {
  const record = records.find((candidate) => templateText(candidate).startsWith(expected));
  assert.ok(record, `missing record whose template begins ${expected}`);
  return record;
}

function templateText(record) {
  return record.definition.template
    .filter(({ kind }) => kind === "text")
    .map(({ text }) => text)
    .join("");
}

function formatDiagnostics(result) {
  if (result.ok) {
    return "";
  }

  return result.diagnostics
    .map((diagnostic) => ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n"))
    .join("\n");
}

async function createProject(files, compilerOptions = {}) {
  const root = await mkdtemp(join(tmpdir(), "semantscript-compile-"));
  const sourceRoot = join(root, "src");
  const outDir = join(root, "dist");
  const coreRoot = join(root, "node_modules", "@semantscript", "core");
  await mkdir(coreRoot, { recursive: true });
  await writeFile(
    join(coreRoot, "package.json"),
    JSON.stringify({ name: "@semantscript/core", type: "module", types: "index.d.ts" }),
  );
  await writeFile(join(coreRoot, "index.d.ts"), coreDeclarations);
  await writeFile(join(root, "package.json"), JSON.stringify({ type: "module" }));

  for (const [fileName, contents] of Object.entries(files)) {
    const target = join(root, fileName);
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, contents);
  }

  const program = ts.createProgram({
    rootNames: Object.keys(files).map((fileName) => join(root, fileName)),
    options: {
      declaration: true,
      inlineSources: true,
      module: ts.ModuleKind.ESNext,
      moduleResolution: ts.ModuleResolutionKind.Bundler,
      noEmitOnError: true,
      outDir,
      rootDir: sourceRoot,
      sourceMap: true,
      strict: true,
      target: ts.ScriptTarget.ES2023,
      ...compilerOptions,
    },
  });

  const fixture = {
    root,
    outDir,
    program,
    async dispose() {
      fixture.program = undefined;
      await rm(root, { force: true, recursive: true });
    },
  };
  return fixture;
}
