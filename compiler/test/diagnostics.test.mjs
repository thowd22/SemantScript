// Every diagnostic family the compiler can emit, rendered once and compared
// with a committed snapshot. Regenerate with UPDATE_SNAPSHOTS=1 after an
// intended wording change and review the diff.
import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import process from "node:process";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import {
  findMalformedSemaSites,
  findSemaSites,
  planSemaCompilation,
} from "../dist/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const snapshotPath = join(here, "fixtures", "diagnostics.snapshot.txt");

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

// One site per diagnostic family. Malformed sites never reach analysis; the
// others fail analysis or definition resolution with a located message.
const malformedSource = `import { sema } from "@semantscript/core";
declare const message: string;
export const untyped = sema\`no type argument \${message}\`;
export const twoTypes = sema<"a" | "b", string>\`two type arguments \${message}\`;
export const noOptions = sema<"a" | "b">()\`configured without options \${message}\`;
export const twoOptions = sema<"a" | "b">({}, {})\`configured with two options \${message}\`;
export const untypedDiagnostic = sema.withConfidence\`no type argument \${message}\`;
export const chained = sema?.withConfidence<"a" | "b">\`optional chain \${message}\`;
export const fine = sema<"a" | "b">\`a canonical site \${message}\`;
`;

const invalidSource = `import { sema, always, never, type BoundedInt } from "@semantscript/core";
interface Nested { child: { flag: boolean } }
enum Mixed { A = "a", B = 1 }
declare const message: string;
declare const repeated: string;
declare const item: { value: string };
declare const score: number;
export const freeText = sema<string>\`free text \${message}\`;
export const nested = sema<Nested>\`nested \${message}\`;
export const mixed = sema<Mixed>\`mixed enum \${message}\`;
export const badBounds = sema<BoundedInt<5, 1>>\`bounds \${message}\`;
export const expression = sema<"yes" | "no">\`member \${item.value}\`;
export const twice = sema<"yes" | "no">\`first \${repeated} second \${repeated}\`;
export const badExample = sema<"yes" | "no">({
  examples: [{ inputs: { message: "x" }, output: "maybe" }],
})\`example \${message}\`;
export const contradictory = sema<"yes" | "no">({
  constraints: [always(() => score > 1, "yes"), never(() => score > 1, "yes")],
})\`constraints \${score}\`;
export const empty = sema<"yes" | "no">\`\${message}\`;
`;

async function createFixtureProgram(files) {
  const root = await mkdtemp(join(tmpdir(), "semantscript-diagnostics-"));
  const coreRoot = join(root, "node_modules", "@semantscript", "core");
  await mkdir(coreRoot, { recursive: true });
  await writeFile(
    join(coreRoot, "package.json"),
    JSON.stringify({
      name: "@semantscript/core",
      type: "module",
      types: "index.d.ts",
    }),
  );
  await writeFile(join(coreRoot, "index.d.ts"), coreDeclarations);
  await writeFile(
    join(root, "package.json"),
    JSON.stringify({ type: "module" }),
  );
  for (const [fileName, contents] of Object.entries(files)) {
    await writeFile(join(root, fileName), contents);
  }
  const program = ts.createProgram({
    rootNames: Object.keys(files).map((fileName) => join(root, fileName)),
    options: {
      module: ts.ModuleKind.ESNext,
      moduleResolution: ts.ModuleResolutionKind.Bundler,
      skipLibCheck: true,
      strict: true,
      target: ts.ScriptTarget.ES2023,
    },
  });
  return {
    root,
    program,
    async dispose() {
      await rm(root, { force: true, recursive: true });
    },
  };
}

function render(diagnostics, root) {
  const host = {
    getCanonicalFileName: (fileName) => fileName,
    getCurrentDirectory: () => root,
    getNewLine: () => "\n",
  };
  return ts
    .formatDiagnostics(diagnostics, host)
    .split(root)
    .join("<root>")
    .split(join(root, "/"))
    .join("<root>/");
}

test("every compiler diagnostic names file, line, column and the sema site, and matches the snapshot", async (t) => {
  const fixture = await createFixtureProgram({
    "malformed.sem.ts": malformedSource,
    "invalid.sem.ts": invalidSource,
  });
  t.after(fixture.dispose);

  const malformed = findMalformedSemaSites(fixture.program);
  assert.equal(malformed.length, 6, "six malformed canonical uses");
  assert.equal(
    findSemaSites(fixture.program).length,
    1 + 9,
    "only canonical sites are found",
  );

  const result = await planSemaCompilation(fixture.program, {
    projectRoot: fixture.root,
  });
  assert.equal(result.ok, false);
  for (const diagnostic of result.diagnostics) {
    assert.equal(diagnostic.category, ts.DiagnosticCategory.Error);
    assert.ok(diagnostic.file, "every diagnostic is located in a file");
    assert.equal(typeof diagnostic.start, "number");
    assert.match(
      String(diagnostic.messageText),
      /^<?[^:]*\.sem\.ts:\d+:\d+: .+ \(site: sema/u,
    );
  }
  const codes = [...new Set(result.diagnostics.map(({ code }) => code))].sort();
  assert.deepEqual(
    codes,
    [9100, 9101, 9102, 9104, 9105, 9110, 9111, 9121, 9124, 9125],
  );

  const rendered = render(result.diagnostics, fixture.root);
  if (process.env.UPDATE_SNAPSHOTS) {
    await writeFile(snapshotPath, rendered);
  }
  const expected = await readFile(snapshotPath, "utf8");
  assert.equal(
    rendered,
    expected,
    "diagnostics differ from the committed snapshot",
  );
});
