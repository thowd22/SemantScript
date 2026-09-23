import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import ts from "typescript";

import { findSemaSites } from "../dist/index.js";

const coreDeclarations = `
export interface ConfiguredSemaTag<T> {
  (strings: TemplateStringsArray, ...inputs: unknown[]): T;
}
export interface DiagnosticSemaTag {
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): { value: T };
  <T>(options: unknown): ConfiguredSemaTag<{ value: T }>;
}
export interface SemaTag {
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): T;
  <T>(options: unknown): ConfiguredSemaTag<T>;
  readonly withConfidence: DiagnosticSemaTag;
}
export declare const sema: SemaTag;
`;

test("finds all four sema forms in nested functions, methods, and arrows", async (t) => {
  const fixture = await createFixtureProgram({
    "shapes.sem.ts": `import { sema } from "@semantscript/core";

function outer() {
  function nested() {
    return sema<"allow" | "deny">\`nested\`;
  }
  return nested();
}

class Policy {
  decide() {
    return sema.withConfidence<boolean>\`method\`;
  }
}

const arrow = () => sema<boolean>({ examples: [] })\`arrow\`;
const configuredDiagnostic = sema.withConfidence<"yes" | "no">({})\`diagnostic\`;
`,
  });
  t.after(fixture.dispose);

  const sites = findSemaSites(fixture.program, fixture.sourceFile("shapes.sem.ts"));

  assert.equal(sites.length, 4);
  assert.deepEqual(
    sites.map(({ configured, resultMode }) => ({ configured, resultMode })),
    [
      { configured: false, resultMode: "value" },
      { configured: false, resultMode: "diagnostic" },
      { configured: true, resultMode: "value" },
      { configured: true, resultMode: "diagnostic" },
    ],
  );
  assert.deepEqual(
    sites.map((site) => site.location.line),
    [5, 12, 16, 17],
  );
  assert.ok(sites.every((site) => site.location.column > 0));
  assert.ok(
    sites.every(
      (site) =>
        site.sourceFile.text.slice(site.location.start, site.location.end) ===
        site.node.getText(site.sourceFile),
    ),
  );
});

test("follows import aliases, namespace access, and re-exports", async (t) => {
  const fixture = await createFixtureProgram({
    "bridge.ts": `export {
  sema as semantic,
  sema as withConfidence,
} from "@semantscript/core";
`,
    "aliases.sem.ts": `import { sema as learned } from "@semantscript/core";
import * as core from "@semantscript/core";
import { semantic } from "./bridge.js";
import * as bridge from "./bridge.js";

const directAlias = learned<boolean>\`alias\`;
const namespaceAlias = core.sema<boolean>\`namespace\`;
const reexportAlias = semantic.withConfidence<boolean>({})\`re-export\`;
const renamedValue = bridge.withConfidence<boolean>\`renamed value\`;
`,
  });
  t.after(fixture.dispose);

  const sites = findSemaSites(fixture.program);

  assert.equal(sites.length, 4);
  assert.deepEqual(
    sites.map((site) => site.node.tag.getText(site.sourceFile)),
    [
      "learned",
      "core.sema",
      "semantic.withConfidence<boolean>({})",
      "bridge.withConfidence",
    ],
  );
  assert.deepEqual(
    sites.map((site) => site.resultMode),
    ["value", "value", "diagnostic", "value"],
  );
});

test("rejects unrelated, shadowed, malformed, and non-sem.ts tags", async (t) => {
  const fixture = await createFixtureProgram({
    "negative.sem.ts": `import { sema as primitive } from "@semantscript/core";

declare const otherTag: typeof primitive;
declare const sema: typeof primitive;
declare const fakeCore: typeof import("@semantscript/core");
const copied = { sema: primitive, withConfidence: primitive.withConfidence };

const unrelated = otherTag<boolean>\`other\`;
const sameSpelling = sema<boolean>\`local\`;
const moduleShapedLocal = fakeCore.sema<boolean>\`module-shaped local\`;
const copiedMember = copied.sema<boolean>\`copied\`;
const copiedDiagnostic = copied.withConfidence<boolean>\`copied diagnostic\`;
const missingType = primitive\`missing type\`;
const malformedOptions = primitive<boolean>()\`missing options\`;
const optionalChain = primitive?.withConfidence<boolean>\`invalid optional chain\`;

function shadowed(primitive: typeof import("@semantscript/core").sema) {
  return primitive<boolean>\`shadowed import alias\`;
}
`,
    "ignored.ts": `import { sema } from "@semantscript/core";
const ignored = sema<boolean>\`not a sem file\`;
`,
  });
  t.after(fixture.dispose);

  assert.deepEqual(findSemaSites(fixture.program), []);
});

async function createFixtureProgram(files) {
  const root = await mkdtemp(join(tmpdir(), "semantscript-discovery-"));
  const coreRoot = join(root, "node_modules", "@semantscript", "core");
  await mkdir(coreRoot, { recursive: true });
  await writeFile(
    join(coreRoot, "package.json"),
    JSON.stringify({ name: "@semantscript/core", type: "module", types: "index.d.ts" }),
  );
  await writeFile(join(coreRoot, "index.d.ts"), coreDeclarations);
  await writeFile(join(root, "package.json"), JSON.stringify({ type: "module" }));

  for (const [fileName, contents] of Object.entries(files)) {
    await writeFile(join(root, fileName), contents);
  }

  const rootNames = Object.keys(files).map((fileName) => join(root, fileName));
  const program = ts.createProgram({
    rootNames,
    options: {
      module: ts.ModuleKind.ESNext,
      moduleResolution: ts.ModuleResolutionKind.Bundler,
      skipLibCheck: true,
      strict: true,
      target: ts.ScriptTarget.ES2023,
    },
  });

  return {
    program,
    sourceFile(fileName) {
      const sourceFile = program.getSourceFile(join(root, fileName));
      assert.ok(sourceFile, `missing fixture source ${fileName}`);
      return sourceFile;
    },
    async dispose() {
      await rm(root, { force: true, recursive: true });
    },
  };
}
