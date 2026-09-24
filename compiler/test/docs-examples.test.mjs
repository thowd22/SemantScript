// The language reference's examples must stay compilable.
import assert from "node:assert/strict";
import {
  mkdir,
  mkdtemp,
  readdir,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import { planSemaCompilation } from "../dist/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const examplesDirectory = join(here, "..", "..", "docs", "examples");
const coreDeclarations = await readFile(
  join(here, "..", "..", "runtime", "dist", "index.d.ts"),
  "utf8",
).catch(() => undefined);

test("every docs/examples program compiles to source IR", async (t) => {
  const names = (await readdir(examplesDirectory)).filter((name) =>
    name.endsWith(".sem.ts"),
  );
  assert.ok(
    names.length >= 5,
    "the language reference has at least five examples",
  );
  const root = await mkdtemp(join(tmpdir(), "semantscript-docs-examples-"));
  t.after(() => rm(root, { recursive: true, force: true }));
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
  assert.ok(
    coreDeclarations,
    "runtime/dist/index.d.ts is produced by the build gate",
  );
  await writeFile(join(coreRoot, "index.d.ts"), coreDeclarations);
  await writeFile(
    join(root, "package.json"),
    JSON.stringify({ type: "module" }),
  );
  for (const name of names) {
    await writeFile(
      join(root, name),
      await readFile(join(examplesDirectory, name), "utf8"),
    );
  }
  const program = ts.createProgram({
    rootNames: names.map((name) => join(root, name)),
    options: {
      module: ts.ModuleKind.NodeNext,
      moduleResolution: ts.ModuleResolutionKind.NodeNext,
      skipLibCheck: true,
      strict: true,
      target: ts.ScriptTarget.ES2023,
      lib: ["lib.es2023.d.ts"],
    },
  });
  const typeErrors = ts.getPreEmitDiagnostics(program);
  assert.deepEqual(
    typeErrors.map((d) => ts.flattenDiagnosticMessageText(d.messageText, "\n")),
    [],
  );

  const result = await planSemaCompilation(program, { projectRoot: root });
  assert.equal(
    result.ok,
    true,
    result.ok
      ? ""
      : result.diagnostics.map((d) => String(d.messageText)).join("\n"),
  );
  const byFile = new Map();
  for (const record of result.value.bundle.functions) {
    byFile.set(record.source.path, (byFile.get(record.source.path) ?? 0) + 1);
  }
  assert.deepEqual(Object.fromEntries([...byFile.entries()].sort()), {
    "confidence.sem.ts": 2,
    "examples-and-constraints.sem.ts": 1,
    "inputs.sem.ts": 1,
    "output-types.sem.ts": 8,
    "refund-decision.sem.ts": 1,
  });
  const outputKinds = result.value.bundle.functions
    .filter((record) => record.source.path === "output-types.sem.ts")
    .map((record) =>
      record.output.kind === "scalar"
        ? record.output.head.sourceKind
        : `object:${record.output.fields.length}`,
    )
    .sort();
  assert.deepEqual(outputKinds, [
    "boolean",
    "bounded-int",
    "bounded-number",
    "number-enum",
    "object:4",
    "ordinal-string",
    "string-enum",
    "string-union",
  ]);
});
