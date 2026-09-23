import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { TextEncoder } from "node:util";

import {
  computeSemanticSha256,
  createFunctionId,
  normalizeProjectRelativeSourcePath,
  semanticJsonSha256,
  serializeIrBundle,
  sha256Hex,
  stringifyExactJson,
} from "../dist/identity.js";

const repositoryRoot = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

test("matches the committed refund semantic hash and function id", async () => {
  const record = JSON.parse(
    await readFile(join(repositoryRoot, "examples", "ir", "refund-decision.v1.json"), "utf8"),
  );

  assert.equal(
    computeSemanticSha256(record),
    "5e4d51edec329295a35965f8d6759f5638ae0570f328f9e15a04904a7191ec92",
  );
  assert.equal(
    createFunctionId(record.source.path, 0, record.semanticSha256),
    "nf_5abdc7d0cbfc5e129dd6b94add4e12a9cc3047e7fb7f6bebaf62c33108ea9222",
  );
});

test("hashes exact bytes and semantic values", () => {
  assert.equal(
    sha256Hex(new TextEncoder().encode("abc")),
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  );
  assert.notEqual(semanticJsonSha256(-0), semanticJsonSha256(0));
});

test("normalizes contained POSIX and Windows source paths", () => {
  assert.equal(
    normalizeProjectRelativeSourcePath("/workspace/project", "/workspace/project/src/main.sem.ts"),
    "src/main.sem.ts",
  );
  assert.equal(
    normalizeProjectRelativeSourcePath("C:\\Work\\Project", "C:\\Work\\Project\\src\\Main.sem.ts"),
    "src/Main.sem.ts",
  );
  assert.throws(
    () => normalizeProjectRelativeSourcePath("/workspace/project", "/workspace/other/main.sem.ts"),
    /inside the compiler project root/,
  );
  assert.throws(
    () => normalizeProjectRelativeSourcePath("C:\\Work\\Project", "D:\\Elsewhere\\main.sem.ts"),
    /inside the compiler project root/,
  );
});

test("writes deterministic exact JSON and preserves negative zero", () => {
  const left = { b: [1, true], a: -0 };
  const right = { a: -0, b: [1, true] };
  const expected = `{
  "a": -0,
  "b": [
    1,
    true
  ]
}
`;

  assert.equal(stringifyExactJson(left), expected);
  assert.equal(stringifyExactJson(right), expected);
  assert.equal(Object.is(JSON.parse(expected).a, -0), true);
  const sparse = new Array(2);
  sparse[1] = 1;
  assert.throws(() => stringifyExactJson(sparse), /dense/);
});

test("writes a fixed two-space LF-terminated IR bundle", async () => {
  const record = JSON.parse(
    await readFile(join(repositoryRoot, "examples", "ir", "refund-decision.v1.json"), "utf8"),
  );
  const value = {
    kind: "semantscript.ir-bundle",
    bundleVersion: 1,
    functions: [record],
    executionPlan: {
      stages: [{ index: 0, functionIds: [record.id] }],
      dependencies: [],
    },
  };
  const bundle = serializeIrBundle(value);

  assert.equal(bundle.endsWith("\n"), true);
  assert.equal(bundle.includes("\r"), false);
  assert.match(bundle, /^\{\n {2}"bundleVersion": 1,/);
  assert.deepEqual(JSON.parse(bundle), value);
});
