import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { TextDecoder } from "node:util";

import {
  SemaInputError,
  serializeCanonicalInputs,
  serializeCanonicalInputsString,
} from "../dist/canonical-input.js";

const number = { kind: "number" };
const string = { kind: "string" };

function schema(type, name = "value") {
  return [{ name, index: 0, tsType: "unknown", type }];
}

function reason(expectedReason) {
  return (error) => {
    assert.ok(error instanceof SemaInputError);
    assert.equal(error.code, "SEMA_INPUT_INVALID");
    assert.equal(error.reason, expectedReason);
    return true;
  };
}

test("matches the canonical-input v1 golden vector byte for byte", () => {
  const example = {
    kind: "object",
    name: "Example",
    fields: [
      { name: "b", optional: false, type: number },
      { name: "zero", optional: false, type: number },
      { name: "text", optional: false, type: string },
      { name: "a", optional: false, type: number },
    ],
  };
  const expected =
    '["semantscript-input",1,[["obj",["object",[["a",["number","3ff0000000000000"]],["b",["number","4000000000000000"]],["text",["string","a\\n\\"é"]],["zero",["number","8000000000000000"]]]]]]]';
  const left = { b: 2, a: 1, text: 'a\n"é', zero: -0 };
  const right = { zero: -0, text: 'a\n"é', a: 1, b: 2 };

  for (const value of [left, right]) {
    const bytes = serializeCanonicalInputs(schema(example, "obj"), { obj: value });
    assert.equal(new TextDecoder().decode(bytes), expected);
    assert.equal(
      createHash("sha256").update(bytes).digest("hex"),
      "3273c3e7b704fce97a131fbdf1c3cd2bef0cd3896c1f4ebbc6857921b534a162",
    );
  }
});

test("encodes every input node and orders top-level inputs by index", () => {
  const entries = [
    {
      name: "second",
      index: 1,
      type: {
        kind: "object",
        name: "Record",
        fields: [
          { name: "optional", optional: true, type: string },
          { name: "tuple", optional: false, type: { kind: "tuple", items: [string, number] } },
        ],
      },
    },
    {
      name: "first",
      index: 0,
      type: {
        kind: "array",
        items: {
          kind: "union",
          variants: [
            { kind: "literal", value: "x" },
            { kind: "enum", name: "Letter", base: "string", values: ["x", "y"] },
          ],
        },
      },
    },
  ];

  assert.equal(
    serializeCanonicalInputsString(entries, {
      second: { tuple: ["ok", 1] },
      first: ["x", "y"],
    }),
    '["semantscript-input",1,[["first",["array",[["union",0,["literal",["string","x"]]],["union",1,["enum","Letter",1,["string","y"]]]]]],["second",["object",[["tuple",["tuple",[["string","ok"],["number","3ff0000000000000"]]]]]]]]]',
  );
});

test("uses bit-exact numbers for primitives, literals, and numeric enums", () => {
  assert.match(
    serializeCanonicalInputsString(schema({ kind: "null" }), { value: null }),
    /\["null"\]/u,
  );
  assert.match(
    serializeCanonicalInputsString(schema({ kind: "boolean" }), { value: false }),
    /\["boolean",false\]/u,
  );
  assert.match(
    serializeCanonicalInputsString(schema(number), { value: -0 }),
    /\["number","8000000000000000"\]/u,
  );
  assert.throws(
    () => serializeCanonicalInputs(schema({ kind: "literal", value: 0 }), { value: -0 }),
    reason("value"),
  );
  assert.equal(
    serializeCanonicalInputsString(
      schema({ kind: "enum", name: "Zero", base: "number", values: [-0, 0] }),
      { value: 0 },
    ),
    '["semantscript-input",1,[["value",["enum","Zero",1,["number","0000000000000000"]]]]]',
  );
});

test("sorts object fields by UTF-8 bytes and emits exact JSON escapes", () => {
  const astral = "\u{10000}";
  const bmp = "\ue000";
  const record = {
    kind: "object",
    name: "Unicode",
    fields: [
      { name: bmp, optional: false, type: string },
      { name: astral, optional: false, type: string },
      { name: "controls", optional: false, type: string },
    ],
  };
  const encoded = serializeCanonicalInputsString(schema(record), {
    value: {
      [bmp]: "bmp",
      controls: '\b\t\n\f\r\u0000\u001f/"\\',
      [astral]: "astral",
    },
  });

  assert.ok(encoded.indexOf(`"${bmp}"`) < encoded.indexOf(`"${astral}"`));
  assert.match(encoded, /\["string","\\b\\t\\n\\f\\r\\u0000\\u001f\/\\"\\\\"\]/u);
  assert.ok(!encoded.endsWith("\n"));
});

test("enforces a UTF-8 byte limit while preserving exact boundary bytes", () => {
  const value = `${"x".repeat(4095)}🙂\n"\\é`;
  const inputs = { value };
  const expected = serializeCanonicalInputs(schema(string), inputs);
  const atBoundary = serializeCanonicalInputs(schema(string), inputs, {
    maximumBytes: expected.byteLength,
  });

  assert.deepEqual(atBoundary, expected);
  assert.throws(
    () =>
      serializeCanonicalInputs(schema(string), inputs, {
        maximumBytes: expected.byteLength - 1,
      }),
    reason("limit"),
  );

  const oversized = "é".repeat(4 * 1024 * 1024);
  assert.throws(
    () =>
      serializeCanonicalInputs(schema(string), { value: oversized }, { maximumBytes: 1024 }),
    reason("limit"),
  );
});

test("rejects unsupported primitives, non-finite numbers, and malformed strings", () => {
  for (const value of [undefined, 1n, Symbol("x"), () => undefined]) {
    assert.throws(() => serializeCanonicalInputs(schema(string), { value }), reason("type"));
  }

  for (const value of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    assert.throws(() => serializeCanonicalInputs(schema(number), { value }), reason("non-finite"));
  }

  assert.throws(
    () => serializeCanonicalInputs(schema(string), { value: "\ud800" }),
    reason("invalid-unicode"),
  );
});

test("rejects proxies, accessors, symbol keys, classes, promises, and cycles", () => {
  const objectType = {
    kind: "object",
    name: "Box",
    fields: [{ name: "item", optional: false, type: string }],
  };
  const symbolObject = { item: "ok", [Symbol("hidden")]: true };
  const cycle = {};
  cycle.self = cycle;
  const cycleInnerType = {
    kind: "object",
    name: "CycleInner",
    fields: [{ name: "self", optional: true, type: string }],
  };
  const cyclicType = {
    kind: "object",
    name: "CyclicShape",
    fields: [{ name: "self", optional: false, type: cycleInnerType }],
  };
  let getterCalls = 0;
  const accessorObject = {};
  Object.defineProperty(accessorObject, "item", {
    enumerable: true,
    get() {
      getterCalls += 1;
      return "bad";
    },
  });
  class Box {
    item = "ok";
  }
  class StringList extends Array {}

  assert.throws(
    () => serializeCanonicalInputs(schema(objectType), { value: new Proxy({ item: "ok" }, {}) }),
    reason("proxy"),
  );
  assert.throws(
    () => serializeCanonicalInputs(schema(objectType), { value: accessorObject }),
    reason("accessor"),
  );
  assert.equal(getterCalls, 0);
  assert.throws(
    () => serializeCanonicalInputs(schema(objectType), { value: symbolObject }),
    reason("symbol-key"),
  );
  assert.throws(
    () => serializeCanonicalInputs(schema(objectType), { value: new Box() }),
    reason("class-instance"),
  );
  assert.throws(
    () => serializeCanonicalInputs(schema({ kind: "array", items: string }), { value: new StringList("ok") }),
    reason("class-instance"),
  );
  assert.throws(
    () => serializeCanonicalInputs(schema(objectType), { value: Promise.resolve("ok") }),
    reason("promise"),
  );
  assert.throws(
    () => serializeCanonicalInputs(schema(cyclicType), { value: cycle }),
    reason("cycle"),
  );
});

test("rejects malformed arrays and object shape mismatches", () => {
  const arrayType = { kind: "array", items: string };
  const sparse = Array(2);
  sparse[1] = "x";
  const decorated = ["x"];
  decorated.note = true;

  assert.throws(() => serializeCanonicalInputs(schema(arrayType), { value: sparse }), reason("sparse-array"));
  assert.throws(
    () => serializeCanonicalInputs(schema(arrayType), { value: decorated }),
    reason("array-property"),
  );
  assert.throws(
    () => serializeCanonicalInputs(schema({ kind: "tuple", items: [string] }), { value: [] }),
    reason("tuple-length"),
  );

  const record = {
    kind: "object",
    name: "Required",
    fields: [{ name: "required", optional: false, type: string }],
  };
  assert.throws(() => serializeCanonicalInputs(schema(record), { value: {} }), reason("missing"));
  assert.throws(
    () => serializeCanonicalInputs(schema(record), { value: { required: "x", extra: true } }),
    reason("extra"),
  );
});

test("allows repeated acyclic references but enforces depth and schema integrity", () => {
  const leaf = {
    kind: "object",
    name: "Leaf",
    fields: [{ name: "text", optional: false, type: string }],
  };
  const pair = { kind: "tuple", items: [leaf, leaf] };
  const shared = { text: "ok" };
  assert.doesNotThrow(() => serializeCanonicalInputs(schema(pair), { value: [shared, shared] }));

  let nestedType = string;
  let nestedValue = "bottom";
  for (let index = 0; index < 101; index += 1) {
    nestedType = { kind: "array", items: nestedType };
    nestedValue = [nestedValue];
  }
  assert.throws(() => serializeCanonicalInputs(schema(nestedType), { value: nestedValue }), reason("limit"));

  assert.throws(
    () => serializeCanonicalInputs([{ name: "a", index: 1, type: string }], { a: "x" }),
    reason("schema"),
  );
  assert.throws(
    () =>
      serializeCanonicalInputs(
        schema({ kind: "union", variants: [{ kind: "string" }, { kind: "string" }] }),
        { value: "x" },
      ),
    reason("schema"),
  );
});
