import assert from "node:assert/strict";
import test from "node:test";
import { TextEncoder } from "node:util";

import { parseStrictJson, StrictJsonError } from "../dist/strict-json.js";

const encoder = new TextEncoder();

test("strict JSON rejects duplicate keys and terminal escaped high surrogates", () => {
  assert.throws(
    () => parseStrictJson(encoder.encode('{"value":1,"value":2}')),
    StrictJsonError,
  );
  assert.throws(() => parseStrictJson(encoder.encode('"\\ud800"')), StrictJsonError);
});

test("strict JSON scans large numeric arrays without copying the remaining payload per number", () => {
  const count = 50_000;
  const source = `[${Array.from({ length: count }, (_, index) => String(index % 10)).join(",")}]`;
  const parsed = parseStrictJson(encoder.encode(source));

  assert.equal(Array.isArray(parsed), true);
  assert.equal(parsed.length, count);
  assert.equal(parsed[0], 0);
  assert.equal(parsed.at(-1), 9);
});
