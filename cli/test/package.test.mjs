import assert from "node:assert/strict";
import test from "node:test";

test("CLI package loads", async () => {
  const cli = await import("../dist/index.js");

  assert.deepEqual(Object.keys(cli), []);
});

