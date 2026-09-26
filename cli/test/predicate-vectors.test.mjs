import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { ConstraintEvaluationError, evaluatePredicate } from "../dist/index.js";

const vectorsPath = join(
  dirname(fileURLToPath(import.meta.url)),
  "../../examples/constraints/predicate-vectors.v1.json",
);

test("explain evaluates constraint predicates exactly as the trainer's vectors expect", async () => {
  const document = JSON.parse(await readFile(vectorsPath, "utf8"));
  assert.equal(document.kind, "semantscript.predicate-vectors");
  assert.equal(document.vectorsVersion, 1);
  assert.ok(document.vectors.length > 40);
  for (const vector of document.vectors) {
    if (vector.expect === "error") {
      assert.throws(
        () => evaluatePredicate(vector.predicate, vector.inputs),
        ConstraintEvaluationError,
        vector.name,
      );
    } else {
      assert.equal(
        evaluatePredicate(vector.predicate, vector.inputs),
        vector.expect,
        vector.name,
      );
    }
  }
});
