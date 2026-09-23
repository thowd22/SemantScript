import assert from "node:assert/strict";
import test from "node:test";

import { semanticJsonString } from "../dist/index.js";

test("semantic JSON rejects a terminal unpaired high surrogate", () => {
  assert.throws(() => semanticJsonString("\ud800"), /unpaired UTF-16 surrogate/);
});
