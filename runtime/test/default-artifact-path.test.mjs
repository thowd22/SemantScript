import assert from "node:assert/strict";
import { resolve } from "node:path";
import test from "node:test";

import {
  DEFAULT_SEMA_ARTIFACT_PATH,
  defaultSemaArtifactPath,
  loadSemaArtifact,
  SEMA_ARTIFACT_ENVIRONMENT_VARIABLE,
} from "../dist/index.js";

test("the default artifact path is .semantscript/artifact under cwd unless the environment overrides it", async () => {
  assert.equal(DEFAULT_SEMA_ARTIFACT_PATH, ".semantscript/artifact");
  assert.equal(SEMA_ARTIFACT_ENVIRONMENT_VARIABLE, "SEMANTSCRIPT_ARTIFACT");
  assert.equal(defaultSemaArtifactPath({}), resolve(".semantscript/artifact"));
  assert.equal(
    defaultSemaArtifactPath({ SEMANTSCRIPT_ARTIFACT: "" }),
    resolve(".semantscript/artifact"),
  );
  assert.equal(
    defaultSemaArtifactPath({ SEMANTSCRIPT_ARTIFACT: "deploy/artifact" }),
    resolve("deploy/artifact"),
  );
  // Loading with no argument goes to the default location, which does not exist here.
  await assert.rejects(loadSemaArtifact(), (error) => {
    assert.match(
      String(error.message),
      /\.semantscript\/artifact|current\.json|artifact/u,
    );
    return true;
  });
});
