import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
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
  assert.equal(
    defaultSemaArtifactPath({}, { entry: undefined }),
    resolve(".semantscript/artifact"),
  );
  assert.equal(
    defaultSemaArtifactPath(
      { SEMANTSCRIPT_ARTIFACT: "" },
      { entry: undefined },
    ),
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

test("the default path is searched upward from the compiled entry script, then from the working directory", async () => {
  const root = await mkdtemp(join(tmpdir(), "semantscript-artifact-search-"));
  try {
    const deployed = join(root, "deploy");
    await mkdir(join(deployed, "dist", "nested"), { recursive: true });
    await mkdir(join(deployed, ".semantscript", "artifact"), {
      recursive: true,
    });
    const elsewhere = join(root, "elsewhere");
    await mkdir(join(elsewhere, ".semantscript", "artifact"), {
      recursive: true,
    });
    const unrelated = join(root, "unrelated");
    await mkdir(unrelated, { recursive: true });

    // The artifact beside dist/ wins over the working directory's.
    assert.equal(
      defaultSemaArtifactPath(
        {},
        {
          entry: join(deployed, "dist", "nested", "server.js"),
          cwd: elsewhere,
        },
      ),
      join(deployed, ".semantscript", "artifact"),
    );
    // Without one beside the entry, the working directory's ancestors are searched.
    assert.equal(
      defaultSemaArtifactPath(
        {},
        {
          entry: join(unrelated, "server.js"),
          cwd: join(elsewhere, "sub", "dir"),
        },
      ),
      join(elsewhere, ".semantscript", "artifact"),
    );
    // Nothing found: the conventional location under the working directory, for the error message.
    assert.equal(
      defaultSemaArtifactPath(
        {},
        { entry: join(unrelated, "server.js"), cwd: unrelated },
      ),
      join(unrelated, ".semantscript", "artifact"),
    );
    // The environment variable always wins.
    assert.equal(
      defaultSemaArtifactPath(
        { SEMANTSCRIPT_ARTIFACT: "/opt/artifact" },
        { entry: join(deployed, "dist", "server.js") },
      ),
      resolve("/opt/artifact"),
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
