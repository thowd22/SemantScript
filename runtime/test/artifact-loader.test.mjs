import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import {
  mkdtemp,
  open,
  readFile,
  rename,
  rm,
  symlink,
  unlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { createFixtureArtifact } from "./fixtures/artifact.mjs";

const { ArtifactLoadError, loadArtifact } = await import("../dist/artifact-loader.js");

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function republish(artifact, mutate, transformBytes) {
  const manifestPath = join(artifact.release, "manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  mutate?.(manifest);
  let bytes = Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`);
  if (transformBytes) bytes = Buffer.from(transformBytes(bytes.toString("utf8")));
  await writeFile(manifestPath, bytes);
  const manifestSha256 = sha256(bytes);
  const release = join(artifact.root, "releases", `sha256-${manifestSha256}`);
  await rename(artifact.release, release);
  await writeFile(
    join(artifact.root, "current.json"),
    `${JSON.stringify({
      kind: "semantscript.artifact-pointer",
      pointerVersion: 1,
      release: `releases/sha256-${manifestSha256}`,
      manifestSha256,
    })}\n`,
  );
  return { ...artifact, manifest, manifestSha256, release };
}

async function withArtifact(run) {
  const root = await mkdtemp(join(tmpdir(), "semantscript-loader-"));
  try {
    const artifact = await createFixtureArtifact(root);
    await run(artifact);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

function hasCode(code) {
  return (error) => error instanceof ArtifactLoadError && error.code === code;
}

function policy(resultMode, confidenceThreshold, thresholdPolicy, fallbackRef) {
  return { resultMode, confidenceThreshold, policy: thresholdPolicy, fallbackRef };
}

function makeFixtureOutputFlat(manifest) {
  const head = manifest.functions[0].heads[0];
  manifest.functions[0].heads = [
    { ...head, outputPath: ["decision"] },
    { ...head, outputPath: ["route"] },
  ];
}

test("loads current pointer and direct release with immutable validated metadata", async () => {
  await withArtifact(async (artifact) => {
    const fromPointer = await loadArtifact(artifact.root);
    const direct = await loadArtifact(artifact.release);

    assert.equal(fromPointer.manifestSha256, artifact.manifestSha256);
    assert.equal(direct.manifestSha256, artifact.manifestSha256);
    assert.equal(fromPointer.resources.length, 4);
    assert.ok(fromPointer.resources.every((entry) => entry.prepared instanceof Uint8Array));
    assert.ok(Object.isFrozen(fromPointer));
    assert.ok(Object.isFrozen(fromPointer.resources));
    assert.ok(Object.isFrozen(fromPointer.manifest));
    assert.ok(Object.isFrozen(fromPointer.functions[0].inputs));
  });
});

test("rejects duplicate JSON keys before schema validation", async () => {
  await withArtifact(async (artifact) => {
    await republish(
      artifact,
      undefined,
      (json) => json.replace('  "artifactVersion": 1,', '  "artifactVersion": 1,\n  "artifactVersion": 1,'),
    );
    await assert.rejects(loadArtifact(artifact.root), hasCode("SEMA_ARTIFACT_INVALID_JSON"));
  });
});

test("rejects invalid relational metadata", async (context) => {
  await context.test("input digest", async () => {
    await withArtifact(async (artifact) => {
      await republish(artifact, (manifest) => {
        manifest.functions[0].inputSchemaSha256 = "f".repeat(64);
      });
      await assert.rejects(loadArtifact(artifact.root), hasCode("SEMA_ARTIFACT_INTEGRITY"));
    });
  });

  await context.test("duplicate resource ref", async () => {
    await withArtifact(async (artifact) => {
      await republish(artifact, (manifest) => {
        manifest.resources[1].ref = manifest.resources[0].ref;
      });
      await assert.rejects(loadArtifact(artifact.root), hasCode("SEMA_ARTIFACT_INVALID_MANIFEST"));
    });
  });

  await context.test("tokenizer sequence limit", async () => {
    await withArtifact(async (artifact) => {
      await republish(artifact, (manifest) => {
        manifest.resources[0].maximumSequenceLength = 8193;
      });
      await assert.rejects(loadArtifact(artifact.root), hasCode("SEMA_ARTIFACT_INVALID_MANIFEST"));
    });
  });

  await context.test("runtime older than the artifact requires", async () => {
    const { VERSION } = await import("../dist/version.js");
    const corePackage = JSON.parse(
      await readFile(new URL("../package.json", import.meta.url), "utf8"),
    );
    assert.equal(VERSION, corePackage.version);
    await withArtifact(async (artifact) => {
      await republish(artifact, (manifest) => {
        manifest.compatibility.minimumRuntimeVersion = "999.0.0";
      });
      // Without runtimeVersion the loader compares against the package's own version.
      await assert.rejects(
        loadArtifact(artifact.root),
        (error) =>
          error.code === "SEMA_ARTIFACT_INCOMPATIBLE" &&
          error.detail === `artifact requires runtime 999.0.0; current is ${VERSION}` &&
          error.message.startsWith(`${error.detail}; next: `),
      );
    });
    await withArtifact(async (artifact) => {
      await republish(artifact, (manifest) => {
        manifest.compatibility.minimumRuntimeVersion = VERSION;
      });
      await loadArtifact(artifact.root);
    });
  });

  await context.test("unsupported opset", async () => {
    await withArtifact(async (artifact) => {
      await republish(artifact, (manifest) => {
        manifest.resources[1].onnx.opset = 19;
      });
      await assert.rejects(loadArtifact(artifact.root), hasCode("SEMA_ARTIFACT_INCOMPATIBLE"));
    });
  });

  await context.test("unreferenced resource", async () => {
    await withArtifact(async (artifact) => {
      await republish(artifact, (manifest) => {
        manifest.resources.push({
          ...manifest.resources[3],
          ref: "head.unreferenced",
          path: "models/heads/unreferenced.onnx",
        });
      });
      await assert.rejects(loadArtifact(artifact.root), hasCode("SEMA_ARTIFACT_INVALID_MANIFEST"));
    });
  });

  await context.test("impossible RFC 3339 calendar date", async () => {
    await withArtifact(async (artifact) => {
      await republish(artifact, (manifest) => {
        manifest.build.createdAt = "2026-02-31T00:00:00Z";
      });
      await assert.rejects(loadArtifact(artifact.root), hasCode("SEMA_ARTIFACT_INVALID_MANIFEST"));
    });
  });

  await context.test("invalid ordinal grids and binary64 collisions", async () => {
    await withArtifact(async (artifact) => {
      await republish(artifact, (manifest) => {
        manifest.functions[0].heads[0].type = {
          kind: "ordinal-number",
          sourceKind: "bounded-number",
          minimum: "0",
          maximum: "1",
          step: "1",
          supportDecimal: ["0", "0.5", "1"],
        };
      });
      await assert.rejects(loadArtifact(artifact.root), hasCode("SEMA_ARTIFACT_INVALID_MANIFEST"));
    });

    await withArtifact(async (artifact) => {
      await republish(artifact, (manifest) => {
        manifest.resources[3].onnx.outputs[0].shape[1] = 2;
        manifest.functions[0].heads[0].type = {
          kind: "ordinal-number",
          sourceKind: "bounded-number",
          minimum: "9007199254740992",
          maximum: "9007199254740993",
          step: "1",
          supportDecimal: ["9007199254740992", "9007199254740993"],
        };
      });
      await assert.rejects(loadArtifact(artifact.root), hasCode("SEMA_ARTIFACT_INVALID_MANIFEST"));
    });
  });
});

test("accepts every implemented canonical input encoding and rejects the rest", async (context) => {
  for (const encoding of ["semantscript.canonical-input/v1", "semantscript.canonical-input/v2"]) {
    await context.test(`accepts ${encoding}`, async () => {
      await withArtifact(async (artifact) => {
        await republish(artifact, (manifest) => {
          manifest.compatibility.canonicalInput = encoding;
        });
        const loaded = await loadArtifact(artifact.root);
        assert.equal(loaded.manifest.compatibility.canonicalInput, encoding);
      });
    });
  }
  await context.test("rejects an unimplemented encoding", async () => {
    await withArtifact(async (artifact) => {
      await republish(artifact, (manifest) => {
        manifest.compatibility.canonicalInput = "semantscript.canonical-input/v3";
      });
      await assert.rejects(loadArtifact(artifact.root), hasCode("SEMA_ARTIFACT_INVALID_MANIFEST"));
    });
  });
});

function quantizationBlock(overrides = {}) {
  return {
    method: "dynamic",
    weightType: "int8",
    perChannel: false,
    reduceRange: false,
    argmaxDisagreementTolerance: 0,
    attestedDisagreementTolerance: 0,
    eceThreshold: 0.1,
    sourceManifestSha256: "a".repeat(64),
    ...overrides,
  };
}

test("validates encoder precision and quantization metadata", async (context) => {
  await context.test("float32 without precision is the historical default", async () => {
    await withArtifact(async (artifact) => {
      const loaded = await loadArtifact(artifact.root);
      assert.equal(loaded.manifest.resources[1].onnx.precision, undefined);
    });
  });

  await context.test("explicit float32 precision loads", async () => {
    await withArtifact(async (artifact) => {
      await republish(artifact, (manifest) => {
        manifest.resources[1].onnx.precision = "float32";
      });
      const loaded = await loadArtifact(artifact.root);
      assert.equal(loaded.manifest.resources[1].onnx.precision, "float32");
    });
  });

  await context.test("int8-dynamic precision with its quantization block loads", async () => {
    await withArtifact(async (artifact) => {
      await republish(artifact, (manifest) => {
        manifest.resources[1].onnx.precision = "int8-dynamic";
        manifest.resources[1].onnx.quantization = quantizationBlock();
      });
      const loaded = await loadArtifact(artifact.root);
      assert.equal(loaded.manifest.resources[1].onnx.precision, "int8-dynamic");
      assert.deepEqual(loaded.manifest.resources[1].onnx.quantization, quantizationBlock());
    });
  });

  const rejected = [
    ["unknown precision", (onnx) => { onnx.precision = "int4"; }],
    ["quantized precision without quantization", (onnx) => { onnx.precision = "int8-dynamic"; }],
    ["quantization on float32", (onnx) => { onnx.quantization = quantizationBlock(); }],
    ["quantization missing a field", (onnx) => {
      onnx.precision = "int8-dynamic";
      const rest = quantizationBlock();
      delete rest.eceThreshold;
      onnx.quantization = rest;
    }],
    ["quantization with an extra field", (onnx) => {
      onnx.precision = "int8-dynamic";
      onnx.quantization = quantizationBlock({ calibrated: true });
    }],
    ["unsupported method", (onnx) => {
      onnx.precision = "int8-dynamic";
      onnx.quantization = quantizationBlock({ method: "static" });
    }],
    ["tolerance outside the unit interval", (onnx) => {
      onnx.precision = "int8-dynamic";
      onnx.quantization = quantizationBlock({ argmaxDisagreementTolerance: 1.5 });
    }],
    ["negative attested tolerance", (onnx) => {
      onnx.precision = "int8-dynamic";
      onnx.quantization = quantizationBlock({ attestedDisagreementTolerance: -1 });
    }],
    ["malformed source digest", (onnx) => {
      onnx.precision = "int8-dynamic";
      onnx.quantization = quantizationBlock({ sourceManifestSha256: "xyz" });
    }],
  ];
  for (const [name, mutate] of rejected) {
    await context.test(name, async () => {
      await withArtifact(async (artifact) => {
        await republish(artifact, (manifest) => {
          mutate(manifest.resources[1].onnx);
        });
        await assert.rejects(loadArtifact(artifact.root), hasCode("SEMA_ARTIFACT_INVALID_MANIFEST"));
      });
    });
  }
});

test("validates every runtime confidence policy relationship", async (context) => {
  const validCases = [
    {
      name: "plain scalar without threshold",
      flat: false,
      runtime: policy("value", null, "none", null),
    },
    {
      name: "diagnostic scalar without threshold",
      flat: false,
      runtime: policy("diagnostic", null, "none", null),
    },
    {
      name: "plain thresholded scalar without fallback",
      flat: false,
      runtime: policy("value", 0.9, "scalar-top1", null),
    },
    {
      name: "plain thresholded scalar with opaque fallback",
      flat: false,
      runtime: policy("value", 0.9, "scalar-top1", "Opaque fallback ref / v1"),
    },
    {
      name: "diagnostic thresholded scalar",
      flat: false,
      runtime: policy("diagnostic", 0.9, "scalar-top1", null),
    },
    {
      name: "plain flat output without threshold",
      flat: true,
      runtime: policy("value", null, "none", null),
    },
    {
      name: "diagnostic flat output without threshold",
      flat: true,
      runtime: policy("diagnostic", null, "none", null),
    },
    {
      name: "plain thresholded flat output without fallback",
      flat: true,
      runtime: policy("value", 0.9, "all-fields", null),
    },
    {
      name: "plain thresholded flat output with opaque fallback",
      flat: true,
      runtime: policy("value", 0.9, "all-fields", "Opaque fallback ref / v1"),
    },
    {
      name: "diagnostic thresholded flat output",
      flat: true,
      runtime: policy("diagnostic", 0.9, "all-fields", null),
    },
  ];

  for (const fixture of validCases) {
    await context.test(fixture.name, async () => {
      await withArtifact(async (artifact) => {
        await republish(artifact, (manifest) => {
          if (fixture.flat) makeFixtureOutputFlat(manifest);
          manifest.functions[0].runtime = fixture.runtime;
        });
        const loaded = await loadArtifact(artifact.root);
        assert.deepEqual(loaded.functions[0].runtime, fixture.runtime);
      });
    });
  }

  const invalidCases = [
    {
      name: "scalar policy without threshold",
      flat: false,
      runtime: policy("value", null, "scalar-top1", null),
    },
    {
      name: "fallback without threshold",
      flat: false,
      runtime: policy("value", null, "none", "fallback.manual"),
    },
    {
      name: "thresholded scalar with no policy",
      flat: false,
      runtime: policy("value", 0.9, "none", null),
    },
    {
      name: "thresholded scalar with flat policy",
      flat: false,
      runtime: policy("value", 0.9, "all-fields", null),
    },
    {
      name: "dead diagnostic scalar fallback",
      flat: false,
      runtime: policy("diagnostic", 0.9, "scalar-top1", "fallback.manual"),
    },
    {
      name: "flat policy without threshold",
      flat: true,
      runtime: policy("diagnostic", null, "all-fields", null),
    },
    {
      name: "thresholded flat output with no policy",
      flat: true,
      runtime: policy("value", 0.9, "none", null),
    },
    {
      name: "thresholded flat output with scalar policy",
      flat: true,
      runtime: policy("value", 0.9, "scalar-top1", null),
    },
    {
      name: "dead diagnostic flat fallback",
      flat: true,
      runtime: policy("diagnostic", 0.9, "all-fields", "fallback.manual"),
    },
  ];

  for (const fixture of invalidCases) {
    await context.test(fixture.name, async () => {
      await withArtifact(async (artifact) => {
        await republish(artifact, (manifest) => {
          if (fixture.flat) makeFixtureOutputFlat(manifest);
          manifest.functions[0].runtime = fixture.runtime;
        });
        await assert.rejects(
          loadArtifact(artifact.root),
          hasCode("SEMA_ARTIFACT_INVALID_MANIFEST"),
        );
      });
    });
  }
});

test("rejects resource quota violations, tampering, and symlinks", async (context) => {
  await context.test("quota", async () => {
    await withArtifact(async (artifact) => {
      await assert.rejects(
        loadArtifact(artifact.root, { maximumResourceBytes: 1 }),
        hasCode("SEMA_ARTIFACT_QUOTA"),
      );
    });
  });

  await context.test("digest", async () => {
    await withArtifact(async (artifact) => {
      const resourcePath = join(artifact.release, artifact.manifest.resources[3].path);
      await writeFile(resourcePath, "tampered");
      await assert.rejects(loadArtifact(artifact.root), hasCode("SEMA_ARTIFACT_INTEGRITY"));
    });
  });

  await context.test("symlink", async (subcontext) => {
    await withArtifact(async (artifact) => {
      const resourcePath = join(artifact.release, artifact.manifest.resources[3].path);
      const target = join(artifact.root, "outside.onnx");
      await writeFile(target, "outside");
      await unlink(resourcePath);
      try {
        await symlink(target, resourcePath);
      } catch (error) {
        if (error && (error.code === "EPERM" || error.code === "EACCES")) {
          subcontext.skip("symlink creation is unavailable");
          return;
        }
        throw error;
      }
      await assert.rejects(loadArtifact(artifact.root), hasCode("SEMA_ARTIFACT_PATH"));
    });
  });
});

test("prepares only after all verification and rolls back backend failures", async () => {
  await withArtifact(async (artifact) => {
    const prepared = [];
    const disposed = [];
    const backend = {
      prepareResource(resource, bytes) {
        assert.ok(bytes.byteLength > 0);
        if (prepared.length === 2) throw new Error("backend failure");
        prepared.push(resource.ref);
        return resource.ref;
      },
      disposeResource(ref) {
        disposed.push(ref);
      },
    };
    await assert.rejects(
      loadArtifact(artifact.root, { backend }),
      hasCode("SEMA_ARTIFACT_RESOURCE"),
    );
    assert.deepEqual(disposed, prepared.toReversed());
  });
});

test("reuses exact standalone file buffers and copies only non-standalone views", async (context) => {
  await context.test("exact standalone buffers", async () => {
    await withArtifact(async (artifact) => {
      const probe = await open(join(artifact.release, "manifest.json"), "r");
      const prototype = Object.getPrototypeOf(probe);
      const descriptor = Object.getOwnPropertyDescriptor(prototype, "readFile");
      await probe.close();
      assert.ok(descriptor?.value);

      const fileBackings = new Set();
      const preparedBackings = [];
      Object.defineProperty(prototype, "readFile", {
        ...descriptor,
        async value(...arguments_) {
          const bytes = await Reflect.apply(descriptor.value, this, arguments_);
          if (
            bytes.buffer instanceof ArrayBuffer &&
            bytes.byteOffset === 0 &&
            bytes.byteLength === bytes.buffer.byteLength
          ) {
            fileBackings.add(bytes.buffer);
          }
          return bytes;
        },
      });

      try {
        await loadArtifact(artifact.root, {
          backend: {
            prepareResource(_resource, bytes) {
              assert.equal(bytes.byteOffset, 0);
              assert.equal(bytes.byteLength, bytes.buffer.byteLength);
              preparedBackings.push(bytes.buffer);
              return bytes;
            },
          },
        });
      } finally {
        Object.defineProperty(prototype, "readFile", descriptor);
      }

      assert.equal(preparedBackings.length, artifact.manifest.resources.length);
      assert.ok(preparedBackings.every((backing) => fileBackings.has(backing)));
    });
  });

  await context.test("non-standalone fallback", async () => {
    await withArtifact(async (artifact) => {
      const probe = await open(join(artifact.release, "manifest.json"), "r");
      const prototype = Object.getPrototypeOf(probe);
      const descriptor = Object.getOwnPropertyDescriptor(prototype, "readFile");
      await probe.close();
      assert.ok(descriptor?.value);

      const sourceBackings = new Set();
      const preparedBackings = [];
      Object.defineProperty(prototype, "readFile", {
        ...descriptor,
        async value(...arguments_) {
          const bytes = await Reflect.apply(descriptor.value, this, arguments_);
          const padded = Buffer.allocUnsafe(bytes.byteLength + 2);
          bytes.copy(padded, 1);
          const view = padded.subarray(1, padded.byteLength - 1);
          sourceBackings.add(view.buffer);
          return view;
        },
      });

      try {
        await loadArtifact(artifact.root, {
          backend: {
            prepareResource(_resource, bytes) {
              assert.equal(bytes.byteOffset, 0);
              assert.equal(bytes.byteLength, bytes.buffer.byteLength);
              preparedBackings.push(bytes.buffer);
              return bytes;
            },
          },
        });
      } finally {
        Object.defineProperty(prototype, "readFile", descriptor);
      }

      assert.equal(preparedBackings.length, artifact.manifest.resources.length);
      assert.ok(preparedBackings.every((backing) => !sourceBackings.has(backing)));
    });
  });
});

test("accepts the optional held-out constraint figure and rejects a malformed one", async (context) => {
  const valid = { sampleSize: 512, violations: 3, violationRate: 3 / 512, seed: 5 };
  await context.test("accepts a release without it and one with it", async () => {
    await withArtifact(async (artifact) => {
      const loaded = await loadArtifact(artifact.root);
      assert.equal(loaded.manifest.functions[0].verification.heldOutConstraints, undefined);
    });
    await withArtifact(async (artifact) => {
      await republish(artifact, (manifest) => {
        manifest.functions[0].verification.heldOutConstraints = valid;
      });
      const loaded = await loadArtifact(artifact.root);
      assert.deepEqual(loaded.manifest.functions[0].verification.heldOutConstraints, valid);
    });
  });
  for (const [name, value] of [
    ["an extra key", { ...valid, violatedChecks: 3 }],
    ["a missing seed", { sampleSize: 512, violations: 3, violationRate: 3 / 512 }],
    ["more violations than inputs", { ...valid, violations: 600, violationRate: 1 }],
    ["a rate that disagrees with the counts", { ...valid, violationRate: 0.5 }],
    ["a negative seed", { ...valid, seed: -1 }],
  ]) {
    await context.test(`rejects ${name}`, async () => {
      await withArtifact(async (artifact) => {
        await republish(artifact, (manifest) => {
          manifest.functions[0].verification.heldOutConstraints = value;
        });
        await assert.rejects(loadArtifact(artifact.root), hasCode("SEMA_ARTIFACT_INVALID_MANIFEST"));
      });
    });
  }
});
