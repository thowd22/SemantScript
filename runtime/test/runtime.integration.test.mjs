import assert from "node:assert/strict";
import { appendFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { createFixtureArtifact, fixtureFunctionId } from "./fixtures/artifact.mjs";

test("loads a verified artifact and executes a synchronous ONNX forward pass", async () => {
  const runtime = await import("../dist/index.js");
  const root = await mkdtemp(join(tmpdir(), "semantscript-runtime-"));
  const brokenRoot = await mkdtemp(join(tmpdir(), "semantscript-runtime-broken-"));
  const abiBrokenRoot = await mkdtemp(join(tmpdir(), "semantscript-runtime-abi-broken-"));
  const replacementRoot = await mkdtemp(join(tmpdir(), "semantscript-runtime-replacement-"));
  const oversizedRoot = await mkdtemp(join(tmpdir(), "semantscript-runtime-oversized-"));

  try {
    const artifact = await createFixtureArtifact(root);
    const handle = await runtime.loadSemaArtifact(root);
    assert.equal(handle.manifestSha256, artifact.manifestSha256);
    assert.deepEqual([...handle.functionIds], [fixtureFunctionId]);
    assert.equal(handle.call(fixtureFunctionId, { facts: { a: 1, b: 2 } }), "review");

    const first = runtime.__sema.call(fixtureFunctionId, { facts: { a: 1, b: 2 } });
    const reordered = runtime.__sema.call(fixtureFunctionId, { facts: { b: 2, a: 1 } });
    assert.equal(first, "review");
    assert.equal(reordered, first);

    assert.throws(
      () => runtime.__sema.call(fixtureFunctionId, { facts: { a: 1 } }),
      runtime.SemaInputError,
    );
    assert.throws(
      () => runtime.__sema.call(`nf_${"f".repeat(64)}`, { facts: { a: 1, b: 2 } }),
      runtime.SemaUnknownFunctionError,
    );

    const broken = await createFixtureArtifact(brokenRoot);
    await appendFile(
      join(broken.release, `models/heads/${fixtureFunctionId}/head-000.onnx`),
      "corruption",
    );
    await assert.rejects(runtime.loadSemaArtifact(brokenRoot), runtime.ArtifactLoadError);
    assert.equal(
      runtime.__sema.call(fixtureFunctionId, { facts: { a: 3, b: 4 } }),
      "review",
      "a failed reload must leave the preceding artifact active",
    );

    await createFixtureArtifact(abiBrokenRoot, { headSource: "wrong-head.onnx" });
    await assert.rejects(
      runtime.loadSemaArtifact(abiBrokenRoot),
      runtime.SemaInferenceInitializationError,
    );
    assert.equal(
      runtime.__sema.call(fixtureFunctionId, { facts: { a: 5, b: 6 } }),
      "review",
      "a post-staging ABI failure must leave the preceding artifact active",
    );

    await createFixtureArtifact(oversizedRoot, {
      transformManifest(manifest) {
        manifest.functions[0].heads[0].type.support[0] = "x".repeat(1_048_577);
      },
    });
    await assert.rejects(
      runtime.loadSemaArtifact(oversizedRoot),
      (error) => error instanceof runtime.ArtifactLoadError && error.code === "SEMA_ARTIFACT_QUOTA",
    );

    await createFixtureArtifact(replacementRoot, {
      encoderSource: "fixed-batch-encoder.onnx",
      adapterSource: "fixed-batch-adapter.onnx",
      headSource: "fixed-batch-head.onnx",
      transformManifest(manifest) {
        for (const resource of manifest.resources) {
          if (resource.format !== "onnx") {
            continue;
          }
          for (const tensor of [...resource.onnx.inputs, ...resource.onnx.outputs]) {
            tensor.shape[0] = 1;
          }
        }
      },
    });
    const replacement = await runtime.loadSemaArtifact(replacementRoot);
    assert.throws(
      () => handle.call(fixtureFunctionId, { facts: { a: 7, b: 8 } }),
      (error) =>
        error instanceof runtime.SemaArtifactInactiveError &&
        error.code === "SEMA_ARTIFACT_INACTIVE" &&
        error.manifestSha256 === artifact.manifestSha256,
      "a stale handle must fail closed instead of dispatching through its replacement",
    );
    assert.equal(
      replacement.call(fixtureFunctionId, { facts: { a: 7, b: 8 } }),
      "review",
      "the active replacement handle must dispatch its own artifact",
    );
    await handle.close();
    assert.equal(
      runtime.__sema.call(fixtureFunctionId, { facts: { a: 7, b: 8 } }),
      "review",
      "closing a stale handle must not unload its replacement",
    );
    await replacement.close();
    assert.throws(
      () => replacement.call(fixtureFunctionId, { facts: { a: 1, b: 2 } }),
      runtime.SemaArtifactInactiveError,
      "a closed handle must fail closed",
    );
    assert.throws(
      () => runtime.__sema.call(fixtureFunctionId, { facts: { a: 1, b: 2 } }),
      runtime.SemaRuntimeNotLoadedError,
    );
  } finally {
    await runtime.closeSemaArtifact();
    await Promise.all([
      rm(root, { recursive: true, force: true }),
      rm(brokenRoot, { recursive: true, force: true }),
      rm(abiBrokenRoot, { recursive: true, force: true }),
      rm(replacementRoot, { recursive: true, force: true }),
      rm(oversizedRoot, { recursive: true, force: true }),
    ]);
  }
});

test("active inference input quota is enforced during canonical serialization", async () => {
  const runtime = await import("../dist/index.js");
  const { serializeCanonicalInputs } = await import("../dist/canonical-input.js");
  const root = await mkdtemp(join(tmpdir(), "semantscript-runtime-input-quota-"));

  try {
    const artifact = await createFixtureArtifact(root);
    const inputs = { facts: { a: 1, b: 2 } };
    const canonical = serializeCanonicalInputs(artifact.manifest.functions[0].inputs, inputs);
    const atBoundary = await runtime.loadSemaArtifact(root, {
      inference: { maximumInputBytes: canonical.byteLength },
    });
    assert.equal(runtime.__sema.call(fixtureFunctionId, inputs), "review");
    await atBoundary.close();

    const belowBoundary = await runtime.loadSemaArtifact(root, {
      inference: { maximumInputBytes: canonical.byteLength - 1 },
    });
    assert.throws(
      () => runtime.__sema.call(fixtureFunctionId, inputs),
      (error) => error instanceof runtime.SemaInputError && error.reason === "limit",
    );
    await belowBoundary.close();
  } finally {
    await runtime.closeSemaArtifact();
    await rm(root, { recursive: true, force: true });
  }
});

test("returns calibrated diagnostics and enforces scalar confidence fallbacks", async () => {
  const runtime = await import("../dist/index.js");
  const diagnosticRoot = await mkdtemp(join(tmpdir(), "semantscript-runtime-diagnostic-"));
  const lowRoot = await mkdtemp(join(tmpdir(), "semantscript-runtime-low-confidence-"));
  const highRoot = await mkdtemp(join(tmpdir(), "semantscript-runtime-high-confidence-"));
  const numberRoot = await mkdtemp(join(tmpdir(), "semantscript-runtime-number-fallback-"));
  const fallbackRef = "fallback.fixture";

  try {
    await createFixtureArtifact(diagnosticRoot, {
      transformManifest(manifest) {
        Object.assign(manifest.functions[0].runtime, {
          resultMode: "diagnostic",
          confidenceThreshold: 1,
          policy: "scalar-top1",
          fallbackRef: null,
        });
      },
    });
    await runtime.loadSemaArtifact(diagnosticRoot);
    const diagnostic = runtime.__sema.call(fixtureFunctionId, { facts: { a: 1, b: 2 } });
    assert.deepEqual(Object.keys(diagnostic), [
      "value",
      "confidence",
      "uncertainty",
      "distribution",
      "expectedValue",
    ]);
    assert.equal(diagnostic.value, "review");
    assert.equal(diagnostic.confidence, Math.max(...diagnostic.distribution.map(({ probability }) => probability)));
    assert.ok(diagnostic.uncertainty >= 0 && diagnostic.uncertainty <= 1);
    assert.deepEqual(
      diagnostic.distribution.map(({ value }) => value),
      ["approve", "deny", "review"],
    );
    assert.ok(
      Math.abs(
        diagnostic.distribution.reduce((sum, { probability }) => sum + probability, 0) - 1,
      ) < 1e-12,
    );
    assert.equal(diagnostic.expectedValue, null);
    assert.ok(diagnostic.confidence < 1, "diagnostic mode returns below its configured threshold");

    await createFixtureArtifact(lowRoot, {
      transformManifest(manifest) {
        Object.assign(manifest.functions[0].runtime, {
          resultMode: "value",
          confidenceThreshold: 1,
          policy: "scalar-top1",
          fallbackRef: null,
        });
      },
    });
    await runtime.loadSemaArtifact(lowRoot);
    assert.throws(
      () => runtime.__sema.call(fixtureFunctionId, { facts: { a: 1, b: 2 } }),
      (error) =>
        error instanceof runtime.SemaConfidenceError &&
        error.code === "SEMA_CONFIDENCE_BELOW_THRESHOLD" &&
        error.functionId === fixtureFunctionId &&
        error.threshold === 1 &&
        error.diagnostic.value === "review",
    );

    let firstFallbackCalls = 0;
    let replacementFallbackCalls = 0;
    let observedInputs;
    let observedDiagnostic;
    let observedThreshold;
    const firstFallback = (inputs, fallbackDiagnostic, threshold) => {
      firstFallbackCalls += 1;
      observedInputs = inputs;
      observedDiagnostic = fallbackDiagnostic;
      observedThreshold = threshold;
      return "approve";
    };
    const registry = new Map([[fallbackRef, firstFallback]]);
    const fallbackRoot = await mkdtemp(join(tmpdir(), "semantscript-runtime-fallback-"));
    try {
      await createFixtureArtifact(fallbackRoot, {
        transformManifest(manifest) {
          Object.assign(manifest.functions[0].runtime, {
            resultMode: "value",
            confidenceThreshold: 1,
            policy: "scalar-top1",
            fallbackRef,
          });
        },
      });
      const pendingLoad = runtime.loadSemaArtifact(fallbackRoot, { fallbacks: registry });
      registry.set(fallbackRef, () => {
        replacementFallbackCalls += 1;
        return "deny";
      });
      await pendingLoad;

      const inputs = { facts: { a: 3, b: 4 } };
      assert.equal(runtime.__sema.call(fixtureFunctionId, inputs), "approve");
      assert.equal(firstFallbackCalls, 1);
      assert.equal(replacementFallbackCalls, 0);
      assert.equal(observedInputs, inputs, "fallback receives the validated original input record");
      assert.equal(observedDiagnostic.value, "review");
      assert.equal(observedThreshold, 1);

      await assert.rejects(
        runtime.loadSemaArtifact(fallbackRoot),
        (error) => error instanceof runtime.SemaFallbackError && error.reason === "missing",
      );
      assert.equal(
        runtime.__sema.call(fixtureFunctionId, { facts: { a: 5, b: 6 } }),
        "approve",
        "a missing fallback on reload must leave the preceding artifact active",
      );

      await runtime.loadSemaArtifact(fallbackRoot, {
        fallbacks: new Map([[fallbackRef, () => "outside-support"]]),
      });
      assert.throws(
        () => runtime.__sema.call(fixtureFunctionId, { facts: { a: 7, b: 8 } }),
        (error) => error instanceof runtime.SemaFallbackError && error.reason === "invalid-result",
      );

      await runtime.loadSemaArtifact(fallbackRoot, {
        fallbacks: new Map([
          [
            fallbackRef,
            () => {
              const cyclic = {};
              cyclic.self = cyclic;
              return cyclic;
            },
          ],
        ]),
      });
      assert.throws(
        () => runtime.__sema.call(fixtureFunctionId, { facts: { a: 9, b: 10 } }),
        (error) => error instanceof runtime.SemaFallbackError && error.reason === "invalid-result",
      );

      const callbackFailure = new Error("fallback failed");
      await runtime.loadSemaArtifact(fallbackRoot, {
        fallbacks: new Map([
          [
            fallbackRef,
            () => {
              throw callbackFailure;
            },
          ],
        ]),
      });
      assert.throws(
        () => runtime.__sema.call(fixtureFunctionId, { facts: { a: 11, b: 12 } }),
        (error) => error === callbackFailure,
      );

      let recurseOnce = true;
      await runtime.loadSemaArtifact(fallbackRoot, {
        fallbacks: new Map([
          [
            fallbackRef,
            (recursiveInputs) => {
              if (recurseOnce) {
                recurseOnce = false;
                return runtime.__sema.call(fixtureFunctionId, recursiveInputs);
              }
              return "approve";
            },
          ],
        ]),
      });
      assert.throws(
        () => runtime.__sema.call(fixtureFunctionId, { facts: { a: 13, b: 14 } }),
        (error) => error instanceof runtime.SemaFallbackError && error.reason === "cycle",
      );
      assert.equal(
        runtime.__sema.call(fixtureFunctionId, { facts: { a: 15, b: 16 } }),
        "approve",
        "fallback invocation stack is cleared after a recursive callback throws",
      );
    } finally {
      await rm(fallbackRoot, { recursive: true, force: true });
    }

    let highFallbackCalls = 0;
    await createFixtureArtifact(highRoot, {
      transformManifest(manifest) {
        Object.assign(manifest.functions[0].runtime, {
          resultMode: "value",
          confidenceThreshold: 0,
          policy: "scalar-top1",
          fallbackRef,
        });
      },
    });
    await runtime.loadSemaArtifact(highRoot, {
      fallbacks: new Map([
        [
          fallbackRef,
          () => {
            highFallbackCalls += 1;
            return "approve";
          },
        ],
      ]),
    });
    assert.equal(runtime.__sema.call(fixtureFunctionId, { facts: { a: 17, b: 18 } }), "review");
    assert.equal(highFallbackCalls, 0, "an inclusive passing threshold must not invoke fallback");

    await createFixtureArtifact(numberRoot, {
      transformManifest(manifest) {
        manifest.functions[0].heads[0].type = {
          kind: "nominal-number",
          support: [0, 1, 2],
        };
        Object.assign(manifest.functions[0].runtime, {
          resultMode: "value",
          confidenceThreshold: 1,
          policy: "scalar-top1",
          fallbackRef,
        });
      },
    });
    await runtime.loadSemaArtifact(numberRoot, {
      fallbacks: new Map([[fallbackRef, () => -0]]),
    });
    assert.throws(
      () => runtime.__sema.call(fixtureFunctionId, { facts: { a: 19, b: 20 } }),
      (error) => error instanceof runtime.SemaFallbackError && error.reason === "invalid-result",
      "fallback support membership uses Object.is and distinguishes -0 from +0",
    );
  } finally {
    await runtime.closeSemaArtifact();
    await Promise.all([
      rm(diagnosticRoot, { recursive: true, force: true }),
      rm(lowRoot, { recursive: true, force: true }),
      rm(highRoot, { recursive: true, force: true }),
      rm(numberRoot, { recursive: true, force: true }),
    ]);
  }
});

test("validates flat fallback results as exact safe support objects", async () => {
  const runtime = await import("../dist/index.js");
  const root = await mkdtemp(join(tmpdir(), "semantscript-runtime-flat-fallback-"));
  const fallbackRef = "fallback.flat-fixture";

  try {
    await createFixtureArtifact(root, {
      transformManifest(manifest) {
        const scalarHead = manifest.functions[0].heads[0];
        manifest.functions[0].heads = [
          { ...cloneJson(scalarHead), outputPath: ["decision"] },
          { ...cloneJson(scalarHead), outputPath: ["route"] },
        ];
        Object.assign(manifest.functions[0].runtime, {
          resultMode: "value",
          confidenceThreshold: 1,
          policy: "all-fields",
          fallbackRef,
        });
      },
    });

    await runtime.loadSemaArtifact(root, {
      fallbacks: new Map([
        [fallbackRef, () => ({ decision: "approve", route: "deny" })],
      ]),
    });
    assert.deepEqual(
      runtime.__sema.call(fixtureFunctionId, { facts: { a: 1, b: 2 } }),
      { decision: "approve", route: "deny" },
    );

    const accessorResult = { route: "deny" };
    Object.defineProperty(accessorResult, "decision", {
      enumerable: true,
      get() {
        return "approve";
      },
    });
    class ClassResult {
      decision = "approve";
      route = "deny";
    }
    const symbolResult = { decision: "approve", route: "deny" };
    symbolResult[Symbol("hidden")] = true;
    const invalidResults = [
      accessorResult,
      Promise.resolve({ decision: "approve", route: "deny" }),
      new ClassResult(),
      new Proxy({ decision: "approve", route: "deny" }, {}),
      symbolResult,
      { decision: "approve" },
      { decision: "approve", route: "deny", extra: false },
    ];
    await runtime.loadSemaArtifact(root, {
      fallbacks: new Map([[fallbackRef, () => invalidResults.shift()]]),
    });
    for (const [index, description] of [
      "accessor",
      "promise",
      "class instance",
      "proxy",
      "symbol key",
      "missing field",
      "extra field",
    ].entries()) {
      assert.throws(
        () => runtime.__sema.call(fixtureFunctionId, { facts: { a: index + 3, b: index + 4 } }),
        (error) => error instanceof runtime.SemaFallbackError && error.reason === "invalid-result",
        `fallback results reject ${description}`,
      );
    }
  } finally {
    await runtime.closeSemaArtifact();
    await rm(root, { recursive: true, force: true });
  }
});

test("allows nested fallbacks while rejecting cross-function invocation cycles", async () => {
  const runtime = await import("../dist/index.js");
  const root = await mkdtemp(join(tmpdir(), "semantscript-runtime-nested-fallback-"));
  const secondFunctionId = `nf_${"7".repeat(64)}`;
  const firstRef = "fallback.first";
  const secondRef = "fallback.second";
  const inputs = { facts: { a: 1, b: 2 } };

  try {
    await createFixtureArtifact(root, {
      transformManifest(manifest) {
        const first = manifest.functions[0];
        Object.assign(first.runtime, {
          resultMode: "value",
          confidenceThreshold: 1,
          policy: "scalar-top1",
          fallbackRef: firstRef,
        });
        const second = cloneJson(first);
        second.id = secondFunctionId;
        second.semanticSha256 = "8".repeat(64);
        second.runtime.fallbackRef = secondRef;
        manifest.functions.push(second);
      },
    });

    const noncyclic = new Map([
      [firstRef, (fallbackInputs) => runtime.__sema.call(secondFunctionId, fallbackInputs)],
      [secondRef, () => "approve"],
    ]);
    await runtime.loadSemaArtifact(root, { fallbacks: noncyclic });
    assert.equal(runtime.__sema.call(fixtureFunctionId, inputs), "approve");

    await runtime.loadSemaArtifact(root, {
      fallbacks: new Map([
        [firstRef, (fallbackInputs) => runtime.__sema.call(secondFunctionId, fallbackInputs)],
        [secondRef, (fallbackInputs) => runtime.__sema.call(fixtureFunctionId, fallbackInputs)],
      ]),
    });
    assert.throws(
      () => runtime.__sema.call(fixtureFunctionId, inputs),
      (error) =>
        error instanceof runtime.SemaFallbackError &&
        error.reason === "cycle" &&
        error.functionId === fixtureFunctionId,
    );

    await runtime.loadSemaArtifact(root, { fallbacks: noncyclic });
    assert.equal(
      runtime.__sema.call(fixtureFunctionId, inputs),
      "approve",
      "a prior cross-function cycle does not poison later fallback invocation",
    );
  } finally {
    await runtime.closeSemaArtifact();
    await rm(root, { recursive: true, force: true });
  }
});

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}
