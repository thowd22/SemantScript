# Runtime

The runtime package loads and verifies immutable application artifacts, validates
and canonically serializes function inputs, executes encoder/adapter/head inference,
applies per-head temperature calibration, and enforces confidence policy.

It performs no prompt interpretation, teacher-model calls, or training in a
deployed application.

The package also exposes the source-language declarations for `sema`, `always`,
`never`, `Ordinal`, `BoundedInt`, `BoundedNumber`, and confidence result types.
Calling an uncompiled `sema` tag throws: compiled applications replace each tag
with an artifact-backed runtime call before execution.

Compiled output imports the internal `__sema` binding and calls
`__sema.call(functionId, inputs)`. Call `loadSemaArtifact(path)` once during
application startup and await it before importing or invoking compiled code. A
successful load atomically activates the complete immutable artifact; a failed
reload leaves the preceding artifact active. `path` may name a release directory
containing `manifest.json` or an artifact root containing `current.json`.

The compiler ABI remains synchronous. ONNX Runtime's JavaScript API is asynchronous,
so the package owns a dedicated worker thread and uses a bounded shared-memory
request/response bridge for each call. Tokenization and encoder/adapter/head ONNX
execution remain in-process, local, and free of Python or network access. A call
made before a successful load, an unknown function ID, malformed inputs, and an
inference failure each throw a distinct typed runtime error.

## Confidence results and fallbacks

`sema.withConfidence<T>` returns the diagnostic shape declared by
`ScalarSemaResult<T>` or `ObjectSemaResult<T>`. Scalar confidence is calibrated
top-1 probability and uncertainty is normalized entropy. A scalar distribution
contains every support value in manifest order. Nominal outputs have a null
`expectedValue`; ordered strings use probability-weighted zero-based rank and
bounded numbers use their probability-weighted numeric value. Flat results expose
one scalar diagnostic per field together with minimum field confidence and maximum
field uncertainty; those aggregates are not a joint probability or entropy.
For a support of size `K`, decoded distribution probabilities must sum to one
within `max(Number.EPSILON, 8 * K * Number.EPSILON)` absolute tolerance.

A plain `sema<T>` call without `@confidence` still returns only `T`. With a
threshold, confidence equal to the threshold passes. A lower scalar confidence—or
any lower field confidence for a flat output—invokes the artifact's configured
synchronous fallback. If the artifact has no fallback reference, the call throws
`SemaConfidenceError` and does not return the low-confidence value. Diagnostic
calls always return diagnostics, including below threshold, and never invoke a
fallback.

Fallback implementations are supplied while loading an artifact:

```ts
await loadSemaArtifact(artifactPath, {
  fallbacks: new Map([
    [
      "refund.manual-review",
      (inputs, diagnostic, requiredConfidence) => {
        // This callback is synchronous and must return the declared output type.
        return "review";
      },
    ],
  ]),
});
```

The runtime snapshots the map when `loadSemaArtifact` is called and resolves every
referenced callback before starting the inference worker. Failed resolution leaves
the previously active artifact unchanged. A fallback receives the validated
original named inputs, the complete diagnostic result, and the required threshold.
Its return is checked exactly against the artifact's scalar support or flat output
fields; promises, proxies, accessors, symbols, class instances, cycles, missing or
extra fields, and out-of-support values throw `SemaFallbackError`. Numeric support
membership uses `Object.is`, so `-0` and `0` remain distinct. Callback-thrown errors
propagate unchanged. Recursive fallback invocation is rejected with
`SemaFallbackError` reason `cycle`, while non-recursive calls to other semantic
functions remain valid.

## Stages and execution plans

`handle.callStage(entries)` (also `__sema.callStage`) runs several functions as
one execution-plan stage. Entries whose canonical inputs are byte-identical share
one encoder pass, and one adapter pass per adapter, before every function's heads
run; the returned `passes` counts (`encoder`, `adapter`, `head`) make the fusion
observable and the results are exactly what independent calls return, including
each function's confidence policy and fallback. `executeSemaPlan(plan, provide)`
runs the compiler's execution plan stage by stage: `provide(stage, resultsSoFar)`
returns the inputs of every function in the stage, so a later stage's inputs can
be built from earlier results, and the outcome maps every function id to its
result with per-stage pass counts.
