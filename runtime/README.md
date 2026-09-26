# Runtime

```sh
npm install @semantscript/core
```

Published as `@semantscript/core` (first publish pending, see
[releasing](../docs/releasing.md)); `VERSION` is the release version it shares
with the other SemantScript packages.

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
containing `manifest.json` or an artifact root containing `current.json`. With no `path`, the runtime resolves the artifact
without environment-specific setup (`defaultSemaArtifactPath()`):
`SEMANTSCRIPT_ARTIFACT` when set; else the first `.semantscript/artifact`
found walking up from the entry script's directory (the compiled output, so a
Docker image or function bundle that ships the artifact beside `dist/` needs
nothing else) and then from the working directory; else
`.semantscript/artifact` under the working directory, where `semantscript init`
and `train` put it. With `{ watch: true }` the runtime also watches the
artifact root's `current.json` and reloads through the same lifecycle whenever
the pointer changes, which is how `semantscript dev` hot-swaps a retrained
head into a running process: a release that fails to load leaves the previous
artifact active (`onReloadError`), a successful one retires the previous
handle and calls `onReload` with the new one; compiled code always goes
through the active artifact. `closeSemaArtifact()` stops the watcher.
`checkSemaArtifact(path)` runs every check a load makes before it starts ONNX
sessions (pointer, digests, schema, ABI compatibility, tensors, opsets and the
model chain) without activating anything, and throws the same
`ArtifactLoadError`; `semantscript releases rollback` uses it to refuse a
release the runtime would not load.

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

## Tracing calls

Two `loadSemaArtifact` options exist for debugging tools such as
`semantscript explain`, and change nothing when absent:

```ts
await loadSemaArtifact(artifactPath, {
  diagnostics: "always",
  observe: (observation) => {
    // { kind: "answered", functionId, inputs, resultMode, confidenceThreshold, diagnostic }
    // or { kind: "missing", functionId, inputs }
  },
});
```

`diagnostics: "always"` makes every function compute its calibrated result, not
only `sema.withConfidence` sites and `@confidence` thresholds; program code still
receives exactly what it would otherwise (the plain value from a value-mode site),
at the cost of the distribution's response bytes on every call. `observe` is
called synchronously for every dispatched call, single or stage entry, before the
confidence policy runs, with the diagnostic result when the function computes one;
a call whose id the artifact lacks is reported as `kind: "missing"` just before
`SemaUnknownFunctionError` is thrown. An observer that throws makes the call throw.

## Routed domains and encoder prefixes

A manifest may carry several adapter resources and several encoder resources.
Every function names its adapter (`adapterRef`) and, when its domain runs a
prefix of the shared encoder (depth routing), the encoder resource it reads
(`encoderRef`; absent means the model's encoder). The worker loads one session
per encoder and adapter and, within a stage, runs one encoder pass per
distinct (encoder, input) pair and one adapter pass per distinct (adapter,
input) pair, so functions of one domain share their prefix pass while
functions of different depths each run their own. `passes.encoder` counts
those passes.

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

## Request scopes

`withSemaScope(fn)` runs `fn` (synchronous or async) inside one request scope:
the sema calls made while it runs share encoder and adapter passes over
identical inputs, the way one execution-plan stage fuses them, without changing
call order or the synchronous ABI. The worker keeps the embeddings a scope
produces (at most 64 per kind) and drops them when `fn` settles;
`semaScopePasses()` reports the passes performed so far in the current scope.
Scopes are what the framework opens per HTTP request.

## Testing without a trained model

`@semantscript/core/testing` builds a stub artifact from the IR bundle
`semantscript build` writes, so application code that calls sema expressions
can be tested with no training and no model files. The stub is an ordinary
artifact: a manifest derived from the bundle (function ids, inputs, heads,
adapters, depth-routed encoders and each function's `@confidence` policy) and
tiny generated ONNX graphs, loaded by `loadSemaArtifact`. The real inference
worker runs every call, so request scopes, stage and scope pass counts, input
validation, confidence thresholds and fallbacks behave as with a trained
artifact; only the value each function answers comes from the test.

```js
import { loadSemaStubArtifact } from "@semantscript/core/testing";
import { decideRefund, refundMethod, refundRisk } from "./dist/refunds.sem.js";

const o1 = {
  customer: { tier: "standard", priorRefunds: 1 },
  order: { total: 88.5, ageDays: 12, status: "paid" },
};
const stub = await loadSemaStubArtifact("dist/semantscript.ir.v1.json", {
  answers: [
    // One value per input, and a default.
    [
      decideRefund,
      { byInput: [{ inputs: o1, value: "review" }], otherwise: "approve" },
    ],
    // A fixed value with its confidence.
    [refundRisk, { value: "high", confidence: 0.6 }],
    // A function of the call's inputs.
    [
      refundMethod,
      {
        compute: ({ payment }) =>
          payment.method === "card" ? "original-payment" : "manual",
      },
    ],
  ],
  fallbacks: new Map(), // any loadSemaArtifact option passes through
});
// ... exercise the application ...
await stub.close(); // closes the artifact and removes the stub
```

- `answers` is a `Map`, a list of `[key, answer]` pairs or an object keyed by
  function id. A key is a function id or the compiled function that calls one
  sema expression (its id is read from the compiled source; `semaFunctionId`
  shows which). Keying by compiled function needs a `Map` or pairs; an
  object's keys must be `nf_` ids.
- A compiled function that calls several sema expressions, such as a staged
  `screenOrder` that asks three questions, has no single id: answer each
  expression by its id. Creating the stub with such a key throws
  `SemaStubError` (`unresolved-function`) listing each expression's source
  position and id. The IR bundle lists them too: every entry of
  `functions` has its `id` and its `source` (`path`, `line`, `column`), so
  `bundle.functions.find((fn) => fn.source.line === 36).id` picks one by
  position.
- An answer is a fixed value, `{ value, confidence }`,
  `{ byInput: [{ inputs, value, confidence? }], otherwise? }` (inputs matched
  on their canonical serialization, so key order does not matter) or
  `{ compute: (inputs) => value | { value, confidence } }`, where `compute`
  is synchronous (sema calls answer synchronously; a returned `Promise` is
  refused) and `inputs` is typed loosely so TypeScript tests can destructure
  it. A flat object
  output's value is `{ value: { ...fields } }` and its confidence one number
  or one per field. Confidence defaults to 1 and must be above one over the
  number of possible values, so the answer stays the top value; the
  diagnostic result (`sema.withConfidence`, a fallback's diagnostic) carries
  that confidence on the answer and the rest spread evenly over the others.
- Values and confidences are checked against the bundle when the stub is
  created; a call to a function without an answer, or an input no `byInput`
  case matches without `otherwise`, throws `SemaStubError` (`unanswered`,
  `unmatched-input`). Messages name the function by its source position and
  id, for example
  `the semantic function at src/refunds.sem.ts:143:10 (nf_f5a0...)`.
- The bundle is an object, a path (resolved against the working directory)
  or a file `URL`, such as
  `new URL("../dist/semantscript.ir.v1.json", import.meta.url)`. A function
  not trained yet gets the trainer's default canonical input (v2), so
  `maximumInputBytes` limits behave as they will after training.
- A loaded stub is the process's active artifact, like a trained one: loading
  another stub (or artifact) replaces it, and closing that one leaves no
  artifact active: calls then throw `SemaRuntimeNotLoadedError` even though
  the earlier stub was never closed. Do not nest stubs (for example one per
  `describe` and one per test); load one at a time, and close it when the
  test ends.
- `createSemaStubArtifact(bundle, { answers, directory? })` writes the
  artifact without loading it and returns its `root` and `dispose()`, for
  loading with your own `loadSemaArtifact` call. Without `directory` it
  writes a temporary directory that `dispose()` removes. A given `directory`
  must be missing, empty or written by an earlier stub; anything else, such
  as a trained artifact root like `.semantscript/artifact` or a release a
  trainer published into a directory a crashed stub left behind, is refused with
  `SemaStubError` (`occupied-directory`) and left untouched. `dispose()`
  removes the stub's release and its `current.json` pointer from it. The answers live in the
  process that created the stub: a stub release loaded anywhere else is
  refused with `SemaStubError` (`unregistered`), so it cannot serve in
  production.

## Packaging for deployment

The runtime's native dependencies ship prebuilt: `onnxruntime-node` carries
its binaries for linux (x64, arm64), macOS and Windows inside the package and
`tokenizers` carries one `.node` binding per platform, so a plain
`npm ci --omit=dev` on the target platform (a Dockerfile, a CI runner, a
function bundle) installs the right one with no manual step, and installs
made on one platform do not carry to another (bundle on the platform you
deploy to, or in the image). Ship three things together: the compiled
JavaScript, `node_modules` and the artifact directory; nothing else is read
at runtime. [`examples/express-app/deploy`](../examples/express-app/deploy)
has a multi-stage Dockerfile and a serverless handler; because that example
links the workspace packages with `file:` dependencies and commits no
lockfile, its Dockerfile uses `npm install --omit=dev --install-links` in
place of `npm ci --omit=dev`, and
[`scripts/cold-start.mjs`](../examples/express-app/scripts/cold-start.mjs)
measures the cold start (process start, artifact load, first response) and
resident memory; the numbers are in that example's README.
