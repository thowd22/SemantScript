---
id: TASK-14.8
title: Stub artifact for testing handlers without training
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-25 15:21'
updated_date: '2026-09-26 02:13'
labels:
  - dx
  - debug
milestone: m-5
dependencies: []
parent_task_id: TASK-14
ordinal: 61000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Application code that calls sema expressions cannot be unit-tested without a trained artifact: the runtime refuses calls before a load, and the only stand-in is the runtime's internal test fixture under runtime/test/fixtures, which the reference application's tests reach into by relative path and re-key by hand. A developer writing a Jest or node:test suite for a controller has no supported way to say what each expression should answer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 @semantscript/core/testing exports a way to create and load a stub artifact from the project's IR bundle in which each function answers a fixed value, a value per input, or a function of its inputs, with a confidence, without any model files
- [ ] #2 The framework's request scopes, pass counts, confidence policy and fallbacks behave with the stub exactly as with a real artifact, and the reference application's tests use it instead of the internal fixture
- [ ] #3 The framework guide documents testing a handler with the stub in under twenty lines
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design: the stub is a real on-disk application artifact (valid v1 manifest, generated tokenizer JSON and tiny generated ONNX graphs) loaded by the ordinary loadSemaArtifact path, so the ONNX worker, canonical input serialization, request-scope sharing and pass counts are the real ones. Only the value is replaced: after the worker answers, the main thread substitutes the stubbed answer (built as the same worker result shape and parsed with parseInferenceResultPayload against the function's head plan) before applyConfidencePolicy, so thresholds, SemaConfidenceError, fallbacks and diagnostics run unchanged. Answers are registered in-process under the stub manifest's sha256 (a per-creation nonce keeps two stubs of one bundle distinct); a stub release loaded with no registered answers is refused with a typed error so a stub can never serve in production.

1. runtime/src/stub-onnx.ts (internal): minimal deterministic ONNX protobuf writer for the three graphs the fixtures use (encoder Cast/Mul/ReduceSum [BATCH,SEQUENCE] -> [BATCH,1]; adapter Identity [BATCH,1]; head Gemm [BATCH,1] -> [BATCH,K], K = support size, 1 for boolean binary-sigmoid), opset 17, ir_version 8, plus the WordLevel tokenizer JSON. Test that the generated graphs pass inspectOnnxContainer and load in onnxruntime (and, if byte-identical is achievable, equal runtime/test/fixtures/*.onnx).
2. runtime/src/stub-registry.ts (internal): Map manifestSha256 -> answerer (functionId, inputs) => inference result; register/unregister/lookup, plus the stub marker check.
3. runtime/src/artifact-runtime.ts: ActiveArtifact gains optional answer; activateArtifact looks it up by staged.manifestSha256 (refuses a stub-marked manifest with none); dispatchArtifactCall and dispatchArtifactStage keep calling runtime.call/callStage (passes, scopes, errors) and then use the stubbed result when present. No change to public index exports.
4. Extract normalizedEntropy/expectedValue/compensatedSum from inference-worker.ts into a shared pure module (e.g. runtime/src/head-math.ts) used by the worker and the stub, so stub diagnostics are computed exactly as the worker computes them.
5. runtime/src/testing.ts (public subpath @semantscript/core/testing; runtime/package.json exports "./testing" with types): 
   - createSemaStubArtifact(bundle | bundlePath, options) -> { root, manifestSha256 }: derives function ids, inputs (inputSchemaSha256), outputSchemaSha256, heads (IR headSpec -> manifest head type: boolean/nominal-string/nominal-number/ordinal-string/ordinal-number, parameterization), adapter and depth-routed encoder refs (application encoder = full if any, else deepest, as the trainer does), runtime policy (resultMode, confidenceThreshold, policy, fallbackRef) from the IR; canonical input from IR trainingProvenance or v1; provenance teacher/baseModel "semantscript-stub"; writes release + current.json atomically like the fixture.
   - loadSemaStubArtifact(bundle, options) -> SemaArtifactHandle: creates in a fresh temp dir unless options.directory, calls loadSemaArtifact(root, options.load) (fallbacks etc. pass through), close() unregisters and removes the temp dir.
   - Answers keyed by function id or by the compiled sema function itself (resolved from its __sema.call("nf_...") literal, must be in the bundle). Forms: a bare scalar or { value, confidence? } (fixed); { byInput: [{ inputs, value, confidence? }], otherwise? } matched by canonical-input equality; { compute: (inputs) => ({ value, confidence? }) }. confidence defaults to 1; a number applies to every field of an object output, or a per-field record. Probability mass: answer gets confidence, the rest split evenly; confidence must exceed 1/K so the answer is the argmax.
   - SemaStubError (code SEMA_STUB_INVALID, reason: unknown-function | unresolved-function | invalid-value | invalid-confidence | unanswered | unmatched-input | unregistered), exported from ./testing. Unstubbed functions throw "unanswered" at call time (closed contract), values validated against head support.
6. Tests beside the feature: runtime/test/testing.test.mjs over a synthetic IR bundle (scalar string, boolean, ordinal number, flat object, a thresholded function with fallbackRef, two adapters and a depth-routed encoder): fixed/per-input/compute answers; diagnostic shape via sema.withConfidence-style diagnostic mode; below-threshold -> SemaConfidenceError without fallback and fallback invoked with it; withSemaScope pass counts equal to the same calls over createFixtureArtifact with the same routing; executeSemaPlan stages; invalid answers and unregistered stub load refused; temp dir removed on close; package.test.mjs imports ../dist/testing.js and the exports map.
7. framework/test: one test driving a controller through handle() over loadSemaStubArtifact and asserting x-sema-passes and a fallback path (keeps existing fixture tests).
8. examples/refund-service/test/app.test.mjs: replace the relative runtime/test/fixtures import and transformManifest re-keying with @semantscript/core/testing; stub decideRefund/refundRisk/refundMethod by compiled function; cover rollback (review/medium), an approve path that writes a row, and a per-input answer; assert x-sema-passes. Build and run it locally (npm install/build/test inside the example, as CI fresh-install does).
9. Docs: docs/framework-guide.md "Testing handlers" rewritten with a node:test example under twenty lines; runtime/README.md testing section; docs/index.md entry; docs/reference-application.md line on the fixture; docs/diagnostics.md SemaStubError; docs/components.md runtime tests line if needed. npx prettier --check docs README.md.
10. Verify: npm run build, npm run lint:node, npm run test:node, refund-service npm test, prettier; record AC evidence; commit and push task-14.8, watch CI.

Risks: the stub manifest must say verification status "passed" (the loader accepts nothing else), mitigated by stub provenance and the unregistered-stub refusal; resolving ids from a compiled function's source text depends on the compiler emitting a literal __sema.call id (fails with a typed error otherwise, and ids are always accepted); a hand-written ONNX writer must match onnxruntime's parser (covered by load tests); examples/express-app still uses the internal fixture (its Docker fixture-artifact script needs a cross-process artifact the in-process stub cannot provide) - report as follow-up rather than change here.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented runtime side: head-math.ts (entropy/expected value shared by worker and stub), stub-onnx.ts (deterministic ONNX writer; reproduces runtime/test/fixtures encoder/adapter/head byte for byte), stub-registry.ts (in-process answers keyed by manifest sha256, SemaStubError, unregistered stub releases refused), artifact-runtime.ts substitutes the stub answer after the real worker call and before applyConfidencePolicy, testing.ts (createSemaStubArtifact, loadSemaStubArtifact, semaFunctionId) exported as @semantscript/core/testing. runtime/test/testing.test.mjs: 8 tests pass, including pass counts equal to createFixtureArtifact with the same routing ({3,3,4} scope, stage and plan).

Framework: framework/test/framework.test.mjs adds a handle()/Next test over loadSemaStubArtifact (passes 1/1/2, x-sema-passes header, fallback on a 0.5 answer below a 0.8 threshold, post-guard sees a stubbed deny). Reference app: examples/refund-service/test/app.test.mjs now uses @semantscript/core/testing keyed by the compiled decideRefund/refundRisk/refundMethod (per-input review for o1 rolls back with x-sema-passes 1/1/2; o3 approves and writes its row with 2/2/3; the no-order path 0/0/0); the relative runtime/test/fixtures import and transformManifest re-keying are gone. Docs: framework-guide Testing handlers (17-line example, run for real inside the example), runtime README testing section, docs/index.md, diagnostics (SemaStubError), components, reference-application, refund-service README. Checks: npm run build ok; lint:node clean; test:node 89/114/22/9/85 pass; refund-service npm test 2 pass 1 skipped (trained); prettier --check clean.

Commit 4ab6e0f pushed to task-14.8; CI run 36209813510 green on all six jobs (Node lint/build/tests, fresh clone + refund-service example tests, Python x2, doctor on Windows and macOS). Follow-up (not in scope): examples/express-app tests and its Docker fixture-artifact script still use runtime/test/fixtures; the Docker fixture needs a cross-process artifact, which the in-process stub deliberately refuses.

Fix round 1 (review findings). Blocking: (1) a given directory is now refused with SemaStubError reason occupied-directory unless it is missing, empty or marked by an earlier stub (.semantscript-stub marker), so a trained root such as .semantscript/artifact is never touched (probe: fake trained current.json unchanged after the refused create); dispose() on a given directory removes the stub's release, its current.json while it still names that release, and the directory once only stub files remain. (2) Stub errors name the function by source position and id, e.g. 'the stub artifact has no answer for the semantic function at src/refunds.sem.ts:143:10 (nf_f5a0...)' (checked through handle() in the refund-service guide example). Advisory fixes: a bundle without a declared canonical input uses v2 (the trainer default), so maximumInputBytes behaves as after training (decideRefund o1 under a 200-byte limit now answers); bundle source may be a file URL and the guide example resolves it with new URL(..., import.meta.url) (still 17 lines, runs green in the example); missing options, misspelled answer forms and non-compiled keys give typed, accurate SemaStubErrors; diagnostics row lists causes and fixes for invalid-bundle and occupied-directory; runtime README documents the directory rules and that a loaded stub is the process-wide active artifact; framework README points to the stub and the guide. Not changed: SemaStubError stays exported from ./testing only; no CommonJS require condition. Checks: npm run build ok; lint:node clean; test:node 89/116/22/9/85 pass 0 fail; runtime/test/testing.test.mjs 10/10; refund-service npm test 2 pass 1 skipped; prettier --check clean.

Fix round 1 commit 4b66c9a pushed; CI run 36210758944 green on all six jobs.
<!-- SECTION:NOTES:END -->
