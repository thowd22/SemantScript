---
id: TASK-5.10
title: 'Runtime: withConfidence result and threshold enforcement'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-22 18:11'
labels:
  - runtime
milestone: m-1
dependencies:
  - TASK-5.9
references:
  - docs/research/laya-analysis.md
documentation:
  - SPEC.md
  - IR.md
  - runtime/README.md
modified_files:
  - IR.md
  - runtime/README.md
  - runtime/src/artifact-loader.ts
  - runtime/src/artifact-runtime.ts
  - runtime/src/confidence-policy.ts
  - runtime/src/index.ts
  - runtime/src/inference-protocol.ts
  - runtime/src/inference-runtime.ts
  - runtime/src/inference-worker.ts
  - runtime/src/strict-json.ts
  - runtime/test/artifact-loader.test.mjs
  - runtime/test/inference-runtime.test.mjs
  - runtime/test/runtime.integration.test.mjs
  - runtime/test/strict-json.test.mjs
parent_task_id: TASK-5
ordinal: 15000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Neural values are probabilistic. Confidence must be accessible without contaminating normal syntax, and @confidence must refuse to return an ordinary value below threshold. Transcript turn 10.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 fn.withConfidence(inputs) returns { value, confidence } using calibrated probabilities
- [x] #2 A function compiled with @confidence(x) throws or invokes the configured fallback when confidence < x
- [x] #3 Plain calls remain unchanged in shape
- [x] #4 confidence is the calibrated top-1 probability and uncertainty is normalized entropy, matching SPEC.md
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconcile SPEC/IR diagnostic result semantics, calibrated head probabilities, entropy, threshold, and fallback contracts with the existing artifact/runtime ABI. 2. Extend worker inference results to carry exact value plus calibrated top-1 confidence, normalized entropy, distribution, and ordinal expected value for scalar and flat outputs while preserving plain call shapes. 3. Enforce configured thresholds for plain calls with typed low-confidence failure or validated fallback routing, while diagnostic calls return below-threshold results for caller decisions. 4. Add deterministic scalar/flat, threshold/fallback, calibration, entropy, exact-number, activation, and package coverage; update runtime/IR documentation. 5. Run focused independent audits followed by the bounded 512 MiB repository gate, record evidence, check all acceptance criteria, and finalize the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented exact SPEC diagnostic results for scalar and flat outputs using temperature-calibrated sigmoid/softmax probabilities, stable support-order ties, normalized entropy, ordered distributions, ordinal expected values, and conservative flat aggregates. Plain unthresholded calls retain their original T shape; diagnostic calls always return diagnostics; thresholded plain calls use inclusive confidence checks and either return T, invoke a snapshotted synchronous fallback, or throw SemaConfidenceError. Artifact loading enforces the threshold/policy/output relationship matrix and resolves fallback registrations before worker startup and atomic activation. Fallback outputs are checked exactly against manifest support with Object.is numeric semantics, safe flat-object contracts, and invocation-cycle recovery. Worker success responses use bounded strict plan-aware JSON decoding with duplicate-key rejection, exact support/field/expected-value validation, direct shared-buffer parsing, linear support validation/number scanning, nested signed-zero preservation, and conservative 25-byte finite-number sizing. Callback-thrown errors propagate unchanged. Confidence thresholds remain artifact policy metadata and are not duplicated in the public diagnostic shape.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @codex
created: 2026-09-22 02:37
---
TASK-1 resolves superseded named-function syntax as expression tags: sema.withConfidence<T> and configured equivalent, not fn.withConfidence(inputs). Scalar diagnostics are {value, confidence, uncertainty, distribution, expectedValue}; flat diagnostics expose per-field scalar results plus minimumFieldConfidence/maximumFieldUncertainty. Diagnostic tags return results below @confidence for caller fallback; plain tags enforce fallback-or-throw.
---

created: 2026-09-22 17:19
---
TASK-5.10 started as the next ready task in ordinal order immediately after TASK-5.9 completed. Work begins with parallel audits of the existing SPEC/IR contract, runtime probability path, and fallback/threshold ABI so implementation preserves plain call shapes and the already-shipped synchronous worker bridge.
---

created: 2026-09-22 17:29
---
Parallel contract, probability, and lifecycle audits converged. The worker will emit stable temperature-calibrated scalar/flat diagnostics only when resultMode or threshold requires them; plain unthresholded calls retain compact values. Loader validation will enforce the q/policy/output matrix. Load-scoped fallback callbacks will be snapshotted and resolved before worker activation, invoked synchronously only for low-confidence plain calls, and their returns validated exactly against manifest support with signed-zero correctness and recursion protection. Public diagnostic shapes remain exactly those in SPEC.md; the IR sentence saying the result records q is treated as stale wording because q remains artifact policy metadata.
---

created: 2026-09-22 17:54
---
Final bounded protocol audit found three issues before repository-wide verification: categorical selection must apply stable tie-breaking to the calibrated probabilities (extreme temperatures can round distinct logits to a tie); main-thread success decoding must validate against the invoked function plan and reject duplicate-key JSON/support/order/expected-value mismatches; and the proven maximum finite-number JSON length is 25 bytes rather than 24. Focused fixes and regression probes are underway; the full suite remains deferred until this audit closes.
---

created: 2026-09-22 18:10
---
Final verification is green after closing both audit rounds. Exact capped runtime suite passed all 7 test files at 249,212 KiB peak RSS. The full NODE_OPTIONS=--max-old-space-size=512 npm run check passed repository lint, formatting, builds, 14 Node test files, and 2 Python tests at 507,116 KiB peak RSS with no swaps. Package dry-run includes confidence-policy and the worker; git diff check passes. Independent lifecycle and final protocol audits report no remaining blockers, including exact-threshold pass behavior and 50,000-support/number linearity probes.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The Node runtime now exposes SPEC-conforming confidence diagnostics and enforces @confidence policies without changing ordinary call shapes. Calibrated top-1 confidence, normalized entropy, complete distributions, ordinal expected values, flat aggregates, typed low-confidence errors, synchronous registered fallbacks, exact fallback contracts, and recursion protection are covered by worker, loader, real-ONNX, protocol-corruption, reload-safety, and 50,000-element scalability tests. The final runtime suite and full repository check pass under a 512 MiB Node cap; full-suite peak RSS was 507,116 KiB with no swaps.
<!-- SECTION:FINAL_SUMMARY:END -->
