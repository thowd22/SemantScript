---
id: TASK-6.1
title: 'Model: shared encoder + per-application adapter + per-function heads'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-24 16:17'
labels:
  - model
milestone: m-2
dependencies:
  - TASK-5.8
parent_task_id: TASK-6
ordinal: 18000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Avoid one model file per function. One artifact holds a shared encoder, a small app adapter and a tiny head per sema site so a controller with 50 expressions shares nearly all compute.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Multiple functions train into one artifact with separate heads
- [x] #2 Adding a head does not require retraining existing heads
- [x] #3 Per-head accuracy is reported and comparable to single-function training
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Model package: add ApplicationAdapter (residual bottleneck MLP over the sentence embedding, zero-initialized output so it starts as identity, ONNX-exportable, shape-preserving) and SharedEncoderApplication (one SentenceEncoder, one adapter, a ModuleDict of ClassificationHeads keyed by function id) with function_model(function_id) returning a per-function view module (encoder, adapter, head; forward = head(adapter(encoder(...)))) so verification, lifecycle and model_state_sha256 keep working per function unchanged. Generalize the ONNX exporter to export the encoder and adapter once and one head per function with chained parity; the single-function API stays as a wrapper.
2. Trainer: new application module with FunctionCorpus(ir, base, adversarial), ApplicationTrainingResult (shared model plus per-function TrainingResult views with their own splits and metrics), train_application (joint fine-tuning of encoder, adapter and all heads over interleaved per-function batches, per-function held-out accuracy, best epoch on the mean) and add_function_head (encoder and adapter frozen, only the new head trains; existing heads and shared weights are byte-identical afterwards). Factor the encoder and head builders out of the single-function trainer and reuse its row preparation, batching, loss, accuracy and schedule helpers.
3. Artifact: export_multi_function_artifact takes one (ir, training view, verification, source IR bytes) per function sharing tokenizer digest, encoder, adapter and application refs; resources are tokenizer, encoder, adapter and one head per function; manifest lists every function; build.sourceIrSha256 is the digest of the ordered per-function source IR digests when there is more than one function; the Python manifest validator accepts N functions and 3+N resources; the existing single-function export delegates to it. Quantization stays single-function and says so.
4. Runtime already loads multi-function manifests (loader validates each function against the shared encoder and adapter); prove it with an end-to-end test exporting a two-function offline artifact and calling both functions from Node.
5. Experiment for the accuracy criterion on the refund inputs: a companion sema function assessRefundRisk (low, medium, high) with explicit rules as constraints, compiled under the same application id so it shares encoder and adapter refs, labeled from the UCI pool by its own constraints (held-out inputs excluded) with a rule-labeled attested set clearly marked as not human or judge adjudicated; train the companion alone, train both jointly, and add the companion as a head on the frozen refund encoder; report per-head held-out accuracy and the refund release accuracy against the committed single-function numbers, export the two-function artifact and run both functions through the Node runtime; record the comparison in the task and a results write-up.
6. Tests: model (adapter shape/identity init, application views, multi-head export parity), trainer (joint training on the offline fixtures, add-head leaves shared and existing head state byte-identical, per-function metrics), artifact (two-function export, manifest shape, Node load and dispatch of both functions, validator rejections), plus lint and the full suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Slice 1 implemented: model package ApplicationAdapter (residual bottleneck, identity at init), SharedEncoderApplication with per-function FunctionModel views, export_application_components (encoder and adapter once, one head per function, chained parity per head; the single-function exporter delegates). Slice 2: trainer application module with FunctionCorpus, train_application (interleaved per-function batches, per-function held-out accuracy, best epoch on the mean) and add_function_head (shared modules and existing heads frozen; verified byte-identical in tests, model_state_sha256 of the existing view unchanged). Slice 3: export_multi_function_artifact (one tokenizer, encoder, adapter and N heads; manifest lists every function; validator accepts N functions; sourceIrSha256 is the digest of the per-function digests when N > 1); quantization stays single-function with an explicit message. Offline tests for all three slices pass; Node runtime loads a two-function artifact and dispatches both functions.

Experiment run 1 (companion with OR-based high rule, no adversarial sidecar): B alone held-out 0.9986 and 80/80 rule-attested; joint (shared encoder and adapter) A held-out 0.9989 with 80/80 judge-attested and B held-out 1.0000 with 79/80; a-then-b (B head only on A's frozen encoder) left A's state byte-identical but B reached only 0.9373 held-out and 0.8375 attested, so head-only addition is cheap but weaker than joint training. Verification of B failed by design: functions with constraints require an adversarial sidecar. Fixed by giving the constraint-label teacher the adversarial protocol without a language model (single-field edits of real inputs across the schema thresholds, labeled by the constraints: 6 boundary cases and 147 counterfactual pairs in under 3 s) and by making the companion's top rule depend on one field so every anchor has a one-field twin (an OR-based top rule left some anchors unflippable). Run 2 with the sidecar is in progress (commit 4025a86).

Experiment run 2 (with the companion's constraint-based adversarial sidecar): B alone held-out 1.0000, 80/80 rule-attested; joint A held-out 0.9968 with 80/80 judge-attested and B held-out 1.0000 with 80/80; a-then-b A byte-identical (80/80) and B head-only 0.9751 held-out, 69/80. Both functions of the joint application verified (A: 0.9968, ECE 0.0019, 3 violations; B: 1.0000, ECE 0.0000, 0 violations) and exported into one artifact (manifest 9ec23d46..., 5 resources, 2 functions) that the Node runtime dispatches (A approve 0.98, B medium 1.00). Two driver defects fixed on the way: the companion was verified without its sidecar, and the Python runtime bridge validated the top decision against the refund support instead of the declared one; the driver can resume measured regimes and seal a report after an environmental failure. Write-up: benchmarks/refund/data/release-application-2026-09-24/README.md. Suites: Python 485 passed, runtime 96 passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Built the shared-encoder application: ApplicationAdapter and SharedEncoderApplication with per-function views in the model package, train_application and add_function_head in the trainer, export_multi_function_artifact publishing one tokenizer, encoder and adapter with one head per function, and a runtime that loads and dispatches multi-function artifacts. Verified with new model, trainer and artifact tests (485 Python and 96 runtime tests passing, including a Node round trip of a two-function artifact) and by the refund experiment: the decision function and a companion risk function trained jointly over one encoder match their single-function accuracy (A 0.9968 held-out with 80/80 attested, B 1.0000 with 80/80), adding a head leaves the existing function byte-identical while the head-only function reaches 0.975, and both verified functions ship in one artifact (manifest 9ec23d46). The head-only gap is recorded as the motivation for per-domain adapters in TASK-6.7. Commits 6e0e733, d8b9006, 4025a86, 4049add, a3b100e and the data commit.
<!-- SECTION:FINAL_SUMMARY:END -->
