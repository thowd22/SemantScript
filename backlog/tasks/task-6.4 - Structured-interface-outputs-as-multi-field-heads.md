---
id: TASK-6.4
title: Structured interface outputs as multi-field heads
status: Done
assignee:
  - '@claude'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-24 16:50'
labels:
  - compiler
  - model
  - runtime
milestone: m-2
dependencies:
  - TASK-6.1
parent_task_id: TASK-6
ordinal: 21000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Flat interface outputs (e.g. ClaimAssessment with category, severity, fraudRisk, requiresHumanReview) should compile to one head per field, not to a serialized object. Transcript turn 10.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A flat interface output type compiles to one head per field in IR
- [x] #2 Runtime assembles the typed object from head outputs with no JSON step
- [x] #3 Per-field accuracy is reported by the verifier
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Findings: the compiler already emits object outputs as one head per field (ObjectOutputSpec with fields[{name, head}], model.heads with JSON-pointer output paths) and the runtime already assembles flat objects from per-field heads with a minimum-field-confidence policy; the gap is the trainer, which is scalar-only (single label per row, one head per corpus, verification and export refuse non-scalar outputs).
1. Contracts: OutputHead(output_path, contract) and derive_output_heads(ir) (scalar: one head with the empty path; object: one per field in IR order, reusing the scalar head derivation per field). TrainingRow gains label_indices (one per head; label_index stays the first for compatibility), TrainingCorpus gains heads (head stays the first), corpus assembly labels every head from the object output, and the calibration split digest adds labelIndices only when there is more than one head so scalar digests are unchanged.
2. Model and training: a FieldHeads module (one ClassificationHead per field, forward returns the concatenated logits with recorded slices) used wherever a function has several heads; the training loop sums per-head losses over the slices, reports exact-match held-out accuracy plus per-field accuracy in EpochMetrics, and the application trainer uses the same head builder and widths.
3. Verification: predictions per head from the logit slices, temperature calibration and CalibrationRecordV1 per head, HeadVerificationV1 per output path, function-level accuracy as exact match over all heads, function-level ECE and Brier as the maximum over heads (the gate stays conservative), pair consistency per head with the function-level minimum; example failures and attested misses count any wrong field; constraint checks evaluate the assembled object output; manifest head metadata per head.
4. Artifact: model contract check per head, runtime head type per field head spec, manifest heads with outputPath [] or [field] and per-head calibration and verification, one head resource per (function, head) named head-<index>.onnx with the IR head ref, validator accepts N unique single-segment paths, ONNX export of every head with chained parity through export_application_components; quantization stays scalar and says so.
5. Tests: contract derivation and row labeling for an object IR; offline training on a two-field fixture (nominal string field plus boolean field) reporting per-field accuracy; offline verification producing two HeadVerificationV1 entries with per-field accuracy and the conservative gate; artifact export of the object function with a Node round trip returning the typed object; a compiler analysis test asserting a flat interface output compiles to one head per field if not already covered; lint and full suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented slices 1-5 of the plan. Contract: OutputHead, derive_output_heads, output_labels, TrainingRow.label_indices, TrainingCorpus.heads. Model: FieldHeads (per-field ClassificationHead, concatenated logits with slices). Training: per-head summed loss, exact-match plus held_out_field_accuracy, train_corpus and application_function derive every head, batch guard uses the total logit count. Verification: per-head temperature/ECE/Brier/accuracy/pair consistency as HeadVerificationV1 with the field JSON pointer; function accuracy is exact match, ECE and Brier the worst head, pair consistency the lowest head. Artifact: one head-<index>.onnx per head, manifest outputPath [field], per-head calibration and verification, all-fields policy when thresholded, validator accepts N distinct single-segment paths; quantization stays single-function scalar and says so. Docs: trainer/README.md. Verification evidence: AC1 compiler/test/analysis.test.mjs asserts a three-field interface compiles to heads [/, /accepted, /score] with one ref each (npm test -w compiler: 68 pass). AC2 trainer/tests/test_object_outputs.py exports a two-field function and a Node script calls it through runtime/dist, receiving a plain {label, flag} object with two head passes in one stage (no JSON step; runtime assembles from head outputs). AC3 the same module verifies the function offline and asserts two HeadVerificationV1 entries with per-field accuracy, EpochMetrics.held_out_field_accuracy == (1.0, 1.0). Suites: pytest trainer/tests model/tests 426 passed, 3 skipped; npm test -w runtime 99 pass; npm run lint clean. Commit: trainer multi-head commit on master.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @codex
created: 2026-09-22 02:37
---
TASK-1 flat-output contract: one independent scalar head per required field; diagnostic results expose fields[k] plus conservative minimumFieldConfidence and maximumFieldUncertainty. @confidence(q) passes only when every field confidence is >= q; no joint distribution is implied.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Flat interface outputs now flow through the whole trainer as one head per field: derive_output_heads/OutputHead/output_labels label every head per row, FieldHeads trains per-field heads over one encoder with summed losses and per-field held-out accuracy, verification calibrates and gates every head (HeadVerificationV1 per field JSON pointer; function ECE/Brier are the worst head's, pair consistency the lowest), and the artifact exporter publishes head-<index>.onnx per head with manifest outputPath [field], per-head metadata and the all-fields policy when thresholded. Compiler and runtime already did their halves; a compiler test now asserts one head per field and a trainer test exports a two-field function that Node calls back as a plain typed object. Verified with pytest trainer/tests model/tests (426 passed), npm test -w compiler (68), npm test -w runtime (99) and npm run lint; commit ba90d25.
<!-- SECTION:FINAL_SUMMARY:END -->
