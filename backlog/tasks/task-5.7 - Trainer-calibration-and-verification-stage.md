---
id: TASK-5.7
title: 'Trainer: calibration and verification stage'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-23 14:30'
labels:
  - trainer
milestone: m-1
dependencies:
  - TASK-5.5
  - TASK-5.6
references:
  - docs/research/laya-analysis.md
modified_files:
  - model/src/semantscript_model/calibration.py
  - model/src/semantscript_model/__init__.py
  - model/tests/test_calibration.py
  - model/tests/test_public_api.py
  - model/README.md
  - trainer/src/semantscript_trainer/verification.py
  - trainer/src/semantscript_trainer/constraints.py
  - trainer/src/semantscript_trainer/__init__.py
  - trainer/tests/test_verification.py
  - trainer/tests/test_public_api.py
  - trainer/README.md
parent_task_id: TASK-5
ordinal: 12000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Confidence must mean something for @confidence thresholds to work, and the artifact must be proven against examples and constraints before it ships. Transcript turn 10 artifact block lists verification fields.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Post-training calibration (e.g. temperature scaling) is applied and expected calibration error is reported
- [x] #2 All gold examples are checked and any miss fails the build
- [x] #3 All constraints are checked on adversarial cases and any violation fails the build
- [x] #4 Verification stats are written into the artifact manifest
- [x] #5 A per-function temperature is fitted on a calibration split held out from training and stored in the artifact
- [x] #6 ECE and Brier score are reported per function and the build fails above a configured ECE threshold
- [x] #7 Pair-consistency accuracy on counterfactual twins is reported per function
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement optional-PyTorch-safe calibration primitives for binary one-logit and categorical logits: bounded positive temperature fitting by held-out NLL, calibrated probabilities, top-1 accuracy, equal-width ECE, and normalized multiclass Brier.
2. Define a schema-aligned scalar verification result with deterministic calibration-split identity and exact manifest projections for per-head calibration/verification plus function-level passing metrics.
3. Reconstruct and identity-check the trained corpus and its deterministic group split from the base dataset and optional adversarial sidecar, collect bounded eval-mode logits from exact canonical-input/v1 text, and use only the training-held-out partition to fit temperature and compute held-out metrics.
4. Recheck every IR gold and external human example after training, evaluate every deterministic constraint against every observed prediction with shared work budgets, and compute counterfactual pair consistency as the fraction of linked pairs whose two members are both correct.
5. Fail the build with a typed gate report on any measured example miss, constraint violation, missing human-authored evidence, or ECE above the configured threshold; abort non-measurable classifier ABI failures with a typed execution error; expose only passing manifest records while retaining measured failed IR diagnostics.
6. Add offline injected model/tokenizer tests, schema/manifest compatibility vectors, bounds and tamper cases, public APIs and documentation; then run focused and memory-bounded full gates plus independent acceptance, semantics/resource, and security audits before finalization.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented deterministic bounded temperature scaling for binary and categorical logits with held-out NLL fitting, equal-width ECE, normalized Brier, and stable support-order ties. Verification reconstructs the exact group-aware split, reruns all gold/external human cases, evaluates every constraint over all observed cases with shared budgets, and reports both-members-correct counterfactual pair consistency; zero pairs use the documented vacuous value 1.0 with pair_count=0. Passing results expose schema-exact IR and manifest projections for TASK-5.8; failed results cannot project. Classifier ABI failures abort with VerificationExecutionError, while completed scalar records have typeErrors=0 because decoded predictions are support members. Resource controls cover canonical bytes, token products, logit products, calibration rows/logits, temperature iterations, and constraint work.

Validation: bounded npm run check passed all 16 Node test files and 317 Python tests with 4 expected skips; peak RSS 771592 KiB, zero swaps. Focused acceptance/schema, semantics/resource, and security audits found no remaining blocker.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-09-23 02:59
---
Started in ordinal order after TASK-5.6 completed. Beginning contract and architecture reconnaissance for temperature fitting, ECE/Brier reporting, gold-example and constraint release gates, counterfactual pair consistency, and verification-manifest persistence; parallel agents are auditing the mathematical, IR/manifest, and adversarial-data requirements before implementation.
---

created: 2026-09-23 03:05
---
Reconnaissance plan recorded. TASK-5.7 will hand TASK-5.8 schema-exact manifest fragments rather than writing ONNX or release directories early; failed verification remains diagnostic IR evidence and cannot produce a deployable manifest record.
---

author: @codex
created: 2026-09-23 03:41
---
Implementation complete pending final audit: added deterministic held-out temperature fitting with ECE/Brier, exact verified-IR and manifest projections, gold/human and all-case constraint gates, counterfactual pair consistency, public APIs/docs, and offline integration tests. Security follow-up now also recomputes the split, shares constraint budgets, caps canonical bytes/tokens/logits/split hashing, validates typed evidence consistency, preserves dependency-free imports, and normalizes injected backend failures. Bounded full repository gate passed with 314 Python tests plus all 16 Node test files, 771996 KiB peak RSS, and zero swaps.
---

author: @codex
created: 2026-09-23 03:45
---
Finalization: all seven acceptance criteria checked from automated evidence; no Definition of Done checklist items were configured. Final bounded npm run check passed after all hardening changes (317 Python passed, 4 expected skips; all 16 Node test files passed; 771592 KiB maximum RSS; zero swaps). Acceptance/schema, semantics/resource, and security agents each reported no remaining blocker.
---

author: @codex
created: 2026-09-23 14:30
---
TASK-5.8 follow-up strengthened VerificationResult with deterministic model-state and exact tokenizer-byte SHA-256 bindings. Verification fingerprints both before inference and rechecks afterward; artifact export rejects stale weights or tokenizer bytes before publication. Covered by the final 357-test bounded repository gate.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added bounded post-training temperature calibration and release verification with gold/human, constraint, ECE, and counterfactual-pair gates; emitted schema-exact IR and manifest-ready evidence; preserved dependency-free imports; and verified the result with the full bounded repository gate (317 Python tests, all 16 Node test files, zero swaps) plus three clean independent audits.
<!-- SECTION:FINAL_SUMMARY:END -->
