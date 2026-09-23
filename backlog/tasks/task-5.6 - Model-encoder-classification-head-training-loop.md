---
id: TASK-5.6
title: 'Model: encoder + classification head training loop'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-23 03:09'
labels:
  - model
milestone: m-1
dependencies:
  - TASK-5.4
references:
  - docs/research/laya-analysis.md
modified_files:
  - pyproject.toml
  - model/README.md
  - model/src/semantscript_model/__init__.py
  - model/src/semantscript_model/classifier.py
  - model/src/semantscript_model/encoder.py
  - model/src/semantscript_model/heads.py
  - model/src/semantscript_model/losses.py
  - model/tests/test_losses.py
  - model/tests/test_primitives.py
  - model/tests/test_public_api.py
  - model/tests/test_smoke.py
  - trainer/README.md
  - trainer/src/semantscript_trainer/__init__.py
  - trainer/src/semantscript_trainer/canonical_input.py
  - trainer/src/semantscript_trainer/training.py
  - trainer/src/semantscript_trainer/training_contract.py
  - trainer/tests/test_canonical_input.py
  - trainer/tests/test_public_api.py
  - trainer/tests/test_smoke.py
  - trainer/tests/test_training.py
  - trainer/tests/test_training_contract.py
  - trainer/tests/test_training_live.py
parent_task_id: TASK-5
ordinal: 11000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Non-autoregressive by design: pretrained 100-300M encoder, one head whose logit count equals the output type cardinality, one forward pass. Transcript turn 9 points 3 and 10.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fine-tunes a pretrained encoder plus a categorical head from the generated dataset
- [x] #2 Head shape is derived from the IR output spec, not hard-coded
- [x] #3 Training run reports accuracy on a held-out split
- [x] #4 Base encoder choice is recorded in the decision log
- [x] #5 Heads are trained with a strictly proper scoring rule loss (log + spherical score), with ranked probability score added for ordinal heads; plain cross-entropy is available as a baseline flag
- [x] #6 Head architecture is a linear or small MLP head per function; a heavier head is only added if the benchmark demands it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the exact Python canonical-input/v1 serializer and stable IR scalar-support/label mapping so training consumes the same bytes and support order as runtime, including signed-zero and decimal supports.
2. Add model primitives for an injectable pretrained sentence encoder, identity adapter seam, and per-function linear or bounded small-MLP head, deriving one-logit binary-sigmoid or K-logit categorical-softmax shape from the IR ABI.
3. Implement numerically stable supervised objectives: log plus spherical score by default, normalized ranked probability score for ordinal heads, and an explicit plain cross-entropy baseline.
4. Implement deterministic group-aware train/held-out splitting and a bounded training loop over base plus adversarial rows, reporting held-out accuracy while keeping counterfactual pairs together and recording dataset/model/config provenance.
5. Keep routine tests fully offline through injected tiny tokenizers/encoders; add opt-in cached-model and ROCm smoke coverage, explicit resource bounds, and training-stack dependency/documentation guidance.
6. Record/confirm the accepted ModernBERT-base choice and immutable revision policy, run focused CPU/GPU tests plus the memory-bounded repository suite, complete independent acceptance/security audits, then check all criteria and finalize.
<!-- SECTION:PLAN:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-09-22 23:36
---
Started immediately after TASK-5.5 completion as the next ready ordinal item. Beginning architecture, dependency, test, and AMD/ROCm training-path reconnaissance before recording the implementation plan; parallel agents are auditing the model skeleton, scoring-rule requirements, and existing research/decision constraints.
---

created: 2026-09-22 23:37
---
Returned to To Do before implementation because the preceding TASK-5.5 final audit found a real parity blocker that must be closed first. Reconnaissance findings are preserved; work will resume in ordinal order after TASK-5.5 is re-finalized.
---

created: 2026-09-22 23:49
---
Resumed in ordinal order after TASK-5.5 was re-finalized. Reconnaissance is complete: model/ is currently a stub; training must consume canonical-input/v1 strings, use deterministic group-aware holdout splitting, derive support/head shape from scalar IR, keep routine tests offline through injected tiny components, and treat PyTorch/Transformers plus ROCm enablement as an explicit optional training stack.
---

created: 2026-09-22 23:51
---
Reconnaissance plan recorded. The accepted decision log already selects ModernBERT-base with DeBERTa-v3-base fallback and Python/PyTorch. Implementation will preserve the binary one-logit runtime ABI while treating it as a two-outcome probability distribution for proper scoring.
---

created: 2026-09-23 02:59
---
Implementation and independent audits complete. The trainer now maps scalar IR supports into binary one-logit or categorical K-logit heads, serializes every row with canonical-input/v1, holds linked adversarial groups together, rejects canonical leakage across the split, fine-tunes the encoder and bounded linear/small-MLP head, trains with proper or explicit cross-entropy loss, and reports per-epoch held-out accuracy plus provenance. The pinned ModernBERT-base revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 completed a real one-step ROCm run on the RX 9070 XT in 19.6s at 2,053,560 KiB peak host RSS and zero swaps. An initial live attempt stopped before model allocation because an unrelated user-site SciPy conflicted with AMD's NumPy 1.26.4; PYTHONNOUSERSITE=1 isolated the validated stack and the rerun passed. The memory-bounded full repository gate then passed all 16 Node test files and 295 Python tests (4 expected opt-in/dependency skips) at 769,968 KiB peak RSS and zero swaps.
---

created: 2026-09-23 03:09
---
TASK-5.7 integration found and corrected one cross-stage contract issue: SPEC 4.1 requires every gold example to participate in training and later verification, so the deterministic group split now keeps all gold groups in the training partition and chooses calibration rows only from non-gold groups. Binary held-out accuracy also now matches runtime tie semantics: a zero logit selects the earlier false support member. Regression coverage was added; TASK-5.6 acceptance remains satisfied.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Phase 1 pretrained-encoder classification training path. Generated base and adversarial rows are converted through exact canonical-input/v1 serialization into a deterministic group-aware train/held-out split; scalar IR derives stable label indices and either the boolean one-logit ABI or categorical K-logit ABI. The pinned ModernBERT-base encoder and a bounded linear or one-hidden-layer MLP head are jointly optimized with log+spherical scoring, plus RPS for ordinal outputs, while cross-entropy remains an explicit baseline. Training reports held-out accuracy and retains dataset/model/config provenance, with fail-closed resource, tensor, leakage, and finite-value checks. Routine tests are offline and injectable; the real pinned ModernBERT revision passed a one-step ROCm fine-tune on the RX 9070 XT. Final validation: all 16 Node test files and 295 Python tests passed (4 expected opt-in/dependency skips), peak full-gate RSS 769,968 KiB with zero swaps.
<!-- SECTION:FINAL_SUMMARY:END -->
