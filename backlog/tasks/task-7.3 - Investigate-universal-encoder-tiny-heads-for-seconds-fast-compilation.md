---
id: TASK-7.3
title: 'Investigate: universal encoder + tiny heads for seconds-fast compilation'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-24 19:19'
labels:
  - research
milestone: m-3
dependencies:
  - TASK-6.1
references:
  - docs/research/laya-analysis.md
parent_task_id: TASK-7
ordinal: 26000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Transcript turn 9 closing question: does every function need separately trained weights, or can a universal semantic encoder be trained once so compilation only fits a tiny head? If yes, compile time drops from minutes to seconds. Laya provides a concrete warm-start path: a universal option-scoring encoder (options as text spans, scored at [MASK] markers) yields initial logits for a new function with zero training; distill those into a fixed head, then refine on generated data. See docs/research/laya-analysis.md section Adopted, item 9.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Experiment compares head-only training on a frozen encoder vs full fine-tuning on the Phase 1 benchmark
- [x] #2 Accuracy gap and compile-time difference are recorded
- [x] #3 A recommendation is written as a backlog decision
- [x] #4 Experiment measures a Laya-style warm start (universal scorer distilled into a fixed head) against head-only and full fine-tuning on time-to-target-accuracy
- [x] #5 The recommendation is framed strictly as compile-time reduction for per-application artifacts; no shipped runtime depends on a universal model
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Findings: the refund benchmark already gives a zero-training Laya number (0.225 on the final set, 80-case release-attested set available for gating), the pooled v4 corpus replays without teacher calls, and sweep_training.py measures release-attested accuracy per training configuration; the trainer cannot freeze the encoder yet.
1. Trainer: TrainingConfig.freeze_encoder (default False) keeps the encoder in eval mode with no gradients so only the head (and adapter, for applications) trains; unit test.
2. Experiment driver benchmarks/refund/program/run_warm_start_experiment.py over the frozen v4 corpus and the compiled refund function, all on the GPU, measuring wall-clock and release-attested accuracy (80 judge-attested cases, the release gate: zero misses) plus held-out accuracy and ECE for: (A) full fine-tuning with the committed release recipe at epoch budgets 1, 2, 4, 8; (B) head-only on the frozen ModernBERT encoder through the trainer at matching budgets; (B') head-only on cached embeddings (encode the corpus once, fit the head on the embeddings with the trainer's proper loss) reporting the encode time separately from the fit time; (C) Laya-style warm start: score every corpus input and release case with the pinned Laya checkpoint through the same choice question the benchmark uses (zero-training accuracy), distill those probabilities into a fixed head on the cached embeddings (accuracy before any generated label is used), then refine on the generated labels and record the time to the release target against (B'). Time-to-target is the first budget whose release misses are zero.
3. Results under benchmarks/refund/data/results-warm-start-2026-09-24 (results.json, environment, README with the tables, accuracy gap and compile-time difference), a Backlog decision framed strictly as compile-time reduction for per-application artifacts (no shipped runtime depends on a universal model), lint and suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Trainer gained TrainingConfig.freeze_encoder (encoder in eval mode, no gradients; unit test in test_training.py, commit 0e5ab2e). Experiment driver benchmarks/refund/program/run_warm_start_experiment.py ran on the frozen pooled v4 corpus (9,409 rows) against the 80 judge-attested release cases on the RX 9070 XT; results.json and README under data/results-warm-start-2026-09-24. AC1/AC2 evidence: full fine-tuning (release recipe) reaches zero attested misses after 1 epoch in 57.8 s (release 1.000, calibrated 0.987, ECE 0.005; 2/4/8 epochs 106/218/434 s all pass); head-only on the frozen encoder through the trainer never passes (release 0.775/0.775/0.763/0.713 at 1/2/4/8 epochs, 31 to 236 s, held-out plateau ~0.89, violations above 10 percent); head-only on cached embeddings encodes 9,489 rows in 25 s and fits 40 epochs in 7.8 s but ends at 0.888 release accuracy (9 misses, 574 violations): gap 22 points on the attested set, compile time moot. AC4 evidence: Laya typed-decisions zero-training release accuracy 0.225 (matches the benchmark baseline), corpus-label agreement 0.382, scoring 92 s; distilled head 0.225; refined 40 epochs 0.875 (10 misses) against 0.888 from random init; time to target not reached by either. AC3/AC5: decision-9 (accepted) recommends short fine-tuning plus the build cache as the compile-time levers, rejects the universal frozen encoder and the Laya warm start as compile paths, and states that shipped artifacts stay per-application encoders plus heads with no runtime dependence on a universal model. Lint clean; benchmark program and training suites 84 passed. Machine note: the reboot cleared /tmp, so the pinned Laya source was re-cloned to /tmp/semantscript-laya-inspect-20260923 at d120d4b.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Answered the universal-encoder question with measurements: on the frozen refund corpus full fine-tuning clears the release-attested target after one epoch (58 s), while head-only training on a frozen ModernBERT-base plateaus near 0.89 held-out and 0.78 to 0.89 attested accuracy with constraint-violation rates far above the gate, even when the head fit itself takes 8 s on cached embeddings; a Laya-distilled warm start starts at 0.225 and adds nothing over random initialisation. Decision-9 records the recommendation: cut compile time with short fine-tuning and the build cache, not a universal frozen encoder, and nothing shipped depends on a universal model. Added TrainingConfig.freeze_encoder, the experiment driver and the committed results; commits 0e5ab2e and the experiment commit.
<!-- SECTION:FINAL_SUMMARY:END -->
