---
id: TASK-5.13
title: 'Experiment: local teacher vs Sonnet 5 reference'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-20 02:43'
updated_date: '2026-09-25 05:48'
labels:
  - trainer
  - benchmark
  - research
milestone: m-1
dependencies:
  - TASK-5.12
  - TASK-5.6
  - TASK-5.11
references:
  - backlog/decisions
parent_task_id: TASK-5
ordinal: 33000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
We want to iterate locally without paying per build, but only if the local teacher is good enough. Decision-1 makes Sonnet 5 the reference. This experiment defines 'good enough' with numbers instead of a guess. Uses the refund-decision IR from the Phase 1 benchmark. Note from Laya: their fine-tuned student (0.766) exceeded its teacher's ceiling (0.735), so compare students on the held-out set, not teachers on label agreement alone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The same set of inputs is labeled by both claude-sonnet-5 and Qwen3-14B and label agreement is reported
- [x] #2 A Sonnet-labeled held-out set exists, is excluded from all training, and is recorded as the fixed evaluation target
- [x] #3 Two student encoders are trained, one per teacher dataset, and their accuracy and calibration on the held-out set are compared
- [x] #4 A backlog decision records whether the local teacher is acceptable for iteration and the measured accuracy gap
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inputs: sample real refund inputs from benchmarks/refund/data/uci-pool/candidates.json with every held-out digest (final set and release verification) excluded; a fixed evaluation set of 300 inputs and a training set of 1,500, disjoint, seeded.
2. Labelers, both over the same policy text and canonical input the benchmark baselines use, one JSON-schema decision per input: claude-sonnet-5 through OpenRouter's Anthropic-format route (structured output, about USD 0.0013 per case; the spend is quoted to the user before the run) and Qwen3-14B through the local Ollama native API with thinking off, temperature 0, seed 1. Cached per input so reruns re-spend nothing. Report Sonnet-versus-Qwen agreement, each against the compiled constraints' unique admissible label, and per rubric step (AC1). The Sonnet-labeled evaluation set is frozen with its digest as the fixed target and never enters either training corpus (AC2).
3. Students: two ModernBERT-base classifiers trained with the committed compact recipe on the same 1,500 inputs, one with Sonnet labels and one with Qwen labels (replay teacher through SyntheticDatasetGenerator, train_classifier, temperature fitted by evaluate_training_result on the calibration split), then scored on the Sonnet-labeled evaluation set and, as a diagnostic, on the judge-attested final set: accuracy, 15-bin ECE, Brier (AC3). Driver benchmarks/refund/program/run_local_teacher_experiment.py with offline tests; results under benchmarks/refund/data/results-local-teacher-2026-09-25.
4. Decision record stating whether the local teacher is acceptable for iteration and the measured gap (AC4).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Progress 2026-09-25: driver benchmarks/refund/program/run_local_teacher_experiment.py (stages sample, label-qwen, label-sonnet, train; per-input label caches; Sonnet budget cap and 4-way concurrency; 6 offline tests). Sampled from the UCI pool with 194 held-out digests excluded and 22 duplicates dropped (7,381 usable): 300 evaluation and 1,500 training inputs, seed 1. Qwen3-14B (ollama qwen3:14b, think off, temperature 0, seed 1) labeled all 1,800 in about 6 minutes at p50 167 ms; against the compiled constraints it agrees on 1,357/1,800 (75.4%): step 1 38/38, step 3 54/84, step 4 155/524 (330 of the 524 suspicious-history cases labeled approve), step 5 1,110/1,154. Sonnet 5 via OpenRouter smoke-tested on 5 inputs: USD 0.0029 per label (960 input tokens, 57-111 output tokens, about 4 s each); the full 1,800 would cost about USD 5.2, awaiting the user's go-ahead on that spend before labeling.

Progress 2026-09-25: inputs sampled from uci-pool/candidates.json (7,597 rows; 194 held-out digests excluded, 22 duplicates; 300 evaluation + 1,500 training, seed 1) in results-local-teacher-2026-09-25/inputs.json. Qwen3-14B (Ollama, think off, temperature 0, seed 1) labeled all 1,800 in about 5 minutes at 167 ms p50: label counts approve 1,470 / review 170 / deny 160; agreement with the compiled constraints 1,357/1,800 (0.754): step 1 stale 38/38, step 3 outside window 54/84, step 4 suspicious 155/524 (330 of the misses are review->approve), step 5 approve 1,110/1,154. Sonnet 5 via OpenRouter smoke: 5/5 labels, 960 input tokens and USD 0.0029 per case, about 4 s per call; full labeling (1,800 cases, about USD 5.2, concurrency 4) started after telling the user the cost. Driver run_local_teacher_experiment.py with 6 offline tests.

Completed 2026-09-25. Sonnet labeling: 1,800 labels at USD 4.96 (960 input tokens, 3.98 s p50 per label, four in flight); Sonnet = constraints on 1,500/1,500 and 300/300. Qwen3-14B = constraints 0.746 / 0.793 (steps: stale 1.000, outside window 0.618, suspicious 0.280, approve 0.960). Evaluation set (300 Sonnet-labeled real inputs, disjoint from training, digest in results.json) frozen as evaluation-set.json. Students (compact recipe, 8 epochs, best epoch 4 both, constraints stripped from the IR copy so labels are used as given, temperature fitted on the trainer's held-out split): Sonnet-taught 0.980 / ECE 0.019 / Brier 0.040 on the evaluation set and 0.794 on the final set; Qwen-taught 0.800 / 0.143 / 0.316 and 0.694. Final-set misses are dominated by fraud cases absent from the real-input pool (step 2), which is why the release corpus adds synthetic threshold-dense cases. Decision-12 recorded. Verified: 6 offline tests, ruff, and the committed records.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-09-23 19:05
---
Handoff readiness note: this story remains To Do and dependency-blocked on TASK-5.11. Reuse the canonical refund function/dataset contracts, held-out evaluator, ROCm training pipeline, leakage ledger, Sonnet 5 training-only CLI teacher, environment capture, and publication controls from TASK-5.11 after its final dataset is frozen. Do not confuse the installed Qwen 2.5 1.5B/7B benchmark baselines with this story local teacher: AC #1 still specifically requires Qwen3-14B. No Sonnet or Qwen3-14B experiment calls, datasets, student runs, results, or decision record exist yet. Start only after TASK-5.11 is complete, then use one fixed Sonnet-labeled evaluation target excluded from both student training corpora.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Measured the local teacher against the Sonnet 5 reference on 1,800 real refund inputs: Sonnet's labels match the compiled constraints on every input; Qwen3-14B agrees on 0.746, failing the two compound conditionals. Two students trained on the same inputs score 0.980 (ECE 0.019) with Sonnet's labels and 0.800 (ECE 0.143) with Qwen's on the frozen Sonnet-labeled evaluation set. Decision-12: Qwen3-14B is not an acceptable labeling teacher; local iteration uses constraint labels, and Sonnet stays the reference for generation. Verified with the driver's offline tests, ruff and the committed results record (USD 4.96 of the key spent on labels).
<!-- SECTION:FINAL_SUMMARY:END -->
