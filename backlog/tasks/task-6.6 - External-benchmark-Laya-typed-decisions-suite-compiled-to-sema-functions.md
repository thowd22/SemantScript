---
id: TASK-6.6
title: 'External benchmark: Laya typed-decisions suite compiled to sema functions'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-20 19:18'
updated_date: '2026-09-24 21:06'
labels:
  - benchmark
milestone: m-2
dependencies:
  - TASK-6.3
  - TASK-6.4
references:
  - docs/research/laya-analysis.md
parent_task_id: TASK-6
ordinal: 40000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Laya's typed-decisions benchmark (400 cases, 2,000 decisions across agent-trace observability, customer service, invoice processing and security incidents) is a public, multi-question-per-input workload with published numbers for Laya (0.766) and Jev (0.727). Expressing each workflow's questions as sema expressions makes it a direct external test of our multi-head execution plan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the four workflows is written as a .sem.ts file whose sema expressions cover every question in the suite
- [x] #2 Accuracy per workflow and overall is reported alongside Laya's and Jev's published numbers, on the suite's own held-out split
- [x] #3 Latency per case (all questions, one execution plan) is reported and compared to Laya's per-question latency
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Findings: the suite (LocalLLaMA/typed-decisions, pinned c76749ec) has four workflows of 300 train and 100 test cases, each case one JSON state and five typed questions (choice over named options, noul yes/no, score over an ordered rubric with per-level descriptions); gold is a three-sample teacher distribution, so the card's ceiling is 0.735 and its ModernBERT-base specialist row is 0.646 at 349 ms per case; Laya publishes 0.766 (fine-tuned) and Jev 0.727. The release gate's attested criterion tolerates no misses, which teacher-noisy labels cannot promise, so verification uses a small high-agreement attested slice and the benchmark score is computed separately on the test split.
1. benchmarks/typed-decisions: generate_programs.py writes one .sem.ts per workflow from the train split's question schema (one sema expression per question over the case's state JSON as a string input: choice as a string-literal union, noul as boolean, score as BoundedInt over the rubric levels, instructions and criteria descriptions in the behavioral text) and a tsconfig; semantscript build compiles them to bundles whose execution plan puts each workflow's five functions in one stage.
2. run_typed_decisions.py: a dataset teacher replays the train split (cases 0-279 train, 280-299 held as attested verification), trains each workflow's five heads jointly over one shared encoder and adapter (ModernBERT-base, the release recipe at a longer sequence length), verifies, exports the application artifact when the gate passes (otherwise records the gate result and exports the ONNX chain for timing only), scores the test split (accuracy per question, per workflow and overall, plus Brier and score MAE from expected values) and times one fused stage per test case (all five questions, one encoder pass) on CPU through the Node runtime's callStage when the artifact exists (else ONNX Runtime directly) and on the GPU in PyTorch.
3. Results and README under benchmarks/typed-decisions/data/results-2026-09-24 next to Laya's and Jev's published numbers and the card's specialist rows, with the specialist/generalist distinction stated; lint and suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
benchmarks/typed-decisions: generate_programs.py writes the four workflows as .sem.ts programs from the pinned dataset's question schema (20 sema expressions, one per question over the case state; choice as string-literal union, noul as boolean, score as BoundedInt over rubric levels; instructions and option/level descriptions in the behavioral text); semantscript build compiles them into one bundle (AC1). run_typed_decisions.py replays the train split through a dataset teacher, trains each workflow's five heads jointly over one ModernBERT-base encoder and adapter (batch 16, sequence 768, lr 3e-5, proper loss, best of 8 epochs), verifies, scores the test split and times one fused stage per case. AC2 evidence (results.json, README): 0.701 overall over 2,000 decisions; customer_service 0.738, agent_trace_observability 0.730, security_incidents 0.684, invoice_processing 0.652; reported next to Laya 0.766 (fine-tuned specialist), Jev 0.727 and Decider 0.768 (general), the card's ModernBERT-base specialist 0.646 and its ceilings 0.704/0.735, with the specialist/generalist distinction stated. AC3 evidence: per case (all five questions, one encoder pass) GPU 12.5/25.2/28.0/22.5 ms p50 (PyTorch eager; ORT has no GPU provider here) and CPU ONNX 53.1/128.8/171.0/112.5 ms p50, compared with Laya's published 32.8 ms per question (T4) and the card's 349 ms per case specialist. Caveat: the release gate refused 18 of 20 heads (ECE over 0.1 on 28 calibration rows, or a miss among five high-confidence attested cases) because the gold is a soft three-sample teacher average, so no artifact was exported and the Node runtime's callStage path was not timed; the CPU number is the deployed ONNX chain measured in Python. Commits ed050a0, d8441b5 and the results commit.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The Laya typed-decisions suite runs as four SemantScript applications: each workflow's five typed questions are one .sem.ts program (20 sema expressions, one head each) and one execution-plan stage answers all five from one encoder pass. On the suite's test split the applications score 0.701 over 2,000 decisions (0.738 customer service, 0.730 agent trace, 0.684 security, 0.652 invoice), above the dataset card's ModernBERT-base specialist (0.646) with the same encoder and at its scenario-understanding ceiling (0.704), below Laya's fine-tuned large checkpoint (0.766) and the zero-shot generalists Jev (0.727) and Decider (0.768). A five-question case costs 12.5 to 28 ms on the GPU and 53 to 171 ms on CPU ONNX, the cost of a single question, against Laya's 32.8 ms per question. The release gate refused most heads on the suite's soft teacher labels, so latency was measured on the ONNX chain rather than a published artifact; the write-up explains why and points to the Laya-initialised large encoder as the measured next step.
<!-- SECTION:FINAL_SUMMARY:END -->
