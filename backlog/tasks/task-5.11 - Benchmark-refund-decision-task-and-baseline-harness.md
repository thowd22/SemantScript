---
id: TASK-5.11
title: 'Benchmark: refund-decision task and baseline harness'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - benchmark
milestone: m-1
dependencies:
  - TASK-5.3
  - TASK-5.8
  - TASK-5.10
parent_task_id: TASK-5
ordinal: 16000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 1 exit criterion. The refund-decision expression from the transcript is the canonical task. Compare our ~200M model against 1B and 7B generative models using structured output and against a traditional LLM structured-output API. Metrics from transcript turn 9 point 10.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A held-out labeled test set for refund decision exists and is not used in training
- [ ] #2 Harness reports accuracy, calibration error, p50/p95 latency, throughput and memory for our model and each baseline
- [ ] #3 Results are committed under benchmarks/ with the exact model versions used
- [ ] #4 A written go/no-go against the exit criterion (p50 < 10ms, accuracy >= 7B baseline) is recorded
<!-- AC:END -->
