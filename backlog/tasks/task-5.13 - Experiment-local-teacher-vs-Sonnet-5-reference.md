---
id: TASK-5.13
title: 'Experiment: local teacher vs Sonnet 5 reference'
status: To Do
assignee: []
created_date: '2026-09-20 02:43'
updated_date: '2026-09-20 19:18'
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
- [ ] #1 The same set of inputs is labeled by both claude-sonnet-5 and Qwen3-14B and label agreement is reported
- [ ] #2 A Sonnet-labeled held-out set exists, is excluded from all training, and is recorded as the fixed evaluation target
- [ ] #3 Two student encoders are trained, one per teacher dataset, and their accuracy and calibration on the held-out set are compared
- [ ] #4 A backlog decision records whether the local teacher is acceptable for iteration and the measured accuracy gap
<!-- AC:END -->
