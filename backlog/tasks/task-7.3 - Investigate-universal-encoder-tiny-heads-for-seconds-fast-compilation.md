---
id: TASK-7.3
title: 'Investigate: universal encoder + tiny heads for seconds-fast compilation'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - research
milestone: m-3
dependencies:
  - TASK-6.1
parent_task_id: TASK-7
ordinal: 26000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Transcript turn 9 closing question: does every function need separately trained weights, or can a universal semantic encoder be trained once so compilation only fits a tiny head? If yes, compile time drops from minutes to seconds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Experiment compares head-only training on a frozen encoder vs full fine-tuning on the Phase 1 benchmark
- [ ] #2 Accuracy gap and compile-time difference are recorded
- [ ] #3 A recommendation is written as a backlog decision
<!-- AC:END -->
