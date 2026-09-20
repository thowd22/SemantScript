---
id: TASK-5
title: 'Phase 1 epic: one sema<T> expression end to end'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-20 02:47'
labels:
  - epic
milestone: m-1
dependencies:
  - TASK-10
references:
  - PLAN.md
priority: high
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The real bet. Prove that a typed natural-language expression can be compiled into a small, fast neural function whose output is constrained by its type and executes alongside ordinary code. Everything else in the project depends on this. See PLAN.md Phase 1 and transcript turn 9 point 10.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The refund-decision sema expression from the transcript compiles, trains, and executes from a .sem.ts file with no manual steps
- [ ] #2 Benchmark meets exit criterion: p50 < 10ms local inference and accuracy >= 7B generative structured-output baseline on the held-out set
<!-- AC:END -->
