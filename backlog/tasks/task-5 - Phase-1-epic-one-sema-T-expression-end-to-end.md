---
id: TASK-5
title: 'Phase 1 epic: one sema<T> expression end to end'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-23 19:05'
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

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-09-23 19:05
---
Phase 1 handoff checkpoint: implementation through TASK-5.10 and TASK-5.12 is complete. TASK-5.11 has a fully implemented, independently audited, memory-bounded harness and verified AMD ROCm/Ollama/Laya smoke paths, but remains In Progress pending authorized training-data generation, two disjoint attested human sets, the Anthropic API baseline, committed final results, and the go/no-go. TASK-5.13 remains dependency-blocked and TASK-5.14 remains unstarted. The Phase 1 acceptance criteria and epic status must remain unchanged until the real TASK-5.11 result proves both p50 under 10 ms and accuracy at least the 7B baseline.
---
<!-- COMMENTS:END -->
