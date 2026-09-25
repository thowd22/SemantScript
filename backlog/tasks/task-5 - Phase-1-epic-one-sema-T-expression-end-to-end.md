---
id: TASK-5
title: 'Phase 1 epic: one sema<T> expression end to end'
status: Done
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-25 05:57'
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
- [x] #1 The refund-decision sema expression from the transcript compiles, trains, and executes from a .sem.ts file with no manual steps
- [x] #2 Benchmark meets exit criterion: p50 < 10ms local inference and accuracy >= 7B generative structured-output baseline on the held-out set
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC1: the release pipeline compiles, trains, verifies, exports and calls the canonical refund expression from its .sem.ts file with no manual step (release-depth-006-2026-09-25). AC2: met by results-depth-006-2026-09-25 (p50 4.82 ms on the CPU runtime, accuracy 1.000 against 0.538 for the 7B baseline in the committed run). The epic stays open because TASK-5.11's structured-output API baseline, TASK-5.13 and TASK-5.17 wait on an API key.

Epic closure 2026-09-25: AC1 is the refund-decision expression compiled, trained and executed through the CLI (init/build/train/test/run) and the reference tutorial; AC2 is now met by one machine-checked record with all five required systems (results-final-2026-09-25, mechanical status go: p50 4.82 ms on the CPU at accuracy 1.000 against the 7B comparator's 0.538, and equal to the Sonnet 5 structured-output baseline's 1.000). Every child task is Done: compiler, trainer, model and runtime primitives, teacher backends, the benchmark and its baselines, the encoder sweep (six of eight criteria, the ~1B point open pending a checkpoint choice), the universal-encoder investigation, the Jev comparator, the local-teacher experiment and the Phase 1 docs.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-09-23 19:05
---
Phase 1 handoff checkpoint: implementation through TASK-5.10 and TASK-5.12 is complete. TASK-5.11 has a fully implemented, independently audited, memory-bounded harness and verified AMD ROCm/Ollama/Laya smoke paths, but remains In Progress pending authorized training-data generation, two disjoint attested human sets, the Anthropic API baseline, committed final results, and the go/no-go. TASK-5.13 remains dependency-blocked and TASK-5.14 remains unstarted. The Phase 1 acceptance criteria and epic status must remain unchanged until the real TASK-5.11 result proves both p50 under 10 ms and accuracy at least the 7B baseline.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 1 delivered one sema<T> expression end to end: the refund decision compiles from a .sem.ts file, trains through the CLI with a pluggable teacher, verifies against attested real inputs and its constraints, exports an immutable artifact and executes from Node at p50 4.82 ms on a CPU with accuracy 1.000 on the 160-case judge-attested final set, equal to Claude Sonnet 5 with structured output and far above the 7B generative comparator (0.538); the exit criteria are met in a machine-checked record (results-final-2026-09-25, status go). Verified by the committed benchmark records, the test suites of every package and the Phase 1 documentation.
<!-- SECTION:FINAL_SUMMARY:END -->
