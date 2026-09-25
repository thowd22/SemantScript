---
id: TASK-6
title: 'Phase 2 epic: shared encoder, many heads, compiler parallelism'
status: Done
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-25 01:29'
labels:
  - epic
milestone: m-2
dependencies:
  - TASK-5
  - TASK-11
ordinal: 17000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Once one expression works, prove the performance thesis: many sema sites share one encoder pass and the compiler schedules them. Transcript turn 9 points 4-6.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A file with several independent sema expressions executes with a single encoder pass
- [x] #2 Dependent expressions execute as ordered inference stages
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed on the subtasks' evidence: TASK-6.3's runtime callStage runs every function of a stage over one encoder pass per distinct input (runtime tests assert passes.encoder), TASK-6.5's stage-scaling results (benchmarks/refund/data/results-stage-scaling-2026-09-24) measure 1, 10 and 50 heads sharing one encoder pass, and TASK-6.2 plus executeSemaPlan run dependent expressions as ordered stages with producers feeding consumers (compiler execution-plan tests, runtime plan tests). TASK-6.7 adds routed domains and depth routing on top.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 2 delivered the shared-encoder application: one encoder and adapter with per-function heads (6.1), the compiler's dependency DAG and staged execution plan (6.2), fused per-stage execution in the runtime (6.3), structured multi-field outputs (6.4), the parallel-head and batch scaling measurements (6.5), the typed-decisions external benchmark (6.6) and compile-time routed domain adapters with depth routing (6.7), which brought the refund artifact under the 10 ms bar.
<!-- SECTION:FINAL_SUMMARY:END -->
