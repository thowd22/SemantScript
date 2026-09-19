---
id: TASK-6
title: 'Phase 2 epic: shared encoder, many heads, compiler parallelism'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
labels:
  - epic
milestone: m-2
dependencies:
  - TASK-5
ordinal: 17000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Once one expression works, prove the performance thesis: many sema sites share one encoder pass and the compiler schedules them. Transcript turn 9 points 4-6.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A file with several independent sema expressions executes with a single encoder pass
- [ ] #2 Dependent expressions execute as ordered inference stages
<!-- AC:END -->
