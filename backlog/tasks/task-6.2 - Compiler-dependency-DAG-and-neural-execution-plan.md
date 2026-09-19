---
id: TASK-6.2
title: 'Compiler: dependency DAG and neural execution plan'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - compiler
milestone: m-2
dependencies:
  - TASK-5.3
parent_task_id: TASK-6
ordinal: 19000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The compiler can see which sema expressions depend only on request inputs and which consume other sema results. Emit an execution plan (stages of fused heads) analogous to a database query plan. Transcript turn 9 points 5-6.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Compiler builds a DAG of sema sites from data dependencies
- [ ] #2 Independent sites are grouped into one stage; dependent sites into later stages
- [ ] #3 The plan is emitted in the IR bundle and is human-readable
<!-- AC:END -->
