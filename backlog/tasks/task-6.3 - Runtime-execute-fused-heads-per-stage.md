---
id: TASK-6.3
title: 'Runtime: execute fused heads per stage'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - runtime
milestone: m-2
dependencies:
  - TASK-6.1
  - TASK-6.2
parent_task_id: TASK-6
ordinal: 20000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consume the execution plan: run the encoder once per stage and fan out to all heads in that stage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A stage with N heads performs one encoder pass and N head passes
- [ ] #2 Stage outputs feed subsequent stages as inputs
- [ ] #3 Results are identical to executing each function independently
<!-- AC:END -->
