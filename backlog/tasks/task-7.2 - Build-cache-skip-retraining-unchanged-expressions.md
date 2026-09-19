---
id: TASK-7.2
title: 'Build cache: skip retraining unchanged expressions'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - cli
  - trainer
milestone: m-3
dependencies:
  - TASK-7.1
parent_task_id: TASK-7
ordinal: 25000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Retraining every function on every build is unacceptable. Function ids are content-addressed; use them to reuse existing heads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A rebuild with no changes performs no training
- [ ] #2 Changing one expression retrains only that head
- [ ] #3 Cache location and invalidation rules are documented
<!-- AC:END -->
