---
id: TASK-7.7
title: 'semantscript dev: watch mode with incremental retraining and hot-swap'
status: To Do
assignee: []
created_date: '2026-09-20 19:41'
labels:
  - cli
  - dx
milestone: m-3
dependencies:
  - TASK-7.2
parent_task_id: TASK-7
ordinal: 43000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Training is the part developers fear. In dev it should be invisible: edit an expression, the changed head retrains in the background using the content-addressed cache, and the running app picks up the new head without restart.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Saving a file retrains only the sema expressions whose IR changed
- [ ] #2 The running runtime swaps in the new head without a process restart
- [ ] #3 Progress and verification results are streamed to the terminal per expression
- [ ] #4 A stale head remains in service until the new one passes verification
<!-- AC:END -->
