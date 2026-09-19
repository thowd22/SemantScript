---
id: TASK-7.1
title: 'CLI: semantscript build | train | test | run'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - cli
milestone: m-3
dependencies:
  - TASK-6.3
parent_task_id: TASK-7
ordinal: 24000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Single entry point wrapping compiler, trainer, verifier and runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 build compiles .sem.ts and emits JS + IR bundle
- [ ] #2 train produces the artifact from the IR bundle
- [ ] #3 test runs verification and reports per-function stats
- [ ] #4 run executes a program with the runtime loaded
<!-- AC:END -->
