---
id: TASK-5.3
title: 'Compiler: rewrite sema sites to runtime calls and emit IR bundle'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - compiler
milestone: m-1
dependencies:
  - TASK-5.2
parent_task_id: TASK-5
ordinal: 8000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final compiler step for Phase 1: replace each sema expression with __sema.call(functionId, { inputs }) and write the IR bundle that the trainer consumes. Function ids must be stable across builds so unchanged expressions can be cached later.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Compiled JS contains a __sema.call with a deterministic function id and the interpolated variables as inputs
- [ ] #2 The natural-language text is absent from the compiled JS
- [ ] #3 An IR bundle (one record per site) is written alongside the JS output
- [ ] #4 Function id is unchanged when the expression text and types are unchanged, and changes when either changes
<!-- AC:END -->
