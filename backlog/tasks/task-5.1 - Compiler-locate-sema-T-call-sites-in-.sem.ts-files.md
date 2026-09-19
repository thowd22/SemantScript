---
id: TASK-5.1
title: 'Compiler: locate sema<T> call sites in .sem.ts files'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
labels:
  - compiler
milestone: m-1
dependencies:
  - TASK-2
  - TASK-4
parent_task_id: TASK-5
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
First step of the compiler pipeline. The transformer must find every sema tagged-template expression in a TS program so later steps can type it and rewrite it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Given a .sem.ts file, the compiler returns every sema<T> tagged-template site with source location
- [ ] #2 Non-sema tagged templates and unrelated identifiers named sema are not matched
- [ ] #3 Unit tests cover nested functions, class methods, and arrow functions
<!-- AC:END -->
