---
id: TASK-5.2
title: 'Compiler: resolve output type T and interpolated input types to IR'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - compiler
milestone: m-1
dependencies:
  - TASK-5.1
parent_task_id: TASK-5
ordinal: 7000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The head shape is derived entirely from T, and the input schema from the ${} interpolations. This is where type safety is established structurally. Uses the TS type checker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 boolean, string-literal union, enum and flat interface output types produce the correct head spec in IR
- [ ] #2 Each ${} interpolation produces an input schema entry with its resolved TS type
- [ ] #3 Unsupported output types (free string, nested objects, arrays) produce a clear compile error naming the site
- [ ] #4 Emitted IR validates against the schema from the IR task
<!-- AC:END -->
