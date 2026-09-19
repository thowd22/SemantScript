---
id: TASK-8.1
title: Controller abstraction and HTTP runtime
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - runtime
milestone: m-4
dependencies:
  - TASK-7.1
parent_task_id: TASK-8
ordinal: 29000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Transcript turn 9 point 9: @Controller / @Post decorators with sema expressions inside handlers; the runtime batches sema evaluation per request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Decorator-based controllers route HTTP requests to handlers
- [ ] #2 sema expressions inside a handler are executed via the execution plan
- [ ] #3 Deterministic require() guards run before and after neural results as written
<!-- AC:END -->
