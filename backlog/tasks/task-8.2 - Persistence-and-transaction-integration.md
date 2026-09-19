---
id: TASK-8.2
title: Persistence and transaction integration
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - runtime
milestone: m-4
dependencies:
  - TASK-8.1
parent_task_id: TASK-8
ordinal: 30000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Data stays out of the weights. Handlers read and write a real database; neural decisions gate transactions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Example handler reads from Postgres, evaluates a sema expression, and commits or rolls back a transaction based on the result
- [ ] #2 No sema expression has ambient database access
<!-- AC:END -->
