---
id: TASK-14.10
title: 'Release management: list, inspect, promote and roll back artifact releases'
status: To Do
assignee: []
created_date: '2026-09-25 15:21'
labels:
  - dx
  - deploy
milestone: m-5
dependencies: []
parent_task_id: TASK-14
ordinal: 63000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every train publishes an immutable release under the artifact root and flips current.json, and the runtime hot-swaps on that pointer, but there is no command to see what releases exist, what each one verified, which is current, or to roll back to the previous one when a new release behaves worse in production. Today that is a manual edit of current.json.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 semantscript releases lists every release under the artifact root with its date, manifest digest, per-function verification summary and which one is current
- [ ] #2 semantscript releases rollback (to the previous or a named release) rewrites the pointer atomically and a running process with watch enabled swaps to it
- [ ] #3 semantscript releases prune removes releases older than a count or age while never removing the current one
<!-- AC:END -->
