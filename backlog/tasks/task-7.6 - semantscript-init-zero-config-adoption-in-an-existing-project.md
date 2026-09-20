---
id: TASK-7.6
title: 'semantscript init: zero-config adoption in an existing project'
status: To Do
assignee: []
created_date: '2026-09-20 19:41'
labels:
  - cli
  - dx
milestone: m-3
dependencies:
  - TASK-7.1
  - TASK-7.5
parent_task_id: TASK-7
ordinal: 42000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A new user should go from an existing repo to a compiling sema expression in one command. Defaults for teacher, encoder, thresholds and artifact location must be sensible enough that no config file is required to start.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 npx semantscript init detects the project's build tool (tsc, Vite, Next, esbuild) and wires the matching plugin
- [ ] #2 A project with no semantscript config builds and trains with documented defaults
- [ ] #3 init adds the artifact directory to the correct place for the detected framework so the runtime finds it in dev and production
- [ ] #4 The whole flow from npm install to first working expression is under five minutes on the documented hardware
<!-- AC:END -->
