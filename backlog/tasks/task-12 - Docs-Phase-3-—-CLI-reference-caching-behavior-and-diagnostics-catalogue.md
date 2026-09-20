---
id: TASK-12
title: 'Docs: Phase 3 — CLI reference, caching behavior and diagnostics catalogue'
status: To Do
assignee: []
created_date: '2026-09-20 02:47'
labels:
  - docs
milestone: m-3
dependencies:
  - TASK-7.1
  - TASK-7.2
  - TASK-7.3
  - TASK-7.4
ordinal: 37000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 3 is where external developers first touch the tool. The CLI, cache rules and error messages are the entire product surface for them, and undocumented behavior here becomes support load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A CLI reference documents every command, flag and exit code for build, train, test and run
- [ ] #2 A page explains the build cache: what is content-addressed, when retraining happens, where the cache lives and how to clear it
- [ ] #3 A diagnostics catalogue lists every compiler and verifier error with its cause and fix
- [ ] #4 The universal-encoder investigation outcome is written up with its recommendation
- [ ] #5 The getting-started tutorial is rewritten to use the CLI end to end
<!-- AC:END -->
