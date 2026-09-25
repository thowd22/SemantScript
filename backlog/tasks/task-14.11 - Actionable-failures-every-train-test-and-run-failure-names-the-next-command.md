---
id: TASK-14.11
title: 'Actionable failures: every train, test and run failure names the next command'
status: To Do
assignee: []
created_date: '2026-09-25 15:21'
labels:
  - dx
  - debug
milestone: m-5
dependencies: []
parent_task_id: TASK-14
ordinal: 64000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The diagnostics catalogue lists every error with a cause and a fix, but the tools themselves do not say it: a verification failure prints the failing cases and stops, a runtime SemaRuntimeNotLoadedError says only that no artifact is loaded, and the trainer's Python tracebacks reach the terminal unwrapped when the interpreter or a package is missing. A developer should not need the catalogue to know what to do next.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every verification failure line in the train report ends with a specific suggestion derived from the evidence: the constraint to add for a rule the misses share, an example for a repeated miss, more cases or epochs for a calibration failure, or the seed retry
- [ ] #2 Runtime errors for a missing or stale artifact name the command that fixes them, and the CLI wraps trainer and interpreter failures into one line with the doctor check to run
- [ ] #3 The diagnostics catalogue's fix column and the tools' messages are generated from one source so they cannot drift
<!-- AC:END -->
