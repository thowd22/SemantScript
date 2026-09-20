---
id: TASK-5.9
title: 'Runtime: load artifact and execute one forward pass in Node'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-20 19:18'
labels:
  - runtime
milestone: m-1
dependencies:
  - TASK-2
  - TASK-4
references:
  - docs/research/laya-analysis.md
parent_task_id: TASK-5
ordinal: 14000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
@semantscript/core must resolve __sema.call(functionId, inputs) to a typed value with no token stream, parsing or validation. Uses ONNX Runtime in-process. Transcript turn 9 point 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Runtime loads an artifact directory and exposes __sema.call(functionId, inputs)
- [ ] #2 Inputs are serialized per the input schema and the head output is mapped back to the TS type value
- [ ] #3 Calling an unknown function id or malformed inputs throws a typed error
- [ ] #4 Runtime has no dependency on Python
- [ ] #5 Inputs are rendered with the canonical serialization from the IR; a test proves key-order invariance
<!-- AC:END -->
