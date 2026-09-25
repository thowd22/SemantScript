---
id: TASK-14.7
title: 'semantscript explain: why an expression answered as it did'
status: To Do
assignee: []
created_date: '2026-09-25 15:21'
labels:
  - dx
  - debug
milestone: m-5
dependencies: []
parent_task_id: TASK-14
ordinal: 60000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a trained expression gives a wrong answer the developer has sema.withConfidence in code and the train report on disk, and nothing in between: no command shows the calibrated distribution for an input, which constraints and examples are active for it, the closest training cases and their labels, or whether the expression's artifact is even the current build. The editor plugin shows verified accuracy but cannot answer for one input.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 semantscript explain <module.js> --call <export> --input <json> prints the value, the calibrated distribution and confidence, every constraint active for the input with whether the answer satisfies it, the gold examples nearest to the input, and the nearest training cases from the cached datasets with their labels and origins
- [ ] #2 The command says when the compiled function id is missing from the loaded artifact (changed since training) and which release and verification metrics the answer came from
- [ ] #3 The diagnostics page documents a wrong-answer workflow that starts with explain and ends with the example or constraint to add
<!-- AC:END -->
