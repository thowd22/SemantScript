---
id: TASK-14.4
title: >-
  Built-in constraints teacher: train an expression with complete constraints
  and no language model from the CLI
status: To Do
assignee: []
created_date: '2026-09-25 15:21'
labels:
  - dx
  - train
milestone: m-5
dependencies: []
parent_task_id: TASK-14
ordinal: 57000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When an expression's constraints decide every input, the constraints are a labeling function: the reference application trains all nine of its expressions that way with no API key (examples/refund-service/train.py, a 400-line custom driver), and decision-12 makes constraint labels the local iteration path. But semantscript train has no such backend: [teacher] offers anthropic and ollama only, so a developer who wrote complete constraints still has to pay a teacher or copy the driver. The built-in path should also apply to a mixed expression: constraints label the inputs they decide and the language-model teacher labels the rest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 [teacher] backend = "constraints" (and semantscript train --teacher constraints) trains any expression whose constraints admit exactly one output for every sampled input, sampling inputs from the IR types with threshold-aware numeric ranges, and generates boundary pairs and counterfactual twins by single-field edits
- [ ] #2 An expression whose constraints are incomplete gets a clear message naming an input the constraints do not decide, or, when a language-model teacher is also configured, uses the constraints for the inputs they decide and the teacher for the rest
- [ ] #3 examples/refund-service/train.py is replaced by the built-in backend and its tests and README still pass
- [ ] #4 The teacher provenance records the constraints teacher and its sampling configuration digest
<!-- AC:END -->
