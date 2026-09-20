---
id: TASK-5.4
title: 'Trainer: synthetic case generation from IR via teacher model'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-20 02:43'
labels:
  - trainer
milestone: m-1
dependencies:
  - TASK-5.12
  - TASK-2
  - TASK-4
parent_task_id: TASK-5
ordinal: 9000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The natural-language spec is consumed at build time only. A teacher model turns the IR (behavior text, input schema, output type, examples) into labeled training cases. Transcript turn 5 describes English -> formal behavior -> generated training set.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Given an IR record, the generator produces N labeled cases whose inputs match the input schema and labels are within the output type
- [ ] #2 Provided examples are included verbatim and marked as gold
- [ ] #3 Generation is resumable and cached so re-running with the same IR does not re-call the teacher
- [ ] #4 Output dataset format is documented
<!-- AC:END -->
