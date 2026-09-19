---
id: TASK-5.5
title: 'Trainer: adversarial case generation around constraints'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - trainer
milestone: m-1
dependencies:
  - TASK-5.4
parent_task_id: TASK-5
ordinal: 10000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deterministic constraints (never/always) define hard boundaries. Training data must be dense around those boundaries so the model learns them, and the verifier can check them. Transcript turn 9 point 8.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 For each never/always constraint, cases are generated on both sides of the boundary
- [ ] #2 Constraint-violating labels are never emitted as training targets
- [ ] #3 Adversarial cases are tagged so the verifier can report constraint accuracy separately
<!-- AC:END -->
