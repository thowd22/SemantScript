---
id: TASK-5.5
title: 'Trainer: adversarial case generation around constraints'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-20 19:36'
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
Deterministic constraints (never/always) define hard boundaries. Training data must be dense around those boundaries so the model learns them, and the verifier can check them. Transcript turn 9 point 8. Beyond constraints, every generated case should get a counterfactual twin: a minimal edit to the inputs that flips the label, with the teacher stating the reason. A small encoder otherwise learns surface correlations (customer name, order size) instead of the actual policy; counterfactual pairs force it to learn the decision boundary. This is one of the central techniques behind Jev-style and Laya-style decision models.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 For each never/always constraint, cases are generated on both sides of the boundary
- [ ] #2 Constraint-violating labels are never emitted as training targets
- [ ] #3 Adversarial cases are tagged so the verifier can report constraint accuracy separately
- [ ] #4 For every generated case, the teacher produces a counterfactual twin: a minimal input edit that changes the label, plus the stated reason for the flip
- [ ] #5 Counterfactual twins are stored as linked pairs so the verifier can report pair-consistency accuracy (both members correct) separately from per-case accuracy
- [ ] #6 Counterfactual generation is a config option with a ratio so data cost can be tuned
<!-- AC:END -->
