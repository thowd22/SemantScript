---
id: TASK-3
title: Decide teacher model and training stack
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
labels:
  - decision
milestone: m-0
dependencies: []
references:
  - PLAN.md
priority: high
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two open decisions block the trainer: which model generates synthetic training data (API vs local) and whether trainer/model are Python/PyTorch with a Node compiler/runtime, or all-TypeScript. PLAN.md section 7 recommends Python for training and Node for compiler/runtime. Record the choice as a Backlog decision so future work does not relitigate it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A backlog decision record states the chosen teacher model and why
- [ ] #2 A backlog decision record states the chosen language split per component and why
- [ ] #3 PLAN.md section 7 is updated to reflect both decisions as resolved
<!-- AC:END -->
