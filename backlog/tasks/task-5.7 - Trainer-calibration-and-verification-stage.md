---
id: TASK-5.7
title: 'Trainer: calibration and verification stage'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - trainer
milestone: m-1
dependencies:
  - TASK-5.5
  - TASK-5.6
parent_task_id: TASK-5
ordinal: 12000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Confidence must mean something for @confidence thresholds to work, and the artifact must be proven against examples and constraints before it ships. Transcript turn 10 artifact block lists verification fields.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Post-training calibration (e.g. temperature scaling) is applied and expected calibration error is reported
- [ ] #2 All gold examples are checked and any miss fails the build
- [ ] #3 All constraints are checked on adversarial cases and any violation fails the build
- [ ] #4 Verification stats are written into the artifact manifest
<!-- AC:END -->
