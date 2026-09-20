---
id: TASK-5.7
title: 'Trainer: calibration and verification stage'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-20 19:18'
labels:
  - trainer
milestone: m-1
dependencies:
  - TASK-5.5
  - TASK-5.6
references:
  - docs/research/laya-analysis.md
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
- [ ] #5 A per-function temperature is fitted on a calibration split held out from training and stored in the artifact
- [ ] #6 ECE and Brier score are reported per function and the build fails above a configured ECE threshold
<!-- AC:END -->
