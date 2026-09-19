---
id: TASK-5.10
title: 'Runtime: withConfidence result and threshold enforcement'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - runtime
milestone: m-1
dependencies:
  - TASK-5.9
parent_task_id: TASK-5
ordinal: 15000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Neural values are probabilistic. Confidence must be accessible without contaminating normal syntax, and @confidence must refuse to return an ordinary value below threshold. Transcript turn 10.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 fn.withConfidence(inputs) returns { value, confidence } using calibrated probabilities
- [ ] #2 A function compiled with @confidence(x) throws or invokes the configured fallback when confidence < x
- [ ] #3 Plain calls remain unchanged in shape
<!-- AC:END -->
