---
id: TASK-6.4
title: Structured interface outputs as multi-field heads
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - compiler
  - model
  - runtime
milestone: m-2
dependencies:
  - TASK-6.1
parent_task_id: TASK-6
ordinal: 21000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Flat interface outputs (e.g. ClaimAssessment with category, severity, fraudRisk, requiresHumanReview) should compile to one head per field, not to a serialized object. Transcript turn 10.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A flat interface output type compiles to one head per field in IR
- [ ] #2 Runtime assembles the typed object from head outputs with no JSON step
- [ ] #3 Per-field accuracy is reported by the verifier
<!-- AC:END -->
