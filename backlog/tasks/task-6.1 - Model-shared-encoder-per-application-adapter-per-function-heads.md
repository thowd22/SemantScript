---
id: TASK-6.1
title: 'Model: shared encoder + per-application adapter + per-function heads'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-19 18:23'
labels:
  - model
milestone: m-2
dependencies:
  - TASK-5.8
parent_task_id: TASK-6
ordinal: 18000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Avoid one model file per function. One artifact holds a shared encoder, a small app adapter and a tiny head per sema site so a controller with 50 expressions shares nearly all compute.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Multiple functions train into one artifact with separate heads
- [ ] #2 Adding a head does not require retraining existing heads
- [ ] #3 Per-head accuracy is reported and comparable to single-function training
<!-- AC:END -->
