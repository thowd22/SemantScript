---
id: TASK-11
title: 'Docs: Phase 2 — execution plans, multi-head artifacts and structured outputs'
status: To Do
assignee: []
created_date: '2026-09-20 02:47'
labels:
  - docs
milestone: m-2
dependencies:
  - TASK-6.5
ordinal: 36000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2 introduces the concepts users will find hardest to reason about: how independent sema expressions fuse into one encoder pass, how dependent ones become stages, and how interface outputs map to heads. Without documentation these look like magic and users cannot predict performance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A concepts page explains the neural execution plan with a worked example showing source, DAG and stages
- [ ] #2 The artifact reference is updated for shared encoder, adapters and multiple heads
- [ ] #3 A page documents structured interface outputs: which shapes are supported, how fields become heads, and what verification reports per field
- [ ] #4 Scaling benchmark results are written up with guidance on when fusing helps
- [ ] #5 The docs index and Phase 1 pages are updated where behavior changed
<!-- AC:END -->
