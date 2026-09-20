---
id: TASK-8
title: 'Phase 4 epic: framework layer'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-20 19:41'
labels:
  - epic
milestone: m-4
dependencies:
  - TASK-7
  - TASK-13
ordinal: 28000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Thin integrations for popular stacks and a reference application that replaces a meaningful share of hand-written logic. Deliberately last and deliberately thin: adoption comes from dropping sema into existing apps (Phase 3), not from owning the whole application. The transcript is explicit that the framework must not be built before the primitive is proven.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reference application serves HTTP requests whose handlers mix deterministic code and sema expressions
<!-- AC:END -->
