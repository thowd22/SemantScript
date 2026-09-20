---
id: TASK-7.8
title: >-
  Editor diagnostics: LSP/TS plugin surfacing accuracy and guidance at the sema
  site
status: To Do
assignee: []
created_date: '2026-09-20 19:41'
labels:
  - dx
milestone: m-3
dependencies:
  - TASK-7.4
  - TASK-7.7
parent_task_id: TASK-7
ordinal: 44000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The biggest usability lever: make tuning a neural function feel like fixing a type error. At each sema site the editor shows verified accuracy, ECE, pair-consistency, example and constraint counts, and actionable diagnostics such as 'no examples and accuracy below threshold'.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A TypeScript language-service plugin shows a hover at each sema site with accuracy, ECE, pair-consistency, and example/constraint counts from the latest artifact
- [ ] #2 Diagnostics appear inline for unsupported types, missing inputs, below-threshold accuracy or ECE, and unverified expressions
- [ ] #3 Works in VS Code with no extension beyond the TS plugin entry in tsconfig
<!-- AC:END -->
