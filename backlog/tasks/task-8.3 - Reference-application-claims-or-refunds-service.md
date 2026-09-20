---
id: TASK-8.3
title: 'Reference application: claims or refunds service'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-20 20:11'
labels:
  - examples
milestone: m-4
dependencies:
  - TASK-8.2
parent_task_id: TASK-8
ordinal: 31000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
End-to-end demonstration of the vision: developers write database interactions, API endpoints and plumbing in TypeScript, and express nearly all business logic as sema expressions. The reference app should look like that, not like a demo with one neural call. It is the artifact people will judge the project by.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Example app lives under examples/ and runs with semantscript run
- [ ] #2 README walks through the source, the compiled output and the artifact
- [ ] #3 The reference app replaces a meaningful share of hand-written business logic with sema expressions, and the walkthrough reports which logic moved, which stayed deterministic and why, with accuracy and latency per expression
- [ ] #4 Business logic in the reference app is expressed as sema expressions wherever it is semantic; deterministic code is limited to persistence, routing, validation, transactions and constraints, and the walkthrough tallies lines of logic in each category
- [ ] #5 The app uses at least three domains so routed adapters and the execution plan are exercised in a realistic shape
<!-- AC:END -->
