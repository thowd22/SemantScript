---
id: TASK-6.6
title: 'External benchmark: Laya typed-decisions suite compiled to sema functions'
status: To Do
assignee: []
created_date: '2026-09-20 19:18'
labels:
  - benchmark
milestone: m-2
dependencies:
  - TASK-6.3
  - TASK-6.4
references:
  - docs/research/laya-analysis.md
parent_task_id: TASK-6
ordinal: 40000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Laya's typed-decisions benchmark (400 cases, 2,000 decisions across agent-trace observability, customer service, invoice processing and security incidents) is a public, multi-question-per-input workload with published numbers for Laya (0.766) and Jev (0.727). Expressing each workflow's questions as sema expressions makes it a direct external test of our multi-head execution plan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the four workflows is written as a .sem.ts file whose sema expressions cover every question in the suite
- [ ] #2 Accuracy per workflow and overall is reported alongside Laya's and Jev's published numbers, on the suite's own held-out split
- [ ] #3 Latency per case (all questions, one execution plan) is reported and compared to Laya's per-question latency
<!-- AC:END -->
