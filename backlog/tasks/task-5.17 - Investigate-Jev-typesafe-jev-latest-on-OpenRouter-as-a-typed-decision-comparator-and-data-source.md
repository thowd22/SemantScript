---
id: TASK-5.17
title: >-
  Investigate Jev (typesafe/jev-latest on OpenRouter) as a typed-decision
  comparator and data source
status: To Do
assignee: []
created_date: '2026-09-23 23:26'
labels:
  - research
  - benchmark
milestone: m-1
dependencies:
  - TASK-5.11
references:
  - 'https://openrouter.ai/api/v1/models'
  - docs/research/laya-analysis.md
  - benchmarks/refund/src/adapters/laya.ts
parent_task_id: TASK-5
ordinal: 49000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The user has OpenRouter credits and access to Jev, published as ~typesafe/jev-latest. Jev is not a general LLM: it is a typesafe decision model of the same kind SemantScript compiles per application, so it belongs alongside Laya as an external comparator rather than in the teacher slot. It is inexpensive, which makes it a candidate source of large volumes of typed decisions and behavior data for distillation once the refund benchmark is finished. The id is not in OpenRouter's public catalog (public lookups of typesafe/jev-latest return 404 on 2026-09-23), so it is account-visible only and needs the user's OPENROUTER_API_KEY to query. Deliberately sequenced after TASK-5.11 so it does not disturb the refund release run. Framed by the north star: any use of Jev must serve per-application accuracy or faster compilation, not a universal model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Jev's API contract is documented from a live authenticated probe: how a typed decision is requested (schema, options, calibration), what the response carries, pricing per request, and latency
- [ ] #2 Jev is run on the frozen refund final set under the benchmark's protocol as a diagnostic comparator and its accuracy, attested-slice accuracy, calibration and latency are reported next to Laya and the Qwen baselines
- [ ] #3 Label agreement between Jev and the judge rubric is measured per policy rule and the cost of generating a 10k-case typed corpus through Jev is estimated
- [ ] #4 A decision record states whether Jev becomes a required benchmark system, a distillation or behavior-data source for Phase 2 and Phase 3, or neither, with the north-star test applied
<!-- AC:END -->
