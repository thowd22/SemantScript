---
id: TASK-5.17
title: >-
  Investigate Jev (typesafe/jev-latest on OpenRouter) as a typed-decision
  comparator and data source
status: To Do
assignee: []
created_date: '2026-09-23 23:26'
updated_date: '2026-09-23 23:30'
labels:
  - research
  - benchmark
milestone: m-1
dependencies:
  - TASK-5.11
references:
  - 'https://anth.us/blog/distilling-jev-into-a-classifier/'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reference read 2026-09-23: Anthus's write-up 'Distilling Jev into a classifier'. Jev is Anthus's hosted model that answers typed questions about text (yes/no, multiple choice, rubric scoring) with a value and a confidence per question, metered per request. They distilled it into a 66M DistilBERT student: 140 human-labeled items calibrated the teacher (a logistic head over Jev's holistic answer plus seven cached answers), the calibrated teacher soft-labeled 5,140 unlabeled items, the student trained 3 epochs on soft targets with cross-entropy against teacher probabilities, and 3,521 held-out items scored it. Teacher 0.890 accuracy vs human labels; soft student 0.912, hard student 0.908, ceiling 0.938; student-teacher agreement 0.940; ECE 0.038 raw to 0.033 calibrated; student latency 5.6 to 15 ms. Lessons they stress: validate teacher calibration before distilling (uncalibrated soft targets teach wrong confidence), gate per slice not on an overall number (one slice failed under one seed while 10 of 11 passed), and a teacher-fallback cascade lost accuracy because the teacher was worse on exactly the items the student deferred. Relevance for us: same recipe as SemantScript's compile path (teacher labels, small encoder, temperature calibration, release gate); worth adopting per-rule gating and soft-target distillation when Jev or Laya act as calibrated teachers. Article gives no request schema, batch option or pricing; get those from the authenticated OpenRouter listing.
<!-- SECTION:NOTES:END -->
