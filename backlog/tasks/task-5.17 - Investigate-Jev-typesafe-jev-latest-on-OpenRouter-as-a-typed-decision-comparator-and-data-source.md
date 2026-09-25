---
id: TASK-5.17
title: >-
  Investigate Jev (typesafe/jev-latest on OpenRouter) as a typed-decision
  comparator and data source
status: Done
assignee:
  - '@claude'
created_date: '2026-09-23 23:26'
updated_date: '2026-09-25 04:17'
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
- [x] #1 Jev's API contract is documented from a live authenticated probe: how a typed decision is requested (schema, options, calibration), what the response carries, pricing per request, and latency
- [x] #2 Jev is run on the frozen refund final set under the benchmark's protocol as a diagnostic comparator and its accuracy, attested-slice accuracy, calibration and latency are reported next to Laya and the Qwen baselines
- [x] #3 Label agreement between Jev and the judge rubric is measured per policy rule and the cost of generating a 10k-case typed corpus through Jev is estimated
- [x] #4 A decision record states whether Jev becomes a required benchmark system, a distillation or behavior-data source for Phase 2 and Phase 3, or neither, with the north-star test applied
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Live probe (done): ~typesafe/jev-latest resolves to typesafe/jev-1.13-20260917 at POST /api/alpha/decisions; chat/completions refuses it. Request {model, state: string|record|array, questions: {key: {type: noul|choice|score, instructions, criteria: record (choice, noul true/false) | ordered array (score)}}}; response {model, answers: {key: {type, choice+probabilities+confidence | noul probability | score+legend+probabilities+confidence}}, usage {input_tokens, output_tokens, cost}, id, provider}. Usage cost 2.4e-5 USD for 566 input tokens (0.042 USD per million input tokens, output free), 148 ms wall for one request. Document in the results README (AC1).
2. Comparator driver benchmarks/refund/program/run_jev_comparator.py: for every case of the frozen final set, one choice question over the policy text plus the customer and order records, the same policy text and support the baselines use, sequential requests, per-call latency, usage cost summed; writes predictions in the refund predictions shape (caseId, inputSha256, value, distribution, latencyMs) plus raw answers, and a results record with accuracy, attested-slice accuracy, 15-bin ECE, Brier, latency p50/p95 and total cost, next to the Laya and Qwen numbers from results-v2-2026-09-23 (AC2). Jev is not a pinned benchmark role, so its predictions are a diagnostic record, not a sealed system entry.
3. Agreement per policy rule (AC3): join Jev's answers to adjudications.json steps 1-5 and report agreement per step; estimate the cost of a 10k-case typed corpus from the measured tokens per case and the price.
4. Decision record (AC4) applying the north-star test: comparator, distillation or behavior-data source, or neither. Spend budget for this task: well under 1 USD of the 50 USD key limit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reference read 2026-09-23: Anthus's write-up 'Distilling Jev into a classifier'. Jev is Anthus's hosted model that answers typed questions about text (yes/no, multiple choice, rubric scoring) with a value and a confidence per question, metered per request. They distilled it into a 66M DistilBERT student: 140 human-labeled items calibrated the teacher (a logistic head over Jev's holistic answer plus seven cached answers), the calibrated teacher soft-labeled 5,140 unlabeled items, the student trained 3 epochs on soft targets with cross-entropy against teacher probabilities, and 3,521 held-out items scored it. Teacher 0.890 accuracy vs human labels; soft student 0.912, hard student 0.908, ceiling 0.938; student-teacher agreement 0.940; ECE 0.038 raw to 0.033 calibrated; student latency 5.6 to 15 ms. Lessons they stress: validate teacher calibration before distilling (uncalibrated soft targets teach wrong confidence), gate per slice not on an overall number (one slice failed under one seed while 10 of 11 passed), and a teacher-fallback cascade lost accuracy because the teacher was worse on exactly the items the student deferred. Relevance for us: same recipe as SemantScript's compile path (teacher labels, small encoder, temperature calibration, release gate); worth adopting per-rule gating and soft-target distillation when Jev or Laya act as calibrated teachers. Article gives no request schema, batch option or pricing; get those from the authenticated OpenRouter listing.

Live probe with the user's OpenRouter key (stored in the git-ignored .env, read from the environment, never written to any record): ~typesafe/jev-latest is served only on POST /api/alpha/decisions (chat/completions refuses it), resolves to typesafe/jev-1.13-20260917, request {model, state: string|record|array, questions: {id: {type: noul|choice|score, instructions, criteria: record | true/false record | ordered array}}}, response {model, answers{choice+probabilities+confidence | noul | score+legend+probabilities+confidence}, usage{input_tokens, output_tokens, cost}, id, provider}; USD 0.042 per million input tokens, output free; 148 ms wall for a three-question probe. Comparator run (program/run_jev_comparator.py, budget-capped, answers cached, 4 offline tests): 160 final cases, one choice question each with the same policy text, hard constraints and canonical inputs as the baselines: accuracy 0.9625 (154/160), 15-bin ECE 0.0307, Brier 0.0456, p50 147.5 ms / p95 221.8 ms over the network, total cost USD 0.00578 (861 input tokens per case). Per rubric step: 1 stale 28/28, 2 fraud 26/26, 3 outside window 39/39, 4 suspicious 9/10, 5 approve 52/57; all six misses are enterprise paid orders aged 33-35 days denied as if the 30-day standard window applied (10/16 on the enterprise 31-60-day slice). 10k-case corpus estimate USD 0.36. Results under benchmarks/refund/data/results-jev-2026-09-25 (README, results.json, raw-answers.json); decision-11 recorded (diagnostic comparator and constraint-filtered corpus source only; not a benchmark system, teacher or runtime dependency). Verified: ruff clean, pytest benchmarks/refund/program 74 passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documented Jev's decisions API from a live authenticated probe (endpoint, typed question and answer schemas, pricing, latency), ran it as a diagnostic comparator on the frozen refund final set under the benchmark's definitions (0.9625, ECE 0.031, p50 148 ms over the network, USD 0.006 in total; next to SemantScript 1.000 at 4.8 ms, Qwen 7B 0.54, Laya 0.225), measured agreement per rubric step (perfect on four rules; every miss is the enterprise 60-day window read as 30 days) and estimated a 10k-case corpus at USD 0.36. Decision-11: a diagnostic comparator and a constraint-filtered corpus source only. Verified by the driver's offline tests, ruff and the committed results record.
<!-- SECTION:FINAL_SUMMARY:END -->
