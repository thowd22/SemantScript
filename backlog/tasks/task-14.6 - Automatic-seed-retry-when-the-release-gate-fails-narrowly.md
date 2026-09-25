---
id: TASK-14.6
title: Automatic seed retry when the release gate fails narrowly
status: To Do
assignee: []
created_date: '2026-09-25 15:21'
labels:
  - dx
  - train
milestone: m-5
dependencies: []
parent_task_id: TASK-14
ordinal: 59000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The release gate is strict and a training seed matters: the refund benchmark's first release needed a second seed, and the Express example on 2026-09-25 failed seeds 1 and 2 on the same handful of near-threshold cases (2.0% and 1.3% violations against a 1% tolerance, ECE above the bar once) before seed 3 published, each retry a manual rerun that the developer had to know to try. The datasets are cached, so a retry costs only GPU time.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When verification fails only on the constraint-violation rate or the ECE threshold by a bounded margin, train retrains with the next seed automatically, up to a configured number of attempts, reusing the cached datasets
- [ ] #2 The report lists every attempt with its seed and gate metrics, and the published release records the seed that passed
- [ ] #3 A gold-example miss or a failure outside the margin does not retry and says why
<!-- AC:END -->
