---
id: TASK-15.2
title: >-
  Retrain the Express example under the held-out gate and make its README
  describe the release on disk
status: To Do
assignee: []
created_date: '2026-09-26 06:27'
labels:
  - dx
  - example
milestone: m-6
dependencies:
  - TASK-15.1
parent_task_id: TASK-15
priority: high
ordinal: 67000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
examples/express-app currently serves release 217d386c (built 2026-09-26T00:48Z, seed 5 of the TASK-14.5 reduced-prompt comparison, refund accuracy 0.924, ECE 0.072), which violates the 90-day rule on stale orders, while examples/express-app/README.md quotes release 5c755d08 (refund 0.987, ECE 0.008) that is no longer under .semantscript/artifact/releases. The cached datasets (examples/express-app/.semantscript/cache: the reduced-prompt dataset caa8e895… and two earlier ones) allow retraining without teacher spend; seed retry (--seed-attempts) is available. Do not spend more than USD 0.50 of the OpenRouter key without the user's approval; if no seed passes from the cache, record the per-seed held-out violation rates and stop. Held-out benchmark inputs never enter training. Related: docs/tutorial-refund-decision.md and docs/deploy.md quote the example's sizes and numbers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 examples/express-app serves a release trained from the cached datasets that passes the held-out constraint gate, and semantscript explain answers deny for standard and enterprise customers with paid and fraudulent orders at 100, 120, 150 and 200 days
- [ ] #2 examples/express-app/README.md, docs/tutorial-refund-decision.md and docs/deploy.md quote the current release's digest, seed, accuracy, ECE, corpus and held-out violation figures and sizes, and semantscript releases list agrees with them
- [ ] #3 If no cached-dataset seed passes, the per-seed corpus and held-out violation rates are recorded in the task and no teacher spend happens without the user's approval
<!-- AC:END -->
