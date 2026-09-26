---
id: TASK-15.5
title: >-
  Int8 derivation for applications: a CLI step that derives, verifies and
  publishes a quantized release
status: To Do
assignee: []
created_date: '2026-09-26 06:27'
labels:
  - dx
  - deploy
milestone: m-6
dependencies: []
parent_task_id: TASK-15
priority: medium
ordinal: 70000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
semantscript package (TASK-14.9) reports int8 quantization as a size lever that would bring the Express example's 653.6 MiB trained release to about 227.6 MiB, under Lambda's 250 MiB zip limit, but marks it 'a measurement only' because no command derives an int8 release for an application: the derivation exists only in benchmarks/refund/program/quantize_release.py (quantize_release_artifact, tolerance-gated verification of the quantized encoder chain against the float32 chain over the release's verification records; results in benchmarks/refund/data/release-int8-2026-09-24 and docs/scaling-results.md, where the measured int8 refund release changed 1 of 80 attested cases and 0.105% of decisions). Make it a first-class step: a CLI command (for example semantscript releases derive --int8 <release> or package --int8) that derives an int8 release from a passing release, verifies it on the quantized graph, publishes it as its own release with provenance (source manifest digest, quantization settings, tolerance, disagreement counts) and lets releases promote/rollback and the runtime treat it like any release. The trained Express release in the main checkout is available read-only for a real measurement; the fixture artifact (runtime/test/fixtures) is the CI evidence. No paid API calls are needed. Related pages: docs/deploy.md, docs/cli-reference.md (releases, package), examples/express-app/README.md size table.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A CLI command derives an int8 release from the current (or a named) release, verifies the quantized chain against the float32 chain over the release's verification records and, when the held-out gate exists, its held-out set, and publishes it as its own release whose manifest records the source digest, quantization settings, tolerance and disagreement figures; the runtime loads it and releases list/show/promote/rollback handle it
- [ ] #2 The derivation refuses to publish when the attested-case or decision disagreements exceed the tolerance (default strict: zero attested disagreements), and the report names the figures and the next command through the generated remedies
- [ ] #3 semantscript package's int8 lever becomes a real step (the report names the command), docs/deploy.md and the Express README give the derived release's measured size and whether it fits lambda-zip, the CI package job exercises the derivation on the fixture artifact, and tests cover the derivation, the refusal and the provenance
<!-- AC:END -->
