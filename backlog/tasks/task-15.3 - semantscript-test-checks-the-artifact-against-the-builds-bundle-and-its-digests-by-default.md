---
id: TASK-15.3
title: >-
  semantscript test checks the artifact against the build's bundle and its
  digests by default
status: To Do
assignee: []
created_date: '2026-09-26 06:27'
labels:
  - dx
  - quality
milestone: m-6
dependencies: []
parent_task_id: TASK-15
priority: medium
ordinal: 68000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-14.11's reviewers confirmed that semantscript test without --bundle passes a stale artifact (the program changed since training, so the compiled function ids are absent from the release) and passes a release whose manifest or resource fails its digest check, because test only reads the recorded verification (cli/src/test-command.ts, cli/src/manifest.ts readArtifactSummary). run and explain already locate the build's IR bundle (dist/semantscript.ir.v1.json from the tsconfig outDir; see cli/src/explain.ts and cli/src/defaults.ts), and releases promote verifies digests through the runtime's checkSemaArtifact (cli/src/releases.ts verifyRelease). The failure messages go through diagnostics/remedies.json and scripts/generate-remedies.mjs (docs/CONTRIBUTING.md#generated-remedies).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Without --bundle, test finds the build's bundle the way run and explain do, and a bundle function absent from the artifact (or an artifact function the bundle no longer has) fails test with exit 1 and the next command; --no-bundle keeps the artifact-only behaviour, and a missing build says to run semantscript build
- [ ] #2 A release whose pointer, manifest or resource fails its digest or symlink check fails test with the runtime's ArtifactLoadError code and remedy instead of passing
- [ ] #3 cli tests cover both failures and the flag; docs/cli-reference.md, cli/README.md and docs/diagnostics.md are updated with the regenerated remedies; the CI fresh-install job still passes
<!-- AC:END -->
