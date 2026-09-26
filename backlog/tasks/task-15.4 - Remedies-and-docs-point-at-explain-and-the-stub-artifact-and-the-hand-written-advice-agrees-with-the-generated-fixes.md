---
id: TASK-15.4
title: >-
  Remedies and docs point at explain and the stub artifact, and the hand-written
  advice agrees with the generated fixes
status: To Do
assignee: []
created_date: '2026-09-26 06:27'
labels:
  - dx
milestone: m-6
dependencies: []
parent_task_id: TASK-15
priority: medium
ordinal: 69000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-14.11 (diagnostics/remedies.json, scripts/generate-remedies.mjs, generated remedies in runtime/src, compiler/src and trainer/src/semantscript_trainer, tables in docs/diagnostics.md) was written while TASK-14.7 (semantscript explain) and TASK-14.8 (@semantscript/core/testing stub artifact) were on other branches, so its remedies do not name them. Its reviewers also found hand-written advice that disagrees with the generated next: lines: the seed-retry stop reason (verification.py / cli.py) carries its own advice ('check the example against the teacher's labels'), and docs/diagnostics.md has hand-written prose for VerificationConfigurationError, TrainingConfigurationError and ArtifactExportError that differs from the generic remedies. The refund tutorial (docs/tutorial-refund-decision.md), docs/build-cache.md and the test docs do not mention explain where wrong answers are discussed. init's .gitignore does not cover .semantscript/artifact.report.traceback.txt, and the trainer-killed remedy suggests --batch-size and --device cpu for --estimate, where they do not apply.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The remedies gold-check-example, test-example-mismatch, unknown-function, runtime-not-loaded, artifact-missing and editor-no-artifact name semantscript explain or @semantscript/core/testing where that is the next step, the generated outputs are regenerated and node scripts/generate-remedies.mjs --check passes
- [ ] #2 The seed-retry stop reason and the hand-written diagnostics prose for VerificationConfigurationError, TrainingConfigurationError and ArtifactExportError no longer contradict the generated fixes (either generated or aligned), the estimate-specific trainer-killed remedy drops the training-only flags, and init's .gitignore covers the traceback file
- [ ] #3 docs/tutorial-refund-decision.md, docs/build-cache.md and the framework testing docs point at explain and the stub artifact where wrong answers or untrained tests are discussed; prettier and the full test suites pass
<!-- AC:END -->
