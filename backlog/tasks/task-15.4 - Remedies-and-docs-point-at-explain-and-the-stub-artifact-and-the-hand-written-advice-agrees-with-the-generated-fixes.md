---
id: TASK-15.4
title: >-
  Remedies and docs point at explain and the stub artifact, and the hand-written
  advice agrees with the generated fixes
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-26 06:27'
updated_date: '2026-09-26 06:41'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. diagnostics/remedies.json (edit only the named entries, in place): gold-check-example -> keep its placeholders, name semantscript explain on a call with the example's inputs (nearest teacher-labelled cases and active constraints show whether the example or the labels are wrong) before correcting/adding examples; test-example-mismatch -> after rebuild+retrain, run semantscript explain --call <export> with that example's inputs, then the train report's next: lines; unknown-function -> keep the build/train/reload fix, add that semantscript explain lists the artifact functions the bundle no longer has; runtime-not-loaded, artifact-missing, editor-no-artifact -> keep prefixes pinned by tests (loadSemaArtifact(), 'run semantscript train to publish an artifact at'), add 'in a test, load a stub with loadSemaStubArtifact() from @semantscript/core/testing (no training)'. No backticks or | in fix texts (generator rule).
2. trainer-killed for --estimate: add one new trainer-process remedy (e.g. estimate-killed) placed directly after trainer-killed (smallest merge surface vs 15.1/15.3), text without --batch-size/--device: run {doctor} and fix its checks, rerun semantscript train --estimate (it trains nothing and sends no request), report a bug if it is killed again. cli/src/train.ts estimate branch (line ~202) uses it; cli/test/cli.test.mjs estimate assertion (~984) pins the full line and asserts no --batch-size.
3. node scripts/generate-remedies.mjs, then --check; regenerates runtime/src + compiler/src remedies.generated.ts, trainer remedies_generated.py and the docs/diagnostics.md tables.
4. Seed-retry stop reason: trainer/src/semantscript_trainer/verification.py:227 drop the parenthetical '(check the example against the teacher's labels and the constraints)' so the gold stop reason only states why it does not retry and the failure's generated next: line is the one advice (cli.py:551 composes '; not retrying: <reason>; next: <first suggestion>'). Keep 'a gold miss is not a seed effect' (pinned by trainer/tests/test_cli.py:1512,1538 and test_seed_retry.py:127); add an assertion that the reason carries no advice of its own. Update docs/diagnostics.md seed-retry table row if its wording changes.
5. docs/diagnostics.md prose (align, no new remedies): VerificationConfigurationError paragraph -> say the no-gold case is caught before training with train-no-gold-examples and the rest end with the train-failed fix (the message names the setting: evaluation ratio, disjoint external cases), internal contract errors being bugs; ArtifactExportError/ArtifactPublicationError -> end with artifact-export-failed (train-path-unwritable when an OSError is the cause, per failures.py); TrainingConfigurationError/TrainingExecutionError -> train-out-of-memory for memory, otherwise train-failed naming the setting. Update the verifier bullet 'Otherwise it asks to check the example against the teacher's labels' to match the new gold-check-example; mention the estimate-specific killed fix in the trainer-process prose.
6. init .gitignore: cli/src/init.ts RESERVED_OUTPUTS add '*.traceback.txt' (covers .semantscript/artifact.report.traceback.txt and a custom --report beside it) and the header comment; cli/test/cli.test.mjs init tests (~1038, ~1064) expect the new line, including the append-to-old-project case; docs/cli-reference.md (init, ~line 91), docs/getting-started.md (~117), docs/build-cache.md (~60) list the traceback entry.
7. Docs AC3: docs/tutorial-refund-decision.md step 4/5 -> when test fails or an answer is wrong, npx semantscript explain dist/refunds.sem.js --call decideRefund --input ... and the diagnostics wrong-answer workflow; tests without training use @semantscript/core/testing. docs/build-cache.md -> explain reads the release's cached dataset (datasets/v1 by trainingProvenance.datasetSha256), so clearing the cache or --no-cache limits explain; after a wrong answer, explain -> add example/constraint -> incremental retrain. docs/framework-guide.md Testing handlers + framework/README.md (+ runtime/README.md testing section if it discusses failures) -> stub for untrained tests, semantscript explain when a test over the trained artifact gets a wrong answer.
8. Verify: npm run build; node scripts/generate-remedies.mjs --check; npm run lint:node; npm run test:node (runtime/compiler/cli); trainer pytest (tests/test_remedies.py, test_cli.py, test_seed_retry.py, test_verification.py, failures tests) and ruff check/format; npx prettier --check docs README.md framework/README.md runtime/README.md. Commit on task-15.4, push, gh run watch CI.
Risks: remedies.json merge with 15.1/15.3 (edit named entries in place, one insertion next to trainer-killed); generated docs table column widths change for whole family tables (regenerate after merge resolves it); explain needs a module and export the trainer cannot know, so train-side fixes name the command generically.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
IMPLEMENT: remedies.json edits (gold-check-example, test-example-mismatch, unknown-function name semantscript explain; runtime-not-loaded, artifact-missing, editor-no-artifact name loadSemaStubArtifact() from @semantscript/core/testing); new estimate-killed trainer-process remedy (no --batch-size/--device) used by cli/src/train.ts --estimate branch; regenerated outputs (generate-remedies --check exit 0). Seed-retry gold stop reason now ends 'a gold miss is not a seed effect' with no advice (tests in test_seed_retry.py and test_cli.py). docs/diagnostics.md prose for VerificationConfigurationError, ArtifactExportError/ArtifactPublicationError, TrainingConfigurationError/TrainingExecutionError aligned with train-no-gold-examples, train-failed, artifact-export-failed, train-path-unwritable, train-out-of-memory; estimate-killed prose. init reserves *.traceback.txt (verified with git check-ignore in a scratch project). explain/stub pointers in tutorial steps 4-5, build-cache.md, framework-guide.md Testing handlers, framework/README.md; reserved-entry lists in cli-reference, getting-started, build-cache, cli/README. Local: npm run build ok, lint:node ok, test:node 369/369 pass, trainer pytest (remedies, failures, seed_retry, verification, cli) 70 passed, ruff clean, prettier --check clean.

Committed and pushed task-15.4 (9cc92a8); CI run 36224327946 green on all 7 jobs (Node lint/build/tests, Python lint and tests dev and dev,training, fresh clone, package, doctor macOS/Windows).
<!-- SECTION:NOTES:END -->
