---
id: TASK-7.1
title: 'CLI: semantscript build | train | test | run'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-24 17:45'
labels:
  - cli
milestone: m-3
dependencies:
  - TASK-6.3
parent_task_id: TASK-7
ordinal: 24000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Single entry point wrapping compiler, trainer, verifier and runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 build compiles .sem.ts and emits JS + IR bundle
- [x] #2 train produces the artifact from the IR bundle
- [x] #3 test runs verification and reports per-function stats
- [x] #4 run executes a program with the runtime loaded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Findings: the CLI package is an empty stub; the compiler exposes compileSemantScriptProgram(program, {projectRoot, encoderRef, adapterRef, bundlePath}); the trainer has every stage as a library call (SyntheticDatasetGenerator, AdversarialDatasetGenerator, train_classifier / train_application, evaluate_training_result, build_verified_ir, export_multi_function_artifact) but no generic bundle-to-artifact driver, only the refund benchmark's bespoke pipeline; the runtime activates one artifact per process for the compiled __sema calls. Trained PyTorch models are not persisted, so verification can only run inside train; test re-checks the shipped artifact.
1. Trainer driver: semantscript_trainer/cli.py with train_bundle(bundle, artifact_root, *, teacher, cache_directory, configs, application id/version, cases, optional injected tokenizer/encoder/weights digest for tests) that, per function of a source IR bundle, generates the synthetic dataset (and the adversarial sidecar when constraints exist), trains one classifier for a single function or the shared-encoder application for several, verifies every function (failing functions are reported and the build stops), binds verified IR with derived provenance counts, exports one content-addressed artifact with a training key over the dataset digests, and returns a per-function JSON report. python -m semantscript_trainer.cli train wires it to a teacher TOML, cache dir, training and verification flags, and writes the report.
2. Node CLI (cli/src): build parses a tsconfig (or a root directory), compiles with the application's encoder/adapter refs, prints diagnostics or a per-function summary and the bundle path; train spawns the Python driver (monorepo PYTHONPATH detected, --python override), streams its log and renders the report; test reads the artifact pointer and manifest, reports per-function verification stats (status, accuracy, ECE, Brier, pair consistency, attested cases, constraint violations, per-head accuracy), loads the artifact and, given the bundle, replays every IR example through the runtime, exiting nonzero on any failed function or example; run loads the artifact, imports the compiled module and optionally calls an export with JSON input, printing the JSON result.
3. Tests: Python test_cli.py builds the refund example bundle with the CLI, trains with a rule-based fake teacher and tiny tokenizer/encoder fakes, and asserts the artifact, report and multi-function path; Node cli.test.mjs covers build (temp project with core declarations, diagnostics exit), train (stub python recording argv, report rendering, exit propagation), test (fixture artifact with passing and failing examples) and run (module calling __sema through the loaded artifact). README for the CLI; lint and suites.
4. Real evidence: build examples/refund.sem.ts, train it with the local Ollama teacher and the pinned ModernBERT encoder, test and run the result on this machine; record the outcome in the task notes (the artifact itself stays out of git).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented per the plan: cli/src (build.ts, train.ts, test-command.ts, run.ts, manifest.ts, table.ts, io.ts, index.ts with runCli) and trainer/src/semantscript_trainer/cli.py (train_bundle + python -m semantscript_trainer.cli train). Design decisions: test reports the verification each function shipped with and replays bundle examples through the runtime, because trained PyTorch models are not persisted and verification is enforced inside train; run activates the artifact, imports the module and calls an export with JSON input, closing the artifact afterwards; train spawns the Python driver (monorepo PYTHONPATH detected, --python override, report JSON rendered), passing training/verification/adversarial flags through unchanged. Verification evidence: AC1 cli/test/cli.test.mjs builds a temp project (bundle with encoder.demo-app refs, __sema.call in the emitted JS, no prompt text) and the real run compiled examples/refund.sem.ts and two other programs; AC2 trainer/tests/test_cli.py trains the compiled refund example and a two-function bundle through train_bundle with a rule teacher and tiny encoder fakes into content-addressed artifacts (report, manifest, pointer, training key), and the real run below published release c67a2c70... from a bundle; AC3 test reports per-function status/accuracy/ECE/Brier/pair consistency/attested/violations/per-head accuracy and example replays, exercised on the runtime fixture artifact (2/2 and 1/2 replays, absent function, failed status) and on the real artifact (2/2); AC4 run calls fixture and real compiled modules with the artifact loaded (positional and file JSON input, promise results, non-callable export, nothing loaded afterwards). Suites: npm test -w cli 7 pass; pytest trainer/tests/test_cli.py 5 pass; full trainer/model/benchmark Python 501 passed; Node 68/99/7/83; npm run lint clean. Real end-to-end on this machine (scratchpad, not committed): (1) examples/refund.sem.ts with the Ollama teacher (qwen2.5:7b-instruct-q4_K_M and glm-4.7-flash) is refused by the strict synthetic generator because a teacher label violates a declared constraint (never fraudulent -> approve at case 21; the backend samples at temperature 0 so seeds do not change it); (2) an unconstrained risk function trained and verified numerically (held-out 1.0) but failed the gate on one gold example because the teacher labelled 46/46 synthetic cases high; (3) a review-tone function passed the whole chain: build, train (verification passed, release c67a2c70...), test 2/2 examples, run returns a value, in 90 s on the pinned ModernBERT encoder. Caveat recorded: that corpus was 63/64 negative with heavy sentence duplication, so the published model answers negative for every review; the strict gate does not detect a collapsed corpus when the held-out split shares the collapse. The trainer's Ollama backend (temperature 0, no duplicate avoidance) is not a usable teacher for real training; the benchmark's Claude CLI teacher (coverage-guided prompts, retry, duplicate avoidance, TASK-5.15) is, and a corpus-diversity gate in the trainer would be a sensible follow-up (not created). Machine change: the openai Python client (1.66.0) and pydantic were installed into .python-packages for the Ollama backend.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
semantscript now has its four commands. build parses a tsconfig, reports type and analysis diagnostics, compiles every sema site under the application's encoder/adapter refs and emits JS plus the IR bundle; train hands the bundle to the new Python driver (semantscript_trainer.cli.train_bundle), which generates synthetic and adversarial data through the configured teacher, trains one classifier or the shared-encoder application, verifies every function, binds verified IR and exports one artifact, then renders the per-function report; test reports each function's shipped verification and replays bundle examples through the runtime; run loads the artifact, imports the compiled module and calls an export with JSON input. Verified with 7 CLI tests (temp project builds, fixture-artifact test/run, stand-in trainer), 5 Python driver tests (real compiled bundles trained with a rule teacher into artifacts, multi-function path, failure path, flag mapping), full suites (Python 501, Node 68/99/7/83) and lint, plus a real run on this machine: a review-tone program went build -> train (Ollama teacher, pinned ModernBERT, verification passed, release c67a2c70...) -> test 2/2 -> run in 90 s. Recorded caveat: the local 7B teacher through the trainer's Ollama backend yields collapsed corpora (all one label, duplicated sentences) that the gate does not catch, and refuses the constrained refund example by violating a constraint, so real training needs the benchmark's Claude CLI teacher or a better backend. Commits 7d370bf, 6c9b3e5.
<!-- SECTION:FINAL_SUMMARY:END -->
