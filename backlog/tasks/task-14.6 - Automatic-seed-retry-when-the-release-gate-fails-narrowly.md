---
id: TASK-14.6
title: Automatic seed retry when the release gate fails narrowly
status: Done
assignee:
  - '@claude'
created_date: '2026-09-25 15:21'
updated_date: '2026-09-26 02:34'
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
- [x] #1 When verification fails only on the constraint-violation rate or the ECE threshold by a bounded margin, train retrains with the next seed automatically, up to a configured number of attempts, reusing the cached datasets
- [x] #2 The report lists every attempt with its seed and gate metrics, and the published release records the seed that passed
- [x] #3 A gold-example miss or a failure outside the margin does not retry and says why
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design (researched 2026-09-25):
- Narrow failure = every gate failure of every failed function is the constraint-violation-rate gate or the ECE gate, with violation rate <= margin x tolerance and ECE <= margin x threshold (margin factor default 2, >= 1). A gold/human example miss, a type error, a rate above margin x tolerance (including any violation under the default zero tolerance) or ECE above margin x threshold never retries, and the report/log says which.
- Retries: seed, seed+1, ... up to --seed-attempts (default 3; 1 disables). Datasets are generated once before the loop (teacher never called again, meter unchanged). A seed past MAXIMUM_SPLIT_SEED stops with a reason.
- Retry settings live in a new SeedRetryConfig (not VerificationConfig) so they stay out of the cache recipe: the cache recipe keeps the configured seed, so an unchanged rebuild reuses the retried release; the effective seed is in each verified IR's trainingProvenance.seed.

Steps:
1. verification.py: add SeedRetryConfig(attempts, margin) with bounds and typed VerificationConfigurationError; add record_count (verification records the rate is measured over) to VerificationResult as an optional, validated field that is not serialized to IR/manifest/cache; add seed_retry_decision(results, config, retry) -> (retry: bool, reason: str) with structured per-gate classification (reuse the _gate_failures conditions, no string parsing). Export from package __init__/public API if the module lists exports (check test_public_api).
2. cli.py train_bundle: new seed_retry: SeedRetryConfig | None param. Wrap training+verification in an attempt loop with attempt_config = dataclasses.replace(resolved_training, seed=base+k). Joint path: deepcopy an injected encoder before attempt 1 so each attempt starts from pristine weights (HF path reloads). Incremental path: re-rehydrate the cached application before each attempt because add_function_head mutates the shared model in place. Provenance seed and TrainingResult.config come from the attempt config (lifecycle.build_verified_ir already asserts provenance.seed == training.config.seed). Log 'verification failed narrowly at seed N (...); retrying with seed N+1 (attempt k of K)' / 'not retrying: ...'.
3. _rehydrate_application: rehydrate each reused function with the seed recorded in its verified IR provenance (so its split matches what it trained on), not the configured seed.
4. Report (reportVersion stays 1, additive like teacher.spend): top-level 'seed' (the seed of the published release, null when failed), 'attempts': [{attempt, seed, status, functions:[{id, status, accuracy, ece, constraintViolations, records, violationRate, failures}]}], 'retry': {attempts, margin, stopReason|null}; per function training.seed. TrainBundleFailure message names the stop reason. Cached/reused report path: seed from provenance, attempts [].
5. argparse: --seed-attempts (int), --seed-retry-margin (float); map into SeedRetryConfig in main. Node CLI cli/src/train.ts: add both to PASSTHROUGH_STRING; renderTrainReport prints an attempts table (attempt, seed, status, per-function violations/rate/ECE) when more than one attempt ran, the passing seed line and the stop reason.
6. Tests: trainer/tests/test_verification.py (SeedRetryConfig bounds; decision: narrow violation, narrow ECE, gold miss, type error, rate beyond margin, zero tolerance, ECE beyond margin). trainer/tests/test_cli.py: monkeypatch cli_module.evaluate_training_result to fail narrowly by seed -> retry publishes, report lists attempts, verified IR provenance seed == passing seed, datasets generated once (CountingTeacher requests unchanged); out-of-margin and gold-miss (ContradictingTeacher) raise without retry and name why; attempts exhausted; attempts=1 disables; rebuild after retry reuses cache without training; incremental-path retry. test_main flag mapping. cli/test/cli.test.mjs + cli/test/fixtures/fake_trainer.py: passthrough of the two flags and rendering of attempts.
7. Docs: docs/cli-reference.md (two flags in the train table + retry paragraph), docs/build-cache.md (retries reuse cached datasets, cost GPU only, recipe keeps configured seed, effective seed in provenance), docs/diagnostics.md (retry/no-retry log lines and report fields; update the constraint/ECE fix columns), cli/README.md flag list, trainer/README.md report paragraph, examples/express-app/README.md (seed lottery now automatic). npx prettier --check docs README.md + touched READMEs.
8. GPU demonstration without API spend: copy the express example's cached datasets/adversarial-datasets/teacher-prices into a scratch cache dir, run semantscript train (examples/express-app bundle) with --teacher .semantscript/teacher.toml, ANTHROPIC_API_KEY from .env only for cache resolution, --max-cost-usd 0.05, --cases 192 --epochs 8 --select-best-epoch --counterfactual-ratio 0.5 --max-constraint-violation-rate 0.01 --device cuda, scratch --artifact: (a) --seed 1 --seed-attempts 5 --seed-retry-margin 3 expected to retry 1..4 and publish seed 5 (prior manual runs: 4,6,5,8 violations of 394, seed 5 passed); (b) --seed 1 default margin 2 expected to stop at seed 4 (8/394 = 2.03% > 2%) naming the margin. Confirm spend USD 0 and published verified IR seed.
9. npm run build, npm run lint:node, npm run test:node, trainer pytest, ruff check/format; commit, push, gh run watch.

Risks: GPU nondeterminism may not reproduce earlier per-seed numbers exactly (demo outcome may differ; report what happened). Default attempts 3 triples worst-case GPU time on a failing build. Retries hide seed sensitivity; the report keeps every attempt visible.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
IMPLEMENT: verification.py gains SeedRetryConfig (attempts 1-20, default 3; margin 1-10, default 2), SeedRetryDecision and seed_retry_decision (structured per-gate classification), and a non-serialized, non-compared VerificationResult.record_count. cli.py train_bundle wraps training+verification in an attempt loop (seed, seed+1, ...), datasets generated once, pristine copy of an injected encoder per joint attempt, cached application restored again per incremental attempt, provenance seed from the attempt's training config, report fields seed/attempts/retry (reportVersion 1, additive) and per-function training.seed; _rehydrate_application restores each reused function on the seed its verified IR records. Flags --seed-attempts/--seed-retry-margin (Python and Node passthrough); renderTrainReport prints an attempts table, the published seed and the stop reason. Tests: trainer/tests/test_seed_retry.py (new), test_verification.py and test_cli.py additions, cli.test.mjs + fake_trainer.py.

Docs: docs/cli-reference.md (flags + '### Seed retry'), docs/build-cache.md ('## Seed retries', recipe keeps configured seed), docs/diagnostics.md (retry/no-retry lines, updated fix columns), cli/README.md, trainer/README.md, examples/express-app/README.md; prettier --check clean.
GPU demo 2026-09-25 (RX 9070 XT, scratch copy of the Express reduced-prompt dataset cache, --max-cost-usd 0.05, report teacher.spend 0 requests / USD 0 in every run): --seed 1 --seed-attempts 5 --seed-retry-margin 3 passed first time (1 of 394; GPU nondeterminism, it failed at 4 the day before). --seed 2 --seed-attempts 5 --seed-retry-margin 3: seed 2 failed 6/394 (1.52%), seed 3 5/394 (1.27%), both 'failed narrowly ... retrying'; seed 4 passed 2/394 and published release 6723f11c...; report seed 4, attempts [2,3,4]; both cached verified IRs trainingProvenance.seed 4; rerun with the same flags: status reused, seed 4, no training. --seed 2 --seed-retry-margin 1.2 --full: 'not retrying: ... violation rate 1.5228% (6 of 394) is outside the retry margin 1.2 x 0.01 = 1.2000%', exit 1, one attempt, nothing published.
Checks: npm run build OK; npm run lint:node OK; npm run lint:python OK; npm run test:node all pass; trainer pytest 508 passed 2 skipped.

FIX round 1 (review: AC2 said the published release must record the passing seed; the manifest had none). The manifest's functions[].trainingProvenance now carries seed. artifact._training_provenance copies it from the verified IR and refuses a mismatch with training.config.seed. The runtime loader accepts seed as an optional non-negative integer (not required to be a safe integer, because seeds go up to 2^63-1), so releases from before this change still load. Also updated: artifact-types.ts, schemas/application-artifact.v1.schema.json and the example manifest. semantscript test shows a seed column (- / null for older releases). Advisory fixes: the attempts table gains an accuracy column; --seed-attempts/--seed-retry-margin are checked at the top of main before the bundle, and for --estimate too, with errors naming the flag; diagnostics lists the three missing stop reasons; the docs cover the split draw, the accuracy tradeoff, dev cycles, components SeedRetryConfig, getting-started and architecture; the cli/README flag list is rewrapped. The test_object_outputs fixture now uses result.config.seed. GPU (Express, --seed 2 --seed-attempts 5 --seed-retry-margin 3 --full, 0 requests, USD 0): seeds 2 (6/394) and 3 (5/394) retried, and seed 4 published release fd330c79. manifest.json trainingProvenance.seed=4 for both functions; semantscript test --bundle shows seed 4 for both and passes 3/3. An older seedless release still reads (seed -). Checks: npm run build OK, lint:node OK, lint:python OK, test:node 89/106/23/8/85 pass 0 fail, trainer pytest 508 passed 2 skipped, prettier clean.

FINALIZE validation 2026-09-26: npm run build exit 0; lint:node and lint:python clean; prettier --check docs/READMEs clean; test:node 89/106/23/8/85 pass 0 fail; trainer pytest 508 passed 2 skipped; 14 retry tests pass (test_seed_retry.py 8, test_cli.py retry/no-retry/gold-miss/type-error/incremental 6). GPU evidence (scratchpad demo, 0 teacher requests, USD 0): fix1-s2.json attempts seeds 2 (6/394 failed), 3 (5/394 failed), 4 (2/394 passed), report seed 4, release fd330c79 manifest trainingProvenance.seed 4 for both functions; margin-s2.json one attempt, stopReason 'violation rate 1.5228% (6 of 394) is outside the retry margin 1.2 x 0.01', nothing published. AC1 (narrow retry on cached datasets), AC2 (attempts listed with seed and metrics; manifest seed = passing seed), AC3 (gold miss / type error / outside-margin do not retry and name why) checked on this evidence.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
semantscript train now retries with the next seed when the release gate fails only on the constraint-violation rate or ECE within a bounded margin (--seed-attempts, default 3, 1 disables; --seed-retry-margin, default 2 x tolerance/threshold), reusing the cached datasets so a retry costs GPU time only. Gold misses, type errors and failures outside the margin stop at once with the reason. The train report lists every attempt (seed, status, accuracy, ECE, violations/records) plus seed and retry.stopReason; the verified IR and now the release manifest's trainingProvenance.seed record the seed that passed, and semantscript test shows it. Documented in cli-reference, build-cache, diagnostics, components, architecture, getting-started and package READMEs. Verified with build, lint, node tests, 508 trainer tests (14 retry-specific) and a zero-spend GPU run on the Express example (seeds 2 and 3 retried, seed 4 published with manifest seed 4; margin 1.2 run stopped naming the margin).
<!-- SECTION:FINAL_SUMMARY:END -->
