---
id: TASK-6.5
title: 'Benchmark: parallel-head and batch scaling'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-24 17:01'
labels:
  - benchmark
milestone: m-2
dependencies:
  - TASK-6.3
parent_task_id: TASK-6
ordinal: 22000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Quantify the shared-encoder thesis.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Latency and throughput are measured for 1, 10 and 50 heads per stage
- [x] #2 Batch scaling curve is recorded
- [x] #3 Results are committed under benchmarks/
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Derive a many-function artifact from the committed compact refund release without retraining: benchmarks/refund/program/derive_multihead_artifact.py clones the verified refund function N times over the same encoder and adapter (new function ids and head refs, hard-linked encoder/tokenizer/adapter, copied head graphs), writes a content-addressed manifest and pointer under data/release-multihead-2026-09-24/artifact (git-ignored like the other bundles), and records the derivation (source manifest digest, N, per-function head digest) next to it. The cloned heads share weights, which is fine for timing: the runtime cost of a head does not depend on its weights.
2. Measure with the fused runtime: benchmarks/refund/program/run-stage-scaling.mjs loads the derived artifact and, after warm-up, times (a) one stage of H in {1, 10, 50} heads over one input, fused (asserting encoder 1, adapter 1, head H passes) against H sequential single calls (unfused baseline), and (b) batch scaling: one stage of B in {1, 2, 4, 8, 16, 32, 64} distinct held-out inputs on one head, reporting stage latency p50/p95, per-decision latency and decisions per second, plus the encoder-only reference. Record node/onnxruntime versions, thread settings and machine facts alongside.
3. Tests: a Python test for the derivation on a small fake release (manifest cloning, hashes, pointer, refusal of non-scalar or multi-function sources) and a Node test that runs the measurement function against the runtime test fixture with extra functions (passes and result shape), so the scripts are exercised without the 571 MB bundle.
4. Commit results.json, environment.json and a README with the tables and what they say about the shared-encoder thesis under data/results-stage-scaling-2026-09-24, index it from benchmarks/refund/README.md, lint and suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Measured on a derived 50-function artifact (data/release-multihead-2026-09-24, manifest 68c28a7f..., built in 58 ms by derive_multihead_artifact.py from the compact release 56c5c5d6... with hard-linked encoder/tokenizer/adapter and byte-copied heads; the clones count head passes, not trained behaviour). run-stage-scaling.mjs, 5 warm-up rounds, 30 iterations per configuration, every fused stage asserted encoder 1 / adapter 1 / head N passes and results byte-equal to N single calls. AC1 evidence (results.json headsPerStage): fused p50 20.99 / 22.47 / 29.72 ms for 1 / 10 / 50 heads against 20.83 / 217.33 / 1132.86 ms for one call per head (9.7x, 38.1x); fused throughput 46.6 / 427.7 / 1640.9 decisions/s; per-decision p50 20.99 / 2.25 / 0.59 ms. AC2 evidence (results.json batchScaling): 1..64 distinct inputs on one head give stage p50 24.1 / 47.1 / 102.0 / 172.2 / 364.6 / 686.6 / 1272.5 ms, per-decision 20 to 25 ms flat and 40 to 49 decisions/s, because the worker runs the encoder once per distinct input (passes encoder B); a padded batched encoder run is the untested throughput lever and would be a separate task. AC3: README, results.json, environment.json committed under data/results-stage-scaling-2026-09-24 plus the derivation record and README under data/release-multihead-2026-09-24; the 572 MB bundle is git-ignored. Caveat recorded in the README: raw callStage timings from the caller's thread, not the run-benchmark.mjs go/no-go protocol, so the 21 ms single-head figure does not replace the 28.5 ms p50 of the compact run. Tests: test_derive_multihead_artifact.py (7) and a program.test.mjs case running measureStageScaling on the runtime fixture with an injected clock; npm test -w benchmarks/refund 83 pass, pytest benchmarks/refund/program 70 pass, npm run lint clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Quantified the shared-encoder thesis on the CPU runtime. A derived 50-function artifact (clones of the verified compact refund function over one encoder and adapter, built without retraining by derive_multihead_artifact.py) shows a fused stage of 1 / 10 / 50 heads at 21.0 / 22.5 / 29.7 ms p50 against 20.8 / 217 / 1,133 ms for one call per head: about 0.18 ms per extra head, 38x at fifty heads, 1,641 decisions per second. The batch curve over 1 to 64 distinct inputs is linear (20 to 25 ms per decision, 40 to 49 decisions/s) because the worker encodes each distinct input separately; a padded batched encoder run is the untested throughput lever. Results, environment and the derivation record are committed under benchmarks/refund/data (results-stage-scaling-2026-09-24, release-multihead-2026-09-24); scripts documented in program/README.md and covered by 7 Python tests and a fixture-artifact Node test. Verified with npm test -w benchmarks/refund (83), pytest benchmarks/refund/program (70) and npm run lint; commit 26c6907.
<!-- SECTION:FINAL_SUMMARY:END -->
