---
id: TASK-5.18
title: 'Latency: bring the refund artifact under 10 ms p50 on the CPU runtime'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-24 12:13'
updated_date: '2026-09-25 01:28'
labels:
  - runtime
  - model
  - benchmark
  - performance
milestone: m-1
dependencies:
  - TASK-5.18.1
  - TASK-6.7
references:
  - benchmarks/refund/data/results-v2-2026-09-23/README.md
parent_task_id: TASK-5
ordinal: 50000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Phase 1 exit benchmark (benchmarks/refund/data/results-v2-2026-09-23/README.md, TASK-5.11) records accuracy met (0.994 on 160 judge-attested real inputs against 0.538 for the 7B comparator) but latency not met: the exported artifact measures p50 38.4 ms and p95 50.5 ms through onnxruntime-node on CPU, against the exit bar of p50 strictly below 10 ms. The cost is arithmetic, not configuration: ModernBERT-base runs 22 layers in fp32 over roughly 110 tokens of canonical input per call, and a steady-state re-measurement outside the harness reproduced the figure (36.6 ms).

Path decided 2026-09-24 after measuring the levers on the committed release. Int8 dynamic quantization (TASK-5.18.2, done) is a measured dead end for this model: 1.38x at the median (27.7 ms), about one decision change per thousand records, and a confidently wrong attested release decision that the zero-attested-miss policy refuses, so it is off the critical path and no int8 artifact enters the benchmark. The two exception-free levers are fewer tokens and fewer layers: the exported ONNX encoder drops from 35.5 ms at 110 tokens to 18.1 ms at 32 tokens (compact canonical encoding, TASK-5.18.1), and PyTorch fp32 drops from 60 ms at 22 layers to 16.7 ms at 6 layers at 110 tokens (compile-time depth routing, TASK-6.7); a 6-layer prefix at 32 tokens measured about 10 ms in the PyTorch proxy, and onnxruntime runs about 1.7x faster than that proxy. Tokens alone do not reach the bar, so this task depends on both. It owns the end-to-end outcome: re-measure the frozen final set under the committed protocol with a float32 artifact that carries the compact encoding and a domain depth, passing the strict release gate with zero attested misses, and update the written go/no-go. Constraint: the canonical serialization is a shared Python and TypeScript contract bound by digests, so an encoding change needs a version in the IR and artifact and a retrained, re-verified release.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The refund benchmark, re-run on the frozen final set under the committed protocol, reports p50 strictly below 10 ms for the SemantScript system
- [x] #2 Attested accuracy on the final set stays at or above 0.99 and 15-bin ECE at or below 0.05 with the faster artifact
- [x] #3 The re-measured predictions, result and updated written go/no-go are committed under benchmarks/refund/data with the new artifact digest and the release evidence that produced it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Release pipeline: add --encoder-depth N; when set, bind the compiled IR to one routed domain at that depth, train through train_application (one adapter over the shared encoder) instead of train_classifier, verify and export through the routed exporter; the pipeline manifest records the depth and the encoder prefix ref.
2. Run the pipeline at depth 6 (and depth 4 as the fallback point) with the committed compact recipe on the frozen v4 corpus under the strict gate.
3. Re-measure the frozen final set under the committed protocol with run-benchmark.mjs (semantscript system from the new pipeline manifest, same warm-up and environment capture), commit predictions, result and environment under benchmarks/refund/data.
4. Update the written go/no-go (results README) with the new artifact digest and release evidence; note the accuracy and ECE criteria.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-24: TASK-5.18.1 done. Compact encoding alone brings the refund release to p50 28.5 ms (from 38.4) with a first-seed strict-gate pass and 160/160 on the final set; the 22-layer encoder's per-layer overhead now dominates at batch 1, so the remaining path to p50 under 10 ms is depth routing (TASK-6.7).

Release pipeline gained --encoder-depth (routed IR, train_application, routed export; the pipeline manifest records encoderDepth and the encoder prefix ref). release-depth-006-2026-09-25 (seed 1, compact recipe, depth 6): strict gate passed first seed, zero attested misses, calibration accuracy 0.9957, ECE 0.0027, 5 violations of 9,489, artifact a3f4b005…, 320 s training. Harness run results-depth-006-2026-09-25 with run-benchmark.mjs (semantscript system, same warm-up protocol and environment capture): 160/160, final-set 15-bin ECE 0.0029, p50 4.82 ms, p95 6.41 ms, 194 requests/s, 816 MiB client RSS; result.json assembled (status incomplete: baselines not re-run, no API key), leakage audit 0 overlap. Written go/no-go amended in results-v2-2026-09-23/README.md.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The refund artifact now meets the Phase 1 latency criterion on the deployed CPU path: a float32 release with the compact encoding and depth routing at 6 of 22 encoder layers, trained by the release pipeline (new --encoder-depth option) under the strict gate with zero attested misses, measures p50 4.82 ms and p95 6.41 ms on the frozen final set under the committed benchmark harness, with accuracy 1.000 (160/160) and 15-bin ECE 0.0029, against 38.40 ms in the committed run. Release evidence, sealed predictions, the assembled result and the amended written go/no-go are committed under benchmarks/refund/data. The mechanical status remains incomplete because the structured-output API baseline still has no key on this machine.
<!-- SECTION:FINAL_SUMMARY:END -->
