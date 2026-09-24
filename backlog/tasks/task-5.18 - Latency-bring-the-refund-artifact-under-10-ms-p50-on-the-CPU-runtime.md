---
id: TASK-5.18
title: 'Latency: bring the refund artifact under 10 ms p50 on the CPU runtime'
status: To Do
assignee: []
created_date: '2026-09-24 12:13'
updated_date: '2026-09-24 13:52'
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
- [ ] #1 The refund benchmark, re-run on the frozen final set under the committed protocol, reports p50 strictly below 10 ms for the SemantScript system
- [ ] #2 Attested accuracy on the final set stays at or above 0.99 and 15-bin ECE at or below 0.05 with the faster artifact
- [ ] #3 The re-measured predictions, result and updated written go/no-go are committed under benchmarks/refund/data with the new artifact digest and the release evidence that produced it
<!-- AC:END -->
