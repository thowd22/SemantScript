---
id: TASK-5.18
title: 'Latency: bring the refund artifact under 10 ms p50 on the CPU runtime'
status: To Do
assignee: []
created_date: '2026-09-24 12:13'
updated_date: '2026-09-24 12:13'
labels:
  - runtime
  - model
  - benchmark
  - performance
milestone: m-1
dependencies:
  - TASK-5.18.1
  - TASK-5.18.2
references:
  - benchmarks/refund/data/results-v2-2026-09-23/README.md
parent_task_id: TASK-5
ordinal: 50000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Phase 1 exit benchmark (benchmarks/refund/data/results-v2-2026-09-23/README.md, TASK-5.11) records accuracy met (0.994 on 160 judge-attested real inputs against 0.538 for the 7B comparator) but latency not met: the exported artifact measures p50 38.4 ms and p95 50.5 ms through onnxruntime-node on CPU, against the exit bar of p50 strictly below 10 ms. The cost is arithmetic, not configuration: ModernBERT-base runs 22 layers in fp32 over roughly 110 tokens of canonical input per call. Two levers are independent of the encoder architecture and compose with depth routing (TASK-6.7): the canonical input encoding spends about 110 tokens on five fields (nested arrays plus hex-encoded binary64 doubles), and the exported ONNX encoder drops from 35.5 ms at 110 tokens to 18.1 ms at 32 tokens; int8 dynamic quantization of the encoder Linear layers measured 2.3x on this CPU (60 ms to 26 ms in PyTorch at full depth). Together they measured 3.5 ms at 6 layers and 32 tokens in PyTorch, and about 10 ms at full precision. This task owns the end-to-end outcome: re-measure the frozen final set under the committed protocol with an artifact that carries both changes and update the written go/no-go. Constraints: the canonical serialization is a shared Python and TypeScript contract bound by digests, so an encoding change needs a version in the IR and artifact and a retrained, re-verified release; quantization must be verified on the quantized graph so calibrated confidences stay honest. Steady-state re-measurement outside the harness reproduced the harness figure (36.6 ms), so the harness is not the cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The refund benchmark, re-run on the frozen final set under the committed protocol, reports p50 strictly below 10 ms for the SemantScript system
- [ ] #2 Attested accuracy on the final set stays at or above 0.99 and 15-bin ECE at or below 0.05 with the faster artifact
- [ ] #3 The re-measured predictions, result and updated written go/no-go are committed under benchmarks/refund/data with the new artifact digest and the release evidence that produced it
<!-- AC:END -->
