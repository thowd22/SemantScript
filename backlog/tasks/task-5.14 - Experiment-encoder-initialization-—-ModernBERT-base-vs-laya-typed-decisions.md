---
id: TASK-5.14
title: >-
  Experiment: encoder size and initialization sweep (base/large/1B, ModernBERT
  vs laya-typed-decisions)
status: To Do
assignee: []
created_date: '2026-09-20 19:18'
updated_date: '2026-09-20 20:05'
labels:
  - model
  - research
milestone: m-1
dependencies:
  - TASK-5.6
references:
  - docs/research/laya-analysis.md
parent_task_id: TASK-5
ordinal: 39000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Laya publishes an Apache-licensed ModernBERT checkpoint already fine-tuned for typed decisions with a proper-scoring-rule objective. It may be a better starting point than the raw pretrained encoder, or its 421M size may cost us the latency target. Decision-3 names ModernBERT-base; this experiment checks whether to revise it. It also settles the size question with numbers: parallel heads amortize the encoder across decisions, but the encoder pass itself scales with depth and weight bytes, so size trades latency for capability. The rule per application should be 'the largest encoder that fits the latency budget', which needs a measured curve.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both initializations are trained on the same refund-decision dataset with the same head and loss
- [ ] #2 Accuracy, ECE, p50 latency in ONNX and artifact size are reported for both
- [ ] #3 Decision-3 is updated with the result
- [ ] #4 Encoder size is swept across at least three points (~150M base, ~400M large, ~1B) with the same data, head and loss
- [ ] #5 For each size: accuracy, pair-consistency, ECE, p50 and p95 latency at batch 1 in ONNX on GPU and on CPU, and artifact size are reported
- [ ] #6 Latency is also reported for 1, 10 and 50 heads sharing one encoder pass at each size, to show the amortization
- [ ] #7 The write-up gives a per-application rule of thumb: which size fits a <10 ms request path, and which is appropriate for batch workloads
- [ ] #8 Encoder size is exposed as a per-application config value with the benchmark-derived default
<!-- AC:END -->
