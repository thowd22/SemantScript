---
id: TASK-5.14
title: 'Experiment: encoder initialization — ModernBERT-base vs laya-typed-decisions'
status: To Do
assignee: []
created_date: '2026-09-20 19:18'
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
Laya publishes an Apache-licensed ModernBERT checkpoint already fine-tuned for typed decisions with a proper-scoring-rule objective. It may be a better starting point than the raw pretrained encoder, or its 421M size may cost us the latency target. Decision-3 names ModernBERT-base; this experiment checks whether to revise it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both initializations are trained on the same refund-decision dataset with the same head and loss
- [ ] #2 Accuracy, ECE, p50 latency in ONNX and artifact size are reported for both
- [ ] #3 Decision-3 is updated with the result
<!-- AC:END -->
