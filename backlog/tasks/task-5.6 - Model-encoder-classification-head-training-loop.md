---
id: TASK-5.6
title: 'Model: encoder + classification head training loop'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-20 19:18'
labels:
  - model
milestone: m-1
dependencies:
  - TASK-5.4
references:
  - docs/research/laya-analysis.md
parent_task_id: TASK-5
ordinal: 11000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Non-autoregressive by design: pretrained 100-300M encoder, one head whose logit count equals the output type cardinality, one forward pass. Transcript turn 9 points 3 and 10.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fine-tunes a pretrained encoder plus a categorical head from the generated dataset
- [ ] #2 Head shape is derived from the IR output spec, not hard-coded
- [ ] #3 Training run reports accuracy on a held-out split
- [ ] #4 Base encoder choice is recorded in the decision log
- [ ] #5 Heads are trained with a strictly proper scoring rule loss (log + spherical score), with ranked probability score added for ordinal heads; plain cross-entropy is available as a baseline flag
- [ ] #6 Head architecture is a linear or small MLP head per function; a heavier head is only added if the benchmark demands it
<!-- AC:END -->
