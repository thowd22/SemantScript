---
id: TASK-6.7
title: 'Experiment: compile-time routed domain adapters (static MoE)'
status: To Do
assignee: []
created_date: '2026-09-20 20:10'
labels:
  - model
  - compiler
  - research
milestone: m-2
dependencies:
  - TASK-6.1
  - TASK-6.2
parent_task_id: TASK-6
ordinal: 46000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Encoder size trades latency for capability because batch-1 inference reads every weight. Sparse experts break that link, but learned token-level routing needs pretrained MoE encoders and far more data than one application has. Our architecture allows a cheaper hybrid: experts are LoRA-sized domain adapters over the shared encoder, grouped by controller, file or an explicit @domain tag, and the compiler selects the adapter statically in the execution plan. No router network, no load-balancing loss, deterministic, and dev-mode retrains only the adapter whose expressions changed. Capacity then scales with the number of domains in the application while per-request latency stays that of one adapter. Aligned with the north star: capacity is per application, organized by the app's own structure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The compiler groups sema expressions into domains by controller, file or an explicit @domain tag and records the adapter per stage in the execution plan
- [ ] #2 The trainer trains one adapter per domain on that domain's data, and the runtime applies the adapter named by the plan with no learned routing
- [ ] #3 On an application with at least three domains, single-adapter and routed-adapter artifacts are compared on per-domain accuracy, pair-consistency, p50 latency and artifact size
- [ ] #4 Interference is measured: adding or retraining one domain does not change the accuracy of the others beyond a stated tolerance
- [ ] #5 A backlog decision records whether routed adapters become the default artifact layout
<!-- AC:END -->
