---
id: TASK-6.7
title: 'Experiment: compile-time routed domain adapters (static MoE)'
status: To Do
assignee: []
created_date: '2026-09-20 20:10'
updated_date: '2026-09-24 12:13'
labels:
  - model
  - compiler
  - research
  - performance
milestone: m-2
dependencies:
  - TASK-6.1
  - TASK-6.2
references:
  - benchmarks/refund/data/results-v2-2026-09-23/README.md
parent_task_id: TASK-6
ordinal: 46000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Encoder size trades latency for capability because batch-1 inference reads every weight. Sparse experts break that link, but learned token-level routing needs pretrained MoE encoders and far more data than one application has. Our architecture allows a cheaper hybrid: experts are LoRA-sized domain adapters over the shared encoder, grouped by controller, file or an explicit @domain tag, and the compiler selects the adapter statically in the execution plan. No router network, no load-balancing loss, deterministic, and dev-mode retrains only the adapter whose expressions changed. Capacity then scales with the number of domains in the application while per-request latency stays that of one adapter. Aligned with the north star: capacity is per application, organized by the structure of the app itself.

Amended 2026-09-24 with depth routing. The Phase 1 refund benchmark (benchmarks/refund/data/results-v2-2026-09-23/README.md) reached 0.994 attested accuracy but p50 38.4 ms on the CPU Node runtime against the 10 ms exit bar, and the cost is the full 22-layer shared-encoder pass, which adapter routing alone leaves untouched. Measured on the benchmark machine (Ryzen 9 9900X, batch 1, 110 tokens): PyTorch fp32 60 ms at 22 layers, 33 ms at 12, 16.7 ms at 6, 11.9 ms at 4; the exported ONNX encoder 35.5 ms at 110 tokens and 18.1 ms at 32. So the static map must route the compute path as well as the adapter: for each domain the compiler records a depth (how many shared-encoder layers run before that domain adapter and its heads), chosen from measured accuracy at each depth; simple domains such as the refund policy run a shallow prefix and only domains that need it run the full stack. Prefixes stay shared, so one encoder still serves the whole application. Compact input encoding and int8 export are tracked separately under the Phase 1 latency task and compose with depth routing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The compiler groups sema expressions into domains by controller, file or an explicit @domain tag and records the adapter per stage in the execution plan
- [ ] #2 The trainer trains one adapter per domain on that domain's data, and the runtime applies the adapter named by the plan with no learned routing
- [ ] #3 On an application with at least three domains, single-adapter and routed-adapter artifacts are compared on per-domain accuracy, pair-consistency, p50 latency and artifact size
- [ ] #4 Interference is measured: adding or retraining one domain does not change the accuracy of the others beyond a stated tolerance
- [ ] #5 A backlog decision records whether routed adapters become the default artifact layout
- [ ] #6 The execution plan records a depth per domain alongside its adapter, and the runtime runs only that prefix of the shared encoder for the stages of that domain
- [ ] #7 For the refund domain, attested final-set accuracy and CPU-runtime p50 latency are reported at three or more depths including the full stack
- [ ] #8 A depth-routed refund artifact meets the Phase 1 latency criterion (p50 below 10 ms on the CPU Node runtime) with attested accuracy within one percentage point of the full-depth artifact, or the decision in criterion 5 records why not
<!-- AC:END -->
