---
id: TASK-6.7
title: 'Experiment: compile-time routed domain adapters (static MoE)'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-20 20:10'
updated_date: '2026-09-24 23:29'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Probe: measure whether ONNX Runtime executes only the layers a fetched shallow output needs from one multi-output encoder graph (versus one prefix graph per depth); the result fixes the artifact layout for depth routing.
2. Model: SentenceEncoder embeds at a set of depths in one pass (hidden_states[d] through the encoder's final norm, full depth unchanged); SharedEncoderApplication holds several adapters keyed by ref, each with a depth, and maps functions to adapters; export writes the encoder once with one pooled output per used depth and one ONNX per adapter, with chained parity per function.
3. IR and compiler: a @domain(name) directive per site with the file stem as default (a class name when the site is inside a class); the execution plan records per function its domain, adapterRef and encoderDepth; build takes --domain-depth name=n (default full depth); schemas, IR.md, docs.
4. Trainer and artifact: train_bundle groups functions by domain, trains one adapter per domain at its depth jointly (an incremental head or a changed domain retrains only that adapter and its heads on the frozen encoder), verifies and exports N adapters plus depth outputs; manifest schema allows several adapter resources and records each function's adapter and encoder output; the runtime loader and worker fetch the named depth output per stage and apply the function's adapter (already per function).
5. Experiments: typed-decisions (four workflows as four domains): single-adapter vs routed-adapter artifacts on per-domain accuracy, pair consistency, p50 and artifact size; interference: retrain one domain and diff the others' heads and accuracy; refund: attested accuracy and CPU p50 at depths including full stack; decision-10 on the default layout and on the Phase 1 latency criterion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Probe (scratch, ModernBERT-base, 49 tokens, CPU ORT): fetching only the depth-4 output of one multi-output encoder graph costs 20.5 ms p50 versus 23.5 ms for the full 22 layers (ORT executes the graph whole), while a standalone 6-layer prefix graph runs in 5.3 ms and a 12-layer one in 15.0 ms with outputs bit-identical to the multi-output graph; so depth routing exports one prefix graph per depth. Implemented so far: model (SentenceEncoder depth embedding through final_norm, PrefixSentenceEncoder, SharedEncoderApplication with adapters by ref and per-adapter depth, export_routed_application_components), trainer (FunctionCorpus model refs, domain_depths, train_application per-domain adapters, add_function_head attaches a fresh adapter for a new domain on the frozen encoder, build cache v2 with encoder and per-adapter files and per-function combined digests plus a model-refs digest, train_bundle reuse and rehydration per domain, artifact export of N encoders and adapters with per-function encoderRef, manifest validation), runtime (StagedInferencePlan.encoders and per-function encoderRef, loader validation, worker encodes per (encoder, input)), compiler (@domain header, diagnostic 9126, default domain from the file name, domainDepths option, model.encoderDepth, plan domains and per-stage adapterRefs, CLI --domain-depth), schemas and docs. Tests: model depth-routing and routed export, runtime routed artifact, compiler domains, trainer end-to-end routed bundle through the Node runtime.

Mechanism committed (all Node and Python suites green). Experiments started: refund depth sweep (run_depth_sweep.py: depths 4, 6, 8, 12, 22 with the compact-release recipe on the frozen v4 corpus, attested release gate at 1 percent violation tolerance, routed artifact export, final-set accuracy and per-call latency through the Node runtime via run-final-set.mjs, and a raw CPU ONNX chain timing); the typed-decisions routed-domains comparison (run_routed_domains.py: single adapter vs four domain adapters, per-domain accuracy, gate metrics, pair consistency, CPU stage latency, artifact bytes, and an interference run adding one domain on the frozen encoder) follows on the GPU.

Refund depth sweep done (results-depth-sweep-2026-09-24): every depth (4, 6, 8, 12, 22) passes the strict gate with zero attested release misses; final set through the CPU Node runtime: depth 4 160/160 at 7.9 ms p50, depth 6 160/160 at 9.8 ms, depth 8 160/160 at 16.0 ms, depth 12 160/160 at 23.8 ms, full depth 159/160 at 29.3 ms; CPU ONNX chain p50 7.2 / 8.7 / 14.9 / 17.0 / 39.4 ms; GPU launch-bound at about 8 ms everywhere; artifact 237 MB at depth 4 vs 599 MB full. AC8's bar (p50 under 10 ms with attested accuracy within one point of full depth) is met at depths 4 and 6 in the direct runtime measurement; the committed-harness rerun belongs to TASK-5.18.
<!-- SECTION:NOTES:END -->
