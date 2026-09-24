---
id: TASK-5.14
title: >-
  Experiment: encoder size and initialization sweep (base/large/1B, ModernBERT
  vs laya-typed-decisions)
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-20 19:18'
updated_date: '2026-09-24 20:14'
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
- [x] #1 Both initializations are trained on the same refund-decision dataset with the same head and loss
- [x] #2 Accuracy, ECE, p50 latency in ONNX and artifact size are reported for both
- [x] #3 Decision-3 is updated with the result
- [ ] #4 Encoder size is swept across at least three points (~150M base, ~400M large, ~1B) with the same data, head and loss
- [ ] #5 For each size: accuracy, pair-consistency, ECE, p50 and p95 latency at batch 1 in ONNX on GPU and on CPU, and artifact size are reported
- [x] #6 Latency is also reported for 1, 10 and 50 heads sharing one encoder pass at each size, to show the amortization
- [x] #7 The write-up gives a per-application rule of thumb: which size fits a <10 ms request path, and which is appropriate for batch workloads
- [x] #8 Encoder size is exposed as a per-application config value with the benchmark-derived default
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Findings: the machine has one 16 GB GPU (12.8 GB free) and CPU-only ONNX Runtime; ModernBERT-base and the Laya typed-decisions checkpoint (a ModernBERT-large encoder inside an 808 MB RL agent, weights in model.safetensors) are cached; ModernBERT-large (395M, pinned 45bb4654) and DeBERTa-v2-xlarge (884M, pinned 1d134961) are downloadable; the trainer caps trainable parameters at 350M and keys the encoder by name and pinned revision in TrainingConfig, which the CLI exposes as --encoder-name/--encoder-revision.
1. Trainer: TrainingConfig.maximum_trainable_parameters (default 350M, hard ceiling 2B) so a deliberate sweep can train larger encoders; unit test.
2. prepare_laya_encoder.py extracts the Laya agent's ModernBERT-large encoder weights into a local Hugging Face directory (git-ignored under benchmarks/refund/data/encoders) with ModernBERT-large's tokenizer, so it loads through the same TrainingConfig path as any encoder.
3. run_encoder_sweep.py replays the frozen v4 corpus and, for each point (ModernBERT-base, Laya-initialised ModernBERT-large, ModernBERT-large, DeBERTa-v2-xlarge as the ~1B point), trains with the same recipe, head and proper loss, verifies against the 80 release-attested cases (accuracy, pair consistency, ECE), exports the ONNX chain (artifact size), measures batch-1 latency on CPU ONNX Runtime (p50/p95, plus one encoder pass with 1, 10 and 50 head passes) and on the GPU in PyTorch eager (ONNX Runtime has no GPU provider here, stated in the write-up). Out-of-memory at the ~1B point is recorded as such, with the smaller batch retried.
4. Results under benchmarks/refund/data/results-encoder-sweep-2026-09-24 with the tables, the rule of thumb (which size fits a <10 ms request path, which suits batch work), an update appended to decision-3, and the CLI README naming the encoder settings and the benchmark-derived default.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ran benchmarks/refund/program/run_encoder_sweep.py on the frozen v4 corpus with one recipe, linear head and proper loss for four points; results and write-up under data/results-encoder-sweep-2026-09-24, decision-3 updated (commit for results follows the driver commits f0d23e0, 9a072bc; trainer gained TrainingConfig.maximum_trainable_parameters, 5ee76c9; prepare_laya_encoder.py extracts the Laya agent's ModernBERT-large encoder, 394.8M parameters). AC1/AC2 evidence: ModernBERT-base vs the Laya-initialised ModernBERT-large on the same data, head and loss: calibrated accuracy 0.9904 vs 0.9936, ECE 0.0038 vs 0.0030, release-attested 80/80 for both, CPU ONNX p50 27.6 vs 93.7 ms, GPU (PyTorch eager, no ONNX GPU provider on this machine) 8.1 vs 13.5 ms, artifact ONNX 597 MB vs 1,580 MB. AC3: decision-3 carries a dated update (base stays default; the Laya-initialised large replaces DeBERTa-v3-base as the named alternative; encoder is a per-application setting). AC6: one encoder pass with 1/10/50 heads on CPU at base 41.5/47.2/40.4 ms stage p50 (0.81 ms per decision at 50), large ~88 ms (1.7 ms per decision); a head pass is 5 microseconds. AC7: the write-up's rule of thumb (GPU base for a <10 ms request path; base on CPU only with fused stages; large-from-Laya for 100 ms batch work). AC8: --encoder-name/--encoder-revision documented in cli/README.md with the benchmark-derived default. Not met: AC4 and AC5 for the ~1B point. DeBERTa-v2-xlarge (884.6M) diverged to non-finite logits within the first optimizer steps at batch 16 / lr 3e-5 and at batch 8 / lr 1e-5, although its forward and backward passes are finite on the GPU in eval and train mode, padded or not, through the trainer's own sentence encoder; the cause was not isolated. Its untrained latency (GPU 11.1 ms p50, CPU PyTorch 850 ms) and the failed ONNX parity export are recorded, so the ~1B row lacks accuracy, pair consistency and ECE. Also observed: pretrained ModernBERT-large reaches the best held-out accuracy (0.9957) but misses three attested window-boundary cases and fails the gate. Options for closing AC4/AC5 need the owner's call: a different ~1B encoder architecture that trains cleanly on this ROCm stack, or bf16/8-bit optimizer support in the trainer for the memory and stability of that size.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-09-23 19:05
---
Handoff readiness note: this story remains To Do and no sweep results are claimed. TASK-5.11 has already pinned and smoke-tested ModernBERT-base on AMD ROCm and pinned/smoke-tested Laya code d120d4ba220711b93c171973118753460310e16b plus checkpoint dd079950600224fb459af2a0cb1d74e1e57ee9cf; those identities and the benchmark measurement/provenance machinery can be reused. The required same refund training dataset is not frozen yet, and the large/~1B size points, multi-head amortization runs, CPU/GPU latency curve, configuration change, and Decision-3 update remain wholly unstarted.
---

author: @claude
created: 2026-09-24 20:14
---
Six of eight criteria are checked with evidence. The ~1B point (AC4, AC5) is open: DeBERTa-v2-xlarge would not train on this GPU stack (non-finite logits at two settings while its forward/backward are finite) and its ONNX export fails parity. To finish, pick either another ~1B encoder-only checkpoint to try or a trainer change (bf16 or 8-bit optimizer) and I will rerun that point.
---
<!-- COMMENTS:END -->
