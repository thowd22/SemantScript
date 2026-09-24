---
id: decision-9
title: >-
  Compile-time reduction: fine-tune the application encoder, do not fit heads on
  a universal frozen encoder
date: "2026-09-24 19:17"
status: accepted
---

## Context

TASK-7.3 asked whether a universal semantic encoder could be trained once so that compiling a function only fits a tiny head, dropping compile time from minutes to seconds, and whether a Laya-style option scorer could warm-start such a head. The experiment (`benchmarks/refund/data/results-warm-start-2026-09-24`) measured full fine-tuning, head-only training on the frozen ModernBERT-base encoder (through the trainer and on cached embeddings) and a Laya-distilled warm start on the frozen refund corpus against the 80 judge-attested release cases. Full fine-tuning reaches zero attested misses after one epoch (58 s on the RX 9070 XT). Head-only never does: it plateaus near 0.89 held-out and 0.78 to 0.89 attested accuracy with constraint-violation rates far above the gate, even though the head fit itself takes 8 s on cached embeddings. The Laya scorer answers the refund question at 0.225, the distilled head reproduces that, and refining from it tracks the random-init head-only curve.

## Decision

- Compile time is reduced by fine-tuning the application's own encoder for fewer epochs (one or two epochs already pass the attested criterion) and by the build cache, which skips retraining for unchanged expressions and trains only a changed expression's head on the application's already fine-tuned encoder. A universal frozen encoder with per-function heads is not adopted as a compile path.
- The Laya-style warm start (universal option scorer distilled into a fixed head) is not adopted; it carries no signal for this task. Its encoder weights may still be evaluated as an initialisation for full fine-tuning under TASK-5.14, a separate question.
- This is strictly a compile-time matter for per-application artifacts. Shipped artifacts remain the application's encoder, adapter and heads; no runtime depends on a universal model.

## Consequences

- `TrainingConfig.freeze_encoder` exists for experiments and for head-only regimes on an application's own encoder, not as a default.
- Future compile-time work targets the fine-tuning epoch (smaller corpora, earlier stopping under the gate, smaller or distilled encoders per TASK-6.7 and TASK-5.14), not head-only training on pretrained embeddings.
