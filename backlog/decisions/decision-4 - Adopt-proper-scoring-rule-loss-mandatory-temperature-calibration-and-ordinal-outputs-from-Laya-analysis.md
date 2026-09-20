---
id: decision-4
title: >-
  Adopt proper-scoring-rule loss, mandatory temperature calibration, and ordinal
  outputs (from Laya analysis)
date: '2026-09-20 19:18'
status: accepted
---
## Context

Laya (Apache 2.0, non-autoregressive ModernBERT decision engine) published code and benchmarks that bear directly on our Phase 1 design. Full analysis: docs/research/laya-analysis.md. Key evidence: ECE 0.466 as shipped vs 0.081 after temperature fitting; ordinal predictions are their weakest primitive; zero-shot decision encoders score near chance; teacher-generated eval sets overstated real-traffic accuracy.

## Decision

- Train heads with a strictly proper scoring rule loss (log score + spherical score; ranked probability score added for ordinal outputs) instead of plain cross-entropy.
- Fit a per-function temperature on a calibration split as a mandatory build step; report ECE and Brier per function in the artifact; fail the build above a configured ECE threshold.
- Add an ordinal output kind to the spec (ordered string-literal unions, bounded integers) with distribution + expected value at runtime.
- Define confidence as calibrated top-1 probability and expose normalized entropy as uncertainty.
- Require canonical, deterministic input serialization in the IR.
- Require every benchmark held-out set to contain real, human-authored cases, not only teacher-generated ones.
- Add Laya as a Phase 1 baseline and its typed-decisions benchmark as an external Phase 2 test.

## Consequences

- TASK-1, TASK-2, TASK-5.6, TASK-5.7, TASK-5.10, TASK-5.11 acceptance criteria updated; new stories for encoder-init comparison and the external benchmark.
- Slightly more trainer complexity in exchange for calibrated probabilities that @confidence can rely on.
- Not adopted: runtime text-encoded options, GRPO training, multilingual routing.


