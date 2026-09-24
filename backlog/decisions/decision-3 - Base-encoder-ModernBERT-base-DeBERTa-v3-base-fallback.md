---
id: decision-3
title: "Base encoder: ModernBERT-base, DeBERTa-v3-base fallback"
date: "2026-09-20 02:42"
status: accepted
---

## Context

Phase 1 needs a pretrained 100-300M encoder to fine-tune with a classification head. It must be small enough to iterate quickly on a 16 GB GPU (or CPU), fast at inference in ONNX, and strong at classification.

## Decision

Start with **ModernBERT-base** (~149M): modern architecture, fast inference, 8k context for structured inputs. Keep **DeBERTa-v3-base** (~184M) as the fallback if ModernBERT underperforms on the refund-decision benchmark.

## Consequences

- Encoder choice is a config value in `model/`; the benchmark harness should be able to run both.
- Revisit once the Phase 3 universal-encoder investigation reports.

## Update 2026-09-24 (TASK-5.14 encoder sweep)

Measured on the refund function with one recipe, head and proper loss
(`benchmarks/refund/data/results-encoder-sweep-2026-09-24`): ModernBERT-base
passes the attested gate (calibrated accuracy 0.990, ECE 0.004) at 27.6 ms
p50 on the CPU ONNX path and 8.1 ms on the GPU; ModernBERT-large initialised
from the Laya typed-decisions encoder passes with 0.994 / 0.003 at 93.7 ms CPU
and 13.5 ms GPU; pretrained ModernBERT-large misses three attested cases and
fails the gate despite the best held-out numbers; DeBERTa-v2-xlarge (the ~1B
point) could not be trained on the ROCm stack (non-finite logits at two
learning rates) and its ONNX export fails parity, so it is unusable here even
before its 3.5 GB size.

- **ModernBERT-base stays the default.** The larger encoders buy at most half a
  point of calibrated accuracy for a 3.2x to 3.4x CPU latency cost and exceed the
  v1 artifact's 1 GiB component limit.
- **DeBERTa-v3-base is no longer the named fallback.** If an application needs
  a larger encoder, the measured alternative is ModernBERT-large initialised
  from the Laya typed-decisions weights: decision-trained initialisation, not
  size, is what improved the attested result at 400M.
- The encoder is a per-application setting (`--encoder-name`,
  `--encoder-revision`): the largest encoder that fits the request path's
  latency budget, with the attested gate deciding.
