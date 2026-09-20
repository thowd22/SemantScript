---
id: decision-3
title: 'Base encoder: ModernBERT-base, DeBERTa-v3-base fallback'
date: '2026-09-20 02:42'
status: accepted
---
## Context

Phase 1 needs a pretrained 100-300M encoder to fine-tune with a classification head. It must be small enough to iterate quickly on a 16 GB GPU (or CPU), fast at inference in ONNX, and strong at classification.

## Decision

Start with **ModernBERT-base** (~149M): modern architecture, fast inference, 8k context for structured inputs. Keep **DeBERTa-v3-base** (~184M) as the fallback if ModernBERT underperforms on the refund-decision benchmark.

## Consequences

- Encoder choice is a config value in `model/`; the benchmark harness should be able to run both.
- Revisit once the Phase 3 universal-encoder investigation reports.


