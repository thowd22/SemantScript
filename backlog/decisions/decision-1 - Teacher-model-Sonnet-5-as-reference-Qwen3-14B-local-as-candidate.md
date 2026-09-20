---
id: decision-1
title: 'Teacher model: Sonnet 5 as reference, Qwen3-14B local as candidate'
date: '2026-09-20 02:42'
status: accepted
---
## Context

The trainer needs a model that turns a sema<T> IR record (behavior text, input schema, output type, examples, constraints) into labeled training cases. This runs at build time only; no LLM runs in production. Constraints: the dev machine has a Radeon RX 9070 XT (16 GB VRAM) under WSL2 where ROCm is not currently reachable, and we want a defined quality bar for any local teacher rather than guessing.

## Decision

Use two teachers behind one interface (`generate(ir, n) -> cases`):

- **Reference teacher:** `claude-sonnet-5` via the Anthropic Python SDK, using structured outputs (`output_config.format`) with a JSON schema derived from the IR so cases are schema-valid by construction. Large runs go through the Batch API (50% cost).
- **Local candidate:** Qwen3-14B (Q4) via Ollama's OpenAI-compatible endpoint. Qwen3-8B as a faster fallback.

The local candidate is evaluated against the reference: label agreement on identical inputs, and student-encoder accuracy on a Sonnet-labeled held-out set (see the Phase 1 teacher-comparison story).

## Consequences

- Estimated cost for a ~20k-case Phase 1 dataset is roughly $20 (about $10 via batch).
- Teacher choice becomes a config swap; iteration can move fully local if the Qwen-trained student is within tolerance.
- Ollama should run natively on Windows (it supports the 9070 XT) and be reached from WSL at localhost:11434 until ROCm-in-WSL is sorted.
- The Sonnet-labeled held-out set is the fixed evaluation target and must never be used for training.


