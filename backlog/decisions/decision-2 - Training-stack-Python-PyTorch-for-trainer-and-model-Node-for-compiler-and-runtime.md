---
id: decision-2
title: >-
  Training stack: Python/PyTorch for trainer and model, Node for compiler and
  runtime
date: '2026-09-20 02:42'
status: accepted
---
## Context

SemantScript spans two ecosystems: the compiler and runtime must live in the TypeScript/Node world (the product is a TypeScript extension and `@semantscript/core` runs in-process in Node), while training a transformer encoder is best served by PyTorch. An all-TypeScript stack was considered.

## Decision

- `compiler/`, `runtime/`, `cli/`: TypeScript on Node.
- `trainer/`, `model/`: Python with PyTorch and Hugging Face Transformers.
- Inference in Node via ONNX Runtime; the model half exports to ONNX and the runtime has no Python dependency.
- The IR bundle (JSON, versioned schema) is the only contract between the two halves.

## Consequences

- Two toolchains to maintain; a single top-level command must run lint and tests for both.
- ROCm PyTorch on WSL2 is unproven for RDNA4; CPU fine-tuning of a ~150M encoder is the fallback, native Linux the reliable path.
- ONNX export must be verified against PyTorch outputs on every artifact build.


