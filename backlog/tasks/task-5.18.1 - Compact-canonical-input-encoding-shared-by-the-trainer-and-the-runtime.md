---
id: TASK-5.18.1
title: Compact canonical input encoding shared by the trainer and the runtime
status: To Do
assignee: []
created_date: '2026-09-24 12:13'
labels:
  - compiler
  - trainer
  - runtime
  - performance
milestone: m-1
dependencies: []
parent_task_id: TASK-5.18
ordinal: 51000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every sema call serializes its inputs to canonical text before tokenization, and the current form is verbose: a five-field refund input becomes about 100 to 115 ModernBERT tokens because it nests typed arrays and encodes every number as a 16-character hexadecimal binary64 string. Encoder cost scales with tokens (the exported ONNX encoder measures 35.5 ms at 110 tokens and 18.1 ms at 32 on the benchmark CPU), so a compact encoding roughly halves per-call latency and shortens every training example. The encoding is a contract shared byte for byte by the Python trainer (serialize_canonical_inputs) and the TypeScript runtime, and models are trained on its exact text, so it must stay deterministic and canonical (stable field order, one spelling per value, no locale or float-formatting drift), be versioned so an artifact declares which encoding it was trained on and the runtime refuses a mismatch, and be followed by a retrained release that passes the existing gate. Parent task records the measurements and the end-to-end target.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Python and TypeScript serializers produce byte-identical output over a shared committed vector set covering integers, non-integral and negative numbers, string unions, nested objects and arrays
- [ ] #2 A representative refund input serializes to at most 40 ModernBERT tokens, measured with the pinned tokenizer
- [ ] #3 The encoding version is recorded in the IR and the artifact manifest, and the runtime refuses to load an artifact whose encoding version it does not implement
- [ ] #4 A refund release trained on the compact encoding passes the release gate with zero attested misses, and its per-call p50 on the CPU runtime is reported next to the 38.4 ms baseline
<!-- AC:END -->
