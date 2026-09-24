---
id: TASK-5.18.2
title: >-
  Int8 encoder export with parity and calibration verified on the quantized
  graph
status: To Do
assignee: []
created_date: '2026-09-24 12:13'
labels:
  - trainer
  - runtime
  - performance
milestone: m-1
dependencies: []
parent_task_id: TASK-5.18
ordinal: 52000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The exported artifact runs the encoder in fp32; dynamic int8 quantization of the Linear layers measured 2.3x faster on the benchmark CPU (60 ms to 26 ms in PyTorch at 22 layers and 110 tokens) with no retraining. The export stage already checks ONNX parity against the verified PyTorch model and the artifact carries calibrated confidences, so quantization has to be part of the verified path rather than a post-processing step: parity must be measured against the quantized graph with a stated tolerance, calibration error re-verified on the quantized outputs before the release is published, and the quantization method recorded in artifact provenance so a benchmark result names exactly what ran. Runtime callers must not change. Parent task records the measurements and the end-to-end target.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The exporter can emit an int8-quantized encoder and records the quantization method and tolerance in the artifact manifest and pipeline manifest
- [ ] #2 Parity and calibration checks run against the quantized ONNX graph, fail the export when the argmax changes on any verification record or ECE exceeds the gate, and the tolerances are tested
- [ ] #3 The Node runtime loads the int8 artifact with no change to callers, and the benchmark harness measures its p50 next to the fp32 figure
- [ ] #4 An end-to-end test covers the quantized export through the runtime with the offline fixture
<!-- AC:END -->
