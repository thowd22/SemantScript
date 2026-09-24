---
id: TASK-5.18.2
title: >-
  Int8 encoder export with parity and calibration verified on the quantized
  graph
status: Done
assignee:
  - '@claude'
created_date: '2026-09-24 12:13'
updated_date: '2026-09-24 13:13'
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
- [x] #1 The exporter can emit an int8-quantized encoder and records the quantization method and tolerance in the artifact manifest and pipeline manifest
- [x] #2 Parity and calibration checks run against the quantized ONNX graph, fail the export when the argmax changes on any verification record or ECE exceeds the gate, and the tolerances are tested
- [x] #3 The Node runtime loads the int8 artifact with no change to callers, and the benchmark harness measures its p50 next to the fp32 figure
- [x] #4 An end-to-end test covers the quantized export through the runtime with the offline fixture
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Model package (semantscript_model/export.py): add quantize_onnx_encoder(source, destination, method, per_channel, reduce_range) over onnxruntime.quantization.quantize_dynamic with optional-dependency guarding, re-run the existing container validation on the result (opset unchanged, standard domains only, no external data, size bound) and return a report (weight type, quantized MatMul count, byte length, digest).
2. Trainer (semantscript_trainer/artifact.py): add quantize_release_artifact(artifact_root, source_manifest_sha256, records, config) that derives a new content-addressed release from a published fp32 release rather than from the in-memory PyTorch model (the trained model is not persisted; the verified fp32 ONNX chain is the reference). It copies tokenizer, adapter and head resources by digest, quantizes the encoder, and publishes through the same staging, exclusive-write, fsync and immutable-release path as export_application_artifact.
3. Quantized-graph verification inside that call: run the fp32 chain and the int8 chain (encoder to adapter to head, batch 1 as the runtime does, tokenizer.json via the tokenizers library, canonical serialization from the manifest input schema) over caller-supplied records (inputs, optional expected label, attested flag). Fail and remove staging when argmax disagreement over all records exceeds ArtifactExportConfig.maximum_quantized_argmax_disagreement_rate (default 0.0), when any attested record disagrees, or when 15-bin ECE on labeled records at the manifest temperature exceeds quantized_ece_threshold (default 0.1, the verification gate). Tolerances validated like the parity tolerances.
4. Manifest contract: encoder resource onnx block gains optional precision (float32 default, int8-dynamic) and, when quantized, a closed quantization object (method, weightType, perChannel, reduceRange, argmaxDisagreementTolerance, eceThreshold, sourceManifestSha256). Update schemas/application-artifact.v1.schema.json, runtime artifact-types.ts and artifact-loader.ts validateOnnx (optional keys, enum, reject quantization on float32), Python manifest validation, and loader tests. build.createdAt is the derivation time; every other block is copied from the source manifest so evidence bindings hold.
5. Driver benchmarks/refund/program/quantize_release.py: reads a release pipeline manifest, replays the frozen corpus and the attested release record exactly like run_release_pipeline (teacher calls forbidden) to build the record set (training rows, calibration rows labeled, attested cases flagged), derives the int8 release, and writes a derived pipeline manifest in a separate output directory (same function and ledger blocks, artifact block pointing at the int8 release, derivedFrom source manifest digest, full quantization report: records checked, disagreements, quantized ECE and accuracy, sizes, seconds) so run-benchmark.mjs can measure it unchanged.
6. Tests: model export test for the quantizer on a tiny encoder with a Linear layer (MatMulInteger present, container valid); trainer tests for the derived release end to end with the offline fixture including the Node runtime load, manifest precision and quantization fields, tolerance failures (forced disagreement and forced ECE failure) with staging cleanup, config validation; runtime loader tests for accepted and rejected precision/quantization blocks; pipeline test for the driver record assembly.
7. Run on the committed seed-2 release (manifest 0ee80669), record the report in the task, then run run-benchmark.mjs with the derived pipeline manifest into a separate results directory and report p50 next to the fp32 38.4 ms figure. Quantization defaults are chosen from the measured probe on real records (dynamic int8 variants: default, per-channel, reduce-range, pre-processed); a variant that flips an attested release case fails step 3 by design and is reported as such.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design change from the plan: the trained PyTorch model is not persisted, so int8 export is a derived release (semantscript_trainer.quantization.quantize_release_artifact) built from a published float32 release, not an export-time option; the float32 ONNX chain is the parity reference. Probe on the seed-2 release before building (1,000 sampled corpus rows, 80 release cases, 160 final cases; onnxruntime CPU): every dynamic int8 variant (int8, int8 per-channel, int8 reduce-range, uint8 per-channel, with and without ORT pre-processing) changes 1 to 2 of the 80 attested release decisions and 0.1 to 0.4 percent of corpus decisions, ECE stays within 0.005 to 0.03, and onnxruntime latency improves only 36.6 to 27.4 ms at 110 tokens (1.3x) and 23.2 to 13.0 ms at 32 tokens (1.8x), far less than the 2.3x PyTorch estimate in the task description. Implemented: model package quantize_onnx_encoder (ORT dynamic quantization, container re-validated, standard opset only); trainer QuantizationConfig/QuantizationRecord/QuantizationReport and quantize_release_artifact with a gate (attested disagreement tolerance default 0, argmax disagreement rate default 0, ECE threshold default 0.1 at the manifest temperature, batch-1 chain through tokenizer.json like the runtime), staging cleanup on refusal and a QuantizationGateError carrying the report; manifest contract gains optional onnx.precision plus a closed quantization block (schema, runtime types, loader with optional-key validation, Python validator); driver benchmarks/refund/program/quantize_release.py replays the corpus and attested record and writes a derived pipeline manifest the benchmark harness reads unchanged. Tests: model 15 passed, trainer test_quantization 12 passed, runtime 89 passed.

Runs on the seed-2 float32 release (manifest 0ee80669...): strict default gate REFUSED (10 of 9,489 decisions changed, rate 0.00105; 1 attested release case changed: standard tier, fraudulent at 179 days, expected deny, int8 says review p=0.94; quantized ECE 0.0066 vs 0.0043), report committed under benchmarks/refund/data/release-int8-2026-09-24 with no release published. Tolerance-recorded run (attested tolerance 2, rate 0.01) published manifest f01067d5... under benchmarks/refund/data/release-int8-tolerant-2026-09-24 (encoder 596.7 MB -> 150.1 MB, 88 MatMulInteger nodes, 1,063 s including 9,489 batch-1 chain comparisons). Benchmark on the final set (results-int8-2026-09-24, same protocol): int8 accuracy 160/160 (its one changed final-set decision fixes the float32 miss by chance), p50 27.74 ms vs 38.40, p95 35.26 vs 50.49, 34.8 req/s vs 24.5, client RSS 599 MiB vs 1,646 MiB. Conclusion: dynamic int8 is a 1.38x lever with about one decision change per thousand; it does not meet the 10 ms bar alone and the Phase 1 zero-attested-miss policy refuses this graph. Full suites: trainer + model + benchmark program 458 passed, 4 skipped; runtime 89 passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added int8 export as a verified derived release: quantize_onnx_encoder in the model package and quantize_release_artifact in the trainer derive a content-addressed int8 release from a published float32 release, run the quantized chain against the float32 chain over every verification record, and refuse unless attested decisions, the decision-change rate and ECE stay within tolerances recorded in the manifest (schema, runtime loader and Python validator extended; older manifests unchanged). The refund driver replays the corpus and attested record and writes a derived pipeline manifest the benchmark reads unchanged. Verified with new model, trainer, runtime-loader and driver tests (458 + 89 passing) and by running it on the seed-2 release: the strict gate refused the graph because one attested release decision changed (report committed), and a tolerance-recorded artifact measured p50 27.7 ms against 38.4 ms float32 with 160/160 on the final set, client RSS 599 MiB against 1,646 MiB. Int8 alone is 1.38x, not enough for the 10 ms bar.
<!-- SECTION:FINAL_SUMMARY:END -->
