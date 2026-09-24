---
id: TASK-5.18.1
title: Compact canonical input encoding shared by the trainer and the runtime
status: Done
assignee:
  - '@claude'
created_date: '2026-09-24 12:13'
updated_date: '2026-09-24 14:18'
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
- [x] #1 The Python and TypeScript serializers produce byte-identical output over a shared committed vector set covering integers, non-integral and negative numbers, string unions, nested objects and arrays
- [x] #2 A representative refund input serializes to at most 40 ModernBERT tokens, measured with the pinned tokenizer
- [x] #3 The encoding version is recorded in the IR and the artifact manifest, and the runtime refuses to load an artifact whose encoding version it does not implement
- [x] #4 A refund release trained on the compact encoding passes the release gate with zero attested misses, and its per-call p50 on the CPU runtime is reported next to the 38.4 ms baseline
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define canonical-input/v2 as a second writer over the same validated typed tree both serializers already build (validation, ordering and error reasons unchanged): top-level inputs as name=value pairs joined by one space; objects {k=v,...} with the existing byte-ordered keys; arrays and tuples [v,...]; null, true, false bare; numbers in ECMAScript Number-to-string spelling (shortest round-trip digits, decimal for 1e-7 <= |x| < 1e21, exponent form otherwise) with -0 spelled -0 so distinct doubles stay distinct; strings bare when they match ^[A-Za-z_][A-Za-z0-9_.-]*$ and are not true/false/null, otherwise quoted with the existing exact-JSON string escaping; literal, enum and union wrappers drop their tags because the value carries them. The encoding is injective and carries no envelope; the artifact declares the version.
2. Python: serialize_canonical_inputs(schema, inputs, *, version=1, maximum_bytes) plus a JS-compatible number formatter derived from repr digits; TypeScript: serializeCanonicalInputs(schema, inputs, {version, maximumBytes}); both keep v1 byte-for-byte.
3. Shared golden vectors in examples/serialization/canonical-input.v2.json (integers, non-integral, negative and signed-zero numbers, exponent-form numbers, string unions, enums, literals, nested objects, arrays, tuples, quoted strings with delimiters and escapes, empty containers, booleans, null); the TypeScript output is generated first and reviewed, then both test suites assert byte identity and digests against the file.
4. Version plumbing: constants for semantscript.canonical-input/v1 and /v2; TrainingConfig.canonical_input_version (default 2, validated) threaded into training row preparation, verification, the release driver (parity text, release predictions, CLI flag, pipeline manifest) and quantization (which reads the source manifest); the lifecycle builder writes canonicalInput into the verified IR trainingProvenance from the training config, the artifact validator accepts it (absent means v1 for older IRs), the IR JSON schema gets the optional enum, and export_application_artifact writes compatibility.canonicalInput from the same config; the runtime loader accepts v1 and v2 and rejects anything else, and dispatchSemaCall serializes with the loaded manifest's version.
5. Tests: Python and TypeScript vector tests; token-budget test on representative refund inputs with the pinned tokenizer (skipped when not cached); loader accept/reject tests; artifact-runtime call through a v2 manifest; training/verification/lifecycle/artifact tests updated for the version field.
6. Retrain the refund release on the v4 corpus with the compact encoding (same recipe as the committed release, seeds until the strict gate passes), benchmark it under the committed protocol into its own results directory, and report p50 next to 38.4 ms in the task and the results write-up.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented canonical-input/v2 as a compact writer over the shared validated typed tree in both serializers (Python canonical_input.py, TypeScript canonical-input.ts): name=value pairs, {k=v} objects, [v] sequences, bare identifiers when unambiguous, JavaScript number spelling with -0 preserved, literal/enum/union tags dropped; injective over validated inputs. Python number spelling cross-checked against Node over 29,000 doubles including subnormals, 1e21 and 1e-7 boundaries: zero mismatches. Shared golden vectors in examples/serialization/canonical-input.v2.json (generated by the TypeScript serializer, seven vectors covering the refund schema, numbers, strings, unions/literals/enums, nested containers, key spelling) are asserted byte for byte by both test suites. Representative refund inputs: 31 to 34 ModernBERT tokens with v2 against 102 to 115 with v1. Version plumbing: TrainingConfig.canonical_input_version (default 2) through row preparation, verification, quantization (reads the source manifest), the release driver (--canonical-input-version, pipeline manifest, parity text, release predictions); the lifecycle builder writes trainingProvenance.canonicalInput, the artifact validator accepts it (absent means v1), the export cross-checks and declares compatibility.canonicalInput, the IR schema lists the enum, and the runtime loader accepts v1 and v2 and rejects others while dispatch serializes with the declared version. Suites: Python 458 passed, runtime 96 passed. Retraining the refund release with the compact encoding on the v4 corpus (same recipe as the committed release) is running; the benchmark follows automatically on a passing seed.

Retrain on the v4 corpus with --canonical-input-version 2 (same recipe as the committed release) PASSED the strict gate on seed 1 (best epoch 7 of 8): zero attested misses, calibration accuracy 0.9968, ECE 0.0036, temperature 1.751, 3 of 9,489 constraint violations, training 344 s (v1 release: 0.9904, 0.0043, 14 violations, 1,290 s, second seed). Artifact manifest 56c5c5d6... declares compatibility.canonicalInput v2 and the verified IR records trainingProvenance.canonicalInput v2. Benchmark on the final set (results-compact-2026-09-24, committed protocol): 160/160, p50 28.49 ms vs 38.40 ms baseline, p95 35.39 vs 50.49, 34.6 req/s, client RSS 1,662 MiB. Tokens are a 1.35x lever at batch 1 because 22 layers of per-layer overhead dominate once the input is short; depth routing remains the lever for the 10 ms bar. Suites: Python 475 passed, runtime 96 passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added canonical-input/v2, a compact text encoding rendered from the same validated typed tree by both the Python trainer and the TypeScript runtime (name=value pairs, {k=v} objects, [v] sequences, bare identifiers, JavaScript number spelling with signed zero, no wrapper tags), injective and asserted byte for byte against shared golden vectors in examples/serialization. Refund inputs drop from 102 to 115 ModernBERT tokens to 31 to 34. The version is carried by TrainingConfig (default 2), recorded in the verified IR and the artifact manifest, and the runtime serializes calls with the declared version while rejecting unimplemented ones. Verified with new vector, token-budget, loader and dispatch tests (475 Python and 96 runtime tests passing) and by retraining the refund release: strict gate passed on seed 1 with zero attested misses, final set 160/160, p50 28.5 ms against the 38.4 ms baseline (commits 9a01b64 and the results commit).
<!-- SECTION:FINAL_SUMMARY:END -->
