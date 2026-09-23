---
id: TASK-5.9
title: 'Runtime: load artifact and execute one forward pass in Node'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-22 17:18'
labels:
  - runtime
milestone: m-1
dependencies:
  - TASK-2
  - TASK-4
references:
  - docs/research/laya-analysis.md
documentation:
  - IR.md
  - runtime/README.md
  - schemas/application-artifact.v1.schema.json
modified_files:
  - .gitignore
  - IR.md
  - package.json
  - package-lock.json
  - schemas/application-artifact.v1.schema.json
  - examples/artifacts/refund-app/manifest.v1.json
  - examples/artifacts/refund-app/current.v1.json
  - compiler/src/semantic-json.ts
  - compiler/test/semantic-json.test.mjs
  - runtime/package.json
  - runtime/README.md
  - runtime/src/artifact-types.ts
  - runtime/src/strict-json.ts
  - runtime/src/artifact-loader.ts
  - runtime/src/canonical-input.ts
  - runtime/src/inference-protocol.ts
  - runtime/src/inference-worker.ts
  - runtime/src/inference-runtime.ts
  - runtime/src/onnx-model.ts
  - runtime/src/artifact-runtime.ts
  - runtime/src/index.ts
  - runtime/test/artifact-loader.test.mjs
  - runtime/test/canonical-input.test.mjs
  - runtime/test/inference-runtime.test.mjs
  - runtime/test/onnx-model.test.mjs
  - runtime/test/strict-json.test.mjs
  - runtime/test/runtime.integration.test.mjs
  - runtime/test/package.test.mjs
  - runtime/test/fixtures/artifact.mjs
  - runtime/test/fixtures/tokenizer.json
  - runtime/test/fixtures/encoder.onnx
  - runtime/test/fixtures/adapter.onnx
  - runtime/test/fixtures/head.onnx
  - runtime/test/fixtures/wrong-head.onnx
  - runtime/test/fixtures/fixed-batch-encoder.onnx
  - runtime/test/fixtures/fixed-batch-adapter.onnx
  - runtime/test/fixtures/fixed-batch-head.onnx
  - runtime/test/fixtures/symbolic-hidden-adapter.onnx
  - runtime/test/fixtures/wide-adapter.onnx
  - scripts/generate-runtime-fixtures.py
parent_task_id: TASK-5
ordinal: 14000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
@semantscript/core must resolve __sema.call(functionId, inputs) to a typed value with no token stream, parsing or validation. Uses ONNX Runtime in-process. Transcript turn 9 point 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime loads an artifact directory and exposes __sema.call(functionId, inputs)
- [x] #2 Inputs are serialized per the input schema and the head output is mapped back to the TS type value
- [x] #3 Calling an unknown function id or malformed inputs throws a typed error
- [x] #4 Runtime has no dependency on Python
- [x] #5 Inputs are rendered with the canonical serialization from the IR; a test proves key-order invariance
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement strict artifact pointer/manifest/resource loading and immutable function-table construction from the v1 schemas and IR relational rules. 2. Add schema-directed canonical input validation/serialization with typed runtime errors and key-order-invariance coverage. 3. Integrate local tokenizer plus ONNX Runtime sessions and bridge the asynchronous native API behind the synchronous compiler ABI without Python. 4. Map calibrated head outputs back to scalar/flat TypeScript values, install the loaded runtime atomically, and cover unknown IDs, malformed inputs, fixture inference, integrity/path failures, and reload safety. 5. Document the runtime API and run package, repository, audit, and independent review gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a strict, quota-bounded artifact loader and atomic activation lifecycle; schema-directed canonical input validation and deterministic serialization; a dedicated worker-thread/SharedArrayBuffer bridge preserving the compiler's synchronous ABI over asynchronous tokenizer and ONNX APIs; exact finite scalar/flat output transport including signed zero; semantic ONNX container and live session ABI validation; fixed/dynamic batch and hidden compatibility; and typed load/input/inference errors. Runtime deployment dependencies are pinned to onnxruntime-node 1.30.0 and tokenizers 0.23.2 only. Confidence thresholds and fallback behavior remain deliberately deferred to TASK-5.10. Resource hashing/verification completes before ownership transfer, exact file buffers avoid duplicate full-model allocations, canonical byte quotas apply during serialization, worker memory is bounded, and failed/stale reloads cannot replace or unload the active artifact.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @codex
created: 2026-09-22 15:02
---
TASK-5.9 started as the next ready item in ordinal order after TASK-5.3. Initial design work is resolving the synchronous compiler ABI against ONNX Runtime's promise-based Node API while preserving in-process execution, local-only artifacts, and atomic runtime activation.
---

created: 2026-09-22 15:24
---
Design audits converged on an async staged loader plus a dedicated worker-thread/SharedArrayBuffer bridge so the existing synchronous __sema.call<T> ABI remains conforming while ONNX Runtime executes in-process. Runtime dependencies are pinned to onnxruntime-node 1.30.0, tokenizers 0.23.2, Ajv 8.20.0, and ajv-formats 3.0.1. The loader will pass tokenizer/model bytes from the same verified buffers; a deterministic tiny artifact will define and test the v1 tokenizer/tensor routing convention.
---

created: 2026-09-22 15:59
---
Implementation is integrated: strict duplicate-key artifact loading; schema-directed canonical serialization and typed errors; worker-thread/SharedArrayBuffer synchronous bridge; verified tokenizer bytes; actual ONNX container, opset/domain/external-data and live tensor metadata checks; atomic activation; calibrated scalar/flat output mapping; and deterministic Node-only fixtures. Focused runtime tests pass under NODE_OPTIONS=--max-old-space-size=512 (7 test files, including real ONNX inference, failed-reload preservation, post-staging ABI rejection, stale-handle isolation, and key-order invariance). Node lint, Python fixture-generator lint, package dry-run, git diff check, and npm audit (0 vulnerabilities) also pass. Independent final audits are now running before the bounded full repository suite.
---

created: 2026-09-22 16:58
---
Final independent audit found four bounded correctness and OOM-safety gaps before finalization: exact signed-zero output transport, dimension-aware live fixed/symbolic ONNX edge compatibility, incremental canonical-input byte quotas before materialization, and avoiding a second full resource buffer allocation. Parallel implementation work is underway; fixed-batch ONNX fixtures and a real replacement-load regression have also been added. The bounded full repository suite remains intentionally deferred until these focused gates pass.
---

created: 2026-09-22 17:17
---
Final verification is green. Exact capped runtime test invocation passed all 7 runtime test files at 248,780 KiB peak RSS; the full NODE_OPTIONS=--max-old-space-size=512 npm run check passed repository lint, formatting, builds, 14 Node test files, and 2 Python tests at 494,052 KiB peak RSS with no swaps. Package dry-run includes the worker, strict Ajv example/pointer validation and digest verification pass, Ruff and git diff checks pass, and npm audit reports 0 vulnerabilities. Correction to comment #2: final deploy-time runtime dependencies are only pinned onnxruntime-node 1.30.0 and tokenizers 0.23.2; Ajv and ajv-formats remain root development dependencies for schema verification, not runtime dependencies.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Node now loads verified v1 artifact directories and resolves compiler-generated __sema.call(functionId, inputs) synchronously through local tokenizer and ONNX inference, with canonical schema-directed inputs, typed errors, exact TypeScript output mapping, atomic reload safety, and no Python runtime dependency. Focused runtime tests and the full repository check pass under a 512 MiB heap; full-suite peak RSS was 494,052 KiB with no swaps. Package dry-run, strict schema/example and pointer-digest checks, Ruff, git diff, and npm audit (0 vulnerabilities) also pass.
<!-- SECTION:FINAL_SUMMARY:END -->
