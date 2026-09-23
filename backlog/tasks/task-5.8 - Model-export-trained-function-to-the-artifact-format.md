---
id: TASK-5.8
title: 'Model: export trained function to the artifact format'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-23 14:30'
labels:
  - model
milestone: m-1
dependencies:
  - TASK-5.7
documentation:
  - model/README.md
  - trainer/README.md
  - DEVELOPING.md
modified_files:
  - pyproject.toml
  - DEVELOPING.md
  - model/README.md
  - model/src/semantscript_model/__init__.py
  - model/src/semantscript_model/export.py
  - model/tests/test_export.py
  - trainer/README.md
  - trainer/src/semantscript_trainer/__init__.py
  - trainer/src/semantscript_trainer/artifact.py
  - trainer/src/semantscript_trainer/semantic_json.py
  - trainer/src/semantscript_trainer/verification.py
  - trainer/tests/test_artifact.py
  - trainer/tests/test_semantic_json.py
  - trainer/tests/test_verification.py
parent_task_id: TASK-5
ordinal: 13000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The runtime is Node and must load the model in-process. Export the encoder+head to ONNX and write the manifest defined in the IR/artifact task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Encoder and head export to ONNX and produce identical outputs to the PyTorch model on a fixed test batch
- [x] #2 Artifact directory matches the documented layout including manifest, calibration and verification stats
- [x] #3 Artifact loads without the training environment installed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend verified training evidence with deterministic model-state and exact tokenizer-resource digests so export rejects stale weights or a tokenizer different from the one verified. Require exact source-IR bytes and the currently underspecified training-key digest as explicit validated provenance inputs.
2. Add a low-level scalar ONNX exporter that writes the trained sentence encoder, a generated identity application adapter, and the trained head as separate single-file opset-17 graphs with the v1 tensor names/shapes; inspect each protobuf and compare the chained ONNX outputs with eval-mode PyTorch outputs on a bounded fixed batch.
3. Build the closed v1 function and application manifest from IR, training, verification, typed semantic-JSON schema digests, exact resource hashes/lengths, and explicit application/build provenance; map TASK-5.7 calibration and verification projections without including training data or definitions.
4. Publish through a same-filesystem private staging directory: write/fsync fixed-path resources first, serialize and validate the manifest once, content-address the release from those exact bytes without overwriting an existing release, then atomically replace the optional current.json pointer; enforce runtime-aligned resource/aggregate/manifest quotas and clean only owned staging paths on failure.
5. Add offline tiny encoder/head parity tests (binary, categorical, and MLP), stale-model/tokenizer and path/quota/tamper cases, exact schema/layout/digest tests, and a separate Node runtime load/inference test proving the release works without Python training packages.
6. Document dependencies and exporter usage, expose optional-dependency-safe public APIs, run focused gates and independent acceptance/security audits, then run the memory-bounded full repository gate; keep a cached ModernBERT export smoke opt-in because its full graph is large.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a low-level opset-17 exporter for separate encoder, identity-adapter, and scalar-head graphs with exact ABI inspection and independent/chained ONNX Runtime parity.

Added deterministic typed semantic JSON plus verification bindings for exact model state and tokenizer bytes, then built a strict scalar artifact manifest from verified IR, training evidence, calibration, and explicit provenance.

Publication uses bounded resources, pre-parse ONNX quotas, hard-capped parity tolerances, descriptor-anchored staging and verification, atomic no-clobber content-addressed releases, atomic pointer replacement, and fail-closed path/inode checks.

Added offline parity, schema, binding, tamper, collision, root-swap, Node load/inference, and public-API regressions; documented training/export dependencies and usage.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @codex
created: 2026-09-23 03:45
---
Started in ordinal order after TASK-5.7 passed all gates. Beginning parallel reconnaissance on the documented artifact layout/manifest schema, ONNX export and PyTorch parity strategy, and dependency-free Node loader compatibility before recording the implementation plan.
---

author: @codex
created: 2026-09-23 03:54
---
Reconnaissance complete. The runtime requires a four-resource encoder→identity-adapter→head chain, exact typed semantic-JSON schema digests, and content-addressed atomic publication. Security review also identified that TASK-5.7 evidence must be bound to the exact model state and tokenizer bytes before export; this cross-stage integrity binding is included in the implementation plan.
---

author: @codex
created: 2026-09-23 13:40
---
Implementation milestone: added deterministic typed semantic-JSON hashing, verification bindings to the exact model state and tokenizer JSON, split opset-17 encoder/identity-adapter/head export with ONNX Runtime parity, bounded content-addressed publication, exact manifest projections, public APIs, and documentation. The focused gate currently passes 37 parity/semantic/verification/artifact tests; a generated artifact also passes the actual Node loader and executes one inference call without Python training code in that process. Independent final acceptance/security/semantic audits are in progress before the full bounded repository gate.
---

author: @codex
created: 2026-09-23 14:00
---
Acceptance hardening: rejected Python bool-as-int aliases for both IR version and dense input indices, with regression coverage. Focused artifact/semantic/verification/ONNX export gate now passes 39 tests; Ruff passes.
---

author: @codex
created: 2026-09-23 14:04
---
Final security audit found four hardening gaps despite green functional gates: release publication must be atomic no-clobber, ONNX quotas must apply before parser/runtime loading, parity tolerances need a strict upper bound, and path checks need directory-descriptor/no-follow anchoring. Keeping the task In Progress while these are fixed and regression-tested.
---

author: @codex
created: 2026-09-23 14:28
---
Security hardening complete: OS-native atomic no-clobber release rename; descriptor-anchored staging/resource/pointer/existing-release operations with root and inode checks; pre-parser ONNX component quotas; and parity tolerances capped at 1e-3. Deterministic concurrent-release and root-swap regressions pass, and the independent security re-audit reports no remaining blockers.
---

author: @codex
created: 2026-09-23 14:30
---
Final verification: bounded npm run check passed on the exact finalized state (16 Node test files; 357 Python passed, 4 opt-in skips; 829056 KiB peak RSS; zero swap). Independent acceptance, semantic, and post-fix security audits report no remaining TASK-5.8 blockers.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exported verified scalar functions as Node-loadable content-addressed artifacts containing tokenizer JSON plus separate opset-17 encoder, adapter, and head graphs. Exact semantic/model/tokenizer bindings, calibration and verification projections, prepublication schema checks, bounded parsing, parity validation, and race-safe atomic publication are enforced. Independent semantic, acceptance, and security audits found no remaining blockers. Final bounded repository gate: all lint/format/build checks passed; 16 Node test files passed; 357 Python tests passed with 4 intentional opt-in skips; peak RSS 829056 KiB; zero swap.
<!-- SECTION:FINAL_SUMMARY:END -->
