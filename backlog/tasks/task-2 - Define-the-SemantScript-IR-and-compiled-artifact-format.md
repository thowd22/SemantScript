---
id: TASK-2
title: Define the SemantScript IR and compiled artifact format
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-22 05:11'
labels:
  - spec
  - compiler
milestone: m-0
dependencies:
  - TASK-1
references:
  - docs/research/laya-analysis.md
documentation:
  - IR.md
modified_files:
  - IR.md
  - schemas/neural-function.v1.schema.json
  - schemas/application-artifact.v1.schema.json
  - schemas/artifact-pointer.v1.schema.json
  - examples/refund.sem.ts
  - examples/ir/refund-decision.v1.json
  - examples/artifacts/refund-app/manifest.v1.json
  - examples/artifacts/refund-app/current.v1.json
  - examples/serialization/canonical-input.v1.json
priority: high
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The compiler emits IR that the trainer consumes and the runtime loads. All three teams need one schema. Transcript turn 10 sketches the shape (function id, inputs, output head, model refs, runtime confidence, training provenance, verification) but no concrete format exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A versioned JSON schema for a NeuralFunction IR record exists and includes: function id, input schema, output head spec, model/adapter/head refs, confidence requirement, training provenance, verification stats
- [x] #2 The on-disk artifact layout for an application (shared encoder + adapter + heads + manifest) is documented
- [x] #3 At least one hand-written example IR file for the refund-decision expression validates against the schema
- [x] #4 The IR defines canonical input serialization: deterministic field order and type-aware rendering so identical inputs with different object key order produce identical model input
- [x] #5 The artifact manifest carries per-function calibration (fitted temperature) and ECE and Brier score
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define a strict JSON Schema 2020-12 NeuralFunction v1 record with recursive input types, scalar and flat output heads, source definition data, model references, runtime confidence policy, training provenance, calibration, and verification evidence. 2. Add a hand-written refund-decision IR instance and validate it against the schema. 3. Define and schema-validate an application artifact manifest plus the shared encoder, adapter, per-function head, runtime metadata, hashing, and compatibility layout. 4. Specify canonical type-aware input serialization with fixed framing, ordering, escaping, and cross-language number rules. 5. Run positive and negative schema validation and independent contract reviews, then record acceptance evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented strict JSON Schema Draft 2020-12 contracts for the lifecycle-aware NeuralFunction IR, deployable application manifest, and atomic artifact pointer. Added real refund source, verified IR, artifact manifest, pointer, and canonical input fixtures.

Validation evidence:
- All three schemas pass Draft202012Validator.check_schema.
- Refund IR, artifact manifest, and pointer validate with FormatChecker.
- Source and trained lifecycle variants validate; invalid lifecycle combinations are rejected.
- Semantic SHA, stable function ID, input/output schema SHA, source SHA, IR file SHA, manifest SHA, and pointer release SHA independently reproduce.
- The canonical input golden vector reproduces SHA-256 3273c3e7b704fce97a131fbdf1c3cd2bef0cd3896c1f4ebbc6857921b534a162 and preserves signed zero.
- Negative cases reject unsafe paths, missing runtime input descriptors, invalid calibration temperatures, noncanonical decimals, unknown fields, and mismatched head parameterization.
- Three independent agent reviews completed; all reported no remaining TASK-2 blocker after fixes.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @codex
created: 2026-09-22 02:37
---
TASK-1 SPEC contract to carry into this task: calibration is one fitted temperature per NeuralFunction/source expression (shared by flat-interface scalar heads); bounded decimal support retains exact canonical decimal spellings; marker types are resolved by @semantscript/core symbol identity.
---

author: @codex
created: 2026-09-22 02:40
---
Clarification after calibration audit: calibration is stored per NeuralFunction, with one fitted temperature per scalar head. A scalar function has one temperature; a flat-interface function has a field-keyed temperature map. Model the manifest accordingly.
---

author: @codex
created: 2026-09-22 02:44
---
TASK-2 uses JSON Schema draft 2020-12. Local Python jsonschema 4.10.3 is available for objective validation; no new dependency is required.
---

created: 2026-09-22 03:12
---
Implemented strict Draft 2020-12 schemas for NeuralFunction IR and the deployable application manifest, plus refund IR and manifest examples. Positive validation passes with format checks; negative cases for missing model metadata, invalid temperature, unknown resource properties, and unsafe paths are rejected. Canonical serialization golden vector SHA-256 is 3273c3e7b704fce97a131fbdf1c3cd2bef0cd3896c1f4ebbc6857921b534a162. Independent agent audits are in progress.
---

author: @codex
created: 2026-09-22 03:38
---
Independent review found and drove fixes for two integration blockers: compiler-emitted IR now has source, trained, and verified lifecycle variants, and deployable functions now embed the exact input descriptors required for runtime validation and canonical serialization. Follow-up fixes add literal unions and optional object fields, stable duplicate ordinals, semantic JSON hashing that preserves signed zero, real fixture identities, canonical decimals, per-head calibration and pair-consistency evidence, explicit sigmoid/softmax ABI, per-function adapter selection, a portable pointer schema, hardened resource paths, and deterministic cache domains.
---

author: @codex
created: 2026-09-22 03:40
---
Final verification passed after the audit-driven corrections. Three independent reviewers report no remaining TASK-2 blocker; all declared hashes and compiler-to-trainer-to-exporter handoffs reproduce.
---

author: @codex
created: 2026-09-22 05:11
---
TASK-5.2 schema-validation follow-up: aligned both v1 schemas with valid empty-string object/output fields. IR head path '/' and manifest path [''] now validate; input/output field name constraints allow empty while type names and interpolation identifiers remain nonempty. All schemas and committed examples were revalidated with Draft202012Validator.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Defined and validated the complete SemantScript v1 compiler-to-trainer-to-runtime contract: lifecycle-aware NeuralFunction IR, deterministic typed input and semantic hashing, immutable shared-encoder/adapter/head artifact layout, calibrated per-head manifest records, secure loading rules, and reproducible refund fixtures. All five acceptance criteria and independent reviews pass.
<!-- SECTION:FINAL_SUMMARY:END -->
