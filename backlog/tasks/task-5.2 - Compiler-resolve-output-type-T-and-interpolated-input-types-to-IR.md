---
id: TASK-5.2
title: 'Compiler: resolve output type T and interpolated input types to IR'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-22 05:28'
labels:
  - compiler
milestone: m-1
dependencies:
  - TASK-5.1
references:
  - docs/research/laya-analysis.md
documentation:
  - compiler/README.md
  - runtime/README.md
modified_files:
  - compiler/src/analyze-site.ts
  - compiler/src/core-symbols.ts
  - compiler/src/decimal.ts
  - compiler/src/index.ts
  - compiler/src/ir-types.ts
  - compiler/src/sema-sites.ts
  - compiler/src/semantic-json.ts
  - compiler/src/source-ir.ts
  - compiler/test/analysis.test.mjs
  - runtime/src/index.ts
  - runtime/test/package.test.mjs
  - package.json
  - package-lock.json
  - SPEC.md
  - IR.md
  - schemas/neural-function.v1.schema.json
  - schemas/application-artifact.v1.schema.json
parent_task_id: TASK-5
ordinal: 7000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The head shape is derived entirely from T, and the input schema from the ${} interpolations. This is where type safety is established structurally. Uses the TS type checker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 boolean, string-literal union, enum and flat interface output types produce the correct head spec in IR
- [x] #2 Each ${} interpolation produces an input schema entry with its resolved TS type
- [x] #3 Unsupported output types (free string, nested objects, arrays) produce a clear compile error naming the site
- [x] #4 Emitted IR validates against the schema from the IR task
- [x] #5 Ordered unions and bounded integers are emitted as ordinal head specs; unordered unions and enums as nominal
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define compiler analysis results and located diagnostics on top of discovered SemaSite nodes. 2. Resolve supported output types by TypeScript symbol identity into IR head specs: boolean, nominal literal unions/enums, Ordinal tuples, bounded integers/numbers, and flat structured interfaces. 3. Resolve each template interpolation into a deterministic input-schema entry while preserving prompt segment order. 4. Assemble schema-valid NeuralFunction IR and report unsupported output/input shapes at the source site. 5. Add focused checker fixtures for every supported/unsupported shape, validate emitted objects against the versioned schema, run the repository gate, and obtain independent audits.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented symbol-aware TypeScript output/input analysis, deterministic semantic ordering, exact bounded-number grids, generic/inherited flat-output resolution, located diagnostics, and schema-valid source-stage IR assembly. Added public core marker/result declarations and exhaustive compiler/runtime coverage. Verified with npm run check, npm audit, schema validation, package dry-runs, git diff --check, and two independent acceptance audits.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @codex
created: 2026-09-22 02:37
---
TASK-1 SPEC adds Ordinal<Tuple>, BoundedInt<Min,Max>, and BoundedNumber<Min,Max,Step>; all are resolved by @semantscript/core export identity. BoundedNumber uses exact decimal-rational grid construction before one-time JS-number conversion. Include these when planning this task.
---

author: @codex
created: 2026-09-22 04:18
---
TASK-5.2 started immediately after TASK-5.1. The implementation will consume symbol-aware discovery results and keep type resolution deterministic and aligned with SPEC.md and neural-function.v1.schema.json.
---

author: @codex
created: 2026-09-22 05:06
---
Implementation now resolves all required scalar/flat output heads, exact decimal grids, generic and inherited marker fields, and recursively typed interpolations (including inferred/intersection objects). Input unions use the IR semantic-JSON byte order. A production source-stage IR builder is validated with Ajv 2020 against the committed schema. Full repository check passes; final independent blocker audit is in progress.
---

created: 2026-09-22 05:28
---
Final audit fixed a false conflict for equivalent generic wrapper aliases by retaining every inherited environment and comparing each field's semantic HeadSpec. Independent repros now pass for equivalent multilayer wrappers, reordered unions, boolean normalization, unused generic arguments, and diamonds; semantically different exact-decimal markers still fail with located diagnostic 9105. Full repository check and npm audit pass.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved sema output types and interpolation types into deterministic v1 IR. Supports boolean, literal unions, enums, Ordinal, BoundedInt, BoundedNumber, generic/inherited flat interfaces, and recursive input schemas; rejects unsupported shapes with located diagnostics. Production IR validates against the committed schema.
<!-- SECTION:FINAL_SUMMARY:END -->
