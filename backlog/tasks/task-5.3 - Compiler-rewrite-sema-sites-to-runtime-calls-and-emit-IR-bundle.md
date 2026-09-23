---
id: TASK-5.3
title: 'Compiler: rewrite sema sites to runtime calls and emit IR bundle'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-22 15:00'
labels:
  - compiler
milestone: m-1
dependencies:
  - TASK-5.2
documentation:
  - compiler/README.md
  - runtime/README.md
  - SPEC.md
modified_files:
  - compiler/src/identity.ts
  - compiler/src/definition.ts
  - compiler/src/rewrite.ts
  - compiler/src/compile.ts
  - compiler/src/index.ts
  - compiler/src/source-ir.ts
  - compiler/test/identity.test.mjs
  - compiler/test/definition.test.mjs
  - compiler/test/compile.test.mjs
  - compiler/package.json
  - compiler/README.md
  - runtime/src/index.ts
  - runtime/test/package.test.mjs
  - runtime/README.md
  - SPEC.md
parent_task_id: TASK-5
ordinal: 8000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final compiler step for Phase 1: replace each sema expression with __sema.call(functionId, { inputs }) and write the IR bundle that the trainer consumes. Function ids must be stable across builds so unchanged expressions can be cached later.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Compiled JS contains a __sema.call with a deterministic function id and the interpolated variables as inputs
- [x] #2 The natural-language text is absent from the compiled JS
- [x] #3 An IR bundle (one record per site) is written alongside the JS output
- [x] #4 Function id is unchanged when the expression text and types are unchanged, and changes when either changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the canonical semantic hash/function-id contract and bundle shape from IR.md and the v1 schema. 2. Add a compiler API that analyzes each discovered site, computes deterministic content-derived identities, and rewrites tagged templates to __sema.call(functionId, { inputs }). 3. Emit JavaScript without prompt text plus a deterministic one-record-per-site IR bundle beside the output. 4. Cover multi-site/nested/configured forms, stable/change-sensitive ids, diagnostics, and disk emission; run full gates and independent audits.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented canonical semantic hashing and stable function IDs; static definition parsing for examples, constraints, and confidence; fail-closed TypeScript rewriting; deterministic source-stage IR bundle emission; source-map prompt stripping; compiler/runtime ABI integration; and bounded definition/compiler tests. Safety hardening seals plans, validates exact Program coverage, snapshots emit inputs, canonicalizes symlinked paths, stages every output, preflights targets, and commits the bundle last. Verification: NODE_OPTIONS=--max-old-space-size=512 npm run check passed; compiler and runtime focused tests passed; npm pack dry-run passed for all workspaces; npm audit reported 0 vulnerabilities; git diff --check passed; three independent final audits reported no blockers.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-09-22 05:28
---
TASK-5.3 started immediately after TASK-5.2. Work will preserve source-stage analysis semantics, remove natural-language prompt text from emitted JavaScript, and derive cache-stable identities only from the canonical semantic payload.
---

created: 2026-09-22 14:10
---
OOM diagnosis: TypeScript 6 semantic diagnostics recursively contextualized generic SemaConstraint<T> entries through SemaOptions<T>. A bounded repro reached ~14.5 GiB RSS before timeout, prior to SemantScript definition parsing. The public options container now uses SemaConstraint<unknown>[] while the compiler explicitly validates each constraint output against the site's exact T. The formerly runaway definition test passes under a 512 MiB V8 cap at ~270 MiB RSS. Compiler identity fixtures now release one Program at a time and compiler test files run with concurrency 1 to bound peak memory.
---

author: @codex
created: 2026-09-22 14:27
---
Independent audits found and the implementation now fixes three additional safety/correctness edges: vacuous constant-false constraints no longer conflict, bracket-form enum members are accepted, and bundle paths cannot overwrite TypeScript sources. Static definition parsing is bounded by node/depth budgets with linear constant folding, stale rewrite plans fail closed, and regression coverage now includes all four sema forms, the committed refund example, semantic changes from examples/constraints/confidence, UTF-8 path ordering, and source-path collision rejection. Focused compiler/runtime tests pass under a 512 MiB V8 cap; repository-wide checks remain intentionally deferred until re-audit completes.
---

author: @codex
created: 2026-09-22 15:00
---
Final verification completed after the OOM-safe implementation and adversarial re-audits. The full repository check passed under a 512 MiB Node heap cap, package dry-runs succeeded, npm audit found 0 vulnerabilities, and all three independent reviewers returned no blockers.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Phase 1 compiler emission step. Every supported sema site now compiles to a collision-safe runtime call with a deterministic content-derived ID, while prompts and inline options are excluded from JavaScript and source maps. The compiler writes a schema-valid deterministic IR record per site and rejects diagnostics, stale/forged/incomplete plans, unsafe output paths, and bounded-definition violations before committing artifacts. The TypeScript 6 OOM was traced to recursive contextual inference in the constraint options type and fixed without weakening compiler validation.
<!-- SECTION:FINAL_SUMMARY:END -->
