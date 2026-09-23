---
id: TASK-6.2
title: 'Compiler: dependency DAG and neural execution plan'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-22 18:52'
labels:
  - compiler
milestone: m-2
dependencies:
  - TASK-5.3
documentation:
  - IR.md
  - compiler/README.md
  - schemas/ir-bundle.v1.schema.json
modified_files:
  - compiler/src/bundle-ir.ts
  - compiler/src/execution-plan.ts
  - compiler/src/compile.ts
  - compiler/src/identity.ts
  - compiler/src/index.ts
  - compiler/test/bundle-ir.test.mjs
  - compiler/test/execution-plan.test.mjs
  - compiler/test/compile.test.mjs
  - compiler/test/identity.test.mjs
  - schemas/ir-bundle.v1.schema.json
  - IR.md
  - compiler/README.md
parent_task_id: TASK-6
ordinal: 19000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The compiler can see which sema expressions depend only on request inputs and which consume other sema results. Emit an execution plan (stages of fused heads) analogous to a database query plan. Transcript turn 9 points 5-6.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Compiler builds a DAG of sema sites from data dependencies
- [x] #2 Independent sites are grouped into one stage; dependent sites into later stages
- [x] #3 The plan is emitted in the IR bundle and is human-readable
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define a versioned, human-readable IR-bundle envelope containing unchanged neural-function records plus a deterministic execution-plan DAG materialized as stages. 2. Build data-dependency edges from compiler-resolved sema result provenance through stable TypeScript symbols and immutable/local aliases, label edges by consumer input names, and diagnose cycles or unsupported ambiguous provenance conservatively. 3. Topologically assign minimum-depth stages with canonical source ordering, integrate the plan into compilation sealing and atomic bundle emission without changing per-function semantic identities. 4. Add schema, direct/alias/chain/diamond/shadowing/determinism/cycle/tamper coverage and update IR/compiler documentation. 5. Run independent audits and focused gates before the bounded full repository suite, then record evidence and finalize all acceptance criteria.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a bounded, symbol-aware compiler DAG over sema sites. Provenance follows aliases, destructuring, derived expressions, same-program imports, initialized mutable bindings, direct assignments, and direct property writes; it uses TypeScript symbol identity, per-consumer memoization, cross-file write indexing, source-order filtering within an execution container, bounded traversal/edge expansion, iterative cycle discovery, and a heap-backed deterministic topological pass.

Added a closed semantscript.ir-bundle v1 envelope with canonical functions, labeled producer/consumer/input dependencies, dense minimum-depth stages, relational validation, deterministic exact-JSON serialization, whole-plan sealing, and atomic emission. Per-function semantic hashes and IDs remain unchanged because plan metadata is outside their semantic projection.

Verification: independent acceptance audit found all three ACs covered. Adversarial audit reproduced and drove fixes for mutable reassignment, cross-file property writes, context-pruned memoization, uncharged edge expansion, quadratic dependency ordering, quadratic ready-queue behavior, and false forward self-cycles; final re-audit found no blocker. Final compiler suite passed 8/8 test files at 534,596 KiB peak RSS with zero swap. Final repository `NODE_OPTIONS=--max-old-space-size=512 npm run check` passed lint, TypeScript build, compiler 8/8 files, runtime 7/7 files, CLI 1/1 file, and Python 2/2 tests at 543,744 KiB peak RSS with zero swap. Compiler package dry-run includes the new emitted modules.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-09-22 18:12
---
TASK-6.2 started as the next ready item in ordinal order after TASK-5.10. The task is scoped to compiler-emitted metadata: it will describe the deterministic DAG and stages for TASK-6.3, without prematurely changing the synchronous runtime call ABI or claiming control-flow batching semantics.
---

created: 2026-09-22 18:20
---
2026-09-22 implementation checkpoint: the plan is compiler metadata over statically proven potential data dependencies. Stages are deterministic minimum-depth topological strata, not an eager execution promise across branches, loops, or call frames; request liveness remains TASK-6.3. The audit found that initialized mutable bindings must not be silently treated as independent, so direct initialized let/var provenance is included conservatively. Function-return and assignment-flow provenance remain explicitly outside this first intraprocedural/symbol-initializer boundary.
---

created: 2026-09-22 18:32
---
2026-09-22 implementation checkpoint: compiler integration now emits a closed versioned semantscript.ir-bundle envelope with per-function records, labeled producer/consumer/input edges, and minimum-depth stages. The builder follows TypeScript symbol identity across initialized aliases, destructuring, derived expressions, cross-file aliases, and conservative initialized mutable bindings; it has bounded provenance traversal and iterative graph-cycle reporting. Plan sealing now covers the whole envelope. Focused compiler suite passes 8/8 test files under NODE_OPTIONS=--max-old-space-size=512 (peak RSS 537,560 KiB, zero swaps); repository lint and diff checks pass. Independent acceptance and adversarial audits are in progress before the full repository gate.
---

created: 2026-09-22 18:52
---
2026-09-22 final verification: all ACs are satisfied. The bounded full repository gate passed at 543,744 KiB peak RSS with zero swap. The earlier assignment-flow scope note is superseded: v1 now indexes direct assignments and direct property writes across Program source files, while indirect mutation behind calls/accessors and interprocedural returns remain outside this metadata plan. TASK-6.3 must preserve/translate the emitted plan into executable runtime scheduling.
---

created: 2026-09-22 18:52
---
TASK-6.2 completed after final acceptance review; all three criteria, implementation notes, documentation references, modified files, final summary, audits, and bounded verification evidence are recorded.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Compiler now emits a deterministic, human-readable, schema-valid execution-plan DAG in a versioned IR bundle. Independent sites share stage 0, consumers occupy minimum later stages, labeled dependencies survive aliases and direct writes, and compiler emission rejects mutated or relationally invalid plans. Bounded full-repository verification and independent audits passed.
<!-- SECTION:FINAL_SUMMARY:END -->
