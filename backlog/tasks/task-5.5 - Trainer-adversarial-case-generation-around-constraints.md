---
id: TASK-5.5
title: 'Trainer: adversarial case generation around constraints'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-22 23:49'
labels:
  - trainer
milestone: m-1
dependencies:
  - TASK-5.4
modified_files:
  - trainer/README.md
  - trainer/src/semantscript_trainer/__init__.py
  - trainer/src/semantscript_trainer/adversarial.py
  - trainer/src/semantscript_trainer/adversarial_contract.py
  - trainer/src/semantscript_trainer/adversarial_prompt.py
  - trainer/src/semantscript_trainer/case_contract.py
  - trainer/src/semantscript_trainer/constraints.py
  - trainer/src/semantscript_trainer/dataset.py
  - trainer/src/semantscript_trainer/teacher.py
  - trainer/src/semantscript_trainer/teachers/anthropic.py
  - trainer/src/semantscript_trainer/teachers/ollama.py
  - trainer/tests/test_adversarial.py
  - trainer/tests/test_anthropic_teacher.py
  - trainer/tests/test_constraints.py
  - trainer/tests/test_dataset.py
  - trainer/tests/test_ollama_teacher.py
  - trainer/tests/test_public_api.py
  - compiler/src/definition.ts
  - compiler/test/definition.test.mjs
parent_task_id: TASK-5
ordinal: 10000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deterministic constraints (never/always) define hard boundaries. Training data must be dense around those boundaries so the model learns them, and the verifier can check them. Transcript turn 9 point 8. Beyond constraints, every generated case should get a counterfactual twin: a minimal edit to the inputs that flips the label, with the teacher stating the reason. A small encoder otherwise learns surface correlations (customer name, order size) instead of the actual policy; counterfactual pairs force it to learn the decision boundary. This is one of the central techniques behind Jev-style and Laya-style decision models.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 For each never/always constraint, cases are generated on both sides of the boundary
- [x] #2 Constraint-violating labels are never emitted as training targets
- [x] #3 Adversarial cases are tagged so the verifier can report constraint accuracy separately
- [x] #4 For every generated case, the teacher produces a counterfactual twin: a minimal input edit that changes the label, plus the stated reason for the flip
- [x] #5 Counterfactual twins are stored as linked pairs so the verifier can report pair-consistency accuracy (both members correct) separately from per-case accuracy
- [x] #6 Counterfactual generation is a config option with a ratio so data cost can be tuned
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a bounded, JavaScript-compatible trainer evaluator for the closed v1 constraint AST and enforce every active always/never rule on gold, synthetic, boundary, counterfactual, and cached cases.
2. Add a separate adversarial-teacher capability and closed structured-output contracts/prompts for locally verifiable boundary pairs and counterfactual twins with reasons; implement it in Anthropic and Ollama without widening the base Teacher protocol.
3. Add an immutable, separately cached semantscript.adversarial-dataset v1 sidecar keyed by the base dataset digest, IR, teacher descriptor, normalized config, and explicit contract revisions. Preserve TASK-5.4 dataset v1 and exact total_cases semantics.
4. Generate and locally verify a predicate-false/predicate-true pair for each two-sided constraint, tag both rows, and fail with a typed bounded error when a compiler-valid predicate has no synthesizable opposite side.
5. Select a deterministic ratio of non-gold base rows, require exactly one changed JSON path plus a different output and bounded nonblank teacher reason, and store both members through unique linked pair records.
6. Add evaluator, generator/cache, provider, tamper, ratio, public API, and regression tests; document count/ratio/failure semantics; run focused and memory-capped repository checks followed by independent audits.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a bounded JavaScript-compatible v1 constraint evaluator with compile-once structural and relational validation, scalar-output support checks, request-wide UTF-16 work limits, and enforcement on gold, synthetic, boundary, counterfactual, and cached rows.

Added a separate AdversarialTeacher capability, closed boundary/counterfactual response contracts and prompts, Anthropic and Ollama implementations, deterministic ratio selection, exact one-path counterfactual verification, stable pair linkage, typed bounded one-sided failures, and an immutable canonical adversarial sidecar cache. TASK-5.4 base dataset size and format remain unchanged.

Narrowed compiler constraint access/equality to JSON-data/scalar semantics so emitted IR matches trainer evaluation. Added provider, evaluator, cache-tamper, ratio, pair-bijection, binary64, aggregate-budget, and compiler parity regressions plus trainer documentation.

Final parity hardening also tracks optional dot and bracket property presence from TypeScript symbols independently of strictNullChecks. Direct equality observation remains supported, while dereference or required-operand use is rejected before IR emission.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @codex
created: 2026-09-22 22:03
---
Started as the next ready ordinal item after completing TASK-5.4 and validating the persistent WSL Ollama GPU service. Beginning reconnaissance of the v1 constraint AST/evaluator and dataset/cache contracts before recording the implementation plan. Parallel read-only audits will cover boundary synthesis and counterfactual pair integration.
---

author: @codex
created: 2026-09-22 22:11
---
Reconnaissance complete. Three read-only audits agree on a separate cached adversarial sidecar rather than changing the closed TASK-5.4 dataset v1. The base generator will still gain local constraint-label enforcement without altering serialized bytes. Ratio 1.0 twins every teacher-generated non-gold base row; ratio 0 disables ordinary twins, while mandatory boundary pairs are independent. Minimal edit is defined and verified as exactly one changed JSON path. Compiler-valid constant or otherwise one-sided predicates make a literal two-side guarantee impossible, so generation will fail explicitly and within a fixed attempt budget rather than omit a side, invent a row, or retry without bound.
---

created: 2026-09-22 23:19
---
Implementation and verification are complete pending the final semantic audit. The memory-bounded full repository check passed with NODE_OPTIONS=--max-old-space-size=512: all 16 Node test files passed, Python reported 227 passed and 1 expected opt-in live-Ollama skip, maximum RSS was 541936 KiB, and no swap was used. Final acceptance and security audits found no blockers.
---

created: 2026-09-22 23:33
---
The first semantic audit found and we fixed four cross-runtime edge cases before finalization: TypeScript configurations without noUncheckedIndexedAccess now reject possibly missing sequence operands, index expressions, and base dereferences before emitting constraint IR; fixed in-range tuple indices remain valid; trainer relational inference preserves unary numeric literals such as -0 object keys; and constraint-label support membership now uses JavaScript numeric equality for signed zero. The repeated bounded full check passes: all 16 Node test files, Python 230 passed with 1 expected opt-in live-Ollama skip, maximum RSS 534136 KiB, and zero swaps.
---

created: 2026-09-22 23:36
---
Finalization: all six acceptance criteria are checked. The final acceptance, security/boundedness, and compiler-to-trainer parity audits found no blockers after the edge-case fixes, and the repeated memory-bounded full repository check passed.
---

created: 2026-09-22 23:37
---
Reopened after a late final audit reproduced one remaining compiler-to-trainer parity gap under strictNullChecks=false: dereferencing an optional property could be emitted by the compiler but is rejected by trainer relational validation. AC2 is temporarily unchecked until optional-property presence provenance and regressions pass the bounded full suite.
---

created: 2026-09-22 23:49
---
Re-finalized after the optional-property gap was fixed. TypeScript optional dot, literal-bracket, and dynamic-key accesses now retain maybe-missing provenance even with strictNullChecks disabled; direct equality remains valid, while dereference and required typed use fail before IR emission. The bounded full check and independent parity re-audit are clean, so AC2 is rechecked.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Delivered deterministic adversarial training-data generation around v1 constraints without changing TASK-5.4 base-dataset counts or format. Every synthesizable constraint receives locally verified false/true boundary rows; selected synthetic rows receive exact one-path, label-flipping twins with bounded reasons and stable pair links; every emitted or cached label is revalidated against the IR and active constraints.

The implementation includes separate Anthropic/Ollama adversarial teacher contracts, a canonical independently keyed sidecar cache, deterministic configurable ratios, typed bounded failures for one-sided predicates, request-wide resource limits, and compiler/trainer semantic parity hardening across optional fields, sequence indices, unary numeric keys, binary64 values, and signed zero.

Final verification: memory-bounded npm run check passed under a 512 MiB Node heap; all 16 Node test files passed; Python reported 230 passed and 1 expected opt-in live-Ollama skip; peak RSS was 533924 KiB with zero swaps. Independent acceptance, security/boundedness, and final compiler-to-trainer semantic re-audits found no remaining blockers.
<!-- SECTION:FINAL_SUMMARY:END -->
