---
id: TASK-1
title: 'Write SPEC.md: sema<T> language surface'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-22 05:11'
labels:
  - spec
milestone: m-0
dependencies: []
references:
  - docs/research/laya-analysis.md
modified_files:
  - SPEC.md
priority: high
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The whole project hinges on one primitive, and every downstream component (compiler, trainer, runtime) needs an agreed contract to build against. The transcript (turns 9-11, 14) settled on neural *expressions* inside ordinary TS functions using a tagged template, but the exact grammar, allowed input/output types and modifier forms were never written down in one place.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SPEC.md defines the sema<T> tagged-template syntax including ${} interpolated inputs
- [x] #2 SPEC.md enumerates v1 output types: boolean, string-literal unions, enums, bounded numbers, flat interfaces of those; free-text string is explicitly out of scope
- [x] #3 SPEC.md defines the examples, constraints (never/always) and @confidence forms and the withConfidence API
- [x] #4 SPEC.md states the input isolation rule: no ambient access to DB, network, filesystem or global state
- [x] #5 SPEC.md defines an ordinal output kind (ordered string-literal unions and bounded integers) distinct from nominal unions, and states what the runtime returns for it (value, expected value, distribution)
- [x] #6 SPEC.md defines confidence as calibrated top-1 probability and uncertainty as normalized entropy, and specifies the withConfidence return shape accordingly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define the TypeScript-valid sema<T> tagged-template grammar, configured-tag overload, interpolated-input rules, examples, constraints, @confidence directive, and sema.withConfidence tag.
2. Specify every supported v1 output kind, including nominal versus ordinal types, bounded numeric support, flat-interface restrictions, and rejected types.
3. Define plain and diagnostic runtime results, calibrated confidence, normalized entropy, ordinal distributions/expectations, threshold/fallback behavior, and isolation guarantees.
4. Add canonical valid and invalid examples, then verify SPEC.md against all six acceptance criteria and downstream compiler/runtime contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created root SPEC.md with the canonical and configured sema tags, explicit interpolation/isolation rules, examples and deterministic constraints, @confidence thresholds, v1 nominal/ordinal/flat outputs, calibrated probability semantics, and scalar/object diagnostic result shapes. Initial checks: git diff --check passes; all six acceptance topics are present. Independent agent reviews are in progress.

Final validation: three independent read-only reviews mapped SPEC.md to all six acceptance criteria. Follow-up audit confirmed the configured and diagnostic tags are valid TypeScript shapes, SemaResult is non-distributive for unions, calibration is per scalar head, and the complete example follows the static-fixture rules. git diff --check reported no tracked-file whitespace errors; no TypeScript toolchain exists yet because repository scaffolding is a later task.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @codex
created: 2026-09-22 02:23
---
Research identified that examples, constraints, @confidence, and fn.withConfidence in the transcript came from the superseded named-function syntax. The plan resolves this with valid TypeScript overloads: sema<T>(options)`...` and sema.withConfidence<T>(options)`...`, while retaining @confidence(q) as a template header directive.
---

author: @codex
created: 2026-09-22 05:11
---
TASK-5.2 conformance follow-up: clarified that plain-object and flat-interface string-literal property names may be empty. This matches valid TypeScript and preserves exact names through IR/artifact JSON Pointer encoding.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the root SemantScript language contract covering sema tagged expressions, explicit inputs and isolation, examples and deterministic constraints, confidence thresholds and diagnostics, supported nominal and ordinal output types, bounded numeric grids, flat interfaces, runtime validation, and compile-time errors. Verified all six acceptance criteria through independent specification reviews and repository consistency checks; recorded downstream contract notes on TASK-2, TASK-5.1, TASK-5.2, TASK-5.10, and TASK-6.4.
<!-- SECTION:FINAL_SUMMARY:END -->
