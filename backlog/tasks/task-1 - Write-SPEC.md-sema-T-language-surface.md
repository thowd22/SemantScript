---
id: TASK-1
title: 'Write SPEC.md: sema<T> language surface'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-20 19:18'
labels:
  - spec
milestone: m-0
dependencies: []
references:
  - docs/research/laya-analysis.md
priority: high
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The whole project hinges on one primitive, and every downstream component (compiler, trainer, runtime) needs an agreed contract to build against. The transcript (turns 9-11, 14) settled on neural *expressions* inside ordinary TS functions using a tagged template, but the exact grammar, allowed input/output types and modifier forms were never written down in one place.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SPEC.md defines the sema<T> tagged-template syntax including ${} interpolated inputs
- [ ] #2 SPEC.md enumerates v1 output types: boolean, string-literal unions, enums, bounded numbers, flat interfaces of those; free-text string is explicitly out of scope
- [ ] #3 SPEC.md defines the examples, constraints (never/always) and @confidence forms and the withConfidence API
- [ ] #4 SPEC.md states the input isolation rule: no ambient access to DB, network, filesystem or global state
- [ ] #5 SPEC.md defines an ordinal output kind (ordered string-literal unions and bounded integers) distinct from nominal unions, and states what the runtime returns for it (value, expected value, distribution)
- [ ] #6 SPEC.md defines confidence as calibrated top-1 probability and uncertainty as normalized entropy, and specifies the withConfidence return shape accordingly
<!-- AC:END -->
