---
id: TASK-2
title: Define the SemantScript IR and compiled artifact format
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
labels:
  - spec
  - compiler
milestone: m-0
dependencies:
  - TASK-1
references:
  - PLAN.md
priority: high
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The compiler emits IR that the trainer consumes and the runtime loads. All three teams need one schema. Transcript turn 10 sketches the shape (function id, inputs, output head, model refs, runtime confidence, training provenance, verification) but no concrete format exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A versioned JSON schema for a NeuralFunction IR record exists and includes: function id, input schema, output head spec, model/adapter/head refs, confidence requirement, training provenance, verification stats
- [ ] #2 The on-disk artifact layout for an application (shared encoder + adapter + heads + manifest) is documented
- [ ] #3 At least one hand-written example IR file for the refund-decision expression validates against the schema
<!-- AC:END -->
