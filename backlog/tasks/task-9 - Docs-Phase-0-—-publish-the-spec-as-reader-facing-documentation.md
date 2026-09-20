---
id: TASK-9
title: 'Docs: Phase 0 — publish the spec as reader-facing documentation'
status: To Do
assignee: []
created_date: '2026-09-20 02:47'
labels:
  - docs
milestone: m-0
dependencies:
  - TASK-1
  - TASK-2
  - TASK-4
ordinal: 34000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 0 produces the contract everyone builds against. A spec that only its authors can read will be re-derived from code later. Turn SPEC.md and the IR schema into documentation a new contributor can read cold, and establish the docs structure the later phases extend.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 docs/ directory exists with an index that links every document produced so far
- [ ] #2 A language reference page covers sema<T> syntax, v1 output types, examples, constraints, @confidence and withConfidence with a runnable example for each
- [ ] #3 An IR and artifact reference page documents every schema field and the on-disk artifact layout
- [ ] #4 A CONTRIBUTING page explains the repo layout, the two toolchains, and how to run lint and tests for both halves
- [ ] #5 Every page was read by someone other than its author and their questions were answered in the text
<!-- AC:END -->
