---
id: TASK-9
title: 'Docs: Phase 0 — publish the spec as reader-facing documentation'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-20 02:47'
updated_date: '2026-09-24 19:33'
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
- [x] #1 docs/ directory exists with an index that links every document produced so far
- [x] #2 A language reference page covers sema<T> syntax, v1 output types, examples, constraints, @confidence and withConfidence with a runnable example for each
- [x] #3 An IR and artifact reference page documents every schema field and the on-disk artifact layout
- [x] #4 A CONTRIBUTING page explains the repo layout, the two toolchains, and how to run lint and tests for both halves
- [ ] #5 Every page was read by someone other than its author and their questions were answered in the text
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. docs/index.md linking every document so far: SPEC.md, PLAN.md, AGENTS.md, the package READMEs (compiler, runtime, trainer, model, cli, examples, benchmarks and the refund benchmark and its program), the schemas, the Backlog decisions, docs/research/laya-analysis.md, the benchmark result write-ups under benchmarks/refund/data, and the new pages.
2. docs/language-reference.md: sema<T> syntax (plain, configured, withConfidence), interpolated inputs and supported values, isolation, examples, always/never constraints, every v1 output kind (boolean, string unions, enums, Ordinal, BoundedInt, BoundedNumber, flat interfaces), @confidence and fallbacks, withConfidence result shapes, runtime errors and compile-time diagnostics, each with an example that lives in docs/examples/*.sem.ts and is compiled by a compiler test so it stays runnable.
3. docs/ir-and-artifact-reference.md: the NeuralFunction IR v1 record field by field (identity, source, definition, inputs, output heads, model binding, runtime policy, training provenance, verification), the IR bundle and execution plan, the application artifact manifest field by field, the pointer, the on-disk release layout, canonical input encodings and digests.
4. docs/CONTRIBUTING.md: repository layout, the Node and Python toolchains and how each is set up on a checkout, lint and tests for both halves, the Backlog workflow, and the machine-specific notes that live outside the repo.
5. AC5 (a reader other than the author) cannot be satisfied by me; the pages will be left for the user's read-through and the task stays open on that criterion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote docs/index.md (links SPEC, IR, AGENTS, PLAN, DEVELOPING, the schemas, golden examples, every package README, all nine Backlog decisions, the Laya analysis and every committed benchmark result directory), docs/language-reference.md (syntax forms, inputs and isolation, examples and constraints, the output-type table for boolean, string unions, string and numeric enums, Ordinal, BoundedInt, BoundedNumber and flat interfaces, @confidence and fallbacks, withConfidence result shapes, runtime errors, the diagnostic code families), docs/ir-and-artifact-reference.md (NeuralFunction record field by field, bundle and execution plan, manifest field by field including resources, heads, policy, verification and provenance, model ABI v1, the release layout and pointer, canonical-input v1 and v2, digests) and docs/CONTRIBUTING.md (layout table, Node and Python toolchains with the exact commands, lint and tests for both halves, npm run check, the Backlog workflow, conventions). AC2 evidence: every snippet quoted in the language reference is a file under docs/examples (refund-decision, examples-and-constraints, output-types with all eight output kinds, confidence, inputs) that compiler/test/docs-examples.test.mjs type-checks and compiles to source IR, asserting the per-file function counts and the eight head kinds; it passes (compiler suite 70). Writing AC3 surfaced two stale contracts that were fixed in the same commit: schemas/application-artifact.v1.schema.json pinned compatibility.canonicalInput to v1 while the exporter emits v2 and the runtime accepts both (now an enum of both), and IR.md lacked the compact encoding (new section 3.1). Runtime 99 and compiler 70 tests pass, lint clean. AC5 (read by someone other than the author, questions answered in the text) needs a human reader; the pages are ready for that read-through and the criterion is left unchecked.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @claude
created: 2026-09-24 19:33
---
Docs are ready for a cold read: docs/index.md, docs/language-reference.md, docs/ir-and-artifact-reference.md, docs/CONTRIBUTING.md. AC5 needs someone other than the author to read them and note questions; I will fold the answers into the pages.
---
<!-- COMMENTS:END -->
