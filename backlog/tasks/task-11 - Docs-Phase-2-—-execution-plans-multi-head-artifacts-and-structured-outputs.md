---
id: TASK-11
title: 'Docs: Phase 2 — execution plans, multi-head artifacts and structured outputs'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-20 02:47'
updated_date: '2026-09-25 03:03'
labels:
  - docs
milestone: m-2
dependencies:
  - TASK-6.5
ordinal: 36000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 2 introduces the concepts users will find hardest to reason about: how independent sema expressions fuse into one encoder pass, how dependent ones become stages, and how interface outputs map to heads. Without documentation these look like magic and users cannot predict performance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A concepts page explains the neural execution plan with a worked example showing source, DAG and stages
- [x] #2 The artifact reference is updated for shared encoder, adapters and multiple heads
- [x] #3 A page documents structured interface outputs: which shapes are supported, how fields become heads, and what verification reports per field
- [x] #4 Scaling benchmark results are written up with guidance on when fusing helps
- [x] #5 The docs index and Phase 1 pages are updated where behavior changed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. New concepts page docs/execution-plans.md: how independent sema expressions fuse into one encoder pass (canonical-input fusion in callStage and request scopes), how dependencies become stages (may-analysis through locals), routed domains and depths per stage; worked example with a runnable docs/examples/execution-plan.sem.ts (compiled by compiler/test/docs-examples.test.mjs) showing the source, the dependency DAG and the emitted stages, plus the refund-service chain.
2. New page docs/structured-outputs.md: supported flat-interface shapes, how each field becomes a head (outputPath JSON pointer, head-NNN files, per-field temperature), what verification reports per field (HeadVerificationV1, worst-head function metrics, exact-match accuracy), the ObjectSemaResult shape and the all-fields confidence policy.
3. Update docs/ir-and-artifact-reference.md for the shared encoder, several adapters and encoder prefixes (bundle domains and per-stage adapterRefs, model.encoderDepth, manifest function encoderRef, resource layout with depth prefixes and per-domain adapters, head-NNN per field).
4. New page docs/scaling-results.md: the parallel-head and batch scaling results (results-stage-scaling-2026-09-24), the depth sweep and typed-decisions one-stage numbers, with guidance on when fusing helps and when it does not.
5. Update docs/index.md and the Phase 1 pages (language reference output types and runtime behavior link to the new pages); verify with the docs examples test, prettier and lint.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added docs/execution-plans.md (worked example docs/examples/execution-plan.sem.ts: four decisions over one support request, one route->template dependency, stages [3,1]; the docs examples test now compiles it and asserts the dependency edge and stage split), docs/structured-outputs.md (allowed flat-interface shapes, real IR output/model block for the Triage example with per-field JSON-pointer heads, per-field training and verification table, ObjectSemaResult and the all-fields policy, when to use one), docs/scaling-results.md (heads-per-stage 1/10/50 at 21.0/22.5/29.7 ms vs 20.8/217/1133 ms, batch 1..64 linear, typed-decisions one-stage cases, depth sweep 4/6/12/22 latency and accuracy, guidance). Updated docs/ir-and-artifact-reference.md (bundle domains and per-stage adapterRefs, model.encoderDepth and routed refs, manifest function encoderRef, release layout with depth prefixes and per-domain adapters, ABI note on per-field heads), docs/index.md (three new pages) and docs/language-reference.md (output types and runtime behavior link to the new pages). Verified: docs-examples test passes (6 example files), lint:node clean, prettier clean on touched docs, every relative link in the touched pages resolves.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 2 documentation: a concepts page on execution plans with a compiler-verified worked example (source, dependency DAG, stages, what the runtime fuses and why chains cost a stage), a structured-outputs page (shapes, field-to-head mapping with the real IR, per-field verification, diagnostic result and policy), a scaling-results page collecting the parallel-head, batch, one-stage application and depth-routing measurements with guidance on when fusing helps, and the IR and artifact reference updated for routed adapters, encoder prefixes and per-field heads. Index and Phase 1 pages link the new material. Verified with the docs examples compile test (which now asserts the worked example's plan), lint, prettier and a link check.
<!-- SECTION:FINAL_SUMMARY:END -->
