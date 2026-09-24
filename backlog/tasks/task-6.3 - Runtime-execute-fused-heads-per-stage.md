---
id: TASK-6.3
title: 'Runtime: execute fused heads per stage'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-24 16:27'
labels:
  - runtime
milestone: m-2
dependencies:
  - TASK-6.1
  - TASK-6.2
parent_task_id: TASK-6
ordinal: 20000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consume the execution plan: run the encoder once per stage and fan out to all heads in that stage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A stage with N heads performs one encoder pass and N head passes
- [x] #2 Stage outputs feed subsequent stages as inputs
- [x] #3 Results are identical to executing each function independently
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Worker protocol: add an invoke-stage message carrying an ordered list of (functionId, canonicalInput) requests. The worker decodes every canonical input, groups the stage by identical canonical bytes, tokenizes and runs the encoder once per distinct input, runs each adapter once per distinct (input, adapterRef), and runs every function's heads; it returns the per-function results in request order plus pass counts (encoder, adapter, head) so the fusion is observable. A single call stays a stage of one.
2. Inference runtime: callStage(requests) mirrors call: validates ids and inputs, sizes the shared response buffer for the whole stage, posts one message, waits synchronously, decodes each result with its function's response schema and returns results plus pass counts; protocol and timeout handling unchanged.
3. Artifact runtime and public API: dispatchArtifactStage applies each function's confidence policy and fallback exactly as a single call does (factored into one helper) and returns results in order; __sema.callStage and handle.callStage expose it; executePlan(plan, provide) runs the compiler's execution plan stage by stage, calling provide(stage, resultsSoFar) for the inputs of that stage's functions, so stage outputs feed later stages, and returns every function's result with per-stage pass counts.
4. Tests: inference-runtime tests with several functions over one fixture encoder proving one encoder pass and N head passes for shared inputs and separate passes for distinct inputs, and byte-identical results to independent calls; artifact-runtime tests for callStage policies and executePlan ordering (a later stage consuming an earlier result); the Python two-function artifact test calls callStage from Node and asserts one encoder pass for both functions.
5. Scope note: the compiler still emits one call per site; rewriting sites into stage calls is the epic-level step that consumes this primitive.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented fused stage execution (commit above): worker invoke-stage groups by canonical bytes (one encoder pass per distinct input, one adapter pass per distinct input and adapter, every head), framed response decoded per result by the single-call parser, pass counts returned; InferenceRuntime.callStage, handle.callStage, __sema.callStage and executeSemaPlan(plan, provide) with plan validation (consecutive stage indices, unique functions, dependencies satisfiable in stage order). Evidence: runtime tests assert passes {encoder 1, adapter 1, head 2} for two functions with identical inputs and {2, 2, 3} when inputs differ, results deep-equal to independent calls, a two-stage plan whose second stage inputs derive from the first stage's result, and the Python two-function artifact exported from the trainer executes both heads with one encoder pass from Node. Suites: runtime 99 passed, repository lint clean, Python 485 passed. The compiler still emits one call per site; rewriting sites into stage calls is the remaining epic-level step.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added stage execution to the runtime: the worker fuses a stage's requests by canonical input so N heads over identical inputs cost one encoder pass and one adapter pass, results are returned in order with observable pass counts, and executeSemaPlan runs the compiler's execution plan stage by stage with earlier results feeding later inputs. Verified with runtime tests (pass counts 1/1/2 for shared inputs, 2/2/3 for distinct inputs, results identical to independent calls, a two-stage plan), the Node round trip of the trainer's two-function artifact, repository lint, 99 runtime and 485 Python tests.
<!-- SECTION:FINAL_SUMMARY:END -->
