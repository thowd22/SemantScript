---
id: TASK-16
title: >-
  Runtime enforcement of always/never constraints: rule violations become
  impossible by construction
status: To Do
assignee: []
created_date: '2026-09-26 19:52'
labels:
  - runtime
  - quality
milestone: m-7
dependencies: []
priority: high
ordinal: 71000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Constraints (always(() => predicate, output) and never(() => predicate, output) in the sema options) shape the training data (the constraints teacher, boundary pairs, counterfactual twins), gate the release (corpus and held-out violation rates, TASK-15.1) and drive explain, but the runtime does not apply them: a shipped model keeps a rule to 99.6% rather than 100%, and every residual miss (the Express example's stale-order misses on 2026-09-26, fixed only by widening the teacher range and retraining) sends the developer back to training. For accuracy-paramount rules the runtime should evaluate the active constraints for each call after the model answers and apply them deterministically, so a rule violation cannot reach the application, the model decides only the inputs the rules leave open, and the held-out gate becomes a quality signal rather than the sole guarantee. Background: the predicate AST is documented in docs/ir-and-artifact-reference.md; the trainer evaluates it in trainer/src/semantscript_trainer/constraints.py and the CLI in cli/src/constraint-eval.ts (TASK-14.7), pinned to each other by examples/constraints/predicate-vectors.v1.json; the runtime (runtime/src/artifact-runtime.ts, artifact-loader.ts, artifact-types.ts, schemas/application-artifact.v1.schema.json) reads the manifest, which today carries no constraints (explain reads them from the IR bundle). Constraints in v1 apply to scalar outputs. The confidence policy (@confidence thresholds, fallbacks, diagnostic results) and the request scopes, pass counts and stub artifact (@semantscript/core/testing) must keep working. The user's rule (2026-09-26): keep it simple, accuracy is paramount.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The artifact manifest records each function's constraints (predicate AST, kind, output and source text) at export, the schema and loader validate them, and older manifests without constraints still load
- [ ] #2 The runtime evaluates the active constraints for every call with the CLI's pinned evaluator (moved or shared, still pinned by the predicate vectors) and applies them deterministically: an always constraint forces its output, a never constraint excludes its output and the model's next-best allowed answer is returned, a conflict or an unevaluable predicate is a typed error with a generated remedy, and the diagnostic result and explain report say when a rule decided the answer and which one
- [ ] #3 Enforcement is measured and documented: the per-call overhead on the Express example is reported (target under 0.1 ms per constraint), the confidence policy, fallbacks, stub artifact, request scopes and pass counts behave unchanged (existing runtime and framework tests pass, new tests cover forced, excluded, conflicting and unevaluable cases), and the Express example's 32-point stale-order grid answers deny with the rule named even when the model is wrong (proved on a deliberately weak seed or the stub)
- [ ] #4 Docs updated: docs/constraints or the language reference, docs/training-pipeline.md (what the gate now guarantees versus what enforcement guarantees), docs/diagnostics.md via the generated remedies, the runtime README and the Express README; CI is green
<!-- AC:END -->
