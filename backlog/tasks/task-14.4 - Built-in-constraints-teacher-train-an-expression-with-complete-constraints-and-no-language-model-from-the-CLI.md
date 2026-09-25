---
id: TASK-14.4
title: >-
  Built-in constraints teacher: train an expression with complete constraints
  and no language model from the CLI
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-25 15:21'
updated_date: '2026-09-25 19:12'
labels:
  - dx
  - train
milestone: m-5
dependencies: []
parent_task_id: TASK-14
ordinal: 57000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When an expression's constraints decide every input, the constraints are a labeling function: the reference application trains all nine of its expressions that way with no API key (examples/refund-service/train.py, a 400-line custom driver), and decision-12 makes constraint labels the local iteration path. But semantscript train has no such backend: [teacher] offers anthropic and ollama only, so a developer who wrote complete constraints still has to pay a teacher or copy the driver. The built-in path should also apply to a mixed expression: constraints label the inputs they decide and the language-model teacher labels the rest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 [teacher] backend = "constraints" (and semantscript train --teacher constraints) trains any expression whose constraints admit exactly one output for every sampled input, sampling inputs from the IR types with threshold-aware numeric ranges, and generates boundary pairs and counterfactual twins by single-field edits
- [ ] #2 An expression whose constraints are incomplete gets a clear message naming an input the constraints do not decide, or, when a language-model teacher is also configured, uses the constraints for the inputs they decide and the teacher for the rest
- [ ] #3 examples/refund-service/train.py is replaced by the built-in backend and its tests and README still pass
- [ ] #4 The teacher provenance records the constraints teacher and its sampling configuration digest
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design (from reading teacher_config.py, teacher.py, dataset.py, adversarial.py, constraints.py, case_contract.py, cli.py, doctor.py, cli/src/{defaults,train,doctor,index,init}.ts, examples/refund-service/{train.py,README.md,package.json,test,scripts/measure.mjs}, docs/teachers.md, docs/reference-application.md, decision-12):

1. New module trainer/src/semantscript_trainer/teachers/constraints.py (generalised from examples/refund-service/train.py):
   - ConstraintsTeacherConfig (frozen, closed): backend="constraints", seed=1, twin_filter=true, maximum_sampling_attempts, ranges (per dotted input path or leaf name: low, high, distribution uniform|log|count, decimals), fallback (optional nested LM TeacherConfig table, anthropic|ollama). configuration_sha256 over a public projection (algorithm version, seed, ranges, knobs, fallback descriptor projection; no secrets).
   - ConstraintSampler per IR: compile_constraints; scalar output support via the case-contract head support; per-path numeric thresholds collected from binary comparisons between an input path and a numeric literal (global literals as fallback), inferred ranges (high = max(10, 2*max threshold), count when thresholds <= 10, log when >= 1000, else uniform; TOML ranges override), 30% near-threshold draws; strings from literals compared to that path plus a fixed pool; boolean, enum, literal, union, null, object (optional fields sometimes omitted), tuple, array (0-3 items). label(inputs) -> the single output with no violated constraint via evaluate_output_contract, else Undecided(allowed outputs).
   - ConstraintsTeacher (Teacher + AdversarialTeacher): descriptor provider "constraints", model "semantscript-constraints/v1" (mixed: provider "constraints+<backend>", model "<fallback model>"), digest = sampling config digest. generate(ir,n): distinct decided samples, keeping only inputs with a single-field counterfactual twin when twin_filter; boundary pairs one field apart with the predicate flipped and both sides decided; counterfactual twins by single-field edits (threshold +-1, enum/union/boolean flips). Per-function, per-purpose reproducible RNG streams (seed:id:purpose:attempt) as in the reference.
   - Pure mode, incomplete constraints: TeacherConfigurationError naming the expression (source path:line, id prefix) and one undecided input as JSON with the outputs it admits (none = contradictory constraints, several = incomplete), plus the fix (add constraints or configure [teacher.fallback]). No constraints or a non-scalar output also fails with a clear message in pure mode.
   - Mixed mode (fallback configured, or a Teacher injected for tests): estimate the decided share from a seeded pilot sample; constraints label ceil(share*n) decided samples, the fallback generates the rest, and any fallback case whose input the constraints decide is relabelled by the constraints (they are the authority). Boundary pairs and twins come from constraints when both sides are decided, otherwise from the fallback (skip-able AdversarialGenerationError path when it cannot). Share 1.0 makes no LM request; no constraints delegates entirely.
   - public sample_decided(ir, n, stream) for held-out sets drawn from a stream the training corpus never uses.
2. teacher_config.py: load_teacher_config accepts the keyword "constraints" (built-in default config, no file) and backend = "constraints" tables; create_teacher builds ConstraintsTeacher (with fallback via the existing LM backends). Keep TeacherConfig for LM backends; union return type. Export new names in __init__.py and test_public_api.
3. cli.py: --teacher takes a path or "constraints"; train_bundle fails fast, before generation, when an expression has no gold examples: "<source>:<line> (<id>) has no gold examples; verification needs at least one attested example per expression".
4. doctor.py: constraints backend -> teacher-config pass (describes seed/ranges/fallback), teacher-key pass "needs no key" (or the fallback's key check), teacher-probe skip "makes no request" (or probes the fallback).
5. Node CLI: defaults.ts BUILT_IN_TEACHERS=["constraints"]; findTeacherConfig/resolveTeacherConfig pass the keyword through unless a file with that name exists; help text (index.ts --teacher <teacher.toml|constraints>), init next-steps line; cli tests for keyword pass-through to train and doctor.
6. Tests without GPU: trainer/tests/test_constraints_teacher.py (sampling types, threshold inference, labels, boundary pairs, twins, undecided-input error text, mixed mode with a fake fallback incl. relabelling and share 1.0 making no calls, descriptor digest stability and change on config, TOML loading/closed keys, keyword); test_cli end-to-end train_bundle with the built-in teacher on the complete-constraints fixture using RuleTokenizer/RuleEncoder on CPU (provenance records provider "constraints" + digest), incomplete examples/refund.sem.ts: pure error vs mixed with RuleTeacher fallback; no-gold-example error; doctor tests.
7. examples/refund-service: delete train.py; add semantscript.teacher.toml (backend constraints, ranges from the old NUMBER_RANGES only if inferred ranges underperform); package.json train -> "semantscript train --cases 800 --epochs 12 --batch-size 16 --learning-rate 5e-5 --max-sequence-length 128 --evaluation-ratio 0.1 --select-best-epoch --local-files-only --device cuda --counterfactual-ratio 0.5 --max-constraint-violation-rate 0.005 --application-id refund-service --report .semantscript/train-report.json && npm run heldout"; small scripts/heldout.py writing heldout.json via sample_decided (stream disjoint from training). Run on the GPU (HSA_ENABLE_DXG_DETECTION=1), npm test (3/3 incl. trained-artifact test), npm run measure; update README (train.py references, the teacher section, provenance, results table and release numbers).
8. Docs: docs/teachers.md (constraints backend section, mixed mode TOML, keyword, gold-example note, writing-a-teacher pointer to the built-in module), docs/cli-reference.md (--teacher, doctor rows), docs/reference-application.md, docs/tutorial-refund-decision.md row, examples/README.md, trainer/README.md teacher backends, docs/getting-started.md if it lists backends; prettier --check.
9. Gates: npm run build, lint:node, test:node, ruff check/format, test:python, prettier; commit, push, gh run watch green.

Risks: generic range inference may not reproduce the reference numbers (fallback: explicit ranges in the example TOML); GPU run ~4+ min and possible strict-gate flakiness near thresholds (the recorded 0.005 tolerance stays); mixed-mode semantics are an interpretation (LM invents inputs; the constraints relabel decided ones); changing the example's teacher identity invalidates its caches (full retrain expected).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
IMPLEMENT: added teachers/constraints.py (ConstraintSampler + ConstraintsTeacher, pure and mixed mode, sample_decided), ConstraintsTeacherConfig/NumberRange in teacher_config.py (keyword 'constraints', backend = "constraints" tables with [teacher.ranges] and [teacher.fallback]), doctor checks for the constraints backend, train_bundle gold-example precondition. New trainer/tests/test_constraints_teacher.py (25 tests, CPU only), 4 test_cli cases (built-in teacher end to end with provenance, pure-mode error and mixed mode on examples/refund.sem.ts, no-gold error, --teacher constraints), 1 doctor test; all pass.

IMPLEMENT (cont.): Node CLI passes --teacher constraints through to train, its preflight and doctor (BUILT_IN_TEACHERS in cli/src/defaults.ts; a file of that name still wins); cli test added. examples/refund-service: train.py deleted; npm run train = semantscript train --teacher constraints with the old recipe (800 cases, 12 epochs, select-best-epoch, cfr 0.5, 0.005 tolerance, cuda) && npm run heldout (scripts/heldout.py, heldout stream, cached training inputs excluded); semantscript devDependency file:../../cli. GPU run 2026-09-25 (11 min wall): release 2f9eb3e890d1, all 9 verified (accuracy 1.0000, ECE 0.0000, 0 violations of 14,456 records, selected epoch 3), teacher constraints/compiled-constraints-v1 digest c58c02b8...; inferred ranges sufficed (no teacher TOML). npm test 3/3; npm run measure: 7 of 9 at 200/200, orders.sem.ts:103 and :122 at 199/200 (amounts just past 2,000 and 5,000). Docs updated (teachers, cli-reference, reference-application, tutorial, index, examples/README, trainer/README, refund-service README). Gates: build, lint:node, test:node, ruff check/format, test:python (577 passed), prettier on touched markdown all green. Commit 64c29cd pushed; CI run 36177879923 success.
<!-- SECTION:NOTES:END -->
