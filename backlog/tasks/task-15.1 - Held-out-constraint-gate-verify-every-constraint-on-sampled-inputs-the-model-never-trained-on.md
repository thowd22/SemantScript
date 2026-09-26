---
id: TASK-15.1
title: >-
  Held-out constraint gate: verify every constraint on sampled inputs the model
  never trained on
status: To Do
assignee: []
created_date: '2026-09-26 06:27'
labels:
  - dx
  - quality
milestone: m-6
dependencies: []
parent_task_id: TASK-15
priority: high
ordinal: 66000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The verifier's constraint check (trainer/src/semantscript_trainer/verification.py: _constraint_violations, fed by _case_records over the reconstructed training corpus) counts violations only on records the model trained on: gold examples, teacher-labelled synthetic cases and the adversarial boundary pairs and counterfactual twins. The Express example's release 217d386c passed with 0 violations and yet answers review or approve for orders at 100, 110, 120, 150 and 200 days under always(() => order.ageDays > 90, 'deny') (semantscript explain in examples/express-app shows it; only the exact boundary cases at 90 and 91 days and the training cases at 132 days answer deny). The gate needs a held-out sample: inputs generated from the function's input types and its constraints' predicates (both sides of each boundary and the interior of each region, plus uniform draws), deterministic by seed, never overlapping the training corpus, scored under the raw model like the existing check, and subject to the same tolerance and seed-retry classification (TASK-14.6). Held-out benchmark inputs must never enter training. Relevant code: verification.py (_constraint_violations, _gate_failures, VerificationConfig.maximum_constraint_violation_rate), constraints.py (predicate evaluator, ConstraintEvaluationBudget), adversarial.py (boundary generation to reuse for sampling), dataset.py (canonical inputs), cli.py (report, seed retry), cli/src/train.ts and test-command.ts (report rendering), docs/training-pipeline.md, docs/diagnostics.md (generated remedies via diagnostics/remedies.json and scripts/generate-remedies.mjs), docs/cli-reference.md. No paid teacher calls are needed: the Express example's cached datasets under examples/express-app/.semantscript/cache allow free retrains, and the constraints teacher (--teacher constraints) trains the refund-service example at no cost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The verifier scores each constraint on a held-out input set generated from the input types and the constraint predicates (boundary, interior and uniform draws, deterministic by --seed, disjoint from the training corpus), and the release manifest records the held-out sample size and violation rate beside the existing corpus figures
- [ ] #2 A model that keeps the rules on the corpus but breaks them on held-out inputs fails the gate (proved with the Express example's cached datasets or a fixture), and the failure names the constraint, several offending inputs with the model's answer, and the next command through the generated remedies
- [ ] #3 Tests cover the sampler (per-constraint boundary and interior coverage, determinism, corpus disjointness, evaluation budget) and the gate; semantscript test and the train report show the held-out figure; docs/training-pipeline.md, docs/cli-reference.md and docs/diagnostics.md describe the check; CI is green
<!-- AC:END -->
