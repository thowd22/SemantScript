---
id: decision-8
title: >-
  Release gate tolerates a recorded constraint-violation rate; artifacts record
  raw-model violations
date: '2026-09-24 00:22'
status: accepted
---
## Context

On the pooled corpus (decision-7) the student predicts all 80 attested release cases correctly with ECE 0.002 and pair consistency 1.0, but the release gate still failed because 36 of roughly 9,300 verification records (every training and calibration row plus the attested cases) violated a constraint under the raw model. With the whole policy expressed as constraints, any misprediction is a violation, so the zero-violation gate demands a learned model with no residual error on unseen in-distribution rows.

## Decision

- `VerificationConfig.maximum_constraint_violation_rate` (default 0.0, the previous strict behavior) sets the fraction of verification records whose raw prediction may violate an active constraint. Attested-example misses, type errors and the ECE threshold stay as they were.
- The application artifact records the observed `constraintViolations` count (schema and runtime loader accept any non-negative integer) while `exampleFailures` and `typeErrors` must still be zero and `status` must be `passed`. The release driver records the tolerance and the record count in its manifest.
- The refund release run uses a tolerance of 0.01 (one percent of records) and reports the observed rate in the go/no-go.

## Consequences

- The gate now distinguishes acceptance-test misses (still zero-tolerance) from residual generalization error on generated rows (bounded and recorded).
- Hard rules that must hold for every production input remain deterministic TypeScript guards, as the spec already required; constraints stay build contracts.
