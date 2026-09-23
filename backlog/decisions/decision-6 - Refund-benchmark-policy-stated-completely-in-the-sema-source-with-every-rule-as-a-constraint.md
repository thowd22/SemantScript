---
id: decision-6
title: >-
  Refund benchmark policy stated completely in the sema source with every rule
  as a constraint
date: '2026-09-23 23:03'
status: accepted
---
## Context

The Phase 1 refund task was compiled from a two-sentence policy plus two constraints. The first complete release run (2026-09-23) trained a 74 percent student and failed the zero-miss release gate on 35 of 80 attested cases. Diagnosis: the Sonnet 5 teacher agreed with the judge rubric on only 87 of 181 outside-window cases and 33 of 78 suspicious-history cases, and the zero-shot baselines resolved the same ambiguities a third way. The user then asked for accuracy as close to 99 percent as possible and allowed Opus 5.5 as a teacher.

## Decision

- The sema source states the full policy in prose and encodes every rule as an `always`/`never` constraint (six constraints: stale-order deny, fraud never approve, fraud review, outside-window deny, suspicious-history review, clean approve). Constraints remain build contracts: generated cases are rejected when they violate one, adversarial pairs cover each rule, and verification counts violations. The runtime does not enforce them.
- The judge rubric is unchanged; the source now mirrors it, so teacher, judge and baselines share one policy. The baselines receive the same six rules in their prompt.
- The training corpus is regenerated with `claude-opus-5-5` through the same CLI teacher (model recorded in provenance) under prompt contract v3, which adds threshold-focused variation hints.

## Consequences

- The benchmark now measures how well a compiled encoder learns an explicit numeric policy from text, not how a model guesses an underspecified one. Label noise from interpretation disappears; residual error is learning error.
- The function id, semantic digest and task-spec digest changed; earlier prediction sets (Sonnet-era baselines) are superseded and must be rerun.
- Reaching the zero-miss release gate depends on the student learning exact numeric thresholds; if it falls just short, a documented gate tolerance is the remaining lever and needs its own decision.
