---
id: decision-7
title: >-
  Train on real-distribution inputs labeled by the compiled constraints, with
  Opus cases replayed and Opus adversarial pairs
date: '2026-09-23 23:54'
status: accepted
---
## Context

With the policy fully explicit (decision-6), the Opus 5.5 corpus trained to 0.97 accuracy on its own held-out split yet only 0.75 on the judge-attested release cases, with dozens of constraint violations. The per-case dump showed half of the real clean approvals predicted as review: the teacher-invented inputs live at a median order total near 1,000 with four prior refunds and almost no day-zero orders, while real refund requests sit near 380 with one prior refund and many same-day cancellations. Sweeping epochs, learning rate and head architecture did not move the release number.

## Decision

- Add a training partition drawn from the real UCI candidate pool (7,597 de-identified refund requests): every held-out input is excluded by semantic digest, the same hash-selected fraud flag is applied, and each input receives the unique output the six compiled constraints admit (7,339 rows labeled; zero ambiguous). The frozen Opus synthetic cases are replayed in front of the pool so threshold-dense teacher data is retained; boundary pairs and counterfactual twins still come from the Claude CLI teacher (Opus 5.5), now anchored on real inputs.
- Provenance is explicit: the teacher descriptor's provider names the rule labeling, and its configuration binds the pool digest, replayed dataset digest, exclusion digest and CLI configuration. The release pipeline reconstructs the teacher from the manifest with teacher calls forbidden.

## Consequences

- The student now learns the policy on the input distribution it is evaluated on, which is what an application would do with its own traffic; the leakage ledger still proves no held-out input entered training.
- Labels for the real partition come from the constraints, not an LLM, which is only possible because the policy is fully constrained. For fuzzier expressions the LLM teacher remains the labeler.
- The 'accuracy' the benchmark reports is agreement with an explicit numeric policy on real inputs, and the write-up must say so.
