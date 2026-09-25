---
id: decision-11
title: >-
  Jev is a diagnostic comparator and a candidate typed-corpus source, not a
  benchmark system, a teacher or a runtime dependency
date: '2026-09-25 04:16'
status: accepted
---
## Context



## Decision



## Consequences


## Context

TASK-5.17 asked what Jev (TypeSafe's hosted decision model, `~typesafe/jev-latest` on OpenRouter, resolved to `typesafe/jev-1.13-20260917`) is to this project: a required benchmark system, a distillation or behavior-data source for Phase 2 and Phase 3, or neither. A live probe documented its contract (`POST /api/alpha/decisions`, typed `noul`/`choice`/`score` questions over a state, calibrated probabilities per option, USD 0.042 per million input tokens, ~150 ms per request over the network). On the frozen refund final set under the benchmark's definitions (`benchmarks/refund/data/results-jev-2026-09-25`) it answers 154 of 160 cases (0.9625, 15-bin ECE 0.031, Brier 0.046) for USD 0.0058 in total, against 1.000 at 4.8 ms for the compiled depth-6 refund artifact, 0.54 for the 7B generative baseline and 0.225 for Laya. All six misses are one systematic error: enterprise paid orders aged 33 to 35 days denied as if the 30-day standard window applied; on the 16 cases where the tier-conditional window decides, it agrees on 10. A 10k-case one-question corpus would cost about USD 0.36.

## Decision

- **Diagnostic comparator, not a benchmark system.** Jev stays outside the pinned benchmark roles and the go/no-go: it runs over the network with no execution-environment evidence, its answers cannot be bound to the policy's hard constraints, and the benchmark's exit criteria are about the compiled artifact's accuracy and local latency. Its result is kept as a diagnostic record next to Laya's, and it is the strongest external point in the typed-decision family the project has measured.
- **A candidate typed-corpus source only behind the constraints.** Its labels are cheap and mostly right but systematically wrong on one conditional; a corpus labeled by Jev may be used for Phase 2 and Phase 3 experiments only through decision-7's path, where every label is checked by the compiled constraints and violating rows are rejected, and the resulting artifact is verified against attested cases as usual. It is not a teacher in the teacher slot: the trainer's teacher contract needs boundary pairs and counterfactual twins around constraints, which a typed-decision model does not produce.
- **Not a runtime dependency.** Nothing shipped calls Jev; the north-star (per-application accuracy at local latency) is met by the compiled artifact, which is better on this policy and thirty times faster with no network.

## Consequences

- `benchmarks/refund/program/run_jev_comparator.py` can be rerun (budget-capped, answers cached) whenever the final set or Jev's version changes; the record notes the resolved model id.
- If a Phase 2 or Phase 3 experiment wants a large typed corpus for a policy whose constraints are complete, Jev is the cheapest labeler measured so far, and the constraint filter is mandatory.
- The typed-decisions benchmark (TASK-6.6) may add Jev as a comparator on its test split using the same driver pattern; that is a separate, optional measurement.
