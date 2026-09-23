---
id: decision-5
title: >-
  Held-out refund cases: real public inputs adjudicated by an independent model
  judge instead of human authorship
date: '2026-09-23 21:32'
status: accepted
---
## Context

TASK-5.11 requires two disjoint held-out refund sets (release verification and final evaluation) whose inputs were not produced by any training teacher and whose labels carry an attestation. The original plan called for human-authored, attested cases. On 2026-09-23 the user decided against human authorship ("I don't think human input is going to get us better results") and asked for the whole pipeline to be automated with Claude Fable 5.1 as the judge. The existing contracts only admit `human-authored` and `other-held-out` origins with a `humanAttestation` block, and a false human attestation would corrupt the benchmark's provenance.

## Decision

- Held-out inputs come from real, public, de-identified transactions: UCI Online Retail II (CC BY 4.0), mapped onto the compiled refund schema by a committed, deterministic extraction script. Real cancellation invoices supply order age (days since the customer's preceding order), order total, and prior refund counts; customer tier is derived from spend rank; a documented, hash-selected subset has `order.status` set to `fraudulent` by the judge so the never-approve constraint is exercised.
- Labels are adjudicated case by case by Claude Fable 5.1 (`claude-fable-5-1`) in an interactive Claude Code session under a committed rubric, with a rationale recorded per case. The judge is a different, stronger model than the Sonnet 5 training teacher and never sees teacher outputs.
- The contracts gain an `independent-judge` case origin and a `judgeAttestation` block (judge identity, rubric digest, case IDs, fixed declaration, evidence digest) across the TypeScript validators, JSON Schemas and the Python pipeline. `human-authored` remains supported. Metrics report the attested slice (human-authored or independent-judge) instead of a human-only slice.
- TASK-5.11 acceptance criterion 5 is rewritten to the independent-judge slice.

## Consequences

- Provenance stays honest: no record claims human authorship. Anyone reading the benchmark can see that labels are model adjudications and can audit the rubric and rationales.
- Label noise is correlated with model-style reasoning rather than human judgment; the measured accuracy is "agreement with a documented policy interpretation applied by a stronger independent model", which is the tradeoff the user accepted. Human-authored slices can still be added later without contract changes.
- The judge must never be used as a training teacher for this benchmark, and the rubric must not be shown to the student or to the Sonnet teacher.
