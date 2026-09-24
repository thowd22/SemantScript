# Shared encoder, one adapter, one head per function (TASK-6.1), 2026-09-24

Two refund functions trained into one artifact over one ModernBERT-base
encoder and one application adapter, with a head each:

- **A**, the canonical refund decision (`nf_65e347f7…`, approve / deny /
  review), with its frozen v4 corpus (9,035 synthetic rows, 374 adversarial)
  and the judge-attested release record (80 cases).
- **B**, the companion refund-risk rating (`refund-risk.sem.ts`,
  `nf_3fb6038f…`, high / low / medium) over the same customer and order
  inputs. Its corpus is the real UCI pool labeled by its own compiled
  constraints (7,340 rows, zero ambiguous), and its adversarial sidecar is 300
  single-field edits of real inputs labeled by the constraints (6 boundary
  cases, 147 counterfactual pairs), generated without a language model. Its
  "attested" set is the 80 release inputs relabeled by its constraints: rule
  labels, not human or judge adjudication.

Recipe for every regime: compact canonical encoding, 8 epochs, batch 16,
learning rate 3e-5, linear schedule with 5 percent warmup, best epoch on the
held-out split, seed 1, adapter bottleneck 64.

## Per-head accuracy

| Regime | A held-out | A attested (80 judge cases) | B held-out | B attested (80 rule cases) | Time |
| --- | --- | --- | --- | --- | --- |
| A alone (committed compact release, single-function trainer) | 0.9968 | 80/80 | | | 344 s |
| B alone (single-function trainer) | | | 1.0000 | 80/80 | 394 s |
| **Joint**: one encoder and adapter, both heads | 0.9968 | 80/80 | 1.0000 | 80/80 | 859 s |
| A first, then B's head on the frozen encoder | 0.9947 | 80/80 | 0.9751 | 69/80 | 691 s |

Held-out means each function's own 10 percent calibration split (941 rows for
A, 764 for B). Joint training over a shared encoder costs neither function
anything against single-function training. Adding a head to a frozen
application is mechanically clean, A's parameters are byte-identical before
and after and its verification stays valid, but a head alone cannot recover
what the frozen encoder never learned for B: 0.975 held-out and 69/80 on the
attested set against 1.000 and 80/80 when trained jointly. That gap is the
motivation for per-domain adapters in TASK-6.7.

## The exported application (`artifact/`, manifest `9ec23d46…`)

Both functions of the joint application were verified separately on their
views of the shared model:

| Function | Verification | Accuracy | ECE | Constraint violations |
| --- | --- | --- | --- | --- |
| A | passed, 80 attested, 0 misses | 0.9968 | 0.0019 | 3 of 9,489 records |
| B | passed, 80 attested, 0 misses | 1.0000 | 0.0000 | 0 of 8,020 records |

The artifact holds five resources (tokenizer, encoder, adapter, two heads) and
two manifest functions sharing `encoder.refund-benchmark` and
`adapter.refund-benchmark`; the Node runtime loads it and dispatches both
functions (`runtimeDiagnostics` in the report). The 572 MB bundle is
git-ignored like the other release artifacts; the verified IRs, the ledger for
A, the report and B's frozen corpus caches are committed here.

Records: `experiment-report.json` (every regime, verification blocks, the
runtime diagnostics), `verified-ir-a.json`, `verified-ir-b.json`,
`training-input-ledger-a.json`, `b-synthetic/`, `b-adversarial/`.
