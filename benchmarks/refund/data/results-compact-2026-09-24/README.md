# Compact canonical encoding (TASK-5.18.1), 2026-09-24

The refund release retrained with `semantscript.canonical-input/v2`, the compact
input encoding, on the same v4 corpus and recipe as the committed float32
release (`data/release-2026-09-23`). The encoding is the only change: v2 renders
the same validated inputs as `customer={priorRefunds=0,tier=standard}
order={ageDays=12,status=paid,total=88.5}` instead of the exact-JSON envelope
with hexadecimal doubles, cutting a refund input from 102 to 115 ModernBERT
tokens to 31 to 34.

## Release (`data/release-compact-2026-09-24`, artifact manifest `56c5c5d6…`)

Passed the strict gate on the first seed (seed 1, best epoch 7 of 8): zero
attested release misses, calibration accuracy 0.9968, ECE 0.0036, temperature
1.75, 3 constraint violations across 9,489 verification records (0.03 percent),
against 0.9904, 0.0043 and 14 violations for the v1 release, which needed two
seeds. Training took 344 s instead of 1,290 s. The verified IR records
`trainingProvenance.canonicalInput` and the artifact manifest declares
`compatibility.canonicalInput`, both `semantscript.canonical-input/v2`; the
runtime serializes calls with that version.

## Final set (160 judge-attested cases), same protocol as the float32 run

| Artifact | Encoding | Accuracy | p50 ms | p95 ms | Requests/s | Client RSS |
| --- | --- | --- | --- | --- | --- | --- |
| `0ee80669…` (committed go/no-go run) | v1 envelope | 0.994 (159/160) | 38.40 | 50.49 | 24.5 | 1,646 MiB |
| `56c5c5d6…` (this run) | v2 compact | 1.000 (160/160) | 28.49 | 35.39 | 34.6 | 1,662 MiB |

Per rule the compact release is 28/28, 26/26, 39/39, 20/20 and 47/47. The
float32 v1 miss (a fraudulent order at 62 days) is now answered correctly, and
no attested case moved the other way; unlike the int8 experiment this is a
strict-gate release with no recorded tolerance.

## What the numbers say

- **Tokens are a 1.35x lever, not 2x.** The encoder at 32 tokens costs about
  23 ms in onnxruntime with default threads (the 18 ms figure in the task
  description used a pinned thread count), so 22 layers at batch 1 are bounded
  by per-layer overhead rather than by token count once the input is short.
  Depth routing (TASK-6.7) is the remaining lever for the 10 ms bar.
- **Shorter text trains better, not just faster.** Higher calibration accuracy,
  lower ECE, fewer violations and a first-seed pass are consistent with the
  model no longer spending capacity on envelope syntax and hexadecimal digits.
- The written go/no-go in `data/results-v2-2026-09-23/README.md` stands: the
  latency criterion is still not met at 28.5 ms.

Records: `predictions-semantscript.json`, `environment.json` (fresh capture,
same machine), `failures.json`; release evidence under
`data/release-compact-2026-09-24` (pipeline manifest, verification with the
gate, release predictions, training-input ledger, verified IR). The 571 MB
artifact bundle is git-ignored like the others.
