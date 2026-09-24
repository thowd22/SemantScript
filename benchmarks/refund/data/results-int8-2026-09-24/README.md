# Int8 encoder follow-up (TASK-5.18.2), 2026-09-24

This directory holds the benchmark of a dynamically quantized (int8) derivation
of the committed float32 refund release (`data/release-2026-09-23`, manifest
`0ee80669…`). It is a latency experiment, not a new release decision: the
written go/no-go in `data/results-v2-2026-09-23/README.md` stands.

## Gate outcome

`quantize_release.py` runs the quantized encoder chain against the float32
chain over every verification record (9,409 corpus rows plus the 80 attested
release cases) before it publishes anything.

- **Strict run (default tolerances), `data/release-int8-2026-09-24`: refused.**
  10 of 9,489 decisions changed (0.105 percent) and one attested release case
  changed: a standard-tier customer with a fraudulent order at 179 days
  (expected `deny`) becomes `review` with probability 0.94. The float32 gate
  admits zero attested misses, so this graph is not releasable under the Phase 1
  policy. The refusal and its report are committed; no release was published.
- **Tolerance-recorded run, `data/release-int8-tolerant-2026-09-24`: published
  for measurement** under `attestedDisagreementTolerance: 2` and
  `argmaxDisagreementTolerance: 0.01`, both written into the artifact manifest
  (encoder `onnx.precision: int8-dynamic`, `onnx.quantization`) and the derived
  pipeline manifest. Quantized ECE 0.0066 against 0.0043 for float32 at the
  verified temperature; corpus accuracy 0.9979 against 0.9985; encoder 596.7 MB
  to 150.1 MB.

## Final set (160 judge-attested cases), same protocol as the float32 run

| Artifact | Accuracy | p50 ms | p95 ms | Requests/s | Client RSS |
| --- | --- | --- | --- | --- | --- |
| float32 `0ee80669…` | 0.994 (159/160) | 38.40 | 50.49 | 24.5 | 1,646 MiB |
| int8-dynamic `f01067d5…` | 1.000 (160/160) | 27.74 | 35.26 | 34.8 | 599 MiB |

The int8 graph is 1.38x faster at the median and uses about a third of the
client memory; p50 stays well above the 10 ms exit bar. It scores 160/160
because its one changed final-set decision happens to fix the float32 miss
(the 62-day fraudulent order now reads `review`), while on the attested release
set the same rounding breaks a case the float32 graph gets right. Both are
single-case flips in the fraud region and neither is evidence of a better or
worse model; the attested miss is the one the release policy counts.

## What this settles

- Dynamic int8 alone gives 1.3 to 1.4x on this CPU, not the 2.3x the PyTorch
  proxy suggested, and it moves roughly one decision in a thousand. It is a
  usable lever only together with fewer tokens (TASK-5.18.1) or fewer layers
  (TASK-6.7), and only under a stated tolerance.
- Quantization is now a verified, refusable step with its tolerances in the
  manifest, so any later int8 release states exactly what it was allowed to
  change.

Records: `predictions-semantscript.json`, `environment.json` (fresh capture,
same machine), the derived pipeline manifest and quantization report under
`data/release-int8-tolerant-2026-09-24`, and the refused report under
`data/release-int8-2026-09-24`. The 145 MB int8 artifact bundle is git-ignored
like the float32 one.
