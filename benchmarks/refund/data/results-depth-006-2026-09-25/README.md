# Depth-routed refund release on the final set (TASK-5.18), 2026-09-25

The Phase 1 exit benchmark re-run for the SemantScript system with a float32
artifact that carries both latency levers: the compact canonical encoding
(`semantscript.canonical-input/v2`, TASK-5.18.1) and compile-time depth
routing (TASK-6.7, decision-10), the refund domain reading the shared
ModernBERT-base encoder after 6 of its 22 layers. Same frozen final set, same
harness, same protocol and machine as the committed go/no-go run
(`results-v2-2026-09-23`) and the compact run (`results-compact-2026-09-24`).

## Release (`data/release-depth-006-2026-09-25`, artifact manifest `a3f4b005…`)

Trained by the release pipeline with `--encoder-depth 6` and the committed
compact recipe (v4 corpus, 9,035 synthetic and 374 adversarial rows, 8 epochs,
batch 16, learning rate 3e-5, linear schedule with 5 percent warmup, best
epoch 7 on the 10 percent calibration split, seed 1, 320 s on the RX 9070 XT).
The strict gate passed on the first seed: zero attested release misses over
the 80 judge-attested cases, calibration accuracy 0.9957, ECE 0.0027, Brier
0.0043, pair consistency 1.000, 5 constraint violations across 9,489 records
(0.05 percent, under decision-8's 1 percent tolerance). The artifact ships the
6-layer prefix as its own ONNX graph (275 MB) with one adapter and one head;
the training-input ledger proves by digest that no final-set input was seen
(leakage audit: 160 evaluation cases, 9,291 training inputs, 0 overlap).

## Final set (160 judge-attested cases), committed protocol

| Artifact                             | Encoding    | Depth | Accuracy            | ECE (15 bins) | p50 ms   | p95 ms   | Requests/s | Peak client RSS |
| ------------------------------------ | ----------- | ----- | ------------------- | ------------- | -------- | -------- | ---------- | --------------- |
| `0ee80669…` (committed go/no-go run) | v1 envelope | 22    | 0.994 (159/160)     | 0.011         | 38.40    | 50.49    | 24.5       | 1,646 MiB       |
| `56c5c5d6…` (compact run)            | v2 compact  | 22    | 1.000 (160/160)     | 0.0036        | 28.49    | 35.39    | 34.6       | 1,662 MiB       |
| `a3f4b005…` (this run)               | v2 compact  | 6     | **1.000** (160/160) | **0.0029**    | **4.82** | **6.41** | 194.2      | 816 MiB         |

Concurrency 1, ten warm-up iterations over synthetic inputs disjoint from the
evaluation set, client-only memory scope, nearest-rank percentiles, 15-bin
top-1 expected calibration error computed from the sealed predictions
(`predictions-semantscript.json`, payload `40e9c009…`). `result.json` is the
assembled machine-checked record; its mechanical status is `incomplete`
because only the SemantScript system ran in this directory (the baselines
were not re-measured and the structured-output API baseline still has no key
on this machine), so the latency and accuracy criteria are read off the
system row against the committed baseline numbers, as the compact run did.

## What the numbers say

- **Latency criterion: met.** p50 4.82 ms against the exit bar of strictly
  below 10 ms, an 8x reduction from the committed run and 5.9x from the
  compact release, on the deployed CPU path through `onnxruntime-node`. The
  client process also halves in memory because it holds a 6-layer encoder.
- **Accuracy and calibration criteria: met.** 160 of 160 attested cases with
  ECE 0.0029, against the 0.99 and 0.05 thresholds, and against 0.538 for the
  7B comparator in the committed run.
- **The direct-call sweep understated the gain.** `run_depth_sweep.py`
  measured depth 6 at 9.8 ms p50 through a bare `handle.call` loop on a
  desktop that was also exporting artifacts; the harness, with the machine
  otherwise idle, measures the same layout at 4.8 ms. The ordering across
  depths in the sweep stands; the absolute number here is the one under the
  committed protocol.
