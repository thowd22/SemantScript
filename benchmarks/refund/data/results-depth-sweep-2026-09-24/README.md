# Depth routing on the refund function (TASK-6.7), 2026-09-24

How many of the shared encoder's 22 layers does the refund decision need?
Measured with `program/run_depth_sweep.py` on the frozen v4 corpus
(`data/pooled-v4-2026-09-24`, 9,035 synthetic and 374 adversarial rows) with
the committed compact-release recipe (`semantscript.canonical-input/v2`, batch
16, learning rate 3e-5, 8 epochs, best epoch by held-out accuracy, seed 1,
linear head, proper-scoring loss), the same machine as the release runs (AMD
Ryzen 9 9900X, RX 9070 XT under ROCm for training, ONNX Runtime 1.30 CPU for
the deployed path, Node 22.22). Raw records: `results.json`; the exported
artifacts are rebuilt by the driver and not committed.

Each depth is one routed domain: the function's IR binds `model.encoderDepth`
and an encoder ref of its own, the trainer fine-tunes the encoder, one adapter
and the head with the adapter reading the hidden state after that many layers
(through the encoder's final normalization), verification runs against the 80
judge-attested release cases under decision-8's 1 percent violation tolerance,
and the artifact carries the encoder prefix as its own ONNX graph. The
decision to export prefixes rather than one multi-output graph came from a
probe on this stack: ONNX Runtime executes a graph whole whatever output is
fetched (the depth-4 output of a multi-output graph cost 20.5 ms against 23.5
ms for all 22 layers), while a standalone 6-layer prefix ran in 5.3 ms with
bit-identical values.

## Gate and final set

| Depth | Train s | Best epoch | Calibrated acc | Pair consistency | ECE    | Brier  | Violations | Release misses | Final set (160) | Final p50 / p95 ms | Artifact |
| ----- | ------- | ---------- | -------------- | ---------------- | ------ | ------ | ---------- | -------------- | --------------- | ------------------ | -------- |
| 4     | 165     | 5          | 0.9936         | 0.989            | 0.0042 | 0.0052 | 16         | 0              | 1.000 (160)     | 7.9 / 12.1         | 237 MB   |
| 6     | 177     | 6          | 0.9936         | 0.9945           | 0.0019 | 0.0058 | 8          | 0              | 1.000 (160)     | 9.8 / 22.0         | 277 MB   |
| 8     | 194     | 8          | 0.9957         | 1.000            | 0.0030 | 0.0047 | 8          | 0              | 1.000 (160)     | 16.0 / 32.1        | 317 MB   |
| 12    | 231     | 8          | 0.9947         | 0.9945           | 0.0004 | 0.0053 | 8          | 0              | 1.000 (160)     | 23.8 / 29.5        | 398 MB   |
| 22    | 316     | 8          | 0.9979         | 1.000            | 0.0012 | 0.0019 | 3          | 0              | 0.994 (159)     | 29.3 / 45.6        | 599 MB   |

Every depth passes the strict gate with zero attested release misses. The
final set is the frozen 160-case judge-attested benchmark set
(`heldout-uci-2026-09-23/final-benchmark-dataset.json`), scored through the
Node runtime by `program/run-final-set.mjs`: every case called once after 20
warm-up calls, each call timed, so accuracy and latency come from the same
pass. The full-depth point misses one case (a paid enterprise order at 67
days, expected `deny`, predicted `approve`), the same shape of miss the encoder
sweep saw with pretrained large; the four shallower points answer all 160.

## Latency and size

| Depth | GPU p50 (PyTorch) | CPU ONNX chain p50 / p95 | CPU encoder-only p50 | Encoder ONNX |
| ----- | ----------------- | ------------------------ | -------------------- | ------------ |
| 4     | 8.2 ms            | 7.2 / 16.0 ms            | 6.9 ms               | 235 MB       |
| 6     | 8.3 ms            | 8.7 / 17.2 ms            | 8.7 ms               | 275 MB       |
| 8     | 8.6 ms            | 14.9 / 16.6 ms           | 14.5 ms              | 315 MB       |
| 12    | 8.5 ms            | 17.0 / 24.1 ms           | 13.9 ms              | 396 MB       |
| 22    | 8.3 ms            | 39.4 / 55.4 ms           | 30.7 ms              | 597 MB       |

The chain rows are 200 timed calls after 20 warm-ups on a 35-token compact
input, batch 1, default ONNX Runtime threads. On the GPU the pass is
launch-bound at every depth (about 8 ms), so depth buys nothing there; on the
CPU the cost is the layer count and the shallow prefixes are where the
request-path budget is met. The desktop is shared and its CPU numbers drift by
several milliseconds between loops (the encoder sweep recorded the same): the
full-depth chain measured 39.4 ms here against 27.6 ms in the encoder sweep,
and the committed compact release scores 18.9 ms p50 through
`run-final-set.mjs` against 28.5 ms under the benchmark harness, which adds
its own per-call bookkeeping. Compare rows within one table; the ordering is
the result, the last millisecond is not.

## What the numbers say

- **The Phase 1 latency criterion is met by depth routing.** At depth 4 and
  depth 6 the refund artifact answers the frozen final set with p50 7.9 ms
  and 9.8 ms through the CPU Node runtime, under the 10 ms bar, with attested
  accuracy 1.000 against 0.994 at full depth (within one percentage point,
  and in fact one case better). The re-measurement under the committed
  benchmark harness with the release pipeline is TASK-5.18's.
- **The refund policy needs four layers.** Calibrated accuracy moves between
  0.9936 and 0.9979 across the sweep and the release gate passes everywhere;
  the full stack's advantage is Brier (0.0019 against 0.0052) and constraint
  violations (3 against 16 at depth 4, 8 at depth 6), which is the
  confidence-shaping side, not decisions. Depth 6 is the better trade for
  this domain: half the violations of depth 4 and the lowest ECE among the
  shallow points, for 2 ms.
- **Prefixes are shared in training, not on disk.** Each exported prefix is
  its own graph, so a routed artifact with several depths carries the shared
  layers once per depth (a 4-layer and a 22-layer domain ship 235 MB plus 597
  MB). ONNX Runtime executing graphs whole is what forces this; an external-
  data layout that shares weight tensors across prefix graphs would remove it
  and is a format question outside this experiment.
