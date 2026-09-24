# Parallel-head and batch scaling (TASK-6.5), 2026-09-24

Quantifies the shared-encoder thesis on the CPU runtime: when several sema
functions read the same input, one stage should pay for the encoder once and for
each head only its own small cost. Measured with
`program/run-stage-scaling.mjs` on a derived artifact
(`data/release-multihead-2026-09-24`, manifest `68c28a7f…`) that clones the
verified compact refund function (`data/release-compact-2026-09-24`, manifest
`56c5c5d6…`) fifty times over its own tokenizer, encoder and adapter. Every
clone's head is a byte copy of the verified head under a fresh function id and
head ref; a head's runtime cost does not depend on its weights, so the clones
count head passes without claiming fifty trained behaviours. Inputs are the first
64 distinct cases of a synthetic v4 corpus file, never an evaluation set.

Protocol: `loadSemaArtifact` with default options (onnxruntime-node 1.30.0, CPU
execution provider, default thread pool), 5 warm-up rounds, then 30 timed
iterations per configuration of `handle.callStage`, wall-clock around the call
from the caller's thread. Every fused stage asserted encoder 1, adapter 1 and
head N passes, and its results were byte-equal to the N single calls made in the
same iteration. Environment in `environment.json`; full statistics (mean, p50,
p95, min, max) in `results.json`.

## Heads per stage, one input

| Heads per stage | Fused p50 ms | Fused p95 ms | One call per head p50 ms | Speedup (p50) | Fused decisions/s | Per-decision p50 ms |
| --------------- | ------------ | ------------ | ------------------------ | ------------- | ----------------- | ------------------- |
| 1               | 20.99        | 24.06        | 20.83                    | 0.99x         | 46.6              | 20.99               |
| 10              | 22.47        | 28.88        | 217.33                   | 9.67x         | 427.7             | 2.25                |
| 50              | 29.72        | 36.98        | 1132.86                  | 38.12x        | 1640.9            | 0.59                |

Fifty heads add 8.7 ms to a 21 ms stage, about 0.18 ms per head, while fifty
independent calls cost fifty encoder passes (1.13 s). The fused stage delivers
1,641 decisions per second from one CPU stage against 43 for one-call-per-head.

## Batch scaling, one head, distinct inputs

| Distinct inputs per stage | Stage p50 ms | Stage p95 ms | Per-decision p50 ms | Decisions/s |
| ------------------------- | ------------ | ------------ | ------------------- | ----------- |
| 1                         | 24.10        | 28.05        | 24.10               | 40.7        |
| 2                         | 47.13        | 52.05        | 23.56               | 41.9        |
| 4                         | 102.04       | 117.95       | 25.51               | 39.2        |
| 8                         | 172.19       | 187.38       | 21.52               | 46.0        |
| 16                        | 364.57       | 535.84       | 22.79               | 41.0        |
| 32                        | 686.55       | 791.41       | 21.45               | 45.0        |
| 64                        | 1272.54      | 1420.07      | 19.88               | 49.4        |

The curve is linear: a stage of B distinct inputs runs the encoder B times
(the recorded passes are encoder B, adapter B, head B), so per-decision cost
stays at 20 to 25 ms and throughput at 40 to 49 decisions per second whatever
the batch. The worker fuses requests by identical canonical input, not by
padding distinct inputs into one encoder run; the small per-decision decline at
64 is the fixed per-stage overhead (worker message, response framing) being
amortized, not batching.

## What the numbers say

- **The thesis holds for heads.** Cost is one encoder pass per distinct input
  plus a fraction of a millisecond per head; an application with fifty sema
  expressions over one record answers all of them in the time of one and a half
  single decisions.
- **Throughput per core is bounded by the encoder, not by heads.** At batch 64
  the runtime is still 20 ms per decision because distinct inputs are encoded
  one at a time; a batched encoder run (one ONNX call over a padded [B, T]
  tensor) is the untested lever for throughput, separate from the depth-routing
  lever (TASK-6.7) for latency.
- These are raw `callStage` timings from the caller's thread, not the
  go/no-go latency protocol of `run-benchmark.mjs` (which measures diagnostic
  calls through the benchmark adapter with its own warm-up corpus); the 21 ms
  single-head figure here is not a replacement for the 28.5 ms p50 recorded in
  `data/results-compact-2026-09-24`.

Records: `results.json`, `environment.json`; derivation record in
`data/release-multihead-2026-09-24/derivation.json`. The 572 MB derived bundle is
git-ignored like the other artifacts.
