# Encoder size and initialisation sweep (TASK-5.14), 2026-09-24

Which encoder should an application fine-tune? Decision-3 chose
ModernBERT-base for Phase 1; this sweep measures the alternatives with one
recipe, head and loss on the same data. Measured with
`program/run_encoder_sweep.py` on the frozen refund corpus
(`data/pooled-v4-2026-09-24`, 9,409 rows) and the canonical refund function
(`nf_65e347f7…`), on the same machine as the release runs (AMD Radeon RX 9070
XT, 16 GB, ROCm; 24-thread Ryzen 9 9900X for CPU numbers). Raw records:
`results.json`.

Recipe for every point: the committed compact-release settings (batch 16,
sequence 128, learning rate 3e-5, evaluation ratio 0.1, seed 1, linear head,
proper-scoring loss, best epoch of 4), then the trainer's verification against
the 80 judge-attested release cases under decision-8's 1 percent
constraint-violation tolerance, then ONNX export of the encoder-adapter-head
chain (float32, no external data) and batch-1 latency on a 35-token refund
input: 200 timed calls after 20 warm-up calls, on CPU through ONNX Runtime
(the deployed path) and on the GPU in PyTorch eager mode, because the ONNX
Runtime build on this machine has only the CPU provider. The amortisation rows
time one encoder pass followed by 1, 10 or 50 head passes on CPU.

## Points

| Point                        | Initialisation                                                 | Parameters | Encoder ONNX |
| ---------------------------- | -------------------------------------------------------------- | ---------- | ------------ |
| `modernbert-base`            | `answerdotai/ModernBERT-base` (decision-3 default)             | 149.0M     | 597 MB       |
| `modernbert-large-laya-init` | ModernBERT-large weights extracted from `laya-typed-decisions` | 394.8M     | 1,580 MB     |
| `modernbert-large`           | `answerdotai/ModernBERT-large`                                 | 394.8M     | 1,580 MB     |
| `deberta-v2-xlarge`          | `microsoft/deberta-v2-xlarge` (the ~1B point, 884.6M)          | 884.6M     | see below    |

The Laya encoder is the same architecture as ModernBERT-large; its weights
were taken from the pinned Laya checkpoint's `encoder.*` tensors
(`program/prepare_laya_encoder.py`, fp16 cast to float32) so the only
difference between the two large points is the initialisation.

## Accuracy and calibration (same data, head and loss)

| Point                        | Train s | Best epoch | Held-out curve             | Release accuracy | Release misses | Calibrated acc | Pair consistency | ECE    | Brier  | Violations | Gate   |
| ---------------------------- | ------- | ---------- | -------------------------- | ---------------- | -------------- | -------------- | ---------------- | ------ | ------ | ---------- | ------ |
| `modernbert-base`            | 160     | 4          | 0.987, 0.988, 0.986, 0.990 | 1.000            | 0              | 0.9904         | 0.9724           | 0.0038 | 0.0082 | 42         | passed |
| `modernbert-large-laya-init` | 357     | 4          | 0.988, 0.993, 0.988, 0.994 | 1.000            | 0              | 0.9936         | 0.9779           | 0.0030 | 0.0055 | 22         | passed |
| `modernbert-large`           | 356     | 4          | 0.976, 0.993, 0.995, 0.996 | 0.9625           | 3              | 0.9957         | 0.9890           | 0.0037 | 0.0045 | 48         | failed |

Pretrained ModernBERT-large has the best held-out numbers and the highest pair
consistency, yet misses three attested cases, all paid orders just outside
their tier window (a standard-tier order at 49 days and two enterprise orders
at 67 days, expected `deny`, predicted `approve`), so it fails the gate. The
Laya-initialised weights, same size and recipe, pass with zero misses, the
lowest ECE and half the base model's constraint violations. Base passes with
the widest margin per parameter.

## Latency and size

| Point                        | GPU p50 / p95 (PyTorch) | CPU ONNX chain p50 / p95 | CPU encoder-only p50 | CPU head-only p50 | Artifact ONNX |
| ---------------------------- | ----------------------- | ------------------------ | -------------------- | ----------------- | ------------- |
| `modernbert-base`            | 8.1 / 23.5 ms           | 27.6 / 44.6 ms           | 40.0 ms\*            | 0.005 ms          | 597 MB        |
| `modernbert-large-laya-init` | 13.5 / 21.8 ms          | 93.7 / 121.5 ms          | 89.0 ms              | 0.005 ms          | 1,580 MB      |
| `modernbert-large`           | 14.1 / 15.6 ms          | 87.6 / 140.0 ms          | 66.4 ms\*            | 0.005 ms          | 1,580 MB      |

\* The encoder-only and stage rows were timed in separate loops after the
chain loop; on this shared desktop their medians drift by tens of milliseconds
between loops (base 27.6 ms as a chain, 40.0 ms encoder-only a minute later),
so compare within a row, not across the two tables to the millisecond. The
chain row is the deployed-path number; it agrees with the committed compact
release (28.5 ms p50 in `results-compact-2026-09-24`).

Heads sharing one encoder pass on CPU (stage p50 and per-decision p50):

| Point                        | 1 head         | 10 heads       | 50 heads       |
| ---------------------------- | -------------- | -------------- | -------------- |
| `modernbert-base`            | 41.5 / 41.5 ms | 47.2 / 4.72 ms | 40.4 / 0.81 ms |
| `modernbert-large-laya-init` | 87.9 / 87.9 ms | 87.7 / 8.77 ms | 88.7 / 1.78 ms |
| `modernbert-large`           | 81.3 / 81.3 ms | 88.1 / 8.81 ms | 86.2 / 1.72 ms |

A head costs 5 microseconds; fifty of them are invisible next to any encoder.
The encoder pass is the whole cost at every size, so size trades latency for
capability exactly as the plan expected and heads do not change the ranking.

## The ~1B point

DeBERTa-v2-xlarge (884.6M parameters, hidden size 1,536) did not train on
this stack: at the shared recipe (batch 16, learning rate 3e-5) and again at
batch 8 with learning rate 1e-5 the classifier logits became non-finite within
the first optimizer steps, although the same model's forward and backward
passes are finite on the GPU in eval and train mode, with and without padding,
through the trainer's own sentence encoder. The failure is recorded in
`results.json` (`trainingAttempts`); its cause was not isolated within this
task's budget, so the ~1B row has no accuracy, pair-consistency or ECE.

What could be measured without training (architecture decides latency, not
weights): batch-1 GPU latency 11.1 ms p50 / 15.1 ms p95 in PyTorch eager, CPU
PyTorch 850 ms p50 (no ONNX number: the exported graph failed the exporter's
PyTorch parity check, so DeBERTa-v2 cannot ship through the v1 ONNX path
either), and 3.5 GB of float32 weights, above the ONNX single-file limit.

## What the numbers say

- **Decision-3 stands: ModernBERT-base remains the default.** On this task it
  passes the gate with 0.990 calibrated accuracy and is the only size whose
  deployed CPU path is under 30 ms and whose GPU pass is under 10 ms. The large
  encoders add at most 0.5 points of calibrated accuracy and cost 3.2x to 3.4x
  on the CPU request path and 1.7x on the GPU.
- **Initialisation matters more than size at 400M.** Decision-trained
  weights (Laya) beat pretrained weights of the same architecture on the
  attested criterion and on calibration; pretrained large is the point that
  fails. If an application needs a larger encoder, start it from
  decision-trained weights and keep the attested gate.
- **Rule of thumb per application.** Under 10 ms per request: ModernBERT-base
  on a GPU (8 ms), or base on CPU with the fused-stage path only when several
  decisions share one input (0.8 ms per decision at 50 heads); no size meets
  10 ms for a single decision on this CPU. Batch and offline workloads that
  tolerate 100 ms: ModernBERT-large from Laya weights buys lower ECE and fewer
  violations. Above that, see the ~1B section. The encoder is a per-application
  setting (`--encoder-name`, `--encoder-revision`; default ModernBERT-base).
- **Artifact format limit.** A 1.58 GB encoder already exceeds the v1
  artifact's 1 GiB component cap (the sweep raised it for measurement only),
  and anything near 1B parameters exceeds the ONNX protobuf's 2 GB single-file
  limit; shipping a large encoder needs an external-data artifact format,
  which v1 forbids.
