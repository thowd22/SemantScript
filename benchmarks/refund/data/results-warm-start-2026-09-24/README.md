# Universal encoder plus tiny heads: can compilation take seconds? (TASK-7.3), 2026-09-24

The question from the plan: does every function need its own fine-tuned
encoder, or can a frozen universal encoder be reused so compiling a function
only fits a tiny head, cutting compile time from minutes to seconds? Measured
with `program/run_warm_start_experiment.py` on the frozen refund corpus
(`data/pooled-v4-2026-09-24`, 9,409 rows: 9,035 synthetic, 374 adversarial)
and the canonical refund function (`nf_65e347f7…`), on the same machine as the
release runs (AMD Radeon RX 9070 XT via ROCm). The target is the release gate's
attested criterion: zero misses on the 80 judge-attested release cases. Every
regime finishes with the trainer's own verification (temperature scaling, ECE,
constraint violations under the 1 percent tolerance of decision-8), so the
numbers are comparable with a real build. Raw records: `results.json`.

## Full fine-tuning against head-only on the frozen encoder

Both regimes use the committed release recipe (batch 16, sequence 128, seed 1,
linear head, evaluation ratio 0.1); head-only keeps ModernBERT-base in eval
mode with no gradients and raises the head learning rate to 5e-3.

| Regime    | Epochs | Wall s | Release accuracy | Release misses | Calibrated acc | ECE    | Violations | Gate   |
| --------- | ------ | ------ | ---------------- | -------------- | -------------- | ------ | ---------- | ------ |
| full      | 1      | 57.8   | 1.000            | 0              | 0.9872         | 0.0054 | 94         | passed |
| full      | 2      | 106.3  | 1.000            | 0              | 0.9936         | 0.0034 | 36         | passed |
| full      | 4      | 217.6  | 1.000            | 0              | 0.9915         | 0.0039 | 41         | passed |
| full      | 8      | 434.1  | 1.000            | 0              | 0.9926         | 0.0066 | 21         | passed |
| head-only | 1      | 30.5   | 0.775            | 18             | 0.7960         | 0.0236 | 1754       | failed |
| head-only | 2      | 59.6   | 0.775            | 18             | 0.8066         | 0.0302 | 1727       | failed |
| head-only | 4      | 117.4  | 0.763            | 19             | 0.8874         | 0.0190 | 987        | failed |
| head-only | 8      | 235.6  | 0.713            | 23             | 0.8874         | 0.0330 | 1073       | failed |

Time to target: full fine-tuning reaches zero release misses after its first
epoch, 58 s. Head-only never reaches it; its best release accuracy is 0.775 and
its held-out accuracy plateaus near 0.89, with a constraint-violation rate above
10 percent. The accuracy gap on the attested set is 22 points, and the
compile-time difference is moot because the cheaper regime does not ship.

## Head-only on cached embeddings

Encoding the corpus and the release cases once with the frozen encoder (9,489
rows) took 25.0 s; fitting a linear head on the cached embeddings for 40 epochs
took 7.8 s, about 0.2 s per epoch with the release cases scored after every
epoch. This is what "seconds-fast compilation" would look like if the head
alone sufficed:

| Fit epoch | Fit s | Held-out accuracy | Release accuracy | Release misses |
| --------- | ----- | ----------------- | ---------------- | -------------- |
| 1         | 0.3   | 0.745             | 0.638            | 29             |
| 6         | 1.3   | 0.849             | 0.813            | 15             |
| 16        | 3.2   | 0.904             | 0.800            | 16             |
| 31        | 6.1   | 0.926             | 0.863            | 11             |
| 40        | 7.8   | 0.936             | 0.888            | 9              |

The trainer's verification of the final head reports calibrated accuracy 0.936,
ECE 0.016, 574 constraint violations (6 percent) and 9 release misses: failed.
The pipeline is fast (33 s end to end including encoding) but the frozen
pretrained embeddings do not separate the refund policy's boundaries (day
windows by tier, prior-refund thresholds) well enough for a linear head.

## Laya-style warm start

The pinned `convaiinnovations/laya-typed-decisions` checkpoint (ModernBERT-large,
option scoring at mask markers) was asked the benchmark's own choice question
for every training row and release case (92 s for 9,489 questions on the GPU):

| Stage                                             | Release accuracy | Release misses | Held-out accuracy |
| ------------------------------------------------- | ---------------- | -------------- | ----------------- |
| Laya zero training (argmax of its probabilities)  | 0.225            | 62             | 0.382 on labels   |
| Distilled into a fixed head, no generated labels  | 0.225            | 62             | 0.370             |
| Refined on the generated labels, 40 cached epochs | 0.875            | 10             | 0.934             |
| Head-only from random init, 40 cached epochs      | 0.888            | 9              | 0.936             |

Laya's universal scorer answers the refund question at 0.225, the same number
the benchmark recorded for it as a baseline, and agrees with the corpus labels
on 38 percent of rows; distillation reproduces that exactly, so the warm start
begins below chance for this task. Refining from it tracks the random-init
head-only curve epoch for epoch and ends slightly behind it (10 misses against
9). Time to target: not reached by either. Total warm-start cost was 129 s
(load, score, distill, refine) against 33 s for head-only from scratch.

## What the numbers say

- **A universal encoder with a tiny head does not compile this function.**
  Frozen ModernBERT-base embeddings, whether trained through the standard loop
  or on cached embeddings, stall at roughly 0.89 held-out and 0.78 to 0.89
  attested accuracy with constraint-violation rates far above the gate. Full
  fine-tuning passes the gate after one epoch, 58 s on this GPU, and the
  committed release recipe's remaining epochs buy calibration and fewer
  violations, not the attested result.
- **The compile-time floor is one fine-tuning epoch, not a head fit.** At the
  corpus size the pipeline uses, the honest "fast compile" is a one- or
  two-epoch fine-tune (58 to 106 s), and the build cache (TASK-7.2) already
  removes retraining for unchanged expressions; incremental heads on an
  application's own fine-tuned encoder (the cache's changed-expression path)
  are a different regime from a universal frozen encoder and were not measured
  here.
- **The Laya warm start is a null result for this task.** A scorer trained on
  other typed-decision workflows brings no signal for the refund policy, so
  distilling it gives the head nothing to keep. It should not be adopted as a
  compile path; TASK-5.14 may still test its encoder weights as an
  initialisation for full fine-tuning, which is a separate question.
- Nothing here changes what ships: artifacts stay per-application encoders plus
  heads, and no runtime depends on a universal model.
