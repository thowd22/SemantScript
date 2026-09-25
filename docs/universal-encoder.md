# Can compilation take seconds? The universal-encoder investigation

Every `sema` expression is a head over an encoder that the trainer fine-tunes
for the application. The plan asked whether that fine-tuning is necessary:
train one universal semantic encoder once, keep it frozen, and compile a new
expression by fitting only its tiny head, in seconds rather than minutes. A
related idea from the Laya typed-decision models was a warm start: a universal
option scorer answers the new question with no training at all, its answers
are distilled into a fixed head, and training refines from there. TASK-7.3
measured both against ordinary fine-tuning; decision-9 records the outcome.

## The experiment

Everything ran on the refund benchmark's frozen corpus (9,409 rows: 9,035
synthetic and 374 adversarial, replayed from cache with no teacher calls) and
the canonical refund function, on the same GPU as the release runs (AMD Radeon
RX 9070 XT). The target is the release gate's attested criterion: zero misses
on the 80 judge-attested release cases, with the trainer's own verification
(temperature scaling, ECE, constraint violations under decision-8's 1%
tolerance) applied to every regime so the numbers are those of a real build.
Records: [`results-warm-start-2026-09-24`](../benchmarks/refund/data/results-warm-start-2026-09-24/README.md).

Three regimes:

- **Full fine-tuning**: the committed release recipe (ModernBERT-base, batch
  16, sequence 128, linear head), 1 to 8 epochs.
- **Head-only**: the same recipe with the encoder frozen in eval mode
  (`TrainingConfig.freeze_encoder`, added for this experiment), head learning
  rate raised to 5e-3; also measured on cached embeddings (encode the corpus
  once, fit the head alone for 40 epochs), which is what "seconds-fast
  compilation" would cost if it worked.
- **Laya warm start**: the pinned `convaiinnovations/laya-typed-decisions`
  checkpoint asked the benchmark's own question for every row, its
  probabilities distilled into a fixed head, then refined on the generated
  labels.

## What was measured

| Regime                         | Epochs | Wall time    | Release accuracy | Release misses | Gate   |
| ------------------------------ | ------ | ------------ | ---------------- | -------------- | ------ |
| Full fine-tuning               | 1      | 58 s         | 1.000            | 0              | passed |
| Full fine-tuning               | 2      | 106 s        | 1.000            | 0              | passed |
| Full fine-tuning               | 8      | 434 s        | 1.000            | 0              | passed |
| Head-only, frozen encoder      | 1      | 31 s         | 0.775            | 18             | failed |
| Head-only, frozen encoder      | 8      | 236 s        | 0.713            | 23             | failed |
| Head-only on cached embeddings | 40     | 33 s total   | 0.888            | 9              | failed |
| Laya zero training             | 0      | 92 s scoring | 0.225            | 62             |        |
| Laya distilled, then refined   | 40     | 129 s total  | 0.875            | 10             | failed |

Full fine-tuning reaches zero attested misses after its first epoch. Head-only
never reaches the target at any budget: it plateaus near 0.89 held-out
accuracy and 0.78 to 0.89 on the attested set, with a constraint-violation
rate above 6% (10% through the trainer), far outside any gate. The accuracy
gap on the attested set is 22 points. The head fit itself is fast (0.2 s per
epoch on cached embeddings), which is the point: the cheap regime does not
ship, so its speed is moot.

The Laya scorer answers the refund question at 0.225, below chance, the same
number the benchmark recorded for it as a baseline; distillation reproduces
that exactly, and refining from it tracks the random-initialisation head-only
curve epoch for epoch and ends slightly behind it.

## Why a frozen encoder is not enough

A pretrained encoder's embedding of a canonical input separates topics, not
policies. The refund decision turns on day windows by tier, prior-refund
thresholds and an order status; those boundaries are not linearly separable in
frozen ModernBERT space, and no head, however well fitted, can draw them
there. Fine-tuning moves the encoder so that they are, and one epoch is enough
for this policy. The universal-scorer warm start fails for the same reason
one level up: its notion of the question is not this application's.

## The recommendation (decision-9)

- Compile time is reduced by fine-tuning the application's own encoder for
  fewer epochs (one or two already pass the attested criterion) and by the
  [build cache](build-cache.md), which skips retraining for unchanged
  expressions and trains only a changed expression's head on the
  application's already fine-tuned encoder. A universal frozen encoder with
  per-function heads is not a compile path.
- The Laya-style warm start is not adopted; it carries no signal for this
  task. Its encoder weights may still be evaluated as an initialisation for
  full fine-tuning, a separate question under the encoder sweep.
- This is strictly a compile-time matter for per-application artifacts.
  Shipped artifacts remain the application's encoder, adapter and heads; no
  runtime depends on a universal model.

What "fast compilation" looks like in practice, then: a first build
fine-tunes for a minute or two per application, and every later build
retrains only the expressions that changed, each as a head on the frozen
application encoder in seconds. Where the encoder pass itself is the cost,
depth routing (decision-10, the [scaling results](scaling-results.md)) cuts
the layers a domain runs rather than sharing a universal model.
