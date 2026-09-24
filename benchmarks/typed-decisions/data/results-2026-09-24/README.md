# Typed-decisions results (TASK-6.6), 2026-09-24

Four workflows of the `LocalLLaMA/typed-decisions` suite (revision
`c76749ec`), each written as one SemantScript program with five `sema`
expressions over the case's JSON state, trained as one application per
workflow (five heads over one ModernBERT-base encoder and adapter, batch 16,
sequence 768, learning rate 3e-5, proper-scoring loss, best of 8 epochs by
held-out accuracy, seed 1) on train cases 0-279, with train cases 280-299 held
back for attested verification, and scored on the suite's own test split: 100
cases and 500 decisions per workflow, 2,000 decisions in all. Raw records:
`results.json`. Machine: AMD Radeon RX 9070 XT (ROCm), Ryzen 9 9900X.

## Accuracy against the published numbers

| System                                         | Mode               | Accuracy on the 2,000 test decisions |
| ---------------------------------------------- | ------------------ | ------------------------------------ |
| Laya typed-decisions checkpoint (Laya README)  | specialist         | 0.766                                |
| meraGPT Decider 1 (dataset card)               | general, zero-shot | 0.768                                |
| teacher self-agreement ceiling (dataset card)  | reference          | 0.735                                |
| TypeSafe Jev 1.13.0 (dataset card)             | general, zero-shot | 0.727                                |
| perfect scenario understanding (dataset card)  | reference          | 0.704                                |
| **SemantScript, four applications (this run)** | **specialist**     | **0.701**                            |
| ModernBERT-base specialist (dataset card)      | specialist         | 0.646                                |
| MiniLM-L6 specialist (dataset card)            | specialist         | 0.587                                |
| prior (dataset card)                           | reference          | 0.470                                |

Per workflow (accuracy over 500 decisions; the card's per-question ceilings
run from 0.56 to 0.94):

| Workflow                    | Accuracy | Train s | Questions (accuracy, Brier against the gold distribution)                                                                             |
| --------------------------- | -------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `customer_service`          | 0.738    | 1,048   | action 0.65 / 0.084, category 0.91 / 0.047, churn_risk 0.74 / 0.126, needs_human 0.74 / 0.080, urgency 0.65 / 0.068                   |
| `agent_trace_observability` | 0.730    | 199     | action 0.68 / 0.091, needs_review 0.88 / 0.053, outcome 0.65 / 0.104, risk 0.81 / 0.090, urgency 0.63 / 0.051                         |
| `security_incidents`        | 0.684    | 425     | credential_compromise 0.70 / 0.029, disposition 0.75 / 0.073, severity 0.64 / 0.068, true_positive 0.76 / 0.069, urgency 0.57 / 0.073 |
| `invoice_processing`        | 0.652    | 533     | discrepancy_severity 0.49 / 0.140, disposition 0.68 / 0.131, duplicate 0.90 / 0.031, matches_order 0.87 / 0.081, urgency 0.32 / 0.088 |

Score questions, expected value against the gold expected score: mean
absolute error 0.26 to 0.61 levels and 82 to 100 percent of decisions within
one level (Jev: 0.391 and 0.952; the card's ModernBERT specialist: 0.444 and
0.931).

## Latency per case, all five questions in one execution-plan stage

| Workflow                    | Mean tokens | GPU p50 / p95 (PyTorch eager) | CPU ONNX p50 / p95 | Per decision (CPU p50) |
| --------------------------- | ----------- | ----------------------------- | ------------------ | ---------------------- |
| `agent_trace_observability` | 118         | 12.5 / 20.7 ms                | 53.1 / 73.2 ms     | 10.6 ms                |
| `customer_service`          | 288         | 25.2 / 45.3 ms                | 128.8 / 361.6 ms   | 25.8 ms                |
| `invoice_processing`        | 312         | 28.0 / 29.9 ms                | 171.0 / 218.0 ms   | 34.2 ms                |
| `security_incidents`        | 244         | 22.5 / 25.0 ms                | 112.5 / 329.9 ms   | 22.5 ms                |

Each case costs one encoder pass plus five head passes; the heads are
microseconds, so a case costs what one of its questions would. Laya reports
32.8 ms per question on a Tesla T4 (its own benchmark), which is roughly 164 ms
for a five-question case if questions are answered one at a time; the dataset
card's ModernBERT-base specialist takes 349 ms per case. The GPU here is a
desktop RX 9070 XT, so the numbers are not a like-for-like hardware comparison,
but on both devices the stage costs the same as a single question. ONNX
Runtime on this machine has only the CPU provider, so the GPU column is
PyTorch; the CPU column is the deployed ONNX chain measured in Python, because
no workflow's artifact was published (next section).

## The release gate and this benchmark

The gate refused 18 of the 20 heads (ECE above 0.1 on 28 calibration rows, or
a miss among the five highest-confidence attested cases held out of the train
split), so no workflow artifact was exported and the Node runtime's `callStage`
path was not timed. That is the gate reading the data correctly: the gold here
is the mean of three teacher samples at temperature 0.7, the card's own ceiling
is 0.735, and a head whose calibration set has 28 rows cannot show a 15-bin ECE
under 0.1 on soft labels. The gate is built for policies whose attested cases
are exact; a benchmark of teacher spread needs a different acceptance
criterion, which this task did not change.

## What the numbers say

- **The multi-head plan holds up on an external suite.** One encoder pass
  answers five typed questions at 0.701 overall, 5.5 points above the card's
  ModernBERT-base specialist (0.646) with the same encoder, and at the
  scenario-understanding ceiling (0.704); two workflows sit at or above the
  teacher self-agreement ceiling.
- **Laya's 0.766 is not matched.** Laya fine-tunes a ModernBERT-large encoder
  with per-question option scoring and two-layer heads; this run used
  ModernBERT-base with linear heads and 280 cases per workflow. The encoder
  sweep (`benchmarks/refund/data/results-encoder-sweep-2026-09-24`) shows the
  Laya-initialised large encoder is the measured next step, at about 3x the
  latency.
- **Specialist against generalist is not a ranking.** Jev and Decider answer
  the questions zero-shot; every number in this run is a specialist fitted on
  the train split, as the card asks reporters to state.
