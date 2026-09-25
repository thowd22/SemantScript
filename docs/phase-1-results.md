# Phase 1 results: the refund benchmark

Phase 1 asked whether one `sema<T>` expression works end to end and whether
it meets two exit criteria: a median of under 10 ms per call on local
inference, and accuracy at least that of a 7B generative model with
structured output on a held-out set. This page is the write-up; every number
comes from digest-bound records under `benchmarks/refund/data/`, each with a
README of its own.

## The task

A three-way refund decision (`approve`, `deny`, `review`) from a customer
record (`tier`, `priorRefunds`) and an order record (`total`, `ageDays`,
`status`). The policy is stated in full in the sema source and every rule is
also an `always`/`never` constraint (decision-6): orders older than 90 days
are denied; fraudulent orders are never approved and are reviewed when not
stale; paid orders outside the tier window (60 days enterprise, 30 standard)
are denied; paid orders inside the window are reviewed when the history is
suspicious and approved otherwise. Source:
`benchmarks/refund/program/refund-with-confidence.sem.ts`, function
`nf_65e347f7…`, semantic digest `f7efe891…`, task spec `ffc5c594…`.

## Held-out data

Inputs are real, de-identified cancellation requests derived from UCI Online
Retail II (CC BY 4.0) by a committed extraction script; a hash-selected 15%
carry the `fraudulent` status. Labels were adjudicated case by case by an
independent model judge (`claude-fable-5-1`, committed rubric, version 1.1)
and recorded under the `independent-judge` origin with a judge attestation;
no record claims human authorship (decision-5). Two disjoint sets: the
release-verification set (80 cases) gates each artifact, and the final
benchmark set (160 cases, payload `5d71fc08…`) is never seen by training,
calibration or verification, which the training-input ledger proves by
digest (leakage audit: 9,291 training inputs, 0 overlap).

## Systems, with exact versions

| Role                  | Model                                                                                | Version or digest                                                                                       | Adapter                                    |
| --------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| SemantScript          | ModernBERT-base fine-tune, depth 6 of 22 layers, one adapter, one head, float32 ONNX | encoder `answerdotai/ModernBERT-base@8949b909…`; artifact manifest `a3f4b005…`                          | Node runtime, `onnxruntime-node` 1.30, CPU |
| Structured-output API | `claude-sonnet-5`, Messages API with JSON-schema structured output                   | immutable model id; run through OpenRouter's Anthropic-format route (upstream `Claude Platform on AWS`) | `refund-anthropic-structured-output` v2    |
| Ollama 7B             | Qwen 2.5 7B instruct Q4_K_M                                                          | Ollama 0.34.3, manifest `845dbda0…`, weights `2bada8a7…`                                                | `refund-ollama-json-schema` v2, GPU        |
| Ollama 1.5B           | Qwen 2.5 1.5B instruct Q4_K_M                                                        | Ollama 0.34.3, manifest `65ec0654…`, weights `183715c4…`                                                | same                                       |
| Laya                  | `convaiinnovations/laya-typed-decisions` option scorer                               | code `d120d4ba…`, checkpoint `dd079950…`                                                                | `refund-laya` v2, CPU                      |

All generative baselines receive the same policy text, the six hard
constraints and the canonical input JSON, and must return a decision plus a
full probability distribution, so calibration is comparable.

## Training the SemantScript system

Corpus (decision-7, `pooled-v4-2026-09-24`): 9,035 synthetic rows (7,535
real-distribution inputs from the UCI pool labeled by the compiled
constraints, 1,500 `claude-opus-5-5` teacher cases, status-flipped twins of
every stale order) plus 374 adversarial cases (12 boundary, 362
counterfactual) from the Claude Code CLI teacher. Recipe: 8 epochs, batch 16,
learning rate 3e-5, linear schedule with 5% warm-up, best epoch on a 10%
calibration split, seed 1, compact canonical encoding (v2), encoder depth 6;
320 s on an RX 9070 XT. Release gate (decision-8, 1% violation tolerance):
zero attested release misses, calibration accuracy 0.9957, ECE 0.0027, 5
violations across 9,489 records.

## Protocol

One machine (Linux 6.18 WSL2, Ryzen 9 9900X, 16 GB, RX 9070 XT), concurrency
1, ten warm-up iterations over eight synthetic inputs disjoint from the
evaluation set, client-only memory scope, nearest-rank latency percentiles,
15-bin top-1 expected calibration error, sealed prediction sets bound to the
dataset digest. `createBenchmarkResult` refuses to assemble sets whose
hardware or protocol identity differs.

## Results and the go/no-go (`results-final-2026-09-25`)

| System                          | Accuracy (160 attested) | 15-bin ECE | p50 ms   | p95 ms | Requests/s |
| ------------------------------- | ----------------------- | ---------- | -------- | ------ | ---------- |
| **SemantScript, depth 6**       | **1.000** (160/160)     | 0.0029     | **4.82** | 6.41   | 194        |
| Structured-output API, Sonnet 5 | **1.000** (160/160)     | 0.0000     | 4,200    | 5,744  | 0.23       |
| Qwen 2.5 7B (comparator)        | 0.538 (86/160)          | 0.428      | 569      | 608    | 1.81       |
| Qwen 2.5 1.5B                   | 0.519 (83/160)          | 0.443      | 319      | 358    | 3.09       |
| Laya                            | 0.225 (36/160)          | 0.411      | 28.2     | 31.9   | 35.5       |

Mechanical decision: **`go`**. Latency 4.82 ms < 10 ms; accuracy 1.000 ≥ the
7B comparator's 0.538; all five required systems present; leakage audit
clean. The compiled function matches Sonnet 5 case for case and answers
about 870 times faster at the median, on a CPU, with no network and no
per-call cost.

How the record got here: the first complete run (`results-v2-2026-09-23`)
met accuracy (0.994) but not latency (38.4 ms at full depth with the v1
encoding); the compact encoding brought it to 28.5 ms
(`results-compact-2026-09-24`); depth routing at 6 layers to 4.82 ms with
accuracy 1.000 (`results-depth-006-2026-09-25`, decision-10); the API
baseline, blocked until a key existed, ran on 2026-09-25 through OpenRouter.
Each earlier README keeps its original text with an amendment note.

## Encoder size: latency against accuracy (`results-encoder-sweep-2026-09-24`)

One recipe, head and loss on the same corpus; batch-1 latency on a 35-token
input; the release gate as above.

| Encoder                        | Parameters | Release accuracy                                                            | Calibrated acc | ECE    | GPU p50 (PyTorch) | CPU ONNX chain p50 | Encoder ONNX   | Gate   |
| ------------------------------ | ---------- | --------------------------------------------------------------------------- | -------------- | ------ | ----------------- | ------------------ | -------------- | ------ |
| ModernBERT-base (default)      | 149 M      | 1.000                                                                       | 0.9904         | 0.0038 | 8.1 ms            | 27.6 ms            | 597 MB         | passed |
| ModernBERT-large, Laya weights | 395 M      | 1.000                                                                       | 0.9936         | 0.0030 | 13.5 ms           | 93.7 ms            | 1,580 MB       | passed |
| ModernBERT-large, pretrained   | 395 M      | 0.9625                                                                      | 0.9957         | 0.0037 | 14.1 ms           | 87.6 ms            | 1,580 MB       | failed |
| DeBERTa-v2-xlarge (~1B)        | 885 M      | did not train (non-finite logits); 11.1 ms GPU, 850 ms CPU PyTorch, no ONNX |                |        |                   |                    | 3.5 GB weights |        |

With heads sharing one encoder pass, a head costs 5 microseconds at every
size: fifty heads over one input cost the same as one, so size trades latency
for capability and heads never change the ranking. Depth routing then cuts
the base encoder's pass itself: 4 layers 7.2 ms, 6 layers 8.7 ms, 12 layers
17.0 ms, 22 layers 39.4 ms on the CPU chain, all at 1.000 on the final set
except the full stack at 0.994 (`results-depth-sweep-2026-09-24`).

**The per-application sizing rule.** The encoder is a per-application
setting (`--encoder-name`, `--encoder-revision`; default ModernBERT-base).
Under 10 ms per request: ModernBERT-base at depth 4 to 6 on a CPU, or full
depth on a GPU; several decisions over one input share the pass and cost
0.8 ms each at fifty heads. Batch and offline work that tolerates 100 ms:
ModernBERT-large from decision-trained (Laya) weights buys lower ECE and
fewer constraint violations, and pretrained large is the point that fails the
gate, so initialization matters more than size at 400 M (decision-3 stands).
Anything near 1 B exceeds the v1 artifact's single-file limits and needs an
external-data format the current artifact forbids.

## What else Phase 1 measured

- **Universal frozen encoder with tiny heads**: never reaches the gate
  (head-only 0.78 to 0.89 attested); compile time is cut by the build cache
  and short fine-tuning instead ([write-up](universal-encoder.md), decision-9).
- **Int8 quantization** (`results-int8-2026-09-24`): a 145 MB artifact
  published only under a recorded tolerance; the strict gate refused it.
- **Jev**, TypeSafe's hosted decision model, as a diagnostic comparator:
  0.9625 on the final set, every miss the enterprise 60-day window read as
  30 days (`results-jev-2026-09-25`, decision-11).
- **Local teacher**: Qwen3-14B labels agree with the policy on 0.746 of real
  inputs and a student trained on them loses 18 points to the Sonnet-taught
  one ([teachers](teachers.md), decision-12).
