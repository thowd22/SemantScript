# Refund benchmark: the complete result (2026-09-25)

The first assembly of the Phase 1 exit benchmark with all five required
systems present, and the first with a mechanical decision: **`go`**. It joins
the depth-routed SemantScript run (`results-depth-006-2026-09-25`), the three
committed baseline runs from the go/no-go record (`results-v2-2026-09-23`:
Qwen 2.5 1.5B and 7B through Ollama, Laya) and the structured-output API
baseline, which ran here for the first time. `result.json` is the
machine-checked record; every prediction set in this directory is the sealed
file its own run produced, byte for byte.

## The structured-output API baseline

The role's pin is `claude-sonnet-5` on the Anthropic Messages API with JSON
schema structured output (`refund-anthropic-structured-output` adapter,
version 2). No Anthropic key exists on this machine; the run used the same
pinned model through OpenRouter's Anthropic-format `/api/v1/messages` route
with the user's OpenRouter key. The transport sends the canonical request body
with the model named `anthropic/claude-sonnet-5` on the wire and strips that
route prefix from the response, so the adapter, prompt, schema and pinned
identity are unchanged; the difference is recorded where it belongs, in the
sealed set's execution backend (`endpoint: "https://openrouter.ai/api"`),
which the contracts and schema were amended to accept beside
`https://api.anthropic.com` (`program/README.md`, "Structured-output API
baseline through OpenRouter"). OpenRouter reported the upstream provider as
`Claude Platform on AWS`. The key was read from the environment and appears
in no record. Cost: about USD 0.53 for the 170 requests (10 warm-ups and 160
cases), USD 0.003 per case.

## Results

Same 160 judge-attested final cases, concurrency 1, ten warm-up iterations
over the same synthetic inputs (digest `30bc6f58…`), client-only memory
scope, nearest-rank percentiles, 15-bin top-1 ECE; hardware identity
(Ryzen 9 9900X, 16 GB, WSL2, RX 9070 XT) identical across the five sets, which
`createBenchmarkResult` checks before it will assemble them.

| System                                           | Accuracy            | Attested | 15-bin ECE | p50 ms   | p95 ms | Requests/s | Where it runs                  |
| ------------------------------------------------ | ------------------- | -------- | ---------- | -------- | ------ | ---------- | ------------------------------ |
| **SemantScript**, depth-6 release `a3f4b005…`    | **1.000** (160/160) | 1.000    | 0.0029     | **4.82** | 6.41   | 194.2      | CPU, Node runtime, no network  |
| Structured-output API, `claude-sonnet-5`         | **1.000** (160/160) | 1.000    | 0.0000     | 4,200    | 5,744  | 0.23       | OpenRouter to Anthropic on AWS |
| Qwen 2.5 7B instruct Q4_K_M, Ollama (comparator) | 0.538 (86/160)      | 0.538    | 0.428      | 569      | 608    | 1.81       | local GPU                      |
| Qwen 2.5 1.5B instruct Q4_K_M, Ollama            | 0.519 (83/160)      | 0.519    | 0.443      | 319      | 358    | 3.09       | local GPU                      |
| Laya typed-decisions checkpoint                  | 0.225 (36/160)      | 0.225    | 0.411      | 28.2     | 31.9   | 35.5       | local CPU                      |

Go/no-go, mechanical: latency criterion `semantscriptP50Ms` 4.82 < 10, passed;
accuracy criterion 1.000 ≥ the 7B comparator's 0.538, passed; no missing
systems; leakage audit 160 evaluation cases against 9,291 training inputs,
0 overlap; status **`go`**.

## What the numbers say

- **Sonnet 5 solves this policy from the text alone**: 160 of 160 with
  probability 1.0 on every answer (so an ECE of zero, which here means
  certainty that happened to be right every time, not a calibrated
  distribution). It is the accuracy ceiling for the task, and the compiled
  function matches it.
- **The cost of that answer is the point of the benchmark.** The API baseline
  takes 4.2 s per decision over the network at USD 0.003 each; the compiled
  artifact takes 4.8 ms on a CPU core with no network and no per-call cost,
  about 870 times faster at the median, and is verified against the policy's
  constraints before release. The size-controlled local generative baselines
  do not get near either: the 7B model reads the same text and answers just
  over half the cases correctly.
- **Both exit criteria are met in one machine-checked record.** The written
  decisions in `results-v2-2026-09-23` (accuracy met, latency not met) and
  `results-depth-006-2026-09-25` (latency met, mechanical status incomplete)
  are superseded by this record; their text is left as written.

Records: `result.json` (assembled result and go/no-go), `predictions-*.json`
(five sealed prediction sets), `environment.json` (captured at this run),
`failures.json` (empty). Assembled with:

```sh
node benchmarks/refund/program/run-benchmark.mjs run \
  --dataset benchmarks/refund/data/heldout-uci-2026-09-23/final-benchmark-dataset.json \
  --output-dir benchmarks/refund/data/results-final-2026-09-25 \
  --warmup-source benchmarks/refund/data/sonnet-pilot-2026-09-23/synthetic/datasets/v1/d3/d35bc8b85dd0fadadc74c46e176d666cd325faf18bc0e0e0231076ad07287c05.json \
  --systems structured-api
# predictions-{semantscript,ollama-1b,ollama-7b,laya}.json copied from their committed runs
node benchmarks/refund/program/run-benchmark.mjs assemble \
  --dataset benchmarks/refund/data/heldout-uci-2026-09-23/final-benchmark-dataset.json \
  --ledger benchmarks/refund/data/release-depth-006-2026-09-25/training-input-ledger.json \
  --output-dir benchmarks/refund/data/results-final-2026-09-25
```
