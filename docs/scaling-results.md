# Scaling results: when fusing helps

The shared-encoder design makes one claim about cost: an application pays for
the encoder once per distinct input and for each sema expression only its head.
This page collects the measurements behind that claim and turns them into
guidance. Every number here is committed under `benchmarks/*/data` with its
environment record; the pages linked from each section have the protocols.

Machine for all CPU figures: AMD Ryzen 9 9900X, WSL2, Node 22, ONNX Runtime
1.30 with the CPU execution provider; GPU figures are an AMD Radeon RX 9070 XT
under ROCm and PyTorch eager, not ONNX.

## Heads per stage: fifty decisions for one and a half

Measured on a derived artifact that clones the verified compact refund
function fifty times over its own encoder and adapter
([`results-stage-scaling-2026-09-24`](../benchmarks/refund/data/results-stage-scaling-2026-09-24/README.md)).
One input, `N` heads in one `callStage`, against `N` independent calls:

| Heads per stage | Fused p50 | One call per head p50 | Speedup | Decisions per second |
| --------------- | --------- | --------------------- | ------- | -------------------- |
| 1               | 21.0 ms   | 20.8 ms               | 1.0x    | 47                   |
| 10              | 22.5 ms   | 217 ms                | 9.7x    | 428                  |
| 50              | 29.7 ms   | 1,133 ms              | 38x     | 1,641                |

Each extra head costs about 0.18 ms. Fifty expressions over one record answer
in the time of one and a half single decisions, and every fused result was
byte-equal to the corresponding single call.

## Distinct inputs per stage: linear, because there is no batching

Same artifact, one head, `B` distinct inputs in one stage:

| Distinct inputs | Stage p50 | Per decision | Decisions per second |
| --------------- | --------- | ------------ | -------------------- |
| 1               | 24 ms     | 24.1 ms      | 41                   |
| 8               | 172 ms    | 21.5 ms      | 46                   |
| 64              | 1,273 ms  | 19.9 ms      | 49                   |

The worker fuses requests by identical canonical input; distinct inputs are
encoded one at a time, so per-decision cost stays at 20 to 25 ms whatever the
stage size. A padded batched encoder run is an untested throughput lever;
until it exists, throughput per core is the single-input encoder latency.

## A real application: five questions from one pass

The typed-decisions benchmark writes each workflow's five questions as five
sema expressions over the case's JSON state, one stage
([`results-2026-09-24`](../benchmarks/typed-decisions/data/results-2026-09-24/README.md)):

| Workflow                    | Mean tokens | CPU ONNX stage p50 | Per decision |
| --------------------------- | ----------- | ------------------ | ------------ |
| `agent_trace_observability` | 118         | 53 ms              | 10.6 ms      |
| `security_incidents`        | 244         | 113 ms             | 22.5 ms      |
| `customer_service`          | 288         | 129 ms             | 25.8 ms      |
| `invoice_processing`        | 312         | 171 ms             | 34.2 ms      |

A case costs what one of its questions would; the stage's cost is the
encoder's, and the encoder's cost is the input length (a full 22-layer
ModernBERT-base at 118 to 312 tokens here).

## Depth routing: the cost of a pass

Fusing shares a pass; depth routing shortens it. The refund policy trained
with its domain cut to the first `n` encoder layers
([`results-depth-sweep-2026-09-24`](../benchmarks/refund/data/results-depth-sweep-2026-09-24/README.md)):

| Depth | CPU chain p50 | Final set accuracy | Encoder ONNX |
| ----- | ------------- | ------------------ | ------------ |
| 4     | 7.2 ms        | 1.000 (160/160)    | 235 MB       |
| 6     | 8.7 ms        | 1.000 (160/160)    | 275 MB       |
| 12    | 17.0 ms       | 1.000 (160/160)    | 396 MB       |
| 22    | 39.4 ms       | 0.994 (159/160)    | 597 MB       |

The [refund service](../examples/refund-service/README.md) runs its three
domains at depth 6 and answers each of its nine expressions at about 3.8 to 5.4 ms
p50 on the CPU. Prefixes are separate graphs on disk, so an application with
domains at two depths ships the shared layers twice.

## Guidance

- **Fuse when several expressions read the same record.** That is the common
  case in a handler (decide, assess risk, pick a method), and it is free: the
  framework opens a request scope, the expressions share the pass, and the
  extra heads cost a fraction of a millisecond each. Ten expressions over one
  record are not ten times slower than one.
- **Fusing does not help across records.** A loop over a hundred orders is a
  hundred encoder passes whether the calls are in one scope or not. Budget
  throughput as `distinct inputs × encoder latency` per core.
- **Chains cost a stage each.** An expression that interpolates another's
  answer encodes a different text and runs after it; the plan makes that
  visible. Keep chains for cases where the earlier answer must change what the
  later one reads.
- **Shorten the pass before parallelizing it.** Depth routing took the refund
  policy from 39 ms to 9 ms with no accuracy loss; input length matters as
  much (the typed-decisions stages scale with tokens). Both act on the one
  term that dominates, the encoder pass.
- **Interfaces and stages are the same cost.** A flat interface with four
  fields and four scalar expressions over the same input both cost one pass
  and four heads; choose by what belongs together (see
  [structured outputs](structured-outputs.md)).
- **Check pass counts, not timings.** `callStage` and request scopes return
  `{ encoder, adapter, head }`; the right shape is one encoder pass per
  distinct input per stage, and a test can assert it.
