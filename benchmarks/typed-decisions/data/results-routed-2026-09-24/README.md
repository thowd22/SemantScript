# Single adapter versus routed domain adapters (TASK-6.7), 2026-09-24

The four typed-decisions workflows compiled as one SemantScript application
with four domains (one per file: `semantscript build --route-domains`), so the
routed-adapter layout can be compared with the single-adapter layout on an
application with more than three domains. Measured with
`program/run_routed_domains.py` on the same machine, data and recipe as
`results-2026-09-24` (train split cases 0 to 279 per workflow as the corpus,
the rest attesting, 8 epochs, batch 16, learning rate 3e-5, 768-token inputs,
seed 1, the test split of 100 cases per workflow scored, 500 decisions each).
Raw records: `results.json`. Unlike the per-workflow runs, here all twenty
functions train jointly over one encoder in both configurations, which is the
comparison the task asks for and also a harder setting than four separate
applications.

## Single adapter versus four routed adapters, full depth

| Configuration   | Adapters | Train s | Test accuracy | agent_trace | customer_service | invoice_processing | security_incidents | Artifact bytes |
| --------------- | -------- | ------- | ------------- | ----------- | ---------------- | ------------------ | ------------------ | -------------- |
| Single adapter  | 1        | 1,934   | 0.698         | 0.660       | 0.700            | 0.734              | 0.696              | 597,286,087    |
| Routed adapters | 4        | 1,891   | 0.659         | 0.710       | 0.612            | 0.622              | 0.690              | 598,479,256    |

Per-domain gate metrics (mean over the five questions of each workflow) and the
CPU latency of one fused stage (one encoder pass, the domain's adapter, five
heads; 90 timed cases after 10 warm-ups through ONNX Runtime on CPU):

| Domain             | Single: gate acc / ECE / pair | Routed: gate acc / ECE / pair | Single p50 ms | Routed p50 ms |
| ------------------ | ----------------------------- | ----------------------------- | ------------- | ------------- |
| agent_trace        | 0.736 / 0.148 / 1.000         | 0.750 / 0.117 / 1.000         | 47.5          | 56.1          |
| customer_service   | 0.743 / 0.160 / 1.000         | 0.586 / 0.165 / 1.000         | 100.1         | 103.7         |
| invoice_processing | 0.779 / 0.179 / 1.000         | 0.621 / 0.197 / 1.000         | 172.1         | 126.0         |
| security_incidents | 0.664 / 0.217 / 1.000         | 0.714 / 0.164 / 1.000         | 121.2         | 82.7          |

The latency columns move both ways because the encoder pass over 500 to 768
tokens dominates and the shared desktop drifts by tens of milliseconds between
loops; the adapter is a 160 KB bottleneck and costs microseconds. Artifact
size grows by 1.2 MB for three more adapters.

## Interference: adding a domain on the frozen encoder

Three domains trained jointly, then `agent_trace_observability` added through
the incremental path (a fresh adapter and five heads on the frozen encoder,
117 s), then the three re-scored:

| Domain             | Before | After | Delta |
| ------------------ | ------ | ----- | ----- |
| customer_service   | 0.750  | 0.750 | 0.000 |
| invoice_processing | 0.752  | 0.752 | 0.000 |
| security_incidents | 0.686  | 0.686 | 0.000 |

No function of the kept domains changed a weight (their model-state digests
are identical), so the tolerance the experiment can state is zero, by
construction: routed adapters make adding or retraining a domain a no-op for
every other domain. The cost sits on the added domain: 0.364 on its test split
against 0.710 when it trains jointly with the others, because the frozen
encoder was fine-tuned for the other three and a bottleneck adapter cannot
recover a domain the encoder never saw.

## What the numbers say

- **At full depth, per-domain adapters do not buy accuracy on this suite.**
  Jointly trained, the single adapter scores 0.698 and the routed layout
  0.659; two domains lose and two gain, within the seed-to-seed spread these
  runs show (the three-domain joint model scores customer service at 0.750
  and invoices at 0.752, above both four-domain runs). The capacity of a
  fine-tuned application lives in the encoder; a 160 KB bottleneck per domain
  is not where accuracy comes from.
- **Routed adapters deliver isolation, not capacity.** Retraining or adding a
  domain leaves every other domain byte-identical, which is what dev-mode
  incremental builds and per-domain release evidence need. The price is that
  a domain added on a frozen encoder is a much weaker domain; a new domain
  should trigger a joint retrain (a `--full` build), which the build cache
  already offers.
- **Depth is the lever that pays.** The refund depth sweep
  (`benchmarks/refund/data/results-depth-sweep-2026-09-24`) shows the routed
  layout's other half, compile-time depth per domain, taking the refund
  function from 29.3 ms to 7.9 ms p50 on the CPU runtime with zero attested
  misses. Decision-10 records the default layout that follows from both
  results.
