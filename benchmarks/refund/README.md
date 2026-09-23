# Refund benchmark foundation

This private workspace owns the machine-checked contracts and deterministic metric
math for the canonical refund-decision benchmark. It does **not** contain a real
held-out dataset, human attestation, model prediction, or performance result.
Those records must only be committed after the cases have actually been authored,
licensed or de-identified, attested, frozen, and run through every required system.

The version-1 JSON Schemas are in the repository `schemas/` directory:

- `refund-benchmark-dataset.v1.schema.json`
- `refund-training-input-ledger.v1.schema.json`
- `refund-benchmark-predictions.v1.schema.json`
- `refund-benchmark-result.v1.schema.json`

Runtime validation is stricter than JSON Schema where relationships are involved.
It requires evaluation-only cases in stable ID order, a semantic-JSON digest for
every input and payload, at least one attested non-teacher case (either
`human-authored` under a human attestation, or `independent-judge` under a judge
attestation that names the judge model, session and rubric digest), and exact
separation from the union of examples, synthetic, adversarial, calibration, and
release-verification inputs. Calibration entries may repeat in
their originating synthetic/adversarial partition; this is recorded lifecycle
reuse, not duplicate training data. Predictions must cover the frozen dataset exactly once and
carry all three probabilities in `approve`, `deny`, `review` support order. The
probabilities must sum to one within `1e-12`; ties use the first support value.

Datasets and training ledgers are accepted only for the canonical compiled refund
function (`nf_955824…`, semantic SHA-256 `29b03f7d…`). Every prediction and
result binds the canonical task-spec digest and a role-specific immutable
model/adapter identity. SemantScript records additionally bind the exact training
ledger, the loaded artifact function's training-dataset digest to the ledger's
base-dataset digest, and the artifact training key to a canonical semantic digest
of the function plus base, optional adversarial, release payload, and release
attestation identities. Baseline evidence must be `null`.

Warmup uses a separately supplied, validated corpus. Exact evaluation inputs may
never be warmed, and the ordered warmup-corpus digest is retained in every
measurement protocol. A publishable comparison also requires identical
OS/architecture/CPU/accelerator/memory identity and identical concurrency,
warmup iterations/corpus, and memory scope across systems. Runtime versions,
capture times, measured duration, and observed peak bytes remain per-system.

Metrics are computed without rounding:

- overall and attested-slice (human-authored or independent-judge) exact accuracy;
- 15 equal-width bins of top-1 expected calibration error;
- nearest-rank p50 and p95 latency;
- concurrency-one throughput from the measured wall duration; and
- peak memory with an explicit `process-tree` or `client-only` scope.

The mechanical exit decision is `go` only when every required system is present,
SemantScript p50 is strictly below 10 ms, and SemantScript accuracy is at least the
7B Ollama baseline. Missing SemantScript, 1B Ollama, 7B Ollama, structured-output
API, or Laya predictions always produces `incomplete`, never an inferred result.

Run the focused checks with:

```sh
npm test --workspace @semantscript/refund-benchmark
```
