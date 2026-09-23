# Benchmarks

This directory owns reproducible evaluation harnesses, baselines, datasets, and
machine-readable results for accuracy, calibration, latency, throughput, and model
size. The refund-decision benchmark is the first end-to-end target.

Benchmark code consumes public compiler, trainer, model, and runtime interfaces; it
must not become a hidden implementation dependency.

The [`refund`](refund/README.md) workspace defines the closed held-out-data,
training-ledger, prediction, result, leakage-audit, and deterministic metric
contracts. It intentionally contains no claimed held-out dataset or benchmark
result until the required human-authored slice has real attestation.
