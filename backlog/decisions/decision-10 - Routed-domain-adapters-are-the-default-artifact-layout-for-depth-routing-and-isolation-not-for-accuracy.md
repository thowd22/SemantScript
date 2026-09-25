---
id: decision-10
title: >-
  Routed domain adapters are the default artifact layout for depth routing and
  isolation, not for accuracy
date: '2026-09-25 01:17'
status: accepted
---

## Context

TASK-6.7 asked whether compile-time routed domain adapters (one LoRA-sized adapter per controller, file or `@domain`, selected statically in the execution plan, no learned routing) should become the default artifact layout, and, amended on 2026-09-24, whether routing the compute path as well (a depth per domain: how many shared-encoder layers run before that domain's adapter) reaches the Phase 1 latency criterion. The mechanism was built end to end (model, trainer, build cache, artifact, runtime, compiler) and two experiments were run.

The refund depth sweep (`benchmarks/refund/data/results-depth-sweep-2026-09-24`) trained the refund function at depths 4, 6, 8, 12 and 22 of ModernBERT-base's 22 layers with the committed compact-release recipe. Every depth passes the strict release gate with zero attested misses. On the frozen 160-case final set through the CPU Node runtime, depth 4 answers 160/160 at 7.9 ms p50 and depth 6 160/160 at 9.8 ms, against 159/160 at 29.3 ms for the full stack; GPU latency is launch-bound at about 8 ms at every depth. A probe showed ONNX Runtime executes a graph whole whatever outputs are fetched (a depth-4 output of a multi-output graph costs as much as all 22 layers), so each depth ships as its own prefix graph (235 MB at depth 4 against 597 MB full).

The typed-decisions comparison (`benchmarks/typed-decisions/data/results-routed-2026-09-24`) trained the four workflows as one application with four domains, jointly, at full depth. A single shared adapter scored 0.698 over 2,000 decisions and four routed adapters 0.659, two domains up and two down, within the spread joint runs show; latency and artifact size are unchanged in substance (one encoder pass dominates, adapters add 1.2 MB). Adding a fourth domain on the frozen encoder left the other three byte-identical (accuracy delta 0.000, no weight changed) but the added domain reached 0.364 against 0.710 when trained jointly.

## Decision

- Routed domain adapters become the default artifact layout, adopted for what they measurably provide: isolation (retraining or adding a domain cannot move another domain's weights or evidence, which dev-mode incremental builds and per-domain release records need) and depth routing (a depth per domain, chosen from measured accuracy at each depth, which is the lever that reaches the request-path budget). They are not adopted as an accuracy mechanism: at full depth a per-domain bottleneck adapter did not add usable capacity to a jointly fine-tuned encoder on this suite, and the write-ups say so.
- The compiler routes a project when any site carries `@domain`, when `--domain-depth` names a depth, or under `--route-domains`; the default domain is the enclosing class, else the file. Unrouted projects keep the single application adapter and the bundle shape of before, so nothing changes for a one-file application unless asked.
- Depth is a per-domain setting recorded in the bundle and the artifact (`model.encoderDepth`, per-function `encoderRef`), chosen by measurement: the refund domain runs at depth 6 (zero misses, half the constraint violations of depth 4, 9.8 ms), a domain that needs the full stack keeps it. The Phase 1 latency criterion is met by depth routing in the direct runtime measurement; the re-measurement under the committed benchmark harness through the release pipeline is TASK-5.18's.
- A new domain, or a domain whose every expression changed, retrains jointly (`--full`) rather than on the frozen encoder, because the incremental path isolates but does not learn a domain the encoder never saw. The incremental path remains the right one for a changed expression inside an existing domain (head-only on the domain's adapter).

## Consequences

- Prefix graphs duplicate shared layers on disk once per depth. A weight-sharing external-data layout for prefix graphs is a format question left open; it does not change the runtime path.
- `semantscript dev` and the build cache treat the adapter ref and depth as part of a function's cache key (cache version 2), so a depth change retrains that domain only.
- Decision-3 (ModernBERT-base) and decision-9 (fine-tune the application encoder) stand; depth routing composes with both and with the compact encoding of TASK-5.18.1.
