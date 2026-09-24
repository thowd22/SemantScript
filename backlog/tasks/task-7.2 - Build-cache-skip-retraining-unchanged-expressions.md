---
id: TASK-7.2
title: 'Build cache: skip retraining unchanged expressions'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-24 18:00'
labels:
  - cli
  - trainer
milestone: m-3
dependencies:
  - TASK-7.1
parent_task_id: TASK-7
ordinal: 25000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Retraining every function on every build is unacceptable. Function ids are content-addressed; use them to reuse existing heads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A rebuild with no changes performs no training
- [x] #2 Changing one expression retrains only that head
- [x] #3 Cache location and invalidation rules are documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Findings: trained PyTorch state is never persisted; only the exported ONNX artifact and the synthetic/adversarial dataset caches survive a build, and the exporter re-validates every function from a TrainingResult and VerificationResult (model state digest, corpus split digest), so reuse must rehydrate exact training state rather than splice manifests. Function ids are content-addressed from the semantic identity, which is the natural cache key; add_function_head already trains one new head on the frozen shared encoder and adapter, leaving every existing head byte-identical.
1. Build cache module (semantscript_trainer/build_cache.py) under the trainer's cache directory: applications/<application-id>/ holds application.json (recipe digest over encoder name and revision, canonical input version, adapter bottleneck, training/verification/adversarial configs and the teacher descriptor; the last release digest; function index), shared.safetensors (exact encoder and adapter state) and functions/<function-id>/ with function.json (semantic digest, recipe digest, dataset and adversarial digests, epoch metrics, selected epoch, the full verification record, tokenizer and model-state digests, verified IR digest), verified-ir.json and head.safetensors. Every file is digest-checked on load; a mismatch is a miss.
2. train_bundle always trains through the shared-encoder application (one adapter even for one function, so every cached state rehydrates the same way) and gains use_cache and full flags. Per bundle function it is reused when its id and semantic digest match a cache record built under the same recipe; otherwise it trains. No changes and the cached release still published at the artifact root: return the existing release with no training and no export. Some reuse: rehydrate the application from shared.safetensors plus cached heads, rebuild each reused function's corpus and split from the cached datasets (no teacher calls) and its TrainingResult and VerificationResult from the record, then train each changed function with add_function_head on the frozen shared modules, verify it, export the whole artifact and store the new records. A recipe change, --full or --no-cache retrains everything jointly. The report gains per-function cache status and a summary.
3. Node CLI: pass --no-cache and --full through and render the cache summary; document location, layout and invalidation rules in cli/README.md and the trainer README.
4. Tests: Python (rebuild with no changes performs no training and keeps the release; changing one expression of a two-function bundle retrains only that head, with the untouched function's verification record and head digest intact and add_function_head the only training call; recipe change and --full retrain everything; a corrupted cache file is a miss) and Node (flag passthrough, cache summary rendering). Lint and suites; real check on this machine with the review-tone program (second build reports reuse in seconds).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented per the plan: semantscript_trainer/build_cache.py (ApplicationCache, CacheRecipe, CachedFunction; exact weights in safetensors; every file digest-checked) and a cache-aware train_bundle (always trains over the shared-encoder application; reuse when id, semantic digest, recipe, shared-state digest and dataset digests match; no-change builds return the published release with no training and no export; changed functions train only their head with add_function_head on the restored frozen shared modules; recipe change, --full or --no-cache retrain jointly and discard old records). Node CLI passes --no-cache, --full and --adapter-bottleneck-size through and renders the cache column and summary. Evidence: AC1 trainer/tests/test_cli.py::test_rebuild_without_changes_performs_no_training_and_keeps_the_release monkeypatches train_application, add_function_head and the exporter to fail if called and asserts the second build reports reused with the same release digest and zero teacher calls; on this machine the review-tone program rebuilt in 0.63 s wall with 'train reused: nothing changed, no training performed' after a 26.5 s first build (release 2d83b2be...), cache 569 MB. AC2 test_changing_one_expression_retrains_only_that_head edits one of two expressions and asserts add_function_head is the only training call, the untouched function's manifest heads, verification and head resource digest are identical to the first release, the artifact loads (semantscript test ok) and a third build reuses both. Also covered: recipe change / --full / --no-cache retrain jointly then reuse again; a corrupted head.safetensors is a miss; a deleted artifact is re-exported from cache without training. AC3 cli/README.md 'Build cache' section and the trainer README document the location, layout, key, reuse conditions, the three outcomes and the incremental-head trade-off. Suites: pytest trainer/tests/test_cli.py 10 passed; full Python 506 passed; Node 68/99/7/83; npm run lint clean. Side note: a pip --target --upgrade install earlier in the session replaced .python-packages/bin and removed the ruff binary; it was reinstalled (ruff 0.16.8).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebuilds no longer retrain unchanged expressions. The trainer keeps a content-addressed build cache under the cache directory (per application: exact encoder and adapter weights; per function: head weights, verified IR bytes, verification record and the digests that bind them to the recipe, shared state and datasets). semantscript train reuses every function whose id, semantic digest, recipe and datasets match: a bundle with no changes performs no training and, when its release is still published, no export (0.63 s on the real encoder against 26.5 s for the build before it); a changed expression trains only its own head on the frozen shared encoder and adapter, leaving every other function's weights and evidence byte-identical; recipe changes, --full and --no-cache retrain everything jointly. Location, layout and invalidation rules are documented in cli/README.md and the trainer README. Verified with 10 driver tests (including monkeypatched no-training proofs and an edit-one-of-two rebuild), the full suites (Python 506, Node 68/99/7/83), lint, and the real timing above; commit b1b95d3.
<!-- SECTION:FINAL_SUMMARY:END -->
