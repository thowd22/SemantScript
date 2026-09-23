---
id: TASK-5.4
title: 'Trainer: synthetic case generation from IR via teacher model'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-22 21:25'
labels:
  - trainer
milestone: m-1
dependencies:
  - TASK-5.12
  - TASK-2
  - TASK-4
documentation:
  - trainer/README.md
modified_files:
  - trainer/src/semantscript_trainer/dataset.py
  - trainer/src/semantscript_trainer/__init__.py
  - trainer/src/semantscript_trainer/teacher.py
  - trainer/src/semantscript_trainer/teachers/anthropic.py
  - trainer/src/semantscript_trainer/teachers/ollama.py
  - trainer/tests/test_dataset.py
  - trainer/tests/test_public_api.py
  - trainer/tests/test_anthropic_teacher.py
  - trainer/tests/test_ollama_teacher.py
  - trainer/tests/training_dataset_v1_empty.golden.json
  - trainer/README.md
parent_task_id: TASK-5
ordinal: 9000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The natural-language spec is consumed at build time only. A teacher model turns the IR (behavior text, input schema, output type, examples) into labeled training cases. Transcript turn 5 describes English -> formal behavior -> generated training set.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Given an IR record, the generator produces N labeled cases whose inputs match the input schema and labels are within the output type
- [x] #2 Provided examples are included verbatim and marked as gold
- [x] #3 Generation is resumable and cached so re-running with the same IR does not re-call the teacher
- [x] #4 Output dataset format is documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a public synthetic-dataset contract with gold/synthetic row origins, exact counts, teacher provenance, stable request/cache identity, and a dataset digest. Treat total_cases as the exact dataset size: validate all IR examples first, retain them in source order as gold, reject total_cases below the gold count, and request only the remainder through the existing backend-neutral CaseGenerator. 2. Define a closed, deterministic JSON v1 dataset envelope and domain-separated cache key over the exact prompt-relevant IR projection, teacher descriptor, total count, and explicit generator/prompt/case contract revisions; exclude source locations and later training/verification metadata. 3. Implement a bounded fail-closed cache with per-key process locking, second-look reads, strict metadata/case revalidation, deterministic UTF-8 serialization, fsynced same-directory staging, verification, atomic publication, and no provider recall on valid hits. Completed files are resumable across processes; provider calls remain intentionally all-or-nothing because the current Teacher protocol has no indexed checkpoint API. 4. Add tests for exact total/gold semantics, zero and invalid counts, all validation boundaries, deterministic keys/digests, cache hits and invalidation, corruption, interrupted writes, and concurrent single-call behavior. 5. Export the API and document the dataset schema, cache/recovery rules, total-count semantics, provenance limitations for mutable model aliases, and downstream usage. 6. Run focused tests, independent audits, and the memory-capped repository suite before finalizing every acceptance criterion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented SyntheticDatasetGenerator with exact total_cases semantics, ordered gold/synthetic DatasetCase snapshots, strict IR/example validation, request/payload/file digests, and a closed deterministic JSON v1 envelope. Cache identity binds the prompt-relevant IR projection, teacher descriptor, count, and contract revisions. Cache reads are bounded and fail closed; per-key process locks, second-look reads, fsynced staging, verification, and atomic replacement prevent duplicate calls and partial publication. IR and returned rows are immutable snapshots, signed zero is preserved through the exported strict loader, and built-in providers enforce aggregate response budgets during decoding. Completed-result resumability is intentional because Teacher.generate is an atomic tuple protocol; mutable model aliases require immutable identities or explicit invalidation.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @codex
created: 2026-09-22 20:43
---
TASK-5.4 started as the newly unblocked, lowest-ordinal ready task after completing TASK-5.12. Scope is limited to deterministic dataset assembly around the existing Teacher boundary: preserve IR examples as gold, request only the remaining cases, validate every case, and make identical generation resumable without another provider call. I am beginning with parallel repository and cache-contract audits before recording the implementation plan.
---

author: @codex
created: 2026-09-22 20:51
---
Parallel design audits agree that total_cases must mean the exact returned/on-disk row count: gold examples consume the first slots and only the remainder is requested from the Teacher. They also identified an important boundary: Teacher.generate is an atomic tuple API, so safe resumability is at the completed, atomically published dataset level; splitting into repeated one-row calls would reuse identical prompt positions and seeds. Cache corruption will fail closed instead of silently spending provider credit. TeacherDescriptor currently identifies the declared model/configuration rather than resolved mutable model weights, so documentation will require immutable model identities or explicit cache invalidation when an alias is retargeted.
---

author: @codex
created: 2026-09-22 21:24
---
Final verification proves all acceptance criteria. Exact-N assembly, schema/output validation, gold-first preservation, strict signed-zero handling, no-call gold/zero paths, fresh-process cache hits, same-key concurrency, cache invalidation, fail-closed corruption/symlink/FIFO handling, provider/publication failures, immutable provenance snapshots, aggregate provider budgets, and pinned v1 bytes/digests are covered by the trainer suite. Independent contract and security re-audits found no remaining blocker. The memory-capped full repository check passed all Node/TypeScript suites and 162 Python tests with one expected opt-in live Ollama skip; peak RSS was 552020 KiB with zero swaps.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Delivered exact-size synthetic dataset generation on the pluggable Teacher boundary. IR examples are validated and preserved as leading gold rows; only the remaining rows call the teacher. Deterministic versioned datasets are cached with provenance digests, process locking, strict revalidation, bounded provider responses, and atomic fail-closed publication, so identical reruns make no provider call. The format, canonicalization, recovery/trust boundary, and mutable-model caveat are documented and pinned by a golden vector. Independent security and contract audits found no remaining blocker; the capped repository check passed all Node/TypeScript suites and 162 Python tests with one opt-in live Ollama skip, 552020 KiB peak RSS, and zero swaps.
<!-- SECTION:FINAL_SUMMARY:END -->
