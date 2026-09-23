---
id: TASK-5.12
title: 'Trainer: pluggable teacher interface with Anthropic and Ollama backends'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-20 02:43'
updated_date: '2026-09-22 22:03'
labels:
  - trainer
milestone: m-1
dependencies:
  - TASK-2
  - TASK-4
references:
  - backlog/decisions
documentation:
  - trainer/README.md
modified_files:
  - pyproject.toml
  - DEVELOPING.md
  - scripts/check-python.mjs
  - trainer/README.md
  - trainer/src/semantscript_trainer/__init__.py
  - trainer/src/semantscript_trainer/case_contract.py
  - trainer/src/semantscript_trainer/strict_json.py
  - trainer/src/semantscript_trainer/teacher.py
  - trainer/src/semantscript_trainer/teacher_config.py
  - trainer/src/semantscript_trainer/teacher_prompt.py
  - trainer/src/semantscript_trainer/teachers/__init__.py
  - trainer/src/semantscript_trainer/teachers/anthropic.py
  - trainer/src/semantscript_trainer/teachers/ollama.py
  - trainer/tests/test_anthropic_teacher.py
  - trainer/tests/test_case_contract.py
  - trainer/tests/test_ollama_teacher.py
  - trainer/tests/test_public_api.py
  - trainer/tests/test_strict_json.py
  - trainer/tests/test_teacher_config.py
  - trainer/tests/test_teacher_contract.py
  - trainer/tests/test_teacher_prompt.py
parent_task_id: TASK-5
ordinal: 32000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Decision-1 uses two teachers: claude-sonnet-5 as the reference and Qwen3-14B via Ollama as the local candidate. The synthetic case generator must not be coupled to either. One interface, two implementations, selected by config.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Teacher interface generate(ir, n) -> cases exists and the case generator depends only on it
- [x] #2 Anthropic backend uses the official Python SDK with structured outputs whose JSON schema is derived from the IR; every returned case validates against the input schema and output type
- [x] #3 Anthropic backend can submit large runs through the Batch API and collect results by custom_id
- [x] #4 Ollama backend targets the OpenAI-compatible endpoint and works against a WSL-native Ollama using the AMD GPU through ROCm/DXG
- [x] #5 Backend and model name are selected by config with no code change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define an immutable provider-neutral Teacher protocol, generated-case/descriptor types, typed failures, deterministic prompt projection, and a generator boundary that depends only on Teacher. 2. Derive one-case structured-output JSON Schema from every v1 IR input/output kind and independently validate strict JSON responses with duplicate-key, non-finite, size/depth/node/count safeguards. 3. Add secret-free, digestible config and a factory selecting Anthropic or Ollama/model/base URL without code changes. 4. Implement Anthropic direct and Message Batch paths with official SDK request shapes, deterministic custom IDs, polling/timeouts, unordered result reconciliation, and exact local validation. 5. Implement Ollama through the OpenAI-compatible /v1 chat endpoint, default localhost configuration plus WSL mirrored-network guidance and configurable gateway fallback. 6. Add injected-client unit/contract tests, an opt-in live Ollama probe, package dependencies/exports/docs, independent audits, and bounded full-suite verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a provider-neutral Teacher protocol and typed generation failures; deterministic IR prompts; strict, bounded JSON parsing; IR-derived response schemas; exact case validation; and configuration-driven provider construction. Added Anthropic direct and Message Batch implementations with schema lowering, deterministic custom IDs, bound resumable handles, unordered reconciliation, and local revalidation. Added the Ollama OpenAI-compatible implementation with structured output, non-thinking requests, configurable local endpoints, and WSL ROCm/DXG deployment guidance. Added injected-client contract coverage, public API coverage, an opt-in live Ollama probe, and source-first Python check isolation.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-09-22 19:00
---
TASK-5.12 started as the next ready ordinal item after TASK-6.2. Reconnaissance confirms the trainer is currently only a package stub, so this task establishes the provider boundary and exact IR-derived case contract without implementing TASK-5.4 caching/dataset assembly. Official current API review selected Anthropic messages.create output_config.format and Message Batches keyed strictly by custom_id; Ollama uses the OpenAI-compatible /v1 endpoint with response_format plus mandatory local validation. A localhost:11434 preflight currently finds no server. The implementation will keep localhost as the mirrored-WSL default, expose base_url for NAT/gateway setups, and add an opt-in live test; no model pull or external billing is authorized.
---

author: @codex
created: 2026-09-22 19:26
---
Core and provider implementation is integrated. The bounded trainer suite passes 115 tests with only the opt-in live Ollama probe skipped. Independent audits caught and fixed pre-transport case-count enforcement, unbounded result materialization, binary64 JSON integer handling, typed configuration failures, and public API exposure. Final work is addressing a real Anthropic SDK schema-transform incompatibility plus batch-handle binding, then running the live Ollama smoke and the capped repository suite.
---

author: @codex
created: 2026-09-22 19:56
---
Acceptance criteria 1, 2, 3, and 5 are verified. The real pinned Anthropic 1.7 transformer passes; provider/core tests pass; the capped full repository check passes at 535392 KiB peak RSS with zero swaps. The full check also exposed a stale .python-packages import precedence issue; scripts/check-python.mjs now places trainer/src and model/src first, and the rerun passes 129 Python tests plus all Node/TypeScript suites. Criterion 4 remains open: no Windows Ollama process or standard Windows Ollama installation was found, while only /usr/local/bin/ollama exists inside WSL.
---

author: @codex
created: 2026-09-22 20:42
---
Criterion 4 is updated to the user-selected WSL-native deployment now that AMD GPU access is available. Live verification used matched Ollama 0.34.2 base and ROCm bundles, HSA_ENABLE_DXG_DETECTION=1, RX 9070 XT gfx1201/ROCm0, and glm-4.7-flash:latest. Startup logs showed all 48 model layers offloaded to ROCm0, and the real OpenAI-client adapter test passed; the full live trainer suite passed 129/129 in 4.25 seconds. The installed 0.16.2 system service remains CPU-only until its base/ROCm bundle and service environment are upgraded with sudo; trainer/README.md now documents those exact persistent-service steps.
---

author: @codex
created: 2026-09-22 22:03
---
2026-09-22 persistent-service verification after user upgrade: /usr/local/bin/ollama client and service are v0.34.3; systemd service is active with HSA_ENABLE_DXG_DETECTION=1. A bounded glm-4.7-flash request returned exactly OK at num_ctx=2048 and num_predict=4. Ollama detected ROCm0 (AMD Radeon RX 9070 XT), reported size_vram=13,001,608,396 bytes, and logged offloaded 48/48 layers to GPU. The model was explicitly unloaded after verification; WSL remained healthy with ~14 GiB available memory. Observed a non-fatal rocBLASLt warning that TensileLibrary_lazy_gfx1201.dat was absent; inference still completed successfully.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Delivered the pluggable teacher layer with config-selected Anthropic and Ollama backends, strict IR-derived structured-output validation, Anthropic batch support, and comprehensive provider/core tests. Verified the Ollama adapter end to end on the WSL-hosted RX 9070 XT through ROCm/DXG: Ollama 0.34.2 detected gfx1201, offloaded all 48 GLM layers, and the live trainer suite passed 129/129. The capped repository check passes all Node, TypeScript, and Python suites at 542336 KiB peak RSS with zero swaps. Persistent GPU service upgrade instructions are documented because applying them requires interactive sudo.
<!-- SECTION:FINAL_SUMMARY:END -->
