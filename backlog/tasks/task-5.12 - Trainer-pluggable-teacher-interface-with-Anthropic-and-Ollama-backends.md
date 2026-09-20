---
id: TASK-5.12
title: 'Trainer: pluggable teacher interface with Anthropic and Ollama backends'
status: To Do
assignee: []
created_date: '2026-09-20 02:43'
labels:
  - trainer
milestone: m-1
dependencies:
  - TASK-2
  - TASK-4
references:
  - backlog/decisions
parent_task_id: TASK-5
ordinal: 32000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Decision-1 uses two teachers: claude-sonnet-5 as the reference and Qwen3-14B via Ollama as the local candidate. The synthetic case generator must not be coupled to either. One interface, two implementations, selected by config.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Teacher interface generate(ir, n) -> cases exists and the case generator depends only on it
- [ ] #2 Anthropic backend uses the official Python SDK with structured outputs whose JSON schema is derived from the IR; every returned case validates against the input schema and output type
- [ ] #3 Anthropic backend can submit large runs through the Batch API and collect results by custom_id
- [ ] #4 Ollama backend targets the OpenAI-compatible endpoint and works against a Windows-native Ollama from WSL via localhost:11434
- [ ] #5 Backend and model name are selected by config with no code change
<!-- AC:END -->
