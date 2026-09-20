---
id: TASK-10
title: 'Docs: Phase 1 — end-to-end tutorial and component guides for the primitive'
status: To Do
assignee: []
created_date: '2026-09-20 02:47'
labels:
  - docs
milestone: m-1
dependencies:
  - TASK-5.11
  - TASK-5.13
ordinal: 35000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 1 is the proof that the primitive works. Its documentation is what lets anyone reproduce that proof and understand each component without reading the transcript. Write it while the details are fresh; the benchmark numbers and teacher comparison belong in the docs, not only in benchmarks/.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A getting-started tutorial takes a reader from a fresh clone to a running refund-decision sema expression with copy-pasteable commands and stated hardware requirements
- [ ] #2 Component guides exist for compiler, trainer, model and runtime describing inputs, outputs, configuration and how to run each in isolation
- [ ] #3 A teacher-configuration page documents the Anthropic and Ollama backends, cost expectations, and the local-vs-reference comparison results
- [ ] #4 The Phase 1 benchmark results and go/no-go are written up with methodology and exact model versions
- [ ] #5 Every public function or CLI flag introduced in Phase 1 has a docstring or help text
- [ ] #6 The docs index links every new page
<!-- AC:END -->
