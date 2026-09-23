---
id: TASK-10
title: 'Docs: Phase 1 — end-to-end tutorial and component guides for the primitive'
status: To Do
assignee: []
created_date: '2026-09-20 02:47'
updated_date: '2026-09-23 19:06'
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
- [ ] #7 The benchmark write-up includes the encoder size vs latency vs accuracy curve and the per-application sizing rule
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-09-23 19:06
---
Handoff readiness note: this story remains To Do and dependency-blocked on the real TASK-5.11 benchmark results and TASK-5.13 teacher decision. Current source material is benchmarks/refund/README.md, benchmarks/refund/program/README.md, and benchmarks/refund/program/CLAUDE_CLI_TRAINING.md. The harness methodology is implemented and audited, but documentation must not present transport/GPU smokes as benchmark results or claim the Phase 1 exit criterion before committed final predictions, environment evidence, and go/no-go exist. TASK-5.14 results are also required for AC #7.
---
<!-- COMMENTS:END -->
