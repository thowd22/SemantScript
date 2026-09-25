---
id: TASK-10
title: 'Docs: Phase 1 — end-to-end tutorial and component guides for the primitive'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-20 02:47'
updated_date: '2026-09-25 05:57'
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
- [x] #1 A getting-started tutorial takes a reader from a fresh clone to a running refund-decision sema expression with copy-pasteable commands and stated hardware requirements
- [x] #2 Component guides exist for compiler, trainer, model and runtime describing inputs, outputs, configuration and how to run each in isolation
- [x] #3 A teacher-configuration page documents the Anthropic and Ollama backends, cost expectations, and the local-vs-reference comparison results
- [x] #4 The Phase 1 benchmark results and go/no-go are written up with methodology and exact model versions
- [x] #5 Every public function or CLI flag introduced in Phase 1 has a docstring or help text
- [x] #6 The docs index links every new page
- [x] #7 The benchmark write-up includes the encoder size vs latency vs accuracy curve and the per-application sizing rule
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. docs/tutorial-refund-decision.md: from a fresh clone to a running refund-decision expression: prerequisites and hardware (Node 22, Python 3.12, GPU or CPU time, disk), clone and build both halves, the refund-decision .sem.ts from docs/examples, build, train (Anthropic key, OpenRouter route, Ollama, or the constraint path the refund service uses when the policy is complete), test, run, every command copy-pasteable and stated where it was verified.
2. docs/components.md: one guide per component (compiler, trainer, model, runtime) with inputs, outputs, configuration and how to run each in isolation, linking the package READMEs for depth.
3. docs/teachers.md: the Anthropic and Ollama backends and the TOML they take, the OpenRouter and Claude CLI routes to Sonnet, cost and time expectations measured this week, the constraint-labeling path, and the local-versus-reference results (results-local-teacher-2026-09-25, decision-12).
4. docs/phase-1-results.md: the Phase 1 benchmark write-up: task, held-out set and adjudication, systems with exact model versions and digests, protocol, the complete five-system result and the go/no-go (results-final-2026-09-25), the encoder size versus latency versus accuracy curve and the per-application sizing rule (results-encoder-sweep-2026-09-24), depth routing.
5. Docstrings for the nine public Python functions and classes that lack one; check TypeScript public exports for JSDoc and add where missing; CLI flags are documented in the CLI reference and the usage text.
6. Link every new page from docs/index.md and the README; prettier, link check, docs test, lint, pytest.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added docs/tutorial-refund-decision.md (fresh clone to a running decideRefund: hardware and time table, clone and build both halves, the expression from examples/express-app, compile, four teacher routes with measured cost, test and run; the compile step was run here, training needs a key), docs/components.md (compiler, trainer, model, runtime: inputs, outputs, configuration, run-alone commands and code, tests), docs/teachers.md (Anthropic and Ollama TOML, OpenRouter route with what is and is not verified, Claude CLI, constraint labels, measured cost table, the local-versus-reference table from results-local-teacher-2026-09-25, how to write a teacher), docs/phase-1-results.md (task, held-out data and adjudication, five systems with exact versions and digests, corpus and recipe, protocol, results-final-2026-09-25 go/no-go, the encoder sweep curve and the per-application sizing rule, depth routing, the other Phase 1 experiments). Docstrings: 9 public Python functions/classes had none, now all do (ast check: 0 missing); JSDoc added to the 33 package-level TypeScript exports that lacked one; CLI flags are covered by the usage text and docs/cli-reference.md. Index and README link every new page. Verified: prettier clean, zero broken relative links, docs examples test passes, lint:node clean, Node suites 89/105/11/8/85 pass, trainer tests 378 pass. Probe note: the trainer's anthropic backend pointed at OpenRouter reached the model but its strict decoder rejected the reply (more than one content block); documented as unverified rather than claimed; the anthropic SDK was installed into .python-packages for the probe.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-09-23 19:06
---
Handoff readiness note: this story remains To Do and dependency-blocked on the real TASK-5.11 benchmark results and TASK-5.13 teacher decision. Current source material is benchmarks/refund/README.md, benchmarks/refund/program/README.md, and benchmarks/refund/program/CLAUDE_CLI_TRAINING.md. The harness methodology is implemented and audited, but documentation must not present transport/GPU smokes as benchmark results or claim the Phase 1 exit criterion before committed final predictions, environment evidence, and go/no-go exist. TASK-5.14 results are also required for AC #7.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 1 documentation: a fresh-clone tutorial to a running refund decision with hardware, time and teacher routes; component guides for compiler, trainer, model and runtime (inputs, outputs, configuration, run alone); a teachers page with backends, measured costs and the local-versus-reference result; a Phase 1 results write-up with exact model versions, the complete go/no-go and the encoder size versus latency versus accuracy curve with the per-application sizing rule; docstrings or JSDoc on every public function that lacked one; index and README links. Verified with prettier, a link check, the docs examples test, lint, the Node suites and the trainer tests.
<!-- SECTION:FINAL_SUMMARY:END -->
