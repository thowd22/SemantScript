---
id: TASK-13
title: 'Docs: Phase 4 — framework guide, reference app walkthrough and public README'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-20 02:47'
updated_date: '2026-09-25 03:16'
labels:
  - docs
milestone: m-4
dependencies:
  - TASK-8.3
ordinal: 38000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 4 is the first time SemantScript looks like a product. The reference application and framework layer need documentation aimed at someone deciding whether to adopt it, not only someone contributing to it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A framework guide covers controllers, request handling, persistence and transactions with sema expressions
- [x] #2 A walkthrough of the reference application explains its source, compiled output and artifact side by side
- [x] #3 The top-level README states what SemantScript is, shows a minimal example, and links the tutorial, references and architecture pages
- [x] #4 An architecture overview page ties compiler, trainer, model and runtime together with one diagram
- [x] #5 All docs pages are reviewed for accuracy against the shipped code and stale content is removed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. docs/framework-guide.md: controllers and decorators, RequestContext and replies, require guards before and after neural results, request scopes and the x-sema-passes header, Express/Nest mounting and middleware, Next.js route handlers, persistence with transactional/rollback/gate over pg or PGlite, a full handler from the refund service, and how to test handlers with the fixture artifact.
2. docs/reference-application.md: the refund service walkthrough at docs level, one expression side by side as source, compiled JavaScript, IR record and artifact manifest entry, the domains and stages, the logic tally and measured results, linking the example README for the full tables.
3. README.md at the repository root: what SemantScript is, the minimal example, how the build works in three lines, status (private workspace packages, repo-local install), and links to the tutorial, references, architecture, examples and benchmarks.
4. docs/architecture.md: one Mermaid diagram tying compiler, trainer, model, runtime, CLI and framework together with the data contracts between them (IR bundle, artifact), and a build-time versus request-time walk.
5. Review every docs page against the shipped code (CONTRIBUTING's cli and examples rows, the index, the language and IR references, the research notes), fix or remove stale content, format, link-check, run the docs test and lint.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added docs/framework-guide.md (controllers and decorators, RequestContext and replies, require guards before and after the model, request scopes and the x-sema-passes header, Express/Nest mounting and middleware, Next.js handlers, transactional/rollback/gate over pg or PGlite with the refund handler in full, testing handlers), docs/reference-application.md (refundRisk as source, compiled JavaScript, IR record and manifest entry side by side; domains and stages; what stayed deterministic and why; results and how to run), README.md at the root (what it is, the minimal example, how it works in three steps, links to tutorial, architecture, references, status as private workspaces), docs/architecture.md (one Mermaid diagram over compiler, trainer, model, runtime, CLI, framework with the IR bundle and artifact as the contracts; build-time and request-time steps; contracts and invariants). Review under AC5: fixed CONTRIBUTING's cli row (init and dev were missing) and examples row (reference application), added the typed-decisions benchmark row, rewrote the index intro, formatted docs/research/laya-analysis.md which failed prettier; grep for stale phase/task references in reader-facing pages found none. Verified: prettier clean over docs and README, zero broken relative links across README and every docs page, docs examples test passes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 4 documentation: a framework guide (controllers, guards, request scopes, Express/Nest/Next.js mounting, transactions gated by decisions), a reference-application walkthrough with one expression shown as source, compiled output, IR and artifact entry, a top-level README stating what SemantScript is with the minimal example and links to the tutorial, references and architecture, and an architecture overview with one diagram tying compiler, trainer, model, runtime, CLI and framework together. All docs pages were reviewed against the shipped code; stale CONTRIBUTING rows were fixed and the index rewritten. Verified with prettier, a link check over README and docs, and the docs examples test.
<!-- SECTION:FINAL_SUMMARY:END -->
