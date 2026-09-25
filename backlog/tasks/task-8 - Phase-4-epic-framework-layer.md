---
id: TASK-8
title: 'Phase 4 epic: framework layer'
status: Done
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-25 05:57'
labels:
  - epic
milestone: m-4
dependencies:
  - TASK-7
  - TASK-13
ordinal: 28000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Thin integrations for popular stacks and a reference application that replaces a meaningful share of hand-written logic. Deliberately last and deliberately thin: adoption comes from dropping sema into existing apps (Phase 3), not from owning the whole application. The transcript is explicit that the framework must not be built before the primitive is proven.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reference application serves HTTP requests whose handlers mix deterministic code and sema expressions
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Epic closure 2026-09-25: AC1 evidence is examples/refund-service, whose three decorated controllers (POST /refunds/:orderId, /tickets/:ticketId/triage, /orders/:orderId/screen) mix require guards, SQL over PGlite and transactional/gate with nine sema expressions in three routed domains; npm test in the example passes 3/3 (bundle shape, the framework path over the runtime's fixture artifact, every controller over the trained artifact through HTTP), and curl examples in its README were exercised. Children Done: TASK-8.1 controllers and request scopes, TASK-8.2 persistence and transactions, TASK-8.3 the reference application, TASK-13 the Phase 4 docs (framework guide, walkthrough, README, architecture).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 4 delivered the thin framework layer (decorated controllers, one sema request scope per handler, deterministic guards, Express/Nest mounting and Next.js handlers, transactions gated by neural decisions with no ambient database access) and the reference application that uses it: an Express API over Postgres whose business policy is nine sema expressions in three routed depth-6 domains, trained from its own constraints, with a walkthrough of source, compiled output and artifact, a logic tally and per-expression accuracy and latency. Verified by the framework and example test suites over PGlite and the trained artifact, and documented in the framework guide, the reference-application page, the architecture overview and the public README.
<!-- SECTION:FINAL_SUMMARY:END -->
