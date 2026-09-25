---
id: TASK-8.2
title: Persistence and transaction integration
status: Done
assignee:
  - '@claude'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-25 01:44'
labels:
  - runtime
milestone: m-4
dependencies:
  - TASK-8.1
parent_task_id: TASK-8
ordinal: 30000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Data stays out of the weights. Handlers read and write a real database; neural decisions gate transactions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Example handler reads from Postgres, evaluates a sema expression, and commits or rolls back a transaction based on the result
- [x] #2 No sema expression has ambient database access
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Framework: a persistence helper over any pg-protocol client (pg Pool/Client, PGlite): transactional(client, fn) runs BEGIN, fn(tx) and COMMIT, or ROLLBACK when fn throws or returns rollback(reason); gate(decision, predicate) expresses 'commit only when the neural decision satisfies the predicate'.
2. Example: an Express handler in examples/express-app that reads customer and order rows from Postgres, evaluates the refund sema expression on those rows only, inserts the refund and commits on approve, and rolls back (no row written) otherwise; PGlite (Postgres 17 in WASM) when no DATABASE_URL, the pg driver otherwise.
3. Ambient access: sema expressions receive only their interpolated inputs; the compiler rejects interpolating a database client (unsupported input type) and the runtime serializes only plain data. A compiler test and a framework test state that.
4. Tests with PGlite: commit path writes the refund row, rollback path leaves the table untouched and the decision recorded nowhere, an exception inside the handler rolls back; docs in the framework README.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Framework persistence helpers (transactional over pg Client, pg Pool or PGlite; rollback; gate) with three PGlite tests: approve commits the refund row, deny and a missing order roll back with nothing written, an exception rolls back and rethrows, a pool is released; and the real fixture runtime's 'review' decision leaving the refunds table empty. Ambient access: compiler test shows interpolating a client object is diagnostic 9112 (unsupported input type); the runtime rejects a client object as input at the call. examples/express-app gains src/refunds.sem.ts (decideRefund over customer and order records) and src/refunds.ts (POST /refunds/:orderId through a decorated controller over PGlite, seeded at startup), built with tspc; the example's node_modules now carries @semantscript/framework and pglite. No local Postgres server exists on this machine, so the Postgres in every test and in the example is PGlite (Postgres 17 in WebAssembly, the same SQL and protocol shape as pg), stated in the READMEs.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added persistence and transaction helpers to @semantscript/framework (transactional, rollback, gate) and a Postgres-backed example handler: POST /refunds/:orderId reads customer and order rows, evaluates the refund sema expression on those values, commits the refund row when the decision is approve and rolls back otherwise. Verified with three PGlite tests (commit, rollback, exception, pool release, and the real runtime's decision gating a write) and a compiler test showing a database client is not an accepted sema input, so no expression has ambient database access; the example builds through tspc. Postgres here is PGlite in process since no server is installed on this machine.
<!-- SECTION:FINAL_SUMMARY:END -->
