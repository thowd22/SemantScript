---
id: TASK-5.1
title: 'Compiler: locate sema<T> call sites in .sem.ts files'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-22 04:17'
labels:
  - compiler
milestone: m-1
dependencies:
  - TASK-2
  - TASK-4
documentation:
  - compiler/README.md
modified_files:
  - compiler/package.json
  - compiler/src/index.ts
  - compiler/src/sema-sites.ts
  - compiler/test/package.test.mjs
  - compiler/README.md
  - package-lock.json
parent_task_id: TASK-5
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
First step of the compiler pipeline. The transformer must find every sema tagged-template expression in a TS program so later steps can type it and rewrite it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Given a .sem.ts file, the compiler returns every sema<T> tagged-template site with source location
- [x] #2 Non-sema tagged templates and unrelated identifiers named sema are not matched
- [x] #3 Unit tests cover nested functions, class methods, and arrow functions
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define a deterministic compiler discovery API that returns source nodes, mode/configuration metadata, and one-based source locations for .sem.ts files. 2. Resolve the canonical sema export from @semantscript/core with the TypeScript checker and follow import aliases, namespace access, and re-exports while rejecting spelling-only matches. 3. Recognize the four SPEC tag shapes: direct/configured value and direct/configured withConfidence. 4. Add unit fixtures for nested functions, class methods, arrows, aliases/re-exports/namespace imports, unrelated tags, shadowed sema identifiers, and non-.sem.ts files. 5. Run the unified repository gate and independent discovery/API reviews, then record acceptance evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a TypeScript-program discovery API for legal sema<T> tagged-template forms in .sem.ts files. Matching is based on the canonical @semantscript/core export identity, supports aliases, namespace imports and re-exports, distinguishes value and diagnostic modes, and reports deterministic source locations. Added regressions for lookalikes, lexical shadowing, copied members, structurally module-shaped locals, malformed/optional-chain forms, and non-.sem.ts files. The repository-wide check and two independent correctness audits pass.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @codex
created: 2026-09-22 02:37
---
TASK-1 SPEC defines four legal AST forms that discovery must cover: sema<T>, sema<T>(options), sema.withConfidence<T>, and sema.withConfidence<T>(options). Detection is by @semantscript/core import symbol identity, including aliases/re-exports.
---

author: @codex
created: 2026-09-22 04:05
---
TASK-5.1 started in backlog order after TASK-4. Implementation will use TypeScript symbol identity from the resolved @semantscript/core module, not identifier text.
---

author: @codex
created: 2026-09-22 04:17
---
Implementation complete and full repository gate passes. Discovery now resolves canonical @semantscript/core symbols across named aliases, namespace imports, and re-exports; regression tests reject shadowed identifiers, structurally module-shaped locals, copied members, malformed/optional-chain tags, and non-.sem.ts files. Awaiting final independent blocker audit before acceptance sign-off.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added deterministic, symbol-aware sema site discovery with source locations and comprehensive traversal/identity tests. All three acceptance criteria are verified and the full repository gate passes.
<!-- SECTION:FINAL_SUMMARY:END -->
