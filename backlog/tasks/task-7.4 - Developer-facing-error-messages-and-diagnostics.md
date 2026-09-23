---
id: TASK-7.4
title: Developer-facing error messages and diagnostics
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-22 04:17'
labels:
  - compiler
  - cli
milestone: m-3
dependencies:
  - TASK-7.1
parent_task_id: TASK-7
ordinal: 27000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Unsupported types, missing inputs, verification failures and low-confidence fallbacks all need clear, located messages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every compile error names file, line and the offending sema site
- [ ] #2 Verification failures list the failing example or constraint
- [ ] #3 Diagnostics are covered by snapshot tests
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @codex
created: 2026-09-22 04:17
---
Discovery audit follow-up: malformed canonical uses such as sema without an explicit type argument or configured forms with zero/multiple option arguments are intentionally omitted by findSemaSites. The diagnostics pipeline must identify these canonical malformed sites separately so they produce located compile errors instead of remaining untransformed runtime tags.
---
<!-- COMMENTS:END -->
