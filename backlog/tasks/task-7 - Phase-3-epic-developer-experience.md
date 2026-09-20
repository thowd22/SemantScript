---
id: TASK-7
title: 'Phase 3 epic: developer experience'
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
updated_date: '2026-09-20 19:41'
labels:
  - epic
milestone: m-3
dependencies:
  - TASK-6
  - TASK-12
ordinal: 23000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the primitive pleasant to use in projects that already exist. The adoption path is: npm install, semantscript init, write one sema expression in an existing file, semantscript dev trains it in the background, the editor shows accuracy. A separate framework is not required for adoption.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A developer can go from a fresh .sem.ts file to a running function using only the semantscript CLI
- [ ] #2 An existing Express or Next.js app adopts sema with one dependency, one init command and one expression, with no other structural changes
<!-- AC:END -->
