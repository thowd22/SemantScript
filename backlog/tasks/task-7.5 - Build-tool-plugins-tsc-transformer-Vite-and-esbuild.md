---
id: TASK-7.5
title: 'Build-tool plugins: tsc transformer, Vite and esbuild'
status: To Do
assignee: []
created_date: '2026-09-20 19:41'
labels:
  - compiler
  - dx
milestone: m-3
dependencies:
  - TASK-5.3
parent_task_id: TASK-7
ordinal: 41000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Adoption depends on sema compiling wherever TypeScript already compiles. A developer with an existing Express, Next or Nest app must not adopt a separate build pipeline. The compiler core from Phase 1 gets thin adapters for the common toolchains.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A ts-patch/ttypescript-compatible transformer compiles sema sites in a plain tsc project with one tsconfig entry
- [ ] #2 A Vite plugin and an esbuild plugin compile sema sites with one line of build config each
- [ ] #3 An existing Express app and an existing Next.js app in examples/ each adopt sema by adding the plugin and one expression, with no other changes
- [ ] #4 Source maps point diagnostics and stack traces at the original .ts line
<!-- AC:END -->
