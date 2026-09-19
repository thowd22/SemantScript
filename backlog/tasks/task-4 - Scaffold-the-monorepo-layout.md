---
id: TASK-4
title: Scaffold the monorepo layout
status: To Do
assignee: []
created_date: '2026-09-19 18:23'
labels:
  - infra
milestone: m-0
dependencies:
  - TASK-3
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PLAN.md section 4 defines the target layout (compiler, trainer, model, runtime, cli, benchmarks, examples). Nothing exists yet. A scaffold with tooling in place lets Phase 1 tasks start in parallel.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Directories compiler/, trainer/, model/, runtime/, cli/, benchmarks/, examples/ exist with a README stating each one's responsibility
- [ ] #2 Node workspace (package.json, tsconfig) builds an empty compiler and runtime package
- [ ] #3 Python project (pyproject) for trainer and model installs and runs an empty test
- [ ] #4 A single top-level command runs lint and tests for both halves
<!-- AC:END -->
