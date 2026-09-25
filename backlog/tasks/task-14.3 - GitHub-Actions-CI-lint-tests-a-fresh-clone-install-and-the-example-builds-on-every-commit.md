---
id: TASK-14.3
title: >-
  GitHub Actions CI: lint, tests, a fresh-clone install and the example builds
  on every commit
status: To Do
assignee: []
created_date: '2026-09-25 15:21'
labels:
  - dx
  - ci
milestone: m-5
dependencies: []
parent_task_id: TASK-14
ordinal: 56000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The repository now lives at github.com/thowd22/SemantScript and has no CI: the lint and test gates run only on the development machine, the Docker example has never been built (TASK-7.9 AC2), the five-minute adoption flow was never timed on a clean machine (TASK-7.6 AC4), and the getting-started commands are verified only by hand. A fresh-clone job is also the only honest proof that the install story works for someone else.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A workflow runs on every push and pull request: Node lint and every Node test suite, Python lint and the CPU-only Python tests, and the docs examples compile
- [ ] #2 A job installs from a fresh checkout on ubuntu-latest exactly as the docs say, builds examples/express-app and examples/refund-service, runs their tests with the fixture artifact, and builds the Express Docker image
- [ ] #3 The workflow's wall time and the fresh-install job's time are recorded in the docs as the measured adoption cost
<!-- AC:END -->
