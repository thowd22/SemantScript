---
id: TASK-14
title: 'Phase 5 epic: developer experience story, install, debug, train, deploy'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-25 15:21'
updated_date: '2026-09-26 05:29'
labels:
  - dx
milestone: m-5
dependencies: []
priority: high
ordinal: 53000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
SemantScript must be extremely easy to use for someone who did not build it: install it, get an expression compiled and trained, understand a wrong answer, and ship the artifact. The first outside-the-benchmark runs on 2026-09-25 (the Express example trained through OpenRouter, examples/express-app/README.md) showed where the friction is. The packages are private and unpublished, so nothing works outside this repository; the Python half is installed by hand with machine-specific environment variables and nothing checks it; a training run gives no cost estimate, no spend cap and no progress in dollars, and it fails whole builds on a single bad teacher case or a narrow seed miss; a policy stated completely as constraints still needs a custom script to train without a language model; a wrong answer cannot be explained from the CLI and a handler cannot be unit-tested without a trained artifact; deploying means copying a 275 to 600 MB directory by hand with no packaging, no release management and no CI to prove a fresh clone works. This epic collects the tasks that close those gaps; each child states the measurable behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A developer on a fresh machine with Node and Python installed gets from zero to a running, trained sema expression using only published packages and the CLI, with every step's expected time and cost stated up front, in under ten minutes of their own time on the documented hardware
- [x] #2 Every failure a developer can hit in install, build, train, test, run and deploy names its cause and the next command to run
- [ ] #3 The reference application trains, tests and deploys through documented commands with no repository checkout, and CI proves it on every commit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Coordinate the eleven child tasks (each implemented on its own branch through worktrees and merged into main), then check the epic's criteria against the children's verified evidence; criteria that need the first public publish (14.1) stay open until the user creates the registry credentials and pushes the release tag.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-26: all eleven children implemented on task-14.x branches through parallel worktrees and merged into main (last merge 10487d9); nine are Done. AC2 checked from probes in an empty directory on main: run with a missing artifact, build without a tsconfig, package outside a project and doctor with a missing interpreter each end with the next command (next:/fix: lines), plus 14.11's verified probes (verification gate failures, trainer import and interpreter failures, broken current.json, stale program, offline encoder download; CI run 36220462273) and 14.9's PACKAGE_OVER_TARGET levers. AC1 and AC3 need the first public publish (TASK-14.1, In Progress): the user must add a LICENSE, create the npm org/scope and NPM_TOKEN, a PyPI trusted publisher or PYPI_TOKEN, and push tag v0.1.0; TASK-14.5 AC4 awaits the user's rescope decision. Not closed: the epic stays In Progress until those land.
<!-- SECTION:NOTES:END -->
