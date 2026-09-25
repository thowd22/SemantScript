---
id: TASK-14
title: 'Phase 5 epic: developer experience story, install, debug, train, deploy'
status: To Do
assignee: []
created_date: '2026-09-25 15:21'
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
- [ ] #2 Every failure a developer can hit in install, build, train, test, run and deploy names its cause and the next command to run
- [ ] #3 The reference application trains, tests and deploys through documented commands with no repository checkout, and CI proves it on every commit
<!-- AC:END -->
