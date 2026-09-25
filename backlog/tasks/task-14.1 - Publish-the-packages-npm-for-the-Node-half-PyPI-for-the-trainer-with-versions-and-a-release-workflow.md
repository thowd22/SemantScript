---
id: TASK-14.1
title: >-
  Publish the packages: npm for the Node half, PyPI for the trainer, with
  versions and a release workflow
status: To Do
assignee: []
created_date: '2026-09-25 15:21'
labels:
  - dx
  - install
milestone: m-5
dependencies: []
parent_task_id: TASK-14
ordinal: 54000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
All four Node packages (@semantscript/core, @semantscript/compiler, @semantscript/framework, the semantscript CLI) and the Python project are private at version 0.0.0 and linked from source, so npx semantscript init, npm install @semantscript/core and pip install semantscript cannot work outside this repository; the examples use file: links and node ../../cli/dist/index.js. Nothing about the project is usable by someone who has not cloned it. The CLI spawns the Python trainer by module name and puts repository source paths on PYTHONPATH when it runs inside the repo, which also has to work from an installed package.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 npm install semantscript @semantscript/core @semantscript/compiler @semantscript/framework and pip install semantscript-trainer (or the chosen name) succeed from the public registries in an empty directory
- [ ] #2 npx semantscript init, build, train, test and run work in that directory against the installed packages with no repository checkout
- [ ] #3 Versions are shared across the packages, recorded in the artifact manifest, and a tagged release workflow publishes all of them from CI
- [ ] #4 The getting-started page and the tutorial use the published packages instead of file: links
<!-- AC:END -->
