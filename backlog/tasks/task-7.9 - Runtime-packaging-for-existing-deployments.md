---
id: TASK-7.9
title: Runtime packaging for existing deployments
status: To Do
assignee: []
created_date: '2026-09-20 19:41'
labels:
  - runtime
  - dx
milestone: m-3
dependencies:
  - TASK-5.9
parent_task_id: TASK-7
ordinal: 45000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dropping sema into an existing app means the artifact must ship with it wherever it already deploys: a Node server, a Docker image, a serverless function, or a Next.js server bundle. The runtime must load from a path relative to the built app with no environment-specific setup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The artifact is resolved relative to the compiled output with an override env var
- [ ] #2 A Docker example and a serverless (cold-start measured) example run the Express app from TASK-TASK-7.5 with the artifact bundled
- [ ] #3 Runtime install pulls the correct ONNX Runtime binary for the platform with no manual steps
- [ ] #4 Cold-start time and memory footprint are documented
<!-- AC:END -->
