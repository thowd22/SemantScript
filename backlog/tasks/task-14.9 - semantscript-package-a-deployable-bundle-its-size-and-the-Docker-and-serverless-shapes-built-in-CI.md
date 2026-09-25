---
id: TASK-14.9
title: >-
  semantscript package: a deployable bundle, its size, and the Docker and
  serverless shapes built in CI
status: To Do
assignee: []
created_date: '2026-09-25 15:21'
labels:
  - dx
  - deploy
milestone: m-5
dependencies:
  - TASK-14.3
parent_task_id: TASK-14
ordinal: 62000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deploying is copying dist/, production node_modules and a 275 to 600 MB artifact directory together by hand; the Express example's Dockerfile has never been built (no Docker on the development machine) and its serverless handler exceeds AWS Lambda's 250 MB unzipped limit with a full-depth float32 artifact. The size levers exist (depth routing brings the encoder to 275 MB, int8 to 145 MB under a recorded tolerance) but nothing tells a developer which applies or produces the bundle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 semantscript package writes a self-contained directory or tarball (compiled output, production dependencies with the platform's native bindings, the current artifact release, a manifest of digests) and prints its size and the size of each part
- [ ] #2 package reports when the bundle exceeds a named target (a Lambda or Cloud Run limit passed as an option) and which lever would fit it: depth routing, int8 with its recorded tolerance, or a smaller encoder
- [ ] #3 CI builds the Express example's Docker image from the package output and runs one request against it; the serverless example is packaged within its platform's limit or the docs state the exact size and why
<!-- AC:END -->
