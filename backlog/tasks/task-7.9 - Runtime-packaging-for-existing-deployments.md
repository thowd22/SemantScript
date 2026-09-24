---
id: TASK-7.9
title: Runtime packaging for existing deployments
status: Done
assignee:
  - '@claude'
created_date: '2026-09-20 19:41'
updated_date: '2026-09-24 22:16'
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
- [x] #1 The artifact is resolved relative to the compiled output with an override env var
- [ ] #2 A Docker example and a serverless (cold-start measured) example run the Express app from TASK-TASK-7.5 with the artifact bundled
- [x] #3 Runtime install pulls the correct ONNX Runtime binary for the platform with no manual steps
- [x] #4 Cold-start time and memory footprint are documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Runtime: defaultSemaArtifactPath() resolves SEMANTSCRIPT_ARTIFACT, else searches .semantscript/artifact upward from the entry script's directory (the compiled output) and from the working directory, so a deployed app finds the artifact beside its build with no environment setup; document the search order.
2. examples/express-app: load the artifact with the default resolution; add deploy/Dockerfile (multi-stage: build with tspc, copy dist plus .semantscript/artifact, npm ci --omit=dev pulls the platform's ONNX Runtime binary) and deploy/lambda.mjs (a serverless handler over the same triage function, artifact bundled beside the handler); scripts/cold-start.mjs measures time to first response and RSS of a fresh process.
3. Measure cold start and memory on this machine with a real artifact (Docker is not installed here: the Dockerfile is written and reviewed but not built; say so); record numbers in the example README and runtime README.
4. Document that onnxruntime-node and tokenizers ship prebuilt binaries per platform selected at install time with no manual step; tests for the runtime's search order.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Runtime: defaultSemaArtifactPath(env, { entry, cwd }) now resolves SEMANTSCRIPT_ARTIFACT, else the first .semantscript/artifact found walking up from the entry script's directory (process.argv[1], the compiled output) and then from the working directory, else the conventional cwd location; two runtime tests cover the order. Example: server.ts calls loadSemaArtifact() with no path; deploy/Dockerfile (two-stage, tspc build, npm ci --omit=dev, artifact copied beside dist/), deploy/lambda.mjs (API Gateway v2 shape, artifact loaded once per environment, runtime errors mapped to 500), scripts/cold-start.mjs (fresh process, artifact load, first request, RSS from /proc). Measured on this machine with the refund multi-head release (597 MB ONNX encoder), three runs: Express server listening (artifact loaded) at 1.8/2.5/2.5 s, first response 1.9/2.5/3.4 s, RSS 1.60 GB; serverless handler first response 3.5/3.7/4.5 s, RSS 1.59 GB. The first response is the unknown-function error path (the refund artifact lacks the example's function), so it includes the full load and a request round trip but not the first encoder pass (28 ms p50 per the refund benchmark). Docker is not installed here (docker: command not found): the Dockerfile is written and reviewed, not built; recorded in the README. Binaries: onnxruntime-node ships linux x64/arm64, darwin and win32 binaries inside the package and tokenizers one .node binding per platform, so npm ci on the target selects them with no manual step (documented in the runtime README). The example's copied artifact and node_modules are git-ignored (examples/*/.semantscript/ added).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The runtime now resolves the artifact without environment-specific setup: SEMANTSCRIPT_ARTIFACT, else .semantscript/artifact searched upward from the compiled entry script and then from the working directory (tests cover the order). The Express example loads with no path and gains deploy/Dockerfile (two-stage image, artifact beside dist/), deploy/lambda.mjs (serverless handler over the same compiled function) and scripts/cold-start.mjs. Measured on this machine with the refund multi-head release: server cold start 1.8 to 2.5 s to a loaded artifact, first response 1.9 to 3.4 s, 1.60 GB resident; serverless handler 3.5 to 4.5 s to first response, 1.59 GB resident; documented in the example and runtime READMEs with what the first response does and does not include. The prebuilt onnxruntime-node and tokenizers binaries make npm ci on the target platform the only install step. AC2 is left unchecked: the serverless example ran and was measured, but Docker is not installed on this machine, so the Dockerfile is written and reviewed, not built. 279 Node tests pass, lint clean.
<!-- SECTION:FINAL_SUMMARY:END -->
