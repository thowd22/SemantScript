---
id: TASK-14.3
title: >-
  GitHub Actions CI: lint, tests, a fresh-clone install and the example builds
  on every commit
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-25 15:21'
updated_date: '2026-09-25 15:48'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add .github/workflows/ci.yml (on push, pull_request, workflow_dispatch; concurrency cancel-in-progress). Jobs on ubuntu-latest: (a) node: setup-node from .nvmrc with npm cache, npm ci, npm run lint:node, npm run build, npm run test:node (includes compiler/test/docs-examples.test.mjs, so docs examples compile; cli and benchmark tests spawn python3, available on the runner), npx prettier --check docs README.md; (b) python: setup-python 3.12, venv + pip install -e '.[dev]' (no training extra, so torch/onnx tests skip), npm run lint:python, then npm ci + npm run build (some Python tests use runtime/dist, cli/dist) and npm run test:python with the venv active; (c) fresh-install: follow CONTRIBUTING/DEVELOPING/example READMEs literally: npm install, npm run build, then in examples/express-app and examples/refund-service npm install, npm run build, npm test; generate the fixture artifact into examples/express-app/.semantscript/artifact and docker build -f examples/express-app/deploy/Dockerfile .; time each step (date +%s / job duration) and print a summary to GITHUB_STEP_SUMMARY.
2. Express example has no tests: add examples/express-app/test/app.test.mjs (node --test) modeled on examples/refund-service/test/app.test.mjs: bundle shape check, then createFixtureArtifact (runtime/test/fixtures/artifact.mjs) re-keyed to triage and decideRefund, POST /tickets and POST /refunds/:id over the fixture. server.ts listens on a hard-coded 3000 at import, so either export createApp from a module (like refund-service's app.ts) or honor PORT env; add "test" script to package.json. Update examples/express-app/README.md.
3. Add examples/express-app/scripts/fixture-artifact.mjs (writes the fixture artifact re-keyed to this app's bundle into .semantscript/artifact) so the Docker job has an artifact without training; document it.
4. Fix deploy/Dockerfile, which cannot build from a clean checkout as written: it does not COPY framework (express-app imports @semantscript/framework); compiler/runtime/framework dist/ are git-ignored so it must build them (npm ci + tsc -b at root inside the build stage, or copy root package.json/package-lock.json/tsconfig*.json); example package-lock.json is git-ignored so npm ci fails (use npm install or commit lock); runtime stage runs npm ci in /app where file:../../runtime resolves to /runtime not /repo/runtime; linked runtime's own deps (onnxruntime-node, tokenizers) are hoisted to the repo root node_modules, which the image lacks. Restructure so the linked packages resolve (e.g. keep /repo layout in the runtime stage, or npm pack the workspaces into tarballs and install those). Add a CI smoke run of the image (docker run, POST /tickets with the fixture artifact) if cheap. Update the README text that says the image was never built.
5. Push to main, watch with gh run watch / gh run view --log-failed, iterate until green; if the torch-free python job reveals tests importing torch at module level, guard them with importorskip. Keep jobs well under 15 min (no training extra).
6. Record measured wall times (whole workflow and fresh-install job, from gh run view --json jobs timings) in docs/CONTRIBUTING.md under a new 'Continuous integration' section (what each job runs, how to reproduce locally, measured adoption cost), mention in docs/getting-started.md if appropriate, add CI badge to README.md top, keep docs/index.md links and prettier passing. Update TASK-7.9 AC2 / TASK-7.6 AC4 evidence only via notes in the final report (not editing other tasks).
7. Run npm run check locally, prettier check, commit with trailers, push, confirm final run green, then finalize per task-finalization guide.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added .github/workflows/ci.yml (node, python, fresh-install jobs). Express example split into src/app.ts (createApp) and src/server.ts (PORT env), added test/app.test.mjs (3 tests, pass locally) and scripts/fixture-artifact.mjs. Dockerfile rebuilt: builds workspaces in-image, installs production tree with --install-links (verified locally: standalone layout serves /tickets and /refunds with the fixture artifact), Dockerfile.dockerignore allowlist. Python: test_training.py parametrize no longer touches torch at collection; two torch-only tests gained @requires_torch (dev-only deps: 412 passed, 45 skipped locally).

CI runs: 36155452753 (node failed: type-aware lint ran before build; fixed by building first in ci.yml and in scripts/check.mjs check mode), 36155743686 (all green: workflow 1m39s, node 1m32s, python 1m32s with 412 passed/45 skipped, fresh-install 1m36s incl. docker build 33s, image 624MB, smoke POST /tickets urgent and POST /refunds/o1 committed:false), 36156428896 on aefa146 (all green: workflow 1m50s, node 1m36s, python 1m13s, fresh-install 1m42s). Docs: docs/CONTRIBUTING.md Continuous integration section with measured adoption cost; README badge; docs/index.md, examples/README.md and express-app README updated. Local npm run check exit 0 (529 passed, 4 skipped with the training extra); prettier --check docs README.md clean.
<!-- SECTION:NOTES:END -->
