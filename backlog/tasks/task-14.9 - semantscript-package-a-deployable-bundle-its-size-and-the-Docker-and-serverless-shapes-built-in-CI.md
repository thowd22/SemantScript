---
id: TASK-14.9
title: >-
  semantscript package: a deployable bundle, its size, and the Docker and
  serverless shapes built in CI
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-25 15:21'
updated_date: '2026-09-26 03:11'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. New cli/src/package.ts: `semantscript package [--project <dir>] [--dist <dir>] [--artifact <root>] [--out <dir>] [--include <path>]... [--target <name>|--max-bytes <n>] [--platform <os>] [--arch <cpu>] [--tarball] [--force] [--json]`, registered in cli/src/index.ts (COMMANDS, USAGE, exports). Typed PackageError (codes such as PACKAGE_NO_DIST, PACKAGE_NO_RELEASE, PACKAGE_OUT_EXISTS, PACKAGE_INSTALL_FAILED, PACKAGE_OVER_TARGET) following ReleaseError.
2. Bundle layout in <out> (default .semantscript/package): package.json, dist/ (compiled output without semantscript.ir.v1.json, which holds prompt text), node_modules/ (production tree), .semantscript/artifact/{current.json, releases/sha256-<current>} only, included files (e.g. deploy/lambda.mjs) at their relative paths, semantscript-package.json manifest (kind semantscript.package, version 1: release digest, platform/arch, node version, per-part bytes, sha256 + size of every file). The runtime finds the artifact by its upward search from dist/, so no env var is needed.
3. Current release: readPointer + scanReleases/resolveRelease from manifest.ts/releases.ts; export verifyRelease, fileSha256, treeSize, formatBytes from releases.ts (or move to a shared module) so the packaged release gets the same integrity, verification and checkSemaArtifact check as promote; copy only that release plus pointerBytes(current).
4. Production dependencies: stage package.json (and package-lock.json when present and no file: specs -> npm ci --omit=dev) in the bundle; file: specs are rewritten to absolute paths and installed with `npm install --omit=dev --install-links` (as the Dockerfile does), with ONNXRUNTIME_NODE_INSTALL=skip (runtime uses the CPU provider only). Then prune native bindings to the target platform/arch (default host): onnxruntime-node bin/napi-v*/<other os|arch> and the CUDA/TensorRT provider libraries; tokenizers *.node for other triples. Measured locally: onnxruntime-node 548 MB -> about 46 MB linux x64 CPU; tokenizers 64 MB -> 5 MB.
5. Size report: table of parts (dist, node_modules with its largest packages, artifact split by resource role encoder/adapter/heads/tokenizer, included files, manifest) and the total apparent bytes (what Lambda counts unzipped); --json emits the same.
6. Targets (AC2): named table with dated source notes, e.g. lambda-zip (250 MB unzipped incl. layers = 262,144,000 bytes), lambda-image (10 GB), cloud-run / cloud-run-functions limits (verify exact figures from provider docs during implementation), plus --max-bytes. When over: exit 1 (bundle still written) with PACKAGE_OVER_TARGET and lever lines computed from the manifest: depth routing (encoder is full depth: estimate from the measured ModernBERT-base depth sweep 4/6/12 layers = 235/275/396 of 597 MB, name `build --domain-depth <domain>=<n>`), int8 dynamic (encoder x 150.1/596.7, only under a recorded tolerance: attested 2, argmax 0.01, results-int8-2026-09-24; the strict gate refused it), both combined, or a smaller encoder (`train --encoder-name`), each with the resulting bundle total and whether it fits; already-routed or int8 encoders are recognised from encoderDepth / onnx.precision so a used lever is not suggested again.
7. Tests cli/test/package.test.mjs over the runtime fixture artifact and a tiny fake project (no network: a project with no dependencies or a file: dep on a local stub; pruning tested on a synthetic node_modules layout): bundle layout, only the current release copied, manifest digests match files, IR bundle excluded, --include, size table parts, over-target report and lever lines against a synthetic large manifest, usage errors, refusing a non-empty --out without --force, invalid pointer/unverified release refused.
8. Express example: deploy/Dockerfile.package (single stage FROM node:22-bookworm-slim, COPY the bundle, USER node, CMD node dist/server.js) with its .dockerignore; README deploy section rewritten around `npx semantscript package`, measured sizes; lambda.mjs shipped via --include deploy/lambda.mjs. Keep the existing Dockerfile and fresh-install job intact.
9. CI: new `package` job beside the others: npm ci, build, example install/build, fixture artifact, `semantscript package --include deploy/lambda.mjs --target lambda-zip` (must fit), invoke the packaged lambda handler once with node, then `docker build` from the bundle with Dockerfile.package, run it and POST /tickets once, print image size.
10. Local measurement (no teacher spend, USD 0): package the Express app against the trained release copied from /home/admin2/SemantScript/examples/express-app/.semantscript/artifact (217d386c, 597 MB full-depth float32 encoder) and record the exact bundle size, the over-target report for lambda-zip and the lever lines in the example README (the serverless shape does not fit with that release; state why and which lever fits). Fixture-artifact bundle size recorded too.
11. Docs: docs/cli-reference.md (`package` section, flags, error codes, synopsis), cli/README.md (package section), docs/diagnostics.md code list if it enumerates CLI codes, docs/CONTRIBUTING.md CI table (package job + measured wall time), examples/express-app/README.md, docs/getting-started or tutorial deploy step if they mention deployment; npx prettier --check docs README.md.
12. Verify: npm run build, npm run lint:node, npm test -w cli, prettier; commit on task-14.9, push, gh run watch until green; then finalization guide and check ACs with evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
IMPLEMENT: cli/src/package.ts adds semantscript package (bundle layout dist without IR bundle, production node_modules via npm install --omit=dev --install-links or npm ci, current release only checked with the exported verifyRelease, --include, semantscript-package.json with sha256+size per file; onnxruntime-node/tokenizers pruned to the target platform, CUDA/TensorRT providers dropped, pruned bindings loaded in a child node on the host). Targets lambda-zip 262,144,000 B, lambda-image 10 GiB, cloud-run-functions 500,000,000 B (provider quota pages read 2026-09-25) or --max-bytes; over target exits 1 PACKAGE_OVER_TARGET with depth/int8/depth+int8/smaller-encoder levers from exact measured bytes (depth sweep results.json, int8 quantization-report.json). Local measurements: fixture bundle 86,641,399 B (82.6 MiB) fits lambda-zip; trained release 217d386c bundle 685,309,329 B (653.6 MiB) over by 403.6 MiB, int8 ~227.6 MiB and depth6/4+int8 ~150.6/140.9 MiB fit, depth alone does not. Packaged server and lambda handler answered urgent locally. Tests cli/test/package.test.mjs (8), docs: docs/deploy.md (new, indexed), cli-reference package section, diagnostics, cli/README, CONTRIBUTING CI row, express README deploy section, Dockerfile.package, CI package job.
<!-- SECTION:NOTES:END -->
