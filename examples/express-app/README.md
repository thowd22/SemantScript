# Express app adopting sema through the tsc transformer

A ticket API on Express 5 compiled by plain `tsc`. SemantScript was adopted by
adding one tsconfig entry and one `.sem.ts` file; the rest of the app is an
ordinary TypeScript project.

What adoption touched:

1. `tsconfig.json`: the `plugins` entry
   `{ "transform": "@semantscript/compiler/transformer", "application": "ticket-triage" }`.
   The build runs through ts-patch's `tspc` instead of `tsc` (that is ts-patch's
   one-line change: `"build": "tspc -p tsconfig.json"`).
2. `src/triage.sem.ts`: the one `sema` expression, `triage(subject, body)`.
3. `src/app.ts` and `src/server.ts`: the `POST /tickets` route in
   `createApp` calls `triage`, and `server.ts` loads the trained artifact once
   with `loadSemaArtifact` before listening on `PORT` (default 3000). The
   artifact load is the runtime half of adoption; every sema program needs it
   once.

Build and run:

```sh
npm install                 # links ../../compiler, ../../runtime and ../../framework, installs express and ts-patch
npm run build               # tspc: dist/*.js, dist/*.js.map and dist/semantscript.ir.v1.json
npm test                    # the bundle and both routes over a fixture artifact, no training needed
semantscript train          # bundle from dist/, artifact to .semantscript/artifact, teacher from ANTHROPIC_API_KEY or --teacher
npm start                   # POST /tickets {"subject": "...", "body": "..."}
```

Run the root `npm install` and `npm run build` first: the linked packages
resolve their own dependencies from the repository's `node_modules`.

`npm test` (`test/app.test.mjs`) checks the compiled bundle, serves both
routes through `createApp` over a fixture artifact keyed to this bundle, and
starts `dist/server.js` with `PORT` and `SEMANTSCRIPT_ARTIFACT` set. The
fixture comes from `scripts/fixture-artifact.mjs`, which reuses the runtime's
test ONNX graphs; it answers the third support value of each function
(`"urgent"` for `triage`, `"review"` for `decideRefund`), so it proves the
wiring and not the policy. `npm run fixture-artifact` writes it to
`.semantscript/artifact` for a smoke run or an image build, and refuses to
replace an existing artifact, such as a trained release, without `--force`.

## A trained release, through OpenRouter

Both expressions carry three gold examples, and `decideRefund` carries the
policy's six rules as constraints (`src/refunds.sem.ts`), which is what makes
a numeric policy learnable from a few hundred teacher cases: the trainer
labels boundary pairs and counterfactual twins against them and the release
gate refuses a model that breaks one. Trained 2026-09-25 with Sonnet 5 through
OpenRouter's Anthropic-format route (`.semantscript/teacher.toml`, git-ignored:
`backend = "anthropic"`, `model = "anthropic/claude-sonnet-5"`,
`base_url = "https://openrouter.ai/api"`, `mode = "direct"`, with
`ANTHROPIC_API_KEY` set to the OpenRouter key for the process):

```sh
semantscript train --teacher .semantscript/teacher.toml --cases 192 --epochs 8 --seed 3 \
  --select-best-epoch --counterfactual-ratio 0.5 --max-constraint-violation-rate 0.01 --device cuda
```

| Expression     | Rows (synthetic + adversarial + gold) | Best epoch | Verified accuracy | ECE   | Pair consistency | Constraint violations |
| -------------- | ------------------------------------- | ---------- | ----------------- | ----- | ---------------- | --------------------- |
| `decideRefund` | 192 + 202                             | 6          | 0.987             | 0.008 | 0.979            | 0 of 394              |
| `triage`       | 192                                   | 6          | 1.000             | 0.000 | 1.000            | 0 of 192              |

Release `5c755d08…`, published under decision-8's recorded 1% violation
tolerance and in fact recording zero raw violations; `semantscript test
--bundle dist/semantscript.ir.v1.json` passes and `semantscript run
dist/refunds.sem.js --call decideRefund --input '[…]'` answers `approve`,
`deny`, `review` on the seeded cases and `deny` on a standard-tier order at 45
days. Two earlier seeds (1 and 2) trained from the same cached datasets failed
the gate on standard-tier orders at 45 days (2.0% and 1.3% violations), so the
seed is part of the record, as it was for the refund benchmark's release.
Generation cost about USD 10 through OpenRouter for the two expressions
(about 600 requests at roughly 8,000 prompt tokens each, plus three attempts
for each of five counterfactual anchors the teacher could not twin); the
seed reruns cost nothing because the datasets were cached.

`dist/semantscript.ir.v1.json` is the IR bundle the trainer consumes. The
transformer writes it every build, so `semantscript train` always sees the
current sites. Source maps point at `src/triage.sem.ts`; a runtime error inside
the sema call (for example starting without an artifact) shows
`src/triage.sem.ts:7` in the stack trace under `--enable-source-maps`.

## A transaction gated by a decision

`src/refunds.ts` mounts `POST /refunds/:orderId` through the framework's
decorated controller: it reads the customer and order rows from Postgres
(PGlite in process, Postgres 17 in WebAssembly, seeded at startup; a `pg`
Pool works the same), evaluates `decideRefund` in `src/refunds.sem.ts` over
those plain values, inserts the refund row and commits when the decision is
`approve`, and rolls back otherwise so nothing is written. The sema expression
receives only the two records; it has no handle on the database.

## Deploying with the artifact

The app calls `loadSemaArtifact()` with no path. The runtime then uses
`SEMANTSCRIPT_ARTIFACT` when set, and otherwise searches for
`.semantscript/artifact` upward from the compiled entry (`dist/server.js`)
and from the working directory, so shipping the artifact directory beside
`dist/` is the whole deployment story. Three things travel together: `dist/`,
`node_modules/` (installed with `npm ci --omit=dev` on the target platform;
`onnxruntime-node` and `tokenizers` carry prebuilt binaries for linux x64 and
arm64, macOS and Windows, nothing to build or download by hand) and
`.semantscript/artifact/`.

- [`deploy/Dockerfile`](deploy/Dockerfile): a two-stage image built from the
  repository root (`docker build -f examples/express-app/deploy/Dockerfile .`).
  The build stage installs the workspace, builds the compiler, runtime and
  framework, compiles the app with `tspc`, and installs the production tree
  with `npm install --omit=dev --install-links`, which packs the `file:`
  packages into real copies with their own dependencies. The runtime stage
  keeps only `dist/` (without the IR bundle, which holds the prompt text),
  that `node_modules` and `.semantscript/artifact`, and runs as the `node`
  user. [`deploy/Dockerfile.dockerignore`](deploy/Dockerfile.dockerignore)
  limits the context to those sources, so local installs never enter it. CI
  builds the image from a fresh clone on every push with the fixture artifact
  and smoke-runs both routes; the image is 624 MB and builds in about 33 s
  without a layer cache
  ([measured](../../docs/CONTRIBUTING.md#continuous-integration)). To ship
  a trained model, train first so `.semantscript/artifact` holds the release,
  then build.
- [`deploy/lambda.mjs`](deploy/lambda.mjs): a serverless handler (API Gateway
  HTTP API event shape) over the same compiled `triage` function. The
  artifact loads once per execution environment on the first invocation and
  stays loaded while the environment is warm.
- [`scripts/cold-start.mjs`](scripts/cold-start.mjs): measures a cold start
  of either shape: a fresh Node process, the artifact load and the first
  request, and the process's resident set after that request.

Cold start and memory, measured 2026-09-24 on the development machine (AMD
Ryzen 9 9900X, 16 GB, WSL2, Node 22.22, CPU inference through ONNX Runtime)
with the refund benchmark's multi-head release
(`benchmarks/refund/data/release-multihead-2026-09-24`, a ModernBERT-base
encoder, 597 MB of ONNX), three runs each:

| Shape              | Artifact loaded (listening) | First response      | Resident set |
| ------------------ | --------------------------- | ------------------- | ------------ |
| Express server     | 1.8 s, 2.5 s, 2.5 s         | 1.9 s, 2.5 s, 3.4 s | 1.60 GB      |
| Serverless handler | (inside first response)     | 3.5 s, 3.7 s, 4.5 s | 1.59 GB      |

The first response in these runs is the runtime's unknown-function error,
because the refund artifact does not carry this example's `triage` function:
it measures process start, module import, the full artifact load (manifest
verification, resource digests, ONNX session creation in the inference
worker) and one request round trip through the loaded runtime, but not the
first encoder pass, which the refund benchmark measures at 28 ms p50 on this
CPU (`results-compact-2026-09-24`). The resident set is dominated by the
float32 encoder; the int8 release under `results-int8-2026-09-24` is the lever
when memory matters more than the last points of accuracy.
