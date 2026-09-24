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
3. `src/server.ts`: the route calls `triage`, and startup loads the trained
   artifact once with `loadSemaArtifact`. The artifact load is the runtime
   half of adoption; every sema program needs it once.

Build and run:

```sh
npm install                 # links ../../compiler and ../../runtime, installs express and ts-patch
npm run build               # tspc: dist/*.js, dist/*.js.map and dist/semantscript.ir.v1.json
semantscript train          # bundle from dist/, artifact to .semantscript/artifact, teacher from ANTHROPIC_API_KEY or --teacher
npm start                   # POST /tickets {"subject": "...", "body": "..."}
```

`dist/semantscript.ir.v1.json` is the IR bundle the trainer consumes. The
transformer writes it every build, so `semantscript train` always sees the
current sites. Source maps point at `src/triage.sem.ts`; a runtime error inside
the sema call (for example starting without an artifact) shows
`src/triage.sem.ts:7` in the stack trace under `--enable-source-maps`.

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
  repository root (`docker build -f examples/express-app/deploy/Dockerfile .`),
  compiling with `tspc` and keeping only the compiled app, production
  dependencies and the artifact. Docker is not installed on the machine this
  was written on, so the image has been reviewed but not built; the runtime
  stage is the same layout the cold-start numbers below were measured in.
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
