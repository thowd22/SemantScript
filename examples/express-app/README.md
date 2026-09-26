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

Build and run. The linked packages resolve their own dependencies from the
repository's `node_modules`, so install and build the repository root first:

```sh
npm install && npm run build          # at the repository root, once
cd examples/express-app
npm install                 # links ../../compiler, ../../runtime and ../../framework, installs express and ts-patch
npm run build               # tspc: dist/*.js, dist/*.js.map and dist/semantscript.ir.v1.json
npm test                    # the bundle and both routes over a fixture artifact, no training needed
npm run fixture-artifact    # optional: a fixture artifact in .semantscript/artifact, to run without training
npx semantscript train      # or train: bundle from dist/, artifact to .semantscript/artifact, teacher from ANTHROPIC_API_KEY or --teacher
npm start                   # POST /tickets {"subject": "...", "body": "..."}
```

The root `npm install` links the workspace `semantscript` CLI into the
repository's `node_modules/.bin`, and the root build compiles it, so
`npx semantscript` works from any directory in the clone; nothing is installed
on your `PATH`. Run before the root build, it says to build first.

`npm test` (`test/app.test.mjs`) checks the compiled bundle, serves both
routes through `createApp` over a fixture artifact keyed to this bundle, and
starts `dist/server.js` with `PORT` and `SEMANTSCRIPT_ARTIFACT` set. The
fixture comes from `scripts/fixture-artifact.mjs`, which reuses the runtime's
test ONNX graphs; it answers the third support value of each function
(`"urgent"` for `triage`, `"review"` for `decideRefund`), so it proves the
wiring and not the policy. `npm run fixture-artifact` writes it to
`.semantscript/artifact` for a smoke run or an image build, and refuses to
replace an existing artifact, such as a trained release, unless you run
`npm run fixture-artifact -- --force` (npm keeps a bare `--force` for itself).

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
The same corpus regenerated on 2026-09-25 with the reduced teacher prompt
(TASK-14.5: constraints sent as source text, no duplicated schema) cost
USD 0.99 for 496 requests under `--max-cost-usd 3`, a tenth of the figure
above. At the matched seed (3) the refund expression verified at the same
accuracy (0.987) with ECE 0.011 and pair consistency 0.926, but 5 of 394
records violated a constraint near the tier window, over the 1% tolerance;
seeds 1, 2 and 4 landed at 1.0%, 1.5% and 2.0%, and seed 5 published
(release `217d386c…`, refund 0.924, ECE 0.072, triage 1.000). So the reduced
prompt buys the same accuracy at a tenth of the cost, and the near-threshold
gate stays a seed lottery on this corpus size either way; the release the
table above describes remains the stronger one.

`train` now draws that lottery itself
([seed retry](../../docs/cli-reference.md#seed-retry)): a failure only on
the violation rate or the ECE, within `--seed-retry-margin` times the gate
(default 2), retrains on the cached datasets with the next seed, up to
`--seed-attempts` runs (default 3). On the reduced-prompt datasets
(2026-09-25, RX 9070 XT, the command above with `--seed 2 --seed-attempts 5
--seed-retry-margin 3`), seed 2 failed at 6 of 394 (1.52%), seed 3 at 5 of
394 (1.27%), and seed 4 passed at 2 of 394 and published; the report listed
all three attempts, both verified IRs and the release manifest recorded seed
4 (`semantscript test` prints it in its `seed` column), the teacher sent no
request (USD 0), and the rebuild with the same flags reused that release. The
first seed to pass publishes: seed 4 passed at accuracy 0.9367, where the
failed seeds 2 and 3 had reached 0.9750 and 0.9873, and the attempts table
shows that tradeoff.
With `--seed-retry-margin 1.2`, seed 2's 1.52% was outside the margin and
the build stopped at once, saying so. GPU training is not bit-for-bit
repeatable, so a seed's violation count can move by a few records between
runs (seed 4 had failed at 8 of 394 the day before).

Generation cost about USD 10 through OpenRouter for the two expressions
(about 600 requests at roughly 8,000 prompt tokens each, plus three attempts
for each of five counterfactual anchors the teacher could not twin); the
seed reruns cost nothing because the datasets were cached.

That release was generated with the teacher prompt of that day. The prompt
has since been made compact and is served through prompt caching
([teachers](../../docs/teachers.md#prompt-size-and-caching)): the
decideRefund case request went from 6,552 input tokens and USD 0.0138 to
USD 0.0071 for the first request and USD 0.0017 once cached. Because the
teacher's configuration digest includes the prompt layout, the next
`semantscript train` with this teacher regenerates both datasets.
Before that paid run, `semantscript train --estimate` with the recipe above
prints:

```text
teacher: anthropic anthropic/claude-sonnet-5; price: OpenRouter price list (2026-09-25) (USD 2 in / 10 out per million tokens)
function      source              requests        input tokens  cached     output tokens  USD              time
------------  ------------------  --------------  ------------  ---------  -------------  ---------------  ------
nf_957c2b2b…  src/refunds.sem.ts  407 (max 1344)  997,984       877,912    38,834         0.81 (max 2.67)  27 min
nf_bcbf93e1…  src/triage.sem.ts   189 (max 189)   253,827       218,080    9,450          0.21 (max 0.21)  13 min
total                             596 (max 1533)  1,251,811     1,095,992  48,284         1.02 (max 2.88)  40 min
```

About USD 1 against the USD 10 the release above cost. The verified accuracy
of a release trained on regenerated datasets has not been measured yet; pass
`--max-cost-usd 3` to bound that run.

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
`dist/` is the whole deployment story. `semantscript package` puts the three
parts together in one directory and says how big it is:

```sh
npm run build                     # and train, so .semantscript/artifact holds a release
npx semantscript package --include deploy/lambda.mjs --target lambda-zip
```

It writes `.semantscript/package/` with `dist/` (without the IR bundle, which
holds the prompt text), the production `node_modules` (the `file:` links to the
repository's `@semantscript/*` packages installed as real copies with
`npm install --omit=dev --install-links`; an app on published packages with a
lockfile gets `npm ci --omit=dev`), `.semantscript/artifact` with the current
release only, `deploy/lambda.mjs`, and `semantscript-package.json`, the
sha256 and size of every file. `onnxruntime-node` and `tokenizers` ship
binaries for every platform; the bundle keeps only the target's (the host by
default, or `--platform linux --arch arm64`), and drops ONNX Runtime's CUDA
and TensorRT providers, since the runtime runs the CPU provider. The command
prints the size of each part and checks the total against the target; the
[CLI reference](../../docs/cli-reference.md#package) lists the
targets and the [deploy guide](../../docs/deploy.md) the size levers.

- [`deploy/Dockerfile.package`](deploy/Dockerfile.package): a single-stage
  image whose build context is the bundle
  (`docker build -f deploy/Dockerfile.package -t ticket-api .semantscript/package`);
  it copies the directory, runs as the `node` user and starts
  `dist/server.js`. The bundle carries native bindings for one platform and
  arch, so package for the image's: on macOS or another non-Linux host pass
  `--platform linux --arch arm64` (Apple silicon, whose Docker builds
  linux/arm64) or `--arch x64`, or the container fails when it loads the
  runtime. CI packages this example with the fixture artifact on
  every push, builds this image from the bundle, sends one `POST /tickets`
  and invokes the packaged Lambda handler once; that image is 314 MB and
  builds in 6 s from the bundle
  ([measured](../../docs/CONTRIBUTING.md#continuous-integration)).
- [`deploy/Dockerfile`](deploy/Dockerfile): the older two-stage image built
  from the repository root (`docker build -f examples/express-app/deploy/Dockerfile .`),
  which installs and compiles inside the build stage and keeps the same three
  parts. [`deploy/Dockerfile.dockerignore`](deploy/Dockerfile.dockerignore)
  limits its context to those sources. CI builds it from a fresh clone on
  every push with the fixture artifact and smoke-runs both routes; the image
  is 624 MB and builds in about 36 s without a layer cache.
- [`deploy/lambda.mjs`](deploy/lambda.mjs): a serverless handler (API Gateway
  HTTP API event shape) over the same compiled `triage` function, shipped in
  the bundle with `--include deploy/lambda.mjs` (its handler is
  `deploy/lambda.handler`). The artifact loads once per execution environment
  on the first invocation and stays loaded while the environment is warm.
- [`scripts/cold-start.mjs`](scripts/cold-start.mjs): measures a cold start
  of either shape: a fresh Node process, the artifact load and the first
  request, and the process's resident set after that request.

Bundle sizes, measured 2026-09-25 on linux/x64 (Node 22.22):

| Artifact                                         | node_modules | Artifact  | Total     | lambda-zip (250 MiB) |
| ------------------------------------------------ | ------------ | --------- | --------- | -------------------- |
| Fixture (`npm run fixture-artifact`)             | 82.3 MiB     | 12.1 KiB  | 82.6 MiB  | fits                 |
| Trained release `217d386c…` (22 layers, float32) | 82.3 MiB     | 570.9 MiB | 653.6 MiB | over by 403.6 MiB    |

The trained release does not fit a Lambda .zip package: its encoder alone is
596,679,464 bytes (569.0 MiB), a full-depth float32 ModernBERT-base, and AWS
counts 262,144,000 bytes unzipped for the function and its layers together.
`package --target lambda-zip` exits 1 with `PACKAGE_OVER_TARGET` and projects
each lever from the measured encoder sizes: depth routing alone does not fit
(about 462, 347 and 309 MiB at 12, 6 and 4 layers, because the 82 MiB of
dependencies stay), depth 6 or 4 with int8 would (about 151 and 141 MiB), int8
alone would (about 228 MiB), and so would an encoder whose graph is at most
165.5 MiB. The int8 figures are measurements, not a step you can run: no int8
derivation exists for applications yet (it has only been measured on the
refund benchmark, where it changed 1 of 80 attested cases and the strict gate
refused it). The levers you can act on today are a smaller encoder
(`train --encoder-name`) and depth routing, which alone is not enough here.
Until one of those fits, ship the trained release as a container:
[`deploy/Dockerfile.package`](deploy/Dockerfile.package) runs the bundle as
the Express server on Cloud Run or any container host, which has no size
limit near 654 MiB. The bundle is also well under Lambda's container image
limit (`--target lambda-image`, 10 GiB), but this example has no Lambda
container recipe: that image would need a Lambda Node.js base image with
`deploy/lambda.handler` as its command, not the Express server image.

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
float32 encoder; the int8 release under `results-int8-2026-09-24` measured how
far quantization would cut it, but that is a measurement on the refund
benchmark, not a step an application can run yet.
