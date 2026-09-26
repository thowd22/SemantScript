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
                            # (calls a paid teacher; the recipe that passed the held-out check is below)
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
gate refuses a model that breaks one, on its corpus and on a held-out sample.
The release on disk passes both parts of that gate. Its teacher is mixed
([teachers](../../docs/teachers.md)): the constraints label every refund input
they decide, and Sonnet 5 through OpenRouter's Anthropic-format route labels
the rest as the `[teacher.fallback]`. The file is
`.semantscript/teacher-constraints.toml` (git-ignored, like the rest of
`.semantscript`), with `ANTHROPIC_API_KEY` set to the OpenRouter key for the
process:

```toml
[teacher]
backend = "constraints"
seed = 1

[teacher.ranges]
"order.ageDays" = { low = 0, high = 240 }

[teacher.fallback]
backend = "anthropic"
model = "anthropic/claude-sonnet-5"
base_url = "https://openrouter.ai/api"
mode = "direct"
max_tokens = 4096
```

The `ageDays` range widens the constraints teacher's inferred 0 to 180 days
(twice the largest threshold) so training also covers orders up to 240 days
old. The older `.semantscript/teacher.toml` is the fallback table on its own,
the language-model teacher that labelled the earlier releases.

### The release on disk

The release this example serves is `0fd67142…`, published
2026-09-26T17:52Z on the development machine (RX 9070 XT). `.semantscript` is
git-ignored (`examples/*/.semantscript/`), so a clone has no trained release:
the figures below describe the development machine's, not what a clone's own
`train` will publish.

```sh
semantscript train --teacher .semantscript/teacher-constraints.toml --cases 384 --epochs 16 \
  --select-best-epoch --counterfactual-ratio 0.5 --max-constraint-violation-rate 0.01 \
  --seed 1 --seed-attempts 3 --device cuda --max-cost-usd 12
```

`train --estimate` priced it beforehand at 517 requests (max 828) and USD 0.70
(max 1.34). The run sent 478 requests (738,080 input tokens, 653,526 of them
cached, and 66,155 output tokens) for USD 0.96 in about 30 minutes, then
passed on its first seed:

| Expression     | Rows (synthetic + adversarial + gold) | Verified accuracy | ECE    | Pair consistency | Corpus violations | Held-out violations |
| -------------- | ------------------------------------- | ----------------- | ------ | ---------------- | ----------------- | ------------------- |
| `decideRefund` | 384 + 394 (3 gold)                    | 0.9872            | 0.0087 | 0.995            | 0 of 778          | 4 of 512 (0.78%)    |
| `triage`       | 384 (3 gold)                          | 1.0000            | 0.0000 | 1.000            | 0 of 384          | no constraints      |

The held-out tolerance is 1% (5 of 512 inputs); `--select-best-epoch` kept
epoch 10 of 16 for both. Of the 381 synthetic refund cases the constraints
decided 374. The fallback labelled the 7 cancelled orders at 90 days or less,
the one region no rule decides, and all 381 synthetic `triage` cases, since
`triage` has no constraints: 97 requests (USD 0.22) went to the refund
function and 381 (USD 0.75) to `triage`. The fallback also built the
counterfactual twins the constraints could not, at `--counterfactual-ratio
0.5`, without the twin failures the local fallbacks had
([teachers](../../docs/teachers.md)). 202 of the 384 refund cases are past 90
days, up to 240.

`semantscript releases list` prints the same release (`0fd67142d16f`,
`2/2 passed`, min accuracy `0.9872`, max ECE `0.0087`, violations `0`), and
`semantscript test` and `releases show 0fd67142` print `4/512` in the
`held-out` column and `1` in the `seed` column. `semantscript explain` answers
`deny` for both tiers, paid and fraudulent orders, at 100, 120, 150 and 200
days (`priorRefunds` 0, a total of 100): 16 of 16, each with the 90-day rule
satisfied:

```sh
npx semantscript explain dist/refunds.sem.js --call decideRefund \
  --input '[{"tier":"standard","priorRefunds":0},{"total":100,"ageDays":150,"status":"fraudulent"}]'
#  value        "deny"
#  confidence   0.9935, uncertainty 0.0395
#  constraints  2 active of 6 (4 inactive)
#    always "deny" when order.ageDays > 90: satisfied
#    never "approve" when order.status === "fraudulent": satisfied
```

`npm start` then serves it: `POST /refunds/o1` (a standard customer's
12-day-old paid order) commits an `approve`, `POST /refunds/o2` (an
enterprise customer's 70-day-old paid order) answers `deny` and writes
nothing, and `POST /tickets` answers `201` with `"urgent"` for a production
outage.

### Retrains from the older cached datasets failed the held-out check

Before this release, the example served `217d386c…` (below), which predates
the held-out check and answered `approve` for paid and `review` for
fraudulent orders at 100, 120, 150 and 200 days, for both tiers: none of the
16 is the `deny` the 90-day rule requires. `.semantscript/cache` still holds
the language-model teacher's reduced-prompt datasets it came from (refund
`caa8e895…` with adversarial set `6833e7fc…`, triage `59f21134…`: the content
sha256 that the train report and `explain` print; the cache files are keyed
`789d3df6…`, `fde84ed3…` and `2cbfbe07…`), 192 cases each. On 2026-09-26
(RX 9070 XT) the recipe
`--teacher .semantscript/teacher.toml --cases 192 --epochs 8 --select-best-epoch --counterfactual-ratio 0.5 --max-constraint-violation-rate 0.01`
ran from them at seeds 1 to 10 with `--seed-attempts 1`, and then the
training-only variants below. Every run sent 0 teacher requests (USD 0);
`triage` passed each time at accuracy 1.000, and `decideRefund` failed each
time on the held-out check (the sample of 512 inputs is drawn from the
build's seed; tolerance 1%, decision-8):

| Seed | Variant                              | Accuracy | ECE    | Corpus violations | Held-out violations |
| ---- | ------------------------------------ | -------- | ------ | ----------------- | ------------------- |
| 1    | recipe above                         | 0.9747   | 0.0227 | 1 of 394          | 60 of 512 (11.7%)   |
| 2    | recipe above                         | 0.9750   | 0.0414 | 6 of 394          | 28 of 512 (5.5%)    |
| 3    | recipe above                         | 0.9873   | 0.0106 | 5 of 394          | 70 of 512 (13.7%)   |
| 4    | recipe above                         | 0.9620   | 0.0302 | 8 of 394          | 53 of 512 (10.4%)   |
| 5    | recipe above                         | 0.9494   | 0.0620 | 8 of 394          | 109 of 512 (21.3%)  |
| 6    | recipe above                         | 0.9625   | 0.0122 | 0 of 394          | 30 of 512 (5.9%)    |
| 7    | recipe above                         | 0.9367   | 0.0703 | 12 of 394         | 72 of 512 (14.1%)   |
| 8    | recipe above                         | 0.9231   | 0.0290 | 8 of 394          | 60 of 512 (11.7%)   |
| 9    | recipe above                         | 0.9750   | 0.0366 | 5 of 394          | 50 of 512 (9.8%)    |
| 10   | recipe above                         | 0.9351   | 0.0876 | 1 of 394          | 43 of 512 (8.4%)    |
| 2    | `--epochs 16`                        | 0.9625   | 0.0306 | 1 of 394          | 25 of 512 (4.9%)    |
| 2    | `--epochs 16` (or 32), best epoch 12 | 0.9875   | 0.0225 | 2 of 394          | 20 of 512 (3.9%)    |
| 2    | `--epochs 16 --batch-size 4`         | 0.9875   | 0.0214 | 0 of 394          | 38 of 512 (7.4%)    |
| 2    | `--learning-rate 5e-5`               | 0.9625   | 0.0450 | 9 of 394          | 60 of 512 (11.7%)   |
| 2    | `--epochs 16 --learning-rate 1e-5`   | 0.9750   | 0.0517 | 2 of 394          | 34 of 512 (6.6%)    |
| 2    | `--head-architecture mlp`            | 0.9750   | 0.0371 | 6 of 394          | 43 of 512 (8.4%)    |
| 2    | without `--select-best-epoch`        | 0.9000   | 0.0770 | 5 of 394          | 49 of 512 (9.6%)    |
| 6    | `--epochs 16`                        | 0.9500   | 0.0276 | 0 of 394          | 28 of 512 (5.5%)    |
| 6    | `--epochs 16` (or 32), best epoch 7  | 0.9625   | 0.0122 | 0 of 394          | 30 of 512 (5.9%)    |
| 6    | `--learning-rate 5e-5`               | 0.9500   | 0.0569 | 2 of 394          | 37 of 512 (7.2%)    |
| 6    | `--epochs 16 --learning-rate 1e-5`   | 0.9625   | 0.0202 | 0 of 394          | 35 of 512 (6.8%)    |
| 6    | `--head-architecture mlp`            | 0.9500   | 0.0493 | 6 of 394          | 77 of 512 (15.0%)   |
| 6    | without `--select-best-epoch`        | 0.9375   | 0.0343 | 2 of 394          | 35 of 512 (6.8%)    |
| 1    | `--epochs 32`, best epoch 12         | 0.9873   | 0.0261 | 0 of 394          | 67 of 512 (13.1%)   |
| 9    | `--epochs 16`, best epoch 5          | 0.9750   | 0.0241 | 6 of 394          | 65 of 512 (12.7%)   |

Variants keep the recipe's other flags; "best epoch" rows pass
`--select-best-epoch`. The best run broke 20 of 512 held-out inputs, four
times the tolerance, and the broken inputs are spread over constraints 0
(past 90 days), 2 (fraudulent within 90 days), 3 (paid outside the tier
window), 4 and 5 (inside the window), not only the 90-day rule. A narrow
corpus gate passes on most of these runs; the held-out check is what fails.
Seed 5 broke 8 of 394 corpus records on this retrain, where it had broken
none when it published `217d386c` (GPU training is not bit-for-bit
repeatable).

Those datasets were thin where the check samples: the language model chose
every refund input, and 51 of the 192 cases were past 90 days, 11 of them
past 120 (the new dataset has 202 of 384 past 90). `train --estimate` priced three routes to more teacher data with the
OpenRouter list of 2026-09-25: gold examples for stale orders (USD 0.81, max
2.67, the refund dataset only), `--cases 384` with the same teacher (USD
2.02, max 5.71, both datasets) and the mixed constraints teacher (USD 0.39,
max 0.78, at 192 cases). The mixed teacher came first, at `--cases 384`, and
its first run published the release above; the other two were not run.

### Earlier releases

- `217d386c…` (2026-09-26T00:48Z, the reduced-prompt datasets above, seed 5
  inferred from its report `.semantscript/train-report-reduced-seed5.json`,
  since the manifest has no seed field): refund accuracy 0.9241, ECE 0.0719,
  0 of 394 corpus violations, never checked on held-out inputs, and it breaks
  the 90-day rule (above). It stays under `.semantscript/artifact/releases`
  with its int8 derivation `f8e22cae…`, and promoting `217d386c` would go
  back to it.
- `5c755d08…` (2026-09-25, the teacher prompt of that day, seed 3, refund
  accuracy 0.987, ECE 0.008, 0 of 394 corpus violations) is no longer under
  `.semantscript/artifact/releases`; generating its datasets cost about USD
  10 for about 600 requests. The same corpus regenerated with the reduced
  teacher prompt (TASK-14.5: constraints sent as source text, no duplicated
  schema) cost USD 0.99 for 496 requests under `--max-cost-usd 3`, and those
  are the cached datasets above. It was not checked on held-out inputs either.

`train` retries seeds itself
([seed retry](../../docs/cli-reference.md#seed-retry)): a failure only on the
violation rate or the ECE, within `--seed-retry-margin` times the gate
(default 2), retrains on the cached datasets with the next seed, up to
`--seed-attempts` runs (default 3). On the reduced-prompt datasets
(2026-09-25, before the held-out check), `--seed 2 --seed-attempts 5
--seed-retry-margin 3` failed seed 2 at 6 of 394 and seed 3 at 5 of 394,
then published seed 4 at 2 of 394, with no teacher request. Under the
held-out check, every run from those datasets is outside the default margin
(the best, 3.9%, is past twice the 1% tolerance), so a retry would stop after
its first seed; the sweep ran one seed per build instead. The release on disk
passed on its first seed, so its `--seed-attempts 3` never retried.

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
  it copies the directory owned by the `node` user (the trainer publishes releases owner-only), runs as that user and starts
  `dist/server.js`. The bundle carries native bindings for one platform and
  arch, so package for the image's: on macOS or another non-Linux host pass
  `--platform linux --arch arm64` (Apple silicon, whose Docker builds
  linux/arm64) or `--platform linux --arch x64` (an Intel Mac or Windows),
  or the container fails when it loads the runtime. CI packages this example with the fixture artifact on
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

Bundle sizes on linux/x64 (Node 22.22), the fixture measured 2026-09-25 and
the trained releases 2026-09-26:

| Artifact                                         | node_modules | Artifact  | Total     | lambda-zip (250 MiB) |
| ------------------------------------------------ | ------------ | --------- | --------- | -------------------- |
| Fixture (`npm run fixture-artifact`)             | 82.3 MiB     | 12.1 KiB  | 82.6 MiB  | fits                 |
| Trained release `0fd67142…` (22 layers, float32) | 82.7 MiB     | 570.9 MiB | 653.9 MiB | over by 403.9 MiB    |
| Int8 release `c7534774…` derived from it (below) | 82.7 MiB     | 145.7 MiB | 228.6 MiB | fits, 21.4 MiB spare |

The float32 bundle is 685,637,515 bytes. The int8 row was derived and
measured on the same machine, and the release sits beside `0fd67142` under
`.semantscript/artifact/releases` (`releases list` prints it, not current):
239,712,504 bytes in total with `deploy/lambda.mjs` included, an encoder of
150,750,065 bytes (143.8 MiB), and the packaged Lambda handler answered
`POST /tickets` from it with `201` and `"urgent"`. `semantscript explain`
on it (promoted for the check, then `0fd67142` promoted back) answered `deny`
on the same 16 stale orders as the float32 release.

The trained release does not fit a Lambda .zip package: its encoder alone is
596,679,464 bytes (569.0 MiB), a full-depth float32 ModernBERT-base, and AWS
counts 262,144,000 bytes unzipped for the function and its layers together.
`package --target lambda-zip` exits 1 with `PACKAGE_OVER_TARGET` and projects
each lever from the measured encoder sizes: depth routing alone does not fit
(about 462, 347 and 309 MiB at 12, 6 and 4 layers, because the 82 MiB of
dependencies stay), depth 6 or 4 with int8 would (about 151 and 141 MiB), int8
alone would (about 228 MiB), and so would an encoder whose graph is at most
165.2 MiB.

`semantscript releases derive --int8` is the int8 step. It checks the int8
chain against the float32 one on the release's own records from the build
cache and, by default, refuses if any decision changes. On `0fd67142` the
strict default gate publishes with `--per-channel`:

```sh
npx semantscript releases derive --int8 --per-channel
```

| Records                                                                                 | Decisions changed | Attested changed | Worst ECE, float32 / int8 | Encoder           | Time on CPU |
| --------------------------------------------------------------------------------------- | ----------------- | ---------------- | ------------------------- | ----------------- | ----------- |
| 1,162: 778 `decideRefund` (384 training, 3 of them gold, 394 adversarial), 384 `triage` | 0                 | 0 of 6           | 0.0039 / 0.0047           | 569.0 → 143.8 MiB | 1 min 23 s  |

The time is wall clock on the development machine (Ryzen 9 9900X, WSL2, CPU
only), and the report is `.semantscript/derive-report-15.2-c7534774.json`.
The check has no held-out set: the train report records the release gate's
held-out sample
([held-out constraint check](../../docs/training-pipeline.md#held-out-constraint-check)),
but `releases derive --int8` does not verify on it yet; the `explain` grid
above is the check that it still denies stale orders. `semantscript releases
promote c7534774` and `semantscript package` again ship it; promoting
`0fd67142` goes back.

The previous release needed a recorded tolerance for the same step:
`217d386c` changed 3 of its 586 records under `--per-channel` (1
`decideRefund`, 2 `triage`, none of the gold examples; int8 ECE 0.0914), and
only `--max-decision-change-rate 0.0052` published its int8 release
`f8e22cae` (2026-09-26, 239,708,539 bytes packaged).

Without the int8 release, ship the trained release as a container:
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
far quantization cuts it on the refund benchmark, and `semantscript releases
derive --int8` derives such a release for an application (above); its memory
has not been measured on this example.
