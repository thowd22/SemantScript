# Deploying

A SemantScript application runs from three things: its compiled output, its
production `node_modules` (with the native ONNX Runtime and tokenizers
bindings) and the artifact release the runtime loads. `semantscript package`
puts them in one directory, checks the release the way the runtime will load
it, and says how big each part is:

```sh
npm run build
npx semantscript train                      # .semantscript/artifact holds a release
npx semantscript package --include deploy/lambda.mjs --target lambda-zip
```

The bundle lands in `.semantscript/package/`, which the `.semantscript/.gitignore`
that `semantscript init` writes keeps out of git (run `init` again on a project
initialised before `package` existed to add the entry). Copy that directory into a
container image or zip it for a function platform; its
`semantscript-package.json` records the SHA-256 and size of every file, so a
deployment can be checked against it. The
[CLI reference](cli-reference.md#package) lists the layout, flags and error
codes.

## What is in the bundle

- `dist/` without `semantscript.ir.v1.json`: the IR bundle holds the prompt
  text and the runtime never reads it.
- `node_modules/` with the production dependencies only. `onnxruntime-node`
  ships its libraries for Linux, macOS and Windows plus a 272 MB CUDA
  provider, and `tokenizers` ships 13 platform binaries; the bundle keeps the
  target platform's (the host's unless `--platform` and `--arch` say
  otherwise) and drops the CUDA and TensorRT providers, because the runtime
  runs ONNX Runtime's CPU provider. On Linux x64 that leaves about 44 MiB of
  ONNX Runtime and 10 MiB of tokenizers.
- `.semantscript/artifact/` with `current.json` and the one release it names.
  A release that fails verification, whose files do not match its manifest,
  or that the runtime's loader refuses is not packaged. The runtime finds the
  directory by searching upward from `dist/`, so no environment variable is
  needed.
- Whatever `--include` names, such as a serverless handler.

## Size targets

`--target` checks the total apparent size (what AWS counts "unzipped")
against a deployment limit. The bundle is always written; over the limit the
command exits 1 with `PACKAGE_OVER_TARGET`.

| Target                | Limit                   | Applies to                                                                                              |
| --------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------- |
| `lambda-zip`          | 262,144,000 B (250 MiB) | an AWS Lambda .zip package and its layers, unzipped                                                     |
| `lambda-image`        | 10 GiB                  | an AWS Lambda container image, uncompressed, including the base image (the bundle is only part of that) |
| `cloud-run-functions` | 500,000,000 B           | Cloud Run functions sources plus modules, uncompressed                                                  |
| `--max-bytes <n>`     | `n` bytes               | anything else                                                                                           |

Cloud Run services and most container platforms have no image size limit;
there the size matters for cold start and memory, not admission. The limits
were read from the providers' quota pages on 2026-09-25.

## The size levers

Almost all of a trained bundle is the encoder: a full-depth float32
ModernBERT-base graph is 596,679,464 bytes, next to about 82 MiB of
dependencies and a few hundred kilobytes of adapter and heads. When a bundle
is over its target, `package` lists each lever with the bundle it would give
and whether that fits, scaling the release's encoder by these measurements:

| Lever                      | Encoder (ModernBERT-base) | Condition                                                                                                                            | Source                           |
| -------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- |
| Depth routing to 12 layers | 395,833,756 B             | every domain at 12 (`build --domain-depth <domain>=12` once per domain), then train; the refund policy kept 160/160 on its final set | `results-depth-sweep-2026-09-24` |
| Depth routing to 6 layers  | 275,297,443 B             | the same at 6                                                                                                                        | same                             |
| Depth routing to 4 layers  | 235,118,960 B             | the same at 4                                                                                                                        | same                             |
| Int8 dynamic quantization  | 150,073,785 B             | `releases derive --int8`, then `releases promote <release>`; the derivation publishes only what its gate admits (below)              | `results-int8-2026-09-24`        |
| Depth and int8             | depth size × 0.2515       | depth routing, then train, then `releases derive --int8`                                                                             | both                             |
| A smaller encoder          | the budget left           | `train --encoder-name <model>`; the report gives the largest encoder graph that fits                                                 | —                                |

The int8 rows project the refund benchmark's measured ratio (the int8 encoder
is 25.15% of the float32 one).
For another encoder the projections are estimates. A lever the release
already uses is not offered again: depth routing when the release ships a
routed encoder prefix (a function `encoderRef`, or a `depth-NNN` encoder named
by `model.encoderRef` when every domain is routed), int8 when an encoder's `onnx.precision` is not
`float32`. A mixed release, where some domains use a prefix and others still
use the full-depth encoder, is offered routing the remaining domains, which
drops the full encoder from the bundle. For an int8 encoder the depth levers
project the float32 prefix that training writes. Depth prefixes are separate graphs, so an application with domains
at two depths ships the shared layers twice ([scaling
results](scaling-results.md#depth-routing-the-cost-of-a-pass)). The depth
projections hold only when every domain is routed: a domain left at full depth
keeps the full encoder in the bundle beside the prefix, so routing one domain
of several makes the bundle larger. A tspc project such as the Express example
sets `domainDepths` in its tsconfig plugin entry instead, since a later
`npm run build` rewrites the IR bundle.

### Deriving an int8 release

`semantscript releases derive --int8` derives an int8 release from the
current release (or a named one) and publishes it beside it:

```sh
npx semantscript releases derive --int8            # strict: no decision may change
npx semantscript releases promote <digest>         # the digest its next: line names
npx semantscript package --target lambda-zip
```

It quantizes the encoder graph (every one, for a depth-routed release) with
ONNX Runtime's dynamic int8 quantization and runs the int8 chain against the
float32 chain on the release's own records, found in the build cache by the
digests the manifest records: each function's training dataset (its gold
examples are the attested records) and its adversarial dataset. With no
tolerance flags it refuses to publish when any decision changed, attested or
not, or when a head's int8 ECE is above 0.1, and it exits 2 with the figures
and the flags that would admit them. A derived release records its source
manifest digest, the quantization settings, the tolerances and the figures in
each quantized encoder's `onnx.quantization` (`releases show` prints them),
the runtime loads it like any release, and `releases promote` or `rollback`
switch to it and back. The check covers the training, gold and adversarial
records only: a held-out set joins it once the release gate records one for
the release. The records are in the build cache, so a release trained on
another machine, or after `.semantscript/cache` was cleared, cannot be
derived until it is retrained where its cache is. The
[CLI reference](cli-reference.md#releases-derive---int8) lists the flags.

The Express example's trained release is the worked case: its bundle is
685,309,690 bytes (653.6 MiB), 403.6 MiB over `lambda-zip`. Depth routing
alone leaves it over (the dependencies stay). `releases derive --int8`
measured it on 2026-09-26 on 586 records (6 attested), on CPU:

| Settings                                          | Decisions changed | Attested changed | Worst int8 ECE | Result                                                          |
| ------------------------------------------------- | ----------------- | ---------------- | -------------- | --------------------------------------------------------------- |
| default                                           | 6 (1.02%)         | 0                | 0.1049         | refused: decisions changed and ECE over 0.1                     |
| `--per-channel`                                   | 3 (0.51%)         | 0                | 0.0914         | refused: decisions changed                                      |
| `--per-channel --max-decision-change-rate 0.0052` | 3 (0.51%)         | 0                | 0.0914         | published; bundle 239,697,592 B (228.6 MiB), fits with 21.4 MiB |

So the strict gate refuses the Express int8 release, and it fits `lambda-zip`
only under a recorded tolerance of 0.52% changed decisions (none attested),
which is the application owner's call; the packaged Lambda handler answered
from that bundle. Without it, the release ships as a container unless a
smaller encoder is trained
([example README](../examples/express-app/README.md#deploying-with-the-artifact)).

## Containers

The Express example's
[`deploy/Dockerfile.package`](../examples/express-app/deploy/Dockerfile.package)
is the whole recipe: `FROM node:22-bookworm-slim`, copy the bundle, run as the
`node` user, start `dist/server.js`. Build the bundle on the image's platform
(or pass `--platform` and `--arch` matching the image: `node:22-bookworm-slim`
is linux/arm64 on Apple silicon), because the native bindings are per
platform. `--platform` and `--arch` prune only `onnxruntime-node` and
`tokenizers`; any other native dependency is installed for the host. CI packages the example on every push, builds this image from the
bundle, sends one request to it and invokes the packaged Lambda handler.
