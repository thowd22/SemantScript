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

| Lever                      | Encoder (ModernBERT-base) | Condition                                                                                                                                                                                             | Source                           |
| -------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| Depth routing to 12 layers | 395,833,756 B             | `build --domain-depth <domain>=12`, then train; the refund policy kept 160/160 on its final set                                                                                                       | `results-depth-sweep-2026-09-24` |
| Depth routing to 6 layers  | 275,297,443 B             | the same at 6                                                                                                                                                                                         | same                             |
| Depth routing to 4 layers  | 235,118,960 B             | the same at 4                                                                                                                                                                                         | same                             |
| Int8 dynamic quantization  | 150,073,785 B             | a measurement only: no int8 derivation exists for applications yet, and the measured int8 refund release changed 1 of 80 attested cases and 0.105 percent of decisions, which the strict gate refused | `results-int8-2026-09-24`        |
| Depth and int8             | depth size × 0.2515       | depth routing, with the int8 half a measurement only                                                                                                                                                  | both                             |
| A smaller encoder          | the budget left           | `train --encoder-name <model>`; the report gives the largest encoder graph that fits                                                                                                                  | —                                |

The int8 rows are there so the report shows how far quantization would go,
not as a step to take: the int8 derivation has only been run on the refund
benchmark, through a refund-specific driver that replays that benchmark's
release pipeline, and no `semantscript` command derives an int8 release for an
application. For another encoder the projections are estimates. A lever the release
already uses is not offered again: depth routing when a function reads a
routed encoder prefix, int8 when an encoder's `onnx.precision` is not
`float32`. Depth prefixes are separate graphs, so an application with domains
at two depths ships the shared layers twice ([scaling
results](scaling-results.md#depth-routing-the-cost-of-a-pass)).

The Express example's trained release is the worked case: its bundle is
685,309,690 bytes (653.6 MiB), 403.6 MiB over `lambda-zip`. Depth routing
alone leaves it over (the dependencies stay); int8, or depth 6 or 4 with
int8, would fit, but no int8 derivation exists for applications yet, so today
it ships as a container unless a smaller encoder is trained
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
