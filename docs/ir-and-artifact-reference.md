# IR and artifact reference

This page documents the records that connect the compiler, trainer and
runtime: the NeuralFunction IR (compiler to trainer), the IR bundle with its
execution plan, the application artifact manifest (trainer to runtime), the
release layout on disk and the canonical input encodings. [`IR.md`](../IR.md)
is the normative text; the JSON Schemas under [`schemas/`](../schemas) are the
machine-checked shapes. All records are UTF-8 JSON without a byte-order mark,
closed to unknown properties, with major versions the consumer must check.

## NeuralFunction IR (`schemas/neural-function.v1.schema.json`)

One record per compiled `sema` expression. The same record advances through
the build: the compiler emits `stage: "source"` with pending training and
verification, training makes it `trained` with complete provenance, and
verification makes it `verified` with metrics. Only a verified, passing record
is exported.

| Field                | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`               | `semantscript.neural-function`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `irVersion`          | `1`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `stage`              | `source`, `trained` or `verified`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `id`                 | `nf_` plus SHA-256 of the semantic-JSON encoding of `["semantscript-function-id", 1, <normalized source path>, <duplicate ordinal>, <semanticSha256>]`. Stable across unrelated edits; changes when the expression's semantics, its file or an identical earlier site changes.                                                                                                                                                                                                                                              |
| `semanticSha256`     | SHA-256 of the semantic JSON of `{ irVersion, definition, inputs, output, runtime }`. Source coordinates, model refs, training and verification are excluded, so retraining does not change identity while changing behavior does.                                                                                                                                                                                                                                                                                          |
| `source`             | `path` (relative to the project root, `/` separated), `line`, `column` of the site and `sourceSha256` of the file. Diagnostic only.                                                                                                                                                                                                                                                                                                                                                                                         |
| `definition`         | `template`: ordered parts, each `{kind:"text", text}` or `{kind:"input", name}`, so text and inputs cannot be confused; `examples`: `{inputs, output}` gold cases; `constraints`: `{kind: always\|never, source, predicate, output}` with the predicate as a portable expression AST.                                                                                                                                                                                                                                       |
| `inputs`             | Ordered `{name, index, tsType, type}` descriptors. `type` is recursive: `string`, `boolean`, `number`, `null`, literal, enum (`name`, `base`, `values`), array, tuple, object (`fields` with `optional`), union (variants sorted by semantic JSON bytes and deduplicated).                                                                                                                                                                                                                                                  |
| `output`             | `{kind:"scalar", tsType, head}` or `{kind:"object", tsType, fields:[{name, tsType, head}]}`.                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `output...head`      | `booleanHead` (support `[false,true]`), `nominalStringHead` (`string-union` or `string-enum`), `nominalNumberHead` (`number-enum`), `ordinalStringHead` (`ordinal-string`, expected value is the zero-based rank) or `ordinalNumberHead` (`bounded-int` or `bounded-number` with canonical decimal `minimum`, `maximum`, `step` and `supportDecimal`).                                                                                                                                                                      |
| `model`              | Logical refs, never paths: `encoder`, `adapter` and `heads: [{outputPath, ref}]`, one head per scalar output (`outputPath` `""`) or per field (JSON Pointer `/field`, with `~0`/`~1` escaping). In a routed bundle `adapter` is the function's domain adapter (`adapter.<app>.<domain>`), `encoder` names the depth prefix it reads (`encoder.<app>.depth-NNN`) and `encoderDepth` is the number of shared-encoder layers that prefix runs (absent for the full stack). Routing is build metadata outside `semanticSha256`. |
| `runtime`            | `resultMode` (`value` or `diagnostic`), `confidenceThreshold` (number or null), `fallbackRef` (string or null), `synchronous: true`.                                                                                                                                                                                                                                                                                                                                                                                        |
| `trainingProvenance` | `{status:"pending"}` in source records; once complete: teacher (`provider`, `model`, `configurationSha256`), `baseModel` (`name`, `revision`, `weightsSha256`), `datasetSha256`, `counts` (examples, synthetic, adversarial, calibration, verification, attestedVerification), `seed`, `trainer` (`version`, `commit`), `trainedAt` and `canonicalInput`.                                                                                                                                                                   |
| `verification`       | `{status:"pending"}` until verified; then `status`, `verifiedAt` and `metrics`: `accuracy`, `ece`, `brier`, `pairConsistency`, per-head `heads[{outputPath, accuracy, pairConsistency, calibration}]`, `exampleFailures`, `constraintViolations`, `typeErrors`.                                                                                                                                                                                                                                                             |

Relational rules the schema cannot express are enforced by every consumer:
unique and dense input indices matching the template, examples covering every
input exactly once, constraint ASTs resolving to declared inputs with outputs
in the support, canonical decimals in fixed notation, exactly one head per
scalar output or field, and refs that resolve. A hand-written record is at
[`examples/ir/refund-decision.v1.json`](../examples/ir/refund-decision.v1.json).

## IR bundle (`schemas/ir-bundle.v1.schema.json`)

`semantscript build` writes one bundle per program: `kind`
(`semantscript.ir-bundle`), `bundleVersion` (`1`), the source-ordered
`functions` array of NeuralFunction records, and `executionPlan`. The plan has
`stages` (`{index, functionIds}`, dense from zero) and `dependencies`
(`{producerFunctionId, consumerFunctionId, consumerInput}`). Independent
functions share stage zero; a dependent function sits one stage after its
deepest producer. Dependencies are a conservative may-analysis of data flow
through variables, aliases, destructuring and direct property writes, so a
stage is a topological stratum, not a promise that its functions execute
together. The plan is build metadata and does not enter semantic identity; the
runtime's `executeSemaPlan` runs it stage by stage, sharing one encoder pass
among functions with identical inputs. The [execution plans](execution-plans.md)
page walks through one.

A routed bundle (built with `--route-domains`, a `--domain-depth`, or an
`@domain` header) adds `domains` to the plan: one entry per compile-time
domain with `name`, `adapterRef`, `encoderRef`, `encoderDepth` (`null` for the
full stack) and its member `functionIds`, and every stage lists the
`adapterRefs` its functions use. An unrouted bundle has neither field and
every function shares the application's single adapter.

## Application artifact manifest (`schemas/application-artifact.v1.schema.json`)

The runtime's contract: one tokenizer, one shared encoder (exported whole
and, for depth-routed domains, once more per prefix depth), one adapter per
domain (one for the whole application when unrouted) and one or more heads per
function, all verified. A runtime never needs source text, examples,
constraints, a teacher or the network.

| Field                | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind`               | `semantscript.application-artifact`; `artifactVersion` `1`; `irVersion` `1`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `compatibility`      | `runtimeAbiVersion` and `modelAbiVersion` (`1`), `canonicalInput` (`semantscript.canonical-input/v1` or `/v2`), `minimumRuntimeVersion` (semver) and `requiredCapabilities`; a runtime checks all of them before opening a model.                                                                                                                                                                                                                                                                                                                                                       |
| `application`        | `id` and semver `version`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `build`              | `createdAt`, `compilerVersion`, `trainerVersion` and `sourceIrSha256` (one function: the digest of its verified IR bytes; several: the digest of the per-function digests joined by newlines).                                                                                                                                                                                                                                                                                                                                                                                          |
| `resources`          | One entry per file: `ref` (logical), `role` (`tokenizer`, `encoder`, `adapter`, `head`), `format` and `formatVersion`, `path` (portable, relative to the release), `byteLength`, `sha256`. The tokenizer entry adds `maximumSequenceLength`; ONNX entries add `onnx`: `opset`, typed `inputs` and `outputs` (`name`, `dtype`, `shape` with `BATCH`/`SEQUENCE` symbols), `externalData: false`, optional `precision` (`float32` or `int8-dynamic`) and `quantization` (method, weight type, per-channel, reduce-range, the disagreement and ECE tolerances, the source manifest digest). |
| `model`              | `tokenizerRef`, `encoderRef`, `adapterRef`: the application-level defaults. When every domain runs the same prefix, `encoderRef` is that prefix and no function overrides it.                                                                                                                                                                                                                                                                                                                                                                                                           |
| `functions[]`        | `id`, `semanticSha256`, the exact `inputs` descriptors plus `inputSchemaSha256` and `outputSchemaSha256`, `adapterRef` (the function's domain adapter), optional `encoderRef` (the encoder resource it reads when its domain's depth differs from the model default), `heads`, `runtime`, `verification`, `trainingProvenance`.                                                                                                                                                                                                                                                         |
| `heads[]`            | `outputPath` (`[]` for a scalar, `["field"]` for a flat output field), `headRef`, `type` (`boolean`, `nominal-string`, `nominal-number`, `ordinal-string` with `support`, or `ordinal-number` with the decimal grid), `parameterization` (`binary-sigmoid` for booleans, `categorical-softmax` otherwise), `calibration` (`temperature-scaling`, `temperature`, `ece`, `brier`, `sampleCount`, `splitSha256`, `eceBins`) and `verification` (`accuracy`, `pairConsistency`).                                                                                                            |
| `runtime`            | `resultMode`, `confidenceThreshold`, `policy` (`none` without a threshold, `scalar-top1` or `all-fields` with one) and `fallbackRef`.                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `verification`       | `status: "passed"`, `accuracy`, `ece`, `brier`, `pairConsistency`, `attestedCases`, `exampleFailures: 0`, `constraintViolations` (the raw-model count the gate tolerated) and `typeErrors: 0`.                                                                                                                                                                                                                                                                                                                                                                                          |
| `trainingProvenance` | `datasetSha256`, `trainingKeySha256`, `seed` (the training seed the function passed verification at; optional, absent in releases built before it was recorded) and the `teacher` and `baseModel` identity strings.                                                                                                                                                                                                                                                                                                                                                                     |

Model ABI v1: the encoder (or a prefix of it) takes `input_ids` and
`attention_mask` (`int64`, `[BATCH, SEQUENCE]`) and emits `sentence_embedding`
(`float32`, `[BATCH, HIDDEN]`); an adapter maps it to `function_embedding` of
the same shape; each head maps that to `logits` shaped `[BATCH, 1]` for a
boolean or `[BATCH, K]` for `K` support values. A flat interface output is one
head per field, each with its own calibration and verification record; see
[structured outputs](structured-outputs.md). The runtime divides logits by the head's temperature
and applies sigmoid or softmax. An illustrative manifest is at
[`examples/artifacts/refund-app/manifest.v1.json`](../examples/artifacts/refund-app/manifest.v1.json).

## Release layout and pointer (`schemas/artifact-pointer.v1.schema.json`)

```text
<artifact-root>/
  current.json                         # {kind, pointerVersion, release, manifestSha256}
  releases/
    sha256-<manifest-sha256>/
      manifest.json
      tokenizer/tokenizer.json
      models/encoder/model.onnx                  # the full encoder (unrouted, or a full-depth domain)
      models/encoder/depth-006.onnx              # one prefix graph per routed depth
      models/adapters/application.onnx           # unrouted: the one adapter
      models/adapters/adapter-<app>-<domain>.onnx  # routed: one per domain
      models/heads/<function-id>/head-000.onnx   # head-001.onnx ... for flat outputs
```

A prefix graph is a complete ONNX model of the shared encoder's first `n`
layers, so an application whose domains use two depths ships the shared layers
twice; the refund service's single depth-6 prefix is 275 MB against 597 MB for
the full stack.

A release directory is named after the SHA-256 of its exact manifest bytes and
is never modified after publication. The exporter writes into a staging
directory on the same filesystem, hashes every resource, writes the manifest
last, validates it and renames the directory into place; `current.json` is
replaced atomically afterwards. The runtime re-hashes the manifest and every
resource through the handle it loads from, rejects symlinks, paths outside the
release and oversized files, checks tensor names, shapes, opsets and the
encoder-adapter-head chain, and only then swaps the new function table into
service.

Releases accumulate: nothing removes an old one on publication, so switching
back is a pointer rewrite. `semantscript releases` lists them, `releases
rollback` and `releases promote` rewrite `current.json` atomically after
checking the target release the same way (every check above except ONNX
session start-up), and `releases prune` deletes old
releases other than the current one (see the
[CLI reference](cli-reference.md#releases)).

## Canonical input encodings

The runtime validates a call's inputs against the function's `inputs`
descriptors and serializes them to one deterministic byte sequence that the
tokenizer consumes; the trainer serializes generated cases the same way, so
training and inference see identical text. Object key order, tuple length and
enum membership are fixed by the schema; strings are not Unicode-normalized.

- `semantscript.canonical-input/v1` is an exact JSON envelope,
  `["semantscript-input",1,[[name, typedValue], ...]]`, with typed nodes for
  null, boolean, string, number (16 hex characters of the binary64 bits, so
  `-0` survives), literal, enum (name, declaration index, value), union
  (variant index, value), array, tuple and object (pairs sorted by UTF-8 key
  bytes). Golden vector: [`examples/serialization/canonical-input.v1.json`](../examples/serialization/canonical-input.v1.json).
- `semantscript.canonical-input/v2` renders the same typed tree as compact
  text: `name=value` pairs separated by spaces, `{k=v,...}` objects with keys
  in the same sorted order, `[v,...]` for arrays and tuples, `true`, `false`,
  `null`, numbers in JavaScript spelling with `-0` preserved, and strings bare
  when they match `[A-Za-z_][A-Za-z0-9_.-]*` and are not `true`, `false` or
  `null`, quoted with JSON escapes otherwise. Literal, enum and union tags are
  dropped because the schema fixes them per path. The refund input
  `customer={priorRefunds=0,tier=standard} order={ageDays=12,status=paid,total=88.5}`
  costs about a third of the tokens of its v1 envelope. Golden vectors:
  [`examples/serialization/canonical-input.v2.json`](../examples/serialization/canonical-input.v2.json).

The artifact's `compatibility.canonicalInput` and each verified record's
`trainingProvenance.canonicalInput` name the encoding the model was trained
with; the trainer defaults to v2 and the runtime loads either. A training
cache key combines the semantic digest, the canonical bytes and the generator
configuration; an inference cache key combines the function id, the manifest
digest and the canonical bytes.

## Digests

Every `*Sha256` is 64 lowercase hex characters. File digests hash exact bytes.
Semantic values use `semantscript.semantic-json/v1`: the typed tree above,
wrapped as `["semantscript-semantic-json",1,node]`, serialized with the exact
byte rules of the canonical encoding, so signed zero and integral floats hash
consistently across TypeScript and Python.
