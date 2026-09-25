# SemantScript documentation

SemantScript adds one compile-time primitive to TypeScript: a typed semantic
expression, `sema<T>`, whose behavior is learned during the build and executed
at runtime by a fixed, non-generative head. This index links every document the
project has produced so far. Start with the language reference if you write
`.sem.ts` files, the IR and artifact reference if you build tooling around the
compiler, trainer or runtime, and CONTRIBUTING if you change the repository.

## Start here

- [Getting started](getting-started.md): add one sema expression to an
  existing Express (or Next.js) app and take it through `init`, `build`,
  `train`, `test` and `run`, with the hardware, teacher and time it needs.
- [Build tools](build-tools/tsc.md): setup and known limitations per adapter,
  [tsc](build-tools/tsc.md), [esbuild](build-tools/esbuild.md),
  [Vite](build-tools/vite.md) and [Next.js](build-tools/next.md).

## Reader-facing references

- [CLI reference](cli-reference.md): every command, flag, default,
  environment variable and exit code of `semantscript`.
- [Build cache](build-cache.md): what is content-addressed, when retraining
  happens, where the caches live and how to clear them.
- [Diagnostics catalogue](diagnostics.md): every compiler diagnostic, editor
  warning, trainer and verifier failure and runtime error, with cause and fix.
- [Language reference](language-reference.md): `sema<T>` syntax, inputs,
  examples, constraints, every v1 output type, `@confidence`,
  `sema.withConfidence`, runtime behavior and compile errors, with runnable
  examples under [`docs/examples`](examples/README.md).
- [IR and artifact reference](ir-and-artifact-reference.md): the NeuralFunction
  IR record field by field, the IR bundle and execution plan, the application
  artifact manifest, the on-disk release layout and the canonical input
  encodings.
- [Execution plans](execution-plans.md): how independent sema expressions
  fuse into one encoder pass and dependent ones become stages, with a worked
  example from source to DAG to stages and what the runtime does with them.
- [Structured outputs](structured-outputs.md): flat interface outputs, how
  each field becomes a head, what verification reports per field, and the
  diagnostic result shape.
- [Scaling results](scaling-results.md): the parallel-head, batch, one-stage
  application and depth-routing measurements, with guidance on when fusing
  helps.
- [CONTRIBUTING](CONTRIBUTING.md): repository layout, the Node and Python
  toolchains, lint and tests for both halves, and the Backlog workflow.

## Normative specifications

- [`SPEC.md`](../SPEC.md): the v1 language contract (source syntax and
  observable runtime behavior). The language reference explains it; the spec
  decides.
- [`IR.md`](../IR.md): the NeuralFunction IR, the application artifact and the
  canonical input encoding. The IR and artifact reference explains it.
- JSON Schemas under [`schemas/`](../schemas): `neural-function.v1`,
  `ir-bundle.v1`, `application-artifact.v1`, `artifact-pointer.v1`, and the
  refund benchmark's dataset, predictions, result, ledger and common contracts.
- Golden vectors under [`examples/`](../examples/README.md): a hand-written IR
  record, a manifest and pointer, and the canonical input vectors for both
  encodings.

## Plans and decisions

- [`PLAN.md`](../PLAN.md): the project plan, phases and success metrics.
- [`AGENTS.md`](../AGENTS.md): the Backlog.md workflow every contributor and
  agent follows.
- [`DEVELOPING.md`](../DEVELOPING.md): environment setup commands (summarized in
  CONTRIBUTING).
- Backlog decisions under [`backlog/decisions/`](../backlog/decisions): teacher
  model, training stack, base encoder, proper-scoring loss and calibration,
  held-out data adjudication, constraint-based policy statement, real-input
  training, the release gate's violation tolerance, compile-time reduction
  (fine-tune the application encoder, not a universal frozen one), and routed
  domain adapters as the default layout for depth routing and isolation.
- [Laya analysis](research/laya-analysis.md): what was adopted from the Laya
  typed-decision models and what was not.
- [Universal encoder investigation](universal-encoder.md): whether a frozen
  universal encoder with tiny heads could make compilation take seconds
  (measured: no), and decision-9's recommendation.

## Component guides

- [Compiler](../compiler/README.md): site discovery, type analysis, definition
  resolution, execution plans, bundle emission, the diagnostics catalogue,
  the build-tool adapters (ts-patch transformer, esbuild, Vite, webpack and
  Turbopack loader) and the editor plugin (hover and diagnostics at sema
  sites).
- [Runtime](../runtime/README.md): artifact loading, canonical input
  serialization, inference, confidence policy, fallbacks, stages and execution
  plans.
- [Trainer](../trainer/README.md): synthetic and adversarial datasets, training,
  calibration and verification, verified IR, artifact export, the bundle driver
  and build cache, quantization, teacher backends.
- [Model](../model/README.md): the encoder, adapter and head modules and the
  ONNX export with parity checks.
- [CLI](../cli/README.md): `semantscript init | build | train | dev | test |
run`, the zero-config defaults, the dev loop and the build cache rules.
- [Framework](../framework/README.md): decorated controllers for Express, Nest
  and Next.js handlers, one sema request scope per request, and deterministic
  guards around neural results.
- [Examples](../examples/README.md), including the
  [Express](../examples/express-app/README.md) and
  [Next.js](../examples/next-app/README.md) applications that adopt sema
  through a build-tool adapter, and the
  [refund service](../examples/refund-service/README.md) reference
  application: three routed domains, nine decisions, trained from its own
  constraints, with a walkthrough of source, compiled output and artifact.

## Benchmarks and results

- [Benchmarks overview](../benchmarks/README.md) and the
  [refund benchmark](../benchmarks/refund/README.md) with its
  [program guide](../benchmarks/refund/program/README.md).
- [Typed-decisions benchmark](../benchmarks/typed-decisions/README.md): the
  external multi-question suite compiled to sema programs, with its results
  under `benchmarks/typed-decisions/data/` (`results-2026-09-24` per workflow,
  `results-routed-2026-09-24` single versus routed domain adapters).
- Committed results under `benchmarks/refund/data/`: the Phase 1 go/no-go
  (`results-v2-2026-09-23`), the int8 experiment (`results-int8-2026-09-24`),
  the compact encoding run (`results-compact-2026-09-24`), the shared-encoder
  application (`release-application-2026-09-24`), stage scaling
  (`results-stage-scaling-2026-09-24`), the warm-start experiment
  (`results-warm-start-2026-09-24`), the encoder sweep
  (`results-encoder-sweep-2026-09-24`), the depth-routing sweep
  (`results-depth-sweep-2026-09-24`) and the depth-routed final-set re-run
  that meets the latency criterion (`results-depth-006-2026-09-25`, release
  `release-depth-006-2026-09-25`); each directory has a README that reads on
  its own.
