# SemantScript documentation

SemantScript adds one compile-time primitive to TypeScript: a typed semantic
expression, `sema<T>`, whose behavior is learned during the build and executed
at runtime by a fixed, non-generative head. This index links every document the
project has produced so far. Start with the language reference if you write
`.sem.ts` files, the IR and artifact reference if you build tooling around the
compiler, trainer or runtime, and CONTRIBUTING if you change the repository.

## Reader-facing references

- [Language reference](language-reference.md): `sema<T>` syntax, inputs,
  examples, constraints, every v1 output type, `@confidence`,
  `sema.withConfidence`, runtime behavior and compile errors, with runnable
  examples under [`docs/examples`](examples/README.md).
- [IR and artifact reference](ir-and-artifact-reference.md): the NeuralFunction
  IR record field by field, the IR bundle and execution plan, the application
  artifact manifest, the on-disk release layout and the canonical input
  encodings.
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
  training, the release gate's violation tolerance, and compile-time reduction
  (fine-tune the application encoder, not a universal frozen one).
- [Laya analysis](research/laya-analysis.md): what was adopted from the Laya
  typed-decision models and what was not.

## Component guides

- [Compiler](../compiler/README.md): site discovery, type analysis, definition
  resolution, execution plans, bundle emission, the diagnostics catalogue and
  the build-tool adapters (ts-patch transformer, esbuild, Vite, webpack and
  Turbopack loader).
- [Runtime](../runtime/README.md): artifact loading, canonical input
  serialization, inference, confidence policy, fallbacks, stages and execution
  plans.
- [Trainer](../trainer/README.md): synthetic and adversarial datasets, training,
  calibration and verification, verified IR, artifact export, the bundle driver
  and build cache, quantization, teacher backends.
- [Model](../model/README.md): the encoder, adapter and head modules and the
  ONNX export with parity checks.
- [CLI](../cli/README.md): `semantscript init | build | train | test | run`,
  the zero-config defaults and the build cache rules.
- [Examples](../examples/README.md), including the
  [Express](../examples/express-app/README.md) and
  [Next.js](../examples/next-app/README.md) applications that adopt sema
  through a build-tool adapter.

## Benchmarks and results

- [Benchmarks overview](../benchmarks/README.md) and the
  [refund benchmark](../benchmarks/refund/README.md) with its
  [program guide](../benchmarks/refund/program/README.md).
- [Typed-decisions benchmark](../benchmarks/typed-decisions/README.md): the
  external multi-question suite compiled to sema programs, with its results
  under `benchmarks/typed-decisions/data/`.
- Committed results under `benchmarks/refund/data/`: the Phase 1 go/no-go
  (`results-v2-2026-09-23`), the int8 experiment (`results-int8-2026-09-24`),
  the compact encoding run (`results-compact-2026-09-24`), the shared-encoder
  application (`release-application-2026-09-24`), stage scaling
  (`results-stage-scaling-2026-09-24`) and the warm-start experiment
  (`results-warm-start-2026-09-24`); each directory has a README that reads on
  its own.
