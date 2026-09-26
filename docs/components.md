# Component guides

Four components, two languages, joined by data. Each section says what the
component takes, what it produces, how it is configured and how to run it on
its own; the package READMEs go deeper.

## Compiler (`@semantscript/compiler`, Node)

**Inputs.** A TypeScript program (a `tsconfig.json` or a `ts.Program`) whose
`.sem.ts` files import `sema`, `always`, `never` and the marker types from
`@semantscript/core`.

**Outputs.** The program's JavaScript with each `sema` site rewritten to
`__sema.call(id, { inputs })`, its source maps, and one IR bundle
(`semantscript.ir.v1.json`) holding a NeuralFunction record per site and the
execution plan. Diagnostics (`TS9100` to `TS9131`) instead of output when a
site is malformed.

**Configuration.** `projectRoot`; `encoderRef`/`adapterRef` or the
`application` name that derives them; `bundlePath`; `domainDepths` and
`routeDomains` for routed adapters; per site, the `@confidence` and `@domain`
headers and the `examples`/`constraints` options.

**Run it alone.**

```sh
npx semantscript build --project tsconfig.json          # CLI
```

```ts
import { compileSemantScriptProgram } from "@semantscript/compiler";
const result = await compileSemantScriptProgram(program, { projectRoot });
if (!result.ok) console.error(result.diagnostics);
```

`planSemaCompilation` plans without writing; the build-tool adapters
(`/transformer`, `/esbuild`, `/vite`, `/loader`) and the editor plugin
(`/ts-plugin`) wrap the same planner. Tests: `npm test -w compiler`.
[Compiler README](../compiler/README.md).

## Trainer (`semantscript_trainer`, Python)

**Inputs.** An IR bundle; a teacher (`[teacher]` TOML: Anthropic, Ollama or
the built-in constraints backend, which labels from the expression's own
constraints, or a Python `Teacher` object); optionally the build cache from the
previous run.

**Outputs.** An immutable artifact release under the artifact root
(`current.json`, `releases/sha256-<manifest>/`), a JSON report per function
(datasets, held-out accuracy, verification metrics and failures, the release
digest), and the updated build cache. A function that fails verification
publishes nothing.

**Configuration.** `TrainingConfig` (encoder name and pinned revision, epochs,
batch size, learning rate, sequence length, evaluation ratio, seed, device,
head architecture, best-epoch selection, canonical input version),
`VerificationConfig` (ECE threshold, constraint-violation tolerance,
calibration bins), `AdversarialGenerationConfig` (counterfactual ratio,
attempts), `SeedRetryConfig` (`--seed-attempts`, `--seed-retry-margin`: how
many seeds a narrow gate miss may try, and how narrow), `--cases`, `--adapter-bottleneck-size`, `--cache-dir`, `--full`,
`--no-cache`. The CLI reference lists every flag.

**Run it alone.**

```sh
PYTHONPATH=trainer/src:model/src python3 -m semantscript_trainer.cli train \
  --bundle dist/semantscript.ir.v1.json --artifact .semantscript/artifact \
  --teacher teacher.toml --cache-dir .semantscript/cache --report train-report.json \
  --cases 200 --epochs 4 --device cuda --local-files-only
```

```python
from semantscript_trainer.cli import train_bundle
result = train_bundle(bundle, ".semantscript/artifact", teacher=teacher, cache_directory=".semantscript/cache")
```

The stages are importable on their own: `SyntheticDatasetGenerator`,
`AdversarialDatasetGenerator`, `train_classifier` / `train_application`,
`evaluate_training_result`, `build_verified_ir`, `export_application_artifact`.
Tests: `pytest trainer/tests`. [Trainer README](../trainer/README.md).

## Model (`semantscript_model`, Python)

**Inputs.** Tokenized canonical input text (`input_ids`, `attention_mask`);
for export, trained modules and one sample batch.

**Outputs.** `SentenceEncoder`: one float32 embedding per input, optionally
from a prefix of the encoder's layers (`depth`). `ApplicationAdapter`: the
function embedding. Heads: raw logits, `[batch, 1]` for a boolean or
`[batch, K]` for `K` support values. `calibrate`: a fitted temperature and
held-out metrics. `export_*`: ONNX graphs with parity checks against PyTorch.

**Configuration.** `EncoderConfig` (model name, full commit hash as
`revision`, `local_files_only`, `trust_remote_code=False`), head
architecture (`linear` or `mlp`), adapter bottleneck size, loss (`proper`
or `cross_entropy`), export opset and precision (`float32` or dynamic int8).

**Run it alone.**

```python
from semantscript_model.encoder import EncoderConfig, SentenceEncoder
encoder = SentenceEncoder(EncoderConfig(model_name="answerdotai/ModernBERT-base",
    revision="8949b909ec900327062f0ebf497f51aef5e6f0c8", local_files_only=True))
embedding = encoder(input_ids, attention_mask, depth=6)
```

Tests: `pytest model/tests` (PyTorch-dependent tests skip without the
`training` extra). [Model README](../model/README.md).

## Runtime (`@semantscript/core`, Node)

**Inputs.** An artifact root or release directory (or none: the default
search) and, per call, a function id with its inputs as plain data.

**Outputs.** The declared output value, or the diagnostic result
(`ScalarSemaResult` / `ObjectSemaResult`) for `sema.withConfidence`; per stage
or scope, the pass counts (`encoder`, `adapter`, `head`). Typed errors for a
missing artifact, an unknown function, invalid inputs, inference failure,
low confidence and bad fallbacks.

**Configuration.** `loadSemaArtifact(path?, { artifact, inference, fallbacks,
watch, onReload, onReloadError })`; `SEMANTSCRIPT_ARTIFACT`; fallbacks are a
map of ref to synchronous function.

**Run it alone.**

```sh
npx semantscript run --artifact .semantscript/artifact dist/app.js --call decideRefund --input '[…]'
```

```ts
import { loadSemaArtifact } from "@semantscript/core";
const handle = await loadSemaArtifact(".semantscript/artifact");
const value = handle.call<"approve" | "deny" | "review">(functionId, {
  customer,
  order,
});
await handle.close();
```

Tests: `npm test -w runtime` (a fixture artifact under `runtime/test/fixtures`
exercises the whole path without a trained model).
[Runtime README](../runtime/README.md).

## The two that tie them together

The **CLI** (`semantscript init | doctor | build | train | dev | test | run | teacher probe`) drives
the four with zero-config defaults ([CLI reference](cli-reference.md)); the
**framework** (`@semantscript/framework`) puts request scopes, guards and
decision-gated transactions around compiled calls in Express, Nest and
Next.js handlers ([framework guide](framework-guide.md)).
