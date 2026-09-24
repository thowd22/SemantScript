# CLI

`semantscript` is the single entry point over the compiler, trainer, verifier
and runtime packages. It coordinates them without owning their core
implementations: `build` is the compiler, `train` is the Python trainer's
bundle driver, `test` and `run` are the runtime.

```text
semantscript build [--project tsconfig.json] [--application <id>] [--bundle <path>]
semantscript train --bundle <path> --artifact <root> --teacher <teacher.toml> [options]
semantscript test  --artifact <root> [--bundle <path>] [--json]
semantscript run   --artifact <root> <module.js> [--call <export>] [--input <json> | --input-file <path>]
```

## build

Reads the TypeScript project (`--project`, default `tsconfig.json`; it must set
`compilerOptions.outDir`), reports configuration and type diagnostics first,
then compiles every `sema` site into a runtime call and writes the IR bundle
(default `<outDir>/semantscript.ir.v1.json`). `--application <id>` names the
artifact's shared encoder and adapter refs (`encoder.<id>`, `adapter.<id>`), so
every function of one program lands in one artifact; the id is lowercase
letters, digits and single dashes. The summary lists each function's id, source
location, output type and head count, the execution plan's stage count and the
emitted files. Prompt text never reaches the emitted JavaScript. Diagnostics
exit with status 1 and emit nothing.

## train

Hands the bundle to `python -m semantscript_trainer.cli train`, which for every
source-stage function generates the synthetic dataset through the configured
teacher (`--teacher` is a TOML file with a `[teacher]` table; see the trainer
README), adds the adversarial sidecar when the function declares constraints
(the teacher must then support boundary and counterfactual generation), trains
one classifier for a single function or the shared-encoder application for
several, verifies every function against the release gate, binds verified IR
and exports one content-addressed artifact under `--artifact`. The trainer
writes a JSON report (`--report`, default `<artifact>.report.json`) that the
CLI renders per function: dataset and adversarial sizes, held-out accuracy,
verification status, accuracy, ECE, attested cases and constraint violations,
plus the published release digest. A function that fails verification stops the
build with the report kept and nothing published; the trainer's exit status is
the command's.

### Build cache

Retraining every function on every build is unacceptable, so `train` keeps a
content-addressed cache under `--cache-dir` (default `.semantscript/cache`),
next to the synthetic and adversarial dataset caches:

```text
.semantscript/cache/applications/<application-id>/
  application.json          recipe digest, last release digest, function index
  shared.safetensors        the application's encoder and adapter weights
  functions/<function-id>/
    function.json           digests, epoch metrics, verification record
    verified-ir.json        exact verified IR bytes
    head.safetensors        the function's head weights
```

The key is the function id, which the compiler derives from the expression's
semantic identity (template text, input and output types, examples, constraints
and runtime policy), so editing an expression changes its id. A cached function
is reused when its id and semantic digest match the bundle, it was built under
the same recipe (encoder name and revision, canonical input version, adapter
size, every training, verification and adversarial setting, and the teacher's
provider, model and configuration digest), its datasets still resolve to the
same digests, and every cache file passes its digest check. Anything else is a
miss for that function.

- **No changes**: nothing trains. When the release the cache last published is
  still the artifact root's current release, nothing is exported either and
  the report says `reused`; otherwise the artifact is re-exported from cached
  weights.
- **One expression changed**: only its head trains, on the frozen shared
  encoder and adapter restored from the cache, so every untouched function
  keeps its verified weights and evidence byte for byte. The new head is
  verified like any other before the whole artifact is exported again.
- **Recipe changed, `--full`, or `--no-cache`**: every function retrains
  jointly (the shared encoder moves), and the previous function records are
  discarded because they were built on the old encoder. `--no-cache` also
  writes nothing.

Deleting the cache directory is always safe; the next build trains from
scratch. The cached weights are exact (safetensors), roughly the size of the
encoder per application. Heads trained incrementally sit on an encoder that
was fine-tuned for the application's earlier expressions; run `--full` before
a release when that matters.

Training options pass through unchanged: `--cases`, `--epochs`, `--batch-size`,
`--learning-rate`, `--max-sequence-length`, `--evaluation-ratio`, `--seed`,
`--device`, `--head-architecture`, `--select-best-epoch`, `--encoder-name`,
`--encoder-revision`, `--local-files-only`, `--ece-threshold`,
`--max-constraint-violation-rate`, `--counterfactual-ratio`,
`--adapter-bottleneck-size`, `--no-cache`, `--full`, `--application-id`,
`--application-version`, `--compiler-version` and `--cache-dir` (default
`.semantscript/cache`). The interpreter is `--python`,
then `SEMANTSCRIPT_PYTHON`, then `python3`; inside this repository the trainer
and model sources (and `.python-packages` when present) are put on `PYTHONPATH`
automatically, and the caller's `PYTHONPATH` is kept after them.

## test

Reads the artifact pointer and manifest and reports, per function, the
verification it shipped with: status, accuracy, ECE, Brier, pair consistency,
attested cases, constraint violations and each head's accuracy. Trained models
are not persisted, so this is the gate `train` already enforced; with
`--bundle` the command also loads the artifact and replays every IR example
through the runtime, comparing outputs by value (diagnostic functions by their
`value`). Any function whose status is not `passed`, any bundle function absent
from the artifact and any example mismatch make the command exit with status 1.
`--json` prints the same as one document.

## run

Loads the artifact, which activates the runtime the compiled `__sema` calls
resolve against, then imports the module. With `--call <export>` it calls that
export with the JSON input (`--input '[1, 2]'` spreads an array as positional
arguments; any other JSON value is the single argument; `--input-file` reads
the same from a file), awaits a returned promise and prints the JSON result.
The module resolves `@semantscript/core` from its own location, and that must
be the same installation the CLI uses. The artifact is closed when the command
returns, so nothing stays loaded in the process.
