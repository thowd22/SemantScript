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

Training options pass through unchanged: `--cases`, `--epochs`, `--batch-size`,
`--learning-rate`, `--max-sequence-length`, `--evaluation-ratio`, `--seed`,
`--device`, `--head-architecture`, `--select-best-epoch`, `--encoder-name`,
`--encoder-revision`, `--local-files-only`, `--ece-threshold`,
`--max-constraint-violation-rate`, `--counterfactual-ratio`,
`--application-id`, `--application-version`, `--compiler-version` and
`--cache-dir` (default `.semantscript/cache`). The interpreter is `--python`,
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
