# CLI reference

`semantscript` is one entry point over the compiler, the Python trainer and
the runtime. This page lists every command, flag, default and exit code as
the shipped source defines them; the [CLI guide](../cli/README.md) explains
the behavior in prose.

```text
semantscript init  [--tool next|vite|esbuild|tsc] [--no-example] [--no-doctor]
                   [--teacher anthropic|openrouter|ollama|constraints] [--teacher-model <id>]
                   [--python <exe>] [--trainer-module <module>]
semantscript doctor [--python <exe>] [--teacher <teacher.toml>|constraints] [--probe request|free|none]
                    [--device auto|cpu|cuda] [--trainer-module <module>] [--no-teacher]
                    [--runtime] [--json]
semantscript build [--project tsconfig.json] [--application <id>] [--bundle <path>]
                   [--route-domains] [--domain-depth <name>=<layers>]...
semantscript train [--bundle <path>] [--artifact <root>] [--teacher <teacher.toml>|constraints]
                   [--estimate] [--max-cost-usd <x>] [options]
semantscript teacher probe [--teacher <teacher.toml>|constraints] [--python <exe>] [--json]
semantscript dev   [build and train options] [--debounce <ms>] [--once]
semantscript test  [--artifact <root>] [--bundle <path>] [--json]
semantscript run   [--artifact <root>] <module.js> [--call <export>] [--input <json> | --input-file <path>]
semantscript releases [list] [--artifact <root>] [--json]
semantscript releases show <release> [--artifact <root>] [--json]
semantscript releases rollback [<release>] [--artifact <root>] [--dry-run] [--json]
semantscript releases promote <release> [--artifact <root>] [--dry-run] [--json]
semantscript releases prune [--keep <n>] [--older-than <duration>] [--artifact <root>] [--dry-run] [--json]
semantscript releases derive --int8 [<release>] [--artifact <root>] [--cache-dir <dir>] [--per-channel]
                     [--reduce-range] [--weight-type int8|uint8] [--max-attested-disagreements <n>]
                     [--max-decision-change-rate <x>] [--ece-threshold <x>] [--promote] [--json]
semantscript explain [--artifact <root>] [--bundle <path>] [--cache-dir <dir>] [--neighbors <n>] [--json]
                     <module.js> --call <export> [--input <json> | --input-file <path>]
semantscript package [--project <dir>] [--dist <dir>] [--artifact <root>] [--out <dir>] [--include <path>]...
                     [--target lambda-zip|lambda-image|cloud-run-functions | --max-bytes <n>]
                     [--platform <os>] [--arch <cpu>] [--force] [--json]
semantscript --version
```

`semantscript --version` (also `-v` or `version`) prints `semantscript`
and the release version every package shares, and exits 0. The trainer's
console script answers the same way: `semantscript-trainer --version`.

## Exit codes

| Code | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0    | The command succeeded. `semantscript help`, `--help` and `-h` (also after a command, as in `semantscript doctor --help`) print the usage and exit 0.                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 1    | The command ran and failed: compile diagnostics, a failed training or verification, a failed `test`, a failed `doctor` check or `train` preflight, a `train` stopped by its `--max-cost-usd` cap, a failed `teacher probe` request, an unknown `--call` export, a refused `releases` switch or prune, a `package` that failed or exceeds its target (the message starts with its code, below), an `explain` whose call reached a function id the artifact lacks or failed with anything but a missing id or a below-threshold confidence, or any other error while working. |
| 2    | Usage: no command, an unknown command, an unknown or malformed option, a missing required value, or an inconsistent combination (`--input` with `--input-file`), and `explain` without `--call`.                                                                                                                                                                                                                                                                                                                                                                            |

`releases derive --int8` also exits 2 when its gate refuses the int8
release ([below](#releases-derive---int8)): the run worked and published
nothing. Usage errors print the message and the usage text to stderr. Every other
failure prints its message to stderr; compile diagnostics carry the file, line,
column, code and site text.

## Defaults and environment

Every command runs without flags in an initialised project:

| Setting            | Resolution                                                                                                                                                                                                                                                                                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Project            | `--project`, else `tsconfig.json` in the working directory.                                                                                                                                                                                                                                                                                                               |
| Bundle             | `--bundle`, else `semantscript.ir.v1.json` under the tsconfig `outDir`, then `.`, `dist`, `out`, `build`.                                                                                                                                                                                                                                                                 |
| Artifact root      | `--artifact`, else `SEMANTSCRIPT_ARTIFACT`, else `.semantscript/artifact`. The runtime's `loadSemaArtifact()` with no path resolves the same way, searching upward from the compiled entry script and then from the working directory.                                                                                                                                    |
| Teacher            | `--teacher` (a file, or the keyword `constraints` for the built-in constraints teacher when no file of that name exists), else the first of `semantscript.teacher.toml`, `teacher.toml`, `.semantscript/teacher.toml`; else, when `ANTHROPIC_API_KEY` is set, `train` writes `.semantscript/teacher.toml` for `claude-sonnet-5` and uses it. The key never enters a file. |
| Python interpreter | `--python`, else `SEMANTSCRIPT_PYTHON`, else `python3` (`python` on Windows). Inside this repository the trainer and model sources (and `.python-packages` when present) are prepended to `PYTHONPATH`.                                                                                                                                                                   |
| npm (`package`)    | `SEMANTSCRIPT_NPM`, else `npm` on the `PATH`. `package` installs the bundle's production dependencies with it: `npm ci --omit=dev` from `package-lock.json` when every dependency comes from a registry, else `npm install --omit=dev --install-links --no-package-lock`, which resolves registry versions afresh rather than from the lockfile.                          |
| Cache directory    | `--cache-dir`, else `.semantscript/cache`.                                                                                                                                                                                                                                                                                                                                |

## `init`

Wires the compiler into an existing project and adds a starter expression.
In a new project (no `tsconfig.json` and no other build tool detected, for
example a directory holding only the `package.json` that `npm install`
wrote, or `--tool tsc` with no `tsconfig.json`), it first starts a
TypeScript project: `tsconfig.json` (NodeNext modules, `src/` compiled to
`dist/`), `src/`, a `build` script `tspc -p tsconfig.json` and `typescript`
in `devDependencies` (each only when absent), and `"type": "module"` when no
code depends on the module type yet: no existing `main` file, no scripts
beyond `npm init`'s placeholder and no `.js` or `.cjs` file in the root or
`src/`. In that case a `type` of `"commonjs"` (which npm 11's `npm init -y`
writes) becomes `"module"` too, and init says so; otherwise `type` is left
alone. Run in a directory without `package.json`, `init` exits 1 and asks
for `npm init -y` first.

| Flag           | Value                            | Effect                                                                                                                                                                                                          |
| -------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--tool`       | `next`, `vite`, `esbuild`, `tsc` | Overrides detection. Detection order: a `next.config.*` or `next` dependency; a `vite.config.*` or `vite` dependency; `esbuild` in the dependencies or a build script; `tsconfig.json`; else a new tsc project. |
| `--no-example` |                                  | Do not write the starter `.sem.ts` (`src/hello.sem.ts`, or `lib/hello.sem.ts` for Next.js). A project that already has a `.sem.ts` file gets none either.                                                       |

What it writes: the adapter entry for the detected tool (see the
[build tool pages](build-tools/tsc.md)), the editor plugin entry
`{ "name": "@semantscript/compiler/ts-plugin" }` in `tsconfig.json`,
`@semantscript/core` and `@semantscript/compiler` in `package.json`,
`.semantscript/.gitignore` reserving `artifact/`, `cache/`, `package/` and
`.package-staging-*/` (on a project initialised earlier it appends whichever
are missing), and the starter
expression. A config it cannot edit safely (missing, unparsable,
`require()`-based, or a `next.config` that already sets `turbopack`,
`serverExternalPackages` or `outputFileTracingIncludes`) is reported as
`manual` with the snippet to add. Edits are idempotent; running `init` twice
changes nothing. Exit 2 for an unknown `--tool`, a project root without
`package.json`, or a `package.json` that is not a JSON object.

`init` ends with the `doctor` checks under `environment (semantscript doctor):`
(with `--probe free`, so no billed teacher request), then says whether the
environment is ready for `train` or how many checks failed. The checks never
change its exit status: the wiring succeeded either way, and a doctor that
cannot run is reported as a line, not an error.

| Flag               | Value                                              | Effect                                                                                                                                                                       |
| ------------------ | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--teacher`        | `anthropic`, `openrouter`, `ollama`, `constraints` | Write `.semantscript/teacher.toml` for that teacher (below). Without the flag, `init` asks on an interactive terminal when no teacher file exists; elsewhere it writes none. |
| `--teacher-model`  | id                                                 | The model the teacher file names (defaults: `claude-sonnet-5`, `anthropic/claude-sonnet-5`, `qwen3:14b`); not for `constraints`.                                             |
| `--no-doctor`      |                                                    | Skip the environment checks.                                                                                                                                                 |
| `--python`         | exe                                                | The interpreter the checks use (default above).                                                                                                                              |
| `--trainer-module` | module                                             | The trainer module the checks run; for tests and forks.                                                                                                                      |

The teacher file never holds a key. `anthropic` writes the Anthropic backend
with `mode = "auto"`; `openrouter` writes the Anthropic backend pointed at
`https://openrouter.ai/api` with `anthropic/claude-sonnet-5` and
`mode = "direct"`, and a comment saying to run with
`ANTHROPIC_API_KEY="$OPENROUTER_API_KEY"`; `ollama` writes the local backend;
`constraints` writes the built-in constraints teacher. An existing
`semantscript.teacher.toml`, `teacher.toml` or `.semantscript/teacher.toml`
is never replaced (the line says `unchanged`), and the closing checks run
against the teacher file `init` wrote or found. Exit 2 for an unknown
`--teacher`.

## `doctor`

Checks everything a build needs before it runs and prints one line per check:
the status (`pass`, `fail`, `warn` or `skip`), the check id and what it found,
with a `fix:` line under every check that did not pass, then the totals. The
[environment guide](environment.md) explains every check and records runs.

| Check              | What it checks                                                                                                                                                                                                                   |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `node`             | The running Node release is 22.13 or later (package.json `engines`).                                                                                                                                                             |
| `runtime-bindings` | `onnxruntime-node` and `tokenizers`, resolved from `@semantscript/core` like the runtime does, load their native binaries on this platform and architecture.                                                                     |
| `python`           | The interpreter the CLI will use (default above) starts and is Python 3.12 or later.                                                                                                                                             |
| `trainer`          | `semantscript_trainer` imports, with its version and location, and its teacher clients `anthropic` and `openai`.                                                                                                                 |
| `model`            | `semantscript_model` imports, with its version and location.                                                                                                                                                                     |
| `torch`            | PyTorch and Transformers import; the build (CUDA, ROCm or CPU-only).                                                                                                                                                             |
| `device`           | The device training runs on: the CUDA or ROCm GPU with its total and free memory, or the CPU with the machine's RAM (a warning). Apple MPS is reported, not used.                                                                |
| `onnxruntime`      | ONNX Runtime and ONNX import (the export and its parity check need both).                                                                                                                                                        |
| `platform-env`     | Environment variables the run needs: `PYTHONNOUSERSITE=1` when user-site packages break the imports, `HSA_ENABLE_DXG_DETECTION=1` when ROCm on WSL2 finds the GPU only with it.                                                  |
| `teacher-config`   | The teacher file `train` would use (default above) exists and is a valid `[teacher]` table, or `--teacher constraints` names the built-in constraints teacher.                                                                   |
| `teacher-key`      | The key is in the environment (`ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`, also for OpenRouter's Anthropic route); Ollama and the constraints teacher need none (with a `[teacher.fallback]`, the fallback's key is checked). |
| `teacher-probe`    | One minimal request to the teacher succeeded, with its latency and tokens; skipped for the constraints teacher, which sends none (its fallback is probed).                                                                       |

| Flag               | Value                     | Effect                                                                                                                                                                                            |
| ------------------ | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--python`         | exe                       | The interpreter to check (default above).                                                                                                                                                         |
| `--teacher`        | path or `constraints`     | The teacher file to check, or `constraints` for the built-in constraints teacher (default above; with no file and `ANTHROPIC_API_KEY` set, the default Anthropic teacher).                        |
| `--probe`          | `request`, `free`, `none` | `request` (default) sends one request: 8 output tokens with thinking off, well under USD 0.001 on Sonnet. `free` sends nothing billed (for Ollama it lists the server's models). `none` skips it. |
| `--device`         | `auto`, `cpu`, `cuda`     | The device `train` will be asked for; `cuda` with no GPU fails, `cpu` passes.                                                                                                                     |
| `--no-teacher`     |                           | Skip the three teacher checks.                                                                                                                                                                    |
| `--runtime`        |                           | Only `node` and `runtime-bindings`: a machine that serves artifacts and never trains.                                                                                                             |
| `--json`           |                           | Print the report as JSON (`kind` `semantscript.doctor-report`, `reportVersion` 1, `checks` with `id`, `status`, `summary`, `fix`).                                                                |
| `--trainer-module` | module                    | The Python module whose `doctor` subcommand runs (default `semantscript_trainer.cli`).                                                                                                            |

When the interpreter was not chosen (no `--python`, no `SEMANTSCRIPT_PYTHON`)
and a `.venv` exists in the working directory or one of its parents, every
failed `python`, `trainer`, `model`, `torch` or `onnxruntime` line also names
that venv's interpreter as the fix (`--python .venv/bin/python` or
`SEMANTSCRIPT_PYTHON`): the CLI does not pick a venv up unless it is activated.
An interpreter older than 3.12 fails `python` even when the trainer cannot
import at all (its sources use Python 3.12 syntax).

Exit 0 when no check failed (warnings and skips included), 1 when any check
failed or the Python side printed a report that breaks the contract, 2 for an
unknown `--probe` value.

## `build`

Compiles every `sema` site in the project to a runtime call and writes one IR
bundle with the execution plan.

| Flag              | Value                         | Effect                                                                                                                                                                                 |
| ----------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--project`, `-p` | path                          | The `tsconfig.json` to compile (default `tsconfig.json`).                                                                                                                              |
| `--application`   | id                            | Names the artifact's encoder and adapter refs (`encoder.<id>`, `adapter.<id>`); default `application`.                                                                                 |
| `--bundle`        | path                          | Where the IR bundle is written; default `semantscript.ir.v1.json` in the tsconfig `outDir`.                                                                                            |
| `--route-domains` |                               | One adapter per domain (the file stem or enclosing controller class by default, or an `@domain` header) even without depths.                                                           |
| `--domain-depth`  | `<name>=<layers>`, repeatable | Runs the named domain on the first `layers` encoder layers before its adapter and makes the bundle routed. `layers` is a positive integer; a name must match a domain the build finds. |

Compile diagnostics exit 1 and nothing is written; a successful build emits the
JavaScript through TypeScript, then the bundle last. Exit 2 for a malformed
`--domain-depth` value. A missing `tsconfig.json` ends with
`next: create <tsconfig.json> with npx -p typescript tsc --init --rootDir . --outDir dist, then run semantscript init to add the SemantScript plugins, or pass --project <tsconfig.json>`
(`init` wires an existing `tsconfig.json` but does not create one), a
`tsconfig.json` that does not parse or selects no inputs with the file to
correct, and a project without `compilerOptions.outDir` with the setting to
add.

## `train`

Runs the Python bundle driver: synthetic and adversarial datasets from the
teacher, training over one shared encoder, calibration, verification, and the
export of an immutable release into the artifact root. Prints one progress
line per expression and a report table; exit 1 when any function fails
verification or the driver fails.

Before the driver starts, `train` runs the `doctor` Python and teacher checks
as a preflight (about 5 seconds, most of it importing PyTorch; no billed
teacher request, though an Ollama server is asked for its model list). It
prints them to stderr and, when any fails, exits 1 without starting the
trainer, so a missing interpreter, package, key or model shows in seconds
instead of minutes into a run.

The driver then stops, before generating anything, when an expression has no
gold example (verification needs at least one attested example per
expression) and names it by source line; with `--teacher constraints` it also
stops on the first sampled input the expression's constraints do not decide,
printing that input and the outputs the constraints admit (see
[teachers](teachers.md)).

Every verification failure in the report ends with a `next:` line: the
constraint to add when the missed gold examples share a one-field rule, an
example to add for a repeated miss, more `--cases` or `--epochs` for a
calibration failure, or the seed retry to run (`--seed-attempts`, or the next
untried `--seed`). The suggestion is derived from the failing cases; the
[diagnostics catalogue](diagnostics.md#verification-failures) says how.

```text
nf_2c6b0e41… verification failures:
  ECE 0.242017778048 exceeds configured threshold 0.1
    next: rerun with --cases 96 (now 48): ECE 0.2420 is measured on 30 calibration rows, and more cases give more of them
```

When the trainer process or its interpreter fails (a package the trainer
imports is missing, the interpreter is too old, any uncaught exception),
`train` prints the lines the trainer wrote before the traceback, then one line
naming the `doctor` check that fixes it, and writes the traceback next to the
report (`<artifact>.report.traceback.txt` by default):

```text
semantscript train: the trainer stopped: ModuleNotFoundError: No module named 'torch'; next: run semantscript doctor and fix its torch check: python3 cannot import torch; full traceback in /srv/app/.semantscript/artifact.report.traceback.txt
```

The printed `doctor` command carries this run's `--python`,
`--trainer-module` and `--teacher` when they were passed. A traceback the
trainer goes on from (a logged warning, Python's shutdown noise) is printed in
place, and only one that ends the run without a report becomes the line. Every
failure the trainer catches itself ends its `error:` line with
`; next: <fix>`: a training package that does not import (its doctor check),
a missing or malformed bundle (`semantscript build`), a broken teacher file
(the `teacher-config` check), an unreachable teacher, an option out of range,
a lack of memory (`--batch-size` or `--device cpu`), an unwritable path, a
failed export, answers the teacher got wrong, an expression's examples or
constraints, and a last-resort fix for anything else. A trainer the operating
system kills (the out-of-memory killer, a native crash) gives
`semantscript train: the trainer was killed by <signal>; next: …` and exit
128 plus the signal number. `train` renders only a report this run wrote,
never one an earlier run left at the path. An interpreter that does not exist
gives
`unable to run <python>: … ENOENT; next: run semantscript doctor --python <python> and fix its python check: …`.
`--estimate` wraps its failures the same way.

| Flag               | Value  | Effect                                                                                                                                                                                                                                                           |
| ------------------ | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--bundle`         | path   | The IR bundle (default above).                                                                                                                                                                                                                                   |
| `--artifact`       | path   | The artifact root to publish into (default above).                                                                                                                                                                                                               |
| `--teacher`        | path   | A TOML file with a `[teacher]` table (`backend = "anthropic"`, `"ollama"` or `"constraints"`), or the keyword `constraints` for the built-in constraints teacher (labels from the expressions' own constraints, no language model); see [teachers](teachers.md). |
| `--cache-dir`      | path   | The build cache (default `.semantscript/cache`); see the [build cache](build-cache.md) page.                                                                                                                                                                     |
| `--report`         | path   | Also write the JSON report here.                                                                                                                                                                                                                                 |
| `--python`         | exe    | The interpreter to run the trainer with.                                                                                                                                                                                                                         |
| `--trainer-module` | module | The Python module to invoke (default `semantscript_trainer.cli`); for tests and forks.                                                                                                                                                                           |
| `--no-preflight`   |        | Skip the environment preflight.                                                                                                                                                                                                                                  |
| `--estimate`       |        | Print what the teacher would cost and exit 0 without the preflight, a teacher request or any training (below).                                                                                                                                                   |
| `--max-cost-usd`   | USD    | Stop before the teacher request that would take the run past this many dollars (below). Exit 2 unless it is a positive number.                                                                                                                                   |

`--estimate` prints, per expression and in total, the teacher requests the
run would send (the expected count and the maximum the generators allow), the
input tokens (and how many of them the prompt cache serves), the output
tokens, the USD cost at the configured backend's price and the wall time
(for a Message Batch, one hour per batch, with the maximum the run can wait in
parentheses). An
expression whose datasets are cached costs nothing; the constraints teacher
and Ollama cost nothing. Tokens are the characters of the exact prompts
divided by a calibrated ratio (2.1 characters per token), and the price comes
from OpenRouter's public price list, the pinned Anthropic list prices or a
`[teacher.pricing]` table ([teachers](teachers.md#cost-estimate-and-spend-cap)).
With `--max-cost-usd` it also says whether the cap covers the expected and
the maximum cost.

With `--max-cost-usd`, every request is reserved against the cap before it
is sent, at more than it can be expected to cost; the request that would pass
the cap is not sent, and the run exits 1 naming the cap, the spend so far and
where the paid work is kept. Every dataset finished before the stop stays
cached and every paid response is kept in the response journal
([build cache](build-cache.md)), so a rerun with a higher cap (or none)
replays them at no cost and continues. Every run prints a running
`teacher: <n> request(s), <tokens>, USD <x>` line every 25 requests and after
each expression, and the report table ends with
`teacher: <n> requests (<m> replayed), USD <x> of the USD <cap> cap`.

Options handed to the trainer unchanged:

| Flag                                                              | Value    | Meaning                                                                                                                                                       |
| ----------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--cases`                                                         | integer  | Synthetic cases generated per expression (gold examples count toward it).                                                                                     |
| `--epochs`, `--batch-size`, `--learning-rate`, `--seed`           | numbers  | The fine-tuning recipe.                                                                                                                                       |
| `--max-sequence-length`                                           | integer  | Tokens per canonical input (default 512).                                                                                                                     |
| `--evaluation-ratio`                                              | fraction | Held-out calibration split (default 0.2).                                                                                                                     |
| `--device`                                                        | name     | `auto`, `cpu`, `cuda`, …                                                                                                                                      |
| `--head-architecture`                                             | name     | `linear` (default) or `mlp`.                                                                                                                                  |
| `--select-best-epoch`                                             |          | Keep the epoch with the best held-out accuracy instead of the last.                                                                                           |
| `--encoder-name`, `--encoder-revision`                            | strings  | The Hugging Face checkpoint and pinned commit every function of the application fine-tunes from.                                                              |
| `--local-files-only`                                              |          | Never download; fail if the checkpoint is not cached.                                                                                                         |
| `--ece-threshold`                                                 | fraction | Verification gate on calibration error (default 0.1).                                                                                                         |
| `--max-constraint-violation-rate`                                 | fraction | Share of raw predictions allowed to violate an active constraint (default 0; the value used is recorded).                                                     |
| `--seed-attempts`                                                 | integer  | Training runs a narrowly failed release gate may use, the first included (default 3, at most 20; 1 turns the seed retry off).                                 |
| `--seed-retry-margin`                                             | factor   | How far past its gate a failure may be and still retry: the violation rate and the ECE each within this many times their gate (default 2, from 1 through 10). |
| `--counterfactual-ratio`                                          | fraction | Share of synthetic cases that get a counterfactual twin (default 1).                                                                                          |
| `--adapter-bottleneck-size`                                       | integer  | Width of the per-domain adapter.                                                                                                                              |
| `--application-id`, `--application-version`, `--compiler-version` | strings  | Recorded in the manifest; `--compiler-version` defaults to the installed `@semantscript/compiler` version.                                                    |
| `--no-cache`                                                      |          | Ignore the build cache and write nothing to it.                                                                                                               |
| `--full`                                                          |          | Retrain every function jointly, discarding cached function records.                                                                                           |

### Seed retry

A training seed can decide a near-threshold release gate: the same datasets
can fail at one seed and pass at the next. The seed drives training noise
(initialisation and batch order) and also draws the held-out calibration and
evaluation split, so each attempt's accuracy and ECE are measured on its own
split; the violation rate is measured over every record and does not depend
on the split. When verification fails only on
the constraint-violation rate or the ECE, and by no more than the margin,
`train` retrains with the next seed (`--seed`, then `--seed` + 1, …) up to
`--seed-attempts` runs in all. "By no more than the margin" means the
violation rate is at most `--seed-retry-margin` times
`--max-constraint-violation-rate` and the ECE at most that many times
`--ece-threshold`. The datasets are generated once, before the first attempt,
so a retry calls no teacher and costs training time only. A retry that
passes publishes as usual. The release manifest records the seed that passed
for each expression (`functions[].trainingProvenance.seed`, which
`semantscript test` prints in its `seed` column), as does the verified IR in
the build cache and the report (`seed`, `functions[].training.seed`), so a
`--no-cache` build keeps it too. The first seed that passes publishes, even
when an earlier attempt that failed was more accurate: compare the
`accuracy` column of the attempts table. Each retry is logged as
`verification failed narrowly at seed <n> (<why>); retrying with seed <n+1> (attempt <k> of <K>) …`.

The build does not retry, and says why on a `not retrying: <reason>` line
and in the report's `retry.stopReason`, when a gold or human example is
missed (a gold miss is not a seed effect), when an output type check fails,
when the rate or the ECE is outside the margin, or when the attempts run
out. Under the default zero violation tolerance any violation is outside
the margin (twice zero is zero), so only a narrow ECE miss retries there. A
build that relaxes the tolerance, like the
[Express example](../examples/express-app/README.md) at 1%, retries a rate
up to 2%. With several expressions, one expression that fails outside the
margin stops the build; on an incremental build only the changed
expressions retrain, and the reused ones keep their recorded seeds.

The report lists every attempt (`attempts[]`: attempt number, seed, status
and, per expression, accuracy, ECE, violations, records, violation rate and
failures), the published seed (`seed`) and the retry settings
(`retry.attempts`, `retry.margin`, `retry.stopReason`); the rendered report
prints the attempts as a table when more than one ran.

## `teacher probe`

Sends one small request through the teacher `train` would use (`--teacher`,
else the teacher file lookup above) and prints the backend, the model and the
host it goes through, the latency, the input and output tokens and the cost
with its price source. The request is one short user message with
`max_tokens` 8 and thinking disabled: 16 input and 4 output tokens, USD
0.000072 on Sonnet 5 through OpenRouter; nothing through Ollama. The
constraints teacher sends no request and reports USD 0; a constraints teacher
with a `[teacher.fallback]` probes the fallback. The key comes from the
environment and is scrubbed from any error. Exit 1 when the request fails
(with the fix), 2 when there is no teacher to probe.

| Flag               | Value  | Effect                                                              |
| ------------------ | ------ | ------------------------------------------------------------------- |
| `--teacher`        | path   | The teacher file, or `constraints`.                                 |
| `--python`         | exe    | The interpreter (default above).                                    |
| `--cache-dir`      | path   | Where the fetched OpenRouter price list is cached for a day.        |
| `--json`           |        | Print the `semantscript.teacher-probe` result instead of the lines. |
| `--trainer-module` | module | The trainer module to invoke; for tests and forks.                  |

## `dev`

`build` then `train`, then watch the project's TypeScript sources and repeat
both on every save. The `train` preflight runs until one training succeeds,
not on every save. Each cycle's `train` applies the [seed retry](#seed-retry)
too, so a narrow miss can retrain up to `--seed-attempts` times before the
cycle ends. Takes every `build` and `train` option plus:

| Flag         | Value        | Effect                                                                  |
| ------------ | ------------ | ----------------------------------------------------------------------- |
| `--debounce` | milliseconds | Wait after the last save before a cycle (default 300).                  |
| `--once`     |              | One cycle, then exit with the train status; what scripts and tests use. |

Watched: `.ts`, `.mts`, `.cts`, `.tsx` and `tsconfig.json` under the project
root, ignoring `node_modules`, `.git`, `.semantscript` and declaration files. A
save during a cycle queues exactly one more. Ctrl-C stops after the current
cycle; exit 0 on interruption, 1 when a `--once` cycle fails.

## `test`

Reads the artifact pointer and manifest and reports every function's shipped
verification (status, accuracy, ECE, Brier, pair consistency, attested cases,
constraint violations, the training seed the release passed at, each head's
accuracy). The seed is `-` in the table and `null` in `--json` for a release
built before the manifest recorded seeds.

| Flag         | Value | Effect                                                                                                                        |
| ------------ | ----- | ----------------------------------------------------------------------------------------------------------------------------- |
| `--artifact` | path  | The artifact root (default above).                                                                                            |
| `--bundle`   | path  | Also load the artifact and replay every IR example through the runtime, comparing by value (diagnostic functions by `value`). |
| `--json`     |       | Print one JSON document instead of the table.                                                                                 |

Exit 1 when any function's status is not `passed`, any bundle function is
absent from the artifact, or any example mismatches. An absent function and a
mismatch each add a `next:` line (retrain on this bundle; rebuild, retrain
and rerun), and so does a function that did not pass (retrain, or roll back),
which `--json` lists as `next`. With nothing published at the artifact root
the command exits 1 with
`no artifact at <root> (current.json is missing); next: run semantscript train …`;
a pointer or release that does not read ends with
`next: run semantscript releases list to find an intact release, then switch to it with semantscript releases rollback <release>, …`,
the same fix `run` prints for that artifact.

## `run`

Loads the artifact, imports the module (so its compiled `__sema` calls
resolve), optionally calls one export and prints the JSON result, then closes
the artifact.

| Flag           | Value  | Effect                                                                                                            |
| -------------- | ------ | ----------------------------------------------------------------------------------------------------------------- |
| `--artifact`   | path   | The artifact root (default above).                                                                                |
| `--call`       | export | The function export to call. Without it the module is only imported (its top level runs) and the command exits 0. |
| `--input`      | JSON   | The arguments: a JSON array is spread as positional arguments, any other JSON value is the single argument.       |
| `--input-file` | path   | The same, read from a file. Exclusive with `--input`.                                                             |

Exactly one module path is required. The module resolves `@semantscript/core`
from its own location, and that must be the installation the CLI uses. Exit 1
when the export is missing or not a function; exit 2 for a missing module
path, both input flags, or input that is not JSON. A missing export's message
ends with `next:` and the module's function exports; an artifact that does not
load ends with the runtime's `next:` clause
([runtime errors](diagnostics.md#runtime-errors)).

## `releases`

Manages the immutable releases under the artifact root without loading any
model. Every `train` publishes `releases/sha256-<manifest digest>/` and points
`current.json` at it; these subcommands show what is there, point
`current.json` at another release, and remove old ones. A `<release>` is a
full manifest digest, `sha256-<digest>`, `releases/sha256-<digest>` or a
unique prefix of at least 7 hex characters. Only directories named
`sha256-<64 hex>` count as releases; the exporter's `.staging-*` directories
and anything else are ignored. "Newest" and "previous" order releases by the
manifest's `build.createdAt` (then by digest), since the exporter keeps no
deployment history.

| Subcommand                  | Effect                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `list` (the default)        | One row per release, newest first: `*` on the one `current.json` names, `build.createdAt`, the first 12 characters of the manifest digest, `application@version`, functions passed out of total, the lowest accuracy, the highest ECE and the summed constraint violations. A release whose manifest does not hash to its name is listed as `invalid` with the reason. `list` and `show` check only the manifest's digest; `rollback` and `promote` check the rest before switching.                                                                                                                                                                                                                        |
| `show <release>`            | That release's per-function verification table, the same columns as `test`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `rollback [<release>]`      | Points `current.json` at the named release or, without one, at the newest release created before the current one. The target is checked first the way the runtime loads it: the manifest hashes to the directory name, every resource is a regular non-symlink file inside the release with its recorded size and SHA-256, every function's verification is `passed`, and the runtime's own loader (`checkSemaArtifact` from `@semantscript/core`, with its default options) accepts it: manifest schema, runtime and model ABI compatibility, tensor names and shapes, opsets and the encoder-adapter-head chain. Only ONNX session start-up and the application's fallback registrations are not checked. |
| `promote <release>`         | The same operation with a required name, for rolling forward again.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `derive --int8 [<release>]` | Derives an int8 release from the current release or the named one and publishes it beside it without moving `current.json` ([below](#releases-derive---int8)).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `prune`                     | Removes releases that are neither current nor kept: with `--keep <n>`, the ones beyond the `n` newest valid releases; with `--older-than <duration>` (`30d`, `12h`, `90m`), the ones created before that; with both, only those matching both. Needs at least one of them. Newest means by `build.createdAt`, not by deployment: after a rollback, `--keep` still counts the newer releases you rolled back from.                                                                                                                                                                                                                                                                                           |

| Flag           | Value    | Effect                                                                                                                     |
| -------------- | -------- | -------------------------------------------------------------------------------------------------------------------------- |
| `--artifact`   | path     | The artifact root (default above).                                                                                         |
| `--json`       |          | Print one JSON document (`kind` `semantscript.releases`, version 1, and `.rollback`, `.promote`, `.prune` for the others). |
| `--dry-run`    |          | `rollback`, `promote`, `prune`: check and report, change nothing.                                                          |
| `--keep`       | count    | `prune`: keep the `n` newest valid releases (`0` keeps only the current one).                                              |
| `--older-than` | duration | `prune`: remove only releases whose `build.createdAt` is older than this.                                                  |

The pointer rewrite is atomic: the new `current.json` (the exact bytes the
trainer's exporter writes) goes to an exclusive temporary file beside it, is
fsynced and renamed over the old one, and the directory is fsynced, so a
reader sees the old pointer or the new one. No release is removed or modified.
A process that loaded the root with `loadSemaArtifact(root, { watch: true })`
swaps to the new release as soon as it loads whole (and keeps the old one in
service if it does not); other processes load it on their next start.

`prune` never removes the release `current.json` names, re-reads the pointer
before each removal (a `train` or `rollback` running at the same time can make
a candidate current), never removes an invalid release or a `.staging-*`
directory, and renames each release to `releases/.pruning-<random>` before
deleting it, so an interrupted prune never leaves a half-deleted directory that
looks like a release (the next prune deletes such leftovers). After the rename
it reads the pointer again and puts the release back if a `rollback` made it
current meanwhile. It refuses to run when `current.json` is missing or invalid
or names a release that is missing or invalid.

| Code                      | Cause                                                                                                                                                                                                                                                                   |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POINTER_INVALID`         | `current.json` is missing (for `prune` or an unnamed `rollback`), not a regular file, or not a v1 pointer whose release is `releases/sha256-<manifestSha256>`, or (for `prune`) names a release that is missing or invalid. A named `rollback` or `promote` repairs it. |
| `RELEASE_NOT_FOUND`       | No release matches the name or prefix.                                                                                                                                                                                                                                  |
| `RELEASE_AMBIGUOUS`       | The prefix matches more than one release.                                                                                                                                                                                                                               |
| `RELEASE_ALREADY_CURRENT` | The target is the current release.                                                                                                                                                                                                                                      |
| `RELEASE_NO_PREVIOUS`     | An unnamed `rollback` found no valid release created before the current one, or the current one is missing or invalid.                                                                                                                                                  |
| `RELEASE_INTEGRITY`       | The target is a symlink, its manifest does not hash to its name, or a resource is missing, crosses a symlink or has the wrong size or digest.                                                                                                                           |
| `RELEASE_UNVERIFIED`      | A function in the target did not pass verification; the runtime refuses such a release.                                                                                                                                                                                 |
| `RELEASE_REJECTED`        | The runtime's loader refuses the target; the message carries its `SEMA_ARTIFACT_*` code, for example `SEMA_ARTIFACT_INCOMPATIBLE` for a release built for another runtime or model ABI.                                                                                 |
| `RELEASE_INVALID`         | `show` named a release whose manifest cannot be read.                                                                                                                                                                                                                   |

`list` adds a line under the table for each release `derive --int8` produced
(`<digest>: int8-dynamic from sha256-<source>: 0 of 586 decisions changed (0
of 6 attested); tolerance …`), `show` prints the same as its `derived` line,
and `--json` gives every release a `derivation` object (`null` for a trained
one): precision, source digest, quantization settings, tolerances and the
recorded figures, which are `null` for an int8 release derived before the
manifest recorded them.

### `releases derive --int8`

Runs `python -m semantscript_trainer.cli derive-int8` (the interpreter and
module resolve as for `train`). It quantizes every encoder graph of the
release (a depth-routed release has one per depth) with ONNX Runtime's
dynamic int8 quantization, copies every other resource unchanged, and then
runs each record through the float32 chain and the int8 chain of its
function: encoder, adapter and heads. A decision is the tuple of the heads'
answers. The records are the release's own, taken from the build cache by
digest:

- the training dataset whose file SHA-256 is the function's
  `trainingProvenance.datasetSha256` (its gold cases, the sema call's
  examples, are the attested records);
- the adversarial dataset the build cache's `function.json` pins, or, with no
  such record, the only one built on that training dataset.

A missing dataset, or two adversarial datasets that could both be the one,
stops the derivation with `int8-records-missing` in the
[diagnostics catalogue](diagnostics.md) instead of verifying on other
records. A held-out set joins the check once the release gate records one;
until then the report says the check ran on the training, gold and
adversarial records only.

The gate refuses the release when more attested records changed decision
than `--max-attested-disagreements` allows (default 0), when the share of all
records that changed is above `--max-decision-change-rate` (default 0), or
when a head's int8 ECE is above `--ece-threshold` (default 0.1). A refusal
publishes nothing, prints the figures and the tolerance flags that would
admit them (remedy `int8-gate-refused`) and exits 2. A release that passes
is published as `releases/sha256-<digest>/` beside its source, and each
quantized encoder's `onnx.quantization` records the source manifest digest,
the settings, the tolerances and a `verification` block: records checked,
attested records, decisions and attested decisions changed, the change rate,
the worst head's float32 and int8 ECE, the datasets checked (kind, function,
SHA-256, count) and the same figures per function. The runtime loads it like
any release. `current.json` does not move: `next:` names
`releases promote <digest>`, and promoting the float32 release again goes
back.

| Flag                           | Value     | Effect                                                                                                                   |
| ------------------------------ | --------- | ------------------------------------------------------------------------------------------------------------------------ |
| `--int8`                       |           | Required: int8 dynamic quantization is the one derivation.                                                               |
| `--cache-dir`                  | path      | The build cache the release was trained with (default `.semantscript/cache`).                                            |
| `--report`                     | path      | Where the trainer writes the JSON report (`kind` `semantscript.derive-report`; default `<artifact>.derive-report.json`). |
| `--weight-type`                | `int8`    | `int8` (default) or `uint8` weights.                                                                                     |
| `--per-channel`                |           | One scale per output channel instead of one per tensor.                                                                  |
| `--reduce-range`               |           | 7-bit weights, for CPUs without VNNI.                                                                                    |
| `--max-attested-disagreements` | count     | Attested records allowed to change decision (default 0). The manifest records the value.                                 |
| `--max-decision-change-rate`   | 0 to 1    | Share of all records allowed to change decision (default 0). The manifest records the value.                             |
| `--ece-threshold`              | 0 to 1    | Largest int8 ECE any head may have (default 0.1). The manifest records the value.                                        |
| `--promote`                    |           | After publishing, run `releases promote` on the new release.                                                             |
| `--python`, `--trainer-module` | exe, name | As for `train`.                                                                                                          |
| `--json`                       |           | Print the derive report instead of the table.                                                                            |

Exit status: 0 published (and promoted with `--promote`), 2 refused by the
gate or a usage error, 1 when the release or its records cannot be found or
the trainer fails (the message ends with a `next:` clause).

A malformed `<release>` (not hex, shorter than 7 characters) or `prune`
without `--keep` or `--older-than` exits 2. After a rollback, the next `train`
publishes its own release again and makes it current: the build cache reuses a
published release only while `current.json` still names it (see
[Build cache](build-cache.md#releases-rollback-and-prune)).

## `explain`

Calls one export the way `run` does, with every sema call's calibrated
distribution recorded, and reports why each call answered as it did. The
artifact is loaded with the runtime's `diagnostics: "always"` and `observe`
options, so program code receives what it would in production (the plain
value from a `sema` site, the diagnostic result from a `sema.withConfidence`
site). The one exception is a `@confidence` answer below its threshold on a
function with a fallback: explain never runs the application's fallbacks, so
the program receives the model's top answer there, and later code in the same
export may take a different path than production would.

| Flag                       | Value  | Effect                                                                                                                                                                                                    |
| -------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--artifact`               | path   | The artifact root (default above).                                                                                                                                                                        |
| `--bundle`                 | path   | The IR bundle for constraints, gold examples and source locations. Default: the build's bundle as above when one exists; without one, explain still shows the answer, the training cases and the release. |
| `--cache-dir`              | path   | The training cache holding the datasets (default `.semantscript/cache`, as for `train`).                                                                                                                  |
| `--neighbors`              | n      | How many nearest gold examples and training cases to list (default 5; 0 lists none).                                                                                                                      |
| `--call`                   | export | Required: the function export to call.                                                                                                                                                                    |
| `--input` / `--input-file` | JSON   | The arguments, as for `run`.                                                                                                                                                                              |
| `--json`                   |        | Print one JSON document (`module`, `export`, `arguments`, `artifact`, `bundle`, `cacheDir`, `distance`, `calls[]`, `staleArtifactFunctions`, `result`, `error`, `ok`) instead of the text.                |

For every sema call the export makes, in order, it prints:

- the function id and its source line (from the bundle) and the inputs;
- the value, confidence, uncertainty and the calibrated distribution (per field
  for an object output), and for a `@confidence` threshold whether the answer
  met it (below it the program gets `SemaConfidenceError`, or the
  application's fallback decides; explain never runs fallbacks and shows the
  model's answer instead; a `sema.withConfidence` site receives the diagnostic
  result whether or not the threshold is met);
- every constraint whose predicate holds for the input, with its source text
  and whether the answer satisfies it, the count of inactive ones and any that
  could not be evaluated (the predicate is evaluated exactly as the trainer
  does; [`examples/constraints/predicate-vectors.v1.json`](../examples/constraints/predicate-vectors.v1.json)
  pins both evaluators);
- the gold examples nearest the input, with their outputs;
- the training cases nearest the input with their labels and origins (`gold`,
  `synthetic (<teacher>)`, `adversarial constraint-boundary` or
  `adversarial counterfactual`), from the dataset the release was trained on:
  the cached `datasets/v1` file whose SHA-256 is the manifest's
  `trainingProvenance.datasetSha256`, plus the adversarial sidecar built from
  it. When that dataset is no longer cached, the newest cached dataset for the
  function is shown and marked as not the release's;
- the function's shipped verification (status, accuracy, ECE, Brier, pair
  consistency, attested cases, constraint violations, per-field accuracy) and
  its teacher, base model and dataset digest.

Above the calls it prints the release directory, application, build time and
manifest digest the answers came from. "Nearest" is a typed distance over the
JSON inputs, not the model's similarity: the mean over leaf fields of `|a-b|`
scaled by the field's spread for numbers, one minus the word-set Jaccard
overlap for strings, and 0 or 1 for everything else (a field on one side only
counts 1).

When a call reaches a function id the loaded artifact lacks, explain says so:
if the bundle has the id, the expression changed since the artifact was
trained; if neither has it, the build or the artifact is stale. It still shows
the bundle's constraints and examples, lists the artifact functions the bundle
no longer has (the likely earlier version), and exits 1; retrain with
`semantscript train`. Exit 0 when every call was explained, even when an
answer violates a constraint or falls below its threshold; exit 1 for a
missing export, a missing function id or any other error from the call.

As with `run`, the module must resolve the same `@semantscript/core` the CLI
uses; when it reaches another copy, the call fails with "no SemantScript
artifact is loaded" and explain says to run the application's own CLI. The
[diagnostics catalogue](diagnostics.md#wrong-answer-workflow) walks from an
explain report to the example or constraint to add.

## `package`

Writes a self-contained directory to deploy: everything the compiled
application needs at run time and nothing else. Run it from the application's
directory after `build` and `train`.

| Path in the bundle          | Contents                                                                                                                                                                                                                                                                                                                                                               |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dist/`                     | The compiled output at its path relative to the project (`dist/` unless `--dist` names another), without `semantscript.ir.v1.json` (the IR bundle holds the prompt text; the runtime needs only the artifact).                                                                                                                                                         |
| `node_modules/`             | The production dependencies. With a `package-lock.json` and only registry dependencies: `npm ci --omit=dev`. With `file:` dependencies (as in this repository's examples): `npm install --omit=dev --install-links` with each `file:` path made absolute, so the linked packages become real copies. Both run with `ONNXRUNTIME_NODE_INSTALL=skip` (no CUDA download). |
| `.semantscript/artifact/`   | `current.json` and the one release it names, checked first exactly as `releases promote` checks a target (integrity, verification, the runtime's loader). Older releases stay behind.                                                                                                                                                                                  |
| `--include` paths           | Each file or directory at its path relative to the project, for example `deploy/lambda.mjs`.                                                                                                                                                                                                                                                                           |
| `package.json`              | The project's, without `devDependencies` (it keeps `"type": "module"` and the scripts).                                                                                                                                                                                                                                                                                |
| `semantscript-package.json` | The manifest (`kind` `semantscript.package`, version 1): the release digest and application, platform, arch, Node version, the install command, the pruned binding paths, the bytes of each part and the path, size and SHA-256 of every other file (symlinks by their target).                                                                                        |

`onnxruntime-node` and `tokenizers` ship prebuilt binaries for every platform
(548 MB and 64 MB installed). The bundle keeps the target's only: for ONNX
Runtime its `bin/napi-v*/<platform>/<arch>` directory without the CUDA and
TensorRT provider libraries (the runtime runs the CPU provider), for
tokenizers the `.node` files of that platform and arch (both libc variants on
Linux, and `darwin-universal` on macOS). When the target is this machine, a
child Node process then loads both bindings from the bundle. The runtime finds
`.semantscript/artifact` by its upward search from `dist/`, so the bundle runs
from any working directory with no variable set.

The command prints the size of each part in bytes (the apparent size, which
is what Lambda counts unzipped): `dist`, `node_modules` with its five largest
packages, the artifact split into encoder, adapter, heads, tokenizer and
manifest, each included path, `package.json`, the manifest and the total.

| Flag          | Value | Effect                                                                                                                                                                                                                    |
| ------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--project`   | path  | The application directory holding `package.json` (default: the working directory).                                                                                                                                        |
| `--dist`      | path  | The compiled output, a directory inside the project (default `dist`); the bundle keeps it at that path, so `main` and the scripts still resolve.                                                                          |
| `--artifact`  | path  | The artifact root (default `SEMANTSCRIPT_ARTIFACT`, else `.semantscript/artifact` in the project).                                                                                                                        |
| `--out`       | path  | Where to write the bundle (default `.semantscript/package` in the project). It is built beside it and renamed into place.                                                                                                 |
| `--include`   | path  | Repeatable: a file or directory inside the project to ship at the same relative path. The compiled output, `node_modules`, `.semantscript`, `package.json` and the manifest are reserved, and it may not contain `--out`. |
| `--target`    | name  | Check the total against a deployment limit (below).                                                                                                                                                                       |
| `--max-bytes` | count | Check the total against this many bytes instead of a named target.                                                                                                                                                        |
| `--platform`  | os    | The `process.platform` to keep native binaries for (default this machine's), for example `linux`.                                                                                                                         |
| `--arch`      | cpu   | The `process.arch` to keep native binaries for (default this machine's), for example `arm64` for Lambda on Graviton.                                                                                                      |
| `--force`     |       | Replace an existing bundle. Without it an existing bundle is refused; a non-empty directory with no `semantscript-package.json` is refused even with it.                                                                  |
| `--json`      |       | Print one JSON document (`kind` `semantscript.package.report`, version 1): the parts, totals, target, encoder facts and levers.                                                                                           |

| Target                | Limit (bytes)  | Source (checked 2026-09-25)                                                                                       |
| --------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------- |
| `lambda-zip`          | 262,144,000    | AWS Lambda quotas: a .zip deployment package is at most 250 MB unzipped, including layers (AWS MB = 1,048,576 B). |
| `lambda-image`        | 10,737,418,240 | AWS Lambda quotas: a container image is at most 10 GB uncompressed, including all layers and the base image.      |
| `cloud-run-functions` | 500,000,000    | Cloud Run functions quotas: 500 MB uncompressed for sources plus modules. Cloud Run services have no image limit. |

Over the target, the bundle is still written, the report lists the size
levers with the bundle each would give and whether it fits, and the command
exits 1 with `PACKAGE_OVER_TARGET`. The levers scale the release's encoder by
sizes measured on ModernBERT-base, so for another encoder they are estimates
([Deploying](deploy.md) explains them): depth routing to 12, 6 and 4 layers
(`build --domain-depth`, given to every domain), int8 dynamic quantization (`releases derive --int8`,
then `releases promote`; its projection is the refund benchmark's measured
int8 ratio, and the derivation's gate decides whether a given release
publishes), both together, and a smaller encoder (`train --encoder-name`,
reported as the largest encoder graph that fits). A lever the release already
uses is not suggested again: depth routing when the release ships a routed
encoder prefix (a function `encoderRef` other than the default encoder, or a
`depth-NNN` encoder ref or path), int8 when an encoder's `onnx.precision` is
not `float32`. A mixed release, with some domains on a prefix and others still
on the full-depth encoder, is offered routing the remaining domains instead,
which drops the full encoder (`encoder.routing` is `none`, `mixed` or `full`
in `--json`). For an already quantized encoder the depth levers project the
float32 prefix that training writes. When the bundle without its encoder is
already over the target, only the smaller-encoder row is shown, saying that
no encoder lever helps.

| Code                                                          | Cause                                                                                                                                     |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `PACKAGE_NO_PROJECT`                                          | The project has no `package.json`, or it is not a JSON object.                                                                            |
| `PACKAGE_NO_DIST`                                             | The compiled output directory does not exist; build first.                                                                                |
| `PACKAGE_NO_RELEASE`                                          | The artifact root has no `current.json`, or it names a release that is not on disk; train first. An invalid pointer is `POINTER_INVALID`. |
| `PACKAGE_OUT_EXISTS`                                          | `--out` holds an earlier bundle and `--force` was not given, is a non-empty directory `package` did not write, or is not a directory.     |
| `PACKAGE_INCLUDE_MISSING`                                     | An `--include` path does not exist.                                                                                                       |
| `PACKAGE_INSTALL_FAILED`                                      | npm did not start or exited non-zero; the message holds npm's first 20 lines and its last 8.                                              |
| `PACKAGE_NO_BINDING`                                          | `onnxruntime-node` or `tokenizers` ships no binary for the requested platform and arch.                                                   |
| `PACKAGE_BINDING_LOAD_FAILED`                                 | The pruned bindings did not load from the bundle on this machine.                                                                         |
| `PACKAGE_OVER_TARGET`                                         | The bundle exceeds `--target` or `--max-bytes`; it is written anyway.                                                                     |
| `RELEASE_INTEGRITY`, `RELEASE_UNVERIFIED`, `RELEASE_REJECTED` | The current release fails the same checks `releases promote` applies (above).                                                             |

An unknown `--target`, `--target` together with `--max-bytes`, a `--max-bytes`
that is not a positive whole number, a positional argument, a `--dist` outside the project, an `--include`
outside the project, on a reserved path or containing `--out`, or an `--out` that is the project,
the compiled output or the artifact (or contains one of them) exits 2.
