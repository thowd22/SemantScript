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
```

## Exit codes

| Code | Meaning                                                                                                                                                                                                                                                                                           |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0    | The command succeeded. `semantscript help`, `--help` and `-h` (also after a command, as in `semantscript doctor --help`) print the usage and exit 0.                                                                                                                                              |
| 1    | The command ran and failed: compile diagnostics, a failed training or verification, a failed `test`, a failed `doctor` check or `train` preflight, a `train` stopped by its `--max-cost-usd` cap, a failed `teacher probe` request, an unknown `--call` export, or any other error while working. |
| 2    | Usage: no command, an unknown command, an unknown or malformed option, a missing required value, or an inconsistent combination (`--input` with `--input-file`).                                                                                                                                  |

Usage errors print the message and the usage text to stderr. Every other
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
| Cache directory    | `--cache-dir`, else `.semantscript/cache`.                                                                                                                                                                                                                                                                                                                                |

## `init`

Wires the compiler into an existing project and adds a starter expression.

| Flag           | Value                            | Effect                                                                                                                                                                                  |
| -------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--tool`       | `next`, `vite`, `esbuild`, `tsc` | Overrides detection. Detection order: a `next.config.*` or `next` dependency; a `vite.config.*` or `vite` dependency; `esbuild` in the dependencies or a build script; `tsconfig.json`. |
| `--no-example` |                                  | Do not write the starter `.sem.ts` (`src/hello.sem.ts`, or `lib/hello.sem.ts` for Next.js). A project that already has a `.sem.ts` file gets none either.                               |

What it writes: the adapter entry for the detected tool (see the
[build tool pages](build-tools/tsc.md)), the editor plugin entry
`{ "name": "@semantscript/compiler/ts-plugin" }` in `tsconfig.json`,
`@semantscript/core` and `@semantscript/compiler` in `package.json`,
`.semantscript/.gitignore` reserving `artifact/` and `cache/`, and the starter
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
`--domain-depth` value.

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
| `--application-id`, `--application-version`, `--compiler-version` | strings  | Recorded in the manifest.                                                                                                                                     |
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
absent from the artifact, or any example mismatches.

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
path, both input flags, or input that is not JSON.
