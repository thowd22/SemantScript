# CLI reference

`semantscript` is one entry point over the compiler, the Python trainer and
the runtime. This page lists every command, flag, default and exit code as
the shipped source defines them; the [CLI guide](../cli/README.md) explains
the behavior in prose.

```text
semantscript init  [--tool next|vite|esbuild|tsc] [--no-example]
semantscript build [--project tsconfig.json] [--application <id>] [--bundle <path>]
                   [--route-domains] [--domain-depth <name>=<layers>]...
semantscript train [--bundle <path>] [--artifact <root>] [--teacher <teacher.toml>] [options]
semantscript dev   [build and train options] [--debounce <ms>] [--once]
semantscript test  [--artifact <root>] [--bundle <path>] [--json]
semantscript run   [--artifact <root>] <module.js> [--call <export>] [--input <json> | --input-file <path>]
```

## Exit codes

| Code | Meaning                                                                                                                                                            |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0    | The command succeeded. `semantscript help`, `--help` and `-h` print the usage and exit 0.                                                                          |
| 1    | The command ran and failed: compile diagnostics, a failed training or verification, a failed `test`, an unknown `--call` export, or any other error while working. |
| 2    | Usage: no command, an unknown command, an unknown or malformed option, a missing required value, or an inconsistent combination (`--input` with `--input-file`).   |

Usage errors print the message and the usage text to stderr. Every other
failure prints its message to stderr; compile diagnostics carry the file, line,
column, code and site text.

## Defaults and environment

Every command runs without flags in an initialised project:

| Setting            | Resolution                                                                                                                                                                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Project            | `--project`, else `tsconfig.json` in the working directory.                                                                                                                                                                                                  |
| Bundle             | `--bundle`, else `semantscript.ir.v1.json` under the tsconfig `outDir`, then `.`, `dist`, `out`, `build`.                                                                                                                                                    |
| Artifact root      | `--artifact`, else `SEMANTSCRIPT_ARTIFACT`, else `.semantscript/artifact`. The runtime's `loadSemaArtifact()` with no path resolves the same way, searching upward from the compiled entry script and then from the working directory.                       |
| Teacher            | `--teacher`, else the first of `semantscript.teacher.toml`, `teacher.toml`, `.semantscript/teacher.toml`; else, when `ANTHROPIC_API_KEY` is set, `train` writes `.semantscript/teacher.toml` for `claude-sonnet-5` and uses it. The key never enters a file. |
| Python interpreter | `--python`, else `SEMANTSCRIPT_PYTHON`, else `python3`. Inside this repository the trainer and model sources (and `.python-packages` when present) are prepended to `PYTHONPATH`.                                                                            |
| Cache directory    | `--cache-dir`, else `.semantscript/cache`.                                                                                                                                                                                                                   |

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

| Flag               | Value  | Effect                                                                                               |
| ------------------ | ------ | ---------------------------------------------------------------------------------------------------- |
| `--bundle`         | path   | The IR bundle (default above).                                                                       |
| `--artifact`       | path   | The artifact root to publish into (default above).                                                   |
| `--teacher`        | path   | A TOML file with a `[teacher]` table (`backend = "anthropic"` or `"ollama"`); see the trainer guide. |
| `--cache-dir`      | path   | The build cache (default `.semantscript/cache`); see the [build cache](build-cache.md) page.         |
| `--report`         | path   | Also write the JSON report here.                                                                     |
| `--python`         | exe    | The interpreter to run the trainer with.                                                             |
| `--trainer-module` | module | The Python module to invoke (default `semantscript_trainer.cli`); for tests and forks.               |

Options handed to the trainer unchanged:

| Flag                                                              | Value    | Meaning                                                                                                   |
| ----------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------- |
| `--cases`                                                         | integer  | Synthetic cases generated per expression (gold examples count toward it).                                 |
| `--epochs`, `--batch-size`, `--learning-rate`, `--seed`           | numbers  | The fine-tuning recipe.                                                                                   |
| `--max-sequence-length`                                           | integer  | Tokens per canonical input (default 512).                                                                 |
| `--evaluation-ratio`                                              | fraction | Held-out calibration split (default 0.2).                                                                 |
| `--device`                                                        | name     | `auto`, `cpu`, `cuda`, …                                                                                  |
| `--head-architecture`                                             | name     | `linear` (default) or `mlp`.                                                                              |
| `--select-best-epoch`                                             |          | Keep the epoch with the best held-out accuracy instead of the last.                                       |
| `--encoder-name`, `--encoder-revision`                            | strings  | The Hugging Face checkpoint and pinned commit every function of the application fine-tunes from.          |
| `--local-files-only`                                              |          | Never download; fail if the checkpoint is not cached.                                                     |
| `--ece-threshold`                                                 | fraction | Verification gate on calibration error (default 0.1).                                                     |
| `--max-constraint-violation-rate`                                 | fraction | Share of raw predictions allowed to violate an active constraint (default 0; the value used is recorded). |
| `--counterfactual-ratio`                                          | fraction | Share of synthetic cases that get a counterfactual twin (default 1).                                      |
| `--adapter-bottleneck-size`                                       | integer  | Width of the per-domain adapter.                                                                          |
| `--application-id`, `--application-version`, `--compiler-version` | strings  | Recorded in the manifest.                                                                                 |
| `--no-cache`                                                      |          | Ignore the build cache and write nothing to it.                                                           |
| `--full`                                                          |          | Retrain every function jointly, discarding cached function records.                                       |

## `dev`

`build` then `train`, then watch the project's TypeScript sources and repeat
both on every save. Takes every `build` and `train` option plus:

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
constraint violations, each head's accuracy).

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
