# CLI

`semantscript` is the single entry point over the compiler, trainer, verifier
and runtime packages. It coordinates them without owning their core
implementations: `build` is the compiler, `train` is the Python trainer's
bundle driver, `test` and `run` are the runtime.

```text
semantscript init  [--tool next|vite|esbuild|tsc] [--no-example]
semantscript build [--project tsconfig.json] [--application <id>] [--bundle <path>] [--domain-depth <name>=<layers>]...
semantscript train [--bundle <path>] [--artifact <root>] [--teacher <teacher.toml>] [options]
semantscript dev   [build and train options] [--debounce <ms>] [--once]
semantscript test  [--artifact <root>] [--bundle <path>] [--json]
semantscript run   [--artifact <root>] <module.js> [--call <export>] [--input <json> | --input-file <path>]
```

## init

Adopts SemantScript in an existing project with no configuration file. `init`
detects the build tool from the project root (a `next.config.*` or `next`
dependency, then a `vite.config.*` or `vite` dependency, then `esbuild` in the
dependencies or a build script, then `tsconfig.json`; `--tool` overrides) and
wires the matching compiler adapter with an idempotent text edit that keeps
comments and formatting:

| Tool    | Edit                                                                                                                                                      |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| tsc     | `plugins: [{ "transform": "@semantscript/compiler/transformer" }]` in `tsconfig.json`; `ts-patch` as a dev dependency and `ts-patch install` in `prepare` |
| Vite    | `import semantscript from "@semantscript/compiler/vite"` and `semantscript()` first in `plugins`                                                          |
| esbuild | the same for `@semantscript/compiler/esbuild` in the first `build.mjs`, `esbuild.config.*` or `scripts/build.*` that calls esbuild                        |
| Next.js | a `turbopack.rules` entry for `*.sem.ts`, `serverExternalPackages` for the runtime and `outputFileTracingIncludes` for the artifact directory             |

Every project also gets the editor plugin entry
`{ "name": "@semantscript/compiler/ts-plugin" }` in `tsconfig.json`, which
puts the verified accuracy and guidance at each sema site in any editor that
runs tsserver (see the compiler README).

A config it cannot edit safely (missing, unparsable, `require()`-based, or a
`next.config` that already sets one of the three keys) is reported as `manual`
with the snippet to add, and nothing is written to it. `init` also adds
`@semantscript/core` and `@semantscript/compiler` to `package.json`, writes
`.semantscript/.gitignore` reserving `artifact/` and `cache/`, and, unless
`--no-example` is passed or a `.sem.ts` file already exists, one starter
expression (`src/hello.sem.ts`, or `lib/hello.sem.ts` for Next.js). It ends
with the next steps: `npm install`, the tool's build, `semantscript train`,
and one `loadSemaArtifact()` call at startup.

### Defaults

Every command works without flags once the project is initialised:

- the bundle is the build's `semantscript.ir.v1.json` under the tsconfig
  `outDir`, then `.`, `dist`, `out` or `build`;
- the artifact root is `SEMANTSCRIPT_ARTIFACT` when set, else
  `.semantscript/artifact`, for `train`, `test`, `run` and the runtime's
  `loadSemaArtifact()` with no argument;
- the teacher is the first of `semantscript.teacher.toml`, `teacher.toml` and
  `.semantscript/teacher.toml`; when none exists and `ANTHROPIC_API_KEY` is
  set, `train` writes `.semantscript/teacher.toml` for the Anthropic backend
  (`claude-sonnet-5`) and uses it. Without a key or a file, `train` asks for
  `--teacher`. The key never enters any file.

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
exit with status 1 and emit nothing. `--domain-depth <name>=<layers>`
(repeatable) routes the named domain (an `@domain` header or a file's name)
through that many shared-encoder layers; see the compiler README on routed
domains. The summary then lists the domains with their depths.

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

### Encoder size

The encoder is a per-application setting: `--encoder-name` and
`--encoder-revision` (a pinned 40-character commit) select the Hugging Face
checkpoint every function of the application is fine-tuned from, and the
default is `answerdotai/ModernBERT-base` at the revision the trainer pins.
Decision-3 chose it for the Phase 1 exit criteria; the encoder sweep under
`benchmarks/refund/data/results-encoder-sweep-2026-09-24` measures the base,
large and ~1B points and gives the rule for choosing another: the largest
encoder that fits the request path's latency budget, because parallel heads
amortise one encoder pass across every decision in a request.

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

## dev

Makes training invisible while developing: `dev` runs `build` then `train`
once, then watches the project's TypeScript sources (`.ts`, `.mts`, `.cts`,
`.tsx` and `tsconfig.json` under the project root, ignoring `node_modules`,
`.git`, `.semantscript` and declaration files) and repeats both on every save,
debounced (`--debounce`, default 300 ms). A save during a run queues exactly
one more run. It takes every `build` and `train` option; `--once` runs a
single cycle and exits with the train status, which is how scripts and tests
use it. Each cycle is numbered on stderr with its reason (`initial build` or
the changed file) and the trainer's progress lines stream underneath, one per
expression (`<function id>: generating …`, `training its head on the frozen
shared encoder`, verification and the published release), before the report
table.

Only what changed retrains: the build cache keyed by function id reuses every
expression whose IR is unchanged, so a saved edit to one expression trains one
head on the frozen shared encoder. A running application that loaded the
artifact with `loadSemaArtifact(path, { watch: true })` (see the runtime
README) swaps in each published release without a restart, and because the
trainer publishes only after verification passes and the runtime replaces an
artifact only once the new release has loaded whole, the stale head stays in
service through a failed build, a failed verification or a broken release.
Ctrl-C stops the loop after the current cycle.

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
