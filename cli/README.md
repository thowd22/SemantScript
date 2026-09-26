# CLI

```sh
npm install semantscript @semantscript/core @semantscript/compiler
pip install "semantscript-trainer[training]"   # for train, dev and teacher probe
```

First publish pending, see [releasing](https://github.com/thowd22/SemantScript/blob/main/docs/releasing.md).

`semantscript` is the single entry point over the compiler, trainer, verifier
and runtime packages. It coordinates them without owning their core
implementations: `build` is the compiler, `train` is the Python trainer's
bundle driver, `test`, `run` and `explain` are the runtime, `releases`
manages the published releases under the artifact root, and `package`
writes the deployable bundle.

```text
semantscript init  [--tool next|vite|esbuild|tsc] [--no-example] [--no-doctor] [--teacher anthropic|openrouter|ollama|constraints] [--teacher-model <id>] [--python <exe>] [--trainer-module <module>]
semantscript doctor [--python <exe>] [--teacher <teacher.toml>|constraints] [--probe request|free|none] [--device auto|cpu|cuda] [--trainer-module <module>] [--no-teacher] [--runtime] [--json]
semantscript build [--project tsconfig.json] [--application <id>] [--bundle <path>] [--domain-depth <name>=<layers>]...
semantscript train [--bundle <path>] [--artifact <root>] [--teacher <teacher.toml>|constraints] [--estimate] [--max-cost-usd <x>] [options]
semantscript teacher probe [--teacher <teacher.toml>|constraints] [--python <exe>] [--json]
semantscript dev   [build and train options] [--debounce <ms>] [--once]
semantscript test  [--artifact <root>] [--bundle <path>] [--json]
semantscript run   [--artifact <root>] <module.js> [--call <export>] [--input <json> | --input-file <path>]
semantscript releases [list | show <release> | rollback [<release>] | promote <release>] [--artifact <root>] [--dry-run] [--json]
semantscript releases prune [--keep <n>] [--older-than <30d>] [--artifact <root>] [--dry-run] [--json]
semantscript explain [--artifact <root>] [--bundle <path>] [--cache-dir <dir>] [--neighbors <n>] [--json] <module.js> --call <export> [--input <json> | --input-file <path>]
semantscript package [--project <dir>] [--dist <dir>] [--artifact <root>] [--out <dir>] [--include <path>]... [--target lambda-zip|lambda-image|cloud-run-functions | --max-bytes <n>] [--platform <os>] [--arch <cpu>] [--force] [--json]
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
`.semantscript/.gitignore` reserving `artifact/`, `cache/`, `package/`,
`.package-staging-*/` and `*.traceback.txt` (the traceback `train` keeps next
to its report; appending any that are missing on a re-run), and, unless
`--no-example` is passed or a `.sem.ts` file already exists, one starter
expression (`src/hello.sem.ts`, or `lib/hello.sem.ts` for Next.js). It ends
with the next steps (`npm install`, the tool's build, `semantscript train`,
and one `loadSemaArtifact()` call at startup) and then the `doctor` checks
below, without a billed teacher request, so a missing Python package, key or
GPU shows before the first `train`. The checks do not change `init`'s exit
status; `--no-doctor` skips them, and `--python` and `--trainer-module` choose
the interpreter and trainer module they run, as for `doctor`.

`--teacher anthropic|openrouter|ollama|constraints` writes
`.semantscript/teacher.toml` for that teacher (`--teacher-model` names another
model); on an interactive terminal with no teacher file, `init` asks instead.
The file never holds a key, an existing teacher file is never replaced, and
the closing checks run against it. `semantscript teacher probe` then sends one
small request through it and prints the model, latency, tokens and cost.

### Defaults

Every command works without flags once the project is initialised:

- the bundle is the build's `semantscript.ir.v1.json` under the tsconfig
  `outDir`, then `.`, `dist`, `out` or `build`;
- the artifact root is `SEMANTSCRIPT_ARTIFACT` when set, else
  `.semantscript/artifact`, for `train`, `test`, `run`, `explain`, `releases` and the runtime's
  `loadSemaArtifact()` with no argument;
- the teacher is the first of `semantscript.teacher.toml`, `teacher.toml` and
  `.semantscript/teacher.toml`; when none exists and `ANTHROPIC_API_KEY` is
  set, `train` writes `.semantscript/teacher.toml` for the Anthropic backend
  (`claude-sonnet-5`) and uses it. Without a key or a file, `train` asks for
  `--teacher`. The key never enters any file. `--teacher constraints` (unless
  a file of that name exists) selects the built-in constraints teacher, which
  labels the cases from the expression's own constraints with no model and no
  key when they decide every input.

## doctor

Checks both toolchains before anything runs and prints one line per check,
`pass`, `fail`, `warn` or `skip`, with a `fix:` line under every check that did
not pass:

- Node: the release (22.13 or later) and the native `onnxruntime-node` and
  `tokenizers` bindings for this platform, loaded the way the runtime loads
  them;
- Python: the interpreter the CLI will use, the trainer and model packages
  with their versions, PyTorch and Transformers, the device training will use
  (CUDA or ROCm with its memory, else the CPU with a warning; Apple MPS is
  reported but the trainer does not use it), ONNX Runtime and ONNX, and the
  platform environment: `PYTHONNOUSERSITE=1` when packages in the user site
  break the imports, `HSA_ENABLE_DXG_DETECTION=1` when ROCm on WSL2 finds the
  GPU only with it. Doctor names a variable only after re-running the imports
  with and without it, so it never suggests one that changes nothing;
- the teacher: the file `train` would use, the key in the environment
  (`ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`, including the OpenRouter case, where the key belongs in
  `ANTHROPIC_API_KEY`), and one minimal request with its latency and tokens
  (`--probe free` sends nothing billed, `--probe none` nothing at all).

The Python checks run as `python -m semantscript_trainer.cli doctor --json`,
which imports PyTorch, Transformers and ONNX Runtime in child interpreters so
a broken package becomes a failed line instead of a traceback. `--runtime`
checks only Node and the bindings, for a machine that serves artifacts and
never trains; `--json` prints the report; `--device` checks the device `train`
will be asked for and `--trainer-module` names another trainer module. When
the default interpreter lacks the packages and a `.venv` sits in the working
directory or above it, the fix lines name that venv's interpreter (`--python`
or `SEMANTSCRIPT_PYTHON`). Exit 1 when any check fails. The
[environment guide](https://github.com/thowd22/SemantScript/blob/main/docs/environment.md) explains each check and records
doctor runs per platform.

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
teacher (`--teacher` is a TOML file with a `[teacher]` table, or the keyword
`constraints` for the built-in constraints teacher; see docs/teachers.md),
adds the adversarial sidecar when the function declares constraints
(the teacher must then support boundary and counterfactual generation), trains
one classifier for a single function or the shared-encoder application for
several, verifies every function against the release gate, binds verified IR
and exports one content-addressed artifact under `--artifact`. The trainer
writes a JSON report (`--report`, default `<artifact>.report.json`) that the
CLI renders per function: dataset and adversarial sizes, held-out accuracy,
verification status, accuracy, ECE, attested cases and constraint violations,
plus the published release digest. A function that fails verification stops the
build with the report kept and nothing published; the trainer's exit status is
the command's. Every expression needs at least one gold example (verification
checks the artifact against them); one with none stops the command before any
generation, with the expression named.

Every failure names the next command. Each verification failure in the report
ends with a `next:` line derived from the failing cases (a constraint to add,
an example to add, more `--cases` or `--epochs`, or the seed retry). A trainer
that dies on an uncaught exception (a missing package, an interpreter too old,
a crash) becomes one line, `semantscript train: the trainer stopped: <exception>;
next: <fix>; full traceback in <artifact>.report.traceback.txt`, naming the
`doctor` check to run with this run's `--python`, `--trainer-module` and
`--teacher`; every failure the trainer catches ends its `error:` line with
`; next: <fix>`, and a trainer the operating system kills names the signal and
the fix. A
traceback the trainer goes on from is printed in place, and only a report this
run wrote is rendered. The fix texts come from `diagnostics/remedies.json`
(docs/diagnostics.md).

Before the trainer starts, `train` runs the `doctor` Python and teacher checks
as a preflight (about 5 seconds; no billed request) and exits 1 with the fix
lines when any fails, instead of failing minutes into a run. `--no-preflight`
skips it.

`--estimate` prints what the teacher would cost instead of running: per
expression and in total, the requests, input tokens (and the share the prompt
cache serves), output tokens, USD at the configured backend's price and the
wall time, without the preflight, a teacher request or PyTorch.
`--max-cost-usd <x>` caps a run: the request that would pass the cap is not
sent, the run exits 1, and every finished dataset and paid response is kept so
the next run replays them free. Every run prints a running
`teacher: … USD …` line, and the report table ends with the run's requests,
replays and cost (docs/teachers.md).

When verification fails only on the constraint-violation rate or the ECE, and
by no more than `--seed-retry-margin` times the gate (default 2), `train`
retrains on the cached datasets with the next seed, up to `--seed-attempts`
runs (default 3; 1 turns it off). A gold-example miss, a type error or a
failure outside the margin stops at once and says why. The report lists every
attempt with its seed and gate metrics, and the published release's manifest
records the seed that passed (`functions[].trainingProvenance.seed`, shown by
`semantscript test`; docs/cli-reference.md#seed-retry).

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
`--max-constraint-violation-rate`, `--seed-attempts`, `--seed-retry-margin`,
`--counterfactual-ratio`, `--adapter-bottleneck-size`, `--no-cache`, `--full`,
`--application-id`, `--application-version`, `--compiler-version` (default: the installed `@semantscript/compiler` version) and
`--cache-dir` (default
`.semantscript/cache`). The interpreter is `--python`,
then `SEMANTSCRIPT_PYTHON`, then `python3` (`python` on Windows); inside this repository the trainer
and model sources (and `.python-packages` when present) are put on `PYTHONPATH`
automatically, and the caller's `PYTHONPATH` is kept after them.

## dev

Makes training invisible while developing: `dev` runs `build` then `train`
once, then watches the project's TypeScript sources (`.ts`, `.mts`, `.cts`,
`.tsx` and `tsconfig.json` under the project root, ignoring `node_modules`,
`.git`, `.semantscript` and declaration files) and repeats both on every save,
debounced (`--debounce`, default 300 ms). A save during a run queues exactly
one more run. It takes every `build` and `train` option (the `train` preflight runs until
one training succeeds, not on every save); `--once` runs a
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
attested cases, constraint violations, the training seed the release passed at
(`-` for releases from before seeds were recorded) and each head's accuracy. Trained models
are not persisted, so this is the gate `train` already enforced; with
`--bundle` the command also loads the artifact and replays every IR example
through the runtime, comparing outputs by value (diagnostic functions by their
`value`). Any function whose status is not `passed`, any bundle function absent
from the artifact and any example mismatch make the command exit with status 1,
each with a `next:` line naming the fix (retrain, rebuild then retrain, or
`semantscript releases rollback`); an artifact whose pointer or release does not
read fails the same way `run` does, naming `releases rollback`. `--json` prints
the same as one document, with the `next:` lines as `next`.

## run

Loads the artifact, which activates the runtime the compiled `__sema` calls
resolve against, then imports the module. With `--call <export>` it calls that
export with the JSON input (`--input '[1, 2]'` spreads an array as positional
arguments; any other JSON value is the single argument; `--input-file` reads
the same from a file), awaits a returned promise and prints the JSON result.
The module resolves `@semantscript/core` from its own location, and that must
be the same installation the CLI uses. The artifact is closed when the command
returns, so nothing stays loaded in the process.

## releases

Every `train` publishes an immutable `releases/sha256-<manifest digest>/`
under the artifact root and points `current.json` at it. `semantscript
releases` (or `releases list`) shows every release newest first by its
manifest's `build.createdAt`, with the digest, the application version, how
many functions passed verification, the lowest accuracy, the highest ECE and
the constraint violations, and marks the current one; `releases show
<release>` prints one release's per-function table, the same columns as
`test`. A release is named by its digest or a unique prefix of at least 7
characters.

`releases rollback` points `current.json` back at the release created before
the current one (or at a named one; `releases promote <release>` rolls
forward). It first checks the target the way the runtime will: the manifest
and every resource hash, no symlinks, every function passed verification, and
the runtime's own loader checks (ABI compatibility, tensors, opsets and the
model chain) accept it; only ONNX session start-up is left out. The pointer is rewritten atomically (temporary
file, fsync, rename) and no release is removed, so a running process loaded
with `watch: true` swaps to the target and a later rollback can swap back.

`releases prune --keep <n>` and/or `--older-than <30d|12h|90m>` delete old
releases to free disk. The current release is never removed, whatever its
age, and neither are invalid releases or an export in progress. `--dry-run`
shows what would go. See the [CLI reference](https://github.com/thowd22/SemantScript/blob/main/docs/cli-reference.md#releases)
for the error codes.

## explain

Answers "why did this expression say that?" for one input. It calls the export
the way `run` does, but loads the artifact with the runtime's
`diagnostics: "always"` and `observe` options, so every sema call the export
makes is recorded with its calibrated distribution while the program still
receives what it would in production (except below a `@confidence` threshold
on a function with a fallback: explain never runs the application's fallback
and returns the model's top answer instead). For each call it prints the value,
confidence, uncertainty and distribution; every constraint whose predicate
holds for the input, with whether the answer satisfies it (predicates are
evaluated with the trainer's semantics, pinned by
[`examples/constraints/predicate-vectors.v1.json`](https://github.com/thowd22/SemantScript/blob/main/examples/constraints/predicate-vectors.v1.json));
the nearest gold examples from the bundle; the nearest training cases with
their labels and origins (gold, synthetic with the teacher, adversarial
constraint-boundary or counterfactual) from the cached dataset whose digest the
manifest records as the release's `datasetSha256`, plus its adversarial
sidecar; and the function's shipped verification, teacher and base model.
Above the calls it names the release (directory, application, build time and
manifest digest). "Nearest" is a typed distance over the JSON inputs, stated
in the output.

A call whose compiled id the artifact lacks is reported as missing (the
expression changed since training when the bundle still has the id), with the
bundle's constraints and examples and the artifact functions the bundle no
longer has; the command then exits 1. A `@confidence` answer below its
threshold is reported, and explain never runs the application's fallbacks. The
bundle is optional (`--bundle`, else the build's); `--cache-dir` defaults to
`.semantscript/cache`, `--neighbors` to 5, and `--json` prints one document.
The [wrong-answer workflow](https://github.com/thowd22/SemantScript/blob/main/docs/diagnostics.md#wrong-answer-workflow)
explains how to read the output.

## package

`semantscript package` writes `.semantscript/package/`, a directory that runs
on its own: `dist/` without the IR bundle (it holds the prompt text), the
production `node_modules` pruned to the target platform's ONNX Runtime and
tokenizers binaries (no CUDA or TensorRT provider; the runtime runs on the
CPU), `.semantscript/artifact` with only the current release, any
`--include` files, and `semantscript-package.json` with the SHA-256 and size
of every file. The current release is checked first the way `releases
promote` checks a target, and on the host platform the pruned bindings are
loaded from the bundle before it is kept. The output is a size table by part
(dist, node_modules with its largest packages, encoder, adapter, heads,
tokenizer, included files) and the total.

`--target lambda-zip|lambda-image|cloud-run-functions` or `--max-bytes <n>`
checks the total. Over it, the bundle is still written, the command exits 1
with `PACKAGE_OVER_TARGET`, and it lists the levers with the bundle each
would give: depth routing (or, for a release with some domains already
routed, routing the rest), int8 quantization (a measurement only: no int8
derivation exists for applications yet), both, or a smaller encoder. See [Deploying](../docs/deploy.md) and the
[CLI reference](../docs/cli-reference.md#package).
