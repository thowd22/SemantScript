# Tutorial: from an empty directory to a running refund decision

This is the end-to-end path: install the packages (from npm and PyPI, or from
a clone of the repository), compile the refund-decision expression, train its
artifact, and call it. Every command was run on the development machine described below;
where a step needs something you must provide (a teacher), the alternatives
are listed with what each costs. The [getting-started guide](getting-started.md)
covers the other direction, adding one expression to an app you already have.

## Hardware and time

| Need              | Used here                                                                                        | Minimum that works                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Node              | 22.22, npm 10                                                                                    | Node 22.13 or later                                                   |
| Python            | 3.12 with PyTorch, Transformers, ONNX, ONNX Runtime                                              | 3.12; the `training` extra installs the rest                          |
| GPU for training  | AMD Radeon RX 9070 XT, 16 GB, ROCm under WSL2                                                    | Any CUDA or ROCm GPU with 8 GB; a CPU works and takes tens of minutes |
| CPU for inference | Ryzen 9 9900X; 4.8 ms per call at depth 6, 28 ms at full depth                                   | Any x64 or arm64 CPU; the ONNX bindings are prebuilt                  |
| Disk              | 600 MB for the encoder checkpoint, 275 to 600 MB per artifact, as much again for the build cache | 3 GB free                                                             |
| Network           | Once, for the ModernBERT-base checkpoint from the Hugging Face Hub, plus the teacher if remote   |                                                                       |

## 1. Install

Two routes. The published packages need no clone; the clone is the
contributor's route and the one the repository's examples use. The steps
below give both routes' commands where they differ, and a few extras (the
keyless reference application and the Express server in step 5) exist only
in the clone.

### From npm and PyPI

```sh
mkdir refund-decision && cd refund-decision
npm init -y                                     # this directory's own package.json
npm install semantscript @semantscript/core @semantscript/compiler @semantscript/framework
python3 -m venv .venv && .venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "semantscript-trainer[training]"   # trainer, model, torch, transformers, onnx, onnxruntime
export SEMANTSCRIPT_PYTHON="$PWD/.venv/bin/python"
npx semantscript init --no-example   # new project: tsconfig.json, type module, build script, transformer, ts-patch; runs doctor
npm install                                     # runs the prepare script: ts-patch install
```

`npm init -y` comes first so that `npm install` installs here: in a directory
with no `package.json`, npm installs into the nearest parent directory that
has one. In a directory with no `tsconfig.json` and no other build tool, `init` starts
a TypeScript project: it writes `tsconfig.json` (NodeNext modules, `src/`
compiled to `dist/`), sets `"type": "module"` and a `build` script that runs
`tspc`, and adds TypeScript (5.9 to 6.x; 7 is not supported yet) to
`devDependencies`. The four npm packages and the trainer share one version, and every artifact
records the compiler and trainer versions that built it. For a GPU, install
the CUDA or ROCm PyTorch build into the venv first
([pytorch.org](https://pytorch.org/get-started/locally/)).

> **First publish pending.** The release workflow is in place
> ([releasing](releasing.md)) but version 0.1.0 is not on npm and PyPI yet, so
> `npm install` and `pip install` above return 404 until it is. Until then
> take the clone route below, or `npm install` the four `npm pack` tarballs
> and `pip install` the clone as the [getting-started guide](getting-started.md)
> describes; this route was run end to end against a local registry holding
> exactly those packs (`init`, `build`, `train --teacher constraints`, `test`
> and `run` in a directory with no checkout, recorded in
> [releasing](releasing.md#local-proof)).

### From a clone

```sh
git clone https://github.com/thowd22/SemantScript.git semantscript && cd semantscript
npm install                    # links the workspaces
npm run build                  # tsc -b: compiler, runtime, cli, framework, refund benchmark
python3 -m venv .venv && .venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev,training]'   # trainer, model, torch, transformers, onnx, onnxruntime
export SEMANTSCRIPT_PYTHON="$PWD/.venv/bin/python"      # the CLI's interpreter, from any directory
npm run check                  # both lints, both test suites: about three minutes
```

The CLI runs `python3` unless told otherwise, and the venv is not activated,
so `SEMANTSCRIPT_PYTHON` points `doctor`, `train` and `dev` at it (on Windows,
`.venv\Scripts\python.exe`; activating the venv works as well).

Without `venv` on the host Python, install into the ignored target instead
(`python3 -m pip install --target .python-packages '.[dev,training]'`; never
add `--upgrade` to a later single-package install into that target, it wipes
the `bin` directory), and leave `SEMANTSCRIPT_PYTHON` unset: the CLI adds that
directory to the path of the default `python3`. On an AMD GPU under WSL, the model README's ROCm section
has the wheel to install.

Then check the environment before anything runs:

```sh
npx semantscript doctor --no-teacher               # published route, in refund-decision/
node cli/bin/semantscript.js doctor --no-teacher   # clone route, at the repository root
```

The teacher is chosen in step 4.

Every line should read `pass` or `skip` (a `warn` on `device` means training will run on
the CPU). A `fail` carries its fix; if `trainer` or `torch` fails and the fix
names `.venv/bin/python`, the variable above is not set in this shell. The
`.venv` keeps the user site out, so `PYTHONNOUSERSITE=1` matters only for an
interpreter that sees it (such as the repository's `.python-packages` route):
on the WSL2 + ROCm machine used here that route needs it because NumPy 2 in the
user site breaks Transformers. Doctor names `HSA_ENABLE_DXG_DETECTION=1` only when ROCm
finds the GPU only with it. The [environment guide](environment.md) records
the full run.

## 2. The expression

The refund decision lives in the Express example, already wired to the
compiler through its `tsconfig.json`. On the published-packages route, fetch
the same file into your project:

```sh
mkdir -p src && curl -fsSLo src/refunds.sem.ts \
  https://raw.githubusercontent.com/thowd22/SemantScript/main/examples/express-app/src/refunds.sem.ts
```

Its core (the full file carries all six policy rules as constraints):

```ts
// examples/express-app/src/refunds.sem.ts
import { always, never, sema } from "@semantscript/core";

export function decideRefund(customer: Customer, order: Order): RefundDecision {
  return sema<RefundDecision>({
    examples: [
      {
        inputs: {
          customer: { priorRefunds: 0, tier: "enterprise" },
          order: { ageDays: 45, status: "paid", total: 129 },
        },
        output: "approve",
      },
    ],
    constraints: [
      never(() => order.status === "fraudulent", "approve"),
      always(() => order.ageDays > 90, "deny"),
    ],
  })`Apply our refund policy. Enterprise customers get 60 days; everyone else gets 30. Suspicious circumstances go to review.
Customer: ${customer}
Order: ${order}`;
}
```

The output type is the support (`"approve" | "deny" | "review"`), the
interpolations are the inputs, the example is an attested case the verifier
must reproduce, and the two constraints are rules the release gate checks.
The text is what a teacher reads to generate the corpus.

## 3. Compile

On the published-packages route, in your project directory:

```sh
npm run build                  # tspc -p tsconfig.json (or npx semantscript build)
ls dist/refunds.sem.js dist/semantscript.ir.v1.json
```

On the clone route, in the example:

```sh
cd examples/express-app
npm install                    # file: links to ../../compiler, ../../runtime, ../../framework
npm run build                  # tspc -p tsconfig.json
ls dist/refunds.sem.js dist/semantscript.ir.v1.json
```

The examples keep `file:` links on purpose: they build against the
repository's sources, so CI tests the tree as it is rather than the last
published release. Steps 4 and 5 run from your project directory on the
published route and from `examples/express-app` on the clone route; the
lines that exist only in a clone are marked.

`dist/refunds.sem.js` now calls the runtime by function id and
`dist/semantscript.ir.v1.json` holds the IR record and the execution plan.
This step was run on 2026-09-25 and takes under a second after the install.

## 4. Train

Pick a teacher. All four routes produce the same artifact layout; the
[teachers page](teachers.md) compares them in detail.

| Route                                    | Command                                                                                                                                                                                                                                              | What it needs            | Measured cost and time for this expression                                                                                                                                                                                                                                                              |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Anthropic API                            | `ANTHROPIC_API_KEY=… npx semantscript train`                                                                                                                                                                                                         | an Anthropic key         | run `semantscript train --estimate` for the figure; a minute of GPU training                                                                                                                                                                                                                            |
| Sonnet 5 through OpenRouter              | `semantscript init --teacher openrouter` writes the `[teacher]` TOML with `backend = "anthropic"`, `model = "anthropic/claude-sonnet-5"`, `base_url = "https://openrouter.ai/api"`, `mode = "direct"`; `ANTHROPIC_API_KEY` set to the OpenRouter key | an OpenRouter key        | about USD 0.017 per request with the prompt of 2026-09-25 (the Express example's two expressions at 192 cases cost about USD 10 including adversarial cases); the compact, cached prompt now costs USD 0.0017 to 0.0071 per request, and `semantscript train --estimate` prints the figure for your run |
| Claude Code CLI on a subscription        | the refund benchmark's `claude_cli_teacher.py` path (`benchmarks/refund/program/CLAUDE_CLI_TRAINING.md`)                                                                                                                                             | a logged-in `claude` CLI | subscription quota, about 4 to 6 s per request                                                                                                                                                                                                                                                          |
| No language model (complete constraints) | `npx semantscript train --teacher constraints` (as `npm run train` does in `examples/refund-service`)                                                                                                                                                | nothing                  | free; nine expressions in about 8 minutes on the GPU                                                                                                                                                                                                                                                    |

For this tutorial's expression a teacher must read the text (its constraints
do not cover every input: `--teacher constraints` stops with one such input
and the outputs the constraints admit), so the first two routes apply, or the
constraints with a language-model `[teacher.fallback]` for the inputs they
leave open ([teachers](teachers.md)). Before paying for a run, know that
this expression has not yet passed the
[held-out constraint check](training-pipeline.md#held-out-constraint-check)
at this size: 27 retrains of it at 192 cases and 8 to 32 epochs all failed
it (details below), so a run at `--cases 200 --epochs 4` will very likely end
with `train failed` and no release, and step 5 then has nothing to test.
Check `--estimate` first and budget for more cases or `examples`. With a key:

```sh
npx semantscript train --cases 200 --epochs 4 --device cuda
```

`npx semantscript doctor` in the same directory checks the teacher you
picked, including one request of well under USD 0.001. `train` itself first
runs the doctor's Python and teacher checks (about five seconds,
no billed request; a missing key or package stops it there with the fix),
finds the bundle under `dist/`, writes `.semantscript/teacher.toml`
when `ANTHROPIC_API_KEY` is set (or takes `--teacher`), generates the cases,
trains ModernBERT-base plus one head, fits the calibration temperature,
verifies the gold example and the constraints, and publishes
`.semantscript/artifact`. Expect the encoder download the first time (600 MB),
then roughly a minute on the GPU or half an hour on a CPU (`--device cpu`).
The report table names any verification failure with the failing cases.

What the Express example's own release shows (it exists on the development
machine only: `examples/*/.semantscript/` is git-ignored): `217d386c`
(2026-09-26, Sonnet 5 through
OpenRouter, `--cases 192 --epochs 8 --seed 5 --select-best-epoch
--counterfactual-ratio 0.5 --max-constraint-violation-rate 0.01`), verified
`decideRefund` at accuracy 0.9241, ECE 0.0719 and 0 of 394 corpus
violations, before the
[held-out constraint check](training-pipeline.md#held-out-constraint-check)
existed, and it answers `approve` for paid orders at 100 to 200 days. No
retrain from its cached datasets passes that check today: seeds 1 to 10 broke
28 to 109 of 512 held-out inputs (5.5% to 21.3%) and the best training-only
variant 20 of 512 (3.9%), against a 1% tolerance
([example README](../examples/express-app/README.md#no-retrain-from-the-cached-datasets-passes-the-held-out-check)).
If your run fails there, add the `examples` entry the failure suggests or
raise `--cases`; both call the teacher again, so check `--estimate` first.

To see the whole flow without any key on the clone route, run the reference
application instead, whose expressions are labeled by their own constraints
(clone only: `examples/refund-service` is not published):

```sh
cd ../refund-service && npm install && npm run build && npm run train && npm test
```

## 5. Test and call

```sh
npx semantscript test
npx semantscript run dist/refunds.sem.js --call decideRefund \
  --input '[{"tier":"standard","priorRefunds":1},{"total":88.5,"ageDays":12,"status":"paid"}]'
"approve"
```

`test` reports the shipped verification (including the `held-out` column,
the inputs of the held-out sample that broke a constraint), checks the
release's digests and replays the build's IR example through the runtime; `run` loads the artifact,
imports the compiled module and calls the export with the JSON arguments. That
is the end of the published route: import `decideRefund` from
`dist/refunds.sem.js` in your own code, after one `await loadSemaArtifact()` at
startup.

On the clone route, the Express example also has a server:

```sh
npm start        # clone only: POST /refunds/:orderId decides over PGlite and commits only on approve
```

The server loads the artifact once at startup with `loadSemaArtifact()` and
no path. The example's release `217d386c` packages to 653.9 MiB (570.9 MiB
of it the float32 artifact, 82.7 MiB `node_modules`) with `semantscript
package`, and `semantscript test` prints `-` in its `held-out` and `seed`
columns because it was published before those fields. A paid order at 100
days shows the answer the held-out check now refuses:

```sh
npx semantscript run dist/refunds.sem.js --call decideRefund \
  --input '[{"tier":"standard","priorRefunds":0},{"total":100,"ageDays":100,"status":"paid"}]'
"approve"
```

## What you have

A compiled function whose behavior lives in a 275 to 600 MB artifact beside
`dist/`, answers in milliseconds on a CPU with no network, and was verified
against its example and constraints before it was published. Changing the
text, examples or constraints and running `train` again retrains only that
expression's head through the [build cache](build-cache.md); `semantscript dev`
does it on every save. The [architecture overview](architecture.md) shows
what each step produced and the [Phase 1 results](phase-1-results.md) what
the same expression measures against generative baselines.
