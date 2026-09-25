# Tutorial: from a fresh clone to a running refund decision

This is the end-to-end path through the repository as it stands: clone, build
both halves, compile the refund-decision expression, train its artifact, and
call it. Every command was run on the development machine described below;
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

## 1. Clone and build

```sh
git clone <this repository> semantscript && cd semantscript
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
node cli/bin/semantscript.js doctor --no-teacher   # the teacher is chosen in step 4
```

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
compiler through its `tsconfig.json`:

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

```sh
cd examples/express-app
npm install                    # file: links to ../../compiler, ../../runtime, ../../framework
npm run build                  # tspc -p tsconfig.json
ls dist/refunds.sem.js dist/semantscript.ir.v1.json
```

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
leave open ([teachers](teachers.md)). With a key:

```sh
npx semantscript train --cases 200 --epochs 4 --device cuda
```

`npx semantscript doctor` in `examples/express-app` checks the teacher you
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

To see the whole flow without any key, run the reference application instead,
whose expressions are labeled by their own constraints:

```sh
cd ../refund-service && npm install && npm run build && npm run train && npm test
```

## 5. Test and call

```sh
npx semantscript test --bundle dist/semantscript.ir.v1.json
npx semantscript run dist/refunds.sem.js --call decideRefund \
  --input '[{"tier":"standard","priorRefunds":1},{"total":88.5,"ageDays":12,"status":"paid"}]'
"approve"
npm start        # POST /refunds/:orderId decides over PGlite and commits only on approve
```

`test` replays the IR's example through the runtime and reports the shipped
verification; `run` loads the artifact, imports the compiled module and calls
the export with the JSON arguments. The server loads the artifact once at
startup with `loadSemaArtifact()` and no path.

## What you have

A compiled function whose behavior lives in a 275 to 600 MB artifact beside
`dist/`, answers in milliseconds on a CPU with no network, and was verified
against its example and constraints before it was published. Changing the
text, examples or constraints and running `train` again retrains only that
expression's head through the [build cache](build-cache.md); `semantscript dev`
does it on every save. The [architecture overview](architecture.md) shows
what each step produced and the [Phase 1 results](phase-1-results.md) what
the same expression measures against generative baselines.
