# Getting started: one sema expression in an existing app

This tutorial starts from an Express application you already have, adds one
decision as a `sema` expression, and takes it through the CLI: `init`, `build`,
`train`, `test` and `run`, then the one line the server needs at startup. It
does not create a SemantScript project; sema is added to yours. A Next.js app
follows the same steps with the differences noted at the end.

## What you need

- **Node 22** and a TypeScript project compiled by `tsc` (the tutorial's
  case), Vite, esbuild or Next.js. The example is Express 5.
- **TypeScript 6** (`npm install -D typescript@6`). The packages accept
  5.9 to 6.x, but the `tsc` route's `ts-patch` 4 needs 6, and TypeScript 7
  is not supported yet: with 7 installed, the install below stops with an
  `ERESOLVE` peer-dependency error.
- **Python 3.12** with the trainer, `semantscript-trainer` from PyPI with
  its `training` extra (PyTorch, Transformers, ONNX and ONNX Runtime); step 2
  installs it. The CLI finds the interpreter through `--python`,
  `SEMANTSCRIPT_PYTHON` or `python3` (`python` on Windows); for a venv that
  is not activated, set `SEMANTSCRIPT_PYTHON` to its interpreter.
- **A GPU for training**, or patience: the encoder is ModernBERT-base, and
  fine-tuning one expression over a few hundred cases takes about a minute on
  a desktop GPU (an AMD RX 9070 XT here; NVIDIA through CUDA works the same)
  and tens of minutes on a CPU. Inference needs no GPU: the runtime is ONNX
  Runtime on the CPU, 4 to 30 ms per call depending on depth.
- **A teacher** to generate the training cases: either `ANTHROPIC_API_KEY`
  in the environment (the default is `claude-sonnet-5`, a few hundred short
  structured-output requests per expression) or a local Ollama server with
  a capable model. An expression whose `always`/`never` constraints decide
  every input needs neither: `npx semantscript train --teacher constraints`
  labels its cases from the constraints themselves (the
  [refund service](../examples/refund-service/README.md) trains all nine of
  its expressions that way; see [teachers](teachers.md)).
- About 700 MB of disk for the artifact (the encoder in ONNX) and as much
  again for the build cache.

`npx semantscript doctor` checks every item of this list and prints one line
each with the fix: the Node release and native bindings, the interpreter, the
trainer, PyTorch and the device it will train on with its memory, ONNX
Runtime, environment variables the platform needs, and the teacher with one
request of well under USD 0.001. The [environment guide](environment.md)
explains each line.

## 1. The app

A ticket API with a route that needs a decision. `src/server.ts`:

```ts
import express from "express";

const app = express();
app.use(express.json());

app.post("/tickets", (request, response) => {
  const { subject, body } = request.body as { subject?: string; body?: string };
  if (typeof subject !== "string" || typeof body !== "string") {
    response.status(400).json({ error: "subject and body are required" });
    return;
  }
  response
    .status(201)
    .json({ id: crypto.randomUUID(), subject, body, priority: "normal" });
});

app.listen(3000);
```

The priority is hard-coded. The rest of the tutorial replaces that one value
with a learned decision and leaves everything else alone.

## 2. Install the packages

All five packages share one version and are released together:

```sh
npm install semantscript @semantscript/core @semantscript/compiler @semantscript/framework
python3 -m venv .venv
.venv/bin/python -m pip install "semantscript-trainer[training]"
export SEMANTSCRIPT_PYTHON="$PWD/.venv/bin/python"
```

`semantscript` is the CLI, `@semantscript/core` the runtime the compiled
code imports, `@semantscript/compiler` the build-time transformer and
`@semantscript/framework` the optional controller layer (the
[framework guide](framework-guide.md); leave it out if you only call
expressions directly). `semantscript-trainer` is the Python trainer the CLI
starts for `train`. For a GPU, install the CUDA or ROCm build of PyTorch
from [pytorch.org](https://pytorch.org/get-started/locally/) into the venv
before the trainer; on Windows the interpreter is `.venv\Scripts\python.exe`.
Every artifact records the compiler and trainer versions that built it
(`build.compilerVersion` and `build.trainerVersion` in its manifest), and
`semantscript doctor` prints the trainer's.

> **First publish pending.** The release workflow that puts these packages
> on npm and PyPI is in place ([releasing](releasing.md)), but version 0.1.0
> has not been published yet, so the commands above return 404 until it is.
> Until then, install from a clone: `npm pack` each workspace of this
> repository (`npm pack -w runtime -w compiler -w framework -w cli` after
> `npm install && npm run build`) and `npm install` the four `.tgz` files, and
> install the trainer with `pip install '<clone>[training]'`; or follow the
> [tutorial](tutorial-refund-decision.md), which works from the clone.

## 3. Wire the compiler

```sh
npx semantscript init
```

`init` finds `tsconfig.json` (in a directory without one it starts a
TypeScript project first: `tsconfig.json`, `"type": "module"` and a `tspc`
build script, see [cli-reference](cli-reference.md#init)), adds the transformer entry and the editor
plugin entry to its `plugins`, adds `@semantscript/core` and
`@semantscript/compiler` to `package.json` if step 2 did not (at the CLI's
own version) with `ts-patch` and a `prepare` script, reserves the build
outputs under `.semantscript/` (the artifact, the cache and the `package`
bundle) in a `.gitignore`, and writes a starter
`src/hello.sem.ts` (pass `--no-example` to skip it). It ends with the
`doctor` checks under `environment (semantscript doctor):`, so a missing
Python package, GPU or teacher shows now; they send no billed request and
never fail `init`. Then:

```sh
npm install
```

`tsconfig.json` now carries:

```json
"plugins": [
  { "name": "@semantscript/compiler/ts-plugin" },
  { "transform": "@semantscript/compiler/transformer" }
]
```

and the build runs through ts-patch's `tspc` instead of `tsc`. In a new
project `init` wrote that `build` script; in an existing one whose `build`
script runs `tsc`, change it to `tspc -p tsconfig.json`. That is the whole adoption cost
for a tsc project; the [build tool pages](build-tools/tsc.md) cover the other
tools.

## 4. Write the expression

`src/triage.sem.ts` (the `.sem.ts` suffix is what the compiler looks for):

```ts
import { sema } from "@semantscript/core";

export type Priority = "urgent" | "normal" | "low";

/** Priority of a support ticket, decided by the trained artifact at request time. */
export function triage(subject: string, body: string): Priority {
  return sema<Priority>({
    examples: [
      {
        inputs: {
          subject: "Checkout is down",
          body: "Every customer sees a 500 since 9am",
        },
        output: "urgent",
      },
      {
        inputs: {
          subject: "Feature idea",
          body: "Could the export include a CSV option?",
        },
        output: "low",
      },
    ],
  })`
    Priority of a customer support ticket. "urgent" when service is down,
    data is lost or a deadline is today; "low" for questions and feature
    ideas; "normal" otherwise.
    Subject: ${subject}
    Body: ${body}
  `;
}
```

The type argument is the output's support: three string literals, so the
runtime can only ever answer one of them. The interpolations are the inputs,
by name. The text is the specification the teacher reads to generate cases.
The examples are gold cases: they are trained on and verification checks the
artifact against them, so `train` stops before generating anything when an
expression has none. Everything else the
[language reference](language-reference.md) allows (constraints,
`@confidence`, interface outputs) is optional; this is the minimum.

Use it in the route:

```ts
import { triage } from "./triage.sem.js";
// …
response.status(201).json({
  id: crypto.randomUUID(),
  subject,
  body,
  priority: triage(subject, body),
});
```

## 5. Build

```sh
npm run build          # tspc -p tsconfig.json
```

(`npx semantscript build` also compiles the project from its
`tsconfig.json`, without the build script, and lists each compiled
expression; see the [CLI reference](cli-reference.md).)

`dist/triage.sem.js` now contains a runtime call instead of the template, and
`dist/semantscript.ir.v1.json` is the IR bundle: one record for `triage` with
its inputs typed from TypeScript, its output support, the text, and the
execution plan (one stage, one function). A malformed expression fails the
build with a `TS91xx` diagnostic at the site; the
[diagnostics catalogue](diagnostics.md) lists them.

## 6. Train

Pick the teacher once, if `init` did not already ask:

```sh
npx semantscript init --teacher anthropic   # or openrouter, ollama, constraints
```

That writes `.semantscript/teacher.toml` with no key in it (the Anthropic
backend reads `ANTHROPIC_API_KEY`; for OpenRouter set it to the OpenRouter
key). Before the first paid run, check the teacher with one small request and
see what the run will cost:

```sh
npx semantscript teacher probe
npx semantscript train --estimate
```

The estimate prints each expression's teacher requests, tokens, USD cost and
time without calling the teacher; `--max-cost-usd <x>` then caps the real
run, stopping before the request that would pass the cap with everything
paid for kept for the next run ([teachers](teachers.md#cost-estimate-and-spend-cap)).
Then:

```sh
npx semantscript train
```

With `ANTHROPIC_API_KEY` set and no teacher file, `train` writes the default
Anthropic teacher file itself. A local model through Ollama costs nothing
(`--teacher ollama` at `init`, see [teachers](teachers.md)).
When the expression's constraints decide every input, no model is needed:
`npx semantscript train --teacher constraints` labels the cases from the
constraints (an input they leave open stops the build with that input named).
`triage` above has no constraints, so it needs a language-model teacher. Either
way `train` first runs the doctor's Python and teacher checks (about five
seconds, no billed request) and stops with the fix if one fails; then it finds
the bundle under `dist/`, generates cases through the
teacher (the default is 64 per expression; `--cases` raises it, and a few hundred is a sensible first setting), trains the
encoder and the head on the GPU (`--device cpu` to force the CPU), fits the
calibration temperature, verifies, and publishes a release under
`.semantscript/artifact`. It prints one line per step and a report table:
accuracy, ECE, Brier score, pair consistency, attested cases, constraint
violations. A function that fails verification is not published and the
command exits 1 with the failing cases named. A narrow miss on the violation
rate or the ECE first retrains with the next seed, up to three runs in all
([seed retry](cli-reference.md#seed-retry)), so a failing build can take up
to three times as long.

Expect about two minutes end to end on a desktop GPU for one expression: most
of it is the teacher requests and the encoder download the first time. Every
later `train` reuses the [build cache](build-cache.md): an unchanged
expression trains nothing, a changed one trains only its head.

## 7. Test and run

```sh
npx semantscript test --bundle dist/semantscript.ir.v1.json
```

reads the release's verification per function and, with `--bundle`, replays
every example in the IR through the runtime. Then call the function directly:

```sh
npx semantscript run dist/triage.sem.js --call triage --input '["Checkout is down", "Every customer sees a 500 since 9am"]'
"urgent"
```

`run` loads the artifact, imports the module, spreads the JSON array as the
arguments and prints the result. When an answer looks wrong, `semantscript
explain` takes the same arguments and shows why: the calibrated distribution,
the constraints active for the input, the nearest gold examples and training
cases, and the release it came from (see the
[wrong-answer workflow](diagnostics.md#wrong-answer-workflow)).

## 8. Load the artifact at startup

One line, before the server listens:

```ts
import { loadSemaArtifact } from "@semantscript/core";

await loadSemaArtifact();
app.listen(3000);
```

With no path the runtime uses `SEMANTSCRIPT_ARTIFACT` when set and otherwise
searches for `.semantscript/artifact` upward from the compiled entry and from
the working directory. A `sema` call before the load resolves throws
`SemaRuntimeNotLoadedError`, so the await matters. Deploying is shipping
`dist/`, `node_modules/` (installed on the target platform; the ONNX and
tokenizer bindings are prebuilt) and `.semantscript/artifact/` together, and
`npx semantscript package` writes exactly that directory with its size
([Deploying](deploy.md)); the
[runtime guide](../runtime/README.md#packaging-for-deployment) has the
container and serverless shapes.

```sh
npm start
curl -X POST localhost:3000/tickets -H 'content-type: application/json' \
  -d '{"subject":"Checkout is down","body":"Every customer sees a 500 since 9am"}'
```

## Iterating

`npx semantscript dev` runs build and train once and then again on every
save, retraining only what changed, and a server started with
`loadSemaArtifact(undefined, { watch: true })` swaps in each new release
without a restart. In the editor, the plugin shows the verified accuracy at
the expression and warns when it has changed since training.

The next things to add, in the order they usually pay off: `examples` (gold
cases the verifier must reproduce), `constraints` (rules the model may never
break, checked at release), and `@confidence(q)` with a fallback for answers
the application should not act on blindly. Constraints are not optional
polish for a policy with numbers in it: the trainer builds its boundary and
counterfactual cases from them, and the Express example's refund expression
went from 0.53 to 0.98 held-out accuracy on the same case budget when its
six rules became constraints. Write every rule you could write as an `if` as
a constraint, and keep the text for the judgment. The
[language reference](language-reference.md) covers each; the
[refund service](../examples/refund-service/README.md) is a whole application
built this way.

## Next.js instead of Express

`semantscript init` detects a Next.js project and edits `next.config.ts`
instead: a `turbopack.rules` entry for `*.sem.ts` with the loader,
`serverExternalPackages` for the runtime's native bindings and
`outputFileTracingIncludes` so a standalone build carries the artifact. The
expression goes in `lib/`, the route handler calls it, and the artifact loads
once per server process (`instrumentation.ts` is the idiomatic place). The
bundle lands beside `tsconfig.json` because Next.js has no `outDir`; `train`
finds it there. The [Next.js page](build-tools/next.md) and the
[example app](../examples/next-app/README.md) have the details and the one
known limitation (Turbopack's composed source maps).
