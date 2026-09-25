# Getting started: one sema expression in an existing app

This tutorial starts from an Express application you already have, adds one
decision as a `sema` expression, and takes it through the CLI: `init`, `build`,
`train`, `test` and `run`, then the one line the server needs at startup. It
does not create a SemantScript project; sema is added to yours. A Next.js app
follows the same steps with the differences noted at the end.

## What you need

- **Node 22** and a TypeScript project compiled by `tsc` (the tutorial's
  case), Vite, esbuild or Next.js. The example is Express 5.
- **Python 3.12** with the trainer installed: from this repository,
  `python3 -m pip install -e '.[dev,training]'` (PyTorch, Transformers, ONNX
  and ONNX Runtime); or `python3 -m pip install --target .python-packages
'.[dev,training]'` when the host Python has no `venv`. The CLI finds the
  interpreter through `--python`, `SEMANTSCRIPT_PYTHON` or `python3`.
- **A GPU for training**, or patience: the encoder is ModernBERT-base, and
  fine-tuning one expression over a few hundred cases takes about a minute on
  a desktop GPU (an AMD RX 9070 XT here; NVIDIA through CUDA works the same)
  and tens of minutes on a CPU. Inference needs no GPU: the runtime is ONNX
  Runtime on the CPU, 4 to 30 ms per call depending on depth.
- **A teacher** to generate the training cases: either `ANTHROPIC_API_KEY`
  in the environment (the default is `claude-sonnet-5`, a few hundred short
  structured-output requests per expression) or a local Ollama server with
  a capable model. No key and no local model means no `train`; the
  [refund service](../examples/refund-service/README.md) shows the third
  path, labeling from complete constraints, which needs neither.
- About 700 MB of disk for the artifact (the encoder in ONNX) and as much
  again for the build cache.

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

## 2. Wire the compiler

```sh
npx semantscript init
```

`init` finds `tsconfig.json`, adds the transformer entry and the editor
plugin entry to its `plugins`, adds `@semantscript/core` and
`@semantscript/compiler` to `package.json` with `ts-patch` and a `prepare`
script, reserves `.semantscript/` in a `.gitignore`, and writes a starter
`src/hello.sem.ts` (pass `--no-example` to skip it). Then:

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

and the build runs through ts-patch's `tspc` instead of `tsc`; change the
`build` script to `tspc -p tsconfig.json`. That is the whole adoption cost
for a tsc project; the [build tool pages](build-tools/tsc.md) cover the other
tools.

## 3. Write the expression

`src/triage.sem.ts` (the `.sem.ts` suffix is what the compiler looks for):

```ts
import { sema } from "@semantscript/core";

export type Priority = "urgent" | "normal" | "low";

/** Priority of a support ticket, decided by the trained artifact at request time. */
export function triage(subject: string, body: string): Priority {
  return sema<Priority>`
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
Everything the [language reference](language-reference.md) allows (examples,
constraints, `@confidence`, interface outputs) is optional; this is the
minimum.

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

## 4. Build

```sh
npm run build          # tspc -p tsconfig.json
```

`dist/triage.sem.js` now contains a runtime call instead of the template, and
`dist/semantscript.ir.v1.json` is the IR bundle: one record for `triage` with
its inputs typed from TypeScript, its output support, the text, and the
execution plan (one stage, one function). A malformed expression fails the
build with a `TS91xx` diagnostic at the site; the
[diagnostics catalogue](diagnostics.md) lists them.

## 5. Train

With `ANTHROPIC_API_KEY` set:

```sh
npx semantscript train
```

Without a key, write `.semantscript/teacher.toml` for a local model first
(`backend = "ollama"`, see the trainer guide) and run the same command. Either
way `train` finds the bundle under `dist/`, generates cases through the
teacher (the default is 64 per expression; `--cases` raises it, and a few hundred is a sensible first setting), trains the
encoder and the head on the GPU (`--device cpu` to force the CPU), fits the
calibration temperature, verifies, and publishes a release under
`.semantscript/artifact`. It prints one line per step and a report table:
accuracy, ECE, Brier score, pair consistency, attested cases, constraint
violations. A function that fails verification is not published and the
command exits 1 with the failing cases named.

Expect about two minutes end to end on a desktop GPU for one expression: most
of it is the teacher requests and the encoder download the first time. Every
later `train` reuses the [build cache](build-cache.md): an unchanged
expression trains nothing, a changed one trains only its head.

## 6. Test and run

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
arguments and prints the result.

## 7. Load the artifact at startup

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
tokenizer bindings are prebuilt) and `.semantscript/artifact/` together; the
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
the application should not act on blindly. The
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
