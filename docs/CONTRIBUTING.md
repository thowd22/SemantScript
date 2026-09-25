# Contributing

This page explains how the repository is laid out, how its two toolchains are
set up, how to run lint and tests for both halves, and how work is tracked.
Environment commands are also in [`DEVELOPING.md`](../DEVELOPING.md).

## Repository layout

The repository is one npm workspace for the Node half and one Python project
for the training half. Both halves share the specifications and schemas at the
root.

| Path                          | Toolchain | What lives there                                                                                                                                                                                                            |
| ----------------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SPEC.md`, `IR.md`            | prose     | The normative language and IR/artifact contracts.                                                                                                                                                                           |
| `schemas/`                    | JSON      | JSON Schemas (Draft 2020-12) for the IR, bundle, artifact, pointer and refund benchmark records.                                                                                                                            |
| `compiler/`                   | Node/TS   | `@semantscript/compiler`: finds `sema` sites in a TypeScript program, resolves types to IR, rewrites sites to runtime calls, emits the IR bundle and execution plan; ships the ts-patch, esbuild, Vite and loader adapters. |
| `runtime/`                    | Node/TS   | `@semantscript/core`: the public `sema` declarations, artifact loading, canonical input serialization, ONNX inference in a worker, calibration and confidence policy.                                                       |
| `cli/`                        | Node/TS   | `semantscript init \| build \| train \| dev \| test \| run`.                                                                                                                                                                |
| `framework/`                  | Node/TS   | `@semantscript/framework`: decorated controllers, per-request sema scopes and guards for Express, Nest and Next.js handlers.                                                                                                |
| `trainer/`                    | Python    | `semantscript_trainer`: datasets, adversarial cases, training, verification, verified IR, artifact export, build cache, teacher backends, the bundle driver.                                                                |
| `model/`                      | Python    | `semantscript_model`: encoder, adapter and head modules, calibration and ONNX export.                                                                                                                                       |
| `benchmarks/refund/`          | both      | The refund benchmark: TypeScript contracts, adapters and driver; Python release pipeline, teacher and experiments; committed data and results under `data/`.                                                                |
| `benchmarks/typed-decisions/` | Python    | The external typed-decisions suite compiled to sema programs, with results under `data/`.                                                                                                                                   |
| `examples/`                   | mixed     | Source fixtures, golden IR, manifest and serialization vectors, and the Express and Next.js adoption apps and the refund-service reference application (standalone packages, not workspaces).                               |
| `docs/`                       | prose     | This documentation and research notes.                                                                                                                                                                                      |
| `backlog/`                    | Markdown  | Backlog.md tasks, decisions and milestones (edited only through the `backlog` CLI).                                                                                                                                         |
| `scripts/`                    | Node      | `check.mjs` and `check-python.mjs`, the lint/test/build gates.                                                                                                                                                              |

The contract between halves is data, never imports: the compiler writes an IR
bundle the trainer reads, and the trainer writes an artifact the runtime loads.
Benchmark code consumes the public interfaces of the packages.

## The Node toolchain

Requirements: Node 22.13 or later and npm 10. From the repository root:

```sh
npm install        # installs and links the workspaces
npm run build      # tsc -b for compiler, runtime, cli, framework and the refund benchmark
npm run lint:node  # eslint (typescript-eslint strict, type-checked)
npm run test:node  # node --test in every workspace (each builds first)
```

Build before linting a fresh clone: the type-checked rules resolve each
workspace's imports of the others through their `dist/` declarations.
`npm run lint` and `npm run check` build first on their own. Prettier is not
part of `lint:node`; CI checks the docs with `npx prettier@3.9.9 --check docs
README.md`.

Package tests live in `<package>/test/*.test.mjs` and run against `dist/`, so
build before testing a single package (`npm test -w compiler` does both).
TypeScript is strict with `noUncheckedIndexedAccess` and
`exactOptionalPropertyTypes`; relative imports carry the `.js` suffix.

## The Python toolchain

Requirements: Python 3.12. The project is `semantscript-python` in
`pyproject.toml` with two source roots, `trainer/src` and `model/src`. Either
create a virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'            # ruff, pytest
.venv/bin/python -m pip install -e '.[dev,training]'   # plus torch, transformers, onnx, onnxruntime
# CPU only: install the pinned torch from the CPU index first, as CI does,
# instead of the multi-GB CUDA build pip picks on Linux x64:
# .venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

or, when the host Python lacks `venv`, install into the ignored local target
`.python-packages` (`pip install --upgrade --target .python-packages '.[dev]'`).
Never run a targeted install with `--upgrade` for a single package afterwards:
pip replaces the whole `.python-packages/bin` directory, which removes the
`ruff` binary the lint gate runs.

```sh
npm run lint:python   # ruff check and ruff format --check
npm run test:python   # pytest over trainer/tests, model/tests and benchmarks/refund/program
```

`scripts/check-python.mjs` picks the interpreter from `SEMANTSCRIPT_PYTHON`, an
active virtual environment, then `.venv`, and puts `trainer/src`, `model/src`
and `.python-packages` on `PYTHONPATH`. To run pytest or a module by hand use
the same layout:

```sh
PYTHONPATH=.:trainer/src:model/src:.python-packages python3 -m pytest trainer/tests -q
```

Tests that need PyTorch or ONNX skip themselves when the training extra is
absent. `node cli/bin/semantscript.js doctor` says which environment variables
a machine needs, and why, after re-running the imports with and without them:
on this project's WSL2 + ROCm machine it reports `PYTHONNOUSERSITE=1` as
needed (NumPy 2 in the user site breaks Transformers), and names
`HSA_ENABLE_DXG_DETECTION=1` only if ROCm finds the GPU only with it. The
[environment guide](environment.md) has the recorded run.

## Running everything

`npm run check` runs the build, both lints and both test suites in sequence
and never installs anything or touches the network. Run it before committing.

## Continuous integration

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every push,
every pull request and on demand, as four parallel jobs on `ubuntu-latest`
(the `python` job runs twice, once per install).
A newer push to a branch other than `main`, or to a pull request, cancels
that ref's run still in progress; every commit on `main` keeps its own run. No job uses
a secret or a teacher: the examples run on a fixture artifact.

| Job                       | What it runs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Reproduce locally                                                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `node`                    | `npm ci`, `npm run build`, `npm run lint:node`, `npm run test:node` (every workspace suite; the compiler's `docs-examples` test compiles every program under `docs/examples`), `npx prettier@3.9.9 --check docs README.md` (pinned, so a Prettier release cannot turn a run red). Python 3.12 is set up because the CLI and refund benchmark tests start `python3` fixture drivers.                                                                                                                                                                 | the same commands                                                                                                                         |
| `python` (`dev`)          | a `.venv` with `pip install -e '.[dev]'` (no training extra: proves the torch-free install and that every test needing PyTorch or ONNX skips cleanly), `npm run lint:python`, then `npm ci` and `npm run build` (some trainer tests drive `runtime/dist` and `cli/dist`) and `npm run test:python`.                                                                                                                                                                                                                                                 | the same commands in a clean virtual environment                                                                                          |
| `python` (`dev,training`) | the same, with the pinned `torch` installed first from the CPU wheel index (`--index-url https://download.pytorch.org/whl/cpu`) and then `pip install -e '.[dev,training]'`: every test that runs on a CPU. Only tests that need a live Ollama server, the ModernBERT CUDA smoke test, a network download of the pinned tokenizer, or a different install (the two environment-specific `test_primitives` cases) skip.                                                                                                                              | the same commands; a GPU is not used                                                                                                      |
| `fresh-install`           | the install exactly as the docs give it, with no npm cache: `npm install` and `npm run build` at the root, then `npm install`, `npm run build` and `npm test` in `examples/express-app` and in `examples/refund-service`, `npm run fixture-artifact` in the Express example, `npx --no-install semantscript --help` and the tutorial's `npx semantscript run` call there (fails if npm did not link the CLI bin), `docker build -f examples/express-app/deploy/Dockerfile .` and a smoke run of the image (`POST /tickets` and `POST /refunds/o1`). | the same commands; `npm run fixture-artifact` refuses to replace an existing artifact unless run as `npm run fixture-artifact -- --force` |

The example installs link `../../compiler`, `../../runtime` and
`../../framework` with `file:` dependencies, and those packages resolve their
own dependencies from the root `node_modules`, so the root `npm install` and
`npm run build` come first. The `node` and `python` jobs set
`ONNXRUNTIME_NODE_INSTALL=skip`, which stops `onnxruntime-node` from fetching
its optional CUDA provider on Linux x64; the `fresh-install` job does not, so
it installs with the defaults a new user gets, apart from the workflow-wide
`npm_config_fund=false` and `npm_config_audit=false`, which only silence npm's
funding and audit messages.

### Measured adoption cost

Wall times on GitHub-hosted `ubuntu-latest` runners, 2026-09-25, from
`gh run view --json jobs`. Run
[36158884114](https://github.com/thowd22/SemantScript/actions/runs/36158884114)
is the current workflow (four jobs), triggered by a pull request; its
`fresh-install` job restored no npm cache (its log shows
`package-manager-cache: false`). Run
[36157665549](https://github.com/thowd22/SemantScript/actions/runs/36157665549),
a push to `main` before the `dev,training` job existed, is the second column,
also with no npm cache. Two earlier green runs are not used for the install
figure: `actions/setup-node@v5` enables npm caching on its own when
`package.json` names a `packageManager`, and their `fresh-install` jobs
restored a warm 181 MB npm cache before the job turned it off.

| Measurement                                                     | Run 36158884114 | Run 36157665549 |
| --------------------------------------------------------------- | --------------- | --------------- |
| Whole workflow, trigger to last job finished (jobs in parallel) | 3 min 16 s      | 1 min 38 s      |
| `node` job                                                      | 1 min 39 s      | 1 min 34 s      |
| `python` (`dev`) job: 412 passed, 45 skipped                    | 1 min 29 s      | 1 min 5 s       |
| `python` (`dev,training`) job: 528 passed, 5 skipped            | 3 min 10 s      | (not yet run)   |
| `fresh-install` job, fresh clone to a smoke-tested image        | 1 min 37 s      | 1 min 32 s      |

The `dev,training` job sets the workflow's wall time: installing CPU PyTorch
and the rest of the training extra takes 54 s and its test run 1 min 41 s.

The `fresh-install` job's steps in run 36158884114, with no npm cache:

| Step                                                                | Time |
| ------------------------------------------------------------------- | ---- |
| Set up Node 22 from `.nvmrc`                                        | 5 s  |
| `npm install` and `npm run build` at the root                       | 14 s |
| Express example: `npm install`, `npm run build`, `npm test`         | 18 s |
| Refund service example: `npm install`, `npm run build`, `npm test`  | 17 s |
| Fixture artifact                                                    | <1 s |
| `npx semantscript --help` and `run` (step added in run 36160701373) | 1 s  |
| `docker build` (no layer cache)                                     | 36 s |
| Smoke run of the image                                              | 3 s  |

This measures the fresh-clone path to the in-repository examples, not the
[getting-started](getting-started.md) flow of adding SemantScript to your own
app with `npx semantscript init`, which CI does not run. A fresh clone to a
built, tested example is under a minute of commands on a hosted runner, and a
deployable image of the Express example about half a minute more. The image
is 624 MB. Its production `node_modules` is 384 MB (the same install
measured on the development machine), of which `onnxruntime-node` is 288 MB
because the package ships its native libraries for Linux, macOS and Windows
together; `tokenizers` is 64 MB and PGlite 26 MB. Training is not part of
this cost: it needs a teacher and preferably a GPU, and is measured in the
[tutorial](tutorial-refund-decision.md) and the
[teachers page](teachers.md).

## Work tracking

Tasks, plans, notes and decisions live in Backlog.md under `backlog/`. Use the
`backlog` CLI, never edit those Markdown files directly, and read
`backlog instructions overview` at the start of a session; the task-creation,
task-execution and task-finalization guides describe when to add a plan, notes,
acceptance-criteria checks and the final summary. Every acceptance criterion is
checked only with verification evidence (a test run, a command output, a
measured result), and follow-up tasks are proposed, not created, without the
project owner's approval.

## Conventions worth knowing

- Digests are SHA-256 over exact bytes; semantic values use the
  `semantscript.semantic-json/v1` encoding so `-0` and integral floats survive.
  Compare JSON by value with those helpers, not by string.
- Prompt text, examples and constraints never reach emitted JavaScript or a
  published artifact; tests assert this.
- Held-out benchmark inputs never enter training, and judge-adjudicated labels
  are never recorded as human-authored.
- Commits describe what changed and why in plain sentences; each Backlog task
  ends with a commit that records its Done state.
