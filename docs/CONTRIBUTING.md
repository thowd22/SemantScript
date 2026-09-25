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
absent. GPU jobs on this project's ROCm machine additionally need
`PYTHONNOUSERSITE=1 HSA_ENABLE_DXG_DETECTION=1`; details that only apply to one
machine are kept out of the repository.

## Running everything

`npm run check` runs the build, both lints and both test suites in sequence
and never installs anything or touches the network. Run it before committing.

## Continuous integration

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every push,
every pull request and on demand, as three parallel jobs on `ubuntu-latest`.
A newer push to a branch other than `main`, or to a pull request, cancels
that ref's run still in progress; every commit on `main` keeps its own run. No job uses
a secret or a teacher: the examples run on a fixture artifact.

| Job             | What it runs                                                                                                                                                                                                                                                                                                                                                                                                   | Reproduce locally                                                                                                                         |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `node`          | `npm ci`, `npm run build`, `npm run lint:node`, `npm run test:node` (every workspace suite; the compiler's `docs-examples` test compiles every program under `docs/examples`), `npx prettier@3.9.9 --check docs README.md` (pinned, so a Prettier release cannot turn a run red). Python 3.12 is set up because the CLI and refund benchmark tests start `python3` fixture drivers.                            | the same commands                                                                                                                         |
| `python`        | a `.venv` with `pip install -e '.[dev]'` (no training extra, so every test that needs PyTorch or ONNX skips), `npm run lint:python`, then `npm ci` and `npm run build` (some trainer tests drive `runtime/dist` and `cli/dist`) and `npm run test:python`.                                                                                                                                                     | the same commands in a clean virtual environment                                                                                          |
| `fresh-install` | the install exactly as the docs give it, with no npm cache: `npm install` and `npm run build` at the root, then `npm install`, `npm run build` and `npm test` in `examples/express-app` and in `examples/refund-service`, `npm run fixture-artifact` in the Express example, `docker build -f examples/express-app/deploy/Dockerfile .` and a smoke run of the image (`POST /tickets` and `POST /refunds/o1`). | the same commands; `npm run fixture-artifact` refuses to replace an existing artifact unless run as `npm run fixture-artifact -- --force` |

The example installs link `../../compiler`, `../../runtime` and
`../../framework` with `file:` dependencies, and those packages resolve their
own dependencies from the root `node_modules`, so the root `npm install` and
`npm run build` come first. The `node` and `python` jobs set
`ONNXRUNTIME_NODE_INSTALL=skip`, which stops `onnxruntime-node` from fetching
its optional CUDA provider on Linux x64; the `fresh-install` job does not, so
it installs with the defaults a new user gets.

### Measured adoption cost

Wall times on GitHub-hosted `ubuntu-latest` runners, 2026-09-25, from
`gh run view --json jobs`. Run
[36157665549](https://github.com/thowd22/SemantScript/actions/runs/36157665549)
has every job green and a `fresh-install` job that restored no npm cache
(its log shows `package-manager-cache: false`). The `fresh-install` job of
run [36155452753](https://github.com/thowd22/SemantScript/actions/runs/36155452753)
also found no npm cache and passed. Two earlier green runs are not used for
the install figure: `actions/setup-node@v5` enables npm caching on its own
when `package.json` names a `packageManager`, and their `fresh-install` jobs
restored a warm 181 MB npm cache before the job turned it off.

| Measurement                                                  | Run 36157665549 | Run 36155452753 |
| ------------------------------------------------------------ | --------------- | --------------- |
| Whole workflow, push to last job finished (jobs in parallel) | 1 min 38 s      | (`node` failed) |
| `node` job                                                   | 1 min 34 s      | (failed)        |
| `python` job (412 passed, 45 skipped)                        | 1 min 5 s       | 1 min 26 s      |
| `fresh-install` job, fresh clone to a smoke-tested image     | 1 min 32 s      | 1 min 50 s      |

The `fresh-install` job's steps in run 36157665549, with no npm cache:

| Step                                                               | Time |
| ------------------------------------------------------------------ | ---- |
| Set up Node 22 from `.nvmrc`                                       | 4 s  |
| `npm install` and `npm run build` at the root                      | 13 s |
| Express example: `npm install`, `npm run build`, `npm test`        | 18 s |
| Refund service example: `npm install`, `npm run build`, `npm test` | 16 s |
| Fixture artifact                                                   | 1 s  |
| `docker build` (no layer cache)                                    | 32 s |
| Smoke run of the image                                             | 2 s  |

The adoption flow a new developer follows, a fresh clone to a built, tested
example, is therefore under a minute of commands on a hosted runner, and a
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
