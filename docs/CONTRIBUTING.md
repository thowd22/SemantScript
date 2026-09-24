# Contributing

This page explains how the repository is laid out, how its two toolchains are
set up, how to run lint and tests for both halves, and how work is tracked.
Environment commands are also in [`DEVELOPING.md`](../DEVELOPING.md).

## Repository layout

The repository is one npm workspace for the Node half and one Python project
for the training half. Both halves share the specifications and schemas at the
root.

| Path                 | Toolchain | What lives there                                                                                                                                                      |
| -------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SPEC.md`, `IR.md`   | prose     | The normative language and IR/artifact contracts.                                                                                                                     |
| `schemas/`           | JSON      | JSON Schemas (Draft 2020-12) for the IR, bundle, artifact, pointer and refund benchmark records.                                                                      |
| `compiler/`          | Node/TS   | `@semantscript/compiler`: finds `sema` sites in a TypeScript program, resolves types to IR, rewrites sites to runtime calls, emits the IR bundle and execution plan.  |
| `runtime/`           | Node/TS   | `@semantscript/core`: the public `sema` declarations, artifact loading, canonical input serialization, ONNX inference in a worker, calibration and confidence policy. |
| `cli/`               | Node/TS   | `semantscript build \| train \| test \| run`.                                                                                                                         |
| `trainer/`           | Python    | `semantscript_trainer`: datasets, adversarial cases, training, verification, verified IR, artifact export, build cache, teacher backends, the bundle driver.          |
| `model/`             | Python    | `semantscript_model`: encoder, adapter and head modules, calibration and ONNX export.                                                                                 |
| `benchmarks/refund/` | both      | The refund benchmark: TypeScript contracts, adapters and driver; Python release pipeline, teacher and experiments; committed data and results under `data/`.          |
| `examples/`          | mixed     | Source fixtures, golden IR, manifest and serialization vectors.                                                                                                       |
| `docs/`              | prose     | This documentation and research notes.                                                                                                                                |
| `backlog/`           | Markdown  | Backlog.md tasks, decisions and milestones (edited only through the `backlog` CLI).                                                                                   |
| `scripts/`           | Node      | `check.mjs` and `check-python.mjs`, the lint/test/build gates.                                                                                                        |

The contract between halves is data, never imports: the compiler writes an IR
bundle the trainer reads, and the trainer writes an artifact the runtime loads.
Benchmark code consumes the public interfaces of the packages.

## The Node toolchain

Requirements: Node 22.13 or later and npm 10. From the repository root:

```sh
npm install        # installs and links the workspaces
npm run build      # tsc -b for compiler, runtime, cli and the refund benchmark
npm run lint:node  # eslint (typescript-eslint strict, type-checked) and prettier
npm run test:node  # node --test in every workspace (each builds first)
```

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

`npm run check` runs both lints, the build, and both test suites in sequence and
never installs anything or touches the network. Run it before committing.

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
