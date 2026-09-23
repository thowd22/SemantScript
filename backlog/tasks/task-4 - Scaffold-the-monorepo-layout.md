---
id: TASK-4
title: Scaffold the monorepo layout
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-22 04:02'
labels:
  - infra
milestone: m-0
dependencies:
  - TASK-3
documentation:
  - DEVELOPING.md
modified_files:
  - .editorconfig
  - .gitattributes
  - .gitignore
  - .nvmrc
  - DEVELOPING.md
  - package.json
  - package-lock.json
  - tsconfig.json
  - tsconfig.base.json
  - eslint.config.js
  - pyproject.toml
  - scripts/check.mjs
  - scripts/check-python.mjs
  - compiler/README.md
  - compiler/package.json
  - compiler/tsconfig.json
  - compiler/src/index.ts
  - compiler/test/package.test.mjs
  - runtime/README.md
  - runtime/package.json
  - runtime/tsconfig.json
  - runtime/src/index.ts
  - runtime/test/package.test.mjs
  - cli/README.md
  - cli/package.json
  - cli/tsconfig.json
  - cli/src/index.ts
  - cli/test/package.test.mjs
  - trainer/README.md
  - trainer/src/semantscript_trainer/__init__.py
  - trainer/tests/test_smoke.py
  - model/README.md
  - model/src/semantscript_model/__init__.py
  - model/tests/test_smoke.py
  - benchmarks/README.md
  - examples/README.md
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PLAN.md section 4 defines the target layout (compiler, trainer, model, runtime, cli, benchmarks, examples). Nothing exists yet. A scaffold with tooling in place lets Phase 1 tasks start in parallel.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Directories compiler/, trainer/, model/, runtime/, cli/, benchmarks/, examples/ exist with a README stating each one's responsibility
- [x] #2 Node workspace (package.json, tsconfig) builds an empty compiler and runtime package
- [x] #3 Python project (pyproject) for trainer and model installs and runs an empty test
- [x] #4 A single top-level command runs lint and tests for both halves
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory PLAN.md, repository state, and available Node/Python toolchains. 2. Scaffold compiler, runtime, and CLI Node workspaces with strict shared TypeScript configuration, smoke exports/tests, and lint/build scripts. 3. Scaffold trainer and model Python packages in one installable pyproject with Ruff and pytest smoke coverage. 4. Add responsibility READMEs for compiler, trainer, model, runtime, CLI, benchmarks, and examples, plus root ignore/tooling files. 5. Provide one root check command that runs lint, builds, and tests both ecosystems; install in isolated local environments, run the complete check, obtain independent audits, and record acceptance evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Scaffolded explicit npm workspaces for compiler, @semantscript/core in runtime, and the semantscript CLI with strict shared TypeScript 6 configuration, type-aware ESLint, project references, built-in Node smoke tests, reproducible npm 10 lockfile, and clean prepack builds. Scaffolded one installable Python distribution with distinct trainer and model src packages, pinned Ruff and pytest tooling, importlib-mode smoke tests, and setuptools multi-root discovery. Added responsibility READMEs for all seven target directories, line-ending and editor policy, generated/ML artifact ignores, and cross-platform development instructions.

Validation evidence:
- npm ci installed 101 packages, audited 105, and reported zero vulnerabilities.
- npm run check is the single root gate and passed Node lint, Python Ruff lint/format, TypeScript project builds, three Node workspace smoke tests, and two Python smoke tests.
- Both Python modules import from the installed .python-packages target, not repository source paths.
- The WSL system lacks ensurepip/python3.12-venv; the documented project-local pip target fallback installed the same wheel and dev tools without changing system Python.
- A clean-tree npm pack --dry-run invokes prepack builds and includes valid JS/type entry points without build metadata or dangling declaration maps.
- Generated dist, cache, Python target, and environment files are ignored; package-lock.json and checked-in artifact examples remain tracked.
- Three independent agent audits completed and report no remaining Node, Python, acceptance, or cross-platform blocker.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @codex
created: 2026-09-22 03:42
---
TASK-4 started after TASK-2 completion. Node 22.22.0 and npm 10.9.4 are available; Python 3.12.3 is available. Python lint/test dependencies are not preinstalled, so verification will use a project-local virtual environment.
---

author: @codex
created: 2026-09-22 04:02
---
TASK-4 final verification passed. The only environment issue was missing Debian ensurepip support; the isolated .python-packages fallback is documented and exercised. Fresh npm ci, the unified check, installed-package imports, and clean-tree pack dry-runs all pass.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the SemantScript monorepo foundation for Node and Python: compiler/core/CLI workspaces, trainer/model packages, all required responsibility directories, reproducible pinned tooling, smoke coverage, package-safe builds, and one cross-platform npm run check gate. All four acceptance criteria and independent audits pass.
<!-- SECTION:FINAL_SUMMARY:END -->
