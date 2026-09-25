---
id: TASK-14.2
title: 'semantscript doctor: check and explain the environment before anything runs'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-25 15:21'
updated_date: '2026-09-25 17:53'
labels:
  - dx
  - install
milestone: m-5
dependencies: []
parent_task_id: TASK-14
ordinal: 55000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two toolchains have to be right before a build works: Node with the native ONNX and tokenizer bindings, and Python with the trainer, PyTorch, a usable device and a reachable teacher. Today nothing checks any of it; a missing trainer surfaces as a Python traceback from semantscript train, a missing GPU as a slow run, and the benchmark machine needs PYTHONNOUSERSITE=1 and HSA_ENABLE_DXG_DETECTION=1 that only memory notes record. init prints next steps but cannot tell the developer what is missing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 semantscript doctor reports, one line each with pass or fail and the fix, the Node version, the native runtime bindings for this platform, the Python interpreter the CLI will use, the trainer and model packages and their versions, PyTorch and the device it will train on (CUDA, ROCm, MPS or CPU) with its memory, ONNX Runtime, the teacher configuration (file found, key present in the environment, one-request probe succeeded), and any platform environment the run needs
- [ ] #2 init runs doctor at the end and train runs its Python and teacher checks first, so a missing piece fails in seconds with the fix named instead of failing minutes into a run
- [ ] #3 Windows, macOS and Linux (including WSL with ROCm) each have a doctor run recorded in the docs
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Findings (UNDERSTAND, 2026-09-25): CLI commands are (args, io: CliIo) => Promise<number> in cli/src, dispatched from COMMANDS in index.ts; usage errors are CliUsageError (exit 2), other failures exit 1. Python is resolved inline in train.ts (--python, SEMANTSCRIPT_PYTHON, python3/python); pythonPath() prepends trainer/src, model/src, .python-packages in the checkout. resolveTeacherConfig() writes .semantscript/teacher.toml as a side effect, so doctor needs a read-only lookup. Trainer device is auto|cpu|cuda only (ROCm through torch.cuda; no MPS). Teacher backends: anthropic (SDK reads ANTHROPIC_API_KEY; OpenRouter via base_url with the OpenRouter key in ANTHROPIC_API_KEY) and ollama (no key). No package __version__: versions come from importlib.metadata (semantscript-python) with a source-checkout fallback. Verified on this machine: torch 2.9.1+rocm7.2 sees the RX 9070 XT with or without HSA_ENABLE_DXG_DETECTION; with the user site on, importing transformers.modeling_utils fails (numpy.long, ~/.local SciPy/NumPy 2) and succeeds with PYTHONNOUSERSITE=1. onnxruntime-node loads its .node binding on import; tokenizers likewise.

1. Python side: trainer/src/semantscript_trainer/doctor.py, a closed JSON report (kind semantscript.doctor-report, version 1, checks [{id, status pass|fail|warn|skip, summary, fix}]) with ids python, trainer, model, torch, device, onnxruntime, platform-env, teacher-config, teacher-key, teacher-probe. Imports nothing heavy at module level; torch/transformers/onnxruntime are probed in subprocesses so a missing or broken package is a fail line, not a traceback. device: CUDA/ROCm (torch.version.hip) name and total/free memory, MPS reported as present but unused by the trainer (CPU), CPU with RAM and a slow-run warning. platform-env: when the import probe fails with the user site on and passes with PYTHONNOUSERSITE=1, fail with that fix; on WSL (/dev/dxg) with a HIP torch, rerun the device probe with HSA_ENABLE_DXG_DETECTION=1 when unset and name it if it changes the outcome. Teacher: load_teacher_config, key present (ANTHROPIC_API_KEY or api_key; OpenRouter base_url with only OPENROUTER_API_KEY set -> fail with the exact fix), probe_teacher(config) sending one minimal request (anthropic: 1 short message, max_tokens small, thinking disabled; ollama: tiny chat completion) reporting latency and usage; the probe function is reusable by TASK-14.5's teacher probe. Wire as 'python -m semantscript_trainer.cli doctor [--teacher X] [--probe|--no-probe] [--device D]' writing JSON to stdout. Tests: trainer/tests/test_doctor.py with fake subprocess results and a fake teacher client (no network).
2. Node side: cli/src/doctor.ts. Extract resolvePython(values, io) into defaults.ts (train.ts uses it) and add findTeacherConfig (read-only variant used by doctor and by resolveTeacherConfig). Node checks: node version (>= the engines/.nvmrc major 22), native bindings: resolve onnxruntime-node and tokenizers from @semantscript/core's location and import them (reports platform/arch and the fix: npm rebuild / reinstall without ONNXRUNTIME_NODE_INSTALL skip issues). Python checks: spawn the interpreter (missing -> fail naming --python/SEMANTSCRIPT_PYTHON), then spawn '<python> -m <trainer-module> doctor --json' with the same PYTHONPATH as train; validate the report with the closed objectOf/stringOf helpers; unknown ids or statuses are a contract error. Output: one line per check 'pass|fail|warn|skip  <id>  <summary>' plus '      fix: ...' for non-pass; --json prints the combined report. Flags: --python, --trainer-module, --teacher, --no-probe, --no-teacher, --runtime (Node and bindings only), --json. Exit 0 when no fail, 1 otherwise. Add to USAGE, COMMANDS and the index.ts exports.
3. train preflight (AC2): runTrain calls the doctor's Python and teacher checks (no paid probe; Ollama reachability is free) before spawning the trainer; any fail prints the lines and exits 1 in seconds. --no-preflight skips it. dev runs the preflight once at start, not on every rebuild. Extend cli/test/fixtures/fake_trainer.py with a 'doctor' subcommand driven by env vars (FAKE_DOCTOR_FAIL etc.) so the CLI tests stay hermetic; add tests: doctor renders pass/fail/fix lines and exit codes, contract rejection of a malformed report, missing interpreter, --runtime, train stops before the trainer on a failing preflight (argv record absent), --no-preflight.
4. init (AC2): after wiring, run doctor with --no-probe and print its lines under 'environment:'; init keeps exit 0 (the wiring succeeded) and says which fixes are needed before train. --no-doctor flag for init; existing init tests pass --no-doctor or a fake SEMANTSCRIPT_PYTHON, plus one test that init prints the doctor section.
5. CI: fresh-install job adds 'npx --no-install semantscript doctor --runtime' in examples/express-app (Node and bindings must pass on a fresh install); python job (dev,training) adds 'node cli/bin/semantscript.js doctor --python .venv/bin/python --no-teacher' (CPU device is a warn, not a fail).
6. Docs: docs/cli-reference.md (synopsis, doctor section, train --no-preflight, init --no-doctor, exit codes), cli/README.md (doctor section, train preflight), new docs/environment.md (what each check means, the fixes, the PYTHONNOUSERSITE/HSA_ENABLE_DXG_DETECTION explanation, recorded doctor runs per platform) linked from docs/index.md; getting-started.md and tutorial mention running doctor first; trainer/README.md and docs/teachers.md mention the probe; CONTRIBUTING.md points at doctor for the ROCm env. Prettier on every touched markdown.
7. Verification and recording: npm run build, npm run lint:node, npm run test:node, ruff check/format, pytest trainer/tests/test_doctor.py and test_cli.py. Record real runs on this WSL2 + ROCm machine: doctor with the Ollama teacher (free probe, qwen2.5 1.5B) and one OpenRouter probe (estimated well under USD 0.001, cap 0.10) using the express example's teacher.toml with ANTHROPIC_API_KEY set from .env for that process only (never printed); a run with the user site on to show the PYTHONNOUSERSITE fail line; a train with a missing interpreter/teacher failing in seconds. Paste the WSL output into docs/environment.md; mark Windows and macOS as not run.
8. Commit in slices, push to origin main, watch CI with gh run watch after each push; red is blocking.

Acceptance criteria expectation: #1 and #2 verifiable here; #3 cannot be demonstrated on this machine (Windows and macOS not available) and stays unchecked unless the user supplies those runs.
Risks: HSA_ENABLE_DXG_DETECTION is not required for torch on this machine today (only for the Ollama service), so doctor reports it only when it changes the outcome; MPS is not a trainer device (follow-up to add it); the teacher probe overlaps TASK-14.5's 'teacher probe' (shared probe_teacher function); train preflight adds a torch import (a few seconds) to every train/dev start; init now spawns Python, which changes its output in tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
IMPLEMENT: trainer/src/semantscript_trainer/doctor.py (closed doctor report, subprocess import probes, PYTHONNOUSERSITE and HSA_ENABLE_DXG_DETECTION detection by retrying the imports, teacher config/key/probe with a reusable probe_teacher) wired as 'python -m semantscript_trainer.cli doctor'; cli/src/doctor.ts (node, runtime-bindings, Python report validation, rendering), train/dev preflight (--no-preflight), init runs doctor (--no-doctor). Tests: trainer/tests/test_doctor.py 14 passed; cli suite 14 passed; npm run test:node all green; lint:node clean; ruff clean; pytest trainer/tests 397 passed 2 skipped.

Docs: new docs/environment.md (checks, detection of PYTHONNOUSERSITE/HSA_ENABLE_DXG_DETECTION, recorded WSL2+ROCm runs: full doctor with one OpenRouter probe of 16 in/4 out tokens (~USD 0.0001), user-site-on run, train preflight stop without a key in 5 s, missing interpreter in 0.3 s; Windows and macOS explicitly not run), linked from docs/index.md and README; cli-reference, cli/README, getting-started, tutorial, teachers, diagnostics, CONTRIBUTING, trainer/README, components, architecture updated. CI: fresh-install runs 'npx --no-install semantscript doctor --runtime'; python (dev,training) runs doctor against .venv with --no-teacher.

CI run 36164465531 (commit 31cf345) green on all four jobs; doctor --runtime passed in fresh-install, doctor against .venv passed in python (dev,training) with device warn (7 s). CI output recorded in docs/environment.md and the steps in CONTRIBUTING's CI table. Criterion #3: Windows native and macOS runs not recorded (not available here); Linux WSL2+ROCm and Ubuntu CI recorded.

FIX round 1 (commits 39c5f2e, d755100; CI runs 36166910601 and 36167400159 green on all six jobs):
- python: an interpreter older than 3.12 now fails the python check even when the trainer cannot import (PEP 695 syntax). The Node fallback compares the version with 3.12. Verified with a 3.11.9 shim: 'fail python Python 3.11.9 (./py311) is older than 3.12', exit 1. A CLI test covers it.
- Ollama free probe: name and name:latest are treated as one model. Real probe of model glm-4.7-flash against a server listing glm-4.7-flash:latest now passes. Tests cover the tagless, :latest and other-tag cases.
- venv: when the interpreter is defaulted and a .venv exists in cwd or a parent, failed python/trainer/model/torch/onnxruntime fixes name --python <venv> or SEMANTSCRIPT_PYTHON (doctor, init and the train preflight). The tutorial exports SEMANTSCRIPT_PYTHON for the venv it creates, and getting-started says the same.
- AC #3: new CI job doctor-platforms runs doctor on windows-latest (native, x64) and macos-latest (arm64). Each run covers the default interpreter (fails, and the fix names the venv), the activated venv (passes with a CPU device warning; Windows RAM from GlobalMemoryStatusEx; macOS reports MPS present but unused) and --runtime. Output from run 36166910601 is recorded in docs/environment.md. Not observed: the Windows 'set NAME=1' fix form (no platform variable was needed) and any GPU on Windows or macOS.
- Advisory fixes: init reports a doctor that breaks its contract instead of exiting 1. ANTHROPIC_AUTH_TOKEN counts as a key. A failing CUDA memory query fails device, not torch. The totals say '1 warning'. '<command> --help' exits 0. Usage and cli/README list --device, --trainer-module and init --python/--trainer-module. The user-site run was re-recorded from the current code with home paths shortened. The free-mode failure text says 'listing the Ollama models failed'.
- Checks: npm run build 0; lint:node 0; lint:python clean; test:node all suites 0 fail (cli 17/17); test:python 543 passed 4 skipped; prettier clean.

FIX round 2 (blocking: non-ASCII output on Windows code-page pipes).
- Verified the reviewer's claim first: in a scratch dir named 项目 with PYTHONIOENCODING=cp1252, doctor printed 'fail trainer ... UnicodeEncodeError' and skipped 8 checks.
- Fix, in both layers: doctor.py writes --json output with ensure_ascii=True; the human report reconfigures stdout with errors=backslashreplace; the nested import probe runs with PYTHONIOENCODING=utf-8 and decodes as UTF-8 with errors=replace. cli/src/doctor.ts sets PYTHONIOENCODING=utf-8 when it captures the Python doctor (doctor, init and the train preflight all use this path).
- Tests: a new CLI assertion (fake trainer mode FAKE_DOCTOR_RAW_TEACHER, non-ASCII dir '项目 café', PYTHONIOENCODING=cp1252) fails without the Node fix with the same UnicodeEncodeError and passes with it. A new pytest (test_cli_doctor_output_survives_a_code_page_stdout) fails without the Python fix and passes with it.
- Real rerun: doctor --probe free with an Ollama qwen3:14b teacher in <scratch>/项目 and <scratch>/café under PYTHONIOENCODING=cp1252: 12 passed, 0 failed, exit 0, with the paths printed intact.
- Advisory items also fixed: a silent-failing interpreter now says 'no error output'; the Windows default interpreter (python) is given in the usage text, cli-reference, diagnostics, environment guide and cli/README; ANTHROPIC_AUTH_TOKEN appears in the cli-reference, diagnostics and cli/README teacher-key text; the tutorial's lowercase sentence start is fixed, and the tutorial now says PYTHONNOUSERSITE matters only for the .python-packages route, not the .venv.
- Checks: build, lint:node, lint:python, test:node (cli 17/17) and test:python (544 passed, 4 skipped) are clean; prettier passes.
<!-- SECTION:NOTES:END -->
