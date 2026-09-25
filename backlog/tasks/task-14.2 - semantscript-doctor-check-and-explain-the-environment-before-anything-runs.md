---
id: TASK-14.2
title: 'semantscript doctor: check and explain the environment before anything runs'
status: To Do
assignee: []
created_date: '2026-09-25 15:21'
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
