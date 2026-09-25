---
id: TASK-14.8
title: Stub artifact for testing handlers without training
status: To Do
assignee: []
created_date: '2026-09-25 15:21'
labels:
  - dx
  - debug
milestone: m-5
dependencies: []
parent_task_id: TASK-14
ordinal: 61000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Application code that calls sema expressions cannot be unit-tested without a trained artifact: the runtime refuses calls before a load, and the only stand-in is the runtime's internal test fixture under runtime/test/fixtures, which the reference application's tests reach into by relative path and re-key by hand. A developer writing a Jest or node:test suite for a controller has no supported way to say what each expression should answer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 @semantscript/core/testing exports a way to create and load a stub artifact from the project's IR bundle in which each function answers a fixed value, a value per input, or a function of its inputs, with a confidence, without any model files
- [ ] #2 The framework's request scopes, pass counts, confidence policy and fallbacks behave with the stub exactly as with a real artifact, and the reference application's tests use it instead of the internal fixture
- [ ] #3 The framework guide documents testing a handler with the stub in under twenty lines
<!-- AC:END -->
