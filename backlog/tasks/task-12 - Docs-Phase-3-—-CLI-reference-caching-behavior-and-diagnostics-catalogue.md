---
id: TASK-12
title: 'Docs: Phase 3 — CLI reference, caching behavior and diagnostics catalogue'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-20 02:47'
updated_date: '2026-09-25 03:12'
labels:
  - docs
milestone: m-3
dependencies:
  - TASK-7.1
  - TASK-7.2
  - TASK-7.3
  - TASK-7.4
ordinal: 37000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 3 is where external developers first touch the tool. The CLI, cache rules and error messages are the entire product surface for them, and undocumented behavior here becomes support load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A CLI reference documents every command, flag and exit code for build, train, test and run
- [x] #2 A page explains the build cache: what is content-addressed, when retraining happens, where the cache lives and how to clear it
- [x] #3 A diagnostics catalogue lists every compiler and verifier error with its cause and fix
- [x] #4 The universal-encoder investigation outcome is written up with its recommendation
- [x] #5 The getting-started tutorial is rewritten to use the CLI end to end
- [x] #6 The getting-started tutorial starts from an existing Express or Next.js app and adds one sema expression, not from a fresh SemantScript project
- [x] #7 A page per supported build tool documents plugin setup and known limitations
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. docs/cli-reference.md: every command (init, build, train, dev, test, run), every flag from the CLI source (build-tool detection, routing flags, trainer pass-through options, defaults and environment variables), exit codes (0, 1 failure, 2 usage) and what each command reads and writes.
2. docs/build-cache.md: what is content-addressed (function id and semantic digest, recipe digest, dataset digests, shared encoder and per-domain adapter state, head state, verified IR), when retraining happens (no change, one expression, new domain, recipe change, --full, --no-cache), where the caches live (.semantscript/cache/applications/<id>, dataset and adversarial caches) and how to clear them.
3. docs/diagnostics.md: catalogue of compiler diagnostics 9100-9131 with cause and fix (from the snapshot fixture and source), verifier gate failures (gold example misses, constraint violations above tolerance, type errors, ECE), trainer bundle errors, and runtime error classes and codes (load, input, inference, confidence, fallback, framework requirement).
4. docs/universal-encoder.md: the TASK-7.3 investigation written up from results-warm-start-2026-09-24 with decision-9's recommendation.
5. docs/getting-started.md: from an existing Express app, add one sema expression, then semantscript init, build, train (Anthropic or Ollama teacher, hardware and time stated), test, run and the artifact load at startup, every command copy-pasteable; the Next.js variant noted.
6. docs/build-tools/{tsc,esbuild,vite,next}.md: setup, defaults, source maps, diagnostics and known limitations per adapter.
7. docs/index.md links; verify with prettier, lint, the docs example test and a link check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added docs/cli-reference.md (init/build/train/dev/test/run flags from cli/src, defaults and environment, exit codes 0/1/2 from index.ts), docs/build-cache.md (layout under .semantscript/cache/applications/<id> with shared.safetensors, adapters/<ref>.safetensors, functions/<id>/{function.json,verified-ir.json,head.safetensors}; keys: function id, semantic digest, model binding, recipe, shared-state and dataset digests, file digests, cacheVersion; retrain matrix incl. new-domain adapters and --full/--no-cache; clearing), docs/diagnostics.md (TS9100-9131 with cause and fix from the snapshot fixture and source, editor 9150-9153, TrainBundleError/teacher/verifier gate failures/VerificationConfigurationError/export and training errors, runtime error classes and codes incl. ArtifactLoadError and SemaInferenceError code unions), docs/universal-encoder.md (results-warm-start-2026-09-24 tables and decision-9), docs/getting-started.md (Express app + one expression through init/build/train/test/run and loadSemaArtifact, requirements and timings stated, Next.js variant), docs/build-tools/{tsc,esbuild,vite,next}.md (setup, options, build behaviour, known limitations incl. Turbopack map composition and native runtime), docs/index.md links. Verified: prettier clean, zero broken relative links across docs, docs examples test passes, lint:node clean. Not verified: the tutorial was not executed end to end on this machine because train needs ANTHROPIC_API_KEY or an Ollama model; its commands are those of examples/express-app and the CLI tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 3 documentation: a CLI reference (every command, flag, default, environment variable and exit code from the source), a build-cache page (content addressing, retraining rules, layout, clearing), a diagnostics catalogue (compiler 9100-9131, editor 9150-9153, trainer and verifier failures, runtime error codes, each with cause and fix), the universal-encoder investigation write-up with decision-9's recommendation, a getting-started tutorial that starts from an existing Express app and uses the CLI end to end (Next.js variant noted), and one page per build tool with setup and known limitations; all linked from the index. Verified with prettier, a link check over docs, the docs examples test and lint; the tutorial's train step is documented as requiring a teacher and was not run here for lack of an API key.
<!-- SECTION:FINAL_SUMMARY:END -->
