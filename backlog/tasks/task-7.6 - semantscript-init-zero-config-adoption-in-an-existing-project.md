---
id: TASK-7.6
title: 'semantscript init: zero-config adoption in an existing project'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-20 19:41'
updated_date: '2026-09-24 21:45'
labels:
  - cli
  - dx
milestone: m-3
dependencies:
  - TASK-7.1
  - TASK-7.5
parent_task_id: TASK-7
ordinal: 42000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A new user should go from an existing repo to a compiling sema expression in one command. Defaults for teacher, encoder, thresholds and artifact location must be sensible enough that no config file is required to start.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 npx semantscript init detects the project's build tool (tsc, Vite, Next, esbuild) and wires the matching plugin
- [x] #2 A project with no semantscript config builds and trains with documented defaults
- [x] #3 init adds the artifact directory to the correct place for the detected framework so the runtime finds it in dev and production
- [ ] #4 The whole flow from npm install to first working expression is under five minutes on the documented hardware
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. CLI: add 'semantscript init' (cli/src/init.ts): detect the build tool (next.config.* > vite.config.* > esbuild in package.json or a build script > tsconfig.json; --tool overrides), wire the matching adapter by editing the config text idempotently (tsconfig plugins entry + ts-patch prepare script; vite.config import and plugins entry; next.config turbopack rule, serverExternalPackages and outputFileTracingIncludes for .semantscript/artifact; esbuild build script plugins entry or printed snippet), add @semantscript/core and @semantscript/compiler to package.json, write .semantscript/.gitignore, and write one starter .sem.ts (src/, or lib/ for Next) unless one exists or --no-example; print the next steps.
2. Defaults so no config is needed: train defaults --bundle to the project's bundle location, --artifact to .semantscript/artifact, --teacher to teacher.toml or .semantscript/teacher.toml, generating the latter (anthropic backend) when ANTHROPIC_API_KEY is set and no file exists; test and run default --artifact the same way; runtime loadSemaArtifact() with no path resolves SEMANTSCRIPT_ARTIFACT then .semantscript/artifact under cwd.
3. Tests: cli.test.mjs init on tsc, Vite, Next and esbuild fixtures (edits, idempotence, invalid config left untouched), train/test/run defaults; runtime default-path test.
4. Docs: CLI README init section and defaults, docs index; measure the timed flow on the Express example as far as the teacher allows and record it honestly.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented cli/src/init.ts (detect next > vite > esbuild > tsc, idempotent text edits that keep comments, manual fallback with snippet when a config cannot be edited safely, package.json deps, .semantscript/.gitignore, starter hello.sem.ts) and cli/src/defaults.ts (bundle from tsconfig outDir then ./dist/out/build; artifact from SEMANTSCRIPT_ARTIFACT else .semantscript/artifact; teacher from three candidate files else a generated .semantscript/teacher.toml for the Anthropic backend when ANTHROPIC_API_KEY is set). Runtime: loadSemaArtifact() with no path resolves the same default (defaultSemaArtifactPath). Tests: three new CLI tests (tsc with comments and idempotence; Vite, Next, esbuild, conflicting next.config left untouched, bare project error; train/test/run defaults through the fake trainer) and a runtime default-path test. Timed flow on a fresh tsc project with linked packages (Ryzen 9 9900X): init 0.21 s, npm install 1.7 s (prepare ran ts-patch install), npm run build 0.5 s compiling the starter site to __sema.call with the bundle in dist/. Training could not be timed: no ANTHROPIC_API_KEY on this machine and the Ollama teacher is unusable for constrained generation, so AC4 stays unchecked.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added 'semantscript init' plus zero-config defaults. init detects the build tool (Next, Vite, esbuild, tsc; --tool overrides), wires the matching compiler adapter with idempotent, comment-preserving edits or reports a manual snippet, adds the packages, reserves .semantscript/{artifact,cache} (Next also gets outputFileTracingIncludes so standalone output carries the artifact) and writes a starter .sem.ts. train/test/run and the runtime's loadSemaArtifact() default to the build's bundle, .semantscript/artifact (SEMANTSCRIPT_ARTIFACT overrides) and a teacher file or a generated Anthropic teacher config when ANTHROPIC_API_KEY is set. Verified by four new tests (CLI init on all four tools, defaults through the fake trainer, runtime default path; 273 Node tests pass, lint clean) and a timed fresh-project flow: init 0.2 s, npm install 1.7 s, build 0.5 s to a compiled sema site. AC4 (whole flow under five minutes) is unchecked: training could not be timed without a teacher key on this machine.
<!-- SECTION:FINAL_SUMMARY:END -->
