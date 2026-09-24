---
id: TASK-7.5
title: 'Build-tool plugins: tsc transformer, Vite and esbuild'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-20 19:41'
updated_date: '2026-09-24 21:35'
labels:
  - compiler
  - dx
milestone: m-3
dependencies:
  - TASK-5.3
parent_task_id: TASK-7
ordinal: 41000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Adoption depends on sema compiling wherever TypeScript already compiles. A developer with an existing Express, Next or Nest app must not adopt a separate build pipeline. The compiler core from Phase 1 gets thin adapters for the common toolchains.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A ts-patch/ttypescript-compatible transformer compiles sema sites in a plain tsc project with one tsconfig entry
- [x] #2 A Vite plugin and an esbuild plugin compile sema sites with one line of build config each
- [x] #3 An existing Express app and an existing Next.js app in examples/ each adopt sema by adding the plugin and one expression, with no other changes
- [x] #4 Source maps point diagnostics and stack traces at the original .ts line
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compiler: split planSemaCompilation into a synchronous core plus digest step; add planSemaCompilationSync and createSemaProgramTransformer(program, plan) that validates the plan seal and returns the before-transformer; add emitSemaSourceFile(program, plan, sourceFile) returning code and inline source map for bundlers.
2. Compiler subpath modules: @semantscript/compiler/transformer (ts-patch program transformer, plans on factory creation, writes the IR bundle beside outDir, reports plan diagnostics through extras.addDiagnostic), @semantscript/compiler/esbuild (onStart plans, onLoad emits .sem.ts through TypeScript with the rewrite), @semantscript/compiler/vite (buildStart plans, transform emits, watchChange re-plans), @semantscript/compiler/loader (webpack/Turbopack loader for Next.js).
3. Tests: ts-patch tspc build of a fixture project via the tsconfig plugins entry; esbuild and Vite builds of fixtures with the one-line config; source-map assertions decoding the mappings back to the .sem.ts line and a node --enable-source-maps stack-trace check.
4. examples/express-app and examples/next-app: minimal existing apps plus the plugin line and one sema expression; README per example; verify the Express one end to end, Next.js as far as the toolchain allows without adding Next to the lockfile.
5. Docs: compiler README section, docs/index.md, CONTRIBUTING; lint, test, commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Compiler: planSemaCompilation split into a synchronous core (planSemaCompilationSync) plus createSemaProgramTransformer and emitSemaSourceFile; new build-tools.ts (loadSemaProjectBuild, planSemaProgramBuild, SemaBuildError) shared by four adapters exported as package subpaths: @semantscript/compiler/transformer (ts-patch), /esbuild, /vite, /loader (webpack/Turbopack). compiler/test/build-tools.test.mjs (10 tests) runs the real tspc, esbuild and Vite builds on a fixture and checks node --enable-source-maps stack traces land on src/main.sem.ts:4:10 for all three; malformed sites fail each build with the TS9100 location. Dev deps added: esbuild 0.28.2, vite 8.3.1, ts-patch 4.0.1, express 5.2.1, @types/express.

Verification: compiler/test/build-tools.test.mjs (10 tests, in npm test -w compiler: 80 pass) runs the real tspc, esbuild and Vite builds on a fixture, checks the bundle against the ir-bundle schema, and runs each output with node --enable-source-maps asserting the stack frame 'at classify (.../src/main.sem.ts:4:10)'; malformed sites fail each adapter with 'src/broken.sem.ts(3,23): error TS9100'. npm run lint:node clean; npm run test:node 269 pass. Examples: examples/express-app built with tspc from the tsconfig plugins entry (dist/triage.sem.js, maps, dist/semantscript.ir.v1.json); started against the refund multihead artifact, POST /tickets reached the compiled call and the SemaUnknownFunctionError trace shows src/triage.sem.ts:7:10. examples/next-app built with Next 16.3.6 (Turbopack and --webpack) after npm install in place; no prompt text in .next/; next start against the same artifact answered POST /api/triage through the compiled call (SemaUnknownFunctionError, expected: the refund artifact does not carry the triage function). Adoption cost recorded honestly: the plugin line plus one .sem.ts file plus one artifact-load call at startup (Express) or in the route module (Next, because Turbopack bundles the source-linked runtime per entry, so instrumentation.ts would load a different copy). Limitation: Turbopack's composed server maps keep the loader output's line (5) instead of the original (7) with or without sourcesContent; the loader's map itself is correct (unit test) and esbuild/Vite/tsc compose it. Packaging: runtime and compiler main exports gained a 'default' condition so require() (ts-patch, webpack, Next externals) can load them; new subpath exports resolve under import and require. Root devDependencies: esbuild 0.28.2, vite 8.3.1, ts-patch 4.0.1 (express removed again; the Express example is a standalone package).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added four build-tool adapters to @semantscript/compiler over a new synchronous planner (planSemaCompilationSync, createSemaProgramTransformer, emitSemaSourceFile, build-tools.ts): a ts-patch transformer (one tsconfig plugins entry), esbuild and Vite plugins (one plugins line each) and a webpack/Turbopack loader for Next.js, all writing the IR bundle and rewriting .sem.ts through TypeScript so source maps and diagnostics point at the original line. Verified by 10 new compiler tests that build a fixture with the real tspc, esbuild and Vite and assert node --enable-source-maps traces at src/main.sem.ts:4:10, plus two runnable examples (examples/express-app via tspc, examples/next-app via the loader) each built and served against a real artifact. Known limitation recorded: Turbopack does not compose the loader's map into its server chunks.
<!-- SECTION:FINAL_SUMMARY:END -->
