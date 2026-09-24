---
id: TASK-7.8
title: >-
  Editor diagnostics: LSP/TS plugin surfacing accuracy and guidance at the sema
  site
status: Done
assignee:
  - '@claude'
created_date: '2026-09-20 19:41'
updated_date: '2026-09-24 22:04'
labels:
  - dx
milestone: m-3
dependencies:
  - TASK-7.4
  - TASK-7.7
parent_task_id: TASK-7
ordinal: 44000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The biggest usability lever: make tuning a neural function feel like fixing a type error. At each sema site the editor shows verified accuracy, ECE, pair-consistency, example and constraint counts, and actionable diagnostics such as 'no examples and accuracy below threshold'.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A TypeScript language-service plugin shows a hover at each sema site with accuracy, ECE, pair-consistency, and example/constraint counts from the latest artifact
- [x] #2 Diagnostics appear inline for unsupported types, missing inputs, below-threshold accuracy or ECE, and unverified expressions
- [x] #3 Works in VS Code with no extension beyond the TS plugin entry in tsconfig
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compiler: add @semantscript/compiler/ts-plugin, a TypeScript language-service plugin (tsconfig plugins entry { name: '@semantscript/compiler/ts-plugin', artifact?: path }). It plans the project's program per request (planSemaCompilationSync over the language service's program, reading source text from the service so unsaved buffers work), reads the artifact pointer and manifest under the configured or default .semantscript/artifact, and decorates getQuickInfoAtPosition (hover at a sema site: function id, verification status, accuracy, ECE, pair consistency, attested cases, example and constraint counts, confidence threshold; or 'not in the latest artifact') and getSemanticDiagnostics (compiler diagnostics 9100 to 9131 inline; warnings for a site absent from or unverified in the artifact, accuracy or ECE beyond the manifest's gate, plus guidance such as adding examples when there are none).
2. Tests: compiler/test/ts-plugin.test.mjs drives the plugin through ts.createLanguageService with a fixture project and a fixture artifact manifest (hover text, diagnostics for a malformed site, unverified site, below-threshold accuracy), and through a real tsserver process launched with the tsconfig plugins entry (open, quickinfo, semanticDiagnosticsSync), which is the path VS Code uses.
3. init wires the plugin entry into tsconfig for every tool; docs: compiler README section, CLI README, docs index, VS Code note (workspace TypeScript or bundled TS with the local plugin).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented compiler/src/ts-plugin.ts (SemaEditorState: plans the language service's program with planSemaCompilationSync reading text from the service so unsaved buffers work, caches the plan per Program, reads current.json and the manifest with an mtime-keyed cache; getQuickInfoAtPosition hover with short id, output type, input/example/constraint counts, confidence threshold and the artifact's status, accuracy, ECE, pair consistency, attested cases and constraint violations; getSemanticDiagnostics adds the compiler's 9100-9131 at the site plus warnings 9150 no artifact, 9151 not in the latest artifact, 9152 accuracy below accuracyThreshold (default 0.95), 9153 ECE above eceThreshold (default 0.1), each with guidance such as adding examples when there are none). Packaging: tsserver loads plugins with require() and resolves them by file layout rather than the exports map, so the entry is compiler/ts-plugin/index.js (CommonJS) which requires dist/ts-plugin.cjs compiled from src/ts-plugin.cts; tsconfig include now covers .cts. init writes the plugin entry into every project's tsconfig. Tests (compiler/test/ts-plugin.test.mjs): in-process through ts.createLanguageService (untrained, weak-accuracy, missing-from-artifact, ECE, configurable thresholds, ordinary hover outside sites, malformed-site hover and diagnostics, non-sema files untouched) and a real tsserver child process opened on a fixture with the tsconfig entry and the workspace as --pluginProbeLocations (the way VS Code launches it): quickinfo returns the plugin hover and semanticDiagnosticsSync returns the 9150 warnings at the site line. VS Code itself was not launched on this machine; the tsserver session is the same protocol path. lint:node clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the @semantscript/compiler/ts-plugin TypeScript language-service plugin, enabled by one tsconfig plugins entry that semantscript init now writes. At every sema site it shows a hover with the function's id and output type, input, example and constraint counts, confidence threshold and, from the latest artifact, verified status, accuracy, ECE, pair consistency, attested cases and constraint violations; inline it reports the compiler's diagnostics (unsupported types, missing or repeated inputs, invalid examples and constraints) and warnings for unverified expressions (no artifact, or absent from the latest release), accuracy below a threshold and ECE above one, each with guidance. Packaged as a CommonJS directory entry because tsserver resolves plugins by file layout and requires them. Verified by three tests: in-process language-service checks of hover text and every diagnostic, a malformed-site case, and a real tsserver session launched the way VS Code launches it (tsconfig entry plus the workspace as plugin probe location) answering quickinfo and semanticDiagnosticsSync; 278 Node tests pass, lint clean. VS Code was not launched on this machine; the tsserver session is the same protocol path.
<!-- SECTION:FINAL_SUMMARY:END -->
