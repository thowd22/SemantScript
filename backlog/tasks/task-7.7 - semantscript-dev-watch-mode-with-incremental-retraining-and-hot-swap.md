---
id: TASK-7.7
title: 'semantscript dev: watch mode with incremental retraining and hot-swap'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-20 19:41'
updated_date: '2026-09-24 21:52'
labels:
  - cli
  - dx
milestone: m-3
dependencies:
  - TASK-7.2
parent_task_id: TASK-7
ordinal: 43000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Training is the part developers fear. In dev it should be invisible: edit an expression, the changed head retrains in the background using the content-addressed cache, and the running app picks up the new head without restart.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Saving a file retrains only the sema expressions whose IR changed
- [x] #2 The running runtime swaps in the new head without a process restart
- [x] #3 Progress and verification results are streamed to the terminal per expression
- [x] #4 A stale head remains in service until the new one passes verification
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Runtime: loadSemaArtifact(path, { watch: true, onReload, onReloadError }) watches the artifact root's current.json and reloads through the lifecycle queue; a failed reload keeps the previous artifact active (hot-swap without restart, stale head until the new release loads); closeSemaArtifact stops the watcher; watcher unref'd.
2. CLI: refactor build into a reusable compileProject; add 'semantscript dev' that runs build then train (trainer stderr streamed, report rendered per expression), watches the project's TypeScript sources with a debounce, reruns on change with one queued rerun while busy, keeps the train cache so only changed expressions retrain, and stops on an AbortSignal (CliIo.signal) or SIGINT.
3. Tests: runtime watch reload (new release swaps in, broken pointer keeps the old one); CLI dev loop with the fake trainer (initial run, edit triggers a rerun with the cache dir, abort ends the loop).
4. Docs: CLI README dev section, runtime README watch option, docs index.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Runtime: loadSemaArtifact(path, { watch, onReload, onReloadError }) watches the artifact root's current.json (fs.watch, non-persistent, 100 ms debounce) and reloads through the lifecycle queue; a failed reload keeps the previous artifact active; closeSemaArtifact and the active handle's close stop the watcher. CLI: build and train refactored into compileProject and runTrain (trainer now spawned asynchronously with stderr inherited so its per-expression progress lines stream; SIGINT wired to CliIo.signal); new 'semantscript dev' runs build+train, watches TypeScript sources under the project root with a debounce, queues one rerun while busy, and exits on abort or --once. Tests: runtime/test/watch-reload.test.mjs (second release swaps in via the pointer, old handle retired, broken pointer keeps the second release, closed runtime ignores further pointer changes); cli dev test with the fake trainer (initial cycle, a saved sema edit triggers cycle 2 with the cache dir and without --no-cache/--full, a file under dist/ does not, abort ends the loop with status 0, --once returns the train status and reports the previous artifact stays in service). lint:node clean; test:node 275 pass. AC1's 'only changed expressions retrain' relies on the TASK-7.2 build cache, which dev passes through unchanged (the fake trainer cannot exercise the cache itself).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added 'semantscript dev' and runtime hot-swap. dev runs build then train, watches the project's TypeScript sources with a debounce and reruns both on every save, queuing one rerun while busy; it passes the build cache through so only expressions whose IR changed retrain (TASK-7.2's cache, keyed by function id), streams the trainer's per-expression progress lines (stderr inherited from an asynchronous spawn) and renders the per-function report, and stops on Ctrl-C (SIGINT wired to CliIo.signal) or --once. The runtime's loadSemaArtifact(path, { watch: true }) watches current.json and reloads through the lifecycle queue: a successful release swaps in without a restart (onReload), a failed one leaves the previous artifact active (onReloadError), and since the trainer publishes only after verification passes, the stale head stays in service until the new one is verified and loaded. Verified by a runtime watched-reload test (swap, retired handle, broken pointer keeps the previous release, closed runtime ignores changes) and a CLI dev-loop test with the fake trainer (initial cycle, edit triggers cycle 2 with the cache, dist/ writes ignored, abort exits 0, --once returns the train status); 275 Node tests pass, lint clean.
<!-- SECTION:FINAL_SUMMARY:END -->
