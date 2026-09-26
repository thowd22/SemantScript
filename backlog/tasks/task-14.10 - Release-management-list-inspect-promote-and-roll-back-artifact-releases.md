---
id: TASK-14.10
title: 'Release management: list, inspect, promote and roll back artifact releases'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-25 15:21'
updated_date: '2026-09-26 02:06'
labels:
  - dx
  - deploy
milestone: m-5
dependencies: []
parent_task_id: TASK-14
ordinal: 63000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every train publishes an immutable release under the artifact root and flips current.json, and the runtime hot-swaps on that pointer, but there is no command to see what releases exist, what each one verified, which is current, or to roll back to the previous one when a new release behaves worse in production. Today that is a manual edit of current.json.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 semantscript releases lists every release under the artifact root with its date, manifest digest, per-function verification summary and which one is current
- [x] #2 semantscript releases rollback (to the previous or a named release) rewrites the pointer atomically and a running process with watch enabled swaps to it
- [x] #3 semantscript releases prune removes releases older than a count or age while never removing the current one
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. cli/src/manifest.ts: factor the per-release reading out of readArtifactSummary into readReleaseSummary(root, releaseName) returning digest, createdAt (manifest.build.createdAt), application id/version and the existing per-function summaries; readArtifactSummary keeps its shape (plus createdAt) and delegates. Add readPointer(root) (strict: kind, pointerVersion 1, release matches releases/sha256-<64hex>, digests agree; symlinked or non-regular current.json rejected) and writePointer(root, digest): pointer JSON byte-identical to the Python exporter (sorted keys, indent 2, trailing newline), written to <root>/.current-<pid>-<random>.tmp with flag wx and mode 0o600, fsync'd, renamed over current.json, directory fsync'd (skipped on Windows), temp unlinked in finally.
2. New cli/src/releases.ts with releasesCommand(args, io) and subcommands:
   - list (default when no subcommand) [--artifact] [--json]: scan <root>/releases for entries named sha256-<64hex> only (skip .staging-*, .pruning-*, anything else; reject symlinked entries as invalid), read each manifest, verify sha256(manifest bytes) equals the directory digest, sort newest first by createdAt (tie-break digest), render a renderTable table: current marker, created, digest (12 chars), app@version, functions passed/total, min accuracy, max ece, total violations; unreadable/mismatching releases listed as invalid with the reason; --json prints kind semantscript.releases v1 with full per-function summaries and current.
   - show <release> [--json]: one release's per-function verification table (reuse the test command's columns; extract the row/header builder from test-command.ts into table.ts or manifest.ts so both use it).
   - rollback [<release>] [--dry-run] [--json]: target is the named release (full digest, sha256-<digest>, releases/sha256-<digest>, or a unique hex prefix of at least 7 chars; ambiguous or unknown is a usage/1 error) or, without one, the newest release created before the current one (by createdAt, then digest); refuse when no current pointer (without a name), when the target is already current, or when there is no older release. Before rewriting, verify the target: non-symlink directory, manifest bytes hash to its digest, kind is semantscript.application-artifact, every manifest resource path stays inside the release, is a regular non-symlink file with matching byteLength and sha256, and every function's verification status is passed (the runtime refuses anything else). Then writePointer. Output: from <old digest> to <new digest> and a note that watching processes reload. Rolling forward to a newer named release is the same operation (this covers the title's promote).
   - prune [--keep <n>] [--older-than <duration: Nd|Nh|Nm>] [--dry-run] [--json]: at least one of --keep/--older-than required (usage error otherwise); a release is removed only when it is not the one current.json names, (with --keep) not among the n newest valid releases, and (with --older-than) its createdAt is older than now minus the duration; invalid/unreadable releases are never removed (reported). Pointer read failure aborts prune (never guess the current one). Before each removal re-read current.json and skip if it now names that release; remove by renaming releases/sha256-X to releases/.pruning-<random> (atomic, so no half-deleted release looks valid) then rm -rf it. Report kept/removed and bytes freed.
   Typed errors: CliUsageError for flag mistakes; a ReleaseError (name ReleaseError, with a code such as RELEASE_NOT_FOUND, RELEASE_AMBIGUOUS, RELEASE_INTEGRITY, RELEASE_UNVERIFIED, RELEASE_NO_PREVIOUS, POINTER_INVALID) for exit 1.
3. cli/src/index.ts: register releases in COMMANDS, add usage lines, export releasesCommand and the pointer helpers.
4. Tests in cli/test/cli.test.mjs (or a new cli/test/releases.test.mjs beside it) using runtime/test/fixtures/artifact.mjs createFixtureArtifact with transformManifest setting distinct application.version and build.createdAt to make three releases: list output (current marker, dates, digests, per-function summary, newest first, staging dir ignored, tampered manifest shown invalid) and --json; show; rollback to previous and to a named/prefixed release; pointer bytes validate against the schema shape and no .current-*.tmp is left; no release removed; errors for unknown/ambiguous/already-current/tampered resource/failed verification; prune --keep 1 and --older-than keep the current release even when it is the oldest, skip invalid ones and staging dirs, --dry-run removes nothing; usage errors exit 2. Runtime swap test (AC2): loadSemaArtifact(root, {watch: true, onReload}) from @semantscript/core, then runCli releases rollback in-process, waitFor onReload with the previous digest and __sema.call succeeding, then closeSemaArtifact (pattern from runtime/test/watch-reload.test.mjs). The rollback command must not call loadSemaArtifact itself (it is process-global and would stop the watcher).
5. Docs: docs/cli-reference.md (synopsis, a releases section with flags, exit codes row), cli/README.md (releases section, command list), a short "Releases: list, roll back, prune" part in docs/build-cache.md or ir-and-artifact-reference release layout (note: after a rollback the next train re-exports the cache's last release and makes it current again, since the cache reuse check compares against current.json; prune frees disk but never touches the cache), docs/index.md CLI line; npx prettier --check docs README.md cli/README.md.
6. Verify: npm run build, npm run lint:node, npm test -w cli (and runtime), prettier; commit on task-14.10 with trailers, push, gh run watch CI.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented semantscript releases (list default, show, rollback, promote, prune) in cli/src/releases.ts; strict readPointer and atomic writePointer (wx temp, fsync, rename, dir fsync; bytes identical to the trainer exporter) in cli/src/manifest.ts; shared verification columns in cli/src/table.ts reused by test. New cli/test/releases.test.mjs (6 tests incl. watch-enabled runtime swap on rollback and promote) passes; npm test -w cli 28/28, lint clean.

Docs: cli-reference releases section with flags and error codes, cli/README releases section, build-cache 'Releases, rollback and prune' (cache re-publishes after a rollback), ir-and-artifact-reference note, docs/index, components and architecture command lists. Checks: npm run build ok, lint:node clean, test:node all pass (cli 28/28), prettier --check docs README.md cli/README.md clean; smoke list/show/rollback --dry-run against the main checkout's Express artifact (read-only).

Committed f5db75d and pushed task-14.10; CI run 36209257443 green (all jobs). AC not checked (finalization is a later stage).

Fix round 1 (4632b30, CI 36210081688 green): (1) rollback/promote now run the runtime's own loader checks: @semantscript/core exports checkSemaArtifact(path), which runs loadArtifact (schema, ABI compatibility, tensors, opsets, chain, digests) and the inference plan's ONNX container checks without starting sessions; verifyRelease calls it and refuses with RELEASE_REJECTED carrying the SEMA_ARTIFACT_* code. New test: a release with minimumRuntimeVersion 99.0.0 is refused by rollback and promote and current.json is unchanged. Remaining gap, documented: ONNX session start-up and app fallback registration are not checked, and the check uses the runtime's default load options. (2) The list test asserts the newest release's actual createdAt instead of a hardcoded 2026; passes with the clock shifted to 2027-01-15. Advisories addressed: prune refuses a pointer naming a missing/invalid release; prune re-reads the pointer after renaming and restores a release a concurrent rollback made current; the subcommand may follow flags; POINTER_INVALID no longer repeats its code and says naming a release repairs the pointer; docs: RELEASE_REJECTED, list checks only the manifest digest, --keep after rollback, docs/index.md and cli/README command lists and defaults, diagnostics.md release-management section, runtime README documents checkSemaArtifact. Checks: npm run build, npm run lint:node clean, npm run test:node 89/105/30/8/85 pass, prettier clean.

Finalization validation: npm run build ok; npm run lint:node clean; node --test cli/test/releases.test.mjs 8/8 pass (AC1: test 1 list newest first with dates, digests, per-function verification summary, current marker; list --json carries full per-function summaries; AC2: test 3 atomic pointer rewrite to previous/named release, no temp left, no release removed, test 8 loadSemaArtifact with watch:true swaps to the rolled-back release and calls succeed; AC3: test 6 prune --keep/--older-than never removes the current release, invalid ones or staging dirs); npm run test:node 89/105/30/8/85 pass; prettier --check clean; manual list/show against a scratch copy of the Express artifact.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added semantscript releases (list default, show, rollback, promote, prune) in cli/src/releases.ts with a strict pointer reader and an atomic pointer writer (temp file, fsync, rename) in cli/src/manifest.ts. Rollback/promote verify the target's digests and resources and run the runtime's own load checks (new checkSemaArtifact in @semantscript/core) before rewriting current.json; prune removes by count or age via an atomic rename, never touching the current release, invalid releases or staging dirs, and restores a release a concurrent rollback made current. Documented in cli-reference, diagnostics, build-cache, ir-and-artifact-reference, index, cli/README and runtime/README. Verified with cli/test/releases.test.mjs (8 tests incl. a watch-enabled runtime swap on rollback), npm run test:node all green, lint and prettier clean.
<!-- SECTION:FINAL_SUMMARY:END -->
