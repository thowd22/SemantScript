---
id: TASK-15.3
title: >-
  semantscript test checks the artifact against the build's bundle and its
  digests by default
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-26 06:27'
updated_date: '2026-09-26 08:44'
labels:
  - dx
  - quality
milestone: m-6
dependencies: []
parent_task_id: TASK-15
priority: medium
ordinal: 68000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-14.11's reviewers confirmed that semantscript test without --bundle passes a stale artifact (the program changed since training, so the compiled function ids are absent from the release) and passes a release whose manifest or resource fails its digest check, because test only reads the recorded verification (cli/src/test-command.ts, cli/src/manifest.ts readArtifactSummary). run and explain already locate the build's IR bundle (dist/semantscript.ir.v1.json from the tsconfig outDir; see cli/src/explain.ts and cli/src/defaults.ts), and releases promote verifies digests through the runtime's checkSemaArtifact (cli/src/releases.ts verifyRelease). The failure messages go through diagnostics/remedies.json and scripts/generate-remedies.mjs (docs/CONTRIBUTING.md#generated-remedies).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Without --bundle, test finds the build's bundle the way run and explain do, and a bundle function absent from the artifact (or an artifact function the bundle no longer has) fails test with exit 1 and the next command; --no-bundle keeps the artifact-only behaviour, and a missing build says to run semantscript build
- [ ] #2 A release whose pointer, manifest or resource fails its digest or symlink check fails test with the runtime's ArtifactLoadError code and remedy instead of passing
- [ ] #3 cli tests cover both failures and the flag; docs/cli-reference.md, cli/README.md and docs/diagnostics.md are updated with the regenerated remedies; the CI fresh-install job still passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. cli/src/test-command.ts: add --no-bundle (boolean); --bundle with --no-bundle is a CliUsageError (exit 2). Order of checks: (a) readSummary as today (missing current.json -> test-no-artifact; unreadable pointer/release -> test-artifact-unreadable, existing tests unchanged); (b) new: dynamic-import checkSemaArtifact from @semantscript/core and run it on the root; an ArtifactLoadError (pointer/manifest/resource digest, symlink or non-regular file, incompatible ABI) is rethrown as 'artifact at <root> fails the runtime's load checks: <code>: <detail>; next: <error.remedy>' -> exit 1 with the runtime's own code and remedy (artifact-corrupt / artifact-path / ...), for --no-bundle too; (c) bundle: --bundle path, else --no-bundle -> none, else the first existing bundleCandidates(io.cwd) (tsconfig outDir, then ., dist, out, build; same lookup as explain/train); none found -> Error ending 'next: <remedy test-no-build>' (exit 1). (d) replay as today, plus the reverse direction: artifact functions whose id the bundle no longer has are listed ('<id>: in the artifact but not in the bundle'), make ok false and add next: <remedy test-function-unbundled>; --json gains bundle (path or null) and unbundledFunctions. Not using releases.verifyRelease: it rewraps ArtifactLoadError as RELEASE_REJECTED/RELEASE_INTEGRITY and also refuses unverified functions, which would hide the ArtifactLoadError code AC2 asks for; checkSemaArtifact alone already does pointer, manifest digest, resource size/digest and symlink checks (runtime/src/artifact-loader.ts selectRelease/readSafeFile/safeDirectory). No edit to releases.ts (15.5 in flight).
2. diagnostics/remedies.json: two new cli-family entries under distinct ids after test-artifact-unreadable: test-no-build (fix: run semantscript build, or pass --bundle <path>, or --no-bundle to check the artifact alone) and test-function-unbundled (fix: the program changed since training: run semantscript build, then semantscript train, and rerun semantscript test; or pass --bundle for the bundle this artifact was trained from). Existing entries left alone (15.4 owns wording). Run node scripts/generate-remedies.mjs (runtime/compiler generated TS, trainer remedies_generated.py, docs/diagnostics.md cli table) and --check.
3. cli/src/index.ts USAGE: test [--artifact <root>] [--bundle <path> | --no-bundle] [--json] with the new description.
4. cli/test/cli.test.mjs: existing artifact-only calls (stats, seeded, seededJson, unverified) get --no-bundle; new test 'test checks the artifact against the build bundle and its digests by default': project dir with tsconfig outDir build-output holding a bundle for the fixture function -> exit 0 with 1/1 examples and no --bundle; bundle with a different id -> exit 1 with absent + unbundled lines and both next: lines, JSON missingFunctions/unbundledFunctions; --no-bundle -> exit 0 with '-' examples; no build -> exit 1 naming semantscript build; --bundle + --no-bundle -> exit 2; tampered resource (same length) -> SEMA_ARTIFACT_INTEGRITY + artifact-corrupt remedy (also with --no-bundle); edited manifest -> SEMA_ARTIFACT_INTEGRITY manifest digest; symlinked current.json -> SEMA_ARTIFACT_PATH + artifact-path remedy.
5. Docs: docs/cli-reference.md (usage line + test section flags table: --bundle default, --no-bundle, digest check, exit conditions), cli/README.md (usage + test section), docs/diagnostics.md (regenerated table; one sentence that test reports ArtifactLoadError codes with the runtime remedy). Light touch on docs/getting-started.md and docs/tutorial-refund-decision.md sentences that say replay happens only with --bundle. npx prettier --check docs README.md cli/README.md.
6. Verify: npm run build, npm run lint:node, npm test -w cli (and npm run test:node), node scripts/generate-remedies.mjs --check, ruff on the regenerated python file, trainer test_remedies. Commit on task-15.3 with trailers, git push -u origin task-15.3, gh run watch until green (fresh-install job does not call semantscript test; release-smoke passes --bundle explicitly).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
IMPLEMENT: cli/src/test-command.ts now (1) reads the summary as before, (2) runs the runtime's checkSemaArtifact on verified releases and turns an ArtifactLoadError into 'artifact at <root> fails the runtime's load checks: <code>: <detail>; next: <remedy>' (exit 1; unverified releases skip it because the runtime refuses them first and the unverified next: line already applies), (3) resolves the bundle: --bundle, --no-bundle (none), else first existing bundleCandidates (tsconfig outDir, ., dist, out, build); none -> 'no semantscript.ir.v1.json under ...; next: <test-no-build>' (exit 1); --bundle with --no-bundle is a usage error (exit 2), (4) replays as before and lists artifact functions missing from the bundle ('<id>: in the artifact but not in the bundle', next: test-function-unbundled); --json gains bundle and unbundledFunctions. New remedies test-no-build and test-function-unbundled (causes kept under the existing column width so docs/diagnostics.md changes by two rows only); generated outputs regenerated. cli tests: existing artifact-only calls use --no-bundle; new test covers no build, --no-bundle, flag conflict, outDir bundle pass, stale ids both directions (text and JSON), tampered resource (SEMA_ARTIFACT_INTEGRITY + artifact-corrupt remedy, with and without --no-bundle), edited manifest digest, symlinked current.json (SEMA_ARTIFACT_PATH + artifact-path remedy). Docs: cli-reference test section and usage, cli/README usage and test section, diagnostics.md sentence + table, getting-started and tutorial now run plain 'npx semantscript test'. Checks: build, lint:node, test:node (all suites pass), generate-remedies --check, ruff, pytest test_remedies, prettier --check.

CI: first push (run 36224304780) failed in Python (dev,training): trainer/tests/test_cli.py run_cli_test ran 'test --artifact ... --json' from the repo root with no build output and now hit test-no-build; it asserts examples is null, so it now passes --no-bundle (commit 2). Local: pytest trainer/tests model/tests 656 passed, 4 skipped. Second push, CI run 36224612805: all 7 jobs green, including Fresh clone, examples and Docker image.

Review fix round 1: test now runs checkSemaArtifact on every release. For a release whose summary already shows an unverified function, only the runtime's 'manifest.functions[N].verification.status must equal "passed"' INVALID_MANIFEST refusal is suppressed, so a manifest hand-edited from passed to failed fails its manifest digest as SEMA_ARTIFACT_INTEGRITY with the artifact-corrupt remedy (the runtime checks pointer and manifest digests before validation; only resource digests are not reached for an unverified release). readSummary failures other than a missing release file (unparseable manifest, current.json symlinked to nothing) run the runtime check first and report its code. --bundle that does not read or is not a semantscript.ir-bundle now fails with the test-no-build next: line. Ids missing both ways print one next: line (test-function-unbundled). findBundle shares defaults.findBuiltBundle with resolveBundlePath. Docs corrected: the bundle lookup is the one train and explain use (not run), and the 'refused before any digest check' sentence now says only resource digests are skipped. Checks: npm run build, lint:node, npm test -w cli 62/62, test:node all pass, generate-remedies --check clean, prettier clean.

Review round 2 fix: a release manifest or release directory symlinked to nothing was treated as a simply missing file and reported a bare ENOENT with the rollback remedy. readSummary now counts a failure as simply missing only when no component between the artifact root and the failing path is a symlink (lstat walk, isSimplyMissing); otherwise it runs checkSemaArtifact and reports SEMA_ARTIFACT_PATH with the runtime remedy. New cli tests cover a dangling manifest symlink and a dangling release-directory symlink; both failed on the previous commit (61/1) and pass now (62/62). Also: explain uses the shared findBuiltBundle lookup; cli-reference exit code 2 row names --bundle with --no-bundle; getting-started says a changed program fails test once semantscript build has rebuilt it.

Review round 3 fix: a pointer naming a release that does not exist (digests disagreeing, SEMA_ARTIFACT_INVALID_POINTER, or consistent, SEMA_ARTIFACT_PATH cannot inspect release directory) and a deleted manifest.json printed a bare ENOENT with no runtime code. readSummary now runs the runtime's checkSemaArtifact in the simply-missing case too and reports its ArtifactLoadError code and detail; SEMA_ARTIFACT_PATH there keeps the rollback remedy (test-artifact-unreadable), any other code keeps the runtime's remedy. checkArtifact is split into loadCheckError/loadCheckFailure. New cli tests cover the three cases; the older dangling-releases test now expects SEMA_ARTIFACT_PATH instead of a bare ENOENT. docs/cli-reference.md and cli/README.md updated. npm test -w cli 62/62, npm run test:node all pass, lint and prettier clean, remedies --check clean.
<!-- SECTION:NOTES:END -->
