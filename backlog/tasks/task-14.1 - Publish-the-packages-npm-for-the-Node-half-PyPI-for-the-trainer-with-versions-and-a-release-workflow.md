---
id: TASK-14.1
title: >-
  Publish the packages: npm for the Node half, PyPI for the trainer, with
  versions and a release workflow
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-25 15:21'
updated_date: '2026-09-26 03:40'
labels:
  - dx
  - install
milestone: m-5
dependencies: []
parent_task_id: TASK-14
ordinal: 54000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
All four Node packages (@semantscript/core, @semantscript/compiler, @semantscript/framework, the semantscript CLI) and the Python project are private at version 0.0.0 and linked from source, so npx semantscript init, npm install @semantscript/core and pip install semantscript cannot work outside this repository; the examples use file: links and node ../../cli/dist/index.js. Nothing about the project is usable by someone who has not cloned it. The CLI spawns the Python trainer by module name and puts repository source paths on PYTHONPATH when it runs inside the repo, which also has to work from an installed package.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 npm install semantscript @semantscript/core @semantscript/compiler @semantscript/framework and pip install semantscript-trainer (or the chosen name) succeed from the public registries in an empty directory
- [ ] #2 npx semantscript init, build, train, test and run work in that directory against the installed packages with no repository checkout
- [ ] #3 Versions are shared across the packages, recorded in the artifact manifest, and a tagged release workflow publishes all of them from CI
- [ ] #4 The getting-started page and the tutorial use the published packages instead of file: links
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Findings (UNDERSTAND stage):
- All four Node workspaces (cli=semantscript, runtime=@semantscript/core, compiler=@semantscript/compiler, framework=@semantscript/framework) are private at 0.0.0; files/exports/bin already exist; cli pins internal deps at exact "0.0.0"; runtime depends on onnxruntime-node/tokenizers; compiler ships ts-plugin/.
- Python: root pyproject.toml, project name semantscript-python 0.0.0, packages semantscript_trainer + semantscript_model from trainer/src and model/src; heavy deps in the training extra. trainer cli.py hard-codes TRAINER_VERSION = "0.0.0" and --compiler-version defaults to "0.0.0"; the Node CLI train never passes --compiler-version, so manifests always say compilerVersion/trainerVersion 0.0.0. runtime loader defaults runtimeVersion to "0.0.0". trainer _git_commit() runs git in the package dir (from site-packages it returns the user's repo commit or 0000000).
- CLI pythonPath() only adds trainer/src, model/src, .python-packages when REPOSITORY_ROOT/trainer/src exists; installed under node_modules that check fails cleanly, so the installed trainer is used by module name. doctor's trainer fix text says pip install -e from the checkout.
- init writes core/compiler deps at the CLI's own package.json version.
- Names: npm semantscript and @semantscript/* and PyPI semantscript / semantscript-trainer all 404 (unclaimed) on 2026-09-25. No LICENSE file in the repo (license choice is the user's).
- Tooling here: no venv/ensurepip, no uv, no docker, no twine/build module; setuptools 84 present (pip wheel --no-deps works); verdaccio 6.x fetchable via npx.

Plan:
1. Shared version source: root package.json "version" (starting 0.1.0) is the one source. Add scripts/version.mjs with `set <semver>` (writes version into the 4 workspace package.json files, their internal @semantscript/* and semantscript dependency pins, pyproject.toml [project].version, trainer/src/semantscript_trainer/_version.py) and `check` (exits 1 naming each mismatch). Add npm scripts version:set / version:check; run version:check in a node test (scripts test or cli test) so CI catches drift without touching ci.yml.
2. Python: rename project to semantscript-trainer, version from _version.py (__version__), add readme, license, urls, classifiers, entry point `semantscript-trainer = semantscript_trainer.cli:main` if cli has main; TRAINER_VERSION imports __version__; semantscript_trainer exports __version__. Tests asserting 0.0.0 via defaults keep passing (explicit values) or updated.
3. Manifest versions: Node CLI train passes --compiler-version (version of the resolved @semantscript/compiler package.json, falling back to the CLI's own) unless the user passes it; trainer records trainer_version=__version__ in manifest build.trainerVersion and IR trainer.version. runtime loader default runtimeVersion = @semantscript/core package version (read via createRequire of ../package.json). Tests: cli test that the passed --compiler-version reaches the driver; python test that the default trainer version equals __version__ and lands in the manifest. _git_commit: return 0000000 unless the package dir is inside a SemantScript checkout (keep existing semantics, avoid recording the user's repo commit) - small, tested.
4. npm metadata on the four packages: drop private, version, license (from user decision; placeholder flagged), repository {type,url,directory}, homepage, bugs, description, keywords, publishConfig {access: public, provenance: true}, README per package already exists (check contents are install-oriented), engines node >=22.13. Keep files lists; verify with npm pack --dry-run that tarballs contain only dist/bin/ts-plugin/README/package.json.
5. doctor/init text: trainer fix line becomes `pip install 'semantscript-trainer[training]'` (checkout editable install kept as the contributor alternative); update matching tests and docs/diagnostics or environment pages where quoted.
6. Release workflow .github/workflows/release.yml (separate from ci.yml, no ci.yml edits): triggers push tags v*, and workflow_dispatch with input dry_run (default true). Jobs: verify (tag == root version via scripts/version.mjs check --tag), build+lint+test (reuse npm ci/build/test:node, python lint/tests on CPU), pack (npm pack each workspace, python -m pip wheel/sdist via `python -m build`), smoke (install tarballs + wheel in an empty temp dir, run npx semantscript --help / init / build / run with the stub artifact from 14.8), publish-npm (npm publish --provenance --access public in dependency order core, compiler, framework, cli, using NODE_AUTH_TOKEN=secrets.NPM_TOKEN; --dry-run when dry_run), publish-pypi (pypa/gh-action-pypi-publish with trusted publishing via id-token: write, or PYPI_TOKEN fallback; skipped in dry run, twine check instead). Refuse a real publish if LICENSE is missing. Exercise with workflow_dispatch dry_run on branch task-14.1 (gh workflow run release.yml --ref task-14.1 -f dry_run=true) - note: workflow_dispatch needs the file on the default branch to be listed, so also trigger dry run on push to branches matching release-dry-run/** or via a pull_request path filter; pick the push-on-branch trigger `release/**` or run the pack+smoke part on push of task-14.1 and verify with gh run watch.
7. scripts/release-smoke.sh (or .mjs): given a dir of tarballs + wheel, create a fresh temp dir outside any checkout, npm init, npm install the tarballs (or from a registry URL), pip install --target the wheel, then run npx semantscript init/build/test/run (and train when --train with --teacher constraints). Used by the workflow and the local proof.
8. Local proof of AC1/AC2: start verdaccio (npx verdaccio@6 with a scratch config, uplink npmjs for third-party deps, local-only user), npm publish the four packages to http://localhost:4873, build the wheel; in a fresh scratchpad dir with no checkout: npm install semantscript @semantscript/core @semantscript/compiler @semantscript/framework --registry local; pip install --no-deps --target <tmp>/py the wheel (host lacks venv; third-party Python deps come from .python-packages on PYTHONPATH, recorded as a caveat); run npx semantscript init, build, train (--teacher constraints, small cases, --local-files-only, GPU), test, run; record outputs and the manifest's compiler/trainer versions in implementation notes. No paid teacher calls.
9. Docs: getting-started.md install section -> npm install semantscript @semantscript/core @semantscript/compiler (+framework) and pip install 'semantscript-trainer[training]', with a note that the first publish is pending (until then, npm pack/clone route); tutorial-refund-decision.md gains a published-packages path (examples keep file: links for CI, note why); README Status section; cli-reference/diagnostics/environment where the trainer install command appears; new docs/releasing.md (version bump, tag, secrets, dry run) linked from docs/index.md and CONTRIBUTING. Prettier check.
10. Verify: npm run build, lint:node, test:node, python lint + tests, prettier; commit on task-14.1, push, gh run watch CI and the release dry run.
Needs user: npm org/scope @semantscript + NPM_TOKEN secret (automation token) and claim of unscoped `semantscript`; PyPI project semantscript-trainer with trusted publisher (or PYPI_TOKEN secret) and a `pypi` GitHub environment; license choice + LICENSE file; push tag v0.1.0 after merge. AC1, AC2 (public registries) and the publish half of AC3 stay unchecked until then.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
IMPLEMENT (2026-09-25):
- Shared version 0.1.0: root package.json is the source; scripts/version.mjs set/check/print writes and checks the 4 published workspaces, every internal pin (incl. benchmarks/refund), package-lock entries, trainer/src/semantscript_trainer/_version.py (pyproject.toml reads it via dynamic version) and runtime/src/version.ts + compiler/src/version.ts (exported as VERSION). cli/test/version.test.mjs runs check in CI and exercises set/check/--tag and PEP 440 spelling. npm install --package-lock-only left the lockfile identical to what set wrote.
- Manifest versions: cli train passes --compiler-version = installed @semantscript/compiler VERSION unless given (cli.test.mjs asserts default and override); trainer TRAINER_VERSION = __version__ (test_cli asserts manifest build.compilerVersion/trainerVersion; test_version.py); runtime default runtimeVersion = package VERSION (artifact-loader test). _git_commit only records a commit for a source checkout (installed copy inside a user repo records 0000000; tested).
- Python project renamed semantscript-trainer with readme, urls, classifiers, console script semantscript-trainer; doctor distribution name + fix lines now say pip install "semantscript-trainer[training]" (clone editable install kept as the alternative) in cli/src/doctor.ts and trainer doctor.py; tests updated.
- npm metadata: private removed; description, keywords, homepage, bugs, repository{directory}, engines, publishConfig.access=public. npm pack --dry-run: only dist/bin/ts-plugin/README/package.json. No license field: repo has no LICENSE (user decision).
- Bug found by the smoke run and fixed: semantscript build failed with 'planned 1 sema rewrites ... but matched 0' once ts-patch install had patched typescript (the setup init writes); cli/src/build.ts now creates its program without the tsconfig plugins.
- .github/workflows/release.yml (ci.yml untouched): verify (version/tag, LICENSE gate for real publish), test, pack (npm pack + python -m build + twine check), smoke (scripts/release-smoke.mjs on the packs, CPU train), publish-npm (dry-run unless v* tag), pypi dry-run / publish-pypi (trusted publishing or PYPI_TOKEN, env pypi), github-release. Dry run on any branch push touching release files.
- Local proof (AC1/AC2 mechanism, no public registry): verdaccio 6.10.4 on 127.0.0.1:4873 with the 4 packs of 0.1.0 published; wheel semantscript_trainer-0.1.0 installed with pip --no-deps --target (host has no venv; torch etc. from .python-packages minus its old semantscript copies). node scripts/release-smoke.mjs --registry http://127.0.0.1:4873/ --version 0.1.0 --device cuda --train-arg=--local-files-only in /tmp (no git checkout): npm install of the 4 packages, init --tool tsc --no-example --teacher constraints, build, train (96 cases, 3 epochs; accuracy 1.0, ece 0.0, 0 violations; USD 0), test passed, run -> true / false; manifest build compilerVersion 0.1.0, trainerVersion 0.1.0. Tutorial published route (npm pkg set, tsconfig, init, npm install, refunds.sem.ts, npm run build, semantscript build, train --estimate) also ran against verdaccio.
- Docs: getting-started (new step 2 Install + first-publish-pending note), tutorial (npm/PyPI route and clone route, file: links explained), new docs/releasing.md (linked from index and CONTRIBUTING), README Status, cli-reference/diagnostics/environment fix text, CONTRIBUTING python project name, trainer + package READMEs.
- Checks: npm run build ok; lint:node ok; test:node 345 pass; ruff check/format ok; pytest 709 passed 4 skipped; prettier --check docs README.md ok.

CI on task-14.1: CI run 36215282158 success (all jobs). Release dry run 36214993058 failed only in publish-npm (npm read release/x.tgz as a GitHub shorthand); fixed with ./release paths. Release dry run 36215282093 success in 4.3 min: verify, test (3.8 min), pack (twine check --strict PASSED wheel+sdist), smoke 3.4 min on hosted CPU (clean venv pip install of the wheel [training] from PyPI deps, npm install of the 4 tarballs in /tmp, init/build/train 96 cases acc 1.0/test/run true,false; manifest compilerVersion+trainerVersion 0.1.0), npm publish --dry-run of core/compiler/framework/cli OK, PyPI dry run OK; publish-pypi and github-release skipped as designed. Recorded in docs/releasing.md.
<!-- SECTION:NOTES:END -->
