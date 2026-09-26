# Releasing

One release publishes five packages at one version:

| Package                   | Registry | Directory                                        | What it is                                                     |
| ------------------------- | -------- | ------------------------------------------------ | -------------------------------------------------------------- |
| `semantscript`            | npm      | [`cli/`](../cli/README.md)                       | the CLI (`npx semantscript`)                                   |
| `@semantscript/core`      | npm      | [`runtime/`](../runtime/README.md)               | the runtime compiled code imports, and `/testing` stubs        |
| `@semantscript/compiler`  | npm      | [`compiler/`](../compiler/README.md)             | the transformer, build-tool adapters and editor plugin         |
| `@semantscript/framework` | npm      | [`framework/`](../framework/README.md)           | controllers, request scopes and guards                         |
| `semantscript-trainer`    | PyPI     | `trainer/src` and `model/src` (`pyproject.toml`) | the Python trainer and model packages the CLI runs for `train` |

The npm packages pin each other at that exact version, so one install always
resolves one release. The CLI hands the installed compiler's version to the
trainer, and every artifact's manifest records both (`build.compilerVersion`
and `build.trainerVersion`); the runtime compares its own version with the
manifest's `compatibility.minimumRuntimeVersion` when it loads an artifact.

> **Status: first publish pending.** Version 0.1.0 is prepared and the
> workflow below has run as a dry run, but nothing is on npm or PyPI yet. The
> steps under [first publish](#first-publish) are the owner's to do.

## One version

The root `package.json` `version` is the source. `scripts/version.mjs` writes
it everywhere else and checks that nothing drifted:

```sh
node scripts/version.mjs set 0.2.0     # every place below, then npm install
node scripts/version.mjs check         # exit 1 naming each mismatch
node scripts/version.mjs check --tag v0.2.0
node scripts/version.mjs print [python]
```

It keeps in step: the four published workspaces' `version`; every
`@semantscript/*` and `semantscript` dependency pin in the workspaces
(including the private refund benchmark); the matching `package-lock.json`
entries; `trainer/src/semantscript_trainer/_version.py`, which `pyproject.toml`
reads (`dynamic = ["version"]`); and `runtime/src/version.ts` and
`compiler/src/version.ts`, the `VERSION` each package exports. A pre-release is
`X.Y.Z-alpha.N`, `-beta.N` or `-rc.N`; the Python spelling follows PEP 440
(`0.2.0-rc.1` becomes `0.2.0rc1`) and npm publishes it under the `next`
dist-tag. `cli/test/version.test.mjs` runs `check` in CI on every push.

## Cutting a release

1. `node scripts/version.mjs set <version>`, `npm install`, `npm run check`.
2. Commit, merge to `main`, and wait for CI.
3. Tag and push: `git tag v<version> && git push origin v<version>`.

The tag starts [`.github/workflows/release.yml`](../.github/workflows/release.yml):

| Job              | What it does                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `verify`         | `version.mjs check --tag`; for a real publish, refuses to go on without a `LICENSE` file and a `license` in every package's metadata                                                                                                                                                                                                                                                                         |
| `test`           | `npm ci`, build, `lint:node`, `test:node`, then a `.venv` with CPU PyTorch and `.[dev,training]`, `lint:python` and `test:python`                                                                                                                                                                                                                                                                            |
| `pack`           | `npm pack` of the four workspaces, `python -m build` (wheel and sdist), `twine check --strict`; uploads them as the `release-<version>` workflow artifact                                                                                                                                                                                                                                                    |
| `smoke`          | [`scripts/release-smoke.mjs`](../scripts/release-smoke.mjs) against those packs in an empty directory outside the checkout: `npm install` of the four tarballs, the wheel with its `training` extra in a venv, then `semantscript init`, `build`, `train --teacher constraints --device cpu`, `test` and `run`, and a check that the manifest records this version as `compilerVersion` and `trainerVersion` |
| `publish-npm`    | `npm publish --provenance --access public` of the tarballs in dependency order (core, compiler, framework, cli), skipping a version already there so a rerun finishes a partial publish                                                                                                                                                                                                                      |
| `publish-pypi`   | uploads the wheel and sdist with `pypa/gh-action-pypi-publish`: trusted publishing, or the `PYPI_TOKEN` secret when set                                                                                                                                                                                                                                                                                      |
| `github-release` | a GitHub release for the tag with the packs attached                                                                                                                                                                                                                                                                                                                                                         |

## Dry runs

Every run that is not a pushed `v*` tag is a dry run: the same `verify`,
`test`, `pack` and `smoke` jobs, `npm publish --dry-run` for each tarball and
no PyPI upload. A push on any branch that changes the release machinery
(`release.yml`, `scripts/version.mjs`, `scripts/release-smoke.mjs`,
`pyproject.toml` or a `package.json`) runs one, and on `main` the workflow can
be started by hand:

```sh
gh workflow run release.yml --ref main                   # dry_run defaults to true
gh workflow run release.yml --ref v0.1.0 -f dry_run=false   # publish an existing tag again
```

The smoke script also runs locally, against tarballs or a registry:

```sh
npm run build
mkdir -p /tmp/packs && for w in runtime compiler framework cli; do npm pack -w $w --pack-destination /tmp/packs; done
python3 -m pip wheel --no-deps -w /tmp/packs .
python3 -m venv /tmp/trainer && /tmp/trainer/bin/pip install "$(ls /tmp/packs/*.whl)[training]"
node scripts/release-smoke.mjs --tarballs /tmp/packs --python /tmp/trainer/bin/python --device cuda
```

It prints each command, stops at the first failure with the directory kept,
and ends with a JSON summary: the installed versions, the answers `run` gave
and the manifest's `build` block. `--registry <url> --version <v>` installs
from a registry instead; `--no-train` stops after the build.

## Local proof

Run (the transcript below is an excerpt) on 2026-09-25 on the development machine (Node 22.22, Python 3.12, RX 9070
XT under ROCm), against a local [Verdaccio](https://verdaccio.org) 6.10.4
registry holding exactly the four `npm pack` tarballs of 0.1.0 (third-party
packages proxied from npmjs.org), and the wheel built by `pip wheel`. The host
Python has no `venv`, so the wheel went in with `pip install --no-deps
--target` and PyTorch, Transformers and ONNX came from the existing
installation; the full dependency resolve of `semantscript-trainer[training]`
from PyPI is what the workflow's `smoke` job proves.

```text
node scripts/release-smoke.mjs --registry http://127.0.0.1:4873/ --version 0.1.0 --device cuda --train-arg=--local-files-only
$ npm install semantscript@0.1.0 @semantscript/core@0.1.0 @semantscript/compiler@0.1.0 @semantscript/framework@0.1.0
$ npx --no-install semantscript init --tool tsc --no-example --teacher constraints
$ npx --no-install semantscript build
compiled 1 neural function(s) for application application
$ npx --no-install semantscript train --teacher constraints --cases 96 --epochs 3 --device cuda --local-files-only
  pass  trainer           semantscript_trainer 0.1.0 from <tmp>/py/semantscript_trainer
nf_145f313e…  src/hold.sem.ts  trained  96     192          1.0000        passed        1.0000    0.0000  3         0
train passed
$ npx --no-install semantscript test --bundle dist/semantscript.ir.v1.json
test passed
$ npx --no-install semantscript run dist/hold.sem.js --call hold --input [{"total":6200},"flag"]
true
$ npx --no-install semantscript run dist/hold.sem.js --call hold --input [{"total":88.5},"clear"]
false
release-smoke: passed
"manifestBuild":{"compilerVersion":"0.1.0","trainerVersion":"0.1.0",…}
```

The tutorial's published-packages route (its `npm pkg set`, `tsconfig.json`,
`init --tool tsc --no-example`, `npm install`, the refund expression and
`npm run build`, then `npx semantscript build` and `train --estimate`) ran the
same way against that registry.

## First publish

Everything above runs without credentials. The real publish needs, once:

1. **A license.** The repository has no `LICENSE` file yet; choose one, add
   it at the root, add `"license"` to the four `package.json` files and
   `license` to `pyproject.toml`'s `[project]`. The workflow refuses to
   publish without them.
2. **npm.** Create the `semantscript` organization on npmjs.com (it owns the
   `@semantscript` scope), publish rights for the unscoped `semantscript`
   name, and a granular access token that can publish both; store it as the
   repository secret `NPM_TOKEN`.
3. **PyPI.** Either add a trusted publisher for the project
   `semantscript-trainer` (owner `thowd22`, repository `SemantScript`,
   workflow `release.yml`, environment `pypi`) and create the `pypi`
   environment in the repository settings, or store an API token as the
   repository secret `PYPI_TOKEN` (the `pypi` environment is still used).
4. **Tag.** `git tag v0.1.0 && git push origin v0.1.0` on the merged `main`,
   then watch the run: `gh run watch $(gh run list --workflow release.yml -L 1 --json databaseId -q '.[0].databaseId')`.

Afterwards `npm view semantscript version` and
`pip index versions semantscript-trainer` show 0.1.0, and the note in the
[getting-started guide](getting-started.md) and the
[tutorial](tutorial-refund-decision.md) can go.
