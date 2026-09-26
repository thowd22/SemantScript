---
id: doc-1
title: Handoff 2026-09-26
type: guide
created_date: '2026-09-26 19:54'
updated_date: '2026-09-26 19:54'
---
# Handoff, 2026-09-26

Written at the end of the session that delivered Phases 5 and 6 so the next session can continue without the conversation.

## Where things stand

- Phases 0 to 6 are on `main` and CI is green (`.github/workflows/ci.yml`, seven jobs).
- The Express example (`examples/express-app`) serves release `f1d1aaac` (seed 3, constraints teacher with Sonnet 5 fallback, teacher range 0 to 730 days): refund accuracy 0.9871, ECE 0.0117, 1 of 778 corpus and 2 of 512 held-out constraint violations, `deny` on 32 stale-order probes out to 1,000 days. It is the only release on disk (`releases prune --keep 0`). `.semantscript/` is git-ignored, so it exists on the development machine only.
- Next task the user asked for: **TASK-16** (milestone m-7), runtime enforcement of `always`/`never` constraints so rule violations become impossible by construction. Not started.
- Waiting on the user: TASK-14.1 first publish (a LICENSE file, the npm org `semantscript` with an `NPM_TOKEN` secret, a PyPI trusted publisher or `PYPI_TOKEN`, then `git tag v0.1.0 && git push origin v0.1.0`); TASK-14.5 AC4 (rescope to the measured cost reduction); TASK-9 AC5 (a reader other than the author); TASK-5.14 (the ~1B encoder point). `backlog task list --exclude-status Done --plain` lists them.

## How tasks have been run

1. One git worktree per task, branched from `main`: `git worktree add -b task-X /home/admin2/SemantScript-wt-X main`, copy `.env` in, symlink `.python-packages` from the main checkout, `npm ci && npm run build` inside it.
2. The orchestration script `.claude/workflows/semantscript-task-worktree.js` (args `task`, `root`, `branch`, `extra`) runs Understand, Implement, three adversarial reviewers, fix rounds and Finalize with opus agents; it commits on the branch with the attribution trailers and pushes the branch only. `extra` carries what other branches are touching and any spend rule.
3. When a branch finishes: merge it into `main` with `--no-ff`, resolve conflicts (for the generated remedy tables take either side and rerun `node scripts/generate-remedies.mjs`; prettier on docs), then the gates: `npm ci`, `npm run build`, `npm run lint:node`, `npm run test:node`, `pytest trainer/tests model/tests --deselect trainer/tests/test_cli.py` (that module takes about an hour; CI runs it), `ruff check` and `ruff format --check` on `trainer` and `model`, `npx prettier --check docs README.md cli`, `node scripts/generate-remedies.mjs --check`. Push `main`, remove the worktree, watch `gh run list --branch main`.
4. Python runs as `PYTHONNOUSERSITE=1 HSA_ENABLE_DXG_DETECTION=1 PYTHONPATH=.:trainer/src:model/src:.python-packages python3 ...`; ruff as `PYTHONPATH=.python-packages python3 -m ruff`. Never `pip install --upgrade --target` (it deletes `.python-packages/bin`).
5. Backlog: only the `backlog` CLI edits tasks; criteria are checked only with command evidence; follow-ups are reported, never created as tasks without approval.

## Secrets and spend

`/home/admin2/SemantScript/.env` holds `OPENROUTER_API_KEY` (git-ignored; the user said it expires around 2026-09-27). Load it with `set -a; . ./.env; set +a`; the trainer's Anthropic backend reads `ANTHROPIC_API_KEY`, so export it as that name for a teacher run. Never print, log or commit it. Usage at handoff: USD 18.75 of 50. The user approved the full budget for Express work but expects `semantscript train --estimate` quoted first and `--max-cost-usd` on every run.

## Decisions to honour

- "Keep it simple, accuracy is paramount": one float32 release; `releases derive --int8` is an optional lever, never the shipped path, until its check covers the held-out sample.
- Held-out benchmark inputs never enter training; judge labels are never recorded as human-authored.
- The Express training recipe and its figures live in `examples/express-app/README.md` under "The release on disk"; the teacher file is `.semantscript/teacher-constraints.toml` (git-ignored).

## Follow-ups reported, not yet tasks

- The int8 derivation check should verify on the release gate's held-out sample.
- A release trained before the held-out gate should say so in `test` and `explain` instead of printing `passed`.
- The held-out sampler should reach past the teacher's inferred range (the 240-day range hid a 365-day miss).
- `explain` with no published release prints a raw ENOENT.
- `init` should git-ignore `.semantscript/artifact.report.json`.
- The `test-example-mismatch` and `test-function-absent` labels still say `--bundle`.
- `minimumRuntimeVersion` is hard-coded to 0.0.0; a CommonJS export condition for the stub; the doctor should suggest a venv on PEP 668 Pythons.
