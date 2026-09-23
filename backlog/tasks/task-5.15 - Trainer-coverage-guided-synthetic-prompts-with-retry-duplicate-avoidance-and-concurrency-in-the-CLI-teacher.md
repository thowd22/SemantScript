---
id: TASK-5.15
title: >-
  Trainer: coverage-guided synthetic prompts with retry, duplicate avoidance and
  concurrency in the CLI teacher
status: Done
assignee:
  - '@claude'
created_date: '2026-09-23 19:31'
updated_date: '2026-09-23 19:45'
labels:
  - trainer
  - benchmark
milestone: m-1
dependencies:
  - TASK-5.4
  - TASK-5.12
references:
  - trainer/src/semantscript_trainer/teacher_prompt.py
  - benchmarks/refund/program/claude_cli_teacher.py
documentation:
  - benchmarks/refund/program/CLAUDE_CLI_TRAINING.md
parent_task_id: TASK-5
ordinal: 47000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first live Sonnet 5 run for the refund benchmark (TASK-5.11, 2026-09-23) showed the synthetic case prompt collapses to one mode case: 10 requests gave 9 deny / 1 approve / 0 review, 8 of 10 at the same order age, and 2 of 3 earlier cases were byte-identical. Every teacher (Claude CLI, Anthropic SDK, Ollama at temperature 0) shares the per-case prompt from teacher_prompt.build_case_messages, which only says 'case i of N' and 'diverse', and the dataset assembler neither dedupes nor rejects duplicates, so a large pilot would freeze a degenerate, deny-dominated corpus. A Claude CLI request also aborts the whole synthetic run on the first schema-invalid, constraint-violating or transport failure, and runs sequentially at 4-9 s per request, which makes a 600-case pilot fragile and slow. The prompt needs a deterministic, IR-derived coverage brief (target label, constraint region, per-field variation hints, rejection feedback) so any provider produces varied, label-covering cases, and the CLI teacher needs bounded per-case retries, duplicate avoidance rounds, a run report, and bounded concurrency. Prompt contract version bumps so existing dataset cache keys do not collide.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 build_case_messages emits a deterministic coverageBrief derived only from the IR and case position: a target output cycling through the output support, a constraint focus cycling through none / satisfy / near-miss per input-dependent constraint with the target adjusted so it never contradicts an active always or never constraint, and per-leaf variation hints for every input type kind
- [x] #2 build_case_messages accepts prior rejection notes and appends them to the user message so a retry tells the model why the previous inputs were rejected; with no notes the prompt is unchanged between calls
- [x] #3 The dataset request digest and the CLI teacher prompt protocol identify the new prompt contract version so caches built under the old prompt are not reused
- [x] #4 ClaudeCliTrainingTeacher.generate retries each case position up to a configured attempt limit on schema, constraint or transport failures, re-requests positions whose inputs duplicate an earlier case for a bounded number of rounds, tolerates residual duplicates, and exposes a run report with request, retry-by-reason and residual-duplicate counts
- [x] #5 ClaudeCliTrainingTeacher runs case positions with configurable bounded concurrency while returning cases in position order; concurrency and attempt limits are part of the configuration projection and digest
- [x] #6 Trainer and benchmark test suites cover the brief schedule, rejection notes, retry paths, duplicate rounds, concurrency ordering and the updated digest, and CLAUDE_CLI_TRAINING.md documents the protocol
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rewrite teacher_prompt.build_case_messages to add a deterministic coverageBrief (target output cycling over output support, constraint focus cycling none/satisfy/near-miss per input-dependent constraint with the target forced consistent with always/never outputs, per-leaf variation hints derived from sha256(function id, index, path)), an optional rejected-attempts section, and a v2 system prompt stating precedence constraints > constraintFocus > targetOutput > variation.
2. Bump promptContractVersion to 2 in the dataset request digest and the CLI teacher prompt protocol to v2.
3. Extend ClaudeCliTeacherConfig with concurrency and maximum_case_attempts in the projection; make generate compile constraints once, run positions through a bounded thread pool, retry each position with rejection notes on schema/constraint failures and silently on transport failures, run bounded duplicate-avoidance rounds keyed on canonical inputs, tolerate residual duplicates, and publish a run report.
4. Update trainer prompt tests and CLI teacher tests (retry, constraint retry, duplicate rounds, residual tolerance, exhaustion, concurrency ordering, new digest); update CLAUDE_CLI_TRAINING.md.
5. Run ruff and the full Python gate; commit.

6. Because Claude Code auto-updated to 2.1.281 mid-session, replace the exact CLI version pin with an observed-version record in provenance plus an optional required pin in config, and run child processes with DISABLE_AUTOUPDATER=1. 7. Add generate_training_corpus.py (compile, synthetic, adversarial, manifest) with an offline stub-CLI test so the pilot is reproducible.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Prompt contract v2 implemented: coverageBrief (target label cycle, constraint focus none/satisfy/near-miss, per-leaf variation hints) plus rejectedAttempts; dataset promptContractVersion 2 and regenerated the pinned empty-dataset golden. CLI teacher: per-position retries with rejection notes, two duplicate-avoidance rounds, residual tolerance, run report, bounded thread pool, observed CLI version in provenance. Live 15-case probe at concurrency 4 (not training data): 15/15 unique inputs, labels approve 3 / deny 7 / review 5, ageDays 4-143 with several at the 90-day boundary, zero rejections, 34 s. Two labels looked policy-wrong where the model chased an unreachable target label, so the system prompt now states label truthfulness outranks targetOutput.

Validation: ruff lint/format clean (66 files); full Python gate 426 passed, 4 intentionally skipped, 17.5 s. Evidence per criterion: AC1/AC2 trainer/tests/test_teacher_prompt.py (brief schedule over 16 positions, variation hints per leaf, rejection notes bounded and absent when empty); AC3 test_dataset golden regenerated under promptContractVersion 2 and the CLI teacher projection asserts promptProtocol v2; AC4 test_claude_cli_teacher.py retry-with-note, constraint retry, duplicate rounds with residual tolerance, exhaustion after all attempts, run report counts; AC5 concurrent ordering, first-failure cancellation, generation settings in the projection digest; AC6 CLAUDE_CLI_TRAINING.md documents coverage, retries, duplicates, concurrency and the observed-version policy. Also added generate_training_corpus.py with an offline stub-CLI test that compiles, generates 4 synthetic + 4 boundary + 2 counterfactual records and writes the manifest.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced the per-case 'case i of N' prompt with a deterministic IR-derived coverage brief (target label cycle, constraint focus none/satisfy/near-miss, per-leaf variation hints, rejection feedback) shared by every teacher, bumped the prompt contract to v2 in the dataset digest and CLI protocol, and gave the Claude CLI teacher bounded per-position retries, two duplicate-avoidance rounds, a run report, a bounded thread pool, and observed-version provenance (Claude Code auto-updates mid-session). Added generate_training_corpus.py to freeze a synthetic-plus-adversarial corpus with a manifest. Verified by the full Python gate (426 passed) and a live 15-case Sonnet 5 probe: 15/15 unique inputs, labels approve 3 / deny 7 / review 5, zero rejections, 34 s at concurrency 4, versus 8/10 unique and 9/1/0 before.
<!-- SECTION:FINAL_SUMMARY:END -->
