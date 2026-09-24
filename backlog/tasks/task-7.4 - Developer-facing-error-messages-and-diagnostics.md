---
id: TASK-7.4
title: Developer-facing error messages and diagnostics
status: Done
assignee:
  - '@claude'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-24 19:26'
labels:
  - compiler
  - cli
milestone: m-3
dependencies:
  - TASK-7.1
parent_task_id: TASK-7
ordinal: 27000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Unsupported types, missing inputs, verification failures and low-confidence fallbacks all need clear, located messages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every compile error names file, line and the offending sema site
- [x] #2 Verification failures list the failing example or constraint
- [x] #3 Diagnostics are covered by snapshot tests
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Findings: analysis and definition diagnostics already carry file:line:column, a code (9101-9125) and the site text; execution-plan diagnostics (9131) locate the site when one exists; but findSemaSites silently drops malformed canonical uses (no or several type arguments, configured calls with zero or several arguments, optional chaining), verification failures are counts ('1 gold/human example prediction(s) failed'), and no diagnostic output is snapshot-tested.
1. Compiler: findMalformedSemaSites reports every tagged template whose tag resolves to core sema or sema.withConfidence but is not a canonical site, with the reason; planSemaCompilation emits them as located diagnostics (code 9100, same file:line:column ... (site: ...) format) before analysis, so such sites become compile errors instead of untransformed runtime tags.
2. Trainer: verification failures name the failing evidence: each missed gold or attested example (row id, inputs, expected and predicted output) and each violated constraint (index, its source text, the case and the predicted output), capped per category; the CLI's train report and rendering print them.
3. Snapshot tests: a compiler diagnostics test compiles one fixture with every diagnostic family (malformed sites, unsupported outputs, invalid interpolations, definition errors) and compares the formatted, path-normalised output with a committed golden file (UPDATE_SNAPSHOTS=1 rewrites it); a verification test asserts the exact failure messages for a gold miss and a constraint violation. Lint and suites.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Compiler: findMalformedSemaSites (sema-sites.ts) plus code 9100 in planSemaCompilation report every core sema / sema.withConfidence tagged template that is not a canonical site (no or several output type arguments, configured call with zero or several options arguments, optional chaining) with the reason, in the file:line:column: message (site: ...) format shared by analysis (9101-9112), definition (9120-9125) and execution-plan (9131) diagnostics; exported from the package and catalogued in compiler/README.md. Trainer: _example_failures and _constraint_violations return per-case details and _gate_failures appends them (up to 10 per category): missed gold/attested examples with id, inputs, expected and predicted output; violated constraints with index, IR source text, case, inputs and predicted output. The CLI train report prints these under the function table. Evidence: AC1 compiler/test/diagnostics.test.mjs asserts every diagnostic from a fixture with all 16 families has a file, a start position and a message matching file:line:column ... (site: sema...), and the rendered output equals the committed snapshot test/fixtures/diagnostics.snapshot.txt (six malformed sites among them, previously silently ignored). AC2 trainer/tests/test_verification.py asserts the exact messages for a gold miss ('gold example base:0: inputs {"score":0} expected false, predicted true') and for constraint violations ('constraint 0 (score >= 10) violated by case base:2: inputs {"score":20} predicted false', three entries). AC3 the snapshot test (UPDATE_SNAPSHOTS=1 regenerates) plus the exact-message trainer assertions; the CLI test checks failures render under the report. Suites: Node 69/99/7/83, Python 507 passed, lint clean. Commits 057a8df and the README follow-up. Runtime errors were reviewed and left as they are: SemaConfidenceError names the function id and threshold and carries the diagnostic; SemaFallbackError names the function, fallback ref and reason.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @codex
created: 2026-09-22 04:17
---
Discovery audit follow-up: malformed canonical uses such as sema without an explicit type argument or configured forms with zero/multiple option arguments are intentionally omitted by findSemaSites. The diagnostics pipeline must identify these canonical malformed sites separately so they produce located compile errors instead of remaining untransformed runtime tags.
---
<!-- COMMENTS:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Compile errors now always point at the expression: malformed canonical uses of sema (missing or extra type arguments, wrong option counts, optional chaining), which the site finder used to skip silently, are reported as located diagnostics (code 9100) in the same file:line:column and site-text format as every analysis, definition and plan diagnostic, and the catalogue is documented. Verification failures name their evidence: each missed gold or attested example with inputs, expected and predicted output, and each violated constraint with its index, source text and case, which the CLI prints under the train report. Diagnostics are snapshot-tested: a compiler fixture with all sixteen families renders to a committed golden file, and the trainer asserts the exact failure messages. Verified with the Node suites (69/99/7/83), Python (507) and lint; commits 057a8df and its README follow-up.
<!-- SECTION:FINAL_SUMMARY:END -->
