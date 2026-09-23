# Claude CLI training teacher

`claude_cli_teacher.py` is an opt-in source of **synthetic training and
adversarial data only** for the refund benchmark. It is not a benchmark adapter,
does not produce human attestations, and must never receive or write either the
release-verification cases or the final held-out benchmark cases.

The pinned identity is:

- model: `claude-sonnet-5`
- Claude Code CLI: `2.1.280 (Claude Code)`
- protocol: `semantscript.refund-training.claude-cli/v1`
- default configuration SHA-256:
  `d56782ca2714eef421da23d8b3bca83564dd8d31162111e2655fa55a214df51c`
- data classification: `synthetic-training-only`

The teacher implements the existing `Teacher` and `AdversarialTeacher`
protocols. `SyntheticDatasetGenerator` persists its ordinary outputs with the
`synthetic` origin, while `AdversarialDatasetGenerator` persists the boundary
and counterfactual origins. The provider identity is deliberately
`anthropic-claude-cli-training-only`, so that it cannot be mistaken for a human
source in provenance review.

## Request isolation

Each request is a fresh, non-interactive process. The command pins Sonnet 5,
passes the trainer's IR-derived JSON Schema, supplies the deterministic trainer
system prompt, disables session persistence, tools, skills, settings sources,
MCP servers, Chrome, permission prompts, and prompt suggestions, and enables
safe and restricted modes. It runs in a new empty temporary working directory.
The exact CLI version is checked before the first request.

Stdout, stderr, stdin, elapsed time, and per-request spend are bounded. On a
timeout or stream overflow the entire process group is terminated. Error
messages report only the failure class/status and never echo prompts, model
output, stderr, environment variables, or authentication material.

The CLI's structured result envelope must be one successful result with no
permission denials, served only by the pinned model (`modelUsage` names exactly
`claude-sonnet-5`), and reported within eight turns. One structured-output
exchange is a thinking or text block, a `StructuredOutput` tool call, and its
tool result; the pinned CLI's turn accounting for that varies with the response
shape (two and three observed), so the bound only guards against runaway
re-prompting. The embedded object is serialized again and validated by the
existing local case, boundary-pair, or counterfactual parser before it can enter
a trainer dataset.

## Intended integration

Construct `ClaudeCliTrainingTeacher`, retain its `provenance` and
`configuration_projection` alongside generation-run evidence, and pass the
teacher only to `SyntheticDatasetGenerator` and `AdversarialDatasetGenerator`.
Do not copy its output into a record whose origin is `human-authored` and do not
use it to fill either attested human set.

This path also does not replace the benchmark's `structured-api` adapter. Claude
Code startup and orchestration are different from a traditional Messages API
latency measurement, so CLI timing is not a valid result for that baseline.

Tests use an injected process runner and local Python subprocesses. They do not
invoke Claude, consume tokens, or generate data. A real request should only be
started after the training-data count and spend are explicitly chosen and the
release/final datasets are sealed independently.
