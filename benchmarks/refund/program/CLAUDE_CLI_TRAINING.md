# Claude CLI training teacher

`claude_cli_teacher.py` is an opt-in source of **synthetic training and
adversarial data only** for the refund benchmark. It is not a benchmark adapter,
does not produce human attestations, and must never receive or write either the
release-verification cases or the final held-out benchmark cases.

The pinned identity is:

- model: `claude-sonnet-5` by default; `ClaudeCliTeacherConfig.model` selects another
  immutable Claude model id (the Opus 5.5 corpus uses `claude-opus-5-5`) and the
  envelope's `modelUsage` must name exactly that model
- Claude Code CLI: the observed `claude --version` string is recorded in
  provenance when the installation is verified. Claude Code auto-updates in the
  background, so an exact pin is not the default; set
  `ClaudeCliTeacherConfig.cli_version` to require one. The protocol was last
  validated against `2.1.281 (Claude Code)`, and child processes run with
  `DISABLE_AUTOUPDATER=1`.
- protocol: `semantscript.refund-training.claude-cli/v1`
- prompt protocol: `semantscript-trainer-ir-prompts/v2`
- default configuration SHA-256: `f0e5a7236c7607ebbba046bef591ba94765f380670051e976fc920e2e455127a`
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
The installed CLI version is read and recorded before the first request.

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

## Coverage, retries, duplicates and concurrency

Every case position is requested with the trainer's shared prompt contract
(`semantscript-trainer-ir-prompts/v2`), whose `coverageBrief` is a pure
function of the IR and the position: it cycles the target label through the
output support, cycles a constraint focus through *none*, *satisfy* and
*near-miss* for each input-dependent constraint, and derives per-leaf
variation hints from a hash of the function id, position and input path.

`generate` retries each position up to `maximum_case_attempts` times. A
schema-invalid or constraint-violating answer is fed back to the model as a
rejection note carrying the refused inputs and the local reason; a transport
failure is retried without a note. After the first pass, positions whose inputs
duplicate an earlier position are re-requested for at most two rounds with a
duplicate note. Residual duplicates are kept rather than discarding the run and
are counted in `last_run_report`, together with request, rejection and failure
counts, so the corpus manifest can record them.

Positions run through a bounded thread pool of `concurrency` workers (1 by
default, 8 at most). Results are returned in position order and the first
failure cancels the remaining work. Concurrency and the attempt limit are part
of the configuration projection and therefore of the provenance digest.

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
