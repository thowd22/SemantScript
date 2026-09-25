# Teachers

A teacher turns an expression's text into training cases: synthetic inputs
with labels, boundary pairs on either side of each constraint, and
counterfactual twins. It runs at build time only; nothing it produces ships
except through the weights it trained. This page covers the backends, what
they cost, and what the local-versus-reference experiment measured.

## Backends

The trainer takes a `[teacher]` TOML table (`--teacher`, or the first of
`semantscript.teacher.toml`, `teacher.toml`, `.semantscript/teacher.toml`)
and validates every response against a JSON Schema derived from the IR; the
provider's own structured-output support is never the validation boundary.

**Anthropic** (the reference, decision-1):

```toml
[teacher]
backend = "anthropic"
model = "claude-sonnet-5"
mode = "auto"            # direct | batch | auto (batch at batch_threshold)
batch_threshold = 32
max_tokens = 2048
```

The official Python SDK with the Messages structured-output API; `batch`
uses Message Batches, reconciled by `custom_id`, and any missing, failed or
truncated item fails the generation rather than yielding a partial dataset.
The key comes from `ANTHROPIC_API_KEY`; `semantscript train` writes this file
for you when the key is set and no file exists.

**Sonnet through OpenRouter.** OpenRouter serves the same model on an
Anthropic-format route (`https://openrouter.ai/api/v1/messages`) that honors
JSON-schema structured output and accepts the OpenRouter key as either
`x-api-key` or a bearer token:

```toml
[teacher]
backend = "anthropic"
model = "anthropic/claude-sonnet-5"
base_url = "https://openrouter.ai/api"
mode = "direct"          # OpenRouter has no Message Batches
max_tokens = 4096
```

Run `semantscript train` with `ANTHROPIC_API_KEY` set to the OpenRouter key
for that process (the SDK reads it; the file holds no secret). Verified
2026-09-25 on the Express example: both expressions generated their cases
through the route. Two things the route does differently, both handled by
the trainer: it turns extended thinking on unless the request says
`thinking: {"type": "disabled"}` (the teacher now sends that, so a case
costs 66 output tokens instead of about 1,900), and its replies can carry a
thinking block beside the text block (the decoder ignores non-text blocks).
A request costs about USD 0.017 through this route (the teacher prompt, the
IR-derived schema and instructions, is about 8,300 input tokens; OpenRouter
caches part of it). A constrained expression needs about twice as many
requests as cases (boundary pairs, counterfactual twins at the configured
ratio, and up to three attempts for an anchor the teacher cannot twin): the
Express example's two expressions at 192 cases each cost about USD 10 and
published on the third training seed, all from the same cached datasets.

**Ollama** (local, OpenAI-compatible endpoint):

```toml
[teacher]
backend = "ollama"
model = "qwen3:14b"
base_url = "http://localhost:11434/v1"
seed = 1
```

Record the model tag and its resolved digest from `/api/tags` in provenance;
a tag can move. The backend supplies a placeholder API key the local server
ignores.

**Claude Code CLI.** The refund benchmark's `claude_cli_teacher.py` shells out
to a logged-in `claude` CLI, so generation draws on a subscription's quota
rather than per-token dollars; it is a training-only teacher with its own
provenance identity (`anthropic-claude-cli-training-only`) and is documented
in `benchmarks/refund/program/CLAUDE_CLI_TRAINING.md`.

**No teacher: constraint labels.** When an expression's constraints are
complete (for every input exactly one output satisfies them), the constraints
label sampled inputs directly and the boundary pairs and counterfactuals come
from single-field edits. The reference application trains all nine of its
expressions this way (`examples/refund-service/train.py`), and the refund
benchmark's release corpus labels its real inputs the same way (decision-7).

## What the teacher is asked for

For every expression the teacher writes the synthetic cases (inputs with
labels). For an expression with constraints it is also asked, per
constraint, for a boundary pair (two inputs one field apart on either side of
the predicate) and, for a share of the cases (`--counterfactual-ratio`,
default 1.0), for a counterfactual twin (one field changed, label changed).
Every proposal is checked locally against the schema and the constraints; a
wrong label is replaced, a twin that changes two fields is retried, and an
anchor the teacher cannot twin is skipped for the next one. This is automatic
whenever constraints are declared, which is why constraints matter more than
the teacher: with the same 93 teacher cases the Express example's refund
expression learned nothing useful without them (0.53) and nearly everything
with them (0.98).

## Cost and time, as measured

| Teacher                           | Per request                                         | Latency                           | Where measured                                                  |
| --------------------------------- | --------------------------------------------------- | --------------------------------- | --------------------------------------------------------------- |
| Sonnet 5 via OpenRouter, label    | USD 0.0028 to 0.0031 (960 input tokens)             | 4.0 s p50                         | 1,800 labels, `results-local-teacher-2026-09-25`                |
| Sonnet 5 via OpenRouter, baseline | USD 0.0031 (decision plus distribution)             | 4.2 s p50                         | 170 requests, `results-final-2026-09-25`                        |
| Sonnet 5 via OpenRouter, teacher  | USD 0.017 per request (8,300-token prompt)          | about 4 s                         | `semantscript train` on the Express example, about 600 requests |
| Claude Code CLI (Sonnet 5)        | subscription quota, about two cents list-equivalent | 4 to 6 s (2.3 s at concurrency 4) | refund pilot corpus, 2026-09-23                                 |
| Qwen3-14B via Ollama, local       | none                                                | 0.17 s p50 on the GPU             | 1,800 labels                                                    |
| Jev (typed-decision model)        | USD 0.00004 (not a teacher; see decision-11)        | 0.15 s                            | 160 cases, `results-jev-2026-09-25`                             |
| Constraint labels                 | none                                                | microseconds                      | refund service, nine expressions                                |

A first build of one expression with 200 cases through Sonnet costs about
USD 0.60 in labels and a minute of GPU time; every later build reuses the
cached datasets and the [build cache](build-cache.md), so an unchanged
expression costs nothing and a changed one retrains only its head.

## Local versus reference (TASK-5.13, decision-12)

Can a local model replace the reference so iteration is free? The same 1,800
real refund inputs were labeled by Sonnet 5 and by Qwen3-14B (thinking off,
temperature 0), then a student was trained on each label set over the same
1,500 inputs and scored on a frozen Sonnet-labeled evaluation set of 300.

| Measure                                    | Sonnet 5  | Qwen3-14B |
| ------------------------------------------ | --------- | --------- |
| Agreement with the compiled constraints    | 1.000     | 0.746     |
| Agreement on the "suspicious history" rule | 1.000     | 0.280     |
| Student accuracy on the evaluation set     | **0.980** | 0.800     |
| Student ECE on the evaluation set          | **0.019** | 0.143     |

Sonnet 5 labels this policy exactly as its constraints do. Qwen3-14B misreads
the two compound conditionals (three or more prior refunds on an order of at
least 1,000; outside the tier window) and the student trained on its labels
inherits the errors: an 18-point gap on the same inputs and target. The
decision: a local generative teacher is not adopted; local iteration uses
constraint labels, and the reference teacher is used for generation where
the text says more than the constraints. Full record:
`benchmarks/refund/data/results-local-teacher-2026-09-25/README.md`.

## Writing a teacher

A teacher is any Python object with a `descriptor` (`TeacherDescriptor`:
provider, model, configuration digest, no secrets) and `generate(ir, n)`
returning `n` `GeneratedCase` values; one that also implements
`generate_boundary_pair(ir, index)` and `generate_counterfactual(ir, anchor)`
serves expressions with constraints. The trainer validates every case
against the IR, drops a case whose label violates an active constraint and
asks for a replacement (three rounds at most, then it fails), refuses
conflicting labels for one input, and caches datasets by the teacher's
identity, so two teachers never share a cache entry. The rule teacher in
`examples/refund-service/train.py` is a complete, dependency-free example.
