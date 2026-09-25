# Teachers

A teacher turns an expression's text into training cases: synthetic inputs
with labels, boundary pairs on either side of each constraint, and
counterfactual twins. It runs at build time only; nothing it produces ships
except through the weights it trained. This page covers the backends, how to
choose one and check it, what a run will cost before it starts and how to cap
it, and what the local-versus-reference experiment measured.

## Choosing a teacher

`semantscript init --teacher anthropic|openrouter|ollama|constraints` (or the
answer to its one question on an interactive terminal) writes
`.semantscript/teacher.toml` for that teacher, with no key in it; an existing
teacher file is never replaced. Then:

```sh
semantscript teacher probe        # one small request: model, latency, tokens, cost
semantscript train --estimate     # what the run would cost and take, without calling the teacher
semantscript train --max-cost-usd 2
```

| Choice        | What it writes                                                                    | Key                                           |
| ------------- | --------------------------------------------------------------------------------- | --------------------------------------------- |
| `anthropic`   | `backend = "anthropic"`, `claude-sonnet-5`, `mode = "auto"`                       | `ANTHROPIC_API_KEY`                           |
| `openrouter`  | the Anthropic backend at `https://openrouter.ai/api`, `anthropic/claude-sonnet-5` | `ANTHROPIC_API_KEY` set to the OpenRouter key |
| `ollama`      | `backend = "ollama"`, `qwen3:14b`                                                 | none                                          |
| `constraints` | `backend = "constraints"`                                                         | none                                          |

`--teacher-model <id>` names another model.

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
With the prompt of that day a request cost about USD 0.017 through this route
(about 8,300 input tokens on average; the decideRefund case prompt alone was
6,552). A constrained expression needs about twice as many requests as cases
(boundary pairs, counterfactual twins at the configured ratio, and up to
three attempts for an anchor the teacher cannot twin): the Express example's
two expressions at 192 cases each cost about USD 10 and published on the
third training seed, all from the same cached datasets. The prompt has since
been reduced and is served through prompt caching (below): the same case
costs USD 0.0071 for the first request and USD 0.0017 from the cache.

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

**Constraints** (built in, no language model and no key, decision-12):

```sh
semantscript train --teacher constraints
```

or, to set its options, a table:

```toml
[teacher]
backend = "constraints"
seed = 1                  # the sampling seed
twin_filter = true        # keep only inputs one field edit can move to another output
near_threshold_share = 0.3
maximum_sampling_attempts = 200000  # draws before it stops looking for more distinct inputs

[teacher.ranges]          # optional; by dotted input path or field name
total = { low = 0, high = 8000, distribution = "log", decimals = 1 }
"order.ageDays" = { low = 0, high = 120 }   # distribution uniform | log | count
```

When an expression's constraints are complete (for every input exactly one
output violates no active `always` or `never`), they are a labelling
function. This backend samples inputs from the IR's input types (objects,
optional fields, unions, enums, literals, booleans, strings, numbers,
tuples, arrays), labels each with the one output the constraints admit, and
builds the adversarial cases the same way: a boundary pair is two labelled
inputs one field apart with the predicate false on one side and true on the
other, and a counterfactual twin is a single-field edit that changes the
label. Sampling is threshold-aware: every comparison between an input path
and a literal (`order.total > 1000`, `ticket.category === "outage"`)
records a threshold or a string for that path, three draws in ten land on
or one step beside a threshold, and a string mostly takes the values the
predicates compare it with. A comparison between two inputs (`p.x > p.y`)
edits one to the other's value or one step beside it, and an array or string
length threshold (`items.length > 3`) is crossed by appending or removing one
item or character, so every edit changes exactly one JSON path. A numeric
range is inferred from the thresholds and the gold examples (at least 0 to
10 and up to twice the largest threshold; exactly twice the largest when the
thresholds are fractional, as for a 0 to 1 score; symmetric around zero, at
least -10 to 10, when a threshold is negative or a comparison holds at or
below zero, as `balance < 0` or `amount <= 0` do, while a count compared only
as `chargebacks > 0` or `=== 0` keeps a range from zero up; `count`, zero
half the time, when every threshold is a small integer; `log` when the range
reaches 1,000; decimals when a threshold or an example has them), and
`[teacher.ranges]` overrides
it. A path compared with no literal of its own, as in an arithmetic predicate
(`pair.a - pair.b > 10`), takes every literal in the predicates as its
thresholds and a range of at least 0 to 100, so there is room for a few hundred
distinct inputs; a number no predicate and no gold example mentions (an
optional field left out of the examples) is drawn from 0 to 100. Arithmetic and
multi-field predicates are harder for the model to learn than a single
threshold: give them more cases or epochs, or a `[teacher.ranges]` entry
centred on the boundary, if verification reports a few violations next to it.
Unions of object variants (`{ kind: "circle"; radius: number } | { kind:
"square"; side: number }`) are sampled, but a predicate on the discriminant
cannot be crossed by a single-field edit (switching the variant changes more
than one JSON path), so the adversarial stage cannot build its boundary pair;
put such a predicate on a plain enum or literal field instead.

**Small input spaces.** Identical inputs, and each case with its
counterfactual twin, share one group in the held-out split, so an expression
with few distinct inputs (a small whole-number range and a boolean, say) can
link its whole corpus into one group and leave nothing to hold out. Two things
keep that from happening. A counterfactual twin first tries number values
spread over the whole range, not only the ones beside a threshold (boundary
pairs keep those), so anchors do not all share the same few twins. And an
expression whose seeded pilot of 2,000 draws holds fewer than 800 distinct
inputs is compact: half of its whole-number draws then carry two decimal
places, so `severity` in `severity >= 8` (range 0 to 16) takes 1,601 values
instead of 17. An input space that stays small, such as two booleans, still
fails with `needs at least two independent row groups`; lower
`--counterfactual-ratio` (0.5, or 0 for a few booleans or enum values) or
widen `[teacher.ranges]`. When fewer than half of an expression's cases are
distinct inputs, `train` prints a warning with the count.

The reference application trains all nine of its expressions this way (`examples/refund-service`, `npm run train`),
and the refund benchmark's release corpus labels its real inputs the same
way (decision-7).

The teacher's identity in provenance is provider `constraints`, model
`compiled-constraints-v3` and, as the configuration digest, the SHA-256 of
the sampling configuration (seed, twin filter, near-threshold share, ranges
and the algorithm version), so changing any of them regenerates the cached
datasets.

An expression whose constraints do not decide some input stops the build
with that input and what the constraints admit:

```text
error: src/refund.sem.ts:17 (nf_7d1c02a6): the constraints do not decide every input, so the
constraints teacher cannot label it alone. For the input {"customer":{"priorRefunds":0,
"tier":"standard"},"order":{"ageDays":12,"status":"paid","total":88}} the constraints admit
"approve", "deny", "review". Add constraints until exactly one output is admissible for every
input, or add a [teacher.fallback] table with a language-model teacher to label the inputs the
constraints leave open (docs/teachers.md)
```

**Mixed: constraints plus a language-model fallback.** Add a fallback table
and the constraints label the inputs they decide while the language model
labels the rest:

```toml
[teacher]
backend = "constraints"

[teacher.fallback]
backend = "anthropic"
model = "claude-sonnet-5"
```

The fallback takes any key of the Anthropic or Ollama tables above. The
constraints label the share of inputs they decide (estimated from a seeded
pilot sample of 512 inputs); the fallback generates the rest; any fallback
case whose input the constraints decide takes the constraints' label; and a
boundary pair or twin the constraints cannot build on both sides comes from
the fallback, its decided side relabelled. An expression the constraints
decide completely sends the fallback no request, and one with no constraints
goes to the fallback entirely. The provenance provider is
`constraints+<fallback backend>`, the model is the fallback's, and the digest
covers the sampling configuration and the fallback's identity.

What has been checked, and what has not: the mixed path is covered by unit
tests with an in-process fake fallback, and one end-to-end run with a local
model passes. An expression `route(ticket: { severity: number; vip: boolean })`
with `always(severity >= 8, "page")` and `never(vip, "ignore")`, a
`qwen3:14b` fallback through Ollama and
`--cases 160 --epochs 10 --select-best-epoch --counterfactual-ratio 0`
published a verified release in about three minutes. The same expression
with counterfactual twins on (the default `--counterfactual-ratio 1`) fails
with `qwen3:14b`, and an earlier run with `qwen2.5:7b` failed a step before,
on a boundary pair whose sides did not straddle the predicate. An anchor
decided by an `always` rule can only change label by moving into the inputs
the constraints leave open, so its twin has to come from the fallback, and
these models propose twins that change two fields, or that the constraints
decide back to the anchor's own output. The error names the expression and
the fallback:

```text
error: only 1 of 10 counterfactual pairs could be generated: base synthetic case 5 did not
yield a valid counterfactual within 3 attempts: src/route.sem.ts:7 (nf_aa334d2c): the
constraints could not build this case, and the [teacher.fallback] teacher (ollama/qwen3:14b)
proposed the counterfactual twin {"ticket":{"severity":16,"vip":false}} of
{"ticket":{"severity":16,"vip":true}}, which has the anchor's own output "page" (the
constraints decide it)
```

With a local fallback, start with `--counterfactual-ratio 0` (or a small
ratio), or make the constraints complete so no case needs the fallback. An
Anthropic fallback has not been run in mixed mode.

Whatever the teacher, verification reproduces every gold example exactly and
refuses an expression with none, so `train` stops before generating anything
when an expression has no `examples` entry and names it.

## Checking a teacher before a run

`semantscript teacher probe` sends one small request through the teacher
`train` would use and prints the backend, the model and host, the latency,
the tokens and the cost with its price source (2026-09-25: 2.3 s, 16 in / 4
out tokens, USD 0.000072 for Sonnet 5 through OpenRouter; 8.3 s including the
model load, USD 0, for Qwen3-14B through Ollama; no request and USD 0 for the
constraints teacher; a mixed constraints teacher probes its fallback).

`semantscript doctor` checks the teacher `train` would use: the file is found
and valid, the key is in the environment, and one minimal request succeeds.
The Anthropic request is one short user message with `max_tokens` 8 and
thinking disabled (about 16 input and 4 output tokens, well under USD 0.001
on Sonnet through OpenRouter); the Ollama request is a tiny chat completion.
The check prints the latency and the tokens, and scrubs the key from any error
text. When `base_url` is OpenRouter and only `OPENROUTER_API_KEY` is set, the
key check fails with the fix `export ANTHROPIC_API_KEY="$OPENROUTER_API_KEY"`
(on Windows, `$env:ANTHROPIC_API_KEY = $env:OPENROUTER_API_KEY` in PowerShell
and `set "ANTHROPIC_API_KEY=%OPENROUTER_API_KEY%"` in cmd), because the
backend reads only `ANTHROPIC_API_KEY`.

For the constraints backend the key check passes with no key and the probe
is skipped (it sends no request); with a `[teacher.fallback]` table both
checks are the fallback's.

`train` runs the same checks first without the billed request (for Ollama it
only asks the server for its model list and fails with `ollama pull <model>`
when the model is missing). The probe is `probe_teacher()` in
`semantscript_trainer.doctor`, for scripts that want the same check.

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

## Prompt size and caching

Every teacher request carries the function's contract. Until 2026-09-25 it
was indented JSON with each constraint's full predicate AST and a copy of the
response schema, which the structured-output format already sends; the
decideRefund case prompt was 20,433 characters. The prompt is now compact
(prompt layout version 4): the stable contract (behavior text, gold
examples, inputs, output, and each constraint as its kind, output and
TypeScript source) is appended to the system prompt as compact JSON, and the
user message carries only the case position and its coverage brief (or the
selected constraint, or the anchor of a twin). The Anthropic backend marks
the system prompt for prompt caching, so from the second request of a kind
the provider reads it at a tenth of the input price; a prompt shorter than
the model's minimum (1,024 tokens on Sonnet 5, 512 on Opus 5, 4,096 on Haiku
4.5) is simply not cached.

Measured 2026-09-25 on the Express example (characters counted locally,
tokens and dollars as Sonnet 5 through OpenRouter reported them for case 6 of
189 and the case after it):

| decideRefund                       | Characters                     | Input tokens                                | Output tokens | USD       |
| ---------------------------------- | ------------------------------ | ------------------------------------------- | ------------- | --------- |
| Case prompt before (layout 3)      | 21,103                         | 6,552                                       | 68            | 0.013784  |
| Case prompt now, first request     | 5,468                          | 2,611 (2,347 written to the cache, 264 not) | 68            | 0.0070755 |
| Case prompt now, next request      | 5,568                          | 2,639 (2,347 read from the cache, 292 not)  | 66            | 0.0017134 |
| Boundary prompt before / now       | 23,641 / 3,727 plus the schema | not sent                                    |               |           |
| Counterfactual prompt before / now | 19,737 / 3,606 plus the schema | not sent                                    |               |           |

Half of the USD 0.017 per request recorded for the old prompt is USD 0.0085;
a request now costs USD 0.0017 once its kind is cached and USD 0.0071 when
it is the first. The triage case prompt went from 3,816 to 2,527 characters.
Qwen3-14B through Ollama answered six cases of each expression with the new
prompt without a schema failure. What has not been measured is the verified
accuracy of a model trained on datasets regenerated with the new prompt; the
full comparison (both Express expressions at 192 cases, then the same recipe
and seed 3) is estimated at USD 1.02 (at most USD 2.88) and has not been run.

A language-model teacher's configuration digest includes the prompt layout
version, so datasets cached with the old prompt are regenerated once for the
Anthropic and Ollama backends; the constraints teacher's datasets, which send
no prompt, are unchanged.

## Cost estimate and spend cap

`semantscript train --estimate` prints, per expression and in total, the
teacher requests the run would send, the input tokens (and the share the
prompt cache serves), the output tokens, the USD cost and the wall time,
without calling the teacher, loading PyTorch or training. On the Express
example with the OpenRouter teacher (`--cases 192 --counterfactual-ratio 0.5`,
no cached datasets):

```text
teacher: anthropic anthropic/claude-sonnet-5; price: OpenRouter price list (2026-09-25) (USD 2 in / 10 out per million tokens)
function      source              requests        input tokens  cached     output tokens  USD              time
------------  ------------------  --------------  ------------  ---------  -------------  ---------------  ------
nf_957c2b2b…  src/refunds.sem.ts  407 (max 1344)  997,984       877,912    38,834         0.81 (max 2.67)  27 min
nf_bcbf93e1…  src/triage.sem.ts   189 (max 189)   253,827       218,080    9,450          0.21 (max 0.21)  13 min
total                             596 (max 1533)  1,251,811     1,095,992  48,284         1.02 (max 2.88)  40 min
time per request: 4 s (pinned default); tokens are characters / 2.1; no teacher request was sent
```

How it counts:

- **Requests.** An expression whose datasets are cached costs nothing. Otherwise the synthetic dataset asks for `cases - gold` cases, one per request; an expression with constraints adds one boundary-pair request per constraint and a twin request for `--counterfactual-ratio` of the synthetic cases. The expected count multiplies a constrained expression's requests by 1.4 (label replacements, repeated attempts, skipped anchors), calibrated on the 2026-09-25 Express run (about 600 requests where 479 were planned, all the extra on the constrained expression); the maximum is the bound the generators enforce (three replacement rounds, three attempts per boundary pair and per anchor, as many skipped anchors as pairs). A mixed constraints teacher counts only the share of inputs its constraints leave open.
- **Tokens.** The characters of the exact prompts the teacher builds, plus the response schema, divided by 2.1: the ratio Sonnet 5 reported for the compact prompts (5,468 characters, 2,611 tokens; the old indented prompt ran at about 3.2). No Claude tokenizer runs offline, so this is the one approximation; output tokens are the size of a serialized case (twice for a pair, plus a reason for a twin) over the same ratio. With direct Anthropic requests the schema and system prompt are counted as cache reads after the first request of each kind when they reach the model's minimum cacheable length. Message Batches (`mode = "batch"`, or `auto` at `batch_threshold`) are priced at half and listed apart; a batch usually ends within an hour.
- **Price.** The constraints teacher and Ollama cost nothing. An Anthropic-backend model is priced from a pinned table of Anthropic list prices (USD per million input / output tokens, 2026-09-25: `claude-sonnet-5` 2 / 10, `claude-opus-5` 5 / 25, `claude-opus-5-5` 4 / 20, `claude-haiku-4-5` 1 / 5, `claude-sonnet-4-6` 3 / 15, `claude-opus-4-8` 5 / 25; cache reads at 0.1 times the input price, writes at 1.25 times, batches at half). With an OpenRouter `base_url` the price comes from OpenRouter's public model list (`GET https://openrouter.ai/api/v1/models`, no key), cached a day in `<cache-dir>/teacher-prices.json`, with the pinned table as the offline fallback; the output names the source.
- **Time.** Requests times the seconds per request: the `[teacher.pricing]` figure, else the mean the last metered run of the same teacher recorded (`<cache-dir>/teacher-stats.json`), else a pinned figure (4 s for Sonnet 5 through OpenRouter; 1 s for Qwen3-14B through Ollama on one RX 9070 XT, both measured on the Express prompts).

Checked on a small run: the estimate for the Express example at `--cases 8
--counterfactual-ratio 0.5` said 26 requests (at most 61) and USD 0.06; the
three runs below sent 19 paid requests in all for USD 0.060, and replayed 8
more from the journal after the stops.

A `[teacher.pricing]` table overrides any figure, and is required for a model
the estimate cannot price (the error names it). It never enters a digest:

```toml
[teacher.pricing]
input_usd_per_million = 3
output_usd_per_million = 15
cache_read_usd_per_million = 0.3    # optional; default 0.1 x input
cache_write_usd_per_million = 3.75  # optional; default 1.25 x input
seconds_per_request = 4
```

`--max-cost-usd <x>` caps a run. Before each request the trainer reserves an
estimate of its cost (its own prompt with a 20% token margin at the
cache-write price, plus the largest answer seen so far and a quarter; a
Message Batch is reserved whole before it is submitted), and a request whose
reservation would pass the cap is not sent. The cap is therefore not a hard
limit to the cent: a single answer much longer than any seen so far (up to the
teacher's `max_tokens`, about USD 0.04 at Sonnet's output price for 4,096
tokens) can take the run past it by that one request. When the cap is
reached: the run exits 1 with
`error: spend cap USD <x> reached …`. Every dataset finished before the stop
stays cached, and every paid response is kept in the response journal
(`<cache-dir>/teacher-responses/`, see the [build cache](build-cache.md)), so
the next run replays them at no cost and continues. Every run prints its
running cost (the second capped run below):

```text
error: spend cap USD 0.02 reached: the next request (about USD 0.0084) would take the run from USD 0.0186 past it after 6 paid request(s). …
teacher: 6 request(s) (1 replayed from the journal), 17,246 in (14,360 cached) / 882 out tokens, USD 0.0186 of the USD 0.02 cap
```

The line appears every 25 requests and after each expression's datasets, the
report table ends with `teacher: <n> requests (<m> replayed), USD <x> of the
USD <cap> cap`, and the JSON report carries the same numbers as
`teacher.spend`. The cost is computed from the token usage each response
reports (OpenRouter's own `usage.cost` matched it to the digit on the probe
requests). Measured on the Express example at `--cases 8`: a first run capped
at USD 0.02 stopped after six requests (it spent USD 0.0223: that run used an
earlier, looser reservation, since tightened to the one above); a second run
with the same cap replayed the journaled boundary pair, paid for six more and
stopped at USD 0.0186; a third with USD 0.10 replayed seven and finished the
datasets for USD 0.0194.

## Cost and time, as measured

| Teacher                           | Per request                                         | Latency                           | Where measured                                                  |
| --------------------------------- | --------------------------------------------------- | --------------------------------- | --------------------------------------------------------------- |
| Sonnet 5 via OpenRouter, label    | USD 0.0028 to 0.0031 (960 input tokens)             | 4.0 s p50                         | 1,800 labels, `results-local-teacher-2026-09-25`                |
| Sonnet 5 via OpenRouter, baseline | USD 0.0031 (decision plus distribution)             | 4.2 s p50                         | 170 requests, `results-final-2026-09-25`                        |
| Sonnet 5 via OpenRouter, teacher  | USD 0.017 per request (8,300-token prompt)          | about 4 s                         | `semantscript train` on the Express example, about 600 requests |
| Same, compact prompt (layout 4)   | USD 0.0017 cached, 0.0071 first of a kind           | 2.6 to 3.0 s                      | two decideRefund case requests, 2026-09-25                      |
| Qwen3-14B via Ollama, teacher     | none                                                | 1.0 s per case                    | six cases per Express expression, compact prompt                |
| Claude Code CLI (Sonnet 5)        | subscription quota, about two cents list-equivalent | 4 to 6 s (2.3 s at concurrency 4) | refund pilot corpus, 2026-09-23                                 |
| Qwen3-14B via Ollama, local       | none                                                | 0.17 s p50 on the GPU             | 1,800 labels                                                    |
| Jev (typed-decision model)        | USD 0.00004 (not a teacher; see decision-11)        | 0.15 s                            | 160 cases, `results-jev-2026-09-25`                             |
| Constraints (built in)            | none                                                | microseconds                      | refund service, nine expressions                                |

A first build of the Express example's constrained `decideRefund` with 192
cases and counterfactual ratio 0.5 through Sonnet on OpenRouter is estimated
at USD 0.81 (at most USD 2.67) with the compact prompt, plus a minute of GPU
time (`semantscript train --estimate` prints the figure for your own run); every later build reuses the
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
identity, so two teachers never share a cache entry. The built-in
constraints teacher (`semantscript_trainer/teachers/constraints.py`) is a
complete, dependency-free example, and `create_teacher` and
`ConstraintsTeacher(fallback=...)` compose it with any other teacher.
