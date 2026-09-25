# Jev as a diagnostic comparator on the refund final set (TASK-5.17), 2026-09-25

Jev is TypeSafe's hosted decision model: a non-generative model that answers
typed questions about a state (boolean, choice, ordered score) with a
probability per option and a confidence. It is reachable through the user's
OpenRouter account as `~typesafe/jev-latest` (resolved by the service to
`typesafe/jev-1.13-20260917`), on a dedicated endpoint, and priced per input
token. It is the same kind of thing SemantScript compiles per application,
so it belongs next to Laya as an external comparator; this run puts it on the
frozen refund final set under the benchmark's definitions. Driver:
`program/run_jev_comparator.py`; raw answers in `raw-answers.json`; every
number below is in `results.json`.

## The API contract (live probe, AC1)

- **Endpoint**: `POST https://openrouter.ai/api/alpha/decisions`, bearer
  authentication with the OpenRouter key. `chat/completions` refuses the model
  (`is a decisions model and cannot be used with the chat/completions
endpoint`). The model is not in the public or the authenticated model
  listing; only the alias resolves.
- **Request**: `{ model, state, questions }`. `state` is a string, a JSON
  record or an array (the run sends a record: the policy text, the hard
  constraints, the input domain note and the canonical customer and order
  records). `questions` is a record keyed by question id; each question has
  `type` (`noul`, `choice` or `score`), `instructions` (string, record or
  array) and `criteria`: a record of option to description for `choice`, a
  record with `true` and `false` descriptions for `noul`, an ordered array of
  level descriptions for `score`. Any other shape is rejected with a field
  path. No temperature, seed or calibration options exist.
- **Response**: `{ model, answers, usage, id, provider }`. A `choice` answer
  carries `choice`, `probabilities` (one per option, two decimals, summing to
  one) and `confidence`; a `noul` answer carries `noul`, the probability of
  yes; a `score` answer carries `score` (probability-weighted level index),
  `legend`, `probabilities` per level and `confidence`. `usage` reports
  `input_tokens`, `output_tokens` and `cost` in USD; `x-generation-id` and
  `x-provider-name: TypeSafe` come back as headers. Context is 32k tokens.
- **Pricing**: USD 0.042 per million input tokens, output free. The run's
  requests averaged 861 input tokens and cost
  USD 0.0000361 per case; a three-question probe (566 tokens)
  cost USD 0.0000238.
- **Latency**: 147 ms p50 and 222 ms p95 per request from
  this machine, network included (one request at a time). Two of 160 requests
  hit a transient HTTP 520 from the gateway and succeeded on retry.

## Results next to the benchmark systems (AC2)

Same 160 judge-attested final cases, same policy text, hard constraints and
canonical inputs the generative baselines receive, one choice question per
case, sequential. Jev is not a pinned benchmark role, so its predictions are a
diagnostic record (`results.json`), not a sealed system entry in a benchmark
result; the numbers use `metrics.ts`'s definitions re-implemented in the
driver (unit-tested against hand-computed values).

| System                                                       | Accuracy (160)  | Attested slice | 15-bin ECE | p50 / p95 ms  | Where it runs                |
| ------------------------------------------------------------ | --------------- | -------------- | ---------- | ------------- | ---------------------------- |
| SemantScript, depth-6 release (results-depth-006-2026-09-25) | 1.000 (160/160) | 1.000          | 0.003      | 4.8 / 6.4     | CPU, Node runtime            |
| SemantScript, full-depth release (results-v2-2026-09-23)     | 0.994 (159/160) | 0.994          | 0.011      | 38.4 / 50.5   | CPU, Node runtime            |
| **Jev `typesafe/jev-1.13-20260917` (this run)**              | 0.963 (154/160) | 0.963          | 0.031      | 147.5 / 221.8 | OpenRouter, network included |
| Qwen 2.5 7B instruct Q4_K_M (Ollama)                         | 0.537 (86/160)  | 0.537          | 0.428      | 569.2 / 608.2 | local GPU                    |
| Qwen 2.5 1.5B instruct Q4_K_M (Ollama)                       | 0.519 (83/160)  | 0.519          | 0.443      | 319.4 / 357.8 | local GPU                    |
| Laya typed-decisions checkpoint                              | 0.225 (36/160)  | 0.225          | 0.411      | 28.2 / 31.9   | local CPU                    |

Jev answers 154 of 160 correctly with a
calibration error of 0.031 and a multiclass Brier score of
0.046: far above the generative baselines and Laya, and
6 cases short of the compiled function's 160. Its top-1 probability averages 0.97; the
misses are not low-confidence abstentions (top-1 from 0.48 to 0.83 on them).

## Agreement per policy rule (AC3)

Agreement with the judge's label, grouped by the rubric step that decided
each case:

| Step | Rule                                                | Agree | Agreement | Confusions       |
| ---- | --------------------------------------------------- | ----- | --------- | ---------------- |
| 1    | stale order (ageDays > 90): deny                    | 28/28 | 1.000     |                  |
| 2    | fraudulent within 90 days: review                   | 26/26 | 1.000     |                  |
| 3    | paid, outside the tier window: deny                 | 39/39 | 1.000     |                  |
| 4    | paid, inside the window, suspicious history: review | 9/10  | 0.900     | review->deny ×1  |
| 5    | paid, inside the window, otherwise: approve         | 52/57 | 0.912     | approve->deny ×5 |

Every miss is the same mistake. All six are enterprise-tier paid orders aged
33 to 35 days, which the policy allows (the enterprise window is 60 days) but
Jev denies, as if the 30-day standard window applied:

- expected `approve`, Jev `deny` (approve 0.15, deny 0.83, review 0.02): {"customer":{"priorRefunds":0,"tier":"enterprise"},"order":{"ageDays":34,"status":"paid","total":252.2}}
- expected `approve`, Jev `deny` (approve 0.39, deny 0.56, review 0.05): {"customer":{"priorRefunds":2,"tier":"enterprise"},"order":{"ageDays":33,"status":"paid","total":576.1}}
- expected `approve`, Jev `deny` (approve 0.15, deny 0.82, review 0.03): {"customer":{"priorRefunds":0,"tier":"enterprise"},"order":{"ageDays":34,"status":"paid","total":534.7}}
- expected `review`, Jev `deny` (approve 0.16, deny 0.54, review 0.30): {"customer":{"priorRefunds":4,"tier":"enterprise"},"order":{"ageDays":33,"status":"paid","total":1588.16}}
- expected `approve`, Jev `deny` (approve 0.34, deny 0.48, review 0.18): {"customer":{"priorRefunds":3,"tier":"enterprise"},"order":{"ageDays":33,"status":"paid","total":304.0}}
- expected `approve`, Jev `deny` (approve 0.26, deny 0.65, review 0.09): {"customer":{"priorRefunds":3,"tier":"enterprise"},"order":{"ageDays":35,"status":"paid","total":700.31}}

On the 16 enterprise paid orders aged 31 to 60 days, the slice
where the tier-conditional window is the whole question, Jev agrees on
10 (0.625); everywhere else it agrees on
144 of 144. The stale rule, the fraud rule, the
outside-window denials and the ordinary approvals are all read correctly; the
one conditional the model does not resolve reliably is the window that depends
on a second field.

**Cost of a 10k-case typed corpus.** At the measured USD 0.0000361 per
case (one question, 861 input tokens with the full
policy in the state), 10,000 cases cost about **USD 0.36**; a
five-question case of typed-decisions size would be a few times that. The constraint is not price but label quality: on this policy Jev's
labels would carry a systematic 3.75% error concentrated on one rule, which
the compiled constraints would catch (every enterprise 31-to-60-day denial
violates the always-approve or always-review constraint) and reject.

## What the numbers say

- **Jev is a strong comparator for the typed-decision family**: 0.96 on a
  policy it has never seen, given only the text, at a hundredth of a cent per
  decision, against 0.54 for a 7B generative model reading the same text.
- **It is not a substitute for the compiled function.** The application's
  own artifact is at 1.000 on this set at 5 ms on the CPU with no network,
  and it is verified against the constraints; Jev is at 0.9625 at 150 ms
  over the network with no way to bind the policy's hard constraints into
  the answer.
- **As a data source it is usable only behind the constraints.** Its errors
  are systematic, not noise, so a corpus labeled by Jev must be filtered by
  the compiled constraints (decision-7's real-input path already does this
  for teacher labels), after which the surviving labels are cheap. The
  north-star test (per-application accuracy or faster compilation) is met
  only in that role; see decision-11.

Records: `results.json` (metrics, agreement, usage, every prediction with its
distribution), `raw-answers.json` (every response as received). The
OpenRouter key was read from the environment and appears in neither.
