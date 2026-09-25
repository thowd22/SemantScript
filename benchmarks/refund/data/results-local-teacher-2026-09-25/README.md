# Local teacher against the Sonnet 5 reference (TASK-5.13), 2026-09-25

Can we iterate locally, without paying per build, with a local model as the
teacher? Decision-1 makes Sonnet 5 the reference; this run puts a number on
"good enough" for the refund task with the local candidate the task names,
Qwen3-14B. The same 1,800 real refund inputs were labeled by both, a
Sonnet-labeled evaluation set was frozen, and two ModernBERT-base students
were trained on the same 1,500 inputs, one per label set. Driver:
`program/run_local_teacher_experiment.py` (offline tests in
`test_local_teacher_experiment.py`); every number below is in `results.json`.

## Inputs

Real, de-identified refund inputs from the UCI pool (`uci-pool/candidates.json`,
7,597 rows): 194 rows whose input digest is in the frozen final set or the
release-verification set were excluded, 22 duplicates dropped, and 1,800 of
the remaining 7,381 were drawn by a seeded shuffle (seed 1): 300 for
evaluation, 1,500 for training, disjoint (`inputs.json`). Every pool row is a
paid order, so the pool has no fraudulent orders; the rubric steps present are
1 (stale), 3 (outside window), 4 (suspicious history) and 5 (approve).

## Labelers

Both received the benchmark baselines' text: the policy, the six hard
constraints, the input domain note and the canonical refund JSON, and
answered one JSON-schema decision.

| Labeler         | Route                                                                   | Settings                                  | p50 per label | Cost                                    |
| --------------- | ----------------------------------------------------------------------- | ----------------------------------------- | ------------- | --------------------------------------- |
| claude-sonnet-5 | OpenRouter, Anthropic-format `/api/v1/messages`, `output_config` schema | provider default sampling, 256 max tokens | 3.98 s        | USD 4.96 for 1,800 (960 tokens each)    |
| qwen3:14b       | Ollama 0.34.3 native `/api/chat`, `format` schema, ROCm                 | `think: false`, temperature 0, seed 1     | 0.17 s        | none (local GPU, about 5 minutes total) |

## Label agreement (AC1)

Against each other and against the compiled constraints' unique admissible
label (the rule every one of these inputs has, since the policy is complete):

| Rows             | Sonnet = Qwen | Sonnet = rule | Qwen = rule |
| ---------------- | ------------- | ------------- | ----------- |
| training (1,500) | 0.746         | **1.000**     | 0.746       |
| evaluation (300) | 0.793         | **1.000**     | 0.793       |

Per rubric step on the training inputs:

| Step | Rule                                        | Rows | Sonnet = Qwen | Qwen = rule |
| ---- | ------------------------------------------- | ---- | ------------- | ----------- |
| 1    | stale order: deny                           | 33   | 1.000         | 1.000       |
| 3    | paid, outside the tier window: deny         | 68   | 0.618         | 0.618       |
| 4    | paid, inside the window, suspicious: review | 440  | 0.280         | 0.280       |
| 5    | paid, inside the window, otherwise: approve | 959  | 0.960         | 0.960       |

Sonnet 5 agrees with the constraints on every one of the 1,800 inputs: for
this policy it is a perfect labeler. Qwen3-14B misreads two conditionals: it
approves 283 of the 440 suspicious-history cases the policy sends to review
(the compound "three or more prior refunds on an order of at least 1000" and
the five-refund threshold), and approves 26 of 68 orders outside their tier
window. Its disagreements with Sonnet: review/approve 283, review/deny 34,
deny/approve 26, approve/deny 25, approve/review 13.

## The evaluation target (AC2)

`evaluation-set.json`: the 300 evaluation inputs with Sonnet's labels, payload
digest recorded in `results.json` (`evaluationSetSha256`). It is
model-labeled, not human-authored and not judge-adjudicated; it never enters
either training corpus (disjoint by construction, the digests are in
`inputs.json`). Because Sonnet's labels equal the rule labels here, it is
also, in effect, the constraints' verdict on 300 real inputs.

## Students (AC3)

Two ModernBERT-base classifiers, the committed compact recipe (batch 16,
sequence 128, learning rate 3e-5, canonical input v2, 8 epochs, best epoch by
the trainer's 10% held-out split, seed 1), trained on the same 1,500 inputs
with the constraints removed from the IR copy so each teacher's labels are
used exactly as given (the dataset generator would otherwise reject the local
teacher's constraint-violating rows). Temperature fitted on the trainer's
held-out split with the model package's fitter; scored on the evaluation set
and, as a diagnostic, on the 160 judge-attested final cases.

| Student taught by | Selected epoch | Evaluation accuracy | Evaluation ECE | Evaluation Brier | Final set accuracy | Final set ECE |
| ----------------- | -------------- | ------------------- | -------------- | ---------------- | ------------------ | ------------- |
| claude-sonnet-5   | 4              | **0.980** (294/300) | **0.019**      | 0.040            | 0.794 (127/160)    | 0.203         |
| qwen3-14b         | 4              | 0.800 (240/300)     | 0.143          | 0.316            | 0.694 (111/160)    | 0.242         |

The gap on the fixed target is 18 accuracy points and seven times the
calibration error. The Qwen student's misses are its teacher's: the
suspicious-history reviews it was taught to approve.

The final-set numbers are low for both because of the pool, not the teachers:
the final set has 26 fraudulent orders (step 2) and the pool has none, so
neither student ever saw one; 15 of the Sonnet student's 33 final-set misses
are fraud cases answered `approve`, and most of the rest are the outside-window
denials, of which the training inputs hold 68. This is why the compiled
release trains on a threshold-dense synthetic corpus plus real inputs
(decision-7) rather than on real inputs alone, and why it scores 1.000 on the
same set.

## What the numbers say

- **Qwen3-14B is not an acceptable labeling teacher for this task.** Three
  quarters of its labels are right, but the errors are systematic on the two
  compound conditionals, and a student inherits them faithfully: 0.80 against
  0.98 on the same inputs and target.
- **Sonnet 5 is exactly as good as the constraints on this policy**, at
  USD 0.0028 and four seconds per label over the network. For a policy stated
  completely as constraints, the constraints label for free and instantly
  (decision-7; the refund service example trains with no teacher at all), so
  Sonnet's value is on policies whose text says more than their constraints.
- **The local iteration path is constraint labeling, not a local LLM.**
  When a policy has judgment in it, the cheap local option measured here
  costs 18 points; use the reference teacher for generation and keep the
  build cache for iteration. Decision-12 records this.

Records: `results.json`, `inputs.json`, `labels-claude-sonnet-5.json`,
`labels-qwen3-14b.json` (every label with latency and usage),
`evaluation-set.json`. The OpenRouter key was read from the environment and
appears in none of them.
