# Training pipeline and release gates

`semantscript train` turns every sema expression of an IR bundle into a
verified head of one application artifact. This page follows one build from
the bundle to the published release and lists every gate a function must pass
on the way. The flags are in the [CLI reference](cli-reference.md#train), the
failure messages and their fixes in the
[diagnostics catalogue](diagnostics.md#verification-failures), and the
caching rules in [build cache](build-cache.md).

## Stages

1. **Datasets.** The teacher labels `--cases` synthetic inputs per expression
   (gold examples count toward it). An expression with constraints also gets
   an adversarial sidecar: a boundary pair per constraint (two inputs one
   field apart with the predicate false on one side and true on the other)
   and counterfactual twins (a single-field edit that changes the label).
   Every case is checked against the expression's types and constraints
   before it is kept. Both datasets are cached, so a rebuild or a seed retry
   sends no teacher request ([teachers](teachers.md)).
2. **Corpus and split.** The gold examples, the synthetic cases and the
   adversarial cases form the training corpus. `--seed` draws a held-out
   calibration and evaluation split from it (`--evaluation-ratio`, default
   0.2); gold examples always train.
3. **Training.** Every function trains over one shared encoder and its
   domain adapter (`--epochs`, `--learning-rate`, `--head-architecture`,
   `--select-best-epoch`). An incremental build trains only the changed heads
   on the frozen shared modules.
4. **Calibration.** One temperature per head is fitted on the held-out split,
   and the split's ECE and Brier score are measured after scaling.
5. **Verification.** The gates below run on the raw (uncalibrated) model.
   A function that fails any gate is not exported. A narrow failure may
   retrain with the next seed ([seed retry](cli-reference.md#seed-retry)).
6. **Export.** Every passing function is written into one immutable release
   under the artifact root, and `current.json` points at it
   ([IR and artifact reference](ir-and-artifact-reference.md)).

## Gates

| Gate                 | Measured on                                                           | Fails when                                                                                   | Seed retry        |
| -------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------- |
| Examples             | Every gold example and attested case                                  | Any prediction differs from the attested output                                              | Never retries     |
| Corpus constraints   | Every corpus record (synthetic, adversarial, gold) and attested case  | Violated checks over records exceed `--max-constraint-violation-rate` (default 0)            | Within the margin |
| Held-out constraints | A sample drawn from the input types and constraint predicates (below) | Inputs that broke a constraint over the sample size exceed `--max-constraint-violation-rate` | Within the margin |
| Output types         | Every decoded prediction                                              | A prediction outside the head's support (only a corrupted head)                              | Never retries     |
| Calibration          | The held-out split after temperature scaling                          | The worst head's ECE exceeds `--ece-threshold` (default 0.1)                                 | Within the margin |
| Pair consistency     | Every counterfactual pair (both members right)                        | Recorded, not gated                                                                          | Not applicable    |

Each failure prints the first failing cases and a `next:` line with the fix
derived from them.

## Held-out constraint check

The corpus constraint check scores only inputs the model trained on. A model
can keep a rule on its boundary pairs and training cases and still break it a
few steps away: the Express example's earlier release `217d386c` passed with zero
corpus violations and answered `review` or `approve` for orders at 100 to 200
days under `always(() => order.ageDays > 90, "deny")`, because its training
inputs past 90 days were the boundary cases at 91 days and a few cases at 132.
The held-out check scores every constraint on inputs the model has never
seen.

**The sample.** For each function with constraints the verifier draws
`--held-out-samples` inputs (default 512, at most 50,000) from the function's
input types and its constraint predicates alone:

- **boundary** inputs: every number on or one step beside a threshold a
  predicate compares it with, plus a single-field edit that flips the
  constraint's predicate, both sides kept;
- **interior** inputs: inputs on each side of the predicate (true, then false)
  whose numbers are all more than one step from every threshold;
- **uniform** inputs: draws spread evenly over each input's inferred range,
  for the rest of the sample (about half).

Numeric ranges are inferred as the constraints teacher infers them (from the
thresholds and the gold examples; a range a constraints-teacher TOML sets is
not applied here), string inputs mostly take the literals the predicates
compare them with, and enums, literals and booleans take their values. The
sample never uses a teacher, a benchmark or any held-out benchmark input, and
it never enters training.

**Disjoint from training.** Every candidate is compared, by its canonical
input encoding, with every corpus row, gold example and attested case, and
repeats are dropped, so no held-out input is a training input. A small input
space the corpus already covers yields fewer inputs; the check scores what is
left and the report records the size it reached.

**Deterministic.** The sample is seeded from the build's first `--seed`, the
function id, the constraint and the category. The same seed draws the same
sample, and every seed retry of one build is scored on that same sample, so a
retry cannot pass by drawing easier inputs.

**Scored like the corpus.** Each sampled input is predicted by the raw model,
and every active constraint is evaluated on its prediction. The held-out rate
is the number of inputs that broke at least one constraint divided by the
sample size (at most 1); the corpus figure counts violated checks over
records, so an input that breaks two constraints counts twice there but once
here. The rate is held to the same `--max-constraint-violation-rate` as the
corpus rate: zero by default, so a single broken held-out input fails the
function. The seed retry classifies it like the corpus rate: under a zero
tolerance it never retries, and under a nonzero one it retries within
`--seed-retry-margin` times the tolerance.

**Budget.** Sampling and scoring share the constraint evaluation budget (10
million predicate steps per request, counted as constraint nodes times
evaluations). The sampler's share is at most 64 draws or predicate
evaluations per requested input, and a size the budget cannot carry for the
function's constraints is refused before sampling, naming the largest
`--held-out-samples` that fits. A category no draw reaches within its share
(a predicate that holds nowhere in the sampled ranges, a region the corpus
already covers) is a coverage shortfall, not an error.

**The failure** names each broken constraint with its count, the seed, and
the first ten offending inputs with the model's answer (one per broken
constraint first, then more of each), then the next step. The Express
example retrained at seed 4 from its cached datasets (2026-09-26, RX 9070 XT,
the README recipe with `--seed 4 --seed-attempts 1`, no teacher request)
kept the corpus within its 1% tolerance (2 of 394 records) and failed here:

```text
held-out constraint check failed on 36 of 512 sampled inputs (0.0703125 exceeds the configured tolerance 0.01; seed 4; broken: constraint 0 on 15, constraint 3 on 10, constraint 4 on 4, constraint 5 on 7; the inputs come from the input types and constraint predicates, none of them a training input):
  - constraint 0 (order.ageDays > 90) violated by held-out input {"customer":{"priorRefunds":0,"tier":"enterprise"},"order":{"ageDays":100,"status":"fraudulent","total":514}}: predicted "review"
  - …
  - constraint 0 (order.ageDays > 90) violated by held-out input {"customer":{"priorRefunds":0,"tier":"enterprise"},"order":{"ageDays":158,"status":"fraudulent","total":372.9}}: predicted "review"
  - and 26 more
  next: add the examples entry { inputs: {"customer": {"priorRefunds": 0, "tier": "enterprise"}, "order": {"ageDays": 100, "status": "fraudulent", "total": 514}}, output: "deny" }: constraint 0 (order.ageDays > 90) is broken on 15 of 512 held-out inputs (seed 4), and the constraints require that output for this one; or rerun with --cases 384 (now 192) so the region is sampled more densely
```

Seeds 3 and 5 on the same datasets broke 70 and 109 of the 512 held-out
inputs (seed 5, which published release `217d386c` before the check, now
also exceeds the corpus tolerance at 8 of 394: GPU training is not
bit-for-bit repeatable). A sweep of seeds 1 to 10 and of training-only
variants (more epochs, other learning rates and batch sizes, an MLP head) on
the same datasets found no passing run: the best broke 20 of 512 (3.9%); its
[README](../examples/express-app/README.md#retrains-from-the-older-cached-datasets-failed-the-held-out-check)
lists every run. A new dataset from the constraints teacher with a Sonnet 5
fallback (`--cases 384 --epochs 16`, USD 0.96) passed at 4 of 512 on its first
seed and published `0fd67142`, which denies orders past 90 days.

When the active constraints require exactly one output for the first
offending input, the fix is that input as an `examples` entry
(`held-out-violation-example`); otherwise it is more `--cases`
(`held-out-violation-more-cases`). An underfit head gets more training first,
as for the other gates.

**Where the figure appears.** The release manifest records it per function as
`verification.heldOutConstraints` (`sampleSize`, `violations`,
`violationRate`, `seed`) beside the corpus `constraintViolations`; a function
without constraints records an empty sample, and a release from before the
check has no such field. The train report carries the same object (plus
`violatedChecks`, the broken checks, and `coverageShortfalls`) under
`functions[].verification.heldOutConstraints` and per attempt, the rendered
report shows a `held-out constraints` column (`3/512 (0.59%)`), and
`semantscript test` and `semantscript releases show` print a `held-out`
column (`-` for a release without the figure or a function without
constraints).

**Cost.** One more forward pass over the sample per function (512 inputs by
default, seconds on a GPU), and no teacher request. Adding the setting
changed the build-cache recipe, so function records cached before the check
retrain once from their cached datasets.
