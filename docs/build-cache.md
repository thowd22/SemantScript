# The build cache

Retraining every expression on every build would make `sema` unusable, so
`semantscript train` keeps a content-addressed cache and trains only what
changed. This page says what the cache keys on, when retraining happens
anyway, where everything lives and how to clear it.

## Where it lives

```text
.semantscript/
  artifact/                                   the published releases (current.json, releases/)
  cache/
    applications/<application-id>/
      application.json                        cache version, recipe digest, encoder and adapter digests, last release, function index
      shared.safetensors                      the application's fine-tuned encoder weights, exact bits
      adapters/<adapter-ref>.safetensors      one per domain (dots in the ref become "__")
      functions/<function-id>/
        function.json                         semantic and recipe digests, dataset digests, epoch metrics, verification record
        verified-ir.json                      the exact verified IR bytes the artifact was exported from
        head.safetensors                      the function's head weights
    datasets/…                                synthetic corpora, keyed by IR, teacher and case count
    adversarial-datasets/…                    boundary pairs and counterfactual sidecars, keyed by IR, base dataset and teacher
    teacher-responses/<teacher digest>/…      every paid direct Anthropic response the run accepted, keyed by the exact request and its occurrence, and every submitted Message Batch's handle (batches/)
    teacher-prices.json                       OpenRouter's price list, fetched at most once a day
    teacher-stats.json                        the mean request latency of the last metered run, per teacher
```

The response journal is what makes a spend cap cheap to hit: a run stopped by
`--max-cost-usd` (or by a crash, a network error or Ctrl-C) keeps every
response it paid for, and the next run with the same teacher and settings
sends the same requests in the same order, so each one is answered from the
journal at no cost and the run carries on where it stopped. A dataset, once
complete, comes from the dataset cache and needs no journal. The journal
holds response text and token counts only, never a key; `--no-cache` neither
reads nor writes it.

The journal keeps only answers worth replaying. A response that breaks the
case or adversarial contract is dropped as soon as it is decoded, and a run
that fails because the teacher's answers were rejected (for example
`did not yield a valid two-sided boundary pair within 3 attempts`) drops every
entry it replayed or wrote and says so
(`note: the teacher's answers were rejected, so the <n> journaled response(s) this run used were discarded …`),
so the rerun asks the teacher again instead of failing on the same answers.
Message Batches (`mode = "batch"`, or `auto` at `batch_threshold`) are
journaled by their handle: the batch ID is written to `teacher-responses/…/batches/`
as soon as the batch is submitted, keyed by the digest of the exact batch
request. A run stopped while a batch is processing (by a crash, Ctrl-C, a
network error or the poll timeout) therefore collects that same batch on the
rerun instead of submitting and paying for a new one; the batch is charged
when it is collected. A collected batch stays recorded, so a rerun that sends
the identical batch request again re-reads its results at no cost for as long
as Anthropic keeps them (29 days); a recorded batch the provider no longer has,
or whose results have expired, is submitted once more. A batch whose results
are rejected is forgotten like a rejected direct response; every item the
provider billed is still charged to the run's meter first. The one gap is a stop in the moment between the
provider accepting a batch and the handle reaching the disk: that batch is not
found again.

`--cache-dir` moves the whole tree; `init` writes `.semantscript/.gitignore`
so neither the artifact, the cache nor the traceback `train` keeps next to its
report (`artifact.report.traceback.txt`) is committed. The weights are stored as
safetensors, so a cache is roughly the size of the application's encoder.

## What is content-addressed

A cached function is reused only when all of these match the bundle and the
current run:

| Key                  | What it covers                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Function id          | Derived by the compiler from the expression's semantic identity: template text, input and output types, examples, constraints and runtime policy, plus its file and duplicate ordinal.                                                                                                                                                                                                                                                                                     |
| Semantic digest      | The same identity without the file, so a moved expression whose text is unchanged still matches when its id is preserved.                                                                                                                                                                                                                                                                                                                                                  |
| Model binding digest | The function's `model` block: its adapter ref, encoder ref and depth. Changing a domain's depth or routing changes it.                                                                                                                                                                                                                                                                                                                                                     |
| Recipe digest        | Encoder name and revision, canonical input version, adapter size, every training, verification and adversarial setting (with the configured `--seed`), and the teacher's provider, model and configuration digest. The seed retry settings are not part of it (below).                                                                                                                                                                                                     |
| Shared-state digest  | The combined digest of the encoder weights and the function's adapter weights the head was trained on; a retrained encoder or adapter invalidates every head on it.                                                                                                                                                                                                                                                                                                        |
| Dataset digests      | The synthetic and adversarial datasets the function trained on. The datasets are cached by content too, so an unchanged expression regenerates nothing; a teacher change regenerates both. A language-model teacher's digest includes its prompt layout version, so the compact prompt of 2026-09-25 (version 4) regenerates Anthropic and Ollama datasets once; the constraints teacher's datasets are unaffected. A `[teacher.pricing]` table is never part of a digest. |
| File digests         | Every cache file is hashed on read; a corrupted or truncated file is a miss for that function, and a damaged shared file is a miss for the whole application.                                                                                                                                                                                                                                                                                                              |

The cache version is part of every record (`cacheVersion`); a cache written by
an older trainer layout is treated as empty.

## When retraining happens

| Situation                                                                                | What trains                                                                                                                                               | What is exported                                                                                                                                                           |
| ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Nothing changed                                                                          | Nothing.                                                                                                                                                  | Nothing, when the cache's last release is still the artifact root's current release (the report says `reused`); otherwise the artifact is re-exported from cached weights. |
| One expression's text, types, examples or constraints changed                            | That function's head only, on the frozen encoder and adapters restored from the cache. It is verified like any other before export.                       | A new release with every other function's weights and verification byte for byte.                                                                                          |
| An expression is added to an existing domain                                             | Its head, as above.                                                                                                                                       | A new release.                                                                                                                                                             |
| An expression is added in a new domain                                                   | A fresh adapter for that domain and the head, on the frozen encoder. Decision-10 notes this trains worse than a joint run; use `--full` before a release. | A new release.                                                                                                                                                             |
| A domain's depth or the routing changed                                                  | Every function whose model binding changed, as heads on the frozen encoder; use `--full` to retrain the encoder for the new layout.                       | A new release.                                                                                                                                                             |
| The recipe changed (any training, verification, adversarial, encoder or teacher setting) | Everything, jointly: the shared encoder moves, so every earlier function record is discarded.                                                             | A new release.                                                                                                                                                             |
| `--full`                                                                                 | Everything, jointly, regardless of the cache; the cache is rewritten.                                                                                     | A new release.                                                                                                                                                             |
| `--no-cache`                                                                             | Everything, jointly; nothing is read from or written to the cache.                                                                                        | A new release.                                                                                                                                                             |

Heads trained incrementally sit on an encoder that was fine-tuned for the
application's earlier expressions. That is by design (it is what keeps a save
cheap in `semantscript dev`), and the reason to run `--full` before a release
that matters.

## Seed retries

When verification fails narrowly, `train` retrains with the next seed
([seed retry](cli-reference.md#seed-retry)). The datasets are generated (or
read from this cache) once, before the first attempt, so every retry reuses
them: a retry calls no teacher, adds nothing to the spend line and costs
training time only. On an incremental build each attempt restores the shared
encoder and the unchanged heads from the cache again and retrains only the
changed heads.

The recipe digest keeps the configured `--seed`, not the seed that passed, so
rebuilding with the same flags reuses a release a retry published instead of
training it again. The seed a function actually trained with is in its
verified IR (`trainingProvenance.seed`), in the published release's manifest
(`functions[].trainingProvenance.seed`) and in the report
(`functions[].training.seed`); when the cache restores a function it rebuilds
that function's held-out split from the recorded seed. `--seed-attempts` and
`--seed-retry-margin` change how many seeds a build may try, not what a head
is, so changing them never invalidates the cache.

## What the cache never does

- It never skips verification: a reused function ships the verification record
  it earned; a retrained one is verified again before anything is exported.
- It never reuses across applications or across recipes: the application id
  is the directory and the recipe digest is in every record.
- It never changes a function's identity: the build cache is keyed by ids the
  compiler derives from source, and the plan's stages and domains are outside
  that identity.

## Releases, rollback and prune

The build cache and the artifact root are separate. Every `train` publishes an
immutable release under `<artifact-root>/releases/` and points `current.json`
at it; `semantscript releases` lists them, `releases rollback` points
`current.json` back at an earlier one and `releases prune` deletes old ones
(see the [CLI reference](cli-reference.md#releases)). Pruning frees disk but
never touches the cache, and never removes the current release.

The cache reuses a published release only while `current.json` still names
the release it last published. After a rollback, the next `train` therefore
exports the cached state again as a new release (its manifest carries a new
`build.createdAt`) and makes it current: roll back to keep an older release in
service, and train again only once the regression is fixed.

## Clearing it

Deleting `.semantscript/cache` (or the `--cache-dir` you passed) is always
safe; the next `train` regenerates datasets through the teacher (paying for
them again: run `semantscript train --estimate` first to see how much) and
trains from scratch. Deleting `teacher-responses/` alone is always safe:
the only cost is that a stopped run pays again for the responses it had. Deleting one `functions/<function-id>` directory retrains that
function's head on the cached shared state. Deleting the artifact root without
the cache makes the next `train` re-export from cached weights without
training. `--no-cache` is the way to ignore the cache for one run without
deleting it.

`semantscript explain` reads the cache too: the training cases it lists
beside a wrong answer come from the dataset the current release was trained
on, the `datasets/v1` file whose SHA-256 is the manifest's
`trainingProvenance.datasetSha256`, plus its adversarial sidecar. After the
cache is deleted, or for a release trained with `--no-cache` (which writes no
dataset), explain shows the newest cached dataset for the function marked as
not the release's, or no training cases at all; the answer, the constraints
and the gold examples still show. Keep the cache while you work through a
wrong answer: explain it, add the example or constraint the
[wrong-answer workflow](diagnostics.md#wrong-answer-workflow) names, and the
next `train` retrains only that function's head from the cached shared state.
