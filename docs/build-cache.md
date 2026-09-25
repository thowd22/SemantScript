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
    adversarial/…                             boundary pairs and counterfactual sidecars, keyed by IR, base dataset and teacher
```

`--cache-dir` moves the whole tree; `init` writes `.semantscript/.gitignore`
so neither the artifact nor the cache is committed. The weights are stored as
safetensors, so a cache is roughly the size of the application's encoder.

## What is content-addressed

A cached function is reused only when all of these match the bundle and the
current run:

| Key                  | What it covers                                                                                                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Function id          | Derived by the compiler from the expression's semantic identity: template text, input and output types, examples, constraints and runtime policy, plus its file and duplicate ordinal.     |
| Semantic digest      | The same identity without the file, so a moved expression whose text is unchanged still matches when its id is preserved.                                                                  |
| Model binding digest | The function's `model` block: its adapter ref, encoder ref and depth. Changing a domain's depth or routing changes it.                                                                     |
| Recipe digest        | Encoder name and revision, canonical input version, adapter size, every training, verification and adversarial setting, and the teacher's provider, model and configuration digest.        |
| Shared-state digest  | The combined digest of the encoder weights and the function's adapter weights the head was trained on; a retrained encoder or adapter invalidates every head on it.                        |
| Dataset digests      | The synthetic and adversarial datasets the function trained on. The datasets are cached by content too, so an unchanged expression regenerates nothing; a teacher change regenerates both. |
| File digests         | Every cache file is hashed on read; a corrupted or truncated file is a miss for that function, and a damaged shared file is a miss for the whole application.                              |

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

## What the cache never does

- It never skips verification: a reused function ships the verification record
  it earned; a retrained one is verified again before anything is exported.
- It never reuses across applications or across recipes: the application id
  is the directory and the recipe digest is in every record.
- It never changes a function's identity: the build cache is keyed by ids the
  compiler derives from source, and the plan's stages and domains are outside
  that identity.

## Clearing it

Deleting `.semantscript/cache` (or the `--cache-dir` you passed) is always
safe; the next `train` regenerates datasets through the teacher and trains
from scratch. Deleting one `functions/<function-id>` directory retrains that
function's head on the cached shared state. Deleting the artifact root without
the cache makes the next `train` re-export from cached weights without
training. `--no-cache` is the way to ignore the cache for one run without
deleting it.
