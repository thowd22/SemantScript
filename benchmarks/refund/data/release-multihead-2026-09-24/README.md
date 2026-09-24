# Fifty-function timing artifact derived from the compact release (TASK-6.5)

`derivation.json` records how `artifact/` (git-ignored, 572 MB, manifest
`68c28a7f…`) was produced by `program/derive_multihead_artifact.py` from the
compact refund release (`data/release-compact-2026-09-24`, manifest
`56c5c5d6…`): fifty clones of the verified refund function over the same
tokenizer, encoder and adapter, each with a fresh function id and head ref and a
byte copy of the verified head graph. The clones exist to count head passes in
`data/results-stage-scaling-2026-09-24`; they are not fifty independently trained
functions and must not be benchmarked for accuracy. Rebuild with:

```bash
python3 benchmarks/refund/program/derive_multihead_artifact.py \
  --source benchmarks/refund/data/release-compact-2026-09-24/artifact \
  --output benchmarks/refund/data/release-multihead-2026-09-24/artifact \
  --functions 50 --record benchmarks/refund/data/release-multihead-2026-09-24/derivation.json
```
