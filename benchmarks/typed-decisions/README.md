# Typed-decisions benchmark

An external test of the multi-head execution plan on the public
[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
suite (pinned revision `c76749ec`): four workflows, 300 train and 100 test
cases each, five typed questions per case over one JSON state (`choice` over
named options, `noul` yes/no, `score` over an ordered rubric). Laya publishes
0.766 on it after fine-tuning and the dataset card lists TypeSafe Jev at 0.727
(general, zero-shot) and a ModernBERT-base specialist at 0.646.

- `program/generate_programs.py` writes one `.sem.ts` per workflow from the
  dataset's question schema: one `sema` expression per question over the
  case's `state` JSON string, `choice` as a string-literal union, `noul` as
  `boolean`, `score` as `BoundedInt` over the rubric levels, with the
  instructions and every option or level description in the behavioral text.
  `semantscript build --project program/tsconfig.json --application typed-decisions`
  compiles all twenty expressions into one bundle whose execution plan puts
  every workflow's five functions in one stage.
- `program/run_typed_decisions.py` replays the train split as the training
  corpus through a dataset teacher (no language model), trains each workflow's
  five heads jointly over one ModernBERT-base encoder and adapter, verifies
  every head with the release gate, scores the test split and times one fused
  stage per case; `program/run-stage-latency.mjs` times the Node runtime's
  `callStage` when an artifact was published.
- Results and the write-up live under `data/results-<date>/`; the dataset
  cache and compiled programs are rebuilt from the pinned dataset and are not
  committed.

This is a specialist measurement in the card's terms: the heads are fitted on
the train split of the very workflows they are scored on.
