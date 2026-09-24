# Refund semantic program and lifecycle bridge

`refund-with-confidence.sem.ts` is the diagnostic compiler input for the refund
benchmark. It emits the complete `approve`, `deny`, `review` distribution needed
for calibration measurement. It deliberately contains no examples: frozen
evaluation cases must not leak through compiler examples into training.

The live Python bridge calls the production boundaries in this order:

1. compile the committed source to exact source-stage IR;
2. train with `train_classifier`;
3. verify with a separately held-out, externally attested release-only human set;
4. bind exact verified IR with `build_verified_ir`;
5. export immutable ONNX artifacts with `export_application_artifact`; and
6. execute the exported function through the deployed Node runtime.

The closed release-verification record carries the function identity, stable case
IDs, complete labeled inputs, semantic input digests, exact human attestation,
attestation digest, and payload digest. The parser validates it against compiler
IR and derives the `GeneratedCase` values used by release verification. Callers
cannot supply those cases plus unrelated digest strings.

The final benchmark crosses this boundary only as a typed opaque identity. Its
digest must differ from the release-verification record and every training
dataset; its cases and labels never enter training, calibration, release
verification, or artifact lifecycle. The training-input ledger is
derived from source IR, base and adversarial datasets, the actual deterministic
training split, and the release record. Calibration inputs may legitimately also
appear in their synthetic or adversarial source partition; leakage auditing uses
the union of every lifecycle partition. Before export, the bridge derives the
same versioned semantic training key as the TypeScript publication boundary from
the canonical function and all ledger source identities, then requires the
provided `ArtifactProvenance.training_key_sha256` to match. This binds both the
release payload and its attestation into the exported manifest without changing
the core artifact format.

Compiler and runtime subprocess output is read incrementally with hard stdout,
stderr, and timeout caps. A breach terminates the whole process group, preventing
an output-flood regression from recreating the earlier WSL OOM. This directory
contains no held-out cases, predictions, benchmark metrics, or claimed results.

`claude_cli_teacher.py` is a separate opt-in Sonnet 5 source for synthetic and
adversarial **training-only** cases. Its fixed CLI/model protocol and isolation
rules are documented in `CLAUDE_CLI_TRAINING.md`; it is never a source of
release-verification or final benchmark cases and does not replace the
traditional structured-output API latency baseline.

`quantize_release.py` derives an int8 release from a published float32 release
(the trained PyTorch model is not persisted, so the verified float32 ONNX chain is
the reference). It replays the frozen corpus and the attested release record
with teacher calls forbidden, runs the quantized encoder chain against the
float32 chain over every record, and publishes only when attested decisions,
the overall decision-change rate and the calibration error stay within the
tolerances recorded in the derived manifest. The derived pipeline manifest names
the source manifest digest and carries the full quantization report, so
`run-benchmark.mjs` measures the int8 artifact without changes.

`refund-risk.sem.ts` is a companion function over the same customer and order
records that answers a different question (refund risk: high, low, medium in the
compiler's support order) with every rule as a constraint, so its corpus can be
labeled from the compiled constraints on the real UCI pool without a language
model. `run_application_experiment.py` uses it for the shared-encoder experiment
(TASK-6.1): it replays the refund corpus for the decision function, labels the
pool for the risk function, trains the risk function alone, both functions
jointly over one encoder and adapter, and the risk head on the frozen decision
encoder, verifies both functions of the joint application, exports one
artifact with two heads and calls both through the Node runtime. The risk
function's attested set is the release inputs relabeled by its constraints,
which the report marks as rule-labeled, not judge-adjudicated.

`derive_multihead_artifact.py` and `run-stage-scaling.mjs` measure the
shared-encoder thesis (TASK-6.5). The first clones a published single-function
scalar release N times over its own tokenizer, encoder and adapter (fresh
function ids and head refs, byte-copied head graphs, hard-linked shared
resources) into a content-addressed artifact plus a derivation record naming the
source manifest digest; the clones exist to count head passes, not to claim
independently trained behaviour. The second loads such an artifact and, after
warm-up, times one fused stage of 1, 10 and 50 heads over one input against one
call per head, and one head over 1 to 64 distinct inputs in a stage (the batch
curve), asserting the encoder/adapter/head pass counts and that fused results
equal the single calls. Inputs come from a synthetic corpus, never from an
evaluation set; results and the environment capture land in the output
directory.

Focused offline checks are bounded and do not train a model or publish results:

```sh
node --test benchmarks/refund/program/program.test.mjs
PYTHONPATH=trainer/src:model/src:.python-packages \
  python -m pytest benchmarks/refund/program/test_pipeline.py -q
```
