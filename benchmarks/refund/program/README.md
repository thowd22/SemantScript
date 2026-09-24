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

Focused offline checks are bounded and do not train a model or publish results:

```sh
node --test benchmarks/refund/program/program.test.mjs
PYTHONPATH=trainer/src:model/src:.python-packages \
  python -m pytest benchmarks/refund/program/test_pipeline.py -q
```
