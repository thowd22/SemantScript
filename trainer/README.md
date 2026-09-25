# Trainer

The trainer consumes NeuralFunction IR, generates synthetic and adversarial cases,
coordinates teacher backends, trains function heads and adapters, fits calibration,
and verifies examples, constraints, types, and quality gates.

It emits trained and verified lifecycle records for the model exporter. It is not
part of the production inference path.

## Synthetic datasets and cache

`SyntheticDatasetGenerator.generate(ir, total_cases)` treats `total_cases` as the
exact size of the returned dataset, not as the number of new teacher requests.
Every IR example is validated first, retained verbatim in source order, and marked
`gold`; only `total_cases - gold` cases are requested from the configured
`CaseGenerator` and marked `synthetic`. A total below the number of gold examples
is rejected before contacting the teacher. A zero total is valid only when the IR
has no examples, and no teacher call is made when the gold examples already fill
the requested total. One dataset is capped at 20,000 rows and 64 MiB of canonical
JSON to bound the completed in-memory tuple and cache footprint.

A teacher case whose label violates an active constraint is a teacher mistake,
not data: the constraints are the authority on those inputs. The generator drops
it and requests the shortfall again, at most `MAXIMUM_REPLACEMENT_ROUNDS` (3)
times, logging the count; if the shortfall remains it raises
`TeacherResponseError`. A malformed case (schema, a mutated IR) stays fatal.

The persisted format is this closed JSON v1 envelope (fixed-shape objects reject
undeclared fields; `inputs` and `output` remain governed by the function IR):

```json
{
  "kind": "semantscript.training-dataset",
  "datasetVersion": 1,
  "payloadSha256": "<64 lowercase hexadecimal characters>",
  "payload": {
    "requestSha256": "<64 lowercase hexadecimal characters>",
    "function": {
      "id": "<function id>",
      "semanticSha256": "<IR semantic digest>"
    },
    "teacher": {
      "provider": "<provider>",
      "model": "<model>",
      "configurationSha256": "<teacher configuration digest>"
    },
    "counts": {
      "total": 2,
      "gold": 1,
      "synthetic": 1
    },
    "cases": [
      { "origin": "gold", "inputs": {}, "output": false },
      { "origin": "synthetic", "inputs": {}, "output": true }
    ]
  }
}
```

`counts.total` equals `counts.gold + counts.synthetic` and the length of
`cases`. `payloadSha256` domain-separately integrity-checks the canonical `payload`.
The returned `dataset_sha256` instead hashes the exact canonical UTF-8 file bytes,
including their final LF, and is deliberately not embedded in the file it hashes.

Canonical JSON v1 uses UTF-8 without a byte-order mark, lexicographically sorted
object keys, no insignificant whitespace, literal non-ASCII characters, finite
numbers only, and a single final LF for the envelope file. Python's JSON encoder
uses the shortest round-trippable binary64 spelling and retains `-0.0`. The
request digest is `SHA-256("semantscript.dataset-request/v1\0" || canonical
request bytes)`; the payload digest is
`SHA-256("semantscript.dataset-payload/v1\0" || canonical payload bytes)`. The
NUL shown as `\0` is one zero byte, not two printable characters. These are
integrity digests, not signatures or message-authentication codes.

The pinned empty-dataset vector is
`trainer/tests/training_dataset_v1_empty.golden.json`: its request, payload, and
exact-file digests are respectively `2c5025cf99b3df4694502ff42d03ad0a06fe5d6b66bf0453184c5b49ca92491f`,
`c740c9738c6cb8bd4e4f55631fa3f46f77893e38d00ee9cc8ff37acc14e20919`, and
`996b74ba663ab374d08c7e37bd169ceec2f2411695d6cbf1e4e9239b64ac474b`.

The domain-separated `requestSha256` cache key covers the canonical prompt-relevant
IR projection (`kind`, `irVersion`, `id`, `semanticSha256`, `definition`, `inputs`,
and `output`), the complete teacher descriptor (`provider`, `model`, and
`configurationSha256`), `total_cases`, and explicit generator, dataset, prompt,
and case-contract revisions. Source locations and later training or verification
metadata are excluded, so they cannot create spurious generation misses.

Cache hits are accepted only after the closed envelope, request identity, metadata,
counts, cases, and both relevant digests have been revalidated. Generation uses a
bounded cache location, a per-key process lock and second-look read, then writes and
fsyncs a same-directory staging file, verifies it, and atomically publishes it.
Corrupt or mismatched committed entries fail closed: they raise an error instead of
silently spending provider credit. Interrupted staging cannot replace a previously
committed result. The hashes detect accidental corruption but do not protect
against an attacker who can rewrite both content and digest; place the cache only
in a trusted, access-controlled local directory.

Resumability is therefore at completed-result granularity. A valid published result
can be reused across processes without another teacher call, but an in-flight
provider request has no per-case checkpoint: the current `Teacher.generate(ir, n)`
protocol returns one atomic tuple, so retrying a failed request may regenerate its
entire synthetic remainder.

The teacher descriptor records a declared model name, not immutable resolved
weights. For reproducible caches, use an immutable model identifier (and record a
resolved Ollama digest where applicable); if a mutable alias or tag is retargeted,
change the configured identity or explicitly invalidate its cached datasets.

## Constraint boundaries and counterfactual sidecars

`AdversarialDatasetGenerator.generate(ir, base_dataset)` preserves the exact-size,
closed training-dataset v1 format above and writes a separate closed
`semantscript.adversarial-dataset` v1 sidecar. TASK-5.6 can train over the base rows
plus the sidecar rows without redefining what `total_cases` means. The sidecar key
covers the exact base-dataset file digest, prompt-relevant IR, adversarial teacher
descriptor, normalized generation configuration, and explicit format, prompt,
pair, generator, and constraint-evaluator revisions. Valid cache hits make no
provider calls; malformed or mismatched committed entries fail closed.

For every `always` or `never` constraint, the teacher proposes a two-row boundary
pair. The trainer independently evaluates the closed v1 predicate AST, requires
one predicate-false and one predicate-true row differing at exactly one JSON path,
validates both rows against the IR, and enforces every active constraint before a
label can enter either the base dataset or sidecar. Boundary rows carry the
`constraint-boundary` tag, constraint index, and recomputed predicate result so
verification can report constraint accuracy separately.

The ordinary counterfactual candidates are the non-gold rows in the base dataset.
`counterfactual_ratio=1.0` twins all of them and `0.0` disables them; intermediate
ratios use the nearest row count (exact halves round upward) after deterministic
digest ranking. Mandatory constraint-boundary rows are independent of this ratio.
Each selected anchor and its teacher-proposed twin are stored as two
`counterfactual` rows linked by a stable pair ID. Locally enforced pair invariants
require exactly one changed JSON path, a different output value, a nonblank bounded
teacher-stated reason, valid input/output types, and no active constraint violation.
The pair table records both case IDs, the base source-row index, changed JSON
Pointer, and reason for later pair-consistency reporting.

```python
from semantscript_trainer import (
    AdversarialDatasetGenerator,
    AdversarialGenerationConfig,
)

sidecar = AdversarialDatasetGenerator(
    teacher,
    cache_directory,
    config=AdversarialGenerationConfig(
        counterfactual_ratio=0.5,
        maximum_attempts=3,
    ),
).generate(ir, base_dataset)
```

The compiler intentionally permits vacuous constant constraints. A constant or
otherwise one-sided predicate cannot truthfully produce both boundary sides, and a
label-flipping one-path twin does not necessarily exist for every behavior. Such
requests fail with a typed, bounded generation error after at most
`maximum_attempts`; the trainer never fabricates a side, silently omits a pair, or
retries without limit. The evaluator uses JavaScript-compatible strict equality,
UTF-16 string ordering/length, truncating remainder semantics, boolean
short-circuiting, and binary64 arithmetic, and fails if evaluation produces a
non-finite intermediate. Constraint ASTs are snapshotted and compiled once per
request. Aggregate AST work, UTF-16 scan work, row counts, and retained canonical
sidecar bytes are independently bounded so large datasets, exponents, or strings
cannot turn validation into an unbounded compiler/trainer workload.

Load compiler IR with `semantscript_trainer.loads_strict_json`, not the standard
library's default `json.loads`. The strict loader applies SemantScript's direct
binary64 numeric interpretation, rejects duplicate keys and non-finite values, and
preserves signed negative zero for literal/type validation before dataset assembly.

An anchor whose counterfactual the teacher cannot produce within
`maximum_attempts` (typically an input with no single-field edit that changes
the label) does not fail the build: the generator logs it and takes the next
synthetic case in the deterministic ranking as a spare, up to as many skips as
pairs wanted, and the cached sidecar records the anchors actually used, which
the loader checks against that ranking.

## Model input serialization boundary

Dataset canonical JSON is the integrity and cache representation; it is not the
encoder input format. Before tokenization, training serializes every base or sidecar
row with the same typed `semantscript.canonical-input/v1` contract used at runtime:

```python
from semantscript_trainer.canonical_input import serialize_canonical_inputs

model_text = serialize_canonical_inputs(
    ir["inputs"],
    case.inputs,
    maximum_bytes=artifact_maximum_input_bytes,
).decode("utf-8", errors="strict")
```

The serializer validates the row against the ordered IR input schema and preserves
the runtime distinctions for binary64 numbers, negative zero, enum members, tuples,
objects, and unions. Its bounded UTF-8 result is the tokenizer input. Do not append
the function definition, constraint prose, teacher prompt, cache metadata, or label.
Base and adversarial-sidecar rows share this path; sidecar inclusion does not change
the base dataset's `total_cases` contract.

## Encoder fine-tuning

`train_classifier` identity-checks the generated base dataset and optional
adversarial sidecar, derives the head support and logit width from the IR output,
makes a deterministic group-aware held-out split, and fine-tunes both the encoder
and a linear or one-hidden-layer MLP head. A scalar output trains one head; a flat
interface output trains one independent head per field (`derive_output_heads`,
`FieldHeads`), every row carries one label index per head, the loss is the sum of
the per-head losses, and `EpochMetrics` reports exact-match accuracy over all
fields plus `held_out_field_accuracy` per field. Counterfactual sources and twins,
and the two sides of each constraint boundary, remain in one partition.

```python
from semantscript_trainer import TrainingConfig, train_classifier

result = train_classifier(
    ir,
    base_dataset,
    adversarial_dataset,
    config=TrainingConfig(
        device="cuda",  # ROCm also uses PyTorch's cuda device name
        epochs=3,
        batch_size=8,
        loss="proper",  # or "cross_entropy" as the baseline
        head_architecture="linear",
    ),
)

print(result.held_out_accuracy)
```

The default loss is log plus spherical score, with normalized ranked probability
score added for ordinal outputs. Training refuses a split without an independent
held-out group, duplicate canonical inputs crossing the split, conflicting labels,
non-finite tensors, or work exceeding the explicit row, text, token, batch-pass,
parameter, and optimizer-step caps. The result retains the exact dataset digests,
model configuration, per-epoch loss, and held-out accuracy needed by calibration
and later artifact provenance.

## Calibration and verification

`verify_training_result` reconstructs the exact training corpus, checks its dataset
digests and split, reruns every gold example, and evaluates every deterministic
constraint against every adversarial prediction. It fits one positive temperature
on the non-gold held-out partition, then reports held-out accuracy, equal-width ECE,
normalized Brier score, and counterfactual pair consistency. A pair is consistent
only when both its anchor and twin match their expected support values; a dataset
with no pairs reports the vacuous value `1.0` and retains an internal pair count of
zero.

```python
from semantscript_trainer import VerificationConfig, verify_training_result

verification = verify_training_result(
    ir,
    result,
    base_dataset,
    adversarial_dataset,
    config=VerificationConfig(ece_threshold=0.05),
)

head_metadata = verification.to_manifest_head_metadata()
function_verification = verification.to_manifest_function_verification()
```

Any gold or external attested-example miss, observed constraint violation beyond the configured `maximum_constraint_violation_rate` (default zero), or ECE
above the configured threshold raises `VerificationGateError` with the complete
failed result attached. Malformed, non-finite, or incorrectly shaped classifier
logits abort measurement with `VerificationExecutionError`; predictions decoded
through a valid head are support members, so completed records have zero output
type errors. A flat interface output is verified per field: each head gets its own
temperature, ECE, Brier, accuracy, and pair consistency (`HeadVerificationV1` with
the field's JSON-pointer `outputPath`), the function-level accuracy is exact match
over every field, the function-level ECE and Brier are the worst head's, and the
pair consistency is the lowest head's, so the gate never passes on an average.
At least one attested gold or external case and a
nonempty held-out calibration partition are required. External attested cases must be
disjoint from training inputs. Failed results can be serialized as IR verification
diagnostics but cannot be projected into artifact-manifest metadata; TASK-5.8
consumes the two passing projections during export.

Verification also fingerprints the exact model state and tokenizer JSON before
inference and checks both again afterward. These internal digests are intentionally
absent from IR and manifest projections; `export_application_artifact` recomputes
them and rejects weights changed after verification or different tokenizer bytes.
Hugging Face fast tokenizers are serialized through
`backend_tokenizer.to_str(pretty=False)`. A trusted injected tokenizer can expose
the same bytes as `semantscript_tokenizer_json`.

## Verified IR lifecycle

`build_verified_ir` is the supported bridge from compiler source IR to artifact
export. It accepts only an exact closed source-stage record with pending
training and verification fields, a matching `TrainingResult`, passing
`VerificationResult`, and explicit `VerifiedIrProvenance`. It derives and checks
the example, synthetic, adversarial, calibration, verification, and attested
counts against the retained split and verification evidence; it also binds the
teacher configuration, dataset digest, training seed, immutable encoder revision,
weights digest, trainer identity, and ordered training/verification timestamps.

```python
from semantscript_trainer import (
    TrainingProvenanceCounts,
    VerifiedIrProvenance,
    build_verified_ir,
)

built = build_verified_ir(
    compiler_source_ir,
    training,
    verification,
    VerifiedIrProvenance(
        teacher=teacher.descriptor,
        base_model_name=training.config.encoder_name,
        base_model_revision=training.config.encoder_revision,
        base_model_weights_sha256=encoder_weights_sha256,
        dataset_sha256=training.base_dataset_sha256,
        counts=TrainingProvenanceCounts(
            examples=human_example_count,
            synthetic=synthetic_count,
            adversarial=adversarial_count,
            calibration=training.held_out_row_count,
            verification=verification_case_count,
            attested_verification=verification.attested_cases,
        ),
        seed=training.config.seed,
        trainer_version="0.1.0",
        trainer_commit="1234abc",
        trained_at="2026-09-23T00:00:00Z",
    ),
)
```

`built.document` returns a fresh export-ready IR object and
`built.source_ir_bytes` is its deterministic UTF-8, UTF-8-key-sorted, two-space
indented JSON representation with LF line endings and one trailing LF. The builder
does not mutate the compiler record. Both the builder and exporter call
`validate_verified_ir_binding`, so hand-assembled or stale lifecycle documents do
not bypass the artifact boundary.

## Application artifact export

`export_application_artifact` accepts the verified IR together with its exact
source bytes, a passing `TrainingResult`/`VerificationResult`, the exact tokenizer
JSON, explicit application/build metadata, and a fixed parity batch. It exports
and validates the three ONNX graphs, projects calibration and verification into
the closed artifact manifest, and publishes this layout:

```text
artifact/
  current.json
  releases/sha256-<manifest-sha256>/
    manifest.json
    tokenizer/tokenizer.json
    models/encoder/model.onnx
    models/adapters/application.onnx
    models/heads/<function-id>/head-000.onnx
    models/heads/<function-id>/head-001.onnx   # one per output head, in IR order
```

A scalar function publishes one head resource with an empty manifest `outputPath`;
a flat interface function publishes one per field with `outputPath` `[field]`,
per-head calibration and verification metadata, and the `all-fields` policy when
it is thresholded. The int8 derivation (`quantize_release_artifact`) still accepts
only single-function scalar releases.

The release is built in a private directory under `releases`, resources are
hashed before the deterministic manifest is written, and the release directory is
published with an OS-native atomic no-clobber rename before `current.json` is
atomically replaced. On POSIX systems, staging writes and existing-release checks
are anchored to opened directory descriptors and refuse path-identity changes.
Existing immutable releases
are reused only after every byte length and digest is rechecked. Export enforces an
8 MiB manifest limit, 64 MiB tokenizer limit, 1 GiB per-resource limit, and 2 GiB
aggregate resource limit.
The per-resource limit is applied to each ONNX component before protobuf parsing
or ONNX Runtime session creation, and parity tolerances are capped at `1e-3`.

```python
import torch

from semantscript_trainer import ArtifactProvenance, export_application_artifact

artifact = export_application_artifact(
    "build/semantscript-artifact",
    verified_ir,
    training,
    verification,
    tokenizer_json=exact_tokenizer_json,
    source_ir_bytes=exact_verified_ir_bytes,
    provenance=ArtifactProvenance(
        application_id="refund-service",
        application_version="1.0.0",
        compiler_version="0.1.0",
        trainer_version="0.1.0",
        created_at="2026-09-23T00:00:00Z",
        training_key_sha256=training_key_sha256,
    ),
    input_ids=torch.tensor([[101, 102]], dtype=torch.int64),
    attention_mask=torch.tensor([[1, 1]], dtype=torch.int64),
)
```

The verified IR must contain complete teacher, immutable base-model, dataset, and
verification provenance matching the supplied training result. The caller must
provide the training-key digest because its cache-key derivation is outside this
export boundary.

## Bundle driver (`semantscript train`)

`semantscript_trainer.cli.train_bundle` turns one compiler IR bundle into one
artifact: for every source-stage function it generates the synthetic dataset
through the configured teacher (and the adversarial sidecar when the function
declares constraints, which needs a teacher with boundary and counterfactual
generation), trains one classifier for a single function or the shared-encoder
application for several, verifies every function, binds verified IR with the
derived provenance counts and exports the artifact with a training key over the
dataset digests. A function that fails verification raises
`TrainBundleFailure` carrying the report; nothing is published. The report
(`semantscript.train-report`) lists, per function, the dataset and adversarial
digests and sizes, held-out accuracy, the verification status, failures and
metrics, plus the published release digest.

```bash
python -m semantscript_trainer.cli train \
  --bundle dist/semantscript.ir.v1.json --artifact dist/artifact \
  --teacher teacher.toml --cache-dir .semantscript/cache --report dist/train-report.json \
  --cases 64 --epochs 3 --device cuda --local-files-only
```

`teacher.toml` holds the `[teacher]` table of the section below (or pass
`--teacher constraints` for the built-in constraints teacher). An expression
with no gold examples stops the driver before any generation, because
verification needs at least one attested example per expression. Training,
verification and adversarial settings map to `TrainingConfig`,
`VerificationConfig` and `AdversarialGenerationConfig` fields; unspecified
flags keep the library defaults. The Node CLI (`cli/`) spawns this module and
renders the report.

Every function trains over one shared encoder and adapter, and
`semantscript_trainer.build_cache` keeps the result under
`<cache-dir>/applications/<application-id>/`: the exact encoder and adapter
weights, and per function the head weights, the verified IR bytes, the
verification record and the digests binding them (recipe, shared state,
datasets, model state). On the next build a function whose id, semantic digest,
recipe, shared-state digest and dataset digests all match is reused without
training; a changed function trains only its head with `add_function_head` on
the restored, frozen shared modules; a recipe change, `--full` or `--no-cache`
retrains everything jointly and discards the old records. Every cached file is
digest-checked and a mismatch is a miss. The `cli/README.md` build-cache section
lists the rules as users see them.

## Environment checks (`semantscript doctor`)

`python -m semantscript_trainer.cli doctor` runs the Python half of
`semantscript doctor` and prints one line per check (`--json` prints the closed
report, kind `semantscript.doctor-report`, `reportVersion` 1, with `checks` of
`id`, `status`, `summary`, `fix`): the interpreter, the trainer and model
packages, PyTorch and Transformers, the device (CUDA or ROCm with its memory,
else the CPU), ONNX Runtime and ONNX, the platform environment, and the
teacher's file, key and probe. `semantscript_trainer.doctor` imports nothing
heavy itself: PyTorch, Transformers and ONNX Runtime are imported in child
interpreters, and when an import fails (or, outside `--quick`, when a variable
is already set) the import is retried with `PYTHONNOUSERSITE=1` and, on WSL2
with a ROCm build, `HSA_ENABLE_DXG_DETECTION=1`, so the report names a variable
only when it changes the outcome. `probe_teacher(config, mode)` sends one
minimal request (`request`) or, for Ollama, only lists the server's models
(`free`), and returns the latency and tokens with the key scrubbed from any
error. Options: `--teacher`, `--default-teacher-model`, `--no-teacher`,
`--probe request|free|none`, `--device`, `--quick`, `--json`; exit 1 when a
check fails. The Node CLI's `train` runs it with `--probe free --quick` before
every training run.

## Teacher backends

`Teacher.generate(ir, n)` is the only case-source contract used by the generator.
Both provider implementations use a one-case JSON Schema derived from the IR and
then independently parse and validate every response. Server-side structured
output is never treated as the validation boundary. Prompts, schemas, backend
options, and model names are deterministic inputs to the secret-free teacher
configuration digest used by later cache/provenance work.

The built-in providers account for response bytes as each direct or batch result
is decoded and stop before retaining more than 64 MiB of aggregate response JSON.
A custom `Teacher` is trusted in-process code and must enforce an equivalent bound
before returning; a caller cannot prevent arbitrary allocations made inside an
untrusted Python implementation before control returns.

Select a provider without changing code by loading a closed TOML table and passing
the resulting config to `create_teacher`:

```toml
[teacher]
backend = "anthropic"
model = "claude-sonnet-5"
mode = "auto"
batch_threshold = 32
max_tokens = 2048
```

Anthropic uses the official Python SDK's Messages structured-output API. `direct`
mode sends one request per case; `batch` uses Message Batches; `auto` selects the
batch path at `batch_threshold`. Batch results are reconciled by `custom_id`, not
response order, and missing, duplicate, failed, canceled, expired, refused, or
truncated items fail the generation rather than producing a partial dataset.

The local backend uses the official OpenAI Python client against Ollama's
OpenAI-compatible endpoint:

```toml
[teacher]
backend = "ollama"
model = "qwen3:14b-q4_K_M"
base_url = "http://localhost:11434/v1"
seed = 1
```

Record both the Ollama model tag and its resolved digest from `/api/tags` in
training provenance; a tag can be updated to point at different model content.
The OpenAI client requires a nonempty API-key value, but a local Ollama server
ignores it; the backend supplies a local placeholder unless one is configured.

The built-in constraints backend (`semantscript_trainer.teachers.constraints`)
needs no model: when an expression's constraints admit exactly one output for
every input, `ConstraintsTeacher` samples inputs from the IR types with
threshold-aware numeric ranges, labels them with that output and builds
boundary pairs and counterfactual twins by single-field edits. Select it with
`--teacher constraints` or a table (optional `[teacher.ranges]` overrides and an
optional `[teacher.fallback]` language-model table for the inputs the
constraints leave open):

```toml
[teacher]
backend = "constraints"
seed = 1

[teacher.fallback]
backend = "ollama"
model = "qwen3:14b"
```

`load_teacher_config` returns a `ConstraintsTeacherConfig` for it and
`create_teacher` a `ConstraintsTeacher`; its descriptor is provider
`constraints` (`constraints+<fallback backend>` in mixed mode) with the
sampling configuration digest. Without a fallback, an input the constraints do
not decide fails generation with that input and the outputs it admits.
Mixed mode is covered by unit tests with a fake fallback and by one passing
local run (`qwen3:14b` with `--counterfactual-ratio 0`); with counterfactual
twins on, local fallbacks have failed to propose valid twins, and the error
names the expression and the fallback (see `docs/teachers.md`).
`ConstraintsTeacher.sample_decided(ir, n, stream=..., exclude=...)` draws a
labelled held-out set from a stream the training corpus never uses. See
`docs/teachers.md` for the sampling rules and mixed mode.

### WSL-native Ollama with an AMD GPU

Run Ollama directly in WSL when ROCm exposes the GPU through `/dev/dxg`. Ollama's
Linux installer can miss AMD-on-WSL GPU detection, so install matching current
base and ROCm bundles, then set `HSA_ENABLE_DXG_DETECTION=1` on the Ollama service.
Do not mix an older Ollama executable with a current ROCm bundle. Verify startup
logs report `library=ROCm` and the expected `gfx` target rather than `library=cpu`.

For the existing WSL service on this development machine, install both bundles
to the same root used by its `ExecStart` (`/usr/local`), then add the DXG setting:

```bash
curl -fsSLo /tmp/ollama-linux-amd64.tar.zst \
  https://ollama.com/download/ollama-linux-amd64.tar.zst
curl -fsSLo /tmp/ollama-linux-amd64-rocm.tar.zst \
  https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst
sudo systemctl stop ollama
sudo tar --zstd -xf /tmp/ollama-linux-amd64.tar.zst -C /usr/local
sudo tar --zstd -xf /tmp/ollama-linux-amd64-rocm.tar.zst -C /usr/local
sudo systemctl edit ollama
```

Add the following override when the editor opens:

```ini
[Service]
Environment="HSA_ENABLE_DXG_DETECTION=1"
```

Then restart and inspect discovery before sending training traffic:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
journalctl -u ollama -n 100 --no-pager
```

For a new service installed under `/usr`, use the paths in Ollama's official
manual-install instructions instead. The two archives must still come from the
same Ollama release.

The default `http://localhost:11434/v1` endpoint then remains correct without a
Windows gateway or firewall exception. The RX 9070 XT live verification used
Ollama 0.34.3, ROCm `gfx1201`, and the existing `glm-4.7-flash:latest` model. The
OpenAI-compatible request sets `reasoning_effort="none"`, preventing thinking
tokens from exhausting the bounded structured-output response.

Windows-native Ollama is still supported. With WSL mirrored networking,
`localhost:11434` reaches the Windows service; under WSL2 NAT, configure
`base_url` with the Windows host gateway. Ollama's local API has no
authentication, so do not expose a wildcard bind outside trusted interfaces.

The routine test suite never contacts a provider, spends API credit, or pulls an
Ollama model. A live Ollama smoke test is opt-in through
`SEMANTSCRIPT_OLLAMA_LIVE_URL` and `SEMANTSCRIPT_OLLAMA_LIVE_MODEL`.

References: [Ollama Linux installation](https://docs.ollama.com/linux),
[GPU support](https://docs.ollama.com/gpu),
[OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility), and
[thinking controls](https://docs.ollama.com/capabilities/thinking).
