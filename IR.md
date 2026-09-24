# SemantScript IR and Application Artifact v1

This document defines the build-time `NeuralFunction` intermediate representation,
the deployable application artifact, and the canonical runtime input encoding for
SemantScript v1. The machine-readable schemas are:

- [`schemas/neural-function.v1.schema.json`](schemas/neural-function.v1.schema.json)
- [`schemas/ir-bundle.v1.schema.json`](schemas/ir-bundle.v1.schema.json)
- [`schemas/application-artifact.v1.schema.json`](schemas/application-artifact.v1.schema.json)
- [`schemas/artifact-pointer.v1.schema.json`](schemas/artifact-pointer.v1.schema.json)

`NeuralFunction` IR is the contract between the compiler and trainer. The
application artifact manifest is the contract between the trainer/exporter and
runtime. A runtime MUST NOT require source text, examples, constraints, a teacher
model, or a network connection.

The schemas use JSON Schema Draft 2020-12. All files MUST be UTF-8 without a byte
order mark. Unknown properties are rejected unless a schema explicitly permits
them.

## 1. Versioning

The root `bundleVersion` and `artifactVersion` fields, and each neural function's
`irVersion` field, are integer major versions. A consumer MUST reject an
unsupported major version. Additions that old consumers can safely ignore require
a new schema and capability, because v1 records and envelopes are closed to
unknown properties.

The artifact separately declares runtime and model ABI versions, the canonical
input encoding, the minimum runtime release, and required named capabilities. A
runtime MUST check all of them before opening model resources.

## 2. NeuralFunction IR

One IR record represents one compiled `sema<T>` source expression. It contains:

- `stage`: `source`, `trained`, or `verified`;
- `id`: the stable source-expression identity;
- `semanticSha256`: a digest of the expression's semantic projection;
- `source`: the diagnostic source location and source-file digest;
- `definition`: the ordered template parts, examples, and portable constraint AST;
- `inputs`: ordered, recursively typed interpolation descriptors;
- `output`: one scalar head or a flat object of scalar heads;
- `model`: logical encoder, adapter, and head references, plus an optional
  `encoderDepth` (the shared-encoder layers the function's domain runs before
  its adapter; absent means the full stack, and the encoder reference then
  names the exported prefix graph);
- `runtime`: result mode, confidence threshold, fallback reference, and the v1
  synchronous-execution requirement;
- `trainingProvenance`: teacher, base model, dataset counts and digest, seed,
  trainer revision, and completion time; and
- `verification`: status and held-out accuracy, ECE, Brier score, example,
  constraint, and type-contract results.

The lifecycle makes the same record usable at every build boundary. A compiler
emits `stage: "source"` with `{ "status": "pending" }` for both training and
verification. Successful training advances it to `trained`, replaces training
provenance with a `complete` record, and leaves verification pending. Verification
advances it to `verified` and records `passed` or `failed` metrics. Only a verified,
passing record can be exported to a deployable manifest.

The `definition.template` array preserves text and interpolation boundaries. It is
not a pre-rendered string, so the compiler and trainer cannot disagree about which
characters came from source text and which value supplied an input.

Input types recursively support `string`, `boolean`, finite `number`, `null`,
literals, unions, string or numeric enums, arrays, fixed tuples, and plain objects
with required or optional fields. Union variants are sorted by their semantic JSON
bytes and deduplicated by the compiler. Index signatures and optional or rest tuple
elements are unsupported in v1 and produce a compile-time diagnostic; empty fixed
tuples are supported. Output heads follow the finite-support rules in `SPEC.md`:
boolean, nominal string/number, ordered string, or an exact decimal grid for
bounded numeric output. Object outputs are flat and assign one scalar head to each
field.

Logical model references are resolved only through an artifact manifest. They are
not paths.

### 2.1 Identity and digests

All SHA-256 fields contain 64 lowercase hexadecimal characters. A file digest is
SHA-256 over the exact file bytes. Embedded semantic values use the
`semantscript.semantic-json/v1` encoding below so signed zero and other JSON-number
edge cases cannot be normalized away.

Semantic JSON recursively maps a JSON value to the same typed nodes used by
canonical input encoding: null, boolean, string, and binary64 number nodes; array
nodes preserving order; and object nodes containing UTF-8-key-sorted name/value
pairs. It wraps the result as
`["semantscript-semantic-json",1,typedNode]` and applies the exact JSON byte rules
from section 3. The parser MUST reject duplicate object keys before this transform.
It MUST parse each JSON numeric token directly to binary64 with round-to-nearest,
ties-to-even, preserve the sign of lexical `-0`, and reject overflow or a
non-finite result before schema validation or hashing.

For semantic identity, the compiler constructs this projection:

```json
{
  "irVersion": 1,
  "definition": "<definition object>",
  "inputs": "<inputs array>",
  "output": "<output object>",
  "runtime": "<runtime object>"
}
```

It stores the SHA-256 of the projection's semantic JSON bytes in
`semanticSha256`. Source coordinates, model references, training configuration,
training provenance, and verification results are excluded. Consequently,
retraining an unchanged expression does not change semantic identity, while
changing learned or observable behavior does.

`inputSchemaSha256` is SHA-256 of semantic JSON for the IR `inputs` array.
`outputSchemaSha256` is SHA-256 of semantic JSON for the IR `output` object. The
manifest embeds the exact `inputs` value as well as its digest so a runtime can
validate and serialize without loading build-time IR.

The stable function ID is `nf_` followed by SHA-256 of the semantic JSON encoding
of:

```json
[
  "semantscript-function-id",
  1,
  "<normalized-source-path>",
  3,
  "<semanticSha256>"
]
```

The integer is the zero-based duplicate ordinal among earlier `sema` sites in that
file with the same `semanticSha256`, not the ordinal among all sites. Normalized
source paths use `/`, are relative to the compiler project root, contain no `.` or
`..` segments, and are case-preserving. Inserting or removing an unrelated
expression therefore does not change an existing function ID, while textually
identical expressions at distinct sites remain separate. Moving a site to another
file or inserting an identical earlier site may change its ID.

A training cache key MUST additionally include the teacher configuration, base
model weights, trainer version, data-generation configuration, seed, and relevant
compiler configuration. It MUST NOT use `semanticSha256` alone.

### 2.2 Execution-plan bundle

The compiler emits a versioned `semantscript.ir-bundle` envelope containing the
ordered `functions` array and an `executionPlan`. Every function appears in
exactly one plan stage. A dependency identifies its producer and consumer function
IDs and the consumer input name carrying the value. Independent functions share
stage zero; every dependent function is assigned the smallest integer greater
than all of its direct producers' stages. Stages are dense, ordered, and contain
function IDs in the same canonical source order as `functions`.

Dependencies are a conservative may-analysis of potential data flow and can
contain false-positive edges. A stage is a topological stratum, not a promise that
all of its functions are live or may be executed eagerly together. In particular,
source branches, loops, and separate call frames remain runtime concerns. The v1
compiler follows symbol identity through initialized variables, aliases,
destructuring, derived expressions, direct assignments, and direct property
writes. A direct write textually after its consumer in the same execution
container is excluded; writes across source files or execution containers remain
conservative potential sources. Mutation hidden behind calls or accessors and
interprocedural return analysis are outside the v1 plan boundary.

A routed bundle additionally records `domains`: one entry per compile-time
domain with its `name`, `adapterRef`, `encoderRef`, `encoderDepth` (`null` for
the full stack) and `functionIds`, and each stage lists the `adapterRefs` it
applies. Every function then belongs to exactly one domain, its `model.adapter`
is the domain's adapter and its `model.encoder` the domain's encoder prefix. An
unrouted bundle omits both fields.

The bundle plan is build metadata and is excluded from each function's semantic
projection. Changing graph topology or routing therefore does not by itself
change an otherwise unchanged function ID or `semanticSha256`.

### 2.3 Semantic validation beyond JSON Schema

JSON Schema validates record shapes. The compiler, trainer, and artifact exporter
MUST also enforce these relational rules:

1. Input names and indices are unique; indices are dense from zero and match
   interpolation order. Template input parts and `inputs` name exactly the same
   values, with no duplicate interpolation identifiers. Union variants have
   unique semantic JSON encodings in ascending byte order.
2. Object field names, output field names, resource references, resource paths,
   function IDs, and head output paths are unique in their respective scopes.
   Empty-string object and output field names are valid. Their IR JSON Pointer is
   `/`, and their manifest output path is the single-segment array `[""]`.
3. Examples contain every input exactly once, contain no additional input, and
   their inputs and output conform to the resolved types.
4. Constraint AST references resolve to declared inputs, all operations type-check,
   and each constraint output belongs to the scalar output support.
5. Enum values are distinct and declaration-ordered. Canonical decimals use fixed
   notation, no leading integer zeroes, no trailing fractional zeroes, and `0`
   rather than `-0`; exponent notation is forbidden. A numeric bounded support is
   strictly ascending and is the exact finite sequence from minimum to maximum by
   a positive step, using decimal rational arithmetic rather than binary
   floating-point iteration. `bounded-int` additionally requires integral minimum
   and maximum and step `1`. Every support value must convert to a distinct finite
   binary64 value.
6. A scalar output has exactly one head at output path `""`. A flat output has
   exactly one head for every field using the single-segment JSON Pointer path for
   that field. The manifest form is `[]` for the scalar path and `[fieldName]` for
   a field; JSON Pointer `~0` and `~1` unescaping is applied during conversion.
   Head types and supports agree with the IR output declaration. Boolean heads use
   `binary-sigmoid`; all other v1 heads use `categorical-softmax`.
7. Every logical reference resolves to exactly one resource of the required role.
   Each manifest function's embedded `inputs` exactly matches its input digest.
   The selected per-function adapter and shared encoder agree with the exported
   head ABI.
8. `resultMode`, threshold, fallback, and threshold policy obey `SPEC.md`. In
   particular, a value-returning expression below a non-null threshold needs a
   configured fallback at runtime or throws `SemaConfidenceError`. A null
   threshold uses policy `none` and a null fallback. A non-null scalar threshold
   uses `scalar-top1`; a non-null flat-output threshold uses `all-fields`. A
   diagnostic call returns calibrated diagnostics even below its configured
   threshold, but the result does not duplicate that artifact policy value and
   never invokes a fallback.
9. Bundle function IDs are unique and canonically source-ordered. Every function
   occurs in exactly one canonically ordered stage; stages are dense from zero and
   use minimum dependency depth. Every dependency names existing producer and
   consumer functions and an existing consumer input, is unique and canonically
   ordered, and points from an earlier stage to a later one.
10. An artifact can be published only from IR with passing verification, no example
    failures, no constraint violations, no output type errors, and complete
    function-level and per-head accuracy, calibration, and pair-consistency metrics.

## 3. Canonical input serialization

`semantscript.canonical-input/v1` turns validated runtime inputs into one
deterministic, type-aware UTF-8 byte sequence. The per-function head is already
bound to the behavioral definition, so source text is not repeated at runtime.

Serialization is schema-directed. The runtime MUST first reject values that do not
match the IR input type, including non-finite numbers, `undefined`, sparse arrays,
wrong tuple lengths, missing required or extra object properties, cyclic objects, class
instances, promises, symbols, functions, own enumerable getters or setters, own
symbol-keyed properties, and enumerable non-index array properties. Strings,
including input and property names, containing unpaired UTF-16 surrogates are
rejected. Validation MUST NOT invoke an accessor.

The encoded value is this JSON array:

```text
["semantscript-input",1,[[inputName,typedValue],...]]
```

Input pairs are ordered by increasing IR input `index`. Typed values use these
forms:

```text
["null"]
["boolean", false]
["boolean", true]
["string", stringValue]
["number", ieee754Hex]
["literal", typedPrimitiveValue]
["enum", enumTypeName, declarationIndex, typedPrimitiveValue]
["union", variantIndex, typedValue]
["array", [typedValue, ...]]
["tuple", [typedValue, ...]]
["object", [[propertyName, typedValue], ...]]
```

Rules:

- `ieee754Hex` is the 16-character lowercase hexadecimal encoding of the
  big-endian IEEE-754 binary64 bits. This preserves `-0`; NaN and infinities are
  invalid.
- A non-JavaScript implementation converts a schema `number` to binary64 exactly
  once using round-to-nearest, ties-to-even, then rejects overflow or a non-finite
  result. It tests booleans before numeric conversion. This rule makes, for
  example, a Python arbitrary-precision integer agree with JavaScript's already
  rounded `number`.
- `enumTypeName` is exactly the IR enum `name`. `declarationIndex` is the zero-based
  position of the matching value in its IR `values` array, and the underlying value
  must equal `values[declarationIndex]` after schema-directed binary64 conversion.
  `typedPrimitiveValue` is the complete `string` or `number` typed node, not raw
  JSON.
- A literal contains the complete primitive typed node. For a union, the compiler
  sorts variants by semantic JSON bytes; the runtime uses the lowest-index variant
  that validates and includes that index in the encoding.
- Array order and tuple order are preserved.
- Object property pairs are sorted by the lexicographic order of their property
  name's UTF-8 bytes. JavaScript insertion order and IR field declaration order do
  not affect the bytes. A missing optional field is omitted; all required fields
  must be present.
- Strings are not Unicode-normalized. Quote MUST use `\"` and reverse solidus
  MUST use `\\`. U+0008, U+0009, U+000A, U+000C, and U+000D MUST use `\b`, `\t`,
  `\n`, `\f`, and `\r`, respectively. Other U+0000 through U+001F characters MUST
  use lowercase `\u00xx`. No other character may be escaped; remaining Unicode
  scalar values are emitted directly as UTF-8, including `/`.
- Envelope versions and enum/union indices use unsigned base-10 JSON integers with
  no leading zeroes.
- JSON structural punctuation has no surrounding whitespace. There is no byte
  order mark and no trailing newline.

For example, both `{b: 2, a: 1, text: "a\n\"é", zero: -0}` and the same object
constructed with another insertion order encode as:

```json
[
  "semantscript-input",
  1,
  [
    [
      "obj",
      [
        "object",
        [
          ["a", ["number", "3ff0000000000000"]],
          ["b", ["number", "4000000000000000"]],
          ["text", ["string", "a\n\"é"]],
          ["zero", ["number", "8000000000000000"]]
        ]
      ]
    ]
  ]
]
```

The SHA-256 digest of those exact UTF-8 bytes is
`3273c3e7b704fce97a131fbdf1c3cd2bef0cd3896c1f4ebbc6857921b534a162`.

Canonical input bytes do not contain a function, schema, or model identity. A
data-generation cache key MUST combine `semanticSha256` with the canonical input
bytes and the complete generator configuration. An inference-cache key MUST
combine function `id`, the active manifest digest, and the canonical input bytes.
This prevents values from different functions or retrained artifacts from sharing
an inference result.

### 3.1 Compact encoding (`semantscript.canonical-input/v2`)

`semantscript.canonical-input/v2` renders the same validated, typed tree as
compact text so that the encoder spends tokens on values rather than on envelope
syntax. Input pairs appear in IR `index` order as `name=value`, separated by one
space. Objects are `{key=value,...}` with the same UTF-8-sorted keys, arrays and
tuples are `[value,...]`, booleans and null are `true`, `false` and `null`,
numbers use ECMAScript `Number` spelling with `-0` preserved, and a string or key
is written bare when it matches `[A-Za-z_][A-Za-z0-9_.-]*` and is not `true`,
`false` or `null`; otherwise it is quoted with the v1 escape rules. The literal,
enum and union tags of v1 are dropped because the schema fixes them per path,
which keeps the encoding injective. An artifact declares the encoding its
functions were trained with in `compatibility.canonicalInput`, and every verified
record repeats it in `trainingProvenance.canonicalInput`; a runtime MUST use
exactly that encoding for the artifact. The golden vectors are in
`examples/serialization/canonical-input.v2.json`.

## 4. Application artifact

An application artifact combines one tokenizer, one shared encoder, a default
application adapter, optional additional statically routed adapters, optional
additional encoder prefixes (depth routing: a prefix of the shared encoder
exported as its own graph, which a function selects with `encoderRef`), and one
or more per-function heads. Each function explicitly selects its `adapterRef`. The
release is immutable and content-addressed:

```text
<artifact-root>/
  current.json                         # optional atomic deployment pointer
  releases/
    sha256-<manifest-sha256>/
      manifest.json
      tokenizer/
        tokenizer.json
      models/
        encoder/
          model.onnx
        adapters/
          application.onnx
        heads/
          <function-id>/
            head-000.onnx
            head-001.onnx              # flat outputs only
```

The release directory suffix is SHA-256 of the exact `manifest.json` bytes. The
manifest stores the path, length, digest, role, format, version, and ABI of every
resource. v1 ONNX resources are single files (`externalData: false`) and require no
custom operator libraries. Function IDs, not source names or raw output field
names, form filesystem paths.

For every function, the manifest includes semantic, input-schema, and
output-schema digests; the complete ordered runtime input descriptors; selected
adapter; result mode and confidence policy; head bindings; passing verification
metrics; and training provenance. Each scalar head carries its own held-out
temperature-scaling record and accuracy/pair-consistency verification:

- fitted temperature greater than zero;
- calibration-set digest and sample count;
- ECE and its bin count; and
- multiclass Brier score.

Thus a scalar function has one calibration record and a flat output has one record
per field. Function-level ECE and Brier score summarize the release gate; they do
not replace per-head calibration.

Temperature scaling divides logits by the one positive fitted temperature before
sigmoid or softmax and minimizes negative log likelihood on the recorded held-out
split. ECE uses the declared number of equal-width bins over calibrated top-1
confidence. The normalized multiclass Brier score for one case is
`sum((p[k] - y[k])^2) / 2`, keeping it in `[0, 1]`. For a scalar function, manifest
accuracy, ECE, and Brier score are that head's held-out metrics. For a flat output,
accuracy is exact whole-object accuracy, while ECE and Brier score are the
unweighted mean of the field-head metrics.

Model ABI v1 uses `binary-sigmoid` for booleans: the head emits shape `[BATCH, 1]`,
the runtime divides that logit by temperature, and sigmoid probability is assigned
to `true` with its complement assigned to `false`. All other heads use
`categorical-softmax`: the head emits one finite logit per support member in stable
support order with shape `[BATCH, K]`, and the runtime applies temperature then
softmax. Cumulative-link and scalar-regression parameterizations are not part of
model ABI v1.

The v1 tokenizer and tensor routing convention is deliberately narrow. The
runtime decodes the canonical-input bytes as UTF-8 and passes that complete string
to the verified Hugging Face `tokenizer.json` with special tokens enabled. Any
normalization, pre-tokenization, or special-token processing needed by the model
MUST be embedded in that file; the runtime does not infer or download tokenizer
configuration. The tokenizer resource's required `maximumSequenceLength` applies
deterministic right truncation after special-token processing; v1 does not pad
because it admits only a batch of one with a dynamic sequence dimension. The
encoder consumes
`input_ids` and `attention_mask` as `int64` tensors shaped `[BATCH, SEQUENCE]` and
produces one `float32` `sentence_embedding` tensor shaped `[BATCH, HIDDEN]`.
The selected adapter consumes `sentence_embedding` and produces one
`float32` `function_embedding` tensor with the same shape. Each selected head
consumes `function_embedding` and produces one `float32` `logits` tensor with
the model-ABI shape above. V1 artifacts using other names, extra model inputs or
outputs, a fixed batch dimension other than one, or an incompatible adjacent
dimension are rejected at load time. Later artifact versions may add explicit edge
mappings without changing this convention retroactively.

Natural-language definitions, examples, constraint source, teacher responses, and
training rows MUST NOT be included in the deployable release. They may exist in a
separate build archive.

### 4.1 Publishing

The exporter writes a new release to a staging directory on the same filesystem,
closes every file, computes and records resource hashes and lengths, writes the
manifest last, validates it, and renames the staging directory to its final
content-addressed name. If a deployment pointer is used, it is written to a new
file and atomically renamed to `current.json` only after the release is complete.
Files and their containing directories are flushed before each rename when the
host exposes a durable flush operation. Published release contents are never
modified in place.

`current.json` validates against `artifact-pointer.v1.schema.json` and contains
`kind`, `pointerVersion`, a portable relative `release` path of the form
`releases/sha256-<digest>`, and `manifestSha256`. The digest suffix and
`manifestSha256` MUST be identical.

### 4.2 Runtime loading

Before inference, a runtime MUST:

1. Parse `current.json`, when used, and the selected release manifest with duplicate
   JSON object keys rejected. Hash the exact manifest bytes and require the selected
   release directory name and deployment-pointer digest to equal that hash.
2. Validate the manifest schema, versions, ABIs, canonical-input identifier,
   minimum runtime release, and all required capabilities.
3. Treat every resource path as relative to the release directory. The schema
   permits only portable ASCII path segments and excludes empty or dot segments,
   colons, control characters, backslashes, and trailing dot/space aliases. Also
   reject symlinks, non-regular files, paths outside the release, and files over
   configured resource and aggregate size quotas.
4. Require unique logical references and paths, then verify every file's exact byte
   length and SHA-256 digest before passing it to the tokenizer or ONNX loader.
   Hash and load through the same safely opened handle or immutable byte buffer to
   avoid a check/use race.
5. Resolve `model` references to resources with roles `tokenizer`, `encoder`, and
   the default `adapter`. Resolve each function's selected adapter and each head
   binding to resources of the declared roles.
6. Check tensor names, dtypes, shapes, opset support, and encoder-to-adapter-to-head
   compatibility before accepting traffic. Inspect the actual ONNX protobuf and
   reject external-data references or unsupported/custom operator domains rather
   than trusting manifest metadata alone.
7. Build an immutable function table keyed by `id`. Only after the complete table
   succeeds may the runtime atomically swap it into service.

At invocation, the runtime finds the function by ID, validates and canonically
serializes inputs, runs tokenizer, encoder, adapter, and the selected head or heads,
applies each head's fitted temperature to its logits, computes the diagnostics in
`SPEC.md`, enforces the confidence policy, and returns the typed value or diagnostic
result. A failure at any loading step leaves the previously active release intact.

Detached signatures over the manifest digest may be required by a deployment
policy. Signatures are outside artifact schema v1; hashes provide integrity but do
not establish publisher authenticity.

## 5. Examples

- [`examples/ir/refund-decision.v1.json`](examples/ir/refund-decision.v1.json) is a
  hand-written build-time IR record.
- [`examples/artifacts/refund-app/manifest.v1.json`](examples/artifacts/refund-app/manifest.v1.json)
  is the corresponding deployable manifest. Its resource metadata is illustrative;
  model files are not included in the example tree.
- [`examples/artifacts/refund-app/current.v1.json`](examples/artifacts/refund-app/current.v1.json)
  is its deployment-pointer example. Like the deliberately omitted illustrative
  model resources, its content-addressed release directory is not materialized in
  the example tree.
- [`examples/serialization/canonical-input.v1.json`](examples/serialization/canonical-input.v1.json)
  records the canonical input golden vector and digest.
