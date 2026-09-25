# Structured outputs

A `sema` expression can return an object: a flat interface whose properties are
each one of the scalar output kinds. Nothing about the model generates JSON.
Every property is its own head over the same encoding, trained together,
calibrated separately and verified field by field. This page says which shapes
are allowed, how a field becomes a head, and what the trainer reports per
field.

## Shapes

```ts
interface Triage {
  urgency: Ordinal<["low", "medium", "high"]>;
  route: "self-service" | "agent";
  abusive: boolean;
  score: BoundedInt<0, 10>;
}

const triage = sema<Triage>`Triage this support message. Message: ${message}`;
```

([`output-types.sem.ts`](examples/output-types.sem.ts))

Every property must be required and must have a supported scalar output type:
`boolean`, a union of string literals, a string or numeric enum, `Ordinal<[...]>`,
`BoundedInt<Min, Max>` or `BoundedNumber<Min, Max, Step>`. `readonly` is
allowed, and so is any string-literal property name (the IR keeps the exact
name and only the artifact's JSON Pointers escape `~` and `/`). Optional
properties, index signatures, methods, nested objects, arrays, tuples, `null`,
`undefined` and unions of objects are compile errors, listed in the
[language reference](language-reference.md#compile-time-errors). One level,
scalar leaves: that is the whole rule.

## From fields to heads

The compiler emits an `object` output with one entry per property:

```json
"output": {
  "kind": "object",
  "tsType": "Triage",
  "fields": [
    { "name": "abusive", "head": { "kind": "nominal", "sourceKind": "boolean", "support": [false, true] } },
    { "name": "route", "head": { "kind": "nominal", "sourceKind": "string-union", "support": ["agent", "self-service"] } },
    { "name": "score", "head": { "kind": "ordinal", "sourceKind": "bounded-int", "minimum": "0", "maximum": "10", "step": "1", "supportDecimal": ["0", "1", "…", "10"], "expectedValue": "numeric" } },
    { "name": "urgency", "head": { "kind": "ordinal", "sourceKind": "ordinal-string", "support": ["low", "medium", "high"], "expectedValue": "zero-based-rank" } }
  ]
},
"model": {
  "encoder": "encoder.main",
  "adapter": "adapter.application",
  "heads": [
    { "outputPath": "/abusive", "ref": "head.096c71be….000" },
    { "outputPath": "/route",   "ref": "head.096c71be….001" },
    { "outputPath": "/score",   "ref": "head.096c71be….002" },
    { "outputPath": "/urgency", "ref": "head.096c71be….003" }
  ]
}
```

Each head is addressed by the field's JSON Pointer (`outputPath`), and a
scalar function is the degenerate case with one head at `""`. The trainer
labels every corpus row once per head (`derive_output_heads`), trains all the
heads of a function over one encoder and adapter with the per-field losses
summed (`FieldHeads`), and reports held-out accuracy per field beside the
function's exact-match accuracy. Every head trains with the trainer's proper-scoring objective (log loss plus
half the spherical loss), and an ordinal field adds the normalized ranked
probability score, exactly as its scalar counterpart would.

The artifact publishes one ONNX head per field,
`models/heads/<function id>/head-000.onnx` onward, and the manifest's `heads[]`
entry for each carries the pointer as a path array (`["route"]`), the head's
type and support, its parameterization (`binary-sigmoid` for a boolean,
`categorical-softmax` otherwise) and its own calibration and verification
records. Model ABI v1 is unchanged: each head maps the function embedding to
`[BATCH, K]` logits, with `K = 1` for a boolean.

## What verification reports per field

Verification is per head, and the gate is on the worst one:

| Reported          | Per field (`heads[].outputPath`)                                           | For the function                      |
| ----------------- | -------------------------------------------------------------------------- | ------------------------------------- |
| Temperature       | one positive temperature fitted on the held-out split                      | none; there is no joint distribution  |
| Accuracy          | that field's held-out accuracy                                             | exact match over every field of a row |
| ECE and Brier     | that field's calibration error and normalized Brier score (`eceBins` bins) | the worst field's                     |
| Pair consistency  | anchor and twin both correct on that field                                 | the lowest field's                    |
| Gold examples     | every field of every example must match                                    | `exampleFailures` counts rows         |
| Constraint checks | evaluated on the whole predicted object                                    | `constraintViolations` counts rows    |

The verified IR carries `verification.metrics.heads[]` with one
`HeadVerificationV1` per field, and the manifest carries the same in each
`heads[]` entry. Because the function-level ECE is the worst head's, a function
never passes the gate on an average across fields.

## What the runtime returns

A plain `sema<Triage>` call returns a `Triage` value, each field decoded from
its own calibrated head. `sema.withConfidence<Triage>` returns
`ObjectSemaResult<Triage>`:

```ts
interface ObjectSemaResult<T extends object> {
  readonly value: T;
  readonly minimumFieldConfidence: number;
  readonly maximumFieldUncertainty: number;
  readonly fields: { readonly [K in keyof T]: ScalarSemaResult<T[K]> };
}
```

`fields` holds one scalar diagnostic per property, with its distribution and,
for ordinal fields, its expected value. The two aggregates are the minimum and
maximum over fields, not a joint probability or entropy. A `@confidence`
threshold on an object output uses the `all-fields` policy: every field's
confidence must reach the threshold, otherwise the fallback runs or
`SemaConfidenceError` is thrown, the same as a scalar whose top-1 confidence
falls short. The manifest records the policy (`scalar-top1` or `all-fields`)
with the threshold.

## When to use one

An interface is the right shape when several answers are read from one input
at once and belong together: a triage record, a screening result. It costs the
same encoder pass as one of its fields would and adds a head per field. Prefer
separate expressions when the fields have different inputs (they would not
share an encoding anyway), when one answer should feed another (the
[execution plan](execution-plans.md) handles that as a chain), or when the
fields are verified and released on different schedules, since an interface is
one function with one verification gate.
