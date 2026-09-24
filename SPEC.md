# SemantScript Language Surface Specification

Status: v1 language contract

Scope: `sema<T>` source syntax and observable runtime behavior

SemantScript extends ordinary TypeScript with one compile-time primitive: a typed
semantic expression. The expression's behavior is learned during the build and is
executed by a fixed, non-generative output head at runtime.

This document specifies the source contract shared by the compiler, trainer, and
runtime. The versioned IR and application artifact layout are specified separately.
The key words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

## 1. Source files and imports

SemantScript source files use the `*.sem.ts` suffix and otherwise follow TypeScript
syntax and semantics. The primitive and its helper types come from
`@semantscript/core`:

```ts
import {
  sema,
  always,
  never,
  type BoundedInt,
  type BoundedNumber,
  type Ordinal,
} from "@semantscript/core";
```

An implementation MUST resolve `sema`, `always`, and `never` by import identity,
not merely by identifier spelling. An unrelated local variable named `sema` is not
a SemantScript primitive.

## 2. Expression grammar

The two expression forms are:

```text
SemaExpression :=
    sema < OutputType > TemplateLiteral
  | sema < OutputType > ( SemaOptions ) TemplateLiteral

DiagnosticSemaExpression :=
    sema.withConfidence < OutputType > TemplateLiteral
  | sema.withConfidence < OutputType > ( SemaOptions ) TemplateLiteral
```

These forms are valid TypeScript: the configured form calls a generic function
that returns a template tag. `sema<T>` returns `T` directly.
`sema.withConfidence<T>` returns the diagnostic result described in section 8.

The public package MUST provide declarations equivalent to these overloads:

```ts
interface SemaTag {
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): T;
  <T>(options: SemaOptions<T>): ConfiguredSemaTag<T>;
  readonly withConfidence: DiagnosticSemaTag;
}

interface ConfiguredSemaTag<T> {
  (strings: TemplateStringsArray, ...inputs: unknown[]): T;
}

interface DiagnosticSemaTag {
  <T>(strings: TemplateStringsArray, ...inputs: unknown[]): SemaResult<T>;
  <T>(options: SemaOptions<T>): ConfiguredDiagnosticSemaTag<T>;
}

interface ConfiguredDiagnosticSemaTag<T> {
  (strings: TemplateStringsArray, ...inputs: unknown[]): SemaResult<T>;
}

declare const sema: SemaTag;

declare function always<T>(
  predicate: () => boolean,
  requiredOutput: T,
): SemaConstraint<T>;

declare function never<T>(
  predicate: () => boolean,
  forbiddenOutput: T,
): SemaConstraint<T>;
```

Standard TypeScript types cannot express the relationship between interpolation
names and `SemaExample.inputs`; the SemantScript compiler performs that additional
static check.

The canonical unconfigured form is:

```ts
type RefundDecision = "approve" | "deny" | "review";

const decision = sema<RefundDecision>`
  Apply our refund policy. Enterprise customers get 60 days;
  everyone else gets 30. Suspicious circumstances go to review.

  Customer: ${customer}
  Order: ${order}
`;
```

The static text is the **behavioral specification**. It is consumed at build time
and MUST NOT be required by the runtime artifact. A `sema` expression is therefore
a compile target, not a runtime prompt.

### 2.1 Interpolated inputs

Every `${identifier}` is an explicit input to the expression. In v1:

- the interpolation MUST be a bare TypeScript identifier;
- every identifier within one template MUST be unique; and
- the identifier spelling is its input name in examples, constraints, and IR.

Property access, calls, operators, and other arbitrary expressions MUST first be
assigned to a named `const`:

```ts
const accountAgeDays = differenceInDays(now, customer.createdAt);

const risk = sema<"low" | "high">`
  Assess account risk.
  Customer: ${customer}
  Account age in days: ${accountAgeDays}
`;
```

This rule gives inputs stable names and makes data-flow and dependency analysis
unambiguous.

### 2.2 Supported input values

A conforming v1 implementation MUST accept an interpolated string, boolean, finite
number, null, supported enum value, readonly array or tuple of supported input
values, or plain object whose own enumerable properties recursively contain
supported input values. Object property order is not semantically significant.
Plain-object property names may be any Unicode scalar string, including the empty
string; only top-level interpolation names use the identifier rule from section
2.1.

`undefined`, `bigint`, `symbol`, functions, promises, cyclic objects, class
instances, and host resources such as file handles MUST be rejected. A later
version may define additional first-class media and domain types.

The compiler rejects statically unsupported input types. Immediately before model
execution, the runtime MUST validate dynamic values for finiteness, cycles, class
instances, promises, and other unsupported values. It MUST throw `SemaInputError`
before inference when a value violates the resolved input schema.

The IR specification defines the canonical, deterministic, type-aware encoding of
these values. Two structurally equal inputs MUST produce identical model input
regardless of JavaScript object insertion order.

## 3. Input isolation

A `sema` expression can observe only its interpolated inputs. It MUST NOT read or
mutate ambient state, including:

- databases or caches;
- the network or HTTP request context;
- the filesystem;
- environment variables;
- clocks, randomness, or process state; or
- module or global mutable state.

Ordinary TypeScript may use libraries and external state before or after a `sema`
expression. The resulting value must be interpolated explicitly if learned
behavior needs it:

```ts
const customer = await db.customers.get(customerId);
const order = await payments.getOrder(orderId);

const decision = sema<RefundDecision>`
  Apply the refund policy.
  Customer: ${customer}
  Order: ${order}
`;
```

The compiler MUST derive the complete runtime input schema from the interpolations
and their resolved TypeScript types. It MUST NOT infer hidden dependencies from
the surrounding lexical scope or from prose in the template.

## 4. Definition options

Examples and deterministic constraints use the configured-tag form. The options
argument MUST be an inline object literal containing only `examples` and/or
`constraints`. Spreads, computed properties, methods, accessors, calls that produce
the options object, and references to an options object are forbidden. Values
inside an example may use the static `const` references defined below.

```ts
const decision = sema<RefundDecision>({
  examples: [
    {
      inputs: {
        customer: enterpriseExample,
        order: orderAt45Days,
      },
      output: "approve",
    },
  ],
  constraints: [
    never(() => order.status === "fraudulent", "approve"),
    always(() => order.ageDays > 90, "deny"),
  ],
})`
  Apply the refund policy.
  Customer: ${customer}
  Order: ${order}
`;
```

Conceptually, the options type is:

```ts
interface SemaOptions<T> {
  readonly examples?: readonly SemaExample<T>[];
  readonly constraints?: readonly SemaConstraint<unknown>[];
}

interface SemaExample<T> {
  readonly inputs: Readonly<Record<string, unknown>>;
  readonly output: T;
}

declare const semaConstraintKind: unique symbol;

interface SemaConstraint<T> {
  readonly [semaConstraintKind]: T;
}
```

The public options container erases the constraint's phantom output parameter to
avoid recursive contextual inference in TypeScript. The SemantScript compiler
still resolves every constraint output against the site's exact `T` and rejects a
non-assignable value; the erasure does not weaken the language rule.

### 4.1 Examples

An example is a gold input/output case used unchanged in training and verification.

- `inputs` MUST have exactly one property for every interpolated input name.
- Every value and the output MUST be assignable to the corresponding source type
  and compile-time serializable.
- Duplicate inputs with different outputs are a compile error.
- A conforming build MUST verify every example; a miss fails verification.

Examples guide learned behavior. They do not add runtime branches or overrides.

A compile-time-serializable expression is recursively one of: a string, boolean,
finite-number, or null literal; a constant enum member; an array or object literal
containing only compile-time-serializable expressions; or a reference to a `const`
in the same source file whose initializer is compile-time serializable. Property
shorthand is allowed when its referenced `const` meets the same rule. Calls,
getters, spreads, computed property names, imports, environment reads, and arbitrary
module execution are forbidden. The compiler MUST analyze this syntax; it MUST NOT
execute the source module to obtain fixture values.

### 4.2 Deterministic constraints

The two constraint constructors are:

```ts
always(predicate, requiredOutput)
never(predicate, forbiddenOutput)
```

`always(p, value)` requires the output to equal `value` whenever `p` is true.
`never(p, value)` requires the output not to equal `value` whenever `p` is true.

In v1, a predicate MUST be a synchronous, zero-argument arrow function with an
expression body. Its free identifiers MUST be a subset of the template's
interpolated identifiers. The supported pure expression subset contains literals,
property and element access, parentheses, comparisons, boolean operators, and
finite-number arithmetic. Calls, assignment, mutation, `await`, `new`, and access
to non-input state are forbidden.

Constraint values MUST be valid scalar outputs. Constraints on a complete flat
interface are reserved for a later version. Statically provable overlap between
contradictory `always` rules, or between an `always` rule and a matching `never`
rule, is a compile error. Other conflicts discovered during generated or
adversarial verification fail the build; the compiler is not required to decide
arbitrary predicate equivalence or overlap.

Literal arithmetic that is statically non-finite is a compile error. If evaluating
a constraint over a generated, example, or verification case produces a non-finite
intermediate, verification fails with the constraint and case identified.

Constraints are hard **build contracts**: they drive boundary and adversarial case
generation, and any observed violation fails verification unless the build records
an explicit violation-rate tolerance. They are not a proof
that an unseen runtime input can never be misclassified. Rules that must hold for
every production input MUST remain deterministic TypeScript guards.

## 5. Output types

Every output has finite, compile-time-known support and maps to one or more fixed
heads. The runtime cannot emit a value outside that support.

### 5.1 Nominal scalar outputs

The nominal output kinds are:

| TypeScript type | Support and order |
| --- | --- |
| `boolean` | `[false, true]` |
| union of two or more string literals | members in Unicode code-point lexical order |
| string enum | two or more distinct constant member values in declaration order |
| numeric enum | two or more distinct constant member values in declaration order |

A plain string-literal union is always nominal. TypeScript does not preserve a
meaningful union-member order, so source spelling MUST NOT imply ordinal semantics.
Heterogeneous enums, computed enum members, and duplicate enum values are invalid.

### 5.2 Ordinal scalar outputs

Ordinal support has a meaningful order and uses an ordinal-aware head and training
objective. It is distinct from a nominal union.

The public marker types MUST preserve their parameters for compiler inspection
while allowing ordinary literal values at the TypeScript boundary. Declarations
equivalent to the following satisfy that contract:

```ts
declare const ordinalKind: unique symbol;
declare const boundedIntKind: unique symbol;
declare const boundedNumberKind: unique symbol;

type Ordinal<Values extends readonly [string, string, ...string[]]> =
  Values[number] & { readonly [ordinalKind]?: Values };

type BoundedInt<Min extends number, Max extends number> =
  number & { readonly [boundedIntKind]?: readonly [Min, Max] };

type BoundedNumber<
  Min extends number,
  Max extends number,
  Step extends number,
> = number & {
  readonly [boundedNumberKind]?: readonly [Min, Max, Step];
};
```

The compiler MUST recognize these types by their exported symbol identity from
`@semantscript/core`, following normal TypeScript aliases and re-exports. A local
lookalike with the same name or structure is not a SemantScript output marker.

An ordered string output uses a nonempty tuple with at least two distinct values:

```ts
type Severity = Ordinal<["low", "medium", "high", "critical"]>;
```

Tuple order is semantic. At the TypeScript boundary, a value of `Severity` is one
of the tuple's string literals.

Bounded numeric outputs use inclusive, finite grids:

```ts
type Stars = BoundedInt<1, 5>;
type ProbabilityPercent = BoundedNumber<0, 100, 0.5>;
```

`BoundedInt<Min, Max>` requires integer literal bounds with `Min < Max` and has
ascending support from `Min` through `Max` in steps of one.
`BoundedNumber<Min, Max, Step>` requires finite numeric literals, `Min < Max`,
`Step > 0`, and an integral `(Max - Min) / Step`; its support is the resulting
ascending finite grid. Both are ordinal. Their brands exist only at compile time;
runtime values are JavaScript numbers.

The compiler MUST interpret the source decimal lexemes for `Min`, `Max`, and `Step`
as exact decimal rational numbers, test divisibility exactly, and enumerate support
as `Min + i * Step` by integer index rather than repeated floating-point addition.
Each support member is then converted once to an IEEE 754 JavaScript number. A grid
is invalid if conversion produces a non-finite value or makes two members equal.
The IR MUST retain the canonical decimal spelling so the TypeScript compiler,
Python trainer, and Node runtime reconstruct the same ordered support.

Bare `number`, numeric-literal unions, unbounded ranges, `NaN`, and infinities are
not v1 output types.

### 5.3 Flat interface outputs

A flat interface represents multiple scalar heads evaluated together:

```ts
interface Triage {
  urgency: Ordinal<["low", "medium", "high"]>;
  route: "self-service" | "agent";
  abusive: boolean;
  score: BoundedInt<0, 10>;
}
```

Every property MUST be required and have one of the supported scalar output kinds.
`readonly` properties are permitted. Optional properties, index signatures,
methods, call signatures, nested interfaces, object unions, arrays, tuples,
`null`, and `undefined` are forbidden as outputs.

String-literal property names, including the empty string, are permitted when they
are valid TypeScript interface properties. The IR preserves the exact property
name and applies JSON Pointer escaping only where the artifact format requires it.

The v1 language reserves and specifies flat interfaces even if a pre-v1 compiler
ships scalar heads first. A conforming v1 implementation MUST implement them as
independent per-field heads, not token generation or JSON decoding.

### 5.4 Explicitly unsupported outputs

Free-text `string` is out of scope for v1 because it requires generation. The
following are also unsupported: bare `number`, `bigint`, `symbol`, `any`, `unknown`,
`never`, `void`, functions, promises, arrays, tuples, nested objects, optional or
nullable values, a single string literal, and unions containing unsupported
members.

Free-text **inputs** remain valid; the prohibition applies only to outputs.

## 6. Confidence requirements

A template MAY begin with an `@confidence` header:

```ts
const decision = sema<RefundDecision>`
  @confidence(0.999)
  Apply the refund policy.
  Customer: ${customer}
  Order: ${order}
`;
```

`@confidence(q)` MUST be the first nonblank line, MUST appear at most once, and
requires a finite decimal `q` in the closed interval `[0, 1]`. It is compiler
syntax and is removed from the behavioral specification before training.

For a plain `sema<T>` expression, an observed confidence greater than or equal to
`q` passes. Below `q`, the runtime MUST invoke the fallback configured for that
compiled expression. If no fallback is configured, it MUST throw
`SemaConfidenceError`; it MUST NOT silently return the low-confidence value.
A fallback registration API is runtime configuration and is outside this language
surface. A fallback MUST return a value assignable to `T`; otherwise the runtime
throws a type-contract error.

Semantically, a registered fallback is a synchronous function receiving the named
inputs, the diagnostic result, and required threshold, and returning `T`. The
artifact's stable expression ID associates that function with the source site.
The runtime and artifact specifications own the concrete registration API and ID
format.

For `sema.withConfidence<T>`, the runtime always returns the diagnostic result,
even below `q`, so the caller can implement a local fallback. The result still
records the calibrated scalar confidence, or per-field confidences, used by the
threshold policy.

## 7. Probability semantics

Each scalar head produces a probability distribution over its finite support.
The build MUST fit one temperature per scalar head on held-out calibration data
before an artifact can pass verification. Calibration remains scoped to the
compiled `NeuralFunction` for a source site: a scalar function has one temperature,
while a flat-interface function has one temperature keyed by each field. All values
below refer to a calibrated scalar-head distribution `p`, not raw logits.

For support size `K`:

- `value` is the support value with the greatest calibrated probability; ties
  select the earlier value in the stable support order;
- `confidence` is `max(p)`, the calibrated top-1 probability; and
- `uncertainty` is normalized entropy:
  `-sum(p[i] * ln(p[i])) / ln(K)`.

The entropy calculation uses the continuous-limit convention `0 * ln(0) = 0`.

`confidence` and `uncertainty` are numbers in `[0, 1]`. The degenerate `K = 1`
convention for uncertainty is zero, although v1 output declarations require at
least two support values.

Confidence and uncertainty measure different things: confidence is probability
assigned to the selected value; uncertainty measures how dispersed the complete
distribution is. Neither is a correctness guarantee.

## 8. `withConfidence` results

The confidence-aware tag has the same behavioral text, interpolation, option, and
output-type rules as `sema<T>`:

```ts
const result = sema.withConfidence<Severity>`
  Assess the incident severity.
  Incident: ${incident}
`;
```

For a scalar output, the result shape is:

```ts
interface DistributionEntry<T> {
  readonly value: T;
  readonly probability: number;
}

interface ScalarSemaResult<T> {
  readonly value: T;
  readonly confidence: number;
  readonly uncertainty: number;
  readonly distribution: readonly DistributionEntry<T>[];
  readonly expectedValue: number | null;
}
```

`distribution` contains every support value exactly once in the stable order from
section 5, and its probabilities sum to one within documented floating-point
tolerance.

For nominal outputs, including booleans and enums, `expectedValue` is `null`.
For `BoundedInt` and `BoundedNumber`, it is the probability-weighted numeric value.
For an ordered string output, it is the probability-weighted zero-based tuple rank:
`0` for the first member through `K - 1` for the last. This rank is metadata, not a
member of the output type.

A plain ordinal `sema<T>` still returns only its selected `T`. Its expected value
and distribution are available through `sema.withConfidence<T>`.

For a flat interface, the result shape is:

```ts
interface ObjectSemaResult<T extends object> {
  readonly value: T;
  readonly minimumFieldConfidence: number;
  readonly maximumFieldUncertainty: number;
  readonly fields: {
    readonly [K in keyof T]: ScalarSemaResult<T[K]>;
  };
}

type SemaResult<T> = [T] extends [boolean | string | number]
  ? ScalarSemaResult<T>
  : [T] extends [object]
    ? ObjectSemaResult<T>
    : never;
```

Each `fields[key]` entry contains that head's full scalar diagnostics. The
`minimumFieldConfidence` and `maximumFieldUncertainty` values are conservative
aggregates, not a calibrated joint probability or joint entropy. Consequently,
`@confidence(q)` on a flat output passes only when every field's calibrated top-1
`confidence` is at least `q`. No joint distribution or joint expected value is
implied.

## 9. Runtime behavior

After compilation, a plain `sema<T>` expression behaves as a typed expression of
`T`; no prompt text, teacher model, token generation, JSON parser, network access,
or ambient input discovery occurs at runtime. A confidence-aware expression has
the result shape in section 8. Both forms execute the same compiled head and differ
only in returned diagnostics and threshold handling.

Evaluation and fallback execution are synchronous from the TypeScript program's
perspective. A runtime target that cannot provide synchronous inference for a site
MUST reject that target during the build; it MUST NOT silently change `T` into
`Promise<T>`.

The natural-language specification, examples, and constraint source are build
inputs. The artifact MAY retain provenance hashes and verification statistics, but
the runtime MUST NOT reinterpret those sources to decide an output.

## 10. Compile-time errors

A conforming compiler MUST report the source location and reject at least:

- a missing, unsupported, or unresolved output type;
- an unsupported interpolation expression or input type;
- duplicate interpolation identifiers;
- malformed or misplaced `@confidence` headers;
- examples with missing, extra, incorrectly typed, or non-serializable values;
- impure, ambient, contradictory, or incorrectly typed constraints;
- an unsupported output member or flat-interface shape; and
- an empty behavioral specification after removing directives and whitespace.

Diagnostics SHOULD name the rejected construct, its resolved TypeScript type, and
the nearest supported alternative.

## 11. Complete example

```ts
import {
  sema,
  type Ordinal,
} from "@semantscript/core";

type RefundDecision = "approve" | "deny" | "review";
type RefundUrgency = Ordinal<["routine", "soon", "urgent"]>;

interface Customer {
  tier: "standard" | "enterprise";
  priorRefunds: number;
}

interface Order {
  ageDays: number;
  status: "paid" | "fraudulent";
  total: number;
}

const enterpriseExample: Customer = {
  tier: "enterprise",
  priorRefunds: 0,
};

const enterpriseOrderAt45Days: Order = {
  ageDays: 45,
  status: "paid",
  total: 129,
};

interface RefundAssessment {
  decision: RefundDecision;
  urgency: RefundUrgency;
  suspicious: boolean;
}

function evaluateRefund(
  customer: Customer,
  order: Order,
): RefundAssessment {
  if (order.ageDays > 90) {
    return {
      decision: "deny",
      urgency: "routine",
      suspicious: false,
    };
  }

  // The application runtime registers a fallback for this expression. Without
  // one, a result below the declared threshold throws SemaConfidenceError.
  return sema<RefundAssessment>({
    examples: [
      {
        inputs: {
          customer: enterpriseExample,
          order: enterpriseOrderAt45Days,
        },
        output: {
          decision: "approve",
          urgency: "soon",
          suspicious: false,
        },
      },
    ],
    // v1 constraints target scalar outputs, so hard object-wide rules stay in
    // ordinary TypeScript guards such as the age check above.
  })`
    @confidence(0.99)
    Apply the refund policy. Prefer review over denial when the circumstances
    are ambiguous or suspicious.

    Customer: ${customer}
    Order: ${order}
  `;
}
```

The ordinary function owns deterministic preprocessing and hard invariants. The
`sema` expression owns the semantic judgment, and its only learned inputs are the
values visible in `${customer}` and `${order}`.
