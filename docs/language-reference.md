# Language reference

This page explains the SemantScript source language for someone writing
`.sem.ts` files. [`SPEC.md`](../SPEC.md) is the normative contract; where the
two differ, the spec decides. Every snippet below is taken from a file under
[`docs/examples`](examples/README.md) that the compiler test suite compiles, so
the examples stay runnable.

## The primitive

SemantScript is ordinary TypeScript plus one primitive: a typed semantic
expression whose behavior is learned at build time and executed at runtime by
a fixed, non-generative output head. Source files end in `.sem.ts` and import
the primitive from `@semantscript/core`; the compiler recognizes `sema`,
`always`, `never` and the marker types by import identity, so a local variable
that happens to be called `sema` is not a SemantScript expression.

```ts
import { sema } from "@semantscript/core";

type RefundDecision = "approve" | "deny" | "review";

export function decideRefund(customer: Customer, order: Order): RefundDecision {
  return sema<RefundDecision>`
    Apply our refund policy. Enterprise customers get 60 days; everyone else
    gets 30. Suspicious circumstances go to review.
    Customer: ${customer}
    Order: ${order}
  `;
}
```

([`refund-decision.sem.ts`](examples/refund-decision.sem.ts))

The static text is the behavioral specification: it is consumed during the
build and never shipped. `${customer}` and `${order}` are the expression's only
inputs. The compiler replaces the site with a call into the runtime, so after
`semantscript build` the function above is a plain typed expression of
`RefundDecision`: no prompt, no token generation, no JSON parsing at runtime.

The four expression forms are:

```text
sema<T>`...`
sema<T>({ examples, constraints })`...`
sema.withConfidence<T>`...`
sema.withConfidence<T>({ examples, constraints })`...`
```

`sema<T>` returns `T`. `sema.withConfidence<T>` returns the diagnostic result
described below. The output type argument is mandatory and there is exactly
one; a configured form takes exactly one inline options object. Anything else
(no type argument, two of them, `sema<T>()`, optional chaining) is a compile
error naming the site.

## Inputs

Every `${identifier}` is an input. In v1 an interpolation must be a bare
identifier, each identifier appears at most once per template, and its spelling
is the input's name in examples, constraints and the IR. Property access, calls
and operators go into a named `const` first:

```ts
export function assessRisk(customer: Customer, nowMs: number): "low" | "high" {
  const accountAgeDays = Math.floor(
    (nowMs - Date.parse(customer.createdAt)) / 86_400_000,
  );
  return sema<"low" | "high">`
    Assess account risk.
    Customer: ${customer}
    Account age in days: ${accountAgeDays}
  `;
}
```

([`inputs.sem.ts`](examples/inputs.sem.ts))

Supported input values are strings, booleans, finite numbers, `null`, enum
values, readonly arrays and tuples of supported values, and plain objects whose
own enumerable properties hold supported values. The compiler derives the
runtime input schema from the interpolations' TypeScript types; the runtime
validates every value against that schema before inference and throws
`SemaInputError` for `undefined`, `bigint`, symbols, functions, promises,
cycles, class instances, non-finite numbers, missing or extra properties.
Object key order never matters: two structurally equal inputs produce identical
model input.

A `sema` expression can observe only its inputs. Databases, the network, the
filesystem, environment variables, clocks, randomness and module state are
invisible to it; fetch what the judgment needs in ordinary TypeScript and
interpolate the value.

## Examples and constraints

The configured form takes an inline object literal with `examples` and
`constraints`:

```ts
const enterpriseExample: Customer = { tier: "enterprise", priorRefunds: 0 };
const orderAt45Days: Order = { ageDays: 45, status: "paid", total: 129 };

export function decideRefund(customer: Customer, order: Order): RefundDecision {
  return sema<RefundDecision>({
    examples: [
      {
        inputs: { customer: enterpriseExample, order: orderAt45Days },
        output: "approve",
      },
    ],
    constraints: [
      never(() => order.status === "fraudulent", "approve"),
      always(() => order.ageDays > 90, "deny"),
    ],
  })`
    Apply our refund policy. Enterprise customers get 60 days; everyone else
    gets 30. Suspicious circumstances go to review.
    Customer: ${customer}
    Order: ${order}
  `;
}
```

([`examples-and-constraints.sem.ts`](examples/examples-and-constraints.sem.ts))

An example is a gold input/output case. Its `inputs` name every interpolated
input exactly once, its values and output must be assignable to the source
types and compile-time serializable (literals, `const` references in the same
file, arrays and object literals of those; no calls, spreads, imports or
computed names), and the build verifies every example: a miss fails
verification. Examples guide learning; they never add runtime branches.

A constraint is a build contract. `always(p, value)` requires the output to be
`value` whenever `p` holds; `never(p, value)` forbids it. The predicate is a
zero-argument arrow function over the interpolated inputs using literals,
property and element access, comparisons, boolean operators and finite
arithmetic. Constraints drive boundary and counterfactual case generation, and
any violation observed during verification fails the build unless the build
records an explicit violation-rate tolerance. They are not a runtime guard:
rules that must hold for every production input stay in ordinary TypeScript
(the `if (order.ageDays > 90)` before the expression, not inside it).
Contradictory constraints that the compiler can prove overlap are a compile
error.

## Output types

Every output has finite, compile-time-known support and maps to one or more
fixed heads; the runtime cannot emit a value outside that support.

```ts
enum Channel {
  Email = "email",
  Chat = "chat",
}
enum Priority {
  Low = 1,
  High = 2,
}
type Severity = Ordinal<["low", "medium", "high", "critical"]>;
type Stars = BoundedInt<1, 5>;
type Percent = BoundedNumber<0, 100, 0.5>;

interface Triage {
  urgency: Ordinal<["low", "medium", "high"]>;
  route: "self-service" | "agent";
  abusive: boolean;
  score: BoundedInt<0, 10>;
}

const spam = sema<boolean>`Is this message spam? Message: ${message}`;
const tone = sema<
  "positive" | "negative" | "neutral"
>`Classify the tone. Message: ${message}`;
const channel = sema<Channel>`Which support channel fits best? Message: ${message}`;
const priority = sema<Priority>`How urgently should we answer? Message: ${message}`;
const severity = sema<Severity>`Rate the incident severity. Message: ${message}`;
const stars = sema<Stars>`How many stars would this reviewer give? Message: ${message}`;
const percent = sema<Percent>`How likely is a refund request, as a percentage? Message: ${message}`;
const triage = sema<Triage>`Triage this support message. Message: ${message}`;
```

([`output-types.sem.ts`](examples/output-types.sem.ts))

| Output type                                              | Kind          | Support and order                                                                                        |
| -------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------- |
| `boolean`                                                | nominal       | `[false, true]`                                                                                          |
| union of two or more string literals                     | nominal       | the literals in Unicode code-point order (source spelling implies no order)                              |
| string enum                                              | nominal       | the distinct constant members in declaration order                                                       |
| numeric enum                                             | nominal       | the distinct constant members in declaration order                                                       |
| `Ordinal<["a", "b", ...]>`                               | ordinal       | the tuple, in tuple order; values are the string literals                                                |
| `BoundedInt<Min, Max>`                                   | ordinal       | the integers from `Min` through `Max`                                                                    |
| `BoundedNumber<Min, Max, Step>`                          | ordinal       | `Min + i * Step` on an exact decimal grid; `(Max - Min) / Step` must be integral and every member finite |
| flat interface of required properties of the kinds above | one head each | each property is an independent scalar head; `readonly` is allowed                                       |

Nominal outputs are classified; ordinal outputs use an order-aware head and
training objective and expose an expected value. Heterogeneous or computed
enums, a single string literal, bare `number`, free-text `string`, `any`,
arrays, tuples, nested objects, optional or nullable values and unions with
unsupported members are compile errors. A flat interface may not have optional
properties, index signatures, methods or nested objects; it is implemented as
independent per-field heads, never as JSON decoding
([structured outputs](structured-outputs.md) covers the heads, the per-field
verification and the diagnostic result).

## Confidence

A template may start with an `@confidence(q)` header, `q` in `[0, 1]`:

```ts
export function severityOrFallback(incident: Incident): Severity {
  return sema<Severity>`
    @confidence(0.9)
    Assess the incident severity.
    Incident: ${incident}
  `;
}
```

([`confidence.sem.ts`](examples/confidence.sem.ts))

For a plain `sema<T>`, a calibrated confidence of at least `q` returns the
value. Below `q` the runtime invokes the fallback registered for that
expression when the artifact was loaded (a synchronous function of the named
inputs, the diagnostic result and the threshold, returning a `T`), and throws
`SemaConfidenceError` when none is registered; it never returns the
low-confidence value silently. For a flat interface every field's confidence
must reach `q`. The header is compiler syntax: it is removed from the
behavioral text before training.

## Domains

A template may carry an `@domain(name)` header, alone or beside `@confidence`,
before the behavior text. It names the compile-time routed domain of the
expression: every expression of a domain shares one adapter over the
application's encoder, and the `semantscript build --domain-depth name=n`
option sets how many shared-encoder layers that domain runs before its adapter
(the full stack by default). Without a header the domain is the enclosing
class's name when the expression sits inside a class (a controller), else the
file's name without `.sem.ts`, both lowercased with dashes, so expressions
grouped by controller or by file are grouped by domain. Naming a domain depth or using one `@domain` header makes
the whole bundle routed; otherwise every expression shares the application's
single adapter and the plan is unchanged.

```ts
const risk = sema<"low" | "high">`
  @domain(fraud)
  Whether this order looks fraudulent.
  Order: ${order}
`;
```

The name is lowercase letters, digits and dashes. Routing is build metadata:
it changes neither the function's semantic identity nor its id.

## Diagnostics with `sema.withConfidence`

The diagnostic form has the same text, input, option and output rules and
always returns the calibrated result, even below a threshold, so the caller
owns the decision:

```ts
export function severityWithDiagnostics(incident: Incident) {
  const result = sema.withConfidence<Severity>`
    Assess the incident severity.
    Incident: ${incident}
  `;
  if (result.confidence < 0.9) {
    return {
      value: "medium" as Severity,
      escalate: true,
      expectedRank: result.expectedValue,
    };
  }
  return {
    value: result.value,
    escalate: false,
    expectedRank: result.expectedValue,
  };
}
```

For a scalar output the result is:

```ts
interface ScalarSemaResult<T> {
  readonly value: T; // the support value with the highest calibrated probability
  readonly confidence: number; // that probability, in [0, 1]
  readonly uncertainty: number; // normalized entropy of the distribution, in [0, 1]
  readonly distribution: readonly { value: T; probability: number }[]; // every support value, stable order
  readonly expectedValue: number | null; // null for nominal outputs
}
```

`expectedValue` is the probability-weighted number for `BoundedInt` and
`BoundedNumber`, the probability-weighted zero-based rank for an `Ordinal`, and
`null` for booleans, unions and enums. Ties select the earlier support value.
For a flat interface the result carries `value`, one `ScalarSemaResult` per
field under `fields`, and the conservative aggregates `minimumFieldConfidence`
and `maximumFieldUncertainty`; no joint distribution is implied.

Each head's probabilities are calibrated with one temperature fitted on
held-out data during the build; confidence is calibrated top-1 probability and
is a measurement, not a guarantee of correctness.

## Runtime behavior

Compiled code imports the runtime and calls each expression's artifact-backed
function by its stable id. An application calls `loadSemaArtifact(path)` once at
startup; the load verifies the artifact and activates it atomically, and a
failed reload leaves the previous artifact active. Evaluation is synchronous
from the program's perspective. The runtime throws distinct typed errors: a call
before a successful load (`SemaRuntimeNotLoadedError`), an unknown function
(`SemaUnknownFunctionError`), invalid inputs (`SemaInputError`), inference
failures and timeouts (`SemaInferenceError` and subclasses), a confidence below
threshold with no fallback (`SemaConfidenceError`) and a fallback returning a
value outside the declared type (`SemaFallbackError`). Several expressions over
the same input can run as one stage sharing the encoder pass, and an
expression that interpolates another's result runs one stage later; the
[execution plans](execution-plans.md) page explains the schedule the compiler
emits and the runtime guide the API.

## Compile-time errors

The compiler rejects, with the file, line and column of the site and the site's
text: malformed uses of `sema` (code 9100), unsupported or unresolved output
types (9101 to 9105), invalid or duplicate interpolations and unsupported input
types (9110 to 9112), invalid options, examples, constraints, `@confidence` and `@domain`
headers (9120 to 9125), and an empty behavioral text (9124). The complete
catalogue and a rendered sample of every family are in the
[compiler guide](../compiler/README.md); `semantscript build` prints them and
emits nothing when any error is present.
