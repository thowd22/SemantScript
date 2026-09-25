# SemantScript

[![CI](https://github.com/thowd22/SemantScript/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/thowd22/SemantScript/actions/workflows/ci.yml)

SemantScript adds one primitive to TypeScript: a typed expression whose
implementation is learned from a natural-language specification at build time
and executed at runtime by a small, fixed classifier. Ordinary code handles
persistence, routing, validation and transactions; `sema` expressions handle
what is semantic. No language model runs in production.

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

The type argument is the whole output space: three literals, three logits.
The interpolations are the inputs. The text is consumed by the build and
never shipped. Before `semantscript train` accepts it, the expression also
needs at least one gold example (`sema<T>({ examples: [...] })`), which
verification must reproduce; see [getting started](docs/getting-started.md).

## How it works

1. **Compile.** The compiler (a `tsc` transformer, or an esbuild, Vite or
   Next.js plugin) rewrites each `sema` site to a call by id and emits an IR
   bundle: inputs typed from TypeScript, the output support, the text,
   examples and constraints, and an execution plan that says which
   expressions can share one encoder pass.
2. **Train.** `semantscript train` generates cases from the text through a
   teacher model (or from complete constraints, with no model at all),
   fine-tunes one shared encoder for the application with one adapter per
   domain and one head per expression, calibrates, verifies against gold
   examples and constraints, and publishes an immutable artifact.
3. **Run.** The application loads the artifact once at startup. Each call is
   one encoder pass plus a head, a few milliseconds on a CPU; expressions
   over the same input in one request share the pass.

Measured on the refund benchmark's 160 judge-attested cases: the compiled
function answers 160 of 160 at 4.8 ms per call on a CPU, the same accuracy
as Claude Sonnet 5 with structured output at 4.2 s per call over the network,
against 54% for a 7B generative model reading the same policy. See the
[benchmarks](benchmarks/README.md) and the
[scaling results](docs/scaling-results.md) for the numbers behind the design.

## Start here

- [Getting started](docs/getting-started.md): add one expression to an
  existing Express or Next.js app and take it through `init`, `build`,
  `train`, `test` and `run`; or start
  [from a fresh clone](docs/tutorial-refund-decision.md) with the refund
  decision itself. `semantscript doctor` checks the environment first
  ([environment guide](docs/environment.md)), and
  `semantscript train --estimate` says what the teacher will cost before a
  paid run ([teachers](docs/teachers.md#cost-estimate-and-spend-cap)).
- [Phase 1 results](docs/phase-1-results.md): the refund benchmark, its
  systems and versions, the go/no-go and the encoder sizing rule.
- [Architecture](docs/architecture.md): compiler, trainer, model, runtime,
  CLI and framework in one diagram.
- [The reference application](docs/reference-application.md): a refund,
  ticket and order service whose policy is nine sema expressions, with source,
  compiled output and artifact side by side.

## References

- [Language reference](docs/language-reference.md), the
  [IR and artifact reference](docs/ir-and-artifact-reference.md),
  [execution plans](docs/execution-plans.md) and
  [structured outputs](docs/structured-outputs.md).
- [CLI reference](docs/cli-reference.md), the
  [build cache](docs/build-cache.md), the
  [diagnostics catalogue](docs/diagnostics.md) and the
  [build tool pages](docs/build-tools/tsc.md).
- [Framework guide](docs/framework-guide.md): controllers, request scopes,
  guards and transactions gated by decisions;
  [component guides](docs/components.md) and [teachers](docs/teachers.md).
- [All documentation](docs/index.md), including the normative
  [`SPEC.md`](SPEC.md) and [`IR.md`](IR.md), the recorded decisions and the
  research notes.

## Status

This repository is the working tree of the project: the packages
(`@semantscript/compiler`, `@semantscript/core`, `@semantscript/framework`,
the `semantscript` CLI and the Python trainer) are private workspaces linked
from source, not published to a registry. To try it, clone the repository,
run `npm install` and `npm run build`, install the Python training extra, and
follow the tutorial or one of the [examples](examples/README.md). Training
needs a GPU to be quick and a teacher (an Anthropic key, a local Ollama
model, or complete constraints); inference needs neither.
[CONTRIBUTING](docs/CONTRIBUTING.md) covers the layout, toolchains and
checks. GitHub Actions runs the lint and test gates, a fresh-clone install of
both examples and the Express Docker build on every push and pull request; a
fresh clone to a running image takes about a minute and a half on a hosted
runner with no npm cache ([measured](docs/CONTRIBUTING.md#continuous-integration)).
