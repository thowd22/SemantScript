# Examples

This directory contains source fixtures, schema-valid IR and artifact examples,
canonical serialization vectors, constraint predicate vectors
([`constraints/predicate-vectors.v1.json`](constraints/predicate-vectors.v1.json),
evaluated by both the trainer and `semantscript explain`), and runnable
reference applications.

Checked-in artifact metadata is illustrative unless its directory explicitly
contains the referenced model resources.

## Reference applications

Each is a standalone package (not a workspace) that links the compiler and
runtime from this repository with `file:` dependencies, so `npm install` inside
it after `npm run build` at the root is enough to build it.

- [`express-app`](express-app/README.md): an Express 5 ticket API compiled by
  plain `tsc`; sema is adopted through the ts-patch transformer entry in
  `tsconfig.json` and one `.sem.ts` file.
- [`next-app`](next-app/README.md): a Next.js 16 App Router project; sema is
  adopted through the loader rule in `next.config.ts` and one `.sem.ts` file.
- [`refund-service`](refund-service/README.md): the reference application. A
  refunds, tickets and orders API over Postgres (PGlite) whose business policy
  is nine sema expressions in three routed, depth-6 domains; trained from its
  own constraints by `semantscript train --teacher constraints` with no
  language-model teacher, with a
  walkthrough of the source, the compiled output, the artifact, the line tally
  and per-expression accuracy and latency.

All three need a trained artifact to answer requests: build, then
`semantscript train` on the emitted IR bundle (or `npm run train` in the
refund service), then start the app with the artifact root. The Express app
and the refund service also carry `npm test` suites that run their routes over
a fixture artifact with no training; CI installs, builds and tests both from a
fresh clone on every push and builds the Express Docker image
([measured](../docs/CONTRIBUTING.md#continuous-integration)).
