# Build tool: tsc (ts-patch transformer)

For a project compiled by plain `tsc`, sema is a TypeScript `before`
transformer loaded through [ts-patch](https://github.com/nonara/ts-patch).
This is what `semantscript init` sets up when it finds a `tsconfig.json` and
no other build tool, and what the [Express example](../../examples/express-app/README.md)
and the [refund service](../../examples/refund-service/README.md) use.

## Setup

`tsconfig.json`:

```json
{
  "compilerOptions": {
    "outDir": "dist",
    "plugins": [
      { "name": "@semantscript/compiler/ts-plugin" },
      {
        "transform": "@semantscript/compiler/transformer",
        "application": "ticket-triage",
        "domainDepths": { "refunds": 6 }
      }
    ]
  }
}
```

`package.json`: `ts-patch` as a dev dependency, `"prepare": "ts-patch
install"` so a fresh clone patches its TypeScript, and the build script
`tspc -p tsconfig.json` in place of `tsc`. `init` writes all three edits
idempotently.

Options on the `transform` entry: `application` (names the artifact's encoder
and adapter refs, default `application`), `bundlePath` (default
`semantscript.ir.v1.json` in `outDir`), `domainDepths` (domain name to
encoder depth) and `routeDomains`. The `name` entry is the editor plugin and
takes `artifact`, `accuracyThreshold` and `eceThreshold`.

## What happens in a build

The transformer plans the whole program `tsc` hands it (every `.sem.ts` file
it includes), writes the IR bundle, and rewrites each `sema` site to
`__sema.call(id, inputs)` as the first `before` transformer. Emitted
JavaScript is what `tsc` would produce otherwise, with the template, examples
and constraints removed. Source maps come from TypeScript's own emit with
each rewritten call mapped to the original tagged template, so a stack trace
under `node --enable-source-maps` names the `.sem.ts` line of the expression.

Diagnostics are reported through ts-patch's `addDiagnostic`, so a malformed
site fails the build like a type error with its `TS91xx` code, file, line and
column ([catalogue](../diagnostics.md)).

## Known limitations

- `tsc` without ts-patch ignores `plugins`, so a build that bypasses `tspc`
  (an editor's build task, a bare `npx tsc`) emits the untransformed tag,
  which throws `sema must be compiled` at runtime. Keep `tspc` in every build
  script and let `prepare` patch the installation.
- `noEmit`, `emitDeclarationOnly`, or an `include` that omits the `.sem.ts`
  files gives the 9130 diagnostic `TypeScript skipped JavaScript emission`;
  the transformer needs JavaScript output for the sema files.
- Project references (`tsc -b`) run the transformer per project; each gets
  its own bundle in its own `outDir`, and `semantscript train` takes one
  bundle at a time.
