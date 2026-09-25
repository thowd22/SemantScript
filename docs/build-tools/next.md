# Build tool: Next.js (webpack and Turbopack loader)

```ts
// next.config.ts
import type { NextConfig } from "next";

const config: NextConfig = {
  turbopack: {
    rules: { "*.sem.ts": { loaders: ["@semantscript/compiler/loader"] } },
  },
  serverExternalPackages: ["@semantscript/core"],
  outputFileTracingIncludes: { "/**": ["./.semantscript/artifact/**"] },
};

export default config;
```

`semantscript init` writes these three keys when none of them is set; a
`next.config` that already sets one is reported as `manual` with the snippet
to merge. The same loader works on the webpack path (`next build --webpack`)
through a `module.rules` entry with `test: /\.sem\.ts$/`.

## Options

Loader options go in the rule (`{ loader: "@semantscript/compiler/loader",
options: { … } }` on webpack; Turbopack rules take the loader name only, so
use the defaults or a `tsconfig` beside the config): `tsconfig`,
`application`, `bundlePath` (default: the tsconfig `outDir`, else beside
`tsconfig.json`, which is where a Next.js project's bundle lands), `domainDepths`
and `routeDomains`.

## What happens in a build

The loader plans the whole TypeScript project once per build process and
re-plans when a `.sem.ts` source's text differs from the program's copy, so
`next dev` picks up edits. Each `.sem.ts` module is returned as rewritten
JavaScript with a source map, and the loader's own map points at the original
expression. No prompt text reaches `.next/`. Diagnostics fail the build with
the file, line and column.

The artifact loads once per server process: with `@semantscript/core`
installed from the registry, `instrumentation.ts` is the idiomatic place for
`await loadSemaArtifact()`; the [example app](../../examples/next-app/README.md)
loads it in the route module because its runtime is linked from source and
Turbopack bundles a linked package into each entry separately.
`outputFileTracingIncludes` makes a standalone output carry the artifact
directory, and `serverExternalPackages` keeps the native ONNX and tokenizer
bindings out of the bundle.

Verified with Next.js 16.3.6 on both `next build` (Turbopack) and
`next build --webpack`: the route compiles, and `next start` with
`SEMANTSCRIPT_ARTIFACT` set answers through the compiled sema call.

## Known limitations

- **Turbopack does not compose the loader's source map** into its server
  chunks: a stack trace names `lib/triage.sem.ts` but keeps the loader
  output's line for the call, whereas esbuild, Vite and webpack compose the
  map and point at the expression's original line. Diagnostics are unaffected.
- A sema expression can only run in server code (route handlers, server
  components, server actions): the runtime is native Node. A client component
  that imports a `.sem.ts` module fails to bundle its runtime.
- Turbopack rules take loader names without options; put per-project
  settings in `tsconfig.json` or accept the defaults.
