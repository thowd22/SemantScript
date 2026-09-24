# Next.js app adopting sema through the loader

A Next.js 16 App Router project with one API route. SemantScript was adopted by
registering the loader in `next.config.ts` and adding one `.sem.ts` file.

What adoption touched:

1. `next.config.ts`: the `turbopack.rules` entry for `*.sem.ts` (and the same
   loader on the webpack path for `next build --webpack`), plus
   `serverExternalPackages` for the runtime's native ONNX and tokenizer
   bindings.
2. `lib/triage.sem.ts`: the one `sema` expression, `triage(subject, body)`.
3. `app/api/triage/route.ts` calls `triage` and loads the trained artifact
   once per server process with `loadSemaArtifact`. The artifact load is the
   runtime half of adoption; every sema program needs it once. With
   `@semantscript/core` installed from the registry, `instrumentation.ts` is
   the idiomatic place for it; in this repository the runtime is linked from
   source and Turbopack bundles a linked package into each entry separately,
   so the route module loads it itself.

Build and run:

```sh
npm install                 # links ../../compiler and ../../runtime, installs next and react
npm run build               # next build: the loader writes semantscript.ir.v1.json beside tsconfig.json
semantscript train --bundle semantscript.ir.v1.json --artifact artifact --teacher teacher.toml
npm start                   # POST /api/triage {"subject": "...", "body": "..."}
```

The loader plans the whole TypeScript project once per build process and
re-plans when a `.sem.ts` file changes, so `next dev` picks up edits. With no
`outDir` in a Next.js tsconfig the bundle lands next to `tsconfig.json`; set
`bundlePath` in the loader options to move it.

Verified on 2026-09-24 with Next.js 16.3.6: `next build` (Turbopack and
`--webpack`) compiles the route, no prompt text reaches `.next/`, and
`next start` with `SEMANTSCRIPT_ARTIFACT` set answers `POST /api/triage`
through the compiled sema call. Turbopack's composed server source maps name
`lib/triage.sem.ts` but keep the loader output's line for the call; the
loader's own map (which esbuild, Vite and webpack compose) points at the
original expression.
