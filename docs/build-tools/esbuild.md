# Build tool: esbuild

```js
import { build } from "esbuild";
import semantscript from "@semantscript/compiler/esbuild";

await build({
  entryPoints: ["src/server.ts"],
  bundle: true,
  platform: "node",
  format: "esm",
  outdir: "dist",
  sourcemap: true,
  packages: "external",
  plugins: [semantscript({ application: "ticket-triage" })],
});
```

`semantscript init` finds the first `build.mjs`, `esbuild.config.*` or
`scripts/build.*` that calls esbuild and inserts the import and the plugin
entry.

## Options

`tsconfig` (default: the nearest `tsconfig.json`), `application`,
`bundlePath` (default: esbuild's `outdir`, or beside `outfile`),
`domainDepths` and `routeDomains`, the same set as every adapter.

## What happens in a build

The plugin builds its own TypeScript program from the tsconfig at each
build start, with the emit shape a bundler expects (per-file JavaScript,
separate source maps with `inlineSources`, no declarations, `module` raised
to ESNext under classic or bundler resolution), writes the IR bundle, and
serves each `.sem.ts` module esbuild asks for as rewritten JavaScript with its
map. esbuild composes that map into its output, so a trace points at the
original expression. Type checking is left to the application's own `tsc`;
sema-site diagnostics still fail the build with the file, line and column as
esbuild errors.

## Known limitations

- `packages: "external"` (or marking `@semantscript/core` external) is
  required for a Node bundle: the runtime loads native ONNX and tokenizer
  bindings that cannot be bundled.
- The project is planned at every build start, including each rebuild in
  esbuild's watch or `context` mode (the previous program is reused so
  rebuilds are cheap), and `tsconfig.json` is registered as a watched file;
  a planning failure fails the build before any module loads.
- The bundle is written to `outdir` unless `bundlePath` says otherwise;
  `semantscript train` finds it there through the tsconfig `outDir` search
  only when the two agree, so pass `--bundle` when they differ.
