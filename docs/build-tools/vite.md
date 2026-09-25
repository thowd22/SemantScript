# Build tool: Vite

```ts
// vite.config.ts
import { defineConfig } from "vite";
import semantscript from "@semantscript/compiler/vite";

export default defineConfig({
  plugins: [semantscript({ application: "ticket-triage" })],
});
```

`semantscript init` adds the import and puts `semantscript()` first in
`plugins` so `.sem.ts` files are rewritten before any other transform sees
them.

## Options

`tsconfig`, `application`, `bundlePath` (default: Vite's `build.outDir`),
`domainDepths` and `routeDomains`.

## What happens in a build

The plugin plans the project from the tsconfig, writes the IR bundle, and
transforms each `.sem.ts` module into rewritten JavaScript with a map Vite
composes into its output. Because Vite empties `build.outDir` before writing,
the plugin writes the bundle after that step, so it is present in the final
output directory. In `vite dev` the plugin re-plans on the next transform
after any watched change, so an edited expression is picked up without a
restart. Diagnostics fail the build as Vite errors with the file, line and
column.

## Known limitations

- Vite is a browser-first bundler; the sema runtime runs in Node (ONNX
  Runtime and the tokenizer are native bindings). Use the plugin for SSR
  builds (`vite build --ssr`) or a Node target with the runtime marked
  external; a client bundle cannot evaluate a `sema` expression.
- The plugin plans once per Vite build and re-plans lazily in dev; a change
  to a type in another file that a sema input uses is seen on the next
  `.sem.ts` transform, not immediately.
