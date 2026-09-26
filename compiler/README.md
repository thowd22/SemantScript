# Compiler

```sh
npm install --save-dev @semantscript/compiler
```

Published as `@semantscript/compiler` (first publish pending, see
[releasing](../docs/releasing.md)); `VERSION` is the release version it shares
with the other SemantScript packages.

The compiler package finds `sema<T>` sites in TypeScript, resolves their input and
output types, emits lifecycle-aware NeuralFunction IR, builds neural dependency
plans, and rewrites source sites to runtime calls.

It owns static analysis and diagnostics. It does not generate training data, train
models, or run inference.

## Site discovery

`findSemaSites(program, sourceFile?)` walks `.sem.ts` source files and returns
each supported `sema<T>` tagged template in source order. Discovery uses the
TypeScript type checker to match the `sema` export from `@semantscript/core`, so
import aliases, namespace imports, and re-exports work without matching local
lookalikes. Each result includes the tagged-template node, output type node,
optional configuration expression, result mode, and a one-based source
location. `start` and `end` are zero-based, end-exclusive UTF-16 offsets, matching
the TypeScript compiler API.

## Type analysis and source IR

`analyzeSemaSite(program, site)` and `analyzeSemaSites(program, sourceFile?)`
resolve output heads, interpolation schemas, and template boundaries into JSON-only
IR values. Successful analyses cover nominal booleans/unions/enums, ordinal marker
types, exact bounded-number grids, flat interface outputs, and recursively typed
inputs. Failures are returned as located TypeScript diagnostics and include the
offending `sema` site.

Every diagnostic the compiler emits carries the source file, line and column of
the `sema` site, its code and the site text (`file:line:column: message (site:
sema<...>\`...\`)`), so `semantscript build`output points at the expression to
fix.`findMalformedSemaSites`reports core`sema`and`sema.withConfidence`tagged templates that are not canonical sites (no or several output type
arguments, a configured call with zero or several options arguments, optional
chaining); they are compile errors rather than untransformed runtime tags.
Codes: 9100 malformed site; 9101 unsupported output type; 9102 invalid enum;
9103 invalid ordinal; 9104 invalid bounds; 9105 invalid flat output field;
9110 invalid interpolation; 9111 duplicate interpolation; 9112 unsupported
input type; 9120 invalid options; 9121 invalid example; 9122 invalid
constraint; 9123 invalid`@confidence`; 9124 empty behavior; 9125 contradictory
constraints; 9126 invalid `@domain`; 9130 compilation configuration; 9131
execution plan. The rendered
output of one fixture per family is snapshot-tested in
`test/fixtures/diagnostics.snapshot.txt`(regenerate with`UPDATE_SNAPSHOTS=1`).

`createSourceNeuralFunctionIr(analysis, options)` wraps a successful analysis in a
complete source-stage NeuralFunction record. The caller supplies build identities,
source hashes, logical model/head references, and any already-parsed examples or
constraints.
Bounded support is limited to 100,000 values so invalid or impractical type
declarations cannot exhaust compiler memory. Static examples and constraints
share a 100,000-node work budget per site and reject nesting beyond 100 levels.
Execution-plan provenance has a 100-level depth limit and a 100,000-unit global
budget covering AST traversal, provenance merging, write indexing, and edge
expansion.

## Compilation and emission

`planSemaCompilation(program, options)` resolves every site without writing files.
It returns deterministic function IDs, a versioned source-stage IR bundle, and
the bundle's exact text. The bundle includes a dependency plan whose dense stages
use the minimum topological depth: independent sites share a stage and consumers
follow all identified potential producers. Dependencies are a conservative
may-analysis through initialized bindings, derived expressions, direct
assignments, and direct property writes, so overwritten values can leave harmless
false-positive edges. They do not imply that sites in different branches, loops,
or call frames execute together. Same-container writes textually after a consumer
are excluded, while cross-file and cross-container writes remain conservative
potential sources. Mutation hidden behind calls or accessors and interprocedural
return analysis are not included in the v1 plan.

IDs include the project-relative source path, the duplicate ordinal among
semantically identical sites in that file, and the SHA-256 of the semantic IR
projection. Execution-plan metadata is outside that projection, so unrelated
source or topology edits do not invalidate an unchanged site.

`emitSemaCompilation(program, plan, options)` runs the SemantScript rewrite as the
first TypeScript `before` transformer. Each tagged template becomes an imported
`__sema.call(functionId, { inputs })`; the prompt and inline compiler options are
not emitted. Emission first verifies that the plan exactly covers the current
`Program`; callers pass the exact, unmodified plan object returned by planning.
Emission uses an internal immutable snapshot so asynchronous filesystem checks
cannot change rewrite coverage. TypeScript outputs are captured before any disk write, source-map
`sourcesContent` is removed, source/output collisions are checked through
canonical filesystem paths (including dangling symlink components), and every
file is fully staged before commits begin. Files are committed atomically with
the IR bundle last. The default bundle is `semantscript.ir.v1.json` in `outDir`.

`compileSemantScriptProgram(program, options)` performs both phases. Callers must
provide `projectRoot`; they may override the logical encoder/adapter references,
source-byte reader, or bundle path. A failed analysis or TypeScript emit returns
diagnostics and does not write the bundle.

`planSemaCompilationSync(program, options)` is the same planner for callers that
run inside a synchronous TypeScript emit or loader; it reads source bytes with
`readFileSync` (or `readSourceBytesSync`) and returns the identical plan.
`createSemaProgramTransformer(program, plan)` validates the plan against the
program and returns the rewrite as a `before` transformer for a caller that
drives `program.emit` itself. `emitSemaSourceFile(program, plan, sourceFile)`
emits one file in memory for a bundler: its JavaScript without the
`sourceMappingURL` comment, and its map without `sourcesContent`.

## Routed domains and depth routing

`planSemaCompilation` takes `domainDepths`, a map of domain name to the number
of shared-encoder layers that domain runs. Each site's domain is its
`@domain(name)` header, else the enclosing class's name (a controller), else
its file name without `.sem.ts` (all lowercased with dashes). When any depth is named or any site carries a header the
plan is routed: every function's `model.adapter` becomes
`<adapterRef>.<domain>`, a domain with a depth gets `model.encoder`
`<encoderRef>.depth-NNN` and `model.encoderDepth`, the execution plan gains a
`domains` list (name, adapter ref, encoder ref, depth, function ids) and each
stage lists the `adapterRefs` it applies. Unrouted projects keep the single
application adapter and the plan shape of before. Routing is outside the
semantic projection, so it never changes a function id; it is part of the
trainer's cache key, so changing a domain's depth retrains that domain only.
Measured on this stack, ONNX Runtime executes a graph whole whatever outputs
are fetched, so the trainer exports one encoder prefix graph per depth in use
and the runtime runs the prefix a function's domain names.

## Build-tool adapters

Adoption means sema compiles wherever the application's TypeScript already
compiles, so the package ships four thin adapters over the entry points above.
Each plans the whole project (one TypeScript program from the nearest
`tsconfig.json`, or the `tsconfig` option), writes the IR bundle, then rewrites
`.sem.ts` files one at a time. Every adapter takes the same options:
`tsconfig`, `application` (names the artifact's encoder and adapter refs,
default `application`), `bundlePath`, and the routing options `domainDepths`
(domain name to encoder depth) and `routeDomains`, which the adapters pass to
`planSemaCompilation` unchanged.

| Adapter                              | One-line adoption                                                                                           | Bundle default                                 |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `@semantscript/compiler/transformer` | tsconfig `"plugins": [{ "transform": "@semantscript/compiler/transformer" }]`, built with ts-patch's `tspc` | `outDir/semantscript.ir.v1.json`               |
| `@semantscript/compiler/esbuild`     | `plugins: [semantscript()]` in the esbuild build options                                                    | esbuild `outdir` (or beside `outfile`)         |
| `@semantscript/compiler/vite`        | `plugins: [semantscript()]` in `vite.config`                                                                | Vite `build.outDir`, rewritten after emptying  |
| `@semantscript/compiler/loader`      | Turbopack `rules: { "*.sem.ts": { loaders: ["@semantscript/compiler/loader"] } }` or a webpack rule         | tsconfig `outDir`, else beside `tsconfig.json` |

The transformer plans from the program tsc hands it and reports planning
diagnostics through ts-patch's `addDiagnostic`, so a malformed site fails the
tsc build with its `TS9100` location like any type error. The bundler adapters
build their own program with the emit shape a bundler expects (per-file
JavaScript, separate source maps with `inlineSources`, no declarations, a
script-style `module` raised to ESNext under classic or bundler resolution) and
leave type checking to the application's own `tsc`; sema-site diagnostics still
fail the build with the file, line and column. The Vite plugin re-plans on the
next transform after any watched change; the loader re-plans when a compiled
source's text differs from the program's copy.

Source maps come from TypeScript's emit, with each rewritten call mapped to the
original tagged template, so bundler-composed maps and `node
--enable-source-maps` stack traces point at the `.sem.ts` line of the sema
expression. `test/build-tools.test.mjs` builds one fixture through `tspc`,
esbuild and Vite and asserts that trace for each. The loader returns the same
map to webpack and Turbopack; Next.js 16.3 with Turbopack was observed to keep
the loader output's positions in its composed server maps (with or without
`sourcesContent`), so a Turbopack stack trace names the `.sem.ts` file but the
emitted line. Diagnostics carry the original position in every adapter. The
subpaths resolve under
both `import` and `require` (ts-patch and webpack load them with `require`, which
Node 22.12+ supports for ESM without top-level await).

Runnable adoption examples: [`examples/express-app`](../examples/express-app)
(tsc transformer), [`examples/next-app`](../examples/next-app) (loader) and
[`examples/refund-service`](../examples/refund-service) (transformer with
routed, depth-6 domains).

## Editor plugin

`@semantscript/compiler/ts-plugin` is a TypeScript language-service plugin,
enabled by one tsconfig entry that `semantscript init` writes:

```json
"plugins": [{ "name": "@semantscript/compiler/ts-plugin" }]
```

It makes tuning a neural function feel like fixing a type error. At every
sema site it adds a hover with the function's short id and output type, its
input, example and constraint counts and confidence threshold, and, from the
latest artifact (`artifact` option, default `.semantscript/artifact` beside
the tsconfig), the verified status, accuracy, ECE, pair consistency, attested
cases and constraint violations. Inline diagnostics carry the compiler's own
errors (unsupported output types, non-identifier or repeated inputs, invalid
examples and constraints, codes 9100 to 9131) at the site, plus warnings for
an expression with no trained artifact (9150), one missing from the latest
artifact because it changed since training (9151), verified accuracy below
`accuracyThreshold` (default 0.95, code 9152) and ECE above `eceThreshold`
(default 0.1, the trainer's gate, code 9153), each with the next thing to try
(add examples when there are none, a constraint for a broken rule, a
confidence threshold with a fallback). The plugin plans the project through
the service's own program, so unsaved buffers are analysed, and rereads the
artifact only when its pointer changes.

The entry is the CommonJS directory `ts-plugin/` at the package root because
tsserver resolves plugins by file layout (not the `exports` map), loads them
with `require` and expects the factory function itself. tsserver probes its
own installation and any `--pluginProbeLocations`; VS Code passes the
workspace folders, so the plugin in the project's `node_modules` loads with
VS Code's bundled TypeScript and no extension. `test/ts-plugin.test.mjs`
drives it both in-process through `ts.createLanguageService` and through a
real `tsserver` session with the workspace as probe location, the path every
editor takes.
