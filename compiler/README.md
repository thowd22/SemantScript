# Compiler

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
sema<...>\`...\`)`), so `semantscript build` output points at the expression to
fix. `findMalformedSemaSites` reports core `sema` and `sema.withConfidence`
tagged templates that are not canonical sites (no or several output type
arguments, a configured call with zero or several options arguments, optional
chaining); they are compile errors rather than untransformed runtime tags.
Codes: 9100 malformed site; 9101 unsupported output type; 9102 invalid enum;
9103 invalid ordinal; 9104 invalid bounds; 9105 invalid flat output field;
9110 invalid interpolation; 9111 duplicate interpolation; 9112 unsupported
input type; 9120 invalid options; 9121 invalid example; 9122 invalid
constraint; 9123 invalid `@confidence`; 9124 empty behavior; 9125 contradictory
constraints; 9130 compilation configuration; 9131 execution plan. The rendered
output of one fixture per family is snapshot-tested in
`test/fixtures/diagnostics.snapshot.txt` (regenerate with `UPDATE_SNAPSHOTS=1`).

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
