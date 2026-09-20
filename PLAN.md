# SemantScript — Project Plan

> **SemantScript is a TypeScript-compatible programming model that makes learned semantic computation a typed language primitive.**
>
> JavaScript → TypeScript → SemantScript

Source of truth for the design rationale: [`semantscript-conversation-transcript.md`](semantscript-conversation-transcript.md) (indexed by turn).
Task tracking: Backlog.md (`backlog board`, `backlog task list --plain`).

---

## 0. North star

**The goal is a semantic system that lives alongside an application's TypeScript and is trained for that specific application — to the point where much of the application's hand-written logic is replaced by `sema` expressions.**

It is *not* a universal or general-purpose decision model. One application → one compiled artifact (shared encoder + that app's adapter + one head per expression), trained on that app's specs, examples and constraints. Universal encoders, warm starts and external benchmarks appear in this plan only as means to faster compilation or fairer measurement; they are never the deliverable. When a task or experiment starts optimizing for generality, it has drifted.

The measure of success is how much of a real application's behavior can be moved into `sema` expressions at production accuracy and latency — not how many tasks one model can do.

**The picture to build toward:** developers write database interactions, API endpoints and plumbing in TypeScript, and express nearly all business logic as `sema` expressions. Deterministic code handles persistence, routing, validation, transactions and hard constraints; learned code handles everything semantic. The reference application (TASK-8.3) must look like this.

## 1. The idea in one paragraph

Ordinary TypeScript gains one new primitive: a **typed expression whose implementation is learned from a natural-language specification** rather than written by hand.

```ts
const decision = sema<"approve" | "deny" | "review">`
    Apply our refund policy. Enterprise customers get 60 days,
    everyone else 30. Suspicious circumstances go to review.
    Customer: ${customer}  Order: ${order}
`;
```

TypeScript stays TypeScript, npm stays npm, existing applications remain valid. The programmer's rule is simple: **code when exact, natural language when semantic.**

## 2. What gets compiled

A `.sem.ts` source file compiles into two artifacts:

| Artifact | Produced by | Contents |
|---|---|---|
| **Plain JavaScript** | normal TS toolchain + our transformer | All deterministic code untouched; each `sema<T>` site rewritten to `__sema.call("<function_id>", { inputs })` |
| **Model artifact** | trainer | One shared encoder + per-app adapter + one tiny *head* per `sema` site, plus per-function metadata: input schema, output type, calibration, verification stats |

The natural-language text is consumed at **build time only** (to generate training data). It is not present in the artifact and no LLM runs in production.

**Type safety is structural, not a decoding constraint.** For `"approve" | "deny" | "review"` the head has exactly three logits; a `boolean` has one; a flat interface is several heads in parallel. The model cannot emit anything outside the type. There is no token stream to parse or validate — the output *is* the type. This is also why inference is a single forward pass and fast.

## 3. Design decisions (locked in from the transcript)

1. **Typed I/O is the contract.** Inputs are explicit interpolated variables — no ambient access to DB, network, filesystem or global state. Outputs are TS types.
2. **`sema<T>` is not a prompt; it is a compile target.** Build-time pipeline: analyze behavior + types → generate training cases → generate adversarial cases → train → calibrate → verify → emit `NeuralFunction` artifact.
3. **Non-autoregressive by default.** Encoder → typed head, one forward pass. No token generation, no constrained decoding, no JSON.
4. **One shared encoder, many tiny heads.** Base encoder (100M–1B) + per-application adapter + per-function heads. Fifty `sema` sites in a controller share nearly all compute.
5. **Parallelism is a compiler feature.** Independent `sema` expressions fuse into one encoder pass with fan-out heads; dependent ones become inference stages. The compiler emits a *neural execution plan* (analogous to a DB query plan).
6. **Three sources of truth for training:** types + natural-language behavior + optional `examples`. Plus deterministic `constraints` (`never(...)`, `always(...)`) that bound fuzzy behavior and drive adversarial case generation.
7. **Confidence is first-class.** `fn.withConfidence(...)` → `{ value, confidence }`; `@confidence(0.999)` makes the runtime route to a configured fallback below threshold.
8. **Expressions are neural, not functions.** Neural expressions live inside ordinary functions → minimal language surface, full TS/npm ecosystem, usable in backend, CLI, batch jobs and frontend alike.
9. **Prove the primitive before the framework.** Order: one `sema` expression → parallel expressions → controllers. Start with a ~100–300M encoder.
10. **v1 output types are head-representable only:** `boolean`, string-literal unions, enums, bounded numbers, flat interfaces of those. Free-text `string` output (requires generation) is out of scope until the primitive is proven.

## 4. Repository layout (target)

```
semantscript/
├── PLAN.md                 this file
├── SPEC.md                 language + artifact spec (Phase 0)
├── compiler/               TS transformer: find sema sites → IR → rewrite call sites, build DAG
├── trainer/                IR → synthetic + adversarial data → train → calibrate → verify   (Python)
├── model/                  encoder, adapters, heads, export                                  (Python)
├── runtime/                Node package: load artifact, run inference, confidence/fallback
├── cli/                    semantscript build | train | test | run
├── benchmarks/             refund-decision benchmark, baselines, results
└── examples/               reference apps
```

Naming conventions: `*.sem.ts` source files; `@semantscript/core` runtime import; `sema<T>` tagged template.

## 5. Phases

### Phase 0 — Specification
Pin the contract before writing code.
- `SPEC.md`: exact `sema<T>` syntax (tagged template, `${}` inputs), v1 output types, `examples` / `constraints` / `@confidence` forms, `withConfidence` API.
- IR / artifact format: function id, input schema, output head spec, model refs, calibration, verification, training provenance.

### Phase 1 — The primitive, end to end  ← *the real bet*
- **Compiler:** TS transformer that finds `sema<T>` sites, resolves `T` and interpolated input types to IR, rewrites the site to a runtime call.
- **Trainer:** given IR, generate labeled cases with a teacher model (+ adversarial cases around `constraints`), fine-tune a pretrained ~100–300M encoder with a classification head, calibrate, verify against `examples`/`constraints`.
- **Runtime:** Node package that loads the artifact and runs one forward pass in-process (ONNX Runtime).
- **Benchmark:** the refund-decision expression from the transcript.

**Exit criterion:** sub-10 ms local p50 and accuracy ≥ a 7B generative model with structured output on a held-out set. If this fails, stop and rethink.

### Phase 2 — Multi-head and parallelism
- Shared encoder + per-function heads in one artifact.
- Compiler builds the dependency DAG; fuses independent expressions into one pass; emits inference stages for dependent ones.
- Structured-interface outputs as multi-field heads.
- Benchmark parallel-head scaling and batch scaling.
- **Compile-time routed domain adapters (static MoE):** experts are LoRA-sized adapters per domain (controller / file / `@domain`), selected statically by the compiler in the execution plan — no router network. Capacity scales with the app's domains; latency stays that of one adapter (TASK-6.7).

### Phase 3 — Developer experience: drop-in adoption
The adoption path is an *existing* project, not a new one:

```
npm i @semantscript/core      # one dependency
npx semantscript init         # detects tsc / Vite / Next / esbuild, wires the plugin, zero config
# write one sema expression in an existing file
npx semantscript dev          # retrains changed expressions in the background, hot-swaps heads
git push                      # CI: semantscript build && test; artifact ships with the app
```

- Build-tool plugins (tsc transformer, Vite, esbuild) so `sema` compiles wherever TS already compiles.
- `semantscript init` with zero-config defaults; `build | train | test | run` CLI.
- `semantscript dev` watch mode: incremental retraining via the content-addressed cache, hot-swap without restart.
- Editor diagnostics (TS language-service plugin): accuracy, ECE, pair-consistency and guidance at each `sema` site.
- Runtime packaging for existing deployments (Node, Docker, serverless, Next server bundle).
- `withConfidence` / `@confidence` + fallbacks.
- Investigate: universal encoder + tiny heads → "compilation in seconds" (compile-time only; see §0).

### Phase 4 — Framework layer (thin)
- Thin integrations for Express, Nest and Next route handlers (decorators/middleware that batch `sema` evaluation per request), not a standalone server.
- Persistence & transaction patterns.
- Reference application that replaces a meaningful share of hand-written logic with `sema` expressions.
- Only after the primitive is proven; adoption comes from Phase 3, not from owning the app.

## 5b. Documentation gate (every phase)

Each phase ends with a documentation story (TASK-9 through TASK-13) and the phase epic depends on it, so a phase is not done until its docs are. Docs live under `docs/` with a single index; each phase adds pages and revises earlier ones where behavior changed.

| Phase | Docs deliverable |
|---|---|
| 0 | Language reference, IR/artifact reference, CONTRIBUTING, docs index |
| 1 | Getting-started tutorial, component guides, teacher configuration, benchmark write-up |
| 2 | Execution-plan concepts, multi-head artifact reference, structured outputs |
| 3 | CLI reference, build-cache behavior, diagnostics catalogue |
| 4 | Framework guide, reference-app walkthrough, public README, architecture overview |

## 6. Success metrics (from transcript, turn 9)

Accuracy · calibration error · P50/P95 latency · throughput · memory · batch scaling · parallel-head scaling · training time · adapter size.

Baselines: our ~200M model vs. 1B generative vs. 7B generative vs. traditional LLM structured output vs. Laya.

**Encoder size is a measured per-application knob, not a global choice.** Parallel heads amortize the encoder pass across every decision in a request, so a larger encoder is cheaper per decision here than in one-call-per-decision systems — but the pass itself scales with depth and weight bytes, so size still trades latency for capability. TASK-5.14 sweeps ~150M / ~400M / ~1B and reports accuracy, pair-consistency, ECE and p50/p95 latency (GPU and CPU, 1/10/50 heads). The rule for each application: the largest encoder that fits its latency budget.

## 7. Decisions

Recorded in `backlog/decisions/` (`backlog decision list`).

| Decision | Choice | Record |
|---|---|---|
| Teacher model for data generation | `claude-sonnet-5` as the **reference** teacher (structured outputs, Batch API for large runs); **Qwen3-14B via Ollama** as the local candidate, measured against the reference (TASK-5.13) | decision-1 |
| Training stack | Python/PyTorch for `trainer/` + `model/`; Node/TypeScript for `compiler/` + `runtime/` + `cli/`; ONNX Runtime for in-process inference | decision-2 |
| Base encoder for Phase 1 | ModernBERT-base (~149M); DeBERTa-v3-base fallback | decision-3 |
| Phase 1 benchmark task | Refund decision (transcript example) | in TASK-5.11 |

Hardware note: dev machine is a Radeon RX 9070 XT (16 GB) under WSL2; ROCm is not currently reachable from WSL. Run Ollama natively on Windows and reach it at `localhost:11434`; CPU fine-tuning of the ~150M encoder is the fallback for training.

## 8. Naming

**SemantScript** · SemantScript Compiler · SemantScript Runtime · SemantScript IR · SemantScript SDK
`semantscript build|train|test|run` · `*.sem.ts` · `@semantscript/core` · `sema<T>`
