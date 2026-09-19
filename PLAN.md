# SemantScript — Project Plan

> **SemantScript is a TypeScript-compatible programming model that makes learned semantic computation a typed language primitive.**
>
> JavaScript → TypeScript → SemantScript

Source of truth for the design rationale: [`semantscript-conversation-transcript.md`](semantscript-conversation-transcript.md) (indexed by turn).
Task tracking: Backlog.md (`backlog board`, `backlog task list --plain`).

---

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

### Phase 3 — Developer experience
- `semantscript build | train | test | run` CLI.
- `withConfidence` / `@confidence` + fallbacks.
- Build cache: unchanged expressions don't retrain.
- Investigate: universal encoder + tiny heads → "compilation in seconds" instead of minutes.

### Phase 4 — Framework layer
- Controllers / HTTP runtime, persistence & transactions integration, an example application.
- Only after the primitive is proven.

## 6. Success metrics (from transcript, turn 9)

Accuracy · calibration error · P50/P95 latency · throughput · memory · batch scaling · parallel-head scaling · training time · adapter size.

Baselines: our ~200M model vs. 1B generative vs. 7B generative vs. traditional LLM structured output.

## 7. Open decisions

| Decision | Options | Status |
|---|---|---|
| Teacher model for data generation | API model (fast start) vs. local model (cost/privacy) | **open** |
| Training stack | Python/PyTorch for trainer+model, Node for compiler+runtime *(recommended)* vs. all-TS | **open** |
| Base encoder for Phase 1 | DeBERTa-v3 / ModernBERT-class, 100–300M | open |
| Phase 1 benchmark task | Refund decision (transcript example) | proposed |

## 8. Naming

**SemantScript** · SemantScript Compiler · SemantScript Runtime · SemantScript IR · SemantScript SDK
`semantscript build|train|test|run` · `*.sem.ts` · `@semantscript/core` · `sema<T>`
