# Laya — analysis and lessons for SemantScript

Source: https://github.com/NandhaKishorM/laya (Apache 2.0), read 2026-09-20. Code inspected: `laya/common.py`, `BENCHMARKS.md`.

## What Laya is

A non-autoregressive "System 1 decision engine": ModernBERT-large (421M) / mmBERT-base (322M) encoder, one forward pass, three typed primitives (`choice`, `score` ordinal, `noul` probability). Claims 33–40 ms/question on a Tesla T4 and 6–7× faster than Jev, 0.766 vs Jev's 0.727 on their typed-decisions benchmark (after fine-tuning).

## Mechanism (from code)

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
```

- Options are rendered as **text spans inside the input**, each preceded by a `[MASK]` marker. A small scorer MLP reads the hidden state at each marker → one logit per option. Softmax over markers.
- Options share a fixed `head_max_len` token budget (192/256), so cardinality above ~20 collapses (banking77: 0.425).
- A 2-layer transformer "decision head" sits on top of the encoder; a type embedding distinguishes choice/score/noul.
- **Loss = strictly proper scoring rule** (log score + 0.5·spherical score, plus ranked-probability-score for ordinal `score` questions), optimized GRPO-style.
- **Confidence = 1 − H(p)/log(k)** (normalized entropy). Per-bucket temperature (`qtype × option-count`) fitted post-hoc.
- An auxiliary `act_head` takes `[CLS]`, top-1 prob, top-1−top-2 margin, entropy and k, and predicts whether to act — a learned abstain signal.

## Their measured results that matter to us

| Finding | Number | Implication for SemantScript |
|---|---|---|
| Zero-shot base on typed-decisions | 0.36 (majority 0.46, random 0.32) | A universal runtime decision encoder is weak; task-specific training is where accuracy comes from. Supports compile-time training. |
| Fine-tuned | 0.766, above the teacher's 0.735 | A small student can exceed its LLM teacher on a narrow task. |
| ECE as shipped → after temperature fit | 0.466 → 0.081 | Ship nothing without a fitted temperature; calibration is the single highest-value step. |
| Option-order flip rate | 0.04–0.23 | Text-encoded options are order-sensitive. Fixed heads are immune; input-field order must still be canonicalized. |
| Moderation on held-out real traffic | 0.53 (hand-picked examples looked fine) | Synthetic/teacher-generated eval sets overstate accuracy. Held-out must include real, non-teacher data. |
| Ordinal `score` | SST-5 0.372 | Ordinal outputs need explicit ordinal treatment (RPS/CDF), and still lag; budget expectations accordingly. |
| Latency, 421M, 512 tokens, T4 | 33–40 ms | Our <10 ms target needs a base-size encoder, short inputs (no question/option tokens) and fp16 ONNX. Plausible, not free. |

## Where SemantScript differs on purpose

Laya is a **runtime** system: question + options are model input, so new questions need no training but every call pays for encoding them and cardinality is capped. SemantScript is **compile-time**: the spec is consumed at build, the output type is a fixed head with N logits, the runtime never sees the text. We trade zero-shot flexibility for accuracy, unbounded cardinality, structured multi-field outputs and lower latency.

## Adopted

1. **Proper-scoring-rule loss** (log + spherical; + RPS for ordinal) as the supervised training objective for heads, instead of plain cross-entropy. We have labels, so no RL/GRPO is needed.
2. **Per-head temperature fitting** on a calibration split as a mandatory build step; ECE reported per function; Brier score reported.
3. **Ordinal output type** in the spec (ordered unions / bounded integers) with RPS-aware training and `expected value + distribution` at runtime.
4. **Confidence definition:** calibrated top-1 probability as `confidence`, normalized entropy exposed as `uncertainty`; a learned abstain predictor is a Phase 3 option, not v1.
5. **Canonical input serialization** in the IR: deterministic field order, type-aware rendering, so results are invariant to object key order.
6. **Held-out evaluation must include real, human-authored cases** — not only teacher-generated data — for every benchmark.
7. **Laya as a baseline** in the Phase 1 benchmark (fairer than a 7B generative model), and `laya-typed-decisions` as a candidate encoder init.
8. **Laya's 400-case / 2,000-decision typed-decisions benchmark** as an external test: four workflows with several questions each is exactly a multi-head, multi-function workload for Phase 2.
9. **Warm-start idea for fast compilation (Phase 3):** a Laya-style universal option-scoring encoder can produce initial logits for a new function with zero training; distill those into a fixed head, then refine. This is a concrete path to "compile in seconds".

## Not adopted

- Text-encoded options at runtime (the core Laya mechanism) — conflicts with compile-time heads.
- GRPO / policy-gradient training — unnecessary with labeled data.
- Multilingual routing — out of scope.
- The 2-layer transformer decision head — start with a linear/MLP head per function; revisit only if accuracy demands it.
