---
id: decision-12
title: >-
  Local teacher: Qwen3-14B is not acceptable as a labeling teacher for
  constraint-heavy policies; iterate locally with constraint labels and reserve
  Sonnet for generation
date: '2026-09-25 05:47'
status: accepted
---
## Context



## Decision



## Consequences


## Context

TASK-5.13 asked whether a local teacher is good enough to iterate with instead of the Sonnet 5 reference (decision-1). On 1,800 real refund inputs (`benchmarks/refund/data/results-local-teacher-2026-09-25`), Sonnet 5 through OpenRouter agreed with the compiled constraints' unique admissible label on every input (1.000) at USD 0.0028 and four seconds per label; Qwen3-14B through Ollama (thinking off, temperature 0) agreed on 0.746, approving 283 of 440 suspicious-history cases the policy sends to review and 26 of 68 orders outside their tier window. Students trained on the same 1,500 inputs with each label set scored 0.980 (ECE 0.019) with Sonnet's labels and 0.800 (ECE 0.143) with Qwen's on the frozen Sonnet-labeled evaluation set: an 18-point gap that is exactly the teacher's systematic error.

## Decision

- **Qwen3-14B is not an acceptable labeling teacher** for policies with compound conditionals like the refund policy; a student inherits its errors. No local generative model is adopted in the teacher slot on the strength of a smaller model reading the same text.
- **Local iteration uses constraint labels, not a local LLM.** For a policy whose rules are stated as constraints, the constraints label real or sampled inputs for free and instantly (decision-7; the refund-service example trains with no language model at all), and the build cache keeps rebuilds cheap. That is the local path.
- **Sonnet 5 remains the reference teacher**, used for generation on policies whose text says more than their constraints, through the Anthropic API, OpenRouter's Anthropic-format route, or the Claude Code CLI on a subscription.

## Consequences

- The Ollama teacher backend stays available for experiments and for future local models; each candidate must be measured the same way (agreement per rubric step, then a student on the fixed target) before it is used to build anything.
- Real-input pools alone under-represent rare rules (the UCI pool has no fraudulent orders), so a training corpus still needs the synthetic threshold-dense cases a reference teacher or the constraints provide.
