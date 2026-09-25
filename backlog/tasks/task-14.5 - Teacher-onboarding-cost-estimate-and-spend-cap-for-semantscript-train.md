---
id: TASK-14.5
title: 'Teacher onboarding, cost estimate and spend cap for semantscript train'
status: To Do
assignee: []
created_date: '2026-09-25 15:21'
labels:
  - dx
  - train
milestone: m-5
dependencies: []
parent_task_id: TASK-14
ordinal: 58000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A training run through a paid teacher gives the developer no idea what it will cost or how long it will take, and no way to bound it. On 2026-09-25 the Express example's two expressions at 192 cases cost about USD 10 through OpenRouter against an estimate of one dollar, because the teacher prompt is about 8,300 tokens per request, a constrained expression needs about twice as many requests as cases, and an anchor the teacher cannot twin costs three attempts. Setting the teacher up is also manual: init assumes ANTHROPIC_API_KEY and anything else means writing a TOML by hand from the trainer README.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 semantscript train --estimate prints, per expression and in total, the number of teacher requests, the input and output tokens, the cost in USD at the configured backend's price and the expected wall time, without calling the teacher
- [ ] #2 --max-cost-usd stops a run before the request that would exceed it, with the cached datasets kept, and every run prints its running cost and request count
- [ ] #3 init asks for or accepts a teacher choice (Anthropic, OpenRouter, Ollama, constraints) and writes the TOML; semantscript teacher probe sends one request through the configured backend and reports the model, latency and cost
- [ ] #4 The teacher prompt is measured and reduced (or served through prompt caching) so that a case costs no more than half of the 2026-09-25 figure at equal verification results on the Express example
<!-- AC:END -->
