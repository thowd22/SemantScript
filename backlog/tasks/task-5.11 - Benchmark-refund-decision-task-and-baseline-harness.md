---
id: TASK-5.11
title: 'Benchmark: refund-decision task and baseline harness'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-19 18:23'
updated_date: '2026-09-23 22:02'
labels:
  - benchmark
milestone: m-1
dependencies:
  - TASK-5.3
  - TASK-5.8
  - TASK-5.10
  - TASK-5.15
references:
  - docs/research/laya-analysis.md
  - 'https://github.com/NandhaKishorM/laya'
  - 'https://huggingface.co/convaiinnovations/laya-typed-decisions'
  - >-
    https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions
documentation:
  - benchmarks/refund/README.md
  - benchmarks/refund/program/README.md
  - benchmarks/refund/program/CLAUDE_CLI_TRAINING.md
parent_task_id: TASK-5
ordinal: 16000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 1 exit criterion. The refund-decision expression from the transcript is the canonical task. Compare our ~200M model against 1B and 7B generative models using structured output and against a traditional LLM structured-output API. Metrics from transcript turn 9 point 10.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A held-out labeled test set for refund decision exists and is not used in training
- [ ] #2 Harness reports accuracy, calibration error, p50/p95 latency, throughput and memory for our model and each baseline
- [ ] #3 Results are committed under benchmarks/ with the exact model versions used
- [ ] #4 A written go/no-go against the exit criterion (p50 < 10ms, accuracy >= 7B baseline) is recorded
- [ ] #5 Laya (laya-typed-decisions checkpoint) is included as a baseline with the same inputs
- [ ] #6 Held-out set includes a slice of real, de-identified public transaction inputs whose labels were adjudicated by an independent model judge under a committed rubric (independent-judge origin with a judge attestation naming the model, session and rubric digest), and accuracy is reported separately on that attested slice
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a dedicated refund-benchmark workspace with closed versioned dataset, training-input-ledger, prediction, and result contracts. Bind cases with typed semantic-JSON digests, require sorted unique evaluation-only cases and explicit non-teacher human attestation, and hard-fail on overlap with every training/adversarial input.
2. Implement deterministic benchmark metrics and orchestration around a common adapter result: exact overall and human-slice accuracy, 15-bin top-1 ECE from complete support-ordered distributions, nearest-rank p50/p95, concurrency-1 throughput, process-tree/client-only memory scopes, immutable model/environment provenance, and a mechanical go/no-go that remains incomplete when a required system is missing.
3. Add the missing public trainer lifecycle builder that converts compiler source IR plus training/teacher/base-model provenance and passing VerificationResult into exact closed verified IR/source bytes; cover identity/count/version/timestamp failures so benchmark code does not hand-assemble production IR.
4. Add a benchmark-specific withConfidence refund source and reproducible compiler -> dataset/adversarial generation -> ROCm training -> calibration/verification -> artifact export -> Node diagnostic-runtime pipeline, with a tiny offline end-to-end fixture and isolated live runners. Benchmark the deployed CPU Node runtime honestly while recording GPU training separately.
5. Implement fixed-heldout-input adapters for the exact pinned local ~1B and ~7B Ollama models, a traditional structured-output API/Claude CLI reference, and the exact Laya laya-typed-decisions checkpoint. Require strict decision-plus-probability output for comparable ECE, stable argmax validation, timeouts, and immutable manifest/checkpoint/prompt identities.
6. Obtain a genuinely human-authored or licensed/de-identified held-out slice with human attestation, freeze it separately from training, run every required system on the same cases/hardware protocol, and commit predictions, machine-readable results, environment/version evidence, and a written exit-criterion decision under benchmarks/.
7. Run contract/metric/adapter/integration tests and memory-bounded repository gates, then independently audit dataset separation, baseline fairness/version pinning, metric math, and the reported go/no-go before finalization.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented and independently re-audited the complete benchmark/lifecycle harness, but not the real benchmark run.

Completed implementation:
- Closed versioned dataset, release-verification, training-ledger, prediction, result, and JSON Schema contracts with canonical semantic digests, cardinality limits, human non-teacher attestations, full lifecycle leakage checks, and exact canonical refund function/task binding.
- Deterministic metrics for overall and human-slice accuracy, 15-bin top-1 ECE, nearest-rank p50/p95, concurrency-1 throughput, sampled memory, and a mechanical incomplete/go/no-go decision.
- Reproducible compiler -> synthetic/adversarial data -> ROCm training -> calibration/release verification -> immutable artifact export -> scoped Node diagnostic runtime pipeline.
- Version-pinned live adapters for Ollama Qwen 2.5 1.5B and 7B Q4_K_M, Anthropic Sonnet 5 structured API, Laya typed decisions, and the SemantScript artifact. Prediction/result records enforce exact role identities and actual execution-backend evidence.
- Training and publication evidence bind the loaded manifest dataset and deterministic artifact training key to the canonical function, base/adversarial datasets, and exact release-verification payload and attestation. Final cases remain opaque to the lifecycle and warmups are separate, unique, and disjoint.
- Claude Code 2.1.278 training-only teacher pins claude-sonnet-5, disables tools/session state, validates strict schemas, bounds process groups/output/time/per-request spend, and cannot produce human or benchmark records.
- OOM fix: subprocess readers remain bounded until parent exit plus pipe EOF and terminate the entire process group on timeout, overflow, or cleanup. Client RSS sampling runs in an independent worker.

Pinned live evidence:
- ModernBERT base revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 completed a bounded AMD ROCm training smoke.
- Ollama 0.34.3 models are installed; both transport smokes reported 100 percent GPU. These smokes are not benchmark results.
- Laya code d120d4ba220711b93c171973118753460310e16b and checkpoint dd079950600224fb459af2a0cb1d74e1e57ee9cf completed one AMD ROCm smoke. This is not a benchmark result.

Latest validation:
- Independent remediation re-audit: no blockers.
- Full bounded repository check: all Node workspaces passed; Python 397 passed and 4 intentionally skipped; 40.90 seconds wall time; peak RSS 832,304 KiB; zero swap.

Handoff / remaining work, in order:
1. Obtain explicit count and total-spend authorization for a small Claude training-data pilot, then freeze the approved synthetic/adversarial training corpus. No paid Claude request has occurred.
2. Obtain two genuinely human-authored or licensed/de-identified, attested, mutually disjoint sets: one release-verification set and one final evaluation set. Neither may be generated or rewritten by a model.
3. Train on AMD ROCm, verify only on the release set, export the artifact, and preserve the derived ledger/training-key evidence.
4. Securely expose ANTHROPIC_API_KEY to the runner for the traditional API baseline; it is currently unset. Do not put secrets in Backlog or chat.
5. Run SemantScript, Ollama 1.5B, Ollama 7B, Anthropic structured API, and Laya against the exact final set under the same warmup/hardware/protocol record.
6. Commit cases, predictions, results, environment evidence, and written go/no-go under benchmarks; independently audit the claims; only then check acceptance criteria and mark Done.

Current status is intentionally In Progress. No real human datasets, benchmark predictions/results, or exit-criterion claim exist.

2026-09-23 resume: Claude CLI teacher pin moved from 2.1.278 to the installed 2.1.280 (Claude Code); default configuration SHA-256 is now d56782ca2714eef421da23d8b3bca83564dd8d31162111e2655fa55a214df51c (doc and test updated). First real request exposed a latent bug: run_bounded_process rejected any empty argv element, but the production command legitimately passes --tools "" and --setting-sources "" (the CLI documents --tools "" as disable-all). The injected FakeRunner tests never hit the real validator. Fixed by requiring only a nonempty argv[0]; added a pass-through test and a stub-executable test that drives the real runner with the production command. Focused suite 20 passed; ruff clean.

Live Sonnet 5 CLI teacher smoke (2026-09-23, not training data, not committed): after the pin/runner fixes, 3 synthetic cases, boundary pairs for both constraints, and 1 counterfactual all returned schema-valid structured output and passed the local parsers. Envelope facts from the pinned CLI 2.1.280: structured output arrives via a StructuredOutput tool call with stop_reason tool_use; num_turns was 2 or 3 per request, so the old exactly-one-turn check was wrong and now bounds turns at 8 while requiring modelUsage to name exactly claude-sonnet-5. Note the strict JSON loader returns integers as binary64 floats, so envelope checks must not use isinstance(int). Per request: 4-9 s wall, USD 0.015-0.023 list cost (auth is the claude.ai Max subscription, so CLI teacher spend is subscription quota, not API billing; ANTHROPIC_API_KEY remains unset). Quality flag: 2 of 3 synthetic cases were identical and matched the first smoke case (standard tier, 1 prior refund, 120-day paid USD 250 order -> deny); per-case prompting with a fixed system prompt gives low diversity and must be addressed before sizing the pilot.

Diversity probe (10 live synthetic cases, 52 s sequential, not training data): 8 unique inputs but 9 deny / 1 approve / 0 review; 9 of 10 used order.status paid and 8 of 10 used ageDays 120. The per-case prompt (fixed system prompt plus 'case i of N') makes Sonnet return near-mode cases, and _assemble_cases neither dedupes nor rejects duplicates, so a large pilot under the current prompt would yield a degenerate, deny-dominated corpus. This is a trainer prompt-design gap shared by the Anthropic SDK and Ollama teachers (Ollama also samples at temperature 0), outside this task's acceptance criteria; needs a scope decision before the pilot is sized.

2026-09-23 pilot started after TASK-5.15 closed: generate_training_corpus.py --synthetic-count 600 --counterfactual-ratio 0.25 --concurrency 4 --maximum-case-attempts 3 into benchmarks/refund/data/sonnet-pilot-2026-09-23 (synthetic-training-only; boundary pairs for both constraints plus ~150 counterfactuals). Expected roughly 750 Sonnet 5 CLI requests on the Max subscription. Results, manifest and digests will be committed when the run completes.

Pilot run 1 (2026-09-23 13:45-14:10): the 600-case synthetic phase completed and is cached under benchmarks/refund/data/sonnet-pilot-2026-09-23/synthetic (600/600 unique inputs; labels approve 96 / deny 305 / review 199; status paid 402 / fraudulent 198; tier enterprise 308 / standard 292; ageDays 0-214 with 47 at exactly 90 and 204 above; zero constraint violations; 15 approve labels outside the policy window flagged as teacher noise). The adversarial phase then aborted on one transient Claude CLI exit status 1 during a counterfactual request: the trainer adversarial generator re-raises TeacherTransportError without retry and the CLI teacher's boundary/counterfactual paths had no transport retry. A raw CLI request succeeded immediately afterwards. Fix in progress: transport retries with linear backoff in the CLI teacher adversarial paths and a --resume mode in generate_training_corpus.py that reloads the cached synthetic set and its saved run report. The synthetic run report for run 1 was lost with the process, so the pilot manifest will record runReport null for that phase.

Pilot corpus frozen and committed (2026-09-23, benchmarks/refund/data/sonnet-pilot-2026-09-23, manifest.json binds function nf_955824..., compiler bundle digest, teacher provenance claude-sonnet-5 via Claude Code 2.1.281, prompt contract v2, and both dataset digests). Synthetic: 600 cases, 600 unique inputs, labels approve 96 / deny 305 / review 199, zero constraint violations. Adversarial (resumed run, 41 min sequential): 4 constraint-boundary cases and 150 counterfactual pairs (304 cases); changed paths ageDays 121, status 28, priorRefunds 1; flip directions cover all six label transitions. Teacher-noise observations for the verification stage: 15 synthetic approve labels sit outside the policy window, and some fraudulent twins are labeled deny rather than review (constraint-valid, policy-debatable). Handoff step 1 is complete; remaining steps still need the two human-authored attested sets, ANTHROPIC_API_KEY for the structured-API baseline, the ROCm training run, and the benchmark runs.

2026-09-23 user decision (decision-5): human authorship replaced by independent-judge adjudication of real public inputs; acceptance criterion 5 rewritten accordingly (the former criterion 6 on Laya is now criterion 5, the new judge criterion is 6). Contract work tracked in TASK-5.16.

2026-09-23 evening: first two real release-pipeline attempts surfaced trainer defects on real data: (1) the canonical input serializer rejected IR input indices because the strict JSON reader yields binary64 floats (fixed, regression test); (2) the corpus assembler refused the frozen corpus because two counterfactual twins relabeled an existing synthetic input (fixed in the adversarial generator, which now rejects and retries label-conflicting proposals under generator contract v2; the adversarial set is being regenerated with --resume). Provenance field humanAuthoredVerification renamed to attestedVerification across IR/artifact contracts. run_release_pipeline.py (train, verify, bind, ledger, export, runtime smoke) and run-benchmark.mjs (per-system prediction sets with one shared environment and warmup protocol, plus result assembly) are committed. Ollama 1.5B/7B and Laya baselines are running now on the frozen final set; SemantScript runs after training; the structured-API baseline stays absent without ANTHROPIC_API_KEY, so the mechanical decision will read incomplete unless a key is provided.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
author: @codex
created: 2026-09-23 14:30
---
Started in ordinal order after TASK-5.8 completed all acceptance, security, and bounded repository gates. Beginning parallel reconnaissance on the human-authored refund dataset/metrics contract, executable SemantScript benchmark path, and fair version-pinned generative/Laya baseline adapters before recording the implementation plan.
---

author: @codex
created: 2026-09-23 15:51
---
Reconnaissance found the current runtime/export path is sound but the benchmark needs its own diagnostic source and independent evaluator; verification metrics cannot be reused because they include training rows. The only acceptance input not implementable autonomously is the genuinely human-authored held-out slice, which requires user authorship/review and attestation rather than AI generation.
---

author: @codex
created: 2026-09-23 15:55
---
Baseline reconnaissance: installed Ollama 0.34.3 currently has only glm-4.7-flash (19 GB), which is not a fair size-controlled substitute. Planned local baselines are qwen2.5 1.5B and 7B instruct Q4_K_M with resolved manifest/blob digests; Laya uses convaiinnovations/laya typed-decisions checkpoint with a pinned Hub commit; the traditional API run must use an immutable Sonnet model ID. Claude CLI 2.1.278 supports --model/--json-schema (the flag is --model, not --mdel), but paid/live calls and model pulls have not started.
---

author: @codex
created: 2026-09-23 15:56
---
Version pin follow-up: upstream Laya code HEAD is d120d4ba220711b93c171973118753460310e16b and convaiinnovations/laya-typed-decisions Hub HEAD is dd079950600224fb459af2a0cb1d74e1e57ee9cf as observed today. Anthropic documents claude-sonnet-5 as a canonical fixed model ID (4.6+ dateless IDs are pinned snapshots, not mutable aliases), so it is suitable as the traditional API model identity.
---

author: @codex
created: 2026-09-23 16:17
---
Verified-IR lifecycle builder integrated and independently revalidated: public build_verified_ir/VerifiedIrProvenance APIs now enforce exact immutable encoder revision, dataset/seed/model bindings, measured training counts, passing verification, and timestamp ordering before export. Focused lifecycle/artifact/public-API gate: 26 passed; Ruff lint/format green. Qwen 2.5 1.5B Q4_K_M pull completed with Ollama ID 65ec06548149; 7B Q4_K_M pull is in progress. No live inference or paid API calls yet.
---

author: @codex
created: 2026-09-23 16:18
---
Both local Ollama baselines are now installed and fully pinned. qwen2.5:1.5b-instruct-q4_K_M resolves to model blob sha256:183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4 (Ollama list ID 65ec06548149); qwen2.5:7b-instruct-q4_K_M resolves to model blob sha256:2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730 (list ID 845dbda0ea48). Ollama client/runtime is 0.34.3. These are provenance captures only; benchmark inference remains pending the frozen held-out dataset and completed harness.
---

author: @codex
created: 2026-09-23 16:19
---
Foundation checkpoint: contract and deterministic metric implementation is green under the focused workspace suite, strict ESLint, root TypeScript build, and all Node workspace tests. Human data/live results remain intentionally unclaimed.
---

author: @codex
created: 2026-09-23 16:28
---
GPU training checkpoint: the focused pinned ModernBERT ROCm smoke exposed and fixed a stale fixture that incorrectly expected two gold rows to create a held-out split. With one gold training row and one synthetic held-out row, the real fine-tune/evaluation smoke passed (1 test, 17.22s) under a 180s timeout. No full suite was run.
---

author: @codex
created: 2026-09-23 16:51
---
Independent integrated audit blocked live/result runs before any claims were made. Fixes now in progress: bind one canonical policy+constraints digest into every baseline prompt; keep the final benchmark set entirely out of train/calibration/release verification; derive Python release cases from a closed validated record rather than arbitrary SHA-shaped strings; derive the leakage ledger from actual lifecycle objects; bound subprocess output to prevent another OOM; add contract cardinality quotas and reject negative-zero prompt ambiguity. The benchmark runner itself is implemented and green offline, but no live or paid runs will start until these audit blockers close.
---

author: @codex
created: 2026-09-23 16:55
---
Pinned SemantScript base-model evidence captured after the successful ROCm smoke: answerdotai/ModernBERT-base revision 8949b909ec900327062f0ebf497f51aef5e6f0c8; model.safetensors 598,635,032 bytes, sha256 340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9; tokenizer.json sha256 9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1; config.json sha256 1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f. These are base artifact identities, not benchmark results.
---

author: @codex
created: 2026-09-23 16:59
---
Ollama provenance correction from full installed manifests: 1.5B manifest sha256 is 65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b and its model-weight blob is 183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4; 7B manifest sha256 is 845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e and its model-weight blob is 2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730. Both manifests also bind shared system 66b9ea09..., template eb440283..., and license 832dd9e0... layers. Adapter/live-transport remediation now distinguishes manifest revision from weight artifact and will re-resolve before inference.
---

author: @codex
created: 2026-09-23 17:06
---
Pinned Laya checkpoint is now cached at Hub revision dd079950600224fb459af2a0cb1d74e1e57ee9cf (808 MiB resolved snapshot). Exact file evidence: model.safetensors sha256 4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e; rl_agent_config.json ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8; encoder/config.json 5268d24ad3b77c8151de5dcb0762ba4391619aad9ab0bda33e36fb083cfeae6d; tokenizer/tokenizer.json 6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30. Code remains pinned separately to d120d4ba220711b93c171973118753460310e16b (Laya 0.3.9). No inference/result claim yet.
---

author: @codex
created: 2026-09-23 17:12
---
Laya live-path smoke (not a benchmark result) passed on AMD ROCm/CUDA with pinned code d120d4b and checkpoint dd079950...: local laya.load(..., device='cuda', fast=False) reported device cuda and returned a complete calibrated three-choice distribution in 13.9s including process/model cold start. Upstream Laya emitted its own warning that the checkpoint's choice:11+ temperature is clamped as uncalibrated; our benchmark uses 3 options but will preserve this warning/evidence. The single smoke prediction is intentionally not committed or interpreted as accuracy/latency.
---

created: 2026-09-23 17:25
---
Offline/live harness checkpoint: final benchmark inputs are now structurally excluded from training and release verification; the release gate is a distinct digest-bound human record; the training ledger is derived from actual source/base/adversarial/split/release artifacts; compiler/runtime output is incrementally capped and process groups are terminated on quota/timeout. Version-pinned Ollama, Anthropic, and persistent offline Laya transports plus the persistent SemantScript runtime adapter and client-only RSS/monotonic measurement helpers are implemented. Root-integrated refund workspace gate is green (10/10 Node test files), focused lifecycle tests are 12/12 green, and Python lint now covers the Laya worker. No benchmark predictions/results or acceptance claims were created; real disjoint human release/final cases remain required.
---

created: 2026-09-23 18:31
---
Hardening and OOM-recovery checkpoint: diagnosed the runaway subprocess pattern as a direct parent exiting while a descendant retained captured pipes; bounded readers now keep the timeout active through EOF and terminate the full process group on timeout/quota/finalization. Publication now fixes the canonical function/task and every system identity; keeps warmups unique, disjoint, and separate from final cases; binds the loaded artifact dataset and deterministic training key to the base/adversarial datasets plus exact release-verification payload/attestation; enforces role pins in runtime and JSON Schema; records actual per-system execution backend/device (including Ollama observed CPU/GPU byte split); uses scoped artifact-handle calls; and samples client RSS from an independent worker. Both pinned Ollama models passed non-benchmark transport smokes at observed 100% GPU; these are not results. Claude Sonnet 5 training-only CLI integration is ready but no paid call/data generation occurred. Independent remediation re-audit found no blockers. Final bounded repository gate passed in 40.90s with peak RSS 832,304 KiB and zero swap: all Node workspaces green; Python 397 passed, 4 intentionally skipped. TASK-5.11 remains In Progress with all acceptance criteria unchecked because two genuinely human-authored, de-identified, disjoint release/final sets, the traditional Anthropic API run, committed predictions/results, and the written go/no-go are still outstanding.
---

author: @claude
created: 2026-09-23 19:12
---
Resumed on handoff step 1 (Sonnet training-data pilot). Preflight found the Claude CLI teacher pins 2.1.278 but the installed CLI is 2.1.280 (Claude Code), so verify_installation would reject every request. Updating the pin, its configuration digest, tests, and CLAUDE_CLI_TRAINING.md, then running a single-case live smoke through ClaudeCliTrainingTeacher against the compiled canonical refund IR. Auth is the claude.ai Max subscription (CLI auth status), not a per-token API key.
---

author: @claude
created: 2026-09-23 19:24
---
Repository Python gate after the CLI teacher fixes: ruff lint/format clean across 64 files; pytest 400 passed, 4 intentionally skipped, 14.6 s. No Node changes were made in this slice. The full task remains In Progress: no training corpus, human sets, benchmark predictions, or go/no-go exist. Pilot count and the synthetic-diversity gap need a user decision before generation starts.
---

author: @claude
created: 2026-09-23 20:54
---
Handoff step 1 done: approved-by-user pilot corpus generated and frozen after fixing the CLI teacher (version pin, empty argv, envelope protocol, transport retries) and the prompt diversity gap (TASK-5.15). Next steps are user-gated: authoring the two disjoint human-attested sets (release verification and final evaluation) and providing an API key for the traditional structured-output baseline. Training and the benchmark runs can proceed once the release set exists.
---
<!-- COMMENTS:END -->
