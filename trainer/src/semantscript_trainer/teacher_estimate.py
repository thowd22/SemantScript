"""``semantscript train --estimate``: what a run's teacher will cost, without calling it.

For every expression of a bundle the estimate counts the teacher requests the
run would send, the input and output tokens, the USD cost at the configured
backend's price and the expected wall time. It builds no client, sends no
request and loads no tokenizer or encoder.

* **Requests.** An expression whose datasets are already cached costs nothing.
  Otherwise the synthetic dataset asks for ``cases - gold`` cases, one per
  request; an expression with constraints adds one boundary-pair request per
  constraint and one counterfactual-twin request per selected anchor
  (``--counterfactual-ratio`` of the synthetic cases). The **expected** count
  multiplies a constrained expression's requests by ``EXPECTED_RETRY_FACTOR``
  (label replacements, repeated boundary and twin attempts, skipped anchors),
  calibrated on the 2026-09-25 Express run (about 600 requests where 479 were
  planned, every extra on the constrained expression). The **maximum** is the
  bound the generators enforce: three replacement rounds, ``maximum_attempts``
  per boundary pair and per anchor, and as many skipped anchors as pairs.
  A mixed constraints teacher sends only the share its constraints leave open
  (its seeded pilot sample).
* **Tokens.** Input tokens are the characters of the exact prompts the teacher
  builds (``teacher_prompt`` and ``adversarial_prompt``) plus the response
  schema, divided by ``teacher_spend.CHARACTERS_PER_TOKEN``; output tokens are
  the size of a serialized case (twice for a pair, plus a reason for a twin),
  divided by the same ratio. With direct Anthropic requests the schema and the
  system prompt are served from the prompt cache after the first request of each
  kind when they reach the model's minimum cacheable length.
* **Time.** Requests times ``seconds_per_request``: the ``[teacher.pricing]``
  figure, else the mean the last metered run of this teacher recorded, else a
  pinned default. Message Batches requests do not run one after another: each
  batch adds ``BATCH_EXPECTED_SECONDS`` (one hour; Anthropic's documentation says
  most batches finish within an hour) to the expected time, and the maximum time
  adds ``poll_timeout_seconds`` (24 hours by default, the longest the run waits)
  for every batch the run can submit: the first synthetic batch and, for a
  constrained expression, one per label-replacement round.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from semantscript_trainer.adversarial import (
    AdversarialDatasetGenerator,
    AdversarialGenerationConfig,
    _ratio_count,
)
from semantscript_trainer.adversarial_contract import (
    build_boundary_pair_schema,
    build_counterfactual_schema,
)
from semantscript_trainer.adversarial_prompt import (
    build_boundary_messages,
    build_counterfactual_messages,
)
from semantscript_trainer.case_contract import build_case_schema
from semantscript_trainer.dataset import MAXIMUM_REPLACEMENT_ROUNDS, SyntheticDatasetGenerator
from semantscript_trainer.teacher import GeneratedCase, NeuralFunctionIr
from semantscript_trainer.teacher_config import (
    AnyTeacherConfig,
    ConstraintsTeacherConfig,
    create_teacher,
)
from semantscript_trainer.teacher_prompt import build_case_messages
from semantscript_trainer.teacher_spend import (
    CHARACTERS_PER_TOKEN,
    Fetcher,
    TeacherPrice,
    language_model_config,
    minimum_cacheable_tokens,
    resolve_price,
    seconds_per_request,
    tokens_for_characters,
)

ESTIMATE_KIND = "semantscript.train-estimate"
ESTIMATE_VERSION = 1
EXPECTED_RETRY_FACTOR = 1.4
TWIN_REASON_CHARACTERS = 200
BATCH_EXPECTED_SECONDS = 3600.0


def estimate_bundle(
    functions: Sequence[NeuralFunctionIr],
    config: AnyTeacherConfig,
    *,
    cache_directory: str | Path,
    cases: int,
    adversarial_config: AdversarialGenerationConfig | None = None,
    use_cache: bool = True,
    fetch: Fetcher | None = None,
) -> dict[str, Any]:
    """The ``semantscript.train-estimate`` document for ``functions`` (see the module
    docstring). Never sends a teacher request."""

    adversarial = adversarial_config or AdversarialGenerationConfig()
    cache_root = Path(cache_directory)
    teacher = create_teacher(config)
    model_config = language_model_config(config)
    price = resolve_price(config, cache_directory=cache_root, fetch=fetch)
    seconds, seconds_source = seconds_per_request(config, cache_directory=cache_root)
    rows = [
        _estimate_function(
            ir,
            config,
            teacher,
            price,
            seconds,
            cache_root=cache_root,
            cases=cases,
            adversarial=adversarial,
            use_cache=use_cache,
        )
        for ir in functions
    ]
    total_keys = (
        "expectedRequests",
        "maximumRequests",
        "batchRequests",
        "inputTokens",
        "cacheReadTokens",
        "outputTokens",
        "costUsd",
        "maximumCostUsd",
        "seconds",
        "batchSeconds",
        "maximumSeconds",
    )
    total = {key: sum(row[key] for row in rows) for key in total_keys}
    for key in ("costUsd", "maximumCostUsd", "seconds", "batchSeconds", "maximumSeconds"):
        total[key] = round(total[key], 4)
    return {
        "kind": ESTIMATE_KIND,
        "estimateVersion": ESTIMATE_VERSION,
        "teacher": {
            "backend": "constraints"
            if isinstance(config, ConstraintsTeacherConfig)
            else cast(Any, config).backend,
            "model": None if model_config is None else model_config.model,
            "fallback": isinstance(config, ConstraintsTeacherConfig) and model_config is not None,
            "mode": None if model_config is None else model_config.mode,
        },
        "price": price.to_json(),
        "secondsPerRequest": seconds,
        "secondsSource": seconds_source,
        "charactersPerToken": CHARACTERS_PER_TOKEN,
        "expectedRetryFactor": EXPECTED_RETRY_FACTOR,
        "cases": cases,
        "counterfactualRatio": adversarial.counterfactual_ratio,
        "functions": rows,
        "total": total,
    }


def _estimate_function(
    ir: NeuralFunctionIr,
    config: AnyTeacherConfig,
    teacher: Any,
    price: TeacherPrice,
    seconds: float,
    *,
    cache_root: Path,
    cases: int,
    adversarial: AdversarialGenerationConfig,
    use_cache: bool,
) -> dict[str, Any]:
    definition = cast(dict[str, Any], ir["definition"])
    examples = cast(list[Any], definition.get("examples") or [])
    constraints = cast(list[Any], definition.get("constraints") or [])
    gold = len(examples)
    total = max(cases, gold)
    synthetic = total - gold
    source = cast(dict[str, Any], ir.get("source") or {})

    dataset_cached = False
    adversarial_cached: bool | None = None if not constraints else False
    if use_cache:
        generator = SyntheticDatasetGenerator(teacher, cache_root)
        dataset_cached = generator.cache_path(ir, total).is_file()
        if dataset_cached and constraints:
            # The adversarial cache key needs the base dataset, which is on disk.
            base = generator.generate(ir, total)
            adversarial_cached = (
                AdversarialDatasetGenerator(teacher, cache_root, config=adversarial)
                .cache_path(ir, base)
                .is_file()
            )

    model_config = language_model_config(config)
    synthetic_requests = 0 if dataset_cached else synthetic
    boundary_requests = 0 if adversarial_cached is not False else len(constraints)
    twin_requests = (
        0
        if adversarial_cached is not False
        else _ratio_count(synthetic, float(adversarial.counterfactual_ratio))
    )
    open_share = 1.0
    if isinstance(config, ConstraintsTeacherConfig):
        # Pure: no request at all. Mixed: the fallback gets what the constraints leave open.
        open_share = 0.0 if model_config is None else 1.0 - teacher.decided_share(ir)
    row: dict[str, Any] = {
        "id": ir["id"],
        "sourcePath": source.get("path"),
        "cached": {"dataset": dataset_cached, "adversarial": adversarial_cached},
        "plannedRequests": {
            "synthetic": math.ceil(synthetic_requests * open_share),
            "boundary": math.ceil(boundary_requests * open_share),
            "counterfactual": math.ceil(twin_requests * open_share),
        },
    }
    planned = row["plannedRequests"]
    planned_total = planned["synthetic"] + planned["boundary"] + planned["counterfactual"]
    factor = EXPECTED_RETRY_FACTOR if constraints else 1.0
    maximum = (
        planned["synthetic"] * (1 + MAXIMUM_REPLACEMENT_ROUNDS if constraints else 1)
        + planned["boundary"] * adversarial.maximum_attempts
        + planned["counterfactual"] * 2 * adversarial.maximum_attempts
    )
    zero = {
        "expectedRequests": 0,
        "maximumRequests": 0,
        "batchRequests": 0,
        "inputTokens": 0,
        "cacheReadTokens": 0,
        "outputTokens": 0,
        "costUsd": 0.0,
        "maximumCostUsd": 0.0,
        "seconds": 0.0,
        "batchSeconds": 0.0,
        "maximumSeconds": 0.0,
    }
    if planned_total == 0 or model_config is None:
        return {**row, **zero}

    anchor = _anchor(examples)
    case_output = len(_compact({"inputs": anchor.inputs, "output": anchor.output}))
    kinds: list[tuple[str, int, list[tuple[str, str]], str, int]] = []
    if planned["synthetic"]:
        kinds.append(
            (
                "synthetic",
                planned["synthetic"],
                [build_case_messages(ir, index, synthetic) for index in range(synthetic)],
                _compact(build_case_schema(ir)),
                case_output,
            )
        )
    if planned["boundary"]:
        kinds.append(
            (
                "boundary",
                planned["boundary"],
                [build_boundary_messages(ir, index) for index in range(len(constraints))],
                _compact(build_boundary_pair_schema(ir)),
                2 * case_output,
            )
        )
    if planned["counterfactual"]:
        kinds.append(
            (
                "counterfactual",
                planned["counterfactual"],
                [build_counterfactual_messages(ir, anchor)],
                _compact(build_counterfactual_schema(ir)),
                case_output + TWIN_REASON_CHARACTERS,
            )
        )

    batch_synthetic = model_config.backend == "anthropic" and (
        model_config.mode == "batch"
        or (model_config.mode == "auto" and planned["synthetic"] >= model_config.batch_threshold)
    )
    caches = model_config.backend == "anthropic"
    minimum_cache = minimum_cacheable_tokens(model_config.model)
    expected_requests = 0
    batch_requests = 0
    input_tokens = cache_read = output_tokens = 0
    cost = 0.0
    for kind, count, messages, schema, output_characters in kinds:
        requests = math.ceil(count * factor)
        expected_requests += requests
        batch = kind == "synthetic" and batch_synthetic
        if batch:
            batch_requests += requests
        # The cacheable prefix is the response schema plus the system prompt (the
        # instructions and the compact contract); the user message varies per request.
        system_tokens = sum(
            tokens_for_characters(len(system) + len(schema)) for system, _ in messages
        ) // len(messages)
        user_tokens = sum(tokens_for_characters(len(user)) for _, user in messages) // len(messages)
        per_output = tokens_for_characters(output_characters)
        cached = caches and not batch and system_tokens >= minimum_cache and requests > 1
        reads = (requests - 1) * system_tokens if cached else 0
        writes = system_tokens if cached else 0
        uncached = requests * (system_tokens + user_tokens) - reads - writes
        input_tokens += uncached + reads + writes
        cache_read += reads
        output_tokens += requests * per_output
        cost += price.cost(
            uncached,
            requests * per_output,
            cache_read_tokens=reads,
            cache_write_tokens=writes,
            batch=batch,
        )
    direct = expected_requests - batch_requests
    direct_seconds = direct * seconds
    batch_seconds = BATCH_EXPECTED_SECONDS if batch_requests else 0.0
    if batch_requests:
        batches = 1 + MAXIMUM_REPLACEMENT_ROUNDS if constraints else 1
        # The maximum direct time: every request past the synthetic ones is direct.
        maximum_direct = (maximum - planned["synthetic"]) * seconds
        maximum_seconds = maximum_direct + batches * model_config.poll_timeout_seconds
    else:
        maximum_seconds = maximum * seconds
    return {
        **row,
        "expectedRequests": expected_requests,
        "maximumRequests": maximum,
        "batchRequests": batch_requests,
        "inputTokens": input_tokens,
        "cacheReadTokens": cache_read,
        "outputTokens": output_tokens,
        "costUsd": round(cost, 4),
        "maximumCostUsd": round(cost * maximum / expected_requests, 4),
        "seconds": round(direct_seconds + batch_seconds, 1),
        "batchSeconds": batch_seconds,
        "maximumSeconds": round(maximum_seconds, 1),
    }


def _anchor(examples: Sequence[Any]) -> GeneratedCase:
    for example in examples:
        if isinstance(example, Mapping) and isinstance(example.get("inputs"), dict):
            return GeneratedCase(inputs=example["inputs"], output=example.get("output"))
    return GeneratedCase(inputs={}, output=None)


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


__all__ = [
    "BATCH_EXPECTED_SECONDS",
    "ESTIMATE_KIND",
    "ESTIMATE_VERSION",
    "EXPECTED_RETRY_FACTOR",
    "estimate_bundle",
]
