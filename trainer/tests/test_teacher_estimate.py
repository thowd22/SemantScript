from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from semantscript_trainer.adversarial import AdversarialGenerationConfig
from semantscript_trainer.dataset import SyntheticDatasetGenerator
from semantscript_trainer.teacher_config import (
    ConstraintsTeacherConfig,
    TeacherConfig,
    create_teacher,
)
from semantscript_trainer.teacher_estimate import (
    BATCH_EXPECTED_SECONDS,
    ESTIMATE_KIND,
    EXPECTED_RETRY_FACTOR,
    estimate_bundle,
)
from semantscript_trainer.teachers.anthropic import AnthropicTeacher

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def refund_ir() -> dict[str, Any]:
    return json.loads(
        (REPOSITORY_ROOT / "examples" / "ir" / "refund-decision.v1.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(autouse=True)
def no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(self: Any) -> Any:
        raise AssertionError("the estimate must not build a teacher client")

    monkeypatch.setattr(AnthropicTeacher, "_get_client", refuse)


def test_counts_requests_tokens_cost_and_time_for_a_constrained_expression(
    tmp_path: Path,
) -> None:
    ir = refund_ir()
    constraints = len(ir["definition"]["constraints"])
    gold = len(ir["definition"]["examples"])
    config = TeacherConfig(backend="anthropic", model="claude-sonnet-5", mode="direct")

    estimate = estimate_bundle(
        [ir],
        config,
        cache_directory=tmp_path,
        cases=64,
        adversarial_config=AdversarialGenerationConfig(counterfactual_ratio=0.5),
    )

    assert estimate["kind"] == ESTIMATE_KIND and estimate["estimateVersion"] == 1
    assert estimate["price"]["source"].startswith("pinned Anthropic list price")
    row = estimate["functions"][0]
    synthetic = 64 - gold
    twins = math.floor(synthetic * 0.5 + 0.5)
    assert row["plannedRequests"] == {
        "synthetic": synthetic,
        "boundary": constraints,
        "counterfactual": twins,
    }
    assert row["expectedRequests"] == sum(
        math.ceil(count * EXPECTED_RETRY_FACTOR) for count in (synthetic, constraints, twins)
    )
    assert row["maximumRequests"] == synthetic * 4 + constraints * 3 + twins * 2 * 3
    assert row["inputTokens"] > row["cacheReadTokens"] > 0
    assert row["outputTokens"] > 0
    assert 0 < row["costUsd"] < row["maximumCostUsd"]
    assert row["seconds"] == pytest.approx(row["expectedRequests"] * 4.0)
    assert estimate["total"]["costUsd"] == pytest.approx(row["costUsd"])


def test_the_reduced_prompt_costs_under_half_the_measured_request_price(tmp_path: Path) -> None:
    config = TeacherConfig.from_mapping(
        {
            "backend": "anthropic",
            "model": "claude-sonnet-5",
            "mode": "direct",
            "pricing": {"input_usd_per_million": 2, "output_usd_per_million": 10},
        }
    )
    estimate = estimate_bundle([refund_ir()], config, cache_directory=tmp_path, cases=64)
    total = estimate["total"]
    # The 2026-09-25 Express run paid about USD 0.017 per request.
    assert total["costUsd"] / total["expectedRequests"] < 0.017 / 2


def test_batch_mode_prices_synthetic_requests_at_the_batch_rate(tmp_path: Path) -> None:
    ir = refund_ir()
    direct = estimate_bundle(
        [ir],
        TeacherConfig(backend="anthropic", model="claude-sonnet-5", mode="direct"),
        cache_directory=tmp_path,
        cases=64,
    )["functions"][0]
    batch = estimate_bundle(
        [ir],
        TeacherConfig(backend="anthropic", model="claude-sonnet-5", mode="auto"),
        cache_directory=tmp_path,
        cases=64,
    )["functions"][0]
    assert batch["batchRequests"] == math.ceil(63 * EXPECTED_RETRY_FACTOR)
    # A batch has an expected wall time (an hour) and a maximum (the poll timeout
    # for the first batch and each replacement round), not a per-request figure.
    direct_requests = batch["expectedRequests"] - batch["batchRequests"]
    assert batch["batchSeconds"] == BATCH_EXPECTED_SECONDS
    assert batch["seconds"] == pytest.approx(direct_requests * 4.0 + BATCH_EXPECTED_SECONDS)
    assert batch["maximumSeconds"] > 4 * 86_400
    assert direct["batchSeconds"] == 0
    assert direct["maximumSeconds"] == pytest.approx(direct["maximumRequests"] * 4.0)


def test_the_constraints_teacher_and_cached_datasets_cost_nothing(tmp_path: Path) -> None:
    constraints = estimate_bundle(
        [refund_ir()], ConstraintsTeacherConfig(), cache_directory=tmp_path, cases=64
    )
    assert constraints["total"]["expectedRequests"] == 0
    assert constraints["total"]["costUsd"] == 0
    assert constraints["price"]["source"].startswith("free")

    ir = refund_ir()
    ir["definition"]["constraints"] = []
    config = TeacherConfig(backend="anthropic", model="claude-sonnet-5")
    cache_path = SyntheticDatasetGenerator(create_teacher(config), tmp_path).cache_path(ir, 64)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("{}")
    cached = estimate_bundle([ir], config, cache_directory=tmp_path, cases=64)["functions"][0]
    assert cached["cached"] == {"dataset": True, "adversarial": None}
    assert (cached["expectedRequests"], cached["costUsd"]) == (0, 0.0)

    fresh = estimate_bundle([ir], config, cache_directory=tmp_path, cases=64, use_cache=False)
    assert fresh["functions"][0]["expectedRequests"] == 63


def test_ollama_is_free_but_takes_time(tmp_path: Path) -> None:
    ir = copy.deepcopy(refund_ir())
    estimate = estimate_bundle(
        [ir], TeacherConfig(backend="ollama", model="qwen3:14b"), cache_directory=tmp_path, cases=8
    )
    total = estimate["total"]
    assert total["costUsd"] == 0 and total["expectedRequests"] > 0 and total["seconds"] > 0
    assert total["cacheReadTokens"] == 0
