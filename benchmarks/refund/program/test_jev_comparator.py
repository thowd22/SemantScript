"""Offline tests for the Jev comparator: request shape, scoring and step agreement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.refund.program.run_jev_comparator import (
    CRITERIA,
    SUPPORT,
    JevComparatorError,
    agreement_by_step,
    build_request,
    parse_answer,
    run,
    score,
)

POLICY = {"policy": "the policy", "hardConstraints": ["rule one"], "taskSpecSha256": "f" * 64}
CASES = [
    {
        "id": "case-1",
        "inputSha256": "1" * 64,
        "expected": "approve",
        "origin": "independent-judge",
        "inputs": {
            "customer": {"priorRefunds": 0, "tier": "standard"},
            "order": {"ageDays": 3, "status": "paid", "total": 50},
        },
    },
    {
        "id": "case-2",
        "inputSha256": "2" * 64,
        "expected": "deny",
        "origin": "independent-judge",
        "inputs": {
            "customer": {"priorRefunds": 1, "tier": "standard"},
            "order": {"ageDays": 120, "status": "paid", "total": 50},
        },
    },
]
ADJUDICATIONS = {"cases": {"case-1": {"step": 5}, "case-2": {"step": 1}}}


def answer(choice: str, probabilities: dict[str, float], cost: float = 1e-5) -> dict:
    return {
        "model": "typesafe/jev-test",
        "answers": {
            "decision": {
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities,
                "confidence": 0.9,
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 5, "cost": cost},
    }


def test_request_carries_the_policy_and_the_canonical_fields_only() -> None:
    request = build_request(POLICY, CASES[0]["inputs"])
    assert request["model"] == "~typesafe/jev-latest"
    assert request["state"]["policy"] == "the policy"
    assert request["state"]["hardConstraints"] == ["rule one"]
    assert request["state"]["customer"] == {"priorRefunds": 0, "tier": "standard"}
    assert request["state"]["order"] == {"ageDays": 3, "status": "paid", "total": 50}
    assert request["questions"]["decision"]["type"] == "choice"
    assert request["questions"]["decision"]["criteria"] == CRITERIA
    assert tuple(CRITERIA) == SUPPORT


def test_parse_answer_orders_the_distribution_by_support() -> None:
    choice, distribution, confidence, usage = parse_answer(
        answer("deny", {"deny": 0.7, "approve": 0.2})
    )
    assert choice == "deny"
    assert distribution == {"approve": 0.2, "deny": 0.7, "review": 0.0}
    assert confidence == 0.9
    assert usage["cost"] == 1e-5
    with pytest.raises(JevComparatorError):
        parse_answer(
            {"answers": {"decision": {"type": "choice", "choice": "maybe", "probabilities": {}}}}
        )


def test_score_matches_the_benchmark_definitions() -> None:
    predictions = [
        {
            "caseId": "case-1",
            "value": "approve",
            "distribution": [
                {"value": v, "probability": p}
                for v, p in (("approve", 0.9), ("deny", 0.05), ("review", 0.05))
            ],
            "latencyMs": 10.0,
        },
        {
            "caseId": "case-2",
            "value": "review",
            "distribution": [
                {"value": v, "probability": p}
                for v, p in (("approve", 0.1), ("deny", 0.3), ("review", 0.6))
            ],
            "latencyMs": 30.0,
        },
    ]
    metrics = score(CASES, predictions)
    assert metrics["accuracy"]["correctCount"] == 1
    assert metrics["accuracy"]["accuracy"] == 0.5
    # bins: 0.9 -> bin 13 (correct), 0.6 -> bin 9 (wrong); ECE = 0.5*|1-0.9| + 0.5*|0-0.6|
    assert metrics["calibration"]["expectedCalibrationError"] == pytest.approx(0.35)
    assert metrics["latency"] == {"p50Ms": 10.0, "p95Ms": 30.0}
    steps = agreement_by_step(CASES, predictions, ADJUDICATIONS)
    assert steps["5"]["agreement"] == 1.0
    assert steps["1"] == {
        "cases": 1,
        "agree": 0,
        "confusions": {"deny->review": 1},
        "agreement": 0.0,
    }


def test_run_caches_raw_answers_and_stops_at_the_budget(tmp_path: Path) -> None:
    calls: list[dict] = []

    def transport(request: dict) -> tuple[dict, float]:
        calls.append(request)
        expected = next(
            c["expected"]
            for c in CASES
            if c["inputs"]
            == {"customer": request["state"]["customer"], "order": request["state"]["order"]}
        )
        return answer(expected, {expected: 1.0}), 12.5

    dataset = {"cases": CASES, "payloadSha256": "d" * 64}
    results = run(
        dataset=dataset,
        adjudications=ADJUDICATIONS,
        policy=POLICY,
        transport=transport,
        output_dir=tmp_path,
        max_cost_usd=1.0,
        log=lambda _m: None,
    )
    assert len(calls) == 2
    assert results["metrics"]["accuracy"]["accuracy"] == 1.0
    assert results["usage"]["costUsd"] == pytest.approx(2e-5)
    assert results["usage"]["estimatedCostFor10kCasesUsd"] == pytest.approx(0.1)
    assert json.loads((tmp_path / "raw-answers.json").read_text()).keys() == {"case-1", "case-2"}
    # A rerun re-spends nothing.
    run(
        dataset=dataset,
        adjudications=ADJUDICATIONS,
        policy=POLICY,
        transport=transport,
        output_dir=tmp_path,
        max_cost_usd=1.0,
        log=lambda _m: None,
    )
    assert len(calls) == 2
    # An exhausted budget stops before the next request.
    (tmp_path / "raw-answers.json").write_text(
        json.dumps(
            {
                "case-1": {
                    "response": answer("approve", {"approve": 1.0}, cost=5.0),
                    "latencyMs": 1.0,
                    "usage": {"cost": 5.0},
                }
            }
        )
    )
    with pytest.raises(JevComparatorError, match="exceeds"):
        run(
            dataset=dataset,
            adjudications=ADJUDICATIONS,
            policy=POLICY,
            transport=transport,
            output_dir=tmp_path,
            max_cost_usd=1.0,
            log=lambda _m: None,
        )
