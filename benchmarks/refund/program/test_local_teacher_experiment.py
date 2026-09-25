"""Offline tests for the local-teacher experiment: sampling, prompts, agreement, caching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.refund.program.run_local_teacher_experiment import (
    SUPPORT,
    ExperimentError,
    ReplayTeacher,
    agreement,
    build_prompt,
    canonical_input,
    label_rows,
    rubric_step,
    sample_inputs,
)

from semantscript_trainer.semantic_json import semantic_json_sha256

POLICY = {
    "policy": "the policy",
    "hardConstraints": ["rule one", "rule two"],
    "taskSpecSha256": "f" * 64,
}


def inputs(prior: int, tier: str, age: int, status: str, total: float) -> dict:
    return {
        "customer": {"priorRefunds": prior, "tier": tier},
        "order": {"ageDays": age, "status": status, "total": total},
    }


def test_sampling_excludes_held_out_digests_and_duplicates_deterministically() -> None:
    rows = [{"inputs": inputs(i % 3, "standard", i, "paid", 10.0 + i)} for i in range(12)]
    rows.append({"inputs": inputs(0, "standard", 0, "paid", 10.0)})  # duplicate of row 0
    excluded = frozenset({semantic_json_sha256(json.loads(canonical_input(rows[1]["inputs"])))})
    evaluation, training, counts = sample_inputs(rows, excluded, evaluation=3, training=5, seed=7)
    again = sample_inputs(rows, excluded, evaluation=3, training=5, seed=7)
    assert (evaluation, training) == again[:2]
    assert counts == {"poolRows": 13, "excludedHeldOut": 1, "duplicateInputs": 1, "usable": 11}
    digests = {row["inputSha256"] for row in evaluation + training}
    assert len(digests) == 8 and not digests & excluded
    with pytest.raises(ExperimentError, match="fewer"):
        sample_inputs(rows, excluded, evaluation=6, training=6, seed=1)


def test_prompt_carries_the_policy_constraints_and_canonical_input() -> None:
    system, user = build_prompt(POLICY, inputs(2, "enterprise", 40, "paid", 99.5))
    assert "structured decision" in system
    assert "Policy: the policy." in user
    assert "1. rule one\n2. rule two" in user
    assert user.endswith(
        '{"customer":{"priorRefunds":2,"tier":"enterprise"},"order":{"ageDays":40,"status":"paid","total":99.5}}'
    )


def test_rubric_steps_follow_the_adjudication_rubric() -> None:
    assert rubric_step(inputs(0, "standard", 91, "fraudulent", 5)) == "1"
    assert rubric_step(inputs(0, "standard", 10, "fraudulent", 5)) == "2"
    assert rubric_step(inputs(0, "enterprise", 61, "paid", 5)) == "3"
    assert rubric_step(inputs(0, "standard", 31, "paid", 5)) == "3"
    assert rubric_step(inputs(5, "standard", 10, "paid", 5)) == "4"
    assert rubric_step(inputs(3, "standard", 10, "paid", 1000)) == "4"
    assert rubric_step(inputs(2, "standard", 1, "paid", 2000)) == "4"
    assert rubric_step(inputs(2, "standard", 2, "paid", 2000)) == "5"
    assert rubric_step(inputs(0, "enterprise", 34, "paid", 252.2)) == "5"


class _Rules:
    def __init__(self, table: dict[str, str | None]) -> None:
        self._table = table

    def label(self, row: dict) -> str | None:
        return self._table[canonical_input(row)]


def test_agreement_counts_pairs_and_rule_matches_per_step() -> None:
    rows = [
        {"inputSha256": "a", "inputs": inputs(0, "standard", 100, "paid", 5)},  # step 1
        {"inputSha256": "b", "inputs": inputs(0, "standard", 5, "paid", 5)},  # step 5
    ]
    sonnet = {"a": {"label": "deny"}, "b": {"label": "approve"}}
    qwen = {"a": {"label": "deny"}, "b": {"label": "review"}}
    rules = _Rules(
        {canonical_input(rows[0]["inputs"]): "deny", canonical_input(rows[1]["inputs"]): None}
    )
    result = agreement(rows, sonnet, qwen, rules)
    assert result["totals"] == {
        "rows": 2,
        "sonnetQwen": 1,
        "sonnetRule": 1,
        "qwenRule": 1,
        "ruleAmbiguous": 1,
        "sonnetQwenRate": 0.5,
        "sonnetRuleRate": 0.5,
        "qwenRuleRate": 0.5,
    }
    assert result["bySteps"]["1"]["sonnetQwen"] == 1 and result["bySteps"]["5"]["sonnetQwen"] == 0
    assert result["disagreements"] == {"sonnet approve / qwen review": 1}


def test_label_rows_caches_and_enforces_the_budget(tmp_path: Path) -> None:
    calls: list[str] = []

    def labeler(system: str, user: str) -> tuple[str, dict]:
        calls.append(user)
        return "review", {"costUsd": 0.5}

    rows = [
        {"inputSha256": "a", "inputs": inputs(0, "standard", 5, "paid", 5)},
        {"inputSha256": "b", "inputs": inputs(1, "standard", 5, "paid", 5)},
    ]
    cache = label_rows(
        rows, labeler, POLICY, tmp_path / "labels.json", name="test", max_cost_usd=2.0
    )
    assert set(cache) == {"a", "b"} and cache["a"]["label"] == "review"
    label_rows(rows, labeler, POLICY, tmp_path / "labels.json", name="test", max_cost_usd=2.0)
    assert len(calls) == 2  # cached
    more = [*rows, {"inputSha256": "c", "inputs": inputs(2, "standard", 5, "paid", 5)}]
    with pytest.raises(ExperimentError, match="exceeds"):
        label_rows(more, labeler, POLICY, tmp_path / "labels.json", name="test", max_cost_usd=0.9)

    def bad(system: str, user: str) -> tuple[str, dict]:
        return "maybe", {}

    with pytest.raises(ExperimentError, match="outside the support"):
        label_rows(more, bad, POLICY, tmp_path / "other.json", name="bad")


def test_replay_teacher_is_bounded_and_identified_by_its_labels() -> None:
    from semantscript_trainer.teacher import GeneratedCase

    cases = [GeneratedCase(inputs(0, "standard", 5, "paid", 5), "approve")]
    teacher = ReplayTeacher("qwen3-14b", cases, "0" * 64)
    assert teacher.descriptor.provider == "replay-labels/qwen3-14b"
    assert teacher.generate({}, 1) == tuple(cases)
    with pytest.raises(ExperimentError):
        teacher.generate({}, 2)
    assert tuple(SUPPORT) == ("approve", "deny", "review")
