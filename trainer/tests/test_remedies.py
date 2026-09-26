"""The printed remedies: one source, the formatter, and the evidence behind each `next:` line."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from semantscript_trainer.remedies import REMEDY_TEMPLATES, remedy, remedy_family
from semantscript_trainer.suggestions import (
    LabelledCase,
    Violation,
    calibration_suggestion,
    example_literal,
    gold_miss_suggestion,
    seed_retry_suggestion,
    shared_rule,
    violation_suggestion,
)

REPOSITORY = Path(__file__).resolve().parents[2]
PLACEHOLDER = re.compile(r"\{([a-z][a-zA-Z0-9]*)\}")


def test_the_generated_module_matches_the_single_source() -> None:
    source = json.loads((REPOSITORY / "diagnostics" / "remedies.json").read_text("utf-8"))
    expected = {
        entry["id"]: {
            "family": entry["family"],
            "fix": entry["fix"],
            "params": list(dict.fromkeys(PLACEHOLDER.findall(entry["fix"]))),
        }
        for entry in source["remedies"]
    }
    assert REMEDY_TEMPLATES == expected, (
        "trainer/src/semantscript_trainer/remedies_generated.py is stale: run "
        "node scripts/generate-remedies.mjs"
    )
    # The catalogue's fix column is generated from the same entries.
    catalogue = (REPOSITORY / "docs" / "diagnostics.md").read_text("utf-8")
    for entry in source["remedies"]:
        assert f"`{PLACEHOLDER.sub(r'<\1>', entry['fix'])}`" in catalogue, entry["id"]


def test_remedy_fills_templates_and_rejects_unknown_ids_and_parameters() -> None:
    assert remedy("type-no-cache") == "rerun with --no-cache to retrain the head from scratch"
    assert remedy_family("module-missing") == "trainer-process"
    assert remedy("run-no-export", exports="a, b").endswith(": a, b")
    with pytest.raises(KeyError, match="unknown remedy"):
        remedy("no-such-remedy")
    with pytest.raises(TypeError, match="needs parameter exports"):
        remedy("run-no-export")
    with pytest.raises(TypeError, match="has no parameter extra"):
        remedy("run-no-export", exports="a", extra=1)


def case(
    case_id: str,
    status: str,
    age: int,
    expected: str,
    predicted: str,
    *,
    teacher: bool = True,
) -> LabelledCase:
    return LabelledCase(
        case_id,
        {"order": {"status": status, "ageDays": age}, "note": "x"},
        expected,
        predicted,
        teacher,
    )


CORPUS = [
    case("s1", "cancelled", 5, "deny", "deny"),
    case("s2", "cancelled", 40, "deny", "deny"),
    case("s3", "paid", 5, "approve", "approve"),
    case("s4", "paid", 95, "deny", "approve"),
    case("s5", "paid", 120, "deny", "approve"),
    case("s6", "paid", 10, "approve", "approve"),
]


def test_misses_sharing_a_one_field_rule_suggest_the_constraint() -> None:
    misses = [
        case("gold-1", "cancelled", 3, "deny", "approve", teacher=False),
        case("gold-2", "cancelled", 60, "deny", "review", teacher=False),
    ]
    corpus = [*CORPUS, *misses]
    assert shared_rule(misses, corpus, []) == ('order.status === "cancelled"', "deny", 2)
    assert gold_miss_suggestion(misses, corpus, []) == remedy(
        "gold-rule-constraint", predicate='order.status === "cancelled"', output='"deny"', count=2
    )
    # A rule a constraint already states is not suggested again.
    stated = [{"kind": "always", "source": 'order.status  ===  "cancelled"', "output": "deny"}]
    assert shared_rule(misses, corpus, stated) is None


def test_a_rule_an_existing_always_constraint_implies_is_not_suggested() -> None:
    misses = [
        case("gold-1", "cancelled", 3, "deny", "approve", teacher=False),
        case("gold-2", "cancelled", 60, "deny", "review", teacher=False),
    ]
    corpus = [*CORPUS, *misses]
    # always(() => order.status !== "paid", "deny") states the rule in other words.

    def required(inputs: object) -> tuple[str, ...]:
        order = inputs["order"]  # type: ignore[index]
        return ("deny",) if order["status"] != "paid" else ()

    stated = [{"kind": "always", "source": 'order.status !== "paid"', "output": "deny"}]
    assert shared_rule(misses, corpus, stated, required) is None
    assert not gold_miss_suggestion(misses, corpus, stated, required).startswith(
        "add the constraint"
    )
    # A constraint that requires another output there does not count as stating it.
    assert shared_rule(misses, corpus, [], lambda _inputs: ("approve",)) == (
        'order.status === "cancelled"',
        "deny",
        2,
    )


def test_an_object_output_never_gets_a_constraint_suggestion() -> None:
    verdict = {"decision": "deny", "reason": "x"}

    def row(case_id: str, status: str, teacher: bool) -> LabelledCase:
        return LabelledCase(
            case_id,
            {"order": {"status": status}},
            verdict,
            {"decision": "approve", "reason": "x"},
            teacher,
        )

    misses = [row("gold-1", "cancelled", False), row("gold-2", "cancelled", False)]
    corpus = [row("s1", "cancelled", True), *misses]
    # The compiler rejects constraints on a flat-interface output (TS9122).
    assert shared_rule(misses, corpus, []) is None
    assert gold_miss_suggestion(misses, corpus, []).startswith("add the examples entry")


def test_a_numeric_bound_is_suggested_only_when_every_labelled_case_agrees() -> None:
    misses = [
        case("gold-1", "paid", 100, "deny", "approve", teacher=False),
        case("gold-2", "cancelled", 130, "deny", "approve", teacher=False),
    ]
    rule = shared_rule(misses, [*CORPUS, *misses], [])
    assert rule == ("order.ageDays >= 100", "deny", 2)
    # A labelled case inside the bound that expects another output rules it out.
    contradicted = [*CORPUS, case("s7", "paid", 110, "approve", "approve"), *misses]
    assert shared_rule(misses, contradicted, []) is None


def test_a_repeated_miss_suggests_an_example_and_a_lone_one_a_check() -> None:
    miss = case("gold-1", "paid", 99, "deny", "approve", teacher=False)
    suggestion = gold_miss_suggestion([miss], [*CORPUS, miss], [])
    assert suggestion == remedy(
        "gold-repeated-miss-example",
        example=example_literal(CORPUS[3].inputs, "deny"),
        count=2,
        expected='"deny"',
        predicted='"approve"',
    )
    assert suggestion.startswith(
        'add the examples entry { inputs: {"order": {"status": "paid", "ageDays": 95}, '
        '"note": "x"}, output: "deny" }'
    )
    lone = case("gold-2", "paid", 1, "review", "approve", teacher=False)
    assert gold_miss_suggestion([lone], [*CORPUS, lone], []) == remedy(
        "gold-check-example", case="gold-2", expected='"review"', predicted='"approve"'
    )
    with pytest.raises(ValueError, match="at least one miss"):
        gold_miss_suggestion([], CORPUS, [])


def test_violations_suggest_an_example_or_denser_data() -> None:
    repeated = [
        Violation(1, "order.ageDays > 90", CORPUS[4]),
        Violation(1, "order.ageDays > 90", CORPUS[3]),
        Violation(0, 'order.status === "fraudulent"', CORPUS[0]),
    ]
    assert violation_suggestion(repeated, 64) == remedy(
        "violation-example",
        example=example_literal(CORPUS[3].inputs, "deny"),
        index=1,
        source="order.ageDays > 90",
        count=2,
    )
    assert violation_suggestion(repeated[2:], 64) == remedy(
        "violation-denser-data", cases=128, current=64
    )


def test_calibration_and_seed_retry_suggestions() -> None:
    small = calibration_suggestion(ece=0.2, rows=40, accuracy=0.8, current_cases=64, epochs=3)
    assert small.startswith("rerun with --cases 128 (now 64): ECE 0.2000 is measured on 40")
    fitted = calibration_suggestion(ece=0.2, rows=400, accuracy=0.97, current_cases=500, epochs=3)
    assert fitted.startswith("rerun with --cases 1000 (now 500)")
    underfit = calibration_suggestion(ece=0.2, rows=400, accuracy=0.8, current_cases=500, epochs=3)
    assert underfit.startswith("rerun with --epochs 5 (now 3): accuracy 0.8000")
    assert seed_retry_suggestion(attempts=1, first_seed=3, last_seed=3, maximum_seed=100) == (
        remedy("seed-retry-attempts", attempts=3)
    )
    assert seed_retry_suggestion(attempts=3, first_seed=3, last_seed=5, maximum_seed=100) == (
        remedy("seed-retry-next-seed", seed=6, first=3, last=5)
    )
    assert seed_retry_suggestion(attempts=3, first_seed=3, last_seed=5, maximum_seed=5) is None
