from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantscript_trainer.teacher_prompt import (
    MAXIMUM_REJECTION_NOTES,
    PROMPT_CONTRACT_VERSION,
    SYSTEM_PROMPT,
    RejectionNote,
    build_case_messages,
    build_coverage_brief,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEAF_PATHS = {
    "customer.priorRefunds",
    "customer.tier",
    "order.ageDays",
    "order.status",
    "order.total",
}


def _refund_ir() -> dict[str, object]:
    return json.loads(
        (REPOSITORY_ROOT / "examples" / "ir" / "refund-decision.v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_builds_deterministic_provider_neutral_prompt() -> None:
    ir = _refund_ir()
    first = build_case_messages(ir, 1, 3)
    second = build_case_messages(ir, 1, 3)

    assert PROMPT_CONTRACT_VERSION == 3
    assert first == second
    assert first[0] == SYSTEM_PROMPT
    assert first[1].startswith("Generate case 2 of 3 from this contract:\n")
    assert '"coverageBrief"' in first[1]
    assert '"responseSchema"' in first[1]
    assert '"approve"' in first[1]
    assert '"constraints"' in first[1]
    assert "trainingProvenance" not in first[1]
    assert "rejectedAttempts" not in first[1]


def test_brief_cycles_target_labels_and_constraint_regions() -> None:
    ir = _refund_ir()
    # Support is [approve, deny, review]; constraints are never-approve-if-fraudulent
    # (index 0) and always-deny-if-older-than-90-days (index 1): 3 labels x 5 regions.
    briefs = [build_coverage_brief(ir, index) for index in range(16)]

    assert [brief["targetOutput"]["value"] for brief in briefs[:3]] == ["approve", "deny", "review"]
    assert all(brief["constraintFocus"]["mode"] == "none" for brief in briefs[:3])

    satisfy_never = briefs[3:6]
    assert all(brief["constraintFocus"]["mode"] == "satisfy" for brief in satisfy_never)
    assert all(brief["constraintFocus"]["constraintIndex"] == 0 for brief in satisfy_never)
    assert all(brief["constraintFocus"]["kind"] == "never" for brief in satisfy_never)
    assert all(brief["targetOutput"]["value"] != "approve" for brief in satisfy_never)

    near_miss_never = briefs[6:9]
    assert all(brief["constraintFocus"]["mode"] == "near-miss" for brief in near_miss_never)
    assert [brief["targetOutput"]["value"] for brief in near_miss_never] == [
        "approve",
        "deny",
        "review",
    ]

    satisfy_always = briefs[9:12]
    assert all(brief["constraintFocus"]["mode"] == "satisfy" for brief in satisfy_always)
    assert all(brief["constraintFocus"]["constraintIndex"] == 1 for brief in satisfy_always)
    assert all(brief["targetOutput"]["value"] == "deny" for brief in satisfy_always)

    assert all(brief["constraintFocus"]["mode"] == "near-miss" for brief in briefs[12:15])
    assert briefs[15]["constraintFocus"]["mode"] == "none"
    assert briefs[15]["targetOutput"]["value"] == "approve"


def test_variation_hints_cover_every_leaf_and_vary_by_position() -> None:
    ir = _refund_ir()
    hints = [build_coverage_brief(ir, index)["variation"] for index in range(12)]

    assert all(set(hint) == LEAF_PATHS for hint in hints)
    assert len({json.dumps(hint, sort_keys=True) for hint in hints}) > 1
    assert all(
        hint["customer.tier"] in {'prefer "enterprise"', 'prefer "standard"'} for hint in hints
    )
    assert all(hint["order.status"] in {'prefer "fraudulent"', 'prefer "paid"'} for hint in hints)
    number_hints = {
        "small",
        "typical",
        "large",
        "unusual but valid",
        "exactly at a threshold named in the behavior or constraints",
        "just inside a threshold named in the behavior or constraints",
        "just outside a threshold named in the behavior or constraints",
    }
    assert all(hint["order.total"] in number_hints for hint in hints)
    assert any("threshold" in hint["order.ageDays"] for hint in hints)
    assert build_coverage_brief(ir, 7) == build_coverage_brief(ir, 7)


def test_brief_without_constraints_or_scalar_support() -> None:
    ir = _refund_ir()
    ir["definition"]["constraints"] = []
    brief = build_coverage_brief(ir, 4)
    assert brief["constraintFocus"]["mode"] == "none"
    assert brief["targetOutput"]["value"] == "deny"

    ir["output"] = {
        "kind": "object",
        "tsType": "Pair",
        "fields": [
            {
                "name": "a",
                "head": {"kind": "nominal", "sourceKind": "boolean", "support": [False, True]},
            }
        ],
    }
    assert "targetOutput" not in build_coverage_brief(ir, 0)


def test_rejection_notes_are_appended_and_bounded() -> None:
    ir = _refund_ir()
    plain = build_case_messages(ir, 0, 2)
    notes = tuple(
        RejectionNote(reason=f"reason {index}", inputs={"order": {"ageDays": index}})
        for index in range(MAXIMUM_REJECTION_NOTES + 3)
    )
    retried = build_case_messages(ir, 0, 2, rejected=notes)

    assert retried[0] == plain[0]
    assert retried[1].startswith(plain[1])
    assert '"rejectedAttempts"' in retried[1]
    assert "reason 0" not in retried[1]
    assert f"reason {MAXIMUM_REJECTION_NOTES + 2}" in retried[1]
    assert build_case_messages(ir, 0, 2, rejected=()) == plain


@pytest.mark.parametrize("rejected", ["not-a-sequence", (object(),), ({"reason": "x"},)])
def test_rejects_invalid_rejection_notes(rejected: object) -> None:
    with pytest.raises(ValueError):
        build_case_messages(_refund_ir(), 0, 1, rejected=rejected)  # type: ignore[arg-type]


@pytest.mark.parametrize("reason", ["", "   "])
def test_rejects_empty_rejection_reason(reason: str) -> None:
    with pytest.raises(ValueError):
        RejectionNote(reason=reason)


@pytest.mark.parametrize(("index", "total"), [(-1, 1), (0, 0), (1, 1), (True, 1)])
def test_rejects_invalid_case_position(index: int, total: int) -> None:
    with pytest.raises(ValueError):
        build_case_messages(_refund_ir(), index, total)
