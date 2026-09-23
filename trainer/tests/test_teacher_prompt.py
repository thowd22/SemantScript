from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantscript_trainer.teacher_prompt import SYSTEM_PROMPT, build_case_messages

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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

    assert first == second
    assert first[0] == SYSTEM_PROMPT
    assert "Generate case 2 of 3" in first[1]
    assert '"responseSchema"' in first[1]
    assert '"approve"' in first[1]
    assert '"constraints"' in first[1]
    assert "trainingProvenance" not in first[1]


@pytest.mark.parametrize(("index", "total"), [(-1, 1), (0, 0), (1, 1), (True, 1)])
def test_rejects_invalid_case_position(index: int, total: int) -> None:
    with pytest.raises(ValueError):
        build_case_messages(_refund_ir(), index, total)
