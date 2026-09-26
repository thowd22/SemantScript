"""The trainer's predicate evaluator against the vectors semantscript explain also runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from semantscript_trainer.constraints import (
    ConstraintEvaluationError,
    evaluate_constraint_predicate,
)

VECTORS = Path(__file__).parents[2] / "examples" / "constraints" / "predicate-vectors.v1.json"


def _vectors() -> list[dict[str, Any]]:
    document = json.loads(VECTORS.read_text(encoding="utf-8"))
    assert document["kind"] == "semantscript.predicate-vectors"
    assert document["vectorsVersion"] == 1
    return document["vectors"]


@pytest.mark.parametrize("vector", _vectors(), ids=lambda vector: vector["name"])
def test_predicate_vector(vector: dict[str, Any]) -> None:
    constraint = {"predicate": vector["predicate"]}
    if vector["expect"] == "error":
        with pytest.raises(ConstraintEvaluationError):
            evaluate_constraint_predicate(constraint, vector["inputs"])
    else:
        assert evaluate_constraint_predicate(constraint, vector["inputs"]) is vector["expect"]
