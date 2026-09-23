"""Deterministic prompts for boundary pairs and counterfactual twins."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from semantscript_trainer.adversarial_contract import (
    build_boundary_pair_schema,
    build_counterfactual_schema,
)
from semantscript_trainer.constraints import constraints_from_ir
from semantscript_trainer.teacher import GeneratedCase, NeuralFunctionIr

BOUNDARY_SYSTEM_PROMPT = """You generate a paired adversarial training example for a typed
SemantScript function. Return only the JSON required by the response schema. The two inputs must
differ at exactly one JSON path, with predicateFalse evaluating false and predicateTrue evaluating
true for the selected constraint. Label both cases according to the complete behavior contract;
all active always and never constraints are mandatory. Do not use Markdown."""

COUNTERFACTUAL_SYSTEM_PROMPT = """You generate a counterfactual twin for a typed SemantScript
function. Return only the JSON required by the response schema. Change exactly one JSON input path
and change the output label. State the behavioral reason for the label flip. The twin must satisfy
the complete behavior contract and all active always and never constraints. Do not use Markdown."""


def build_boundary_messages(
    ir: NeuralFunctionIr,
    constraint_index: int,
) -> tuple[str, str]:
    constraints = constraints_from_ir(ir)
    if isinstance(constraint_index, bool) or not isinstance(constraint_index, int):
        raise ValueError("constraint index must be an integer")
    if constraint_index < 0 or constraint_index >= len(constraints):
        raise ValueError("constraint index is out of range")
    projection = _projection(ir)
    projection.update(
        {
            "selectedConstraint": {
                "index": constraint_index,
                "value": constraints[constraint_index],
            },
            "responseSchema": build_boundary_pair_schema(ir),
        }
    )
    return BOUNDARY_SYSTEM_PROMPT, _request("Generate the boundary pair", projection)


def build_counterfactual_messages(
    ir: NeuralFunctionIr,
    anchor: GeneratedCase,
) -> tuple[str, str]:
    if not isinstance(anchor, GeneratedCase):
        raise ValueError("counterfactual anchor must be a GeneratedCase")
    projection = _projection(ir)
    projection.update(
        {
            "anchor": {"inputs": anchor.inputs, "output": anchor.output},
            "responseSchema": build_counterfactual_schema(ir),
        }
    )
    return COUNTERFACTUAL_SYSTEM_PROMPT, _request(
        "Generate one minimally edited counterfactual twin",
        projection,
    )


def _projection(ir: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return {
            "functionId": ir["id"],
            "definition": ir["definition"],
            "inputs": ir["inputs"],
            "output": ir["output"],
        }
    except KeyError as error:
        raise ValueError(f"IR is missing required prompt field {error.args[0]!r}") from error


def _request(instruction: str, projection: Mapping[str, Any]) -> str:
    payload = json.dumps(
        projection,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return f"{instruction} from this contract:\n{payload}"


__all__ = [
    "BOUNDARY_SYSTEM_PROMPT",
    "COUNTERFACTUAL_SYSTEM_PROMPT",
    "build_boundary_messages",
    "build_counterfactual_messages",
]
