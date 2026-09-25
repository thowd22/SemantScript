"""Deterministic prompts for boundary pairs and counterfactual twins."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from semantscript_trainer.constraints import constraints_from_ir
from semantscript_trainer.teacher import GeneratedCase, NeuralFunctionIr
from semantscript_trainer.teacher_prompt import build_contract

BOUNDARY_SYSTEM_PROMPT = """You generate a paired adversarial training example for a typed
SemantScript function. Return only the JSON required by the response schema. The two inputs must
differ at exactly one JSON path, with predicateFalse evaluating false and predicateTrue evaluating
true for the selected constraint. Label both cases according to the complete behavior contract;
all active always and never constraints are mandatory. Do not use Markdown. The function contract
follows as compact JSON; the user message names the selected constraint."""

COUNTERFACTUAL_SYSTEM_PROMPT = """You generate a counterfactual twin for a typed SemantScript
function. Return only the JSON required by the response schema. Change exactly one JSON input path
and change the output label. State the behavioral reason for the label flip. The twin must satisfy
the complete behavior contract and all active always and never constraints. Do not use Markdown.
The function contract follows as compact JSON; the user message gives the anchor case."""


def build_boundary_messages(
    ir: NeuralFunctionIr,
    constraint_index: int,
) -> tuple[str, str]:
    """Return the system and user prompts asking a teacher for a boundary pair of one constraint.

    The system prompt carries the compact function contract (shared by every boundary
    request for the function, so prompt caching can serve it); the user message names
    the selected constraint by index, kind, output and TypeScript source.
    """
    constraints = constraints_from_ir(ir)
    if isinstance(constraint_index, bool) or not isinstance(constraint_index, int):
        raise ValueError("constraint index must be an integer")
    if constraint_index < 0 or constraint_index >= len(constraints):
        raise ValueError("constraint index is out of range")
    constraint = constraints[constraint_index]
    selected = {
        "index": constraint_index,
        "kind": constraint.get("kind"),
        "output": constraint.get("output"),
        "source": constraint.get("source"),
    }
    return (
        _system(BOUNDARY_SYSTEM_PROMPT, ir),
        f"Generate the boundary pair for this selectedConstraint:\n{_dump(selected)}",
    )


def build_counterfactual_messages(
    ir: NeuralFunctionIr,
    anchor: GeneratedCase,
) -> tuple[str, str]:
    """Return the system and user prompts asking a teacher for a counterfactual twin of ``anchor``."""
    if not isinstance(anchor, GeneratedCase):
        raise ValueError("counterfactual anchor must be a GeneratedCase")
    return (
        _system(COUNTERFACTUAL_SYSTEM_PROMPT, ir),
        "Generate one minimally edited counterfactual twin of this anchor:\n"
        + _dump({"inputs": anchor.inputs, "output": anchor.output}),
    )


def _system(instructions: str, ir: Mapping[str, Any]) -> str:
    return f"{instructions}\n\nContract:\n{build_contract(ir)}"


def _dump(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


__all__ = [
    "BOUNDARY_SYSTEM_PROMPT",
    "COUNTERFACTUAL_SYSTEM_PROMPT",
    "build_boundary_messages",
    "build_counterfactual_messages",
]
