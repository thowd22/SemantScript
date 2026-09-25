"""Closed provider response contracts for adversarial generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from semantscript_trainer.case_contract import build_case_schema, validate_case
from semantscript_trainer.strict_json import (
    DEFAULT_STRICT_JSON_LIMITS,
    StrictJsonError,
    StrictJsonLimits,
    loads_strict_json,
)
from semantscript_trainer.teacher import (
    MAXIMUM_COUNTERFACTUAL_REASON_BYTES,
    BoundaryPairProposal,
    CounterfactualProposal,
    GeneratedCase,
    JsonValue,
    NeuralFunctionIr,
    TeacherResponseError,
)


def build_boundary_pair_schema(ir: NeuralFunctionIr) -> dict[str, JsonValue]:
    """Build the closed schema for a false/true constraint-boundary proposal."""

    case_schema = build_case_schema(ir)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["predicateFalse", "predicateTrue"],
        "properties": {
            "predicateFalse": cast(JsonValue, case_schema),
            "predicateTrue": cast(JsonValue, case_schema),
        },
    }


def build_counterfactual_schema(ir: NeuralFunctionIr) -> dict[str, JsonValue]:
    """Build the closed schema for a twin plus teacher-stated flip reason."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["twin", "reason"],
        "properties": {
            "twin": cast(JsonValue, build_case_schema(ir)),
            "reason": {
                "type": "string",
                "minLength": 1,
                # This is a provider hint; the strict local boundary is UTF-8 bytes.
                "maxLength": MAXIMUM_COUNTERFACTUAL_REASON_BYTES,
            },
        },
    }


def parse_boundary_pair_response(
    ir: NeuralFunctionIr,
    response: str | bytes | bytearray,
    *,
    limits: StrictJsonLimits = DEFAULT_STRICT_JSON_LIMITS,
) -> BoundaryPairProposal:
    """Parse and validate a teacher's strict-JSON boundary-pair response for the IR."""
    value = _parse_object(response, limits=limits)
    if set(value) != {"predicateFalse", "predicateTrue"}:
        raise TeacherResponseError(
            "boundary response must contain exactly predicateFalse and predicateTrue"
        )
    return BoundaryPairProposal(
        predicate_false=_parse_case(ir, value["predicateFalse"], "predicateFalse"),
        predicate_true=_parse_case(ir, value["predicateTrue"], "predicateTrue"),
    )


def parse_counterfactual_response(
    ir: NeuralFunctionIr,
    response: str | bytes | bytearray,
    *,
    limits: StrictJsonLimits = DEFAULT_STRICT_JSON_LIMITS,
) -> CounterfactualProposal:
    """Parse and validate a teacher's strict-JSON counterfactual response for the IR."""
    value = _parse_object(response, limits=limits)
    if set(value) != {"twin", "reason"}:
        raise TeacherResponseError("counterfactual response must contain exactly twin and reason")
    reason = value["reason"]
    if not isinstance(reason, str):
        raise TeacherResponseError("counterfactual reason must be a string")
    return CounterfactualProposal(
        twin=_parse_case(ir, value["twin"], "twin"),
        reason=reason,
    )


def _parse_object(
    response: str | bytes | bytearray,
    *,
    limits: StrictJsonLimits,
) -> dict[str, Any]:
    try:
        value = loads_strict_json(response, limits=limits)
    except StrictJsonError as error:
        raise TeacherResponseError(f"teacher returned invalid adversarial JSON: {error}") from error
    if not isinstance(value, dict):
        raise TeacherResponseError("adversarial response must be a JSON object")
    return value


def _parse_case(ir: NeuralFunctionIr, raw: object, context: str) -> GeneratedCase:
    if not isinstance(raw, Mapping) or set(raw) != {"inputs", "output"}:
        raise TeacherResponseError(f"adversarial {context} must contain exactly inputs and output")
    inputs = raw["inputs"]
    if not isinstance(inputs, dict):
        raise TeacherResponseError(f"adversarial {context} inputs must be an object")
    return validate_case(
        ir,
        GeneratedCase(inputs=inputs, output=cast(JsonValue, raw["output"])),
    )


__all__ = [
    "build_boundary_pair_schema",
    "build_counterfactual_schema",
    "parse_boundary_pair_response",
    "parse_counterfactual_response",
]
