import math
from copy import deepcopy

import pytest

from semantscript_trainer.case_contract import (
    MAXIMUM_CASE_COUNT,
    build_case_schema,
    parse_case_response,
    validate_case,
    validate_case_count,
    validate_generated_cases,
)
from semantscript_trainer.strict_json import StrictJsonLimits
from semantscript_trainer.teacher import (
    GeneratedCase,
    TeacherConfigurationError,
    TeacherResponseError,
)


def test_builds_one_closed_case_schema_for_every_input_kind() -> None:
    schema = build_case_schema(all_inputs_ir())

    assert schema["required"] == ["inputs", "output"]
    assert schema["additionalProperties"] is False
    inputs = schema["properties"]["inputs"]
    assert inputs["additionalProperties"] is False
    assert inputs["required"] == input_names(all_inputs_ir())
    assert inputs["properties"]["pair"] == {
        "type": "array",
        "prefixItems": [{"type": "number"}, {"type": "boolean"}],
        "items": False,
        "minItems": 2,
        "maxItems": 2,
    }
    profile = inputs["properties"]["profile"]
    assert profile["required"] == ["name"]
    assert set(profile["properties"]) == {"name", "nickname"}
    assert inputs["properties"]["choice"]["anyOf"] == [
        {"const": "manual"},
        {"type": "number"},
    ]


def test_validates_all_input_kinds_and_optional_fields() -> None:
    ir = all_inputs_ir()
    case = valid_all_inputs_case()

    assert validate_case(ir, case) is case
    with_optional = deepcopy(case.inputs)
    with_optional["profile"]["nickname"] = "A"
    assert validate_case(ir, GeneratedCase(inputs=with_optional, output="approve"))


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda values: values.pop("text"), "input names"),
        (lambda values: values.__setitem__("extra", 1), "input names"),
        (lambda values: values.__setitem__("score", True), "finite number"),
        (lambda values: values.__setitem__("score", float("inf")), "finite number"),
        (lambda values: values.__setitem__("score", 10**10_000), "finite number"),
        (lambda values: values.__setitem__("nothing", False), "must be null"),
        (lambda values: values.__setitem__("exact", "other"), "literal"),
        (lambda values: values.__setitem__("negativeZero", 0.0), "literal"),
        (lambda values: values.__setitem__("status", "unknown"), "enum member"),
        (lambda values: values.__setitem__("numeric", 0.0), "enum member"),
        (lambda values: values.__setitem__("tags", "tag"), "must be an array"),
        (lambda values: values.__setitem__("pair", [1]), "tuple length"),
        (lambda values: values.__setitem__("choice", False), "union variant"),
        (lambda values: values["profile"].pop("name"), "missing or additional"),
        (lambda values: values["profile"].__setitem__("extra", "x"), "missing or additional"),
    ],
)
def test_rejects_invalid_inputs(mutate, match: str) -> None:
    values = deepcopy(valid_all_inputs_case().inputs)
    mutate(values)

    with pytest.raises(TeacherResponseError, match=match):
        validate_case(all_inputs_ir(), GeneratedCase(inputs=values, output="approve"))


@pytest.mark.parametrize(
    "head,valid,invalid",
    [
        (
            {"kind": "nominal", "sourceKind": "boolean", "support": [False, True]},
            True,
            1,
        ),
        (
            {
                "kind": "nominal",
                "sourceKind": "string-union",
                "support": ["approve", "deny"],
            },
            "deny",
            False,
        ),
        (
            {"kind": "nominal", "sourceKind": "number-enum", "support": [1, 2]},
            2.0,
            True,
        ),
        (
            {
                "kind": "ordinal",
                "sourceKind": "ordinal-string",
                "support": ["low", "high"],
                "expectedValue": "zero-based-rank",
            },
            "high",
            "middle",
        ),
        (
            {
                "kind": "ordinal",
                "sourceKind": "bounded-int",
                "minimum": "0",
                "maximum": "2",
                "step": "1",
                "supportDecimal": ["0", "1", "2"],
                "expectedValue": "numeric",
            },
            1.0,
            1.5,
        ),
        (
            {
                "kind": "ordinal",
                "sourceKind": "bounded-number",
                "minimum": "-0.5",
                "maximum": "0.5",
                "step": "0.5",
                "supportDecimal": ["-0.5", "0", "0.5"],
                "expectedValue": "numeric",
            },
            0.5,
            0.25,
        ),
    ],
)
def test_validates_every_scalar_head_support(head: dict, valid, invalid) -> None:
    ir = scalar_ir(head)
    inputs = {"value": "x"}

    assert validate_case(ir, GeneratedCase(inputs=inputs, output=valid))
    assert build_case_schema(ir)["properties"]["output"] == {
        "enum": head.get("support", [float(item) for item in head.get("supportDecimal", [])])
    }
    with pytest.raises(TeacherResponseError, match="outside the declared output support"):
        validate_case(ir, GeneratedCase(inputs=inputs, output=invalid))


def test_validates_flat_output_with_exact_and_empty_field_names() -> None:
    ir = flat_output_ir()
    case = GeneratedCase(inputs={"value": "x"}, output={"": True, "rank": 1})

    assert validate_case(ir, case)
    output_schema = build_case_schema(ir)["properties"]["output"]
    assert output_schema["required"] == ["", "rank"]
    assert output_schema["additionalProperties"] is False
    with pytest.raises(TeacherResponseError, match="fields must exactly match"):
        validate_case(ir, GeneratedCase(inputs={"value": "x"}, output={"": True}))


def test_parse_case_response_uses_strict_json_and_preserves_signed_zero() -> None:
    ir = scalar_ir({"kind": "nominal", "sourceKind": "number-enum", "support": [-0.0, 1]})
    case = parse_case_response(ir, b'{"inputs":{"value":"x"},"output":-0}')

    assert isinstance(case.output, float)
    assert math.copysign(1.0, case.output) == -1.0
    with pytest.raises(TeacherResponseError, match="invalid JSON"):
        parse_case_response(ir, '{"inputs":{},"inputs":{},"output":-0}')
    with pytest.raises(TeacherResponseError, match="UTF-8 byte length"):
        parse_case_response(
            ir,
            '{"inputs":{"value":"x"},"output":-0}',
            limits=StrictJsonLimits(maximum_bytes=4),
        )


def test_validates_generated_case_count_and_values() -> None:
    ir = scalar_ir(string_head())
    case = GeneratedCase(inputs={"value": "x"}, output="approve")
    assert validate_generated_cases(ir, [case, case], expected_count=2) == (case, case)

    with pytest.raises(TeacherResponseError, match="returned 1 cases; expected 2"):
        validate_generated_cases(ir, [case], expected_count=2)
    with pytest.raises(TeacherResponseError, match="GeneratedCase"):
        validate_generated_cases(ir, [{"inputs": {}, "output": "approve"}], expected_count=1)


def test_generated_case_validation_stops_after_one_excess_item() -> None:
    ir = scalar_ir(string_head())
    case = GeneratedCase(inputs={"value": "x"}, output="approve")
    consumed = 0

    def unbounded_cases():
        nonlocal consumed
        while True:
            consumed += 1
            yield case

    with pytest.raises(TeacherResponseError, match="more than 2 cases"):
        validate_generated_cases(ir, unbounded_cases(), expected_count=2)
    assert consumed == 3


@pytest.mark.parametrize("value", [-1, True, 1.5, MAXIMUM_CASE_COUNT + 1])
def test_rejects_invalid_or_excessive_case_counts(value) -> None:
    with pytest.raises(TeacherConfigurationError, match="case count"):
        validate_case_count(value)


def all_inputs_ir() -> dict:
    types = [
        ("text", {"kind": "string"}),
        ("flag", {"kind": "boolean"}),
        ("score", {"kind": "number"}),
        ("nothing", {"kind": "null"}),
        ("exact", {"kind": "literal", "value": "fixed"}),
        ("negativeZero", {"kind": "literal", "value": -0.0}),
        (
            "status",
            {"kind": "enum", "name": "Status", "base": "string", "values": ["new", "old"]},
        ),
        (
            "numeric",
            {"kind": "enum", "name": "Numeric", "base": "number", "values": [-0.0, 1]},
        ),
        ("tags", {"kind": "array", "items": {"kind": "string"}}),
        (
            "pair",
            {"kind": "tuple", "items": [{"kind": "number"}, {"kind": "boolean"}]},
        ),
        (
            "profile",
            {
                "kind": "object",
                "name": "Profile",
                "fields": [
                    {"name": "name", "optional": False, "type": {"kind": "string"}},
                    {"name": "nickname", "optional": True, "type": {"kind": "string"}},
                ],
            },
        ),
        (
            "choice",
            {
                "kind": "union",
                "variants": [{"kind": "literal", "value": "manual"}, {"kind": "number"}],
            },
        ),
    ]
    return {
        "inputs": [
            {"name": name, "index": index, "tsType": name, "type": input_type}
            for index, (name, input_type) in enumerate(types)
        ],
        "output": {"kind": "scalar", "tsType": "Decision", "head": string_head()},
    }


def valid_all_inputs_case() -> GeneratedCase:
    return GeneratedCase(
        inputs={
            "text": "hello",
            "flag": True,
            "score": 1.25,
            "nothing": None,
            "exact": "fixed",
            "negativeZero": -0.0,
            "status": "new",
            "numeric": -0.0,
            "tags": ["a", "b"],
            "pair": [3, False],
            "profile": {"name": "Ada"},
            "choice": "manual",
        },
        output="approve",
    )


def scalar_ir(head: dict) -> dict:
    return {
        "inputs": [{"name": "value", "index": 0, "tsType": "string", "type": {"kind": "string"}}],
        "output": {"kind": "scalar", "tsType": "Output", "head": head},
    }


def flat_output_ir() -> dict:
    return {
        "inputs": [{"name": "value", "index": 0, "tsType": "string", "type": {"kind": "string"}}],
        "output": {
            "kind": "object",
            "tsType": "Result",
            "fields": [
                {
                    "name": "",
                    "head": {"kind": "nominal", "sourceKind": "boolean", "support": [False, True]},
                },
                {
                    "name": "rank",
                    "head": {"kind": "nominal", "sourceKind": "number-enum", "support": [1, 2]},
                },
            ],
        },
    }


def string_head() -> dict:
    return {
        "kind": "nominal",
        "sourceKind": "string-union",
        "support": ["approve", "deny"],
    }


def input_names(ir: dict) -> list[str]:
    return [entry["name"] for entry in ir["inputs"]]
