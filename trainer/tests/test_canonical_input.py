from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from semantscript_trainer.canonical_input import (
    CanonicalInputError,
    serialize_canonical_inputs,
    serialize_canonical_inputs_string,
)

NUMBER = {"kind": "number"}
STRING = {"kind": "string"}


def schema(type_spec: dict[str, Any], name: str = "value") -> list[dict[str, Any]]:
    return [{"name": name, "index": 0, "tsType": "unknown", "type": type_spec}]


def assert_reason(reason: str):
    return pytest.raises(CanonicalInputError, check=lambda error: error.reason == reason)


def test_matches_canonical_input_v1_golden_vector_byte_for_byte() -> None:
    fixture_path = (
        Path(__file__).parents[2] / "examples" / "serialization" / "canonical-input.v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected = fixture["canonicalUtf8"].encode("utf-8")

    for value in fixture["equivalentValues"]:
        actual = serialize_canonical_inputs(schema(fixture["input"]["type"], "obj"), {"obj": value})
        assert actual == expected
        assert hashlib.sha256(actual).hexdigest() == fixture["sha256"]
        assert math.copysign(1.0, value["zero"]) == -1.0


def test_encodes_all_nodes_and_orders_top_level_inputs_by_index() -> None:
    entries = [
        {
            "name": "second",
            "index": 1,
            "type": {
                "kind": "object",
                "name": "Record",
                "fields": [
                    {"name": "optional", "optional": True, "type": STRING},
                    {
                        "name": "tuple",
                        "optional": False,
                        "type": {"kind": "tuple", "items": [STRING, NUMBER]},
                    },
                ],
            },
        },
        {
            "name": "first",
            "index": 0,
            "type": {
                "kind": "array",
                "items": {
                    "kind": "union",
                    "variants": [
                        {"kind": "literal", "value": "x"},
                        {
                            "kind": "enum",
                            "name": "Letter",
                            "base": "string",
                            "values": ["x", "y"],
                        },
                    ],
                },
            },
        },
    ]

    assert serialize_canonical_inputs_string(
        entries,
        {"second": {"tuple": ["ok", 1]}, "first": ["x", "y"]},
    ) == (
        '["semantscript-input",1,[["first",["array",[["union",0,["literal",'
        '["string","x"]]],["union",1,["enum","Letter",1,["string","y"]]]]]],'
        '["second",["object",[["tuple",["tuple",[["string","ok"],["number",'
        '"3ff0000000000000"]]]]]]]]]'
    )


def test_numbers_literals_and_numeric_enums_are_bit_exact() -> None:
    assert '["null"]' in serialize_canonical_inputs_string(
        schema({"kind": "null"}), {"value": None}
    )
    assert '["boolean",false]' in serialize_canonical_inputs_string(
        schema({"kind": "boolean"}), {"value": False}
    )
    assert '["number","8000000000000000"]' in serialize_canonical_inputs_string(
        schema(NUMBER), {"value": -0.0}
    )

    with assert_reason("value"):
        serialize_canonical_inputs(schema({"kind": "literal", "value": 0}), {"value": -0.0})

    assert serialize_canonical_inputs_string(
        schema({"kind": "enum", "name": "Zero", "base": "number", "values": [-0.0, 0]}),
        {"value": 0},
    ) == ('["semantscript-input",1,[["value",["enum","Zero",1,["number","0000000000000000"]]]]]')


def test_object_fields_use_utf8_order_and_exact_json_escapes() -> None:
    astral = "\U00010000"
    bmp = "\ue000"
    record = {
        "kind": "object",
        "name": "Unicode",
        "fields": [
            {"name": bmp, "optional": False, "type": STRING},
            {"name": astral, "optional": False, "type": STRING},
            {"name": "controls", "optional": False, "type": STRING},
        ],
    }

    encoded = serialize_canonical_inputs_string(
        schema(record),
        {
            "value": {
                bmp: "bmp",
                "controls": '\b\t\n\f\r\u0000\u001f/"\\',
                astral: "astral",
            }
        },
    )

    assert encoded.index(f'"{bmp}"') < encoded.index(f'"{astral}"')
    assert '["string","\\b\\t\\n\\f\\r\\u0000\\u001f/\\"\\\\"]' in encoded
    assert not encoded.endswith("\n")


def test_enforces_utf8_byte_limit_at_exact_boundary() -> None:
    value = f'{"x" * 4095}🙂\n"\\é'
    expected = serialize_canonical_inputs(schema(STRING), {"value": value})

    assert (
        serialize_canonical_inputs(
            schema(STRING),
            {"value": value},
            maximum_bytes=len(expected),
        )
        == expected
    )
    with assert_reason("limit"):
        serialize_canonical_inputs(
            schema(STRING),
            {"value": value},
            maximum_bytes=len(expected) - 1,
        )
    with assert_reason("limit"):
        serialize_canonical_inputs(
            schema(STRING),
            {"value": "é" * (1024 * 1024)},
            maximum_bytes=1024,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_numbers(value: float) -> None:
    with assert_reason("non-finite"):
        serialize_canonical_inputs(schema(NUMBER), {"value": value})


def test_rejects_invalid_unicode_host_objects_cycles_and_shape_mismatches() -> None:
    with assert_reason("invalid-unicode"):
        serialize_canonical_inputs(schema(STRING), {"value": "\ud800"})
    with assert_reason("invalid-unicode"):
        serialize_canonical_inputs(
            schema(
                {
                    "kind": "object",
                    "name": "InvalidKey",
                    "fields": [],
                }
            ),
            {"value": {"\ud800": "bad"}},
        )

    class Box:
        def __init__(self) -> None:
            self.item = "ok"

    object_type = {
        "kind": "object",
        "name": "Box",
        "fields": [{"name": "item", "optional": False, "type": STRING}],
    }
    with assert_reason("class-instance"):
        serialize_canonical_inputs(schema(object_type), {"value": Box()})

    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    cyclic_type = {
        "kind": "object",
        "name": "Cycle",
        "fields": [
            {
                "name": "self",
                "optional": False,
                "type": {
                    "kind": "object",
                    "name": "Inner",
                    "fields": [{"name": "self", "optional": True, "type": STRING}],
                },
            }
        ],
    }
    with assert_reason("cycle"):
        serialize_canonical_inputs(schema(cyclic_type), {"value": cycle})

    with assert_reason("tuple-length"):
        serialize_canonical_inputs(
            schema({"kind": "tuple", "items": [STRING]}),
            {"value": []},
        )
    with assert_reason("missing"):
        serialize_canonical_inputs(schema(object_type), {"value": {}})
    with assert_reason("extra"):
        serialize_canonical_inputs(
            schema(object_type),
            {"value": {"item": "ok", "extra": True}},
        )


def test_allows_shared_acyclic_values_and_rejects_schema_ambiguity_and_depth() -> None:
    leaf = {
        "kind": "object",
        "name": "Leaf",
        "fields": [{"name": "text", "optional": False, "type": STRING}],
    }
    shared = {"text": "ok"}
    serialize_canonical_inputs(
        schema({"kind": "tuple", "items": [leaf, leaf]}),
        {"value": [shared, shared]},
    )

    nested_type: dict[str, Any] = STRING
    nested_value: object = "bottom"
    for _ in range(101):
        nested_type = {"kind": "array", "items": nested_type}
        nested_value = [nested_value]
    with assert_reason("limit"):
        serialize_canonical_inputs(schema(nested_type), {"value": nested_value})

    with assert_reason("schema"):
        serialize_canonical_inputs(
            [{"name": "a", "index": 1, "type": STRING}],
            {"a": "x"},
        )
    with assert_reason("schema"):
        serialize_canonical_inputs(
            schema({"kind": "union", "variants": [STRING, STRING]}),
            {"value": "x"},
        )


def test_accepts_binary64_integral_schema_indices_from_strict_json() -> None:
    integral = schema({"kind": "number"})
    floating = [dict(integral[0], index=0.0)]
    assert serialize_canonical_inputs(floating, {"value": 1}) == serialize_canonical_inputs(
        integral, {"value": 1}
    )
    with pytest.raises(CanonicalInputError, match="non-negative safe integer"):
        serialize_canonical_inputs([dict(integral[0], index=0.5)], {"value": 1})
