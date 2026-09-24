from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from semantscript_trainer.canonical_input import (
    CANONICAL_INPUT_V1,
    CANONICAL_INPUT_V2,
    CanonicalInputError,
    canonical_input_version,
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


V2_FIXTURE = Path(__file__).parents[2] / "examples" / "serialization" / "canonical-input.v2.json"


def test_matches_every_canonical_input_v2_golden_vector_byte_for_byte() -> None:
    fixture = json.loads(V2_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["encoding"] == CANONICAL_INPUT_V2
    names = {vector["name"] for vector in fixture["vectors"]}
    assert {
        "refund-representative",
        "numbers",
        "strings",
        "unions-literals-enums",
        "nested-containers",
        "key-spelling",
    } <= names
    for vector in fixture["vectors"]:
        actual = serialize_canonical_inputs(vector["schema"], vector["inputs"], version=2)
        assert actual == vector["canonicalUtf8"].encode("utf-8"), vector["name"]
        assert hashlib.sha256(actual).hexdigest() == vector["sha256"], vector["name"]
        assert (
            serialize_canonical_inputs_string(vector["schema"], vector["inputs"], version=2)
            == vector["canonicalUtf8"]
        )
        # The exact-JSON envelope is untouched by the compact encoding.
        assert serialize_canonical_inputs(vector["schema"], vector["inputs"]).startswith(
            b'["semantscript-input",1,'
        )
    numbers = next(vector for vector in fixture["vectors"] if vector["name"] == "numbers")
    assert math.copysign(1.0, numbers["inputs"]["n"]["zero"]) == -1.0


def test_compact_encoding_is_injective_over_look_alike_values() -> None:
    mixed = schema({"kind": "union", "variants": [NUMBER, STRING]})
    rendered = {
        serialize_canonical_inputs_string(mixed, {"value": candidate}, version=2)
        for candidate in (12, "12", 12.0, "12.0", -0.0, 0.0, "0", "-0", "true", "null")
    }
    assert rendered == {
        f"value={spelling}"
        for spelling in ("12", '"12"', '"12.0"', "-0", "0", '"0"', '"-0"', '"true"', '"null"')
    }
    booleans = schema({"kind": "union", "variants": [{"kind": "boolean"}, STRING]})
    assert serialize_canonical_inputs_string(booleans, {"value": True}, version=2) == "value=true"
    assert (
        serialize_canonical_inputs_string(booleans, {"value": "true"}, version=2) == 'value="true"'
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, "1"),
        (-1.5, "-1.5"),
        (1e21, "1e+21"),
        (1e20, "100000000000000000000"),
        (1e-7, "1e-7"),
        (1e-6, "0.000001"),
        (5e-324, "5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
        (2**53, "9007199254740992"),
        (0.1 + 0.2, "0.30000000000000004"),
        (123456789012345680000.0, "123456789012345680000"),
        (1.5e300, "1.5e+300"),
        (-2.5e-10, "-2.5e-10"),
    ],
)
def test_compact_numbers_use_javascript_spelling(value: float, expected: str) -> None:
    assert serialize_canonical_inputs_string(schema(NUMBER), {"value": value}, version=2) == (
        f"value={expected}"
    )


def test_compact_encoding_enforces_the_byte_limit_and_version_choice() -> None:
    entries = schema(STRING)
    text = serialize_canonical_inputs(entries, {"value": "x" * 40}, version=2)
    assert text == b"value=" + b"x" * 40
    with assert_reason("limit"):
        serialize_canonical_inputs(entries, {"value": "x" * 40}, version=2, maximum_bytes=10)
    with pytest.raises(ValueError, match="version must be 1 or 2"):
        serialize_canonical_inputs(entries, {"value": "x"}, version=3)
    with pytest.raises(ValueError, match="version must be 1 or 2"):
        serialize_canonical_inputs(entries, {"value": "x"}, version=True)  # type: ignore[arg-type]
    assert canonical_input_version(CANONICAL_INPUT_V1) == 1
    assert canonical_input_version(CANONICAL_INPUT_V2) == 2
    assert canonical_input_version("semantscript.canonical-input/v3") is None
    # Validation failures are shared: the compact writer never sees an invalid input.
    with assert_reason("missing"):
        serialize_canonical_inputs(entries, {}, version=2)
    with assert_reason("type"):
        serialize_canonical_inputs(entries, {"value": 5}, version=2)
