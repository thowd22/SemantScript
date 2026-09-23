import math

import pytest

from semantscript_trainer.strict_json import StrictJsonError, StrictJsonLimits, loads_strict_json


def test_parses_utf8_and_preserves_integer_negative_zero() -> None:
    value = loads_strict_json(b'{"message":"caf\xc3\xa9","zero":-0}')

    assert value == {"message": "caf\u00e9", "zero": -0.0}
    assert isinstance(value, dict)
    assert math.copysign(1.0, value["zero"]) == -1.0


def test_parses_integral_tokens_directly_to_binary64() -> None:
    assert loads_strict_json("9007199254740993") == 9007199254740992.0
    assert isinstance(loads_strict_json("1"), float)

    with pytest.raises(StrictJsonError, match="finite binary64"):
        loads_strict_json("1" + "0" * 400)


@pytest.mark.parametrize(
    "document,match",
    [
        ('{"value":1,"value":2}', "duplicate object property"),
        ("NaN", "non-finite"),
        ("Infinity", "non-finite"),
        ("-Infinity", "non-finite"),
        ("1e9999", "finite binary64"),
        ('"\\ud800"', "unpaired Unicode surrogate"),
        ("\ufeffnull", "byte order mark"),
    ],
)
def test_rejects_ambiguous_or_nonportable_json(document: str, match: str) -> None:
    with pytest.raises(StrictJsonError, match=match):
        loads_strict_json(document)


def test_rejects_invalid_utf8() -> None:
    with pytest.raises(StrictJsonError, match="valid UTF-8"):
        loads_strict_json(b'"\xff"')


def test_enforces_utf8_byte_limit_not_character_count() -> None:
    document = '"\u00e9"'
    assert loads_strict_json(document, limits=StrictJsonLimits(maximum_bytes=4)) == "\u00e9"

    with pytest.raises(StrictJsonError, match="UTF-8 byte length 3"):
        loads_strict_json(document, limits=StrictJsonLimits(maximum_bytes=3))


def test_enforces_depth_before_json_decoder_recursion() -> None:
    assert loads_strict_json('"[not nesting]"', limits=StrictJsonLimits(maximum_depth=0)) == (
        "[not nesting]"
    )
    assert loads_strict_json("[0]", limits=StrictJsonLimits(maximum_depth=1)) == [0]

    with pytest.raises(StrictJsonError, match="maximum depth 2"):
        loads_strict_json("[[[0]]]", limits=StrictJsonLimits(maximum_depth=2))


def test_enforces_node_limit_iteratively() -> None:
    assert loads_strict_json("[0,1]", limits=StrictJsonLimits(maximum_nodes=3)) == [0, 1]

    with pytest.raises(StrictJsonError, match="maximum node count 2"):
        loads_strict_json("[0,1]", limits=StrictJsonLimits(maximum_nodes=2))


@pytest.mark.parametrize(
    "arguments",
    [
        {"maximum_bytes": 0},
        {"maximum_bytes": True},
        {"maximum_depth": -1},
        {"maximum_depth": True},
        {"maximum_nodes": 0},
    ],
)
def test_rejects_invalid_limits(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        StrictJsonLimits(**arguments)


def test_rejects_non_text_documents() -> None:
    with pytest.raises(TypeError, match="str, bytes, or bytearray"):
        loads_strict_json(12)  # type: ignore[arg-type]
