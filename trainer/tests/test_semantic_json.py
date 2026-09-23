from __future__ import annotations

import hashlib

import pytest

from semantscript_trainer.semantic_json import (
    semantic_json_bytes,
    semantic_json_sha256,
    semantic_json_string,
)


def test_matches_compiler_fixed_vector_byte_for_byte() -> None:
    # Produced by compiler/src/semantic-json.ts under Node 22. The two non-ASCII
    # keys also distinguish its UTF-8 ordering from JavaScript's default UTF-16
    # string ordering.
    value = {
        "z": None,
        "a": True,
        "n": -0.0,
        "u": "é🙂\n",
        "\ue000": "bmp",
        "\U00010000": "astral",
        "values": [1, 1.5, 5e-324, 1.7976931348623157e308],
    }
    expected = (
        '["semantscript-semantic-json",1,["object",[["a",["boolean",true]],'
        '["n",["number","8000000000000000"]],["u",["string","é🙂\\n"]],'
        '["values",["array",[["number","3ff0000000000000"],["number",'
        '"3ff8000000000000"],["number","0000000000000001"],["number",'
        '"7fefffffffffffff"]]]],["z",["null"]],["\ue000",["string","bmp"]],'
        '["\U00010000",["string","astral"]]]]]'
    )

    assert semantic_json_string(value) == expected
    assert semantic_json_bytes(value) == expected.encode("utf-8")
    assert (
        semantic_json_sha256(value)
        == "a78899420e05556d84766508a3fbdc2da944526248c397800f023b70c1c2e033"
    )
    assert hashlib.sha256(semantic_json_bytes(value)).hexdigest() == semantic_json_sha256(value)


def test_numbers_use_javascript_binary64_semantics() -> None:
    assert semantic_json_string(0.0).endswith('["number","0000000000000000"]]')
    assert semantic_json_string(-0.0).endswith('["number","8000000000000000"]]')
    assert semantic_json_string(2**53 + 1).endswith('["number","4340000000000000"]]')


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 10**1000])
def test_rejects_numbers_outside_finite_binary64(value: int | float) -> None:
    with pytest.raises(TypeError, match="numbers must be finite"):
        semantic_json_bytes(value)


@pytest.mark.parametrize(
    "value",
    ["\ud800", "\udfff", {"\ud800": None}, ["ok", "\udfff"]],
)
def test_rejects_unpaired_unicode_surrogates(value: object) -> None:
    with pytest.raises(TypeError, match="unpaired UTF-16 surrogates"):
        semantic_json_string(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [b"bytes", ("tuple",), {1: "non-string key"}])
def test_rejects_non_json_values(value: object) -> None:
    with pytest.raises(TypeError, match="semantic JSON"):
        semantic_json_bytes(value)  # type: ignore[arg-type]


def test_uses_json_stringify_compatible_escaping_without_ascii_folding() -> None:
    encoded = semantic_json_string('\b\t\n\f\r\u0000\u001f/"\\é\u2028\u2029')

    assert encoded == (
        '["semantscript-semantic-json",1,["string",'
        '"\\b\\t\\n\\f\\r\\u0000\\u001f/\\"\\\\é\u2028\u2029"]]'
    )
    assert not encoded.endswith("\n")
