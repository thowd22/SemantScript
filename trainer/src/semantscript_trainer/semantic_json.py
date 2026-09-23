"""Compiler-compatible ``semantscript-semantic-json`` version 1 encoding."""

from __future__ import annotations

import hashlib
import json
import math
import struct

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type TypedJsonNode = list[str | bool | int | list[TypedJsonNode] | list[list[str | TypedJsonNode]]]

_DOMAIN = "semantscript-semantic-json"
_VERSION = 1


def semantic_json_bytes(value: JsonValue, /) -> bytes:
    """Encode a JSON value using the compiler's typed semantic JSON v1 format."""

    return semantic_json_string(value).encode("utf-8")


def semantic_json_string(value: JsonValue, /) -> str:
    """Return the exact diagnostic string emitted by the TypeScript compiler."""

    document = [_DOMAIN, _VERSION, _to_typed_node(value)]
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


def semantic_json_sha256(value: JsonValue, /) -> str:
    """Return the lowercase SHA-256 digest of typed semantic JSON v1 bytes."""

    return hashlib.sha256(semantic_json_bytes(value)).hexdigest()


def _to_typed_node(value: JsonValue) -> TypedJsonNode:
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["boolean", value]
    if type(value) in (int, float):
        return ["number", _number_to_binary64_hex(value)]
    if type(value) is str:
        _assert_unicode_scalar_string(value)
        return ["string", value]
    if type(value) is list:
        return ["array", [_to_typed_node(item) for item in value]]
    if type(value) is dict:
        entries: list[list[str | TypedJsonNode]] = []
        for name, entry_value in value.items():
            if type(name) is not str:
                raise TypeError("semantic JSON object keys must be strings")
            _assert_unicode_scalar_string(name)
            entries.append([name, _to_typed_node(entry_value)])
        entries.sort(key=lambda entry: entry[0].encode("utf-8"))
        return ["object", entries]
    raise TypeError("semantic JSON values must contain only JSON-compatible data")


def _number_to_binary64_hex(value: int | float) -> str:
    try:
        binary64 = float(value)
    except OverflowError as error:
        raise TypeError("semantic JSON numbers must be finite") from error
    if not math.isfinite(binary64):
        raise TypeError("semantic JSON numbers must be finite")
    return struct.pack(">d", binary64).hex()


def _assert_unicode_scalar_string(value: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise TypeError("semantic JSON strings cannot contain unpaired UTF-16 surrogates")


__all__ = [
    "JsonValue",
    "semantic_json_bytes",
    "semantic_json_sha256",
    "semantic_json_string",
]
