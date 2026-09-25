"""Resource-bounded strict JSON parsing for untrusted teacher responses."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import cast

from .teacher import JsonValue


class StrictJsonError(ValueError):
    """JSON is malformed, ambiguous, or exceeds a configured resource limit."""


@dataclass(frozen=True, slots=True)
class StrictJsonLimits:
    """Byte, depth and node ceilings a strict JSON document may not exceed."""

    maximum_bytes: int = 8 * 1024 * 1024
    maximum_depth: int = 100
    maximum_nodes: int = 100_000

    def __post_init__(self) -> None:
        for name in ("maximum_bytes", "maximum_nodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.maximum_depth, bool)
            or not isinstance(self.maximum_depth, int)
            or self.maximum_depth < 0
        ):
            raise ValueError("maximum_depth must be a non-negative integer")


DEFAULT_STRICT_JSON_LIMITS = StrictJsonLimits()


def loads_strict_json(
    document: str | bytes | bytearray,
    *,
    limits: StrictJsonLimits = DEFAULT_STRICT_JSON_LIMITS,
) -> JsonValue:
    """Parse one UTF-8 JSON value with duplicate keys and ambiguity rejected."""

    text, size = _decode(document)
    if size > limits.maximum_bytes:
        raise StrictJsonError(
            f"JSON document exceeds maximum UTF-8 byte length {limits.maximum_bytes}"
        )
    if text.startswith("\ufeff"):
        raise StrictJsonError("JSON document must not contain a byte order mark")
    _check_lexical_depth(text, limits.maximum_depth)

    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_parse_integer,
            parse_float=_parse_float,
            parse_constant=_reject_constant,
        )
    except StrictJsonError:
        raise
    except (ValueError, OverflowError, RecursionError) as error:
        raise StrictJsonError(f"invalid JSON: {error}") from error

    _check_value(value, limits.maximum_nodes)
    return cast(JsonValue, value)


def _decode(document: str | bytes | bytearray) -> tuple[str, int]:
    if isinstance(document, str):
        try:
            encoded = document.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise StrictJsonError("JSON contains an unpaired Unicode surrogate") from error
        return document, len(encoded)
    if isinstance(document, (bytes, bytearray)):
        encoded = bytes(document)
        try:
            return encoded.decode("utf-8", errors="strict"), len(encoded)
        except UnicodeDecodeError as error:
            raise StrictJsonError("JSON document is not valid UTF-8") from error
    raise TypeError("JSON document must be str, bytes, or bytearray")


def _check_lexical_depth(text: str, maximum_depth: int) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > maximum_depth:
                raise StrictJsonError(f"JSON exceeds maximum depth {maximum_depth}")
        elif character in "]}":
            depth -= 1


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise StrictJsonError(f"duplicate object property {name!r}")
        result[name] = value
    return result


def _parse_integer(token: str) -> float:
    # IR semantic JSON requires every numeric token to be rounded directly to
    # binary64, including lexically integral values and signed zero.
    return _parse_float(token)


def _parse_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise StrictJsonError("JSON number does not fit finite binary64")
    return value


def _reject_constant(token: str) -> object:
    raise StrictJsonError(f"non-finite JSON constant {token} is forbidden")


def _check_value(value: object, maximum_nodes: int) -> None:
    work: list[object] = [value]
    nodes = 0
    while work:
        current = work.pop()
        nodes += 1
        if nodes > maximum_nodes:
            raise StrictJsonError(f"JSON exceeds maximum node count {maximum_nodes}")
        if isinstance(current, str):
            try:
                current.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise StrictJsonError("JSON contains an unpaired Unicode surrogate") from error
        elif isinstance(current, list):
            work.extend(current)
        elif isinstance(current, dict):
            for name, child in current.items():
                try:
                    name.encode("utf-8", errors="strict")
                except UnicodeEncodeError as error:
                    raise StrictJsonError("JSON contains an unpaired Unicode surrogate") from error
                work.append(child)


__all__ = [
    "DEFAULT_STRICT_JSON_LIMITS",
    "StrictJsonError",
    "StrictJsonLimits",
    "loads_strict_json",
]
