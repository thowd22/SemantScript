"""Canonical, type-aware serialization of validated SemantScript inputs."""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

MAXIMUM_INPUT_DEPTH = 100
MAXIMUM_INPUT_NODES = 100_000
MAXIMUM_SAFE_INTEGER = 2**53 - 1

type InputPath = tuple[str | int, ...]
type InputErrorReason = Literal[
    "accessor",
    "array-property",
    "class-instance",
    "cycle",
    "extra",
    "invalid-unicode",
    "limit",
    "missing",
    "no-union-variant",
    "non-finite",
    "promise",
    "proxy",
    "schema",
    "sparse-array",
    "symbol-key",
    "tuple-length",
    "type",
    "value",
]
type ExactJson = str | int | bool | list["ExactJson"]

_FATAL_REASONS = frozenset(
    {
        "accessor",
        "array-property",
        "class-instance",
        "cycle",
        "invalid-unicode",
        "limit",
        "non-finite",
        "promise",
        "proxy",
        "sparse-array",
        "symbol-key",
    }
)
_MISSING = object()


class CanonicalInputError(TypeError):
    """A schema or input cannot be represented as canonical-input/v1."""

    code = "SEMA_INPUT_INVALID"

    def __init__(self, reason: InputErrorReason, path: InputPath, message: str) -> None:
        super().__init__(f"{_format_path(path)}: {message}")
        self.reason = reason
        self.path = path


@dataclass(slots=True)
class _WorkBudget:
    remaining: int = MAXIMUM_INPUT_NODES


@dataclass(slots=True)
class _EncodeContext:
    active: set[int]
    budget: _WorkBudget


def serialize_canonical_inputs(
    schema: Sequence[Mapping[str, Any]],
    inputs: object,
    *,
    maximum_bytes: int | None = None,
) -> bytes:
    """Validate and serialize inputs as ``semantscript.canonical-input/v1`` bytes."""

    if maximum_bytes is not None and (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes < 1
        or maximum_bytes > MAXIMUM_SAFE_INTEGER
    ):
        raise ValueError("maximum_bytes must be a positive safe integer")

    envelope = _encode_envelope(schema, inputs)
    writer = _ExactJsonWriter(maximum_bytes)
    _write_exact_json(envelope, writer)
    return writer.finish()


def serialize_canonical_inputs_string(
    schema: Sequence[Mapping[str, Any]],
    inputs: object,
) -> str:
    """Return the diagnostic string form used by cross-language golden tests."""

    return serialize_canonical_inputs(schema, inputs).decode("utf-8", errors="strict")


def _encode_envelope(
    schema: Sequence[Mapping[str, Any]],
    inputs: object,
) -> ExactJson:
    ordered = _validate_and_order_schema(schema)
    values = _plain_object(inputs, (), "input record")
    expected_names = {entry[0] for entry in ordered}

    for name in values:
        _require_string(name, (name,) if isinstance(name, str) else ())
        if name not in expected_names:
            _fail("extra", (name,), f"unexpected input {_quote_message(name)}")

    context = _EncodeContext(active=set(), budget=_WorkBudget())
    pairs: list[ExactJson] = []
    for name, _, type_spec in ordered:
        path = (name,)
        value = values.get(name, _MISSING)
        if value is _MISSING:
            _fail("missing", path, f"missing required input {_quote_message(name)}")
        pairs.append([name, _encode_value(type_spec, value, path, context, 0)])
    return ["semantscript-input", 1, pairs]


def _encode_value(
    type_spec: Mapping[str, Any],
    value: object,
    path: InputPath,
    context: _EncodeContext,
    depth: int,
) -> ExactJson:
    _consume_work(context.budget, path, depth)
    _reject_host_object(value, path)
    kind = type_spec.get("kind")

    if kind == "null":
        if value is not None:
            _expected_type(path, "null")
        return ["null"]
    if kind == "boolean":
        if type(value) is not bool:
            _expected_type(path, "boolean")
        return ["boolean", value]
    if kind == "string":
        if type(value) is not str:
            _expected_type(path, "string")
        _require_unicode(value, path)
        return ["string", value]
    if kind == "number":
        return _encode_number(value, path)
    if kind == "literal":
        expected = type_spec.get("value", _MISSING)
        if expected is _MISSING:
            _schema_failure("literal input schema requires value")
        encoded = _encode_primitive(value, path)
        if not _same_primitive(value, expected):
            _fail("value", path, "value does not match the required literal")
        return ["literal", encoded]
    if kind == "enum":
        return _encode_enum(type_spec, value, path)
    if kind == "array":
        return _encode_array(type_spec, value, path, context, depth)
    if kind == "tuple":
        return _encode_tuple(type_spec, value, path, context, depth)
    if kind == "object":
        return _encode_object(type_spec, value, path, context, depth)
    if kind == "union":
        return _encode_union(type_spec, value, path, context, depth)
    _schema_failure(f"unsupported input kind {kind!r}")


def _encode_primitive(value: object, path: InputPath) -> ExactJson:
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["boolean", value]
    if type(value) is str:
        _require_unicode(value, path)
        return ["string", value]
    if _is_number(value):
        return _encode_number(value, path)
    _expected_type(path, "primitive literal")


def _encode_number(value: object, path: InputPath) -> ExactJson:
    return ["number", _number_hex(value, path)]


def _encode_enum(
    type_spec: Mapping[str, Any],
    value: object,
    path: InputPath,
) -> ExactJson:
    name = _schema_string(type_spec.get("name"), "enum name")
    base = type_spec.get("base")
    values = _schema_sequence(type_spec.get("values"), f"enum {name} values")

    if base == "string":
        if type(value) is not str:
            _expected_type(path, f"member of enum {name}")
        _require_unicode(value, path)
        index = next(
            (position for position, candidate in enumerate(values) if candidate == value),
            -1,
        )
        encoded: ExactJson = ["string", value]
    elif base == "number":
        encoded = _encode_number(value, path)
        key = _number_hex(value, path)
        index = next(
            (
                position
                for position, candidate in enumerate(values)
                if _is_number(candidate) and _number_hex(candidate, ("$schema", name)) == key
            ),
            -1,
        )
    else:
        _schema_failure("enum input schema base must be 'string' or 'number'")

    if index < 0:
        _fail("value", path, f"value is not a member of enum {name}")
    return ["enum", name, index, encoded]


def _encode_array(
    type_spec: Mapping[str, Any],
    value: object,
    path: InputPath,
    context: _EncodeContext,
    depth: int,
) -> ExactJson:
    if type(value) is not list:
        _expected_type(path, "array")
    if len(value) > context.budget.remaining:
        _fail(
            "limit",
            path,
            f"input work exceeds the limit of {MAXIMUM_INPUT_NODES} nodes",
        )
    item_type = _schema_mapping(type_spec.get("items"), "array items")
    return _with_active(
        value,
        path,
        context,
        lambda: [
            "array",
            [
                _encode_value(item_type, item, (*path, index), context, depth + 1)
                for index, item in enumerate(value)
            ],
        ],
    )


def _encode_tuple(
    type_spec: Mapping[str, Any],
    value: object,
    path: InputPath,
    context: _EncodeContext,
    depth: int,
) -> ExactJson:
    if type(value) is not list:
        _expected_type(path, "tuple")
    items = _schema_sequence(type_spec.get("items"), "tuple items")
    if len(value) != len(items):
        _fail(
            "tuple-length",
            path,
            f"expected tuple length {len(items)}, received {len(value)}",
        )

    def encode() -> ExactJson:
        encoded: list[ExactJson] = []
        for index, (item_type, item) in enumerate(zip(items, value, strict=True)):
            encoded.append(
                _encode_value(
                    _schema_mapping(item_type, "tuple item"),
                    item,
                    (*path, index),
                    context,
                    depth + 1,
                )
            )
        return ["tuple", encoded]

    return _with_active(value, path, context, encode)


def _encode_object(
    type_spec: Mapping[str, Any],
    value: object,
    path: InputPath,
    context: _EncodeContext,
    depth: int,
) -> ExactJson:
    name = _schema_string(type_spec.get("name"), "object name")
    values = _plain_object(value, path, f"object {name}")
    fields = _object_fields(type_spec, name)
    by_name = {field_name: (optional, field_type) for field_name, optional, field_type in fields}

    for key in values:
        _require_string(key, (*path, key) if isinstance(key, str) else path)
        if key not in by_name:
            _fail("extra", (*path, key), f"unexpected property {_quote_message(key)}")

    def encode() -> ExactJson:
        pairs: list[tuple[str, ExactJson]] = []
        for field_name, optional, field_type in fields:
            field_path = (*path, field_name)
            field_value = values.get(field_name, _MISSING)
            if field_value is _MISSING:
                if not optional:
                    _fail(
                        "missing",
                        field_path,
                        f"missing required property {_quote_message(field_name)}",
                    )
                continue
            pairs.append(
                (
                    field_name,
                    _encode_value(field_type, field_value, field_path, context, depth + 1),
                )
            )
        pairs.sort(key=lambda pair: pair[0].encode("utf-8", errors="strict"))
        return ["object", [[field_name, encoded] for field_name, encoded in pairs]]

    return _with_active(values, path, context, encode)


def _encode_union(
    type_spec: Mapping[str, Any],
    value: object,
    path: InputPath,
    context: _EncodeContext,
    depth: int,
) -> ExactJson:
    variants = _schema_sequence(type_spec.get("variants"), "union variants")
    for index, raw_variant in enumerate(variants):
        variant = _schema_mapping(raw_variant, "union variant")
        try:
            return ["union", index, _encode_value(variant, value, path, context, depth + 1)]
        except CanonicalInputError as error:
            if error.reason in _FATAL_REASONS:
                raise
    _fail("no-union-variant", path, "value does not match any union variant")


def _with_active(
    value: object,
    path: InputPath,
    context: _EncodeContext,
    encode: Any,
) -> ExactJson:
    identity = id(value)
    if identity in context.active:
        _fail("cycle", path, "cyclic input values are not supported")
    context.active.add(identity)
    try:
        return encode()
    finally:
        context.active.remove(identity)


def _validate_and_order_schema(
    schema: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, int, Mapping[str, Any]], ...]:
    if type(schema) not in (list, tuple):
        _schema_failure("input schema must be an array")

    entries: list[tuple[str, int, Mapping[str, Any]]] = []
    for raw_entry in schema:
        entry = _schema_mapping(raw_entry, "input entry")
        name = _schema_string(entry.get("name"), "input name")
        index = entry.get("index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index > MAXIMUM_SAFE_INTEGER
        ):
            _schema_failure("input index must be a non-negative safe integer")
        entries.append((name, index, _schema_mapping(entry.get("type"), f"input {name} type")))

    ordered = tuple(sorted(entries, key=lambda entry: entry[1]))
    names: set[str] = set()
    budget = _WorkBudget()
    for expected_index, (name, index, type_spec) in enumerate(ordered):
        if index != expected_index:
            _schema_failure(
                f"input indices must be unique and dense from zero; missing index {expected_index}"
            )
        _require_unicode(name, ("$schema", expected_index, "name"))
        if name in names:
            _schema_failure(f"duplicate input name {_quote_message(name)}")
        names.add(name)
        _validate_input_type(type_spec, set(), budget, 0)
    return ordered


def _validate_input_type(
    type_spec: Mapping[str, Any],
    active: set[int],
    budget: _WorkBudget,
    depth: int,
) -> tuple[Any, ...]:
    _consume_work(budget, ("$schema",), depth)
    identity = id(type_spec)
    if identity in active:
        _schema_failure("input schemas cannot be recursive at runtime")
    active.add(identity)
    try:
        kind = type_spec.get("kind")
        if kind in ("null", "boolean", "string", "number"):
            return (kind,)
        if kind == "literal":
            value = type_spec.get("value", _MISSING)
            if value is _MISSING:
                _schema_failure("literal input schema requires value")
            return ("literal", _primitive_schema_key(value))
        if kind == "enum":
            name = _schema_string(type_spec.get("name"), "enum name")
            base = type_spec.get("base")
            values = _schema_sequence(type_spec.get("values"), f"enum {name} values")
            _require_unicode(name, ("$schema", "enum", "name"))
            if not name or not values:
                _schema_failure("enum input schemas require a name and at least one value")
            keys: list[tuple[Any, ...]] = []
            for value in values:
                if base == "string":
                    if type(value) is not str:
                        _schema_failure("string enums can contain only strings")
                    _require_unicode(value, ("$schema", name))
                elif base == "number":
                    if not _is_number(value):
                        _schema_failure("number enums can contain only finite numbers")
                    _finite_float(value, ("$schema", name), schema=True)
                else:
                    _schema_failure("enum input schema base must be 'string' or 'number'")
                keys.append(_primitive_schema_key(value))
            if len(set(keys)) != len(keys):
                _schema_failure(f"enum {name} contains duplicate values")
            return ("enum", name, base, tuple(keys))
        if kind == "array":
            item = _schema_mapping(type_spec.get("items"), "array items")
            return ("array", _validate_input_type(item, active, budget, depth + 1))
        if kind == "tuple":
            items = _schema_sequence(type_spec.get("items"), "tuple items")
            return (
                "tuple",
                tuple(
                    _validate_input_type(
                        _schema_mapping(item, "tuple item"), active, budget, depth + 1
                    )
                    for item in items
                ),
            )
        if kind == "object":
            name = _schema_string(type_spec.get("name"), "object name")
            _require_unicode(name, ("$schema", "object", "name"))
            fields = _object_fields(type_spec, name)
            keys: list[tuple[Any, ...]] = []
            for field_name, optional, field_type in fields:
                keys.append(
                    (
                        field_name,
                        optional,
                        _validate_input_type(field_type, active, budget, depth + 1),
                    )
                )
            return ("object", name, tuple(keys))
        if kind == "union":
            variants = _schema_sequence(type_spec.get("variants"), "union variants")
            if len(variants) < 2:
                _schema_failure("union input schemas require at least two variants")
            keys = tuple(
                _validate_input_type(
                    _schema_mapping(variant, "union variant"), active, budget, depth + 1
                )
                for variant in variants
            )
            if len(set(keys)) != len(keys):
                _schema_failure("union input schema contains duplicate variants")
            return ("union", keys)
        _schema_failure(f"unsupported input kind {kind!r}")
    finally:
        active.remove(identity)


def _object_fields(
    type_spec: Mapping[str, Any],
    object_name: str,
) -> tuple[tuple[str, bool, Mapping[str, Any]], ...]:
    raw_fields = _schema_sequence(type_spec.get("fields"), f"object {object_name} fields")
    fields: list[tuple[str, bool, Mapping[str, Any]]] = []
    names: set[str] = set()
    for raw_field in raw_fields:
        field = _schema_mapping(raw_field, f"object {object_name} field")
        name = _schema_string(field.get("name"), "object field name")
        _require_unicode(name, ("$schema", object_name, "field"))
        if name in names:
            _schema_failure(f"object {object_name} contains duplicate field {_quote_message(name)}")
        optional = field.get("optional")
        if type(optional) is not bool:
            _schema_failure(f"object {object_name} field optional must be boolean")
        names.add(name)
        fields.append(
            (
                name,
                optional,
                _schema_mapping(field.get("type"), f"object {object_name} field type"),
            )
        )
    return tuple(fields)


class _ExactJsonWriter:
    __slots__ = ("_bytes", "_maximum_bytes")

    def __init__(self, maximum_bytes: int | None) -> None:
        self._bytes = bytearray()
        self._maximum_bytes = maximum_bytes

    def write_ascii(self, value: str) -> None:
        self.write_bytes(value.encode("ascii"))

    def write_text(self, value: str) -> None:
        for start in range(0, len(value), 4096):
            self.write_bytes(value[start : start + 4096].encode("utf-8", errors="strict"))

    def write_bytes(self, value: bytes) -> None:
        if self._maximum_bytes is not None and len(self._bytes) + len(value) > self._maximum_bytes:
            _fail(
                "limit",
                (),
                f"canonical input exceeds the {self._maximum_bytes} byte limit",
            )
        self._bytes.extend(value)

    def write_string(self, value: str) -> None:
        self.write_ascii('"')
        unescaped_start = 0
        escapes = {
            "\b": "\\b",
            "\t": "\\t",
            "\n": "\\n",
            "\f": "\\f",
            "\r": "\\r",
            '"': '\\"',
            "\\": "\\\\",
        }
        for index, character in enumerate(value):
            code = ord(character)
            escape = escapes.get(character)
            if escape is None and code <= 0x1F:
                escape = f"\\u00{code:02x}"
            if escape is not None:
                self.write_text(value[unescaped_start:index])
                self.write_ascii(escape)
                unescaped_start = index + 1
            elif 0xD800 <= code <= 0xDFFF:
                raise TypeError("canonical input strings cannot contain unpaired UTF-16 surrogates")
        self.write_text(value[unescaped_start:])
        self.write_ascii('"')

    def finish(self) -> bytes:
        return bytes(self._bytes)


def _write_exact_json(value: ExactJson, writer: _ExactJsonWriter) -> None:
    if type(value) is str:
        writer.write_string(value)
        return
    if type(value) is bool:
        writer.write_ascii("true" if value else "false")
        return
    if type(value) is int:
        if value < 0 or value > MAXIMUM_SAFE_INTEGER:
            raise TypeError("canonical input envelope integers must be unsigned safe integers")
        writer.write_ascii(str(value))
        return
    if type(value) is not list:
        raise TypeError("canonical input envelope contains an unsupported value")
    writer.write_ascii("[")
    for index, item in enumerate(value):
        if index:
            writer.write_ascii(",")
        _write_exact_json(item, writer)
    writer.write_ascii("]")


def _number_hex(value: object, path: InputPath) -> str:
    return struct.pack(">d", _finite_float(value, path)).hex()


def _finite_float(value: object, path: InputPath, *, schema: bool = False) -> float:
    if not _is_number(value):
        if schema:
            _schema_failure("number schemas can contain only finite numbers")
        _expected_type(path, "number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        if schema:
            _schema_failure("number schemas can contain only finite numbers")
        raise CanonicalInputError("non-finite", path, "numbers must be finite") from error
    if not math.isfinite(result):
        if schema:
            _schema_failure("number schemas can contain only finite numbers")
        _fail("non-finite", path, "numbers must be finite")
    return result


def _primitive_schema_key(value: object) -> tuple[Any, ...]:
    if value is None:
        return ("null",)
    if type(value) is bool:
        return ("boolean", value)
    if type(value) is str:
        _require_unicode(value, ("$schema", "literal"))
        return ("string", value)
    if _is_number(value):
        return ("number", struct.pack(">d", _finite_float(value, ("$schema",), schema=True)))
    _schema_failure("literal input schemas require a finite primitive value")


def _same_primitive(value: object, expected: object) -> bool:
    if _is_number(expected):
        return _is_number(value) and _number_hex(value, ()) == _number_hex(expected, ("$schema",))
    return type(value) is type(expected) and value == expected


def _plain_object(value: object, path: InputPath, expected: str) -> dict[str, object]:
    _reject_host_object(value, path)
    if type(value) is not dict:
        _expected_type(path, expected)
    return value


def _reject_host_object(value: object, path: InputPath) -> None:
    if value is None or type(value) in (bool, int, float, str, list, dict):
        return
    if callable(value):
        return
    _fail("class-instance", path, "class instances and host objects are not supported")


def _consume_work(budget: _WorkBudget, path: InputPath, depth: int) -> None:
    if depth > MAXIMUM_INPUT_DEPTH:
        _fail(
            "limit",
            path,
            f"input nesting exceeds the limit of {MAXIMUM_INPUT_DEPTH}",
        )
    budget.remaining -= 1
    if budget.remaining < 0:
        _fail(
            "limit",
            path,
            f"input work exceeds the limit of {MAXIMUM_INPUT_NODES} nodes",
        )


def _schema_mapping(value: object, description: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        _schema_failure(f"{description} must be an object")
    return value


def _schema_sequence(value: object, description: str) -> Sequence[Any]:
    if type(value) not in (list, tuple):
        _schema_failure(f"{description} must be an array")
    return value


def _schema_string(value: object, description: str) -> str:
    if type(value) is not str:
        _schema_failure(f"{description} must be a string")
    return value


def _require_string(value: object, path: InputPath) -> None:
    if type(value) is not str:
        _fail("type", path, "object property names must be strings")
    _require_unicode(value, path)


def _require_unicode(value: str, path: InputPath) -> None:
    # Scan without first materializing another potentially large UTF-8 buffer.
    # Valid Python strings represent astral scalars directly; surrogate code
    # points can only arrive through deliberately malformed host input.
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        _fail(
            "invalid-unicode",
            path,
            "strings cannot contain unpaired UTF-16 surrogates",
        )


def _is_number(value: object) -> bool:
    return type(value) in (int, float)


def _expected_type(path: InputPath, expected: str) -> NoReturn:
    _fail("type", path, f"expected {expected}")


def _schema_failure(message: str) -> NoReturn:
    _fail("schema", ("$schema",), message)


def _fail(reason: InputErrorReason, path: InputPath, message: str) -> NoReturn:
    raise CanonicalInputError(reason, path, message)


def _quote_message(value: str) -> str:
    writer = _ExactJsonWriter(None)
    try:
        writer.write_string(value)
    except TypeError:
        # Diagnostics must still be constructible for the invalid-surrogate case
        # that the canonical payload writer itself rejects.
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return writer.finish().decode("utf-8")


def _format_path(path: InputPath) -> str:
    if not path:
        return "inputs"
    pieces = ["inputs"]
    for segment in path:
        pieces.append(
            f"[{segment}]" if isinstance(segment, int) else f"[{_quote_message(segment)}]"
        )
    return "".join(pieces)


__all__ = [
    "MAXIMUM_INPUT_DEPTH",
    "MAXIMUM_INPUT_NODES",
    "CanonicalInputError",
    "InputErrorReason",
    "InputPath",
    "serialize_canonical_inputs",
    "serialize_canonical_inputs_string",
]
