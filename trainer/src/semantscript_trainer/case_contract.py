"""IR-derived structured-output schema and exact generated-case validation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, NoReturn, cast

from .strict_json import (
    DEFAULT_STRICT_JSON_LIMITS,
    StrictJsonError,
    StrictJsonLimits,
    loads_strict_json,
)
from .teacher import (
    GeneratedCase,
    JsonValue,
    NeuralFunctionIr,
    TeacherConfigurationError,
    TeacherResponseError,
)

MAXIMUM_CASE_COUNT = 100_000


def validate_case_count(n: int) -> int:
    """Return ``n`` when it is a non-negative case count within the configured maximum."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise TeacherConfigurationError("case count must be a non-negative integer")
    if n > MAXIMUM_CASE_COUNT:
        raise TeacherConfigurationError(f"case count exceeds maximum {MAXIMUM_CASE_COUNT}")
    return n


def build_case_schema(ir: NeuralFunctionIr) -> dict[str, JsonValue]:
    """Build the closed one-case JSON Schema sent to structured-output providers."""

    inputs = _ir_inputs(ir)
    properties: dict[str, JsonValue] = {}
    required: list[JsonValue] = []
    for entry in inputs:
        name = _string_member(entry, "name", "IR input")
        properties[name] = cast(JsonValue, _input_type_schema(_mapping_member(entry, "type", name)))
        required.append(name)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["inputs", "output"],
        "properties": {
            "inputs": {
                "type": "object",
                "additionalProperties": False,
                "required": required,
                "properties": properties,
            },
            "output": _output_schema(_ir_output(ir)),
        },
    }


def parse_case_response(
    ir: NeuralFunctionIr,
    response: str | bytes | bytearray,
    *,
    limits: StrictJsonLimits = DEFAULT_STRICT_JSON_LIMITS,
) -> GeneratedCase:
    """Strictly parse and validate exactly one provider case response."""

    try:
        value = loads_strict_json(response, limits=limits)
    except StrictJsonError as error:
        raise TeacherResponseError(f"teacher returned invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise TeacherResponseError("teacher response must be a JSON object")
    if set(value) != {"inputs", "output"}:
        raise TeacherResponseError("teacher response must contain exactly inputs and output")
    inputs = value["inputs"]
    if not isinstance(inputs, dict):
        raise TeacherResponseError("teacher response inputs must be a JSON object")
    generated = GeneratedCase(inputs=inputs, output=value["output"])
    return validate_case(ir, generated)


def validate_case(ir: NeuralFunctionIr, case: GeneratedCase) -> GeneratedCase:
    """Validate one already-decoded case against IR inputs and output support."""

    if not isinstance(case, GeneratedCase):
        raise TeacherResponseError("teacher cases must be GeneratedCase values")
    entries = _ir_inputs(ir)
    expected_names = [_string_member(entry, "name", "IR input") for entry in entries]
    if set(case.inputs) != set(expected_names) or len(case.inputs) != len(expected_names):
        raise TeacherResponseError(
            "generated case input names must exactly match the neural-function inputs"
        )
    for entry in entries:
        name = _string_member(entry, "name", "IR input")
        _validate_input_value(
            case.inputs[name],
            _mapping_member(entry, "type", name),
            f"inputs.{name}",
        )
    _validate_output_value(case.output, _ir_output(ir), "output")
    return case


def validate_constraint_output(ir: NeuralFunctionIr, value: JsonValue) -> None:
    """Validate one v1 constraint label against the declared scalar head."""

    output = _ir_output(ir)
    if output.get("kind") != "scalar":
        _bad_ir("v1 constraints require a scalar output")
    head = _mapping_member(output, "head", "constraint output")
    if not any(_same_constraint_scalar(value, item) for item in _head_support(head)):
        _invalid("constraint output", "is outside the declared output support")


def validate_generated_cases(
    ir: NeuralFunctionIr,
    cases: Iterable[GeneratedCase],
    *,
    expected_count: int,
) -> tuple[GeneratedCase, ...]:
    """Validate exactly ``expected_count`` teacher cases against the IR and return them as a tuple."""
    expected = validate_case_count(expected_count)
    try:
        iterator = iter(cases)
    except TypeError as error:
        raise TeacherResponseError("teacher cases must be iterable") from error

    resolved: list[GeneratedCase] = []
    for index in range(expected):
        try:
            case = next(iterator)
        except StopIteration as error:
            raise TeacherResponseError(
                f"teacher returned {index} cases; expected {expected}"
            ) from error
        resolved.append(validate_case(ir, case))

    try:
        next(iterator)
    except StopIteration:
        return tuple(resolved)
    raise TeacherResponseError(f"teacher returned more than {expected} cases; expected {expected}")


def _ir_inputs(ir: NeuralFunctionIr) -> list[Mapping[str, Any]]:
    value = ir.get("inputs")
    if not isinstance(value, list):
        raise TeacherConfigurationError("IR inputs must be an array")
    entries: list[Mapping[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise TeacherConfigurationError("IR input entries must be objects")
        name = _string_member(raw, "name", "IR input")
        if name in names or raw.get("index") != index:
            raise TeacherConfigurationError("IR input names must be unique and indices dense")
        names.add(name)
        entries.append(raw)
    return entries


def _ir_output(ir: NeuralFunctionIr) -> Mapping[str, Any]:
    value = ir.get("output")
    if not isinstance(value, Mapping):
        raise TeacherConfigurationError("IR output must be an object")
    return value


def _input_type_schema(spec: Mapping[str, Any]) -> dict[str, JsonValue]:
    kind = spec.get("kind")
    if kind in ("string", "boolean", "number", "null"):
        return {"type": {"null": "null"}.get(cast(str, kind), cast(str, kind))}
    if kind == "literal":
        return {"const": cast(JsonValue, spec.get("value"))}
    if kind == "enum":
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            _bad_ir("enum input requires nonempty values")
        return {"enum": cast(list[JsonValue], values)}
    if kind == "array":
        return {
            "type": "array",
            "items": _input_type_schema(_mapping_member(spec, "items", "array")),
        }
    if kind == "tuple":
        items = spec.get("items")
        if not isinstance(items, list):
            _bad_ir("tuple input requires items")
        schemas = [_input_type_schema(_as_mapping(item, "tuple item")) for item in items]
        return {
            "type": "array",
            "prefixItems": cast(list[JsonValue], schemas),
            "items": False,
            "minItems": len(schemas),
            "maxItems": len(schemas),
        }
    if kind == "object":
        fields = spec.get("fields")
        if not isinstance(fields, list):
            _bad_ir("object input requires fields")
        properties: dict[str, JsonValue] = {}
        required: list[JsonValue] = []
        for raw in fields:
            field = _as_mapping(raw, "object field")
            name = _string_member(field, "name", "object field", allow_empty=True)
            if name in properties:
                _bad_ir(f"duplicate object field {name!r}")
            properties[name] = cast(
                JsonValue, _input_type_schema(_mapping_member(field, "type", name))
            )
            optional = field.get("optional")
            if not isinstance(optional, bool):
                _bad_ir(f"object field {name!r} has invalid optional flag")
            if not optional:
                required.append(name)
        return {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }
    if kind == "union":
        variants = spec.get("variants")
        if not isinstance(variants, list) or len(variants) < 2:
            _bad_ir("union input requires at least two variants")
        return {
            "anyOf": cast(
                list[JsonValue],
                [_input_type_schema(_as_mapping(item, "union variant")) for item in variants],
            )
        }
    _bad_ir(f"unsupported input kind {kind!r}")


def _output_schema(spec: Mapping[str, Any]) -> dict[str, JsonValue]:
    kind = spec.get("kind")
    if kind == "scalar":
        return _head_schema(_mapping_member(spec, "head", "scalar output"))
    if kind == "object":
        fields = spec.get("fields")
        if not isinstance(fields, list) or not fields:
            _bad_ir("object output requires fields")
        properties: dict[str, JsonValue] = {}
        required: list[JsonValue] = []
        for raw in fields:
            field = _as_mapping(raw, "output field")
            name = _string_member(field, "name", "output field", allow_empty=True)
            if name in properties:
                _bad_ir(f"duplicate output field {name!r}")
            properties[name] = cast(JsonValue, _head_schema(_mapping_member(field, "head", name)))
            required.append(name)
        return {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }
    _bad_ir(f"unsupported output kind {kind!r}")


def _head_schema(head: Mapping[str, Any]) -> dict[str, JsonValue]:
    return {"enum": _head_support(head)}


def _head_support(head: Mapping[str, Any]) -> list[JsonValue]:
    support = head.get("support")
    if isinstance(support, list) and support:
        return cast(list[JsonValue], support)
    decimals = head.get("supportDecimal")
    if not isinstance(decimals, list) or not decimals:
        _bad_ir("output head requires support or supportDecimal")
    integral = head.get("sourceKind") == "bounded-int"
    result: list[JsonValue] = []
    for token in decimals:
        if not isinstance(token, str):
            _bad_ir("supportDecimal values must be strings")
        try:
            decimal = Decimal(token)
        except InvalidOperation as error:
            raise TeacherConfigurationError(f"invalid support decimal {token!r}") from error
        value: int | float = int(decimal) if integral else float(decimal)
        if not math.isfinite(value):
            _bad_ir("support decimal must convert to finite binary64")
        result.append(value)
    return result


def _validate_input_value(value: JsonValue, spec: Mapping[str, Any], path: str) -> None:
    kind = spec.get("kind")
    if kind == "string":
        if not isinstance(value, str):
            _invalid(path, "must be a string")
        return
    if kind == "boolean":
        if not isinstance(value, bool):
            _invalid(path, "must be a boolean")
        return
    if kind == "number":
        if not _finite_number(value):
            _invalid(path, "must be a finite number")
        return
    if kind == "null":
        if value is not None:
            _invalid(path, "must be null")
        return
    if kind == "literal":
        if not _same_json_scalar(value, cast(JsonValue, spec.get("value"))):
            _invalid(path, "does not equal its literal type")
        return
    if kind == "enum":
        values = spec.get("values")
        if not isinstance(values, list) or not any(
            _same_json_scalar(value, cast(JsonValue, item)) for item in values
        ):
            _invalid(path, "is not an enum member")
        return
    if kind == "array":
        if not isinstance(value, list):
            _invalid(path, "must be an array")
        item_spec = _mapping_member(spec, "items", "array")
        for index, item in enumerate(value):
            _validate_input_value(item, item_spec, f"{path}[{index}]")
        return
    if kind == "tuple":
        items = spec.get("items")
        if not isinstance(value, list) or not isinstance(items, list) or len(value) != len(items):
            _invalid(path, "must have the exact tuple length")
        for index, (item, item_spec) in enumerate(zip(value, items, strict=True)):
            _validate_input_value(item, _as_mapping(item_spec, "tuple item"), f"{path}[{index}]")
        return
    if kind == "object":
        if not isinstance(value, dict):
            _invalid(path, "must be an object")
        fields = spec.get("fields")
        if not isinstance(fields, list):
            _bad_ir("object input requires fields")
        by_name = {
            _string_member(
                _as_mapping(item, "object field"), "name", "object field", allow_empty=True
            ): _as_mapping(item, "object field")
            for item in fields
        }
        extras = set(value) - set(by_name)
        missing = {
            name
            for name, field in by_name.items()
            if not field.get("optional") and name not in value
        }
        if extras or missing:
            _invalid(path, "has missing or additional fields")
        for name, item in value.items():
            _validate_input_value(
                item, _mapping_member(by_name[name], "type", name), f"{path}.{name}"
            )
        return
    if kind == "union":
        variants = spec.get("variants")
        if not isinstance(variants, list):
            _bad_ir("union input requires variants")
        for variant in variants:
            try:
                _validate_input_value(value, _as_mapping(variant, "union variant"), path)
                return
            except TeacherResponseError:
                pass
        _invalid(path, "does not match any union variant")
    _bad_ir(f"unsupported input kind {kind!r}")


def _validate_output_value(value: JsonValue, spec: Mapping[str, Any], path: str) -> None:
    if spec.get("kind") == "scalar":
        _validate_head_value(value, _mapping_member(spec, "head", path), path)
        return
    if spec.get("kind") != "object" or not isinstance(value, dict):
        _invalid(path, "must match the declared output shape")
    fields = spec.get("fields")
    if not isinstance(fields, list):
        _bad_ir("object output requires fields")
    by_name = {
        _string_member(
            _as_mapping(item, "output field"), "name", "output field", allow_empty=True
        ): _as_mapping(item, "output field")
        for item in fields
    }
    if set(value) != set(by_name) or len(value) != len(by_name):
        _invalid(path, "fields must exactly match the declared output")
    for name, item in value.items():
        _validate_head_value(item, _mapping_member(by_name[name], "head", name), f"{path}.{name}")


def _validate_head_value(value: JsonValue, head: Mapping[str, Any], path: str) -> None:
    if not any(_same_json_scalar(value, item) for item in _head_support(head)):
        _invalid(path, "is outside the declared output support")


def _same_json_scalar(left: JsonValue, right: JsonValue) -> bool:
    if _finite_number(left) and _finite_number(right):
        if float(left) == 0 and float(right) == 0:
            return math.copysign(1.0, float(left)) == math.copysign(1.0, float(right))
        return float(left) == float(right)
    return type(left) is type(right) and left == right


def _same_constraint_scalar(left: JsonValue, right: JsonValue) -> bool:
    if _finite_number(left) and _finite_number(right):
        return float(left) == float(right)
    return type(left) is type(right) and left == right


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _mapping_member(value: Mapping[str, Any], name: str, context: str) -> Mapping[str, Any]:
    result = value.get(name)
    if not isinstance(result, Mapping):
        _bad_ir(f"{context} requires object member {name!r}")
    return result


def _string_member(
    value: Mapping[str, Any], name: str, context: str, *, allow_empty: bool = False
) -> str:
    result = value.get(name)
    if not isinstance(result, str) or (not allow_empty and not result):
        _bad_ir(f"{context} requires string member {name!r}")
    return result


def _as_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _bad_ir(f"{context} must be an object")
    return value


def _bad_ir(message: str) -> NoReturn:
    raise TeacherConfigurationError(message)


def _invalid(path: str, message: str) -> NoReturn:
    raise TeacherResponseError(f"generated case {path} {message}")


__all__ = [
    "MAXIMUM_CASE_COUNT",
    "build_case_schema",
    "parse_case_response",
    "validate_case",
    "validate_case_count",
    "validate_constraint_output",
    "validate_generated_cases",
]
