"""Bounded evaluation and enforcement of v1 deterministic constraints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final, cast

from semantscript_trainer.case_contract import build_case_schema, validate_constraint_output
from semantscript_trainer.teacher import (
    GeneratedCase,
    JsonValue,
    NeuralFunctionIr,
    TeacherConfigurationError,
    TeacherResponseError,
)

MAXIMUM_CONSTRAINT_DEPTH = 100
MAXIMUM_CONSTRAINT_NODES = 100_000
MAXIMUM_CONSTRAINT_COUNT = 10_000
MAXIMUM_CONSTRAINT_EVALUATION_STEPS = 10_000_000
MAXIMUM_CONSTRAINT_STRING_SCAN_UNITS = 1_000_000
MAXIMUM_CONSTRAINT_AGGREGATE_STRING_SCAN_UNITS = 10_000_000
MAXIMUM_CONSTRAINT_TYPE_OPERATIONS = 10_000_000
MAXIMUM_CONSTRAINT_TYPE_VARIANTS = 10_000


class ConstraintError(RuntimeError):
    """Base class for invalid or unsatisfied deterministic constraints."""


class ConstraintConfigurationError(ConstraintError, ValueError):
    """The IR contains a malformed constraint AST."""


class ConstraintEvaluationError(ConstraintError):
    """A constraint could not be evaluated safely for one input."""


class ConstraintViolationError(ConstraintError):
    """A labeled case violates an active always/never constraint."""


class _Undefined:
    __slots__ = ()


_UNDEFINED: Final = _Undefined()
_NO_LITERAL: Final = object()


@dataclass(frozen=True, slots=True)
class _StaticType:
    kind: str
    literal: Any = _NO_LITERAL
    fields: tuple[tuple[str, tuple[_StaticType, ...], bool], ...] = ()
    item: tuple[_StaticType, ...] = ()
    elements: tuple[tuple[_StaticType, ...], ...] = ()


@dataclass(slots=True)
class _TypeBudget:
    remaining: int = MAXIMUM_CONSTRAINT_TYPE_OPERATIONS

    def consume(self, count: int = 1) -> None:
        self.remaining -= count
        if self.remaining < 0:
            raise ConstraintConfigurationError(
                "constraint relational validation exceeds maximum operation budget "
                f"{MAXIMUM_CONSTRAINT_TYPE_OPERATIONS}"
            )


@dataclass(slots=True)
class ConstraintEvaluationBudget:
    """Shared UTF-16 work allowance for one dataset/adversarial request."""

    remaining_string_units: int = MAXIMUM_CONSTRAINT_AGGREGATE_STRING_SCAN_UNITS

    def __post_init__(self) -> None:
        if (
            isinstance(self.remaining_string_units, bool)
            or not isinstance(self.remaining_string_units, int)
            or not 0
            <= self.remaining_string_units
            <= MAXIMUM_CONSTRAINT_AGGREGATE_STRING_SCAN_UNITS
        ):
            raise ConstraintConfigurationError(
                "aggregate UTF-16 scan budget must be an integer between 0 and "
                f"{MAXIMUM_CONSTRAINT_AGGREGATE_STRING_SCAN_UNITS}"
            )

    def new_case(self) -> _EvaluationBudget:
        return _EvaluationBudget(aggregate=self)


@dataclass(slots=True)
class _EvaluationBudget:
    aggregate: ConstraintEvaluationBudget
    remaining_string_units: int = MAXIMUM_CONSTRAINT_STRING_SCAN_UNITS

    def consume_string_units(self, count: int, label: str) -> None:
        self.remaining_string_units -= count
        if self.remaining_string_units < 0:
            raise ConstraintEvaluationError(
                f"{label} exceeds maximum UTF-16 scan units {MAXIMUM_CONSTRAINT_STRING_SCAN_UNITS}"
            )
        self.aggregate.remaining_string_units -= count
        if self.aggregate.remaining_string_units < 0:
            raise ConstraintEvaluationError(
                f"constraint request exceeds maximum aggregate UTF-16 scan units "
                f"{MAXIMUM_CONSTRAINT_AGGREGATE_STRING_SCAN_UNITS}"
            )


@dataclass(frozen=True, slots=True)
class CompiledConstraints:
    """One structurally validated constraint set reusable across many rows."""

    _values: tuple[Mapping[str, Any], ...]
    node_count: int

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self._values[index]

    def ensure_evaluation_budget(self, case_count: int) -> None:
        if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count < 0:
            raise ConstraintConfigurationError(
                "constraint evaluation case count must be a non-negative integer"
            )
        if self.node_count * case_count > MAXIMUM_CONSTRAINT_EVALUATION_STEPS:
            raise ConstraintConfigurationError(
                "constraint evaluation exceeds maximum aggregate step budget "
                f"{MAXIMUM_CONSTRAINT_EVALUATION_STEPS}"
            )

    def evaluate(
        self,
        index: int,
        inputs: Mapping[str, JsonValue],
        /,
        *,
        budget: ConstraintEvaluationBudget | None = None,
    ) -> bool:
        session = budget if budget is not None else ConstraintEvaluationBudget()
        return self._evaluate(index, inputs, session.new_case())

    def _evaluate(
        self,
        index: int,
        inputs: Mapping[str, JsonValue],
        budget: _EvaluationBudget,
    ) -> bool:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise ConstraintConfigurationError("constraint index is out of range")
        if not isinstance(inputs, Mapping):
            raise ConstraintEvaluationError(f"constraint {index} inputs must be an object")
        value = _evaluate(self._values[index]["predicate"], inputs, index, budget)
        if not isinstance(value, bool):
            raise ConstraintEvaluationError(
                f"constraint {index} predicate did not evaluate to a boolean"
            )
        return value

    def validate_case(
        self,
        case: GeneratedCase,
        /,
        *,
        budget: ConstraintEvaluationBudget | None = None,
    ) -> tuple[int, ...]:
        active: list[int] = []
        case_budget = (budget if budget is not None else ConstraintEvaluationBudget()).new_case()
        for index, constraint in enumerate(self._values):
            if not self._evaluate(index, case.inputs, case_budget):
                continue
            active.append(index)
            required_or_forbidden = constraint["output"]
            equal = _strict_equal(
                case.output,
                required_or_forbidden,
                case_budget,
                f"constraint {index}",
            )
            kind = constraint["kind"]
            if kind == "always" and not equal:
                raise ConstraintViolationError(
                    f"case output violates active always constraint {index}: "
                    "the required output was not emitted"
                )
            if kind == "never" and equal:
                raise ConstraintViolationError(
                    f"case output violates active never constraint {index}: "
                    "the forbidden output was emitted"
                )
        return tuple(active)

    def evaluate_output_contract(
        self,
        inputs: Mapping[str, JsonValue],
        output: JsonValue,
        /,
        *,
        budget: ConstraintEvaluationBudget | None = None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return active and violated constraint indexes under one case budget.

        Unlike :meth:`validate_case`, this diagnostic form evaluates every
        constraint instead of stopping on the first violation. Predicate work and
        output equality scans share the same per-case and aggregate budgets.
        """

        session = budget if budget is not None else ConstraintEvaluationBudget()
        case_budget = session.new_case()
        active: list[int] = []
        violations: list[int] = []
        for index, constraint in enumerate(self._values):
            if not self._evaluate(index, inputs, case_budget):
                continue
            active.append(index)
            equal = _strict_equal(
                output,
                constraint["output"],
                case_budget,
                f"constraint {index}",
            )
            if (constraint["kind"] == "always" and not equal) or (
                constraint["kind"] == "never" and equal
            ):
                violations.append(index)
        return tuple(active), tuple(violations)


def compile_constraints(ir: NeuralFunctionIr) -> CompiledConstraints:
    """Snapshot and validate a bounded v1 constraint set once for row reuse."""

    if not isinstance(ir, Mapping):
        raise ConstraintConfigurationError("IR must be an object")
    type_budget = _TypeBudget()
    try:
        build_case_schema(ir)
        input_types = _constraint_input_types(ir, type_budget)
    except TeacherConfigurationError as error:
        raise ConstraintConfigurationError(f"IR case contract is invalid: {error}") from error

    definition = ir.get("definition")
    if not isinstance(definition, Mapping):
        raise ConstraintConfigurationError("IR definition must be an object")
    raw_constraints = definition.get("constraints")
    if not isinstance(raw_constraints, list):
        raise ConstraintConfigurationError("IR definition constraints must be an array")
    if len(raw_constraints) > MAXIMUM_CONSTRAINT_COUNT:
        raise ConstraintConfigurationError(
            f"IR constraint count exceeds maximum {MAXIMUM_CONSTRAINT_COUNT}"
        )

    constraints: list[Mapping[str, Any]] = []
    total_nodes = 0
    for index, original in enumerate(raw_constraints):
        try:
            raw = deepcopy(original)
        except (TypeError, ValueError, RecursionError) as error:
            raise ConstraintConfigurationError(
                f"constraint {index} could not be snapshotted: {error}"
            ) from error
        if not isinstance(raw, Mapping) or set(raw) != {
            "kind",
            "source",
            "predicate",
            "output",
        }:
            raise ConstraintConfigurationError(
                f"constraint {index} must contain exactly kind, source, predicate, and output"
            )
        if raw.get("kind") not in ("always", "never"):
            raise ConstraintConfigurationError(
                f"constraint {index} kind must be 'always' or 'never'"
            )
        if not isinstance(raw.get("source"), str) or not raw["source"]:
            raise ConstraintConfigurationError(
                f"constraint {index} source must be a nonempty string"
            )
        predicate = raw.get("predicate")
        if not isinstance(predicate, Mapping):
            raise ConstraintConfigurationError(f"constraint {index} predicate must be an object")
        total_nodes += _validate_expression(predicate, constraint_index=index)
        if total_nodes > MAXIMUM_CONSTRAINT_NODES:
            raise ConstraintConfigurationError(
                f"IR constraints exceed maximum aggregate node count {MAXIMUM_CONSTRAINT_NODES}"
            )
        predicate_types = _infer_expression_types(
            predicate,
            input_types,
            constraint_index=index,
            budget=type_budget,
        )
        _require_type_kinds(
            predicate_types,
            {"boolean"},
            f"constraint {index} predicate",
        )
        try:
            validate_constraint_output(ir, cast(JsonValue, raw["output"]))
        except (TeacherConfigurationError, TeacherResponseError) as error:
            raise ConstraintConfigurationError(
                f"constraint {index} output is invalid: {error}"
            ) from error
        constraints.append(raw)
    return CompiledConstraints(tuple(constraints), total_nodes)


def constraints_from_ir(ir: NeuralFunctionIr) -> tuple[Mapping[str, Any], ...]:
    """Return and structurally validate the constraints in a v1 IR record."""

    return tuple(compile_constraints(ir))


def evaluate_constraint_predicate(
    constraint: Mapping[str, Any],
    inputs: Mapping[str, JsonValue],
    /,
    *,
    constraint_index: int | None = None,
) -> bool:
    """Evaluate one closed v1 predicate with JavaScript-compatible scalar semantics."""

    label = _constraint_label(constraint_index)
    predicate = constraint.get("predicate")
    if not isinstance(predicate, Mapping):
        raise ConstraintConfigurationError(f"{label} predicate must be an object")
    _validate_expression(predicate, constraint_index=constraint_index)
    if not isinstance(inputs, Mapping):
        raise ConstraintEvaluationError(f"{label} inputs must be an object")
    value = _evaluate(
        predicate,
        inputs,
        constraint_index,
        ConstraintEvaluationBudget().new_case(),
    )
    if not isinstance(value, bool):
        raise ConstraintEvaluationError(f"{label} predicate did not evaluate to a boolean")
    return value


def validate_case_constraints(
    ir: NeuralFunctionIr,
    case: GeneratedCase,
    /,
) -> tuple[int, ...]:
    """Reject any labeled case that violates an active constraint.

    The returned indices identify active predicates and are useful to later
    verifier stages without trusting a teacher's description of the case.
    """

    return compile_constraints(ir).validate_case(case)


def predicate_depends_on_inputs(constraint: Mapping[str, Any], /) -> bool:
    """Return whether a structurally valid predicate contains an input node."""

    predicate = constraint.get("predicate")
    if not isinstance(predicate, Mapping):
        raise ConstraintConfigurationError("constraint predicate must be an object")
    _validate_expression(predicate, constraint_index=None)
    pending: list[Mapping[str, Any]] = [predicate]
    while pending:
        current = pending.pop()
        node = current["node"]
        if node == "input":
            return True
        if node in ("property", "unary"):
            child = current["object"] if node == "property" else current["operand"]
            pending.append(_mapping(child, f"{node} child"))
        elif node == "index":
            pending.append(_mapping(current["object"], "index object"))
            pending.append(_mapping(current["index"], "index expression"))
        elif node == "binary":
            pending.append(_mapping(current["left"], "binary left"))
            pending.append(_mapping(current["right"], "binary right"))
    return False


def json_values_equal(left: JsonValue, right: JsonValue, /) -> bool:
    """Compare JSON scalars using JavaScript strict-equality number rules."""

    return _strict_equal(
        left,
        right,
        ConstraintEvaluationBudget().new_case(),
        "JSON value comparison",
    )


def _validate_expression(
    expression: Mapping[str, Any],
    *,
    constraint_index: int | None,
) -> int:
    pending: list[tuple[Mapping[str, Any], int]] = [(expression, 1)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > MAXIMUM_CONSTRAINT_NODES:
            raise ConstraintConfigurationError(
                f"{_constraint_label(constraint_index)} predicate exceeds maximum node count "
                f"{MAXIMUM_CONSTRAINT_NODES}"
            )
        if depth > MAXIMUM_CONSTRAINT_DEPTH:
            raise ConstraintConfigurationError(
                f"{_constraint_label(constraint_index)} predicate exceeds maximum depth "
                f"{MAXIMUM_CONSTRAINT_DEPTH}"
            )
        node = current.get("node")
        if node == "literal":
            _exact_keys(current, {"node", "value"}, constraint_index)
            value = current["value"]
            finite_number = True
            if _is_number(value):
                try:
                    finite_number = math.isfinite(float(value))
                except (OverflowError, ValueError):
                    finite_number = False
            valid_unicode = True
            if isinstance(value, str):
                try:
                    value.encode("utf-8", errors="strict")
                except UnicodeEncodeError:
                    valid_unicode = False
            if not _is_json_scalar(value) or not finite_number or not valid_unicode:
                raise ConstraintConfigurationError(
                    f"{_constraint_label(constraint_index)} literal must be a finite JSON scalar"
                )
            continue
        if node == "input":
            _exact_keys(current, {"node", "name"}, constraint_index)
            if not isinstance(current["name"], str) or not current["name"]:
                raise ConstraintConfigurationError(
                    f"{_constraint_label(constraint_index)} input name must be nonempty"
                )
            continue
        if node == "property":
            _exact_keys(current, {"node", "object", "property"}, constraint_index)
            if not isinstance(current["property"], str):
                raise ConstraintConfigurationError(
                    f"{_constraint_label(constraint_index)} property name must be a string"
                )
            pending.append((_mapping(current["object"], "property object"), depth + 1))
            continue
        if node == "index":
            _exact_keys(current, {"node", "object", "index"}, constraint_index)
            pending.append((_mapping(current["object"], "index object"), depth + 1))
            pending.append((_mapping(current["index"], "index expression"), depth + 1))
            continue
        if node == "unary":
            _exact_keys(current, {"node", "operator", "operand"}, constraint_index)
            if current["operator"] not in ("!", "+", "-"):
                raise ConstraintConfigurationError(
                    f"{_constraint_label(constraint_index)} has an unsupported unary operator"
                )
            pending.append((_mapping(current["operand"], "unary operand"), depth + 1))
            continue
        if node == "binary":
            _exact_keys(current, {"node", "operator", "left", "right"}, constraint_index)
            if current["operator"] not in (
                "===",
                "!==",
                "<",
                "<=",
                ">",
                ">=",
                "&&",
                "||",
                "+",
                "-",
                "*",
                "/",
                "%",
                "**",
            ):
                raise ConstraintConfigurationError(
                    f"{_constraint_label(constraint_index)} has an unsupported binary operator"
                )
            pending.append((_mapping(current["left"], "binary left"), depth + 1))
            pending.append((_mapping(current["right"], "binary right"), depth + 1))
            continue
        raise ConstraintConfigurationError(
            f"{_constraint_label(constraint_index)} has unsupported expression node {node!r}"
        )
    return nodes


def _constraint_input_types(
    ir: Mapping[str, Any],
    budget: _TypeBudget,
) -> dict[str, tuple[_StaticType, ...]]:
    raw_inputs = ir.get("inputs")
    if not isinstance(raw_inputs, list):
        raise ConstraintConfigurationError("IR inputs must be an array")
    result: dict[str, tuple[_StaticType, ...]] = {}
    for raw in raw_inputs:
        if not isinstance(raw, Mapping):
            raise ConstraintConfigurationError("IR input entries must be objects")
        name = raw.get("name")
        spec = raw.get("type")
        if not isinstance(name, str) or not name or not isinstance(spec, Mapping):
            raise ConstraintConfigurationError("IR input entries require name and type")
        result[name] = _types_from_input_spec(spec, f"input {name!r}", budget)
    return result


def _types_from_input_spec(
    spec: Mapping[str, Any],
    label: str,
    budget: _TypeBudget,
) -> tuple[_StaticType, ...]:
    budget.consume()
    kind = spec.get("kind")
    if kind in ("string", "boolean", "number", "null"):
        return (_StaticType(cast(str, kind)),)
    if kind == "literal":
        return (_literal_static_type(spec.get("value"), label),)
    if kind == "enum":
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            raise ConstraintConfigurationError(f"{label} enum must contain values")
        return _merge_types(
            (_literal_static_type(value, label) for value in values),
            label,
            budget,
        )
    if kind == "array":
        item = spec.get("items")
        if not isinstance(item, Mapping):
            raise ConstraintConfigurationError(f"{label} array item type is invalid")
        return (_StaticType("array", item=_types_from_input_spec(item, label, budget)),)
    if kind == "tuple":
        items = spec.get("items")
        if not isinstance(items, list):
            raise ConstraintConfigurationError(f"{label} tuple items are invalid")
        elements: list[tuple[_StaticType, ...]] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise ConstraintConfigurationError(f"{label} tuple item type is invalid")
            elements.append(_types_from_input_spec(item, label, budget))
        return (_StaticType("tuple", elements=tuple(elements)),)
    if kind == "object":
        raw_fields = spec.get("fields")
        if not isinstance(raw_fields, list):
            raise ConstraintConfigurationError(f"{label} object fields are invalid")
        fields: list[tuple[str, tuple[_StaticType, ...], bool]] = []
        for raw_field in raw_fields:
            if not isinstance(raw_field, Mapping):
                raise ConstraintConfigurationError(f"{label} object field is invalid")
            field_name = raw_field.get("name")
            field_type = raw_field.get("type")
            optional = raw_field.get("optional")
            if (
                not isinstance(field_name, str)
                or not isinstance(field_type, Mapping)
                or not isinstance(optional, bool)
            ):
                raise ConstraintConfigurationError(f"{label} object field is invalid")
            fields.append(
                (
                    field_name,
                    _types_from_input_spec(field_type, f"{label}.{field_name}", budget),
                    optional,
                )
            )
        return (_StaticType("object", fields=tuple(fields)),)
    if kind == "union":
        variants = spec.get("variants")
        if not isinstance(variants, list) or len(variants) < 2:
            raise ConstraintConfigurationError(f"{label} union variants are invalid")
        resolved: list[_StaticType] = []
        for variant in variants:
            if not isinstance(variant, Mapping):
                raise ConstraintConfigurationError(f"{label} union variant is invalid")
            resolved.extend(_types_from_input_spec(variant, label, budget))
        return _merge_types(resolved, label, budget)
    raise ConstraintConfigurationError(f"{label} has unsupported kind {kind!r}")


def _literal_static_type(value: Any, label: str) -> _StaticType:
    if value is None:
        return _StaticType("null", literal=None)
    if isinstance(value, bool):
        return _StaticType("boolean", literal=value)
    if isinstance(value, str):
        return _StaticType("string", literal=value)
    if _is_number(value):
        try:
            number = float(value)
        except (OverflowError, ValueError) as error:
            raise ConstraintConfigurationError(f"{label} literal is not finite") from error
        if not math.isfinite(number):
            raise ConstraintConfigurationError(f"{label} literal is not finite")
        return _StaticType("number", literal=number)
    raise ConstraintConfigurationError(f"{label} literal must be a JSON scalar")


def _merge_types(
    values,
    label: str,
    budget: _TypeBudget,
) -> tuple[_StaticType, ...]:
    result: list[_StaticType] = []
    seen: set[_StaticType] = set()
    for value in values:
        budget.consume()
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) > MAXIMUM_CONSTRAINT_TYPE_VARIANTS:
            raise ConstraintConfigurationError(
                f"{label} exceeds maximum type variants {MAXIMUM_CONSTRAINT_TYPE_VARIANTS}"
            )
    if not result:
        raise ConstraintConfigurationError(f"{label} has no supported value type")
    return tuple(result)


def _infer_expression_types(
    expression: Mapping[str, Any],
    inputs: Mapping[str, tuple[_StaticType, ...]],
    *,
    constraint_index: int,
    budget: _TypeBudget,
) -> tuple[_StaticType, ...]:
    budget.consume()
    label = f"constraint {constraint_index}"
    node = expression["node"]
    if node == "literal":
        return (_literal_static_type(expression["value"], label),)
    if node == "input":
        name = expression["name"]
        if name not in inputs:
            raise ConstraintConfigurationError(f"{label} references unknown input {name!r}")
        return inputs[name]
    if node == "property":
        object_types = _infer_expression_types(
            _mapping(expression["object"], "property object"),
            inputs,
            constraint_index=constraint_index,
            budget=budget,
        )
        return _property_result_types(
            object_types,
            expression["property"],
            label,
            budget,
        )
    if node == "index":
        object_types = _infer_expression_types(
            _mapping(expression["object"], "index object"),
            inputs,
            constraint_index=constraint_index,
            budget=budget,
        )
        index_types = _infer_expression_types(
            _mapping(expression["index"], "index expression"),
            inputs,
            constraint_index=constraint_index,
            budget=budget,
        )
        return _index_result_types(object_types, index_types, label, budget)
    if node == "unary":
        operand_types = _infer_expression_types(
            _mapping(expression["operand"], "unary operand"),
            inputs,
            constraint_index=constraint_index,
            budget=budget,
        )
        operator = expression["operator"]
        expected = "boolean" if operator == "!" else "number"
        _require_type_kinds(operand_types, {expected}, f"{label} unary {operator} operand")
        if operator in ("+", "-") and all(
            operand.literal is not _NO_LITERAL for operand in operand_types
        ):
            return _merge_types(
                (
                    _StaticType(
                        "number",
                        literal=(
                            float(operand.literal) if operator == "+" else -float(operand.literal)
                        ),
                    )
                    for operand in operand_types
                ),
                f"{label} unary {operator} result",
                budget,
            )
        return (_StaticType(expected),)

    left_types = _infer_expression_types(
        _mapping(expression["left"], "binary left"),
        inputs,
        constraint_index=constraint_index,
        budget=budget,
    )
    right_types = _infer_expression_types(
        _mapping(expression["right"], "binary right"),
        inputs,
        constraint_index=constraint_index,
        budget=budget,
    )
    operator = expression["operator"]
    if operator in ("+", "-", "*", "/", "%", "**"):
        _require_type_kinds(left_types, {"number"}, f"{label} {operator} left operand")
        _require_type_kinds(right_types, {"number"}, f"{label} {operator} right operand")
        return (_StaticType("number"),)
    if operator in ("&&", "||"):
        _require_type_kinds(left_types, {"boolean"}, f"{label} {operator} left operand")
        _require_type_kinds(right_types, {"boolean"}, f"{label} {operator} right operand")
        return (_StaticType("boolean"),)
    if operator in ("<", "<=", ">", ">="):
        left_kinds = _type_kinds(left_types)
        right_kinds = _type_kinds(right_types)
        if not (left_kinds == right_kinds == {"number"} or left_kinds == right_kinds == {"string"}):
            raise ConstraintConfigurationError(
                f"{label} ordered comparison requires two numbers or two strings"
            )
        return (_StaticType("boolean"),)
    scalar_kinds = {"number", "string", "boolean", "null", "undefined"}
    _require_type_kinds(left_types, scalar_kinds, f"{label} equality left operand")
    _require_type_kinds(right_types, scalar_kinds, f"{label} equality right operand")
    return (_StaticType("boolean"),)


def _property_result_types(
    object_types: tuple[_StaticType, ...],
    property_name: str,
    label: str,
    budget: _TypeBudget,
) -> tuple[_StaticType, ...]:
    result: list[_StaticType] = []
    for object_type in object_types:
        budget.consume()
        if object_type.kind in ("string", "array", "tuple") and property_name == "length":
            result.append(_StaticType("number"))
            continue
        if object_type.kind != "object":
            raise ConstraintConfigurationError(
                f"{label} cannot read JSON-data property {property_name!r} from {object_type.kind}"
            )
        field = next(
            (entry for entry in object_type.fields if entry[0] == property_name),
            None,
        )
        if field is None:
            raise ConstraintConfigurationError(
                f"{label} references undeclared property {property_name!r}"
            )
        result.extend(field[1])
        if field[2]:
            result.append(_StaticType("undefined"))
    return _merge_types(result, f"{label} property {property_name!r}", budget)


def _index_result_types(
    object_types: tuple[_StaticType, ...],
    index_types: tuple[_StaticType, ...],
    label: str,
    budget: _TypeBudget,
) -> tuple[_StaticType, ...]:
    _require_type_kinds(index_types, {"number", "string"}, f"{label} index")
    result: list[_StaticType] = []
    for object_type in object_types:
        for index_type in index_types:
            budget.consume()
            literal = index_type.literal
            if object_type.kind == "object":
                if literal is _NO_LITERAL:
                    raise ConstraintConfigurationError(
                        f"{label} object index must have a finite literal or enum key"
                    )
                key = _static_property_key(literal, label)
                field = next(
                    (entry for entry in object_type.fields if entry[0] == key),
                    None,
                )
                if field is None:
                    raise ConstraintConfigurationError(
                        f"{label} references undeclared property {key!r}"
                    )
                result.extend(field[1])
                if field[2]:
                    result.append(_StaticType("undefined"))
                continue
            if object_type.kind == "array":
                if literal == "length":
                    result.append(_StaticType("number"))
                elif index_type.kind == "number" or (
                    literal is not _NO_LITERAL and _array_index(literal) is not None
                ):
                    result.extend(object_type.item)
                    result.append(_StaticType("undefined"))
                else:
                    raise ConstraintConfigurationError(
                        f"{label} array index must be numeric or 'length'"
                    )
                continue
            if object_type.kind == "tuple":
                if literal == "length":
                    result.append(_StaticType("number"))
                    continue
                if literal is _NO_LITERAL:
                    if index_type.kind != "number":
                        raise ConstraintConfigurationError(
                            f"{label} tuple index must be numeric or 'length'"
                        )
                    for element in object_type.elements:
                        result.extend(element)
                    result.append(_StaticType("undefined"))
                    continue
                position = _array_index(literal)
                if position is None:
                    raise ConstraintConfigurationError(
                        f"{label} tuple index must be numeric or 'length'"
                    )
                if position < len(object_type.elements):
                    result.extend(object_type.elements[position])
                else:
                    result.append(_StaticType("undefined"))
                continue
            if object_type.kind == "string":
                if literal == "length":
                    result.append(_StaticType("number"))
                elif index_type.kind == "number" or (
                    literal is not _NO_LITERAL and _array_index(literal) is not None
                ):
                    result.extend((_StaticType("string"), _StaticType("undefined")))
                else:
                    raise ConstraintConfigurationError(
                        f"{label} string index must be numeric or 'length'"
                    )
                continue
            raise ConstraintConfigurationError(
                f"{label} cannot index JSON-data value of type {object_type.kind}"
            )
    return _merge_types(result, f"{label} index result", budget)


def _static_property_key(value: Any, label: str) -> str:
    if isinstance(value, str):
        return value
    if _is_number(value):
        try:
            return _js_number_string(float(value))
        except (OverflowError, ValueError) as error:
            raise ConstraintConfigurationError(f"{label} object index is not finite") from error
    raise ConstraintConfigurationError(f"{label} object index must be string or number")


def _type_kinds(values: tuple[_StaticType, ...]) -> set[str]:
    return {value.kind for value in values}


def _require_type_kinds(
    values: tuple[_StaticType, ...],
    expected: set[str],
    label: str,
) -> None:
    actual = _type_kinds(values)
    if not actual or not actual <= expected:
        rendered = ", ".join(sorted(actual)) or "none"
        raise ConstraintConfigurationError(
            f"{label} has incompatible type {rendered}; expected {' or '.join(sorted(expected))}"
        )


def _evaluate(
    expression: Mapping[str, Any],
    inputs: Mapping[str, JsonValue],
    constraint_index: int | None,
    budget: _EvaluationBudget,
) -> Any:
    node = expression["node"]
    label = _constraint_label(constraint_index)
    if node == "literal":
        return expression["value"]
    if node == "input":
        name = expression["name"]
        if name not in inputs:
            raise ConstraintEvaluationError(f"{label} references missing input {name!r}")
        return inputs[name]
    if node == "property":
        value = _evaluate(
            _mapping(expression["object"], "property object"),
            inputs,
            constraint_index,
            budget,
        )
        property_name = expression["property"]
        if isinstance(value, Mapping):
            return value.get(property_name, _UNDEFINED)
        if isinstance(value, (list, str)) and property_name == "length":
            return len(value) if isinstance(value, list) else _utf16_length(value, budget, label)
        raise ConstraintEvaluationError(
            f"{label} cannot read property {property_name!r} from this value"
        )
    if node == "index":
        value = _evaluate(
            _mapping(expression["object"], "index object"),
            inputs,
            constraint_index,
            budget,
        )
        index = _evaluate(
            _mapping(expression["index"], "index expression"),
            inputs,
            constraint_index,
            budget,
        )
        return _read_index(value, index, label, budget)
    if node == "unary":
        operand = _evaluate(
            _mapping(expression["operand"], "unary operand"),
            inputs,
            constraint_index,
            budget,
        )
        operator = expression["operator"]
        if operator == "!":
            if not isinstance(operand, bool):
                raise ConstraintEvaluationError(f"{label} unary ! operand is not boolean")
            return not operand
        number = _number(operand, f"{label} unary {operator} operand")
        return _finite_result(number if operator == "+" else -number, label)

    operator = expression["operator"]
    left = _evaluate(
        _mapping(expression["left"], "binary left"),
        inputs,
        constraint_index,
        budget,
    )
    if operator == "&&":
        if not isinstance(left, bool):
            raise ConstraintEvaluationError(f"{label} && left operand is not boolean")
        if not left:
            return False
        right = _evaluate(
            _mapping(expression["right"], "binary right"),
            inputs,
            constraint_index,
            budget,
        )
        if not isinstance(right, bool):
            raise ConstraintEvaluationError(f"{label} && right operand is not boolean")
        return right
    if operator == "||":
        if not isinstance(left, bool):
            raise ConstraintEvaluationError(f"{label} || left operand is not boolean")
        if left:
            return True
        right = _evaluate(
            _mapping(expression["right"], "binary right"),
            inputs,
            constraint_index,
            budget,
        )
        if not isinstance(right, bool):
            raise ConstraintEvaluationError(f"{label} || right operand is not boolean")
        return right

    right = _evaluate(
        _mapping(expression["right"], "binary right"),
        inputs,
        constraint_index,
        budget,
    )
    if operator == "===":
        return _strict_equal(left, right, budget, label)
    if operator == "!==":
        return not _strict_equal(left, right, budget, label)
    if operator in ("<", "<=", ">", ">="):
        return _ordered_compare(left, right, operator, label, budget)

    left_number = _number(left, f"{label} {operator} left operand")
    right_number = _number(right, f"{label} {operator} right operand")
    try:
        if operator == "+":
            result = left_number + right_number
        elif operator == "-":
            result = left_number - right_number
        elif operator == "*":
            result = left_number * right_number
        elif operator == "/":
            if right_number == 0:
                raise ConstraintEvaluationError(f"{label} produced a non-finite intermediate")
            result = left_number / right_number
        elif operator == "%":
            if right_number == 0:
                raise ConstraintEvaluationError(f"{label} produced a non-finite intermediate")
            result = math.fmod(left_number, right_number)
        else:
            result = left_number**right_number
    except ConstraintEvaluationError:
        raise
    except (ArithmeticError, OverflowError, ValueError) as error:
        raise ConstraintEvaluationError(f"{label} produced a non-finite intermediate") from error
    if isinstance(result, complex):
        raise ConstraintEvaluationError(f"{label} produced a non-finite intermediate")
    return _finite_result(result, label)


def _read_index(
    value: Any,
    index: Any,
    label: str,
    budget: _EvaluationBudget,
) -> Any:
    if isinstance(value, Mapping):
        key = _property_key(index, label)
        return value.get(key, _UNDEFINED)
    if isinstance(value, list):
        if index == "length":
            return len(value)
        position = _array_index(index)
        return value[position] if position is not None and position < len(value) else _UNDEFINED
    if isinstance(value, str):
        if index == "length":
            return _utf16_length(value, budget, label)
        position = _array_index(index)
        return _UNDEFINED if position is None else _utf16_unit_at(value, position, budget, label)
    raise ConstraintEvaluationError(f"{label} cannot index this value")


def _property_key(value: Any, label: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or value is None or value is _UNDEFINED:
        raise ConstraintEvaluationError(f"{label} object index is not a string or number")
    if isinstance(value, (int, float)):
        number = _number(value, f"{label} object index")
        return _js_number_string(number)
    raise ConstraintEvaluationError(f"{label} object index is not a string or number")


def _array_index(value: Any) -> int | None:
    if isinstance(value, str):
        if value == "0":
            return 0
        if not value or value[0] == "0" or not value.isascii() or not value.isdigit():
            return None
        integer = int(value)
        return integer if integer < 2**32 - 1 else None
    if not _is_number(value):
        return None
    try:
        number = _number(value, "array index")
    except ConstraintEvaluationError:
        return None
    if number < 0 or not number.is_integer():
        return None
    integer = int(number)
    return integer if integer < 2**32 - 1 else None


def _ordered_compare(
    left: Any,
    right: Any,
    operator: str,
    label: str,
    budget: _EvaluationBudget,
) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        comparison = _compare_utf16(left, right, budget, label)
    elif _is_number(left) and _is_number(right):
        comparable_left = _number(left, f"{label} left comparison operand")
        comparable_right = _number(right, f"{label} right comparison operand")
        comparison = (comparable_left > comparable_right) - (comparable_left < comparable_right)
    else:
        raise ConstraintEvaluationError(
            f"{label} ordered comparison operands must both be numbers or both be strings"
        )
    if operator == "<":
        return comparison < 0
    if operator == "<=":
        return comparison <= 0
    if operator == ">":
        return comparison > 0
    return comparison >= 0


def _strict_equal(
    left: Any,
    right: Any,
    budget: _EvaluationBudget,
    label: str,
) -> bool:
    if left is _UNDEFINED or right is _UNDEFINED:
        return left is right
    if _is_number(left) and _is_number(right):
        return _number(left, f"{label} equality operand") == _number(
            right, f"{label} equality operand"
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, (dict, list)):
        return left is right
    if isinstance(left, str):
        return _compare_utf16(left, right, budget, label) == 0
    return bool(left == right)


def _number(value: Any, context: str) -> float:
    if not _is_number(value):
        raise ConstraintEvaluationError(f"{context} is not a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ConstraintEvaluationError(f"{context} is not a finite number") from error
    if not math.isfinite(number):
        raise ConstraintEvaluationError(f"{context} is not a finite number")
    return number


def _finite_result(value: int | float, label: str) -> int | float:
    try:
        finite = math.isfinite(float(value))
    except OverflowError:
        finite = False
    if not finite:
        raise ConstraintEvaluationError(f"{label} produced a non-finite intermediate")
    return float(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool, int, float))


def _utf16_length(value: str, budget: _EvaluationBudget, label: str) -> int:
    budget.consume_string_units(len(value), label)
    astral = sum(ord(character) > 0xFFFF for character in value)
    budget.consume_string_units(astral, label)
    return len(value) + astral


def _utf16_unit_at(
    value: str,
    position: int,
    budget: _EvaluationBudget,
    label: str,
) -> str | _Undefined:
    for index, unit in enumerate(_iter_utf16_units(value, budget, label)):
        if index == position:
            return chr(unit)
    return _UNDEFINED


def _compare_utf16(
    left: str,
    right: str,
    budget: _EvaluationBudget,
    label: str,
) -> int:
    left_units = iter(_iter_utf16_units(left, budget, label))
    right_units = iter(_iter_utf16_units(right, budget, label))
    sentinel = object()
    while True:
        left_unit = next(left_units, sentinel)
        right_unit = next(right_units, sentinel)
        if left_unit is sentinel:
            return 0 if right_unit is sentinel else -1
        if right_unit is sentinel:
            return 1
        if left_unit != right_unit:
            return -1 if cast(int, left_unit) < cast(int, right_unit) else 1


def _iter_utf16_units(
    value: str,
    budget: _EvaluationBudget,
    label: str,
):
    for character in value:
        codepoint = ord(character)
        if codepoint <= 0xFFFF:
            budget.consume_string_units(1, label)
            yield codepoint
            continue
        codepoint -= 0x10000
        budget.consume_string_units(2, label)
        yield 0xD800 + (codepoint >> 10)
        yield 0xDC00 + (codepoint & 0x3FF)


def _js_number_string(value: float) -> str:
    if value == 0:
        return "0"
    absolute = abs(value)
    token = repr(value).lower()
    if 1e-6 <= absolute < 1e21:
        fixed = format(Decimal(token), "f")
        return fixed.rstrip("0").rstrip(".") if "." in fixed else fixed
    if "e" not in token:
        token = f"{token}e+0"
    mantissa, exponent = token.split("e", 1)
    sign = ""
    if exponent.startswith(("+", "-")):
        sign, exponent = exponent[0], exponent[1:]
    exponent = exponent.lstrip("0") or "0"
    if sign != "-":
        sign = "+"
    return f"{mantissa}e{sign}{exponent}"


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConstraintConfigurationError(f"constraint {context} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    constraint_index: int | None,
) -> None:
    if set(value) != expected:
        raise ConstraintConfigurationError(
            f"{_constraint_label(constraint_index)} expression has unknown or missing fields"
        )


def _constraint_label(index: int | None) -> str:
    return "constraint" if index is None else f"constraint {index}"


__all__ = [
    "MAXIMUM_CONSTRAINT_AGGREGATE_STRING_SCAN_UNITS",
    "MAXIMUM_CONSTRAINT_COUNT",
    "MAXIMUM_CONSTRAINT_DEPTH",
    "MAXIMUM_CONSTRAINT_EVALUATION_STEPS",
    "MAXIMUM_CONSTRAINT_NODES",
    "MAXIMUM_CONSTRAINT_STRING_SCAN_UNITS",
    "CompiledConstraints",
    "ConstraintConfigurationError",
    "ConstraintError",
    "ConstraintEvaluationBudget",
    "ConstraintEvaluationError",
    "ConstraintViolationError",
    "compile_constraints",
    "constraints_from_ir",
    "evaluate_constraint_predicate",
    "json_values_equal",
    "predicate_depends_on_inputs",
    "validate_case_constraints",
]
