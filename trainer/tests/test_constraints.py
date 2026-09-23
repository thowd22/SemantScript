from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import semantscript_trainer.dataset as dataset_module
from semantscript_trainer import (
    ConstraintConfigurationError,
    ConstraintEvaluationError,
    ConstraintViolationError,
    DatasetConfigurationError,
    GeneratedCase,
    SyntheticDatasetGenerator,
    TeacherDescriptor,
    TeacherResponseError,
    evaluate_constraint_predicate,
    validate_case_constraints,
)
from semantscript_trainer.constraints import ConstraintEvaluationBudget, compile_constraints


def literal(value: Any) -> dict[str, Any]:
    return {"node": "literal", "value": value}


def input_node(name: str) -> dict[str, Any]:
    return {"node": "input", "name": name}


def binary(operator: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"node": "binary", "operator": operator, "left": left, "right": right}


def constraint(kind: str, predicate: dict[str, Any], output: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "source": "test predicate",
        "predicate": predicate,
        "output": output,
    }


def test_evaluates_boundary_and_enforces_always_and_never() -> None:
    contract = ir(
        constraints=[
            constraint("always", binary(">", input_node("score"), literal(10)), True),
            constraint("never", input_node("blocked"), True),
        ]
    )

    assert validate_case_constraints(
        contract,
        GeneratedCase(inputs={"score": 11, "blocked": False, "text": "x"}, output=True),
    ) == (0,)
    with pytest.raises(ConstraintViolationError, match="always constraint 0"):
        validate_case_constraints(
            contract,
            GeneratedCase(inputs={"score": 11, "blocked": False, "text": "x"}, output=False),
        )
    with pytest.raises(ConstraintViolationError, match="never constraint 1"):
        validate_case_constraints(
            contract,
            GeneratedCase(inputs={"score": 0, "blocked": True, "text": "x"}, output=True),
        )


def test_uses_javascript_strict_numeric_remainder_and_utf16_semantics() -> None:
    strict = constraint("always", binary("!==", input_node("blocked"), literal(1)), True)
    remainder = constraint(
        "always",
        binary(
            "===",
            binary("%", input_node("score"), literal(3)),
            literal(-1),
        ),
        True,
    )
    ordered = constraint("always", binary("<", input_node("text"), literal("\ue000")), True)

    assert evaluate_constraint_predicate(strict, {"blocked": True}) is True
    assert evaluate_constraint_predicate(remainder, {"score": -4}) is True
    assert evaluate_constraint_predicate(ordered, {"text": "😀"}) is True


def test_short_circuits_dead_nonfinite_arithmetic() -> None:
    divide_by_zero = binary("/", literal(1), literal(0))
    dead = constraint(
        "always",
        binary("&&", literal(False), binary(">", divide_by_zero, literal(0))),
        True,
    )
    live = constraint(
        "always",
        binary("||", literal(False), binary(">", divide_by_zero, literal(0))),
        True,
    )

    assert evaluate_constraint_predicate(dead, {}) is False
    with pytest.raises(ConstraintEvaluationError, match="non-finite"):
        evaluate_constraint_predicate(live, {})


def test_compile_rejects_unknown_input_even_in_dead_short_circuit_branch() -> None:
    contract = ir(
        constraints=[
            constraint(
                "always",
                binary("&&", literal(False), input_node("missing")),
                True,
            )
        ]
    )

    with pytest.raises(ConstraintConfigurationError, match="unknown input 'missing'"):
        compile_constraints(contract)


def test_compile_rejects_operand_and_predicate_type_mismatches() -> None:
    wrong_operand = ir(
        constraints=[
            constraint(
                "always",
                binary(
                    ">",
                    binary("+", input_node("text"), literal(1)),
                    literal(0),
                ),
                True,
            )
        ]
    )
    wrong_predicate = ir(constraints=[constraint("always", input_node("score"), True)])

    with pytest.raises(ConstraintConfigurationError, match="incompatible type string"):
        compile_constraints(wrong_operand)
    with pytest.raises(ConstraintConfigurationError, match=r"predicate.*incompatible type number"):
        compile_constraints(wrong_predicate)


def test_compile_rejects_constraint_labels_outside_scalar_output_support() -> None:
    outside_support = ir(constraints=[constraint("always", literal(True), "other")])
    structured_output = ir(constraints=[constraint("always", literal(True), {"decision": True})])
    structured_output["output"] = {
        "kind": "object",
        "tsType": "Result",
        "fields": [
            {
                "name": "decision",
                "tsType": "boolean",
                "head": {
                    "kind": "nominal",
                    "sourceKind": "boolean",
                    "support": [False, True],
                },
            }
        ],
    }

    with pytest.raises(ConstraintConfigurationError, match="outside the declared output support"):
        compile_constraints(outside_support)
    with pytest.raises(ConstraintConfigurationError, match="require a scalar output"):
        compile_constraints(structured_output)


def test_compile_matches_javascript_signed_zero_constraint_membership() -> None:
    contract = ir(constraints=[constraint("always", literal(True), 0.0)])
    contract["output"]["head"] = {
        "kind": "nominal",
        "sourceKind": "numeric-enum",
        "support": [-0.0, 1.0],
    }

    assert len(compile_constraints(contract)) == 1


def test_compile_accepts_json_data_property_index_and_optional_scalar_access() -> None:
    nested_score = {
        "node": "property",
        "object": {
            "node": "property",
            "object": input_node("record"),
            "property": "nested",
        },
        "property": "score",
    }
    first_value = {
        "node": "index",
        "object": input_node("values"),
        "index": literal(0),
    }
    optional = {
        "node": "property",
        "object": input_node("record"),
        "property": "optional",
    }
    contract = ir(
        constraints=[
            constraint(
                "always",
                binary(
                    "&&",
                    binary("===", nested_score, first_value),
                    binary("!==", optional, literal("blocked")),
                ),
                True,
            )
        ]
    )
    contract["inputs"] = [
        {
            "name": "record",
            "index": 0,
            "tsType": "RecordValue",
            "type": {
                "kind": "object",
                "fields": [
                    {
                        "name": "nested",
                        "optional": False,
                        "type": {
                            "kind": "object",
                            "fields": [
                                {
                                    "name": "score",
                                    "optional": False,
                                    "type": {"kind": "number"},
                                }
                            ],
                        },
                    },
                    {
                        "name": "optional",
                        "optional": True,
                        "type": {"kind": "string"},
                    },
                ],
            },
        },
        {
            "name": "values",
            "index": 1,
            "tsType": "readonly number[]",
            "type": {"kind": "array", "items": {"kind": "number"}},
        },
    ]

    assert len(compile_constraints(contract)) == 1


def test_compile_rejects_possibly_missing_sequence_operands_but_accepts_fixed_tuple() -> None:
    indexed = {
        "node": "index",
        "object": input_node("value"),
        "index": literal(0),
    }
    arithmetic = binary(
        ">",
        binary("+", indexed, literal(1)),
        literal(2),
    )
    array_contract = ir(constraints=[constraint("always", arithmetic, True)])
    array_contract["inputs"] = [
        {
            "name": "value",
            "index": 0,
            "tsType": "readonly number[]",
            "type": {"kind": "array", "items": {"kind": "number"}},
        }
    ]
    string_contract = ir(
        constraints=[
            constraint(
                "always",
                binary("<", indexed, literal("z")),
                True,
            )
        ]
    )
    string_contract["inputs"] = [
        {
            "name": "value",
            "index": 0,
            "tsType": "string",
            "type": {"kind": "string"},
        }
    ]
    tuple_contract = ir(constraints=[constraint("always", arithmetic, True)])
    tuple_contract["inputs"] = [
        {
            "name": "value",
            "index": 0,
            "tsType": "readonly [number]",
            "type": {"kind": "tuple", "items": [{"kind": "number"}]},
        }
    ]

    with pytest.raises(ConstraintConfigurationError, match="number, undefined"):
        compile_constraints(array_contract)
    with pytest.raises(ConstraintConfigurationError, match="ordered comparison"):
        compile_constraints(string_contract)
    assert len(compile_constraints(tuple_contract)) == 1


def test_compile_and_evaluate_unary_signed_zero_object_index() -> None:
    contract = ir(
        constraints=[
            constraint(
                "always",
                binary(
                    "===",
                    {
                        "node": "index",
                        "object": input_node("record"),
                        "index": {
                            "node": "unary",
                            "operator": "-",
                            "operand": literal(0),
                        },
                    },
                    literal(1),
                ),
                True,
            )
        ]
    )
    contract["inputs"] = [
        {
            "name": "record",
            "index": 0,
            "tsType": "Record<'0', number>",
            "type": {
                "kind": "object",
                "fields": [
                    {
                        "name": "0",
                        "optional": False,
                        "type": {"kind": "number"},
                    }
                ],
            },
        }
    ]

    assert validate_case_constraints(
        contract,
        GeneratedCase(inputs={"record": {"0": 1}}, output=True),
    ) == (0,)


@pytest.mark.parametrize(
    "predicate",
    [
        {"node": "unary", "operator": "!", "operand": literal(False)},
        binary("===", {"node": "unary", "operator": "+", "operand": literal(2)}, literal(2)),
        binary("===", {"node": "unary", "operator": "-", "operand": literal(2)}, literal(-2)),
        binary("!==", literal(True), literal(1)),
        binary("<", literal(1), literal(2)),
        binary("<=", literal(2), literal(2)),
        binary(">", literal(2), literal(1)),
        binary(">=", literal(2), literal(2)),
        binary("&&", literal(True), literal(True)),
        binary("||", literal(False), literal(True)),
        binary("===", binary("+", literal(2), literal(3)), literal(5)),
        binary("===", binary("-", literal(5), literal(3)), literal(2)),
        binary("===", binary("*", literal(2), literal(3)), literal(6)),
        binary("===", binary("/", literal(6), literal(3)), literal(2)),
        binary("===", binary("%", literal(7), literal(3)), literal(1)),
        binary("===", binary("**", literal(2), literal(3)), literal(8)),
    ],
)
def test_all_v1_unary_and_binary_operators(predicate: dict[str, Any]) -> None:
    assert evaluate_constraint_predicate(constraint("always", predicate, True), {}) is True


def test_property_length_index_and_missing_optional_follow_js_data_access() -> None:
    length = constraint(
        "always",
        binary(
            "===",
            {"node": "property", "object": input_node("text"), "property": "length"},
            literal(2),
        ),
        True,
    )
    missing = constraint(
        "always",
        binary(
            "===",
            {
                "node": "index",
                "object": input_node("record"),
                "index": literal("optional"),
            },
            literal(None),
        ),
        True,
    )

    assert evaluate_constraint_predicate(length, {"text": "😀"}) is True
    assert evaluate_constraint_predicate(missing, {"record": {}}) is False


def test_binary64_rounding_and_huge_exponent_match_bounded_js_number_semantics() -> None:
    rounded = constraint(
        "always",
        binary("===", literal(2**53 + 1), literal(2**53)),
        True,
    )
    huge_exponent = constraint(
        "always",
        binary(
            ">",
            binary("**", input_node("score"), literal(1_000_000_000)),
            literal(0),
        ),
        True,
    )
    huge_input = constraint(
        "always",
        binary("===", input_node("score"), input_node("score")),
        True,
    )

    assert evaluate_constraint_predicate(rounded, {}) is True
    with pytest.raises(ConstraintEvaluationError, match="non-finite"):
        evaluate_constraint_predicate(huge_exponent, {"score": 2})
    with pytest.raises(ConstraintEvaluationError, match="finite number"):
        evaluate_constraint_predicate(huge_input, {"score": 10**1000})


@pytest.mark.parametrize(
    ("value", "index", "expected"),
    [
        ([7], "0", 7),
        ([7], "length", 1),
        ("abc", "0", "a"),
        ("abc", "length", 3),
    ],
)
def test_string_and_array_bracket_properties_follow_js(
    value: Any, index: str, expected: Any
) -> None:
    predicate = constraint(
        "always",
        binary(
            "===",
            {
                "node": "index",
                "object": input_node("value"),
                "index": literal(index),
            },
            literal(expected),
        ),
        True,
    )

    assert evaluate_constraint_predicate(predicate, {"value": value}) is True


@pytest.mark.parametrize(
    ("number", "key"),
    [(1e21, "1e+21"), (1e-7, "1e-7"), (1e-6, "0.000001")],
)
def test_numeric_object_indices_use_javascript_number_keys(number: float, key: str) -> None:
    predicate = constraint(
        "always",
        binary(
            "===",
            {
                "node": "index",
                "object": input_node("record"),
                "index": literal(number),
            },
            literal("found"),
        ),
        True,
    )

    assert evaluate_constraint_predicate(predicate, {"record": {key: "found"}}) is True


def test_large_string_equality_stops_at_utf16_scan_budget_without_materializing_units() -> None:
    predicate = constraint(
        "always",
        binary("===", input_node("text"), input_node("text")),
        True,
    )

    with pytest.raises(ConstraintEvaluationError, match="UTF-16 scan units"):
        evaluate_constraint_predicate(predicate, {"text": "x" * 500_001})


def test_expression_depth_is_bounded_before_recursive_evaluation() -> None:
    expression: dict[str, Any] = literal(True)
    for _ in range(101):
        expression = {"node": "unary", "operator": "!", "operand": expression}

    with pytest.raises(ConstraintConfigurationError, match="maximum depth"):
        evaluate_constraint_predicate(constraint("always", expression, True), {})


def test_base_dataset_rejects_constraint_violating_gold_before_teacher_call(
    tmp_path: Path,
) -> None:
    contract = ir(
        constraints=[constraint("always", input_node("blocked"), False)],
        examples=[
            {
                "inputs": {"score": 0, "blocked": True, "text": "x"},
                "output": True,
            }
        ],
    )
    teacher = FakeTeacher(())

    with pytest.raises(DatasetConfigurationError, match="always constraint"):
        SyntheticDatasetGenerator(teacher, tmp_path).generate(contract, 1)
    assert teacher.calls == []


def test_base_dataset_rejects_constraint_violating_synthetic_target(tmp_path: Path) -> None:
    contract = ir(constraints=[constraint("never", input_node("blocked"), True)])
    teacher = FakeTeacher(
        (GeneratedCase(inputs={"score": 0, "blocked": True, "text": "x"}, output=True),)
    )

    with pytest.raises(TeacherResponseError, match="never constraint"):
        SyntheticDatasetGenerator(teacher, tmp_path).generate(contract, 1)


def test_dataset_rejects_aggregate_constraint_work_before_teacher_call(tmp_path: Path) -> None:
    constraints = [constraint("always", input_node("blocked"), False) for _ in range(501)]
    contract = ir(constraints=constraints)
    teacher = FakeTeacher(())

    with pytest.raises(DatasetConfigurationError, match="aggregate step budget"):
        SyntheticDatasetGenerator(teacher, tmp_path).cache_path(contract, 20_000)
    assert teacher.calls == []


def test_dataset_reuses_aggregate_utf16_budget_across_synthetic_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = ir(
        constraints=[
            constraint(
                "always",
                binary("===", input_node("text"), input_node("text")),
                True,
            )
        ]
    )
    cases = (
        GeneratedCase(inputs={"score": 0, "blocked": False, "text": "abc"}, output=True),
        GeneratedCase(inputs={"score": 1, "blocked": False, "text": "abc"}, output=True),
    )
    teacher = FakeTeacher(cases)
    monkeypatch.setattr(
        dataset_module,
        "ConstraintEvaluationBudget",
        lambda: ConstraintEvaluationBudget(remaining_string_units=10),
    )

    with pytest.raises(TeacherResponseError, match="aggregate UTF-16 scan units"):
        SyntheticDatasetGenerator(teacher, tmp_path).generate(contract, 2)

    assert len(teacher.calls) == 1


class FakeTeacher:
    def __init__(self, cases: tuple[GeneratedCase, ...]) -> None:
        self.cases = cases
        self.calls: list[tuple[dict[str, Any], int]] = []
        self.descriptor = TeacherDescriptor("fake", "constraints-v1", "a" * 64)

    def generate(self, value: dict[str, Any], n: int, /) -> tuple[GeneratedCase, ...]:
        self.calls.append((value, n))
        return self.cases


def ir(
    *,
    constraints: list[dict[str, Any]],
    examples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "semantscript.neural-function",
        "irVersion": 1,
        "stage": "lowered",
        "id": "nf_" + "4" * 64,
        "semanticSha256": "5" * 64,
        "source": {
            "path": "constraint.sem.ts",
            "line": 1,
            "column": 1,
            "sourceSha256": "6" * 64,
        },
        "definition": {
            "template": [{"kind": "text", "text": "Classify."}],
            "examples": examples or [],
            "constraints": constraints,
        },
        "inputs": [
            {"name": "score", "index": 0, "tsType": "number", "type": {"kind": "number"}},
            {
                "name": "blocked",
                "index": 1,
                "tsType": "boolean",
                "type": {"kind": "boolean"},
            },
            {"name": "text", "index": 2, "tsType": "string", "type": {"kind": "string"}},
        ],
        "output": {
            "kind": "scalar",
            "tsType": "boolean",
            "head": {
                "kind": "nominal",
                "sourceKind": "boolean",
                "support": [False, True],
            },
        },
    }
