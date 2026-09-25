"""The built-in constraints teacher: sampling, labels, pairs, modes and configuration.

Everything here runs on the CPU with no model: the teacher only samples inputs and
evaluates constraints, and the dataset and adversarial generators it feeds are the
production validators.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from semantscript_trainer import (
    AdversarialDatasetGenerator,
    AdversarialTeacher,
    ConstraintsTeacherConfig,
    GeneratedCase,
    NumberRange,
    SyntheticDatasetGenerator,
    Teacher,
    TeacherConfig,
    TeacherConfigurationError,
    TeacherDescriptor,
    create_teacher,
    load_teacher_config,
    validate_case_constraints,
)
from semantscript_trainer.adversarial import AdversarialGenerationError
from semantscript_trainer.case_contract import validate_case
from semantscript_trainer.teacher_config import teacher_config_from_mapping
from semantscript_trainer.teachers import ConstraintsTeacher
from semantscript_trainer.teachers.constraints import ConstraintSampler

# --- IR helpers ---------------------------------------------------------------


def path(*names: str) -> dict[str, Any]:
    node: dict[str, Any] = {"node": "input", "name": names[0]}
    for name in names[1:]:
        node = {"node": "property", "object": node, "property": name}
    return node


def lit(value: Any) -> dict[str, Any]:
    return {"node": "literal", "value": value}


def op(operator: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"node": "binary", "operator": operator, "left": left, "right": right}


def rule(kind: str, predicate: dict[str, Any], output: Any, source: str = "rule") -> dict[str, Any]:
    return {"kind": kind, "source": source, "predicate": predicate, "output": output}


CUSTOMER = {
    "kind": "object",
    "fields": [
        {"name": "priorRefunds", "optional": False, "type": {"kind": "number"}},
        {
            "name": "tier",
            "optional": False,
            "type": {
                "kind": "union",
                "variants": [
                    {"kind": "literal", "value": "enterprise"},
                    {"kind": "literal", "value": "standard"},
                ],
            },
        },
    ],
}
ORDER = {
    "kind": "object",
    "fields": [
        {"name": "ageDays", "optional": False, "type": {"kind": "number"}},
        {
            "name": "status",
            "optional": False,
            "type": {
                "kind": "union",
                "variants": [
                    {"kind": "literal", "value": "fraudulent"},
                    {"kind": "literal", "value": "paid"},
                ],
            },
        },
        {"name": "total", "optional": False, "type": {"kind": "number"}},
    ],
}

AGE = path("order", "ageDays")
STATUS = path("order", "status")
TIER = path("customer", "tier")
PRIOR = path("customer", "priorRefunds")


def within_window() -> dict[str, Any]:
    return op(
        "||",
        op("&&", op("===", TIER, lit("enterprise")), op("<=", AGE, lit(60))),
        op("&&", op("===", TIER, lit("standard")), op("<=", AGE, lit(30))),
    )


def complete_constraints() -> list[dict[str, Any]]:
    """A refund policy that decides every input: deny, review or approve."""

    return [
        rule("always", op(">", AGE, lit(90)), "deny", "order.ageDays > 90"),
        rule(
            "always",
            op("&&", op("<=", AGE, lit(90)), op("===", STATUS, lit("fraudulent"))),
            "review",
            "fraudulent within 90 days",
        ),
        rule(
            "always",
            op(
                "&&",
                op("&&", op("<=", AGE, lit(90)), op("===", STATUS, lit("paid"))),
                {"node": "unary", "operator": "!", "operand": within_window()},
            ),
            "review",
            "paid outside the tier window",
        ),
        rule(
            "always",
            op(
                "&&",
                op("&&", op("===", STATUS, lit("paid")), within_window()),
                op("<=", PRIOR, lit(2)),
            ),
            "approve",
            "paid within the tier window, at most two prior refunds",
        ),
        rule(
            "always",
            op(
                "&&",
                op("&&", op("===", STATUS, lit("paid")), within_window()),
                op(">", PRIOR, lit(2)),
            ),
            "review",
            "paid within the tier window, more than two prior refunds",
        ),
        rule("never", op(">", PRIOR, lit(2)), "approve", "customer.priorRefunds > 2"),
    ]


def refund_ir(
    constraints: list[dict[str, Any]] | None = None,
    *,
    examples: list[dict[str, Any]] | None = None,
    function_id: str = "nf_" + "a" * 64,
) -> dict[str, Any]:
    return {
        "kind": "semantscript.neural-function",
        "irVersion": 1,
        "stage": "source",
        "id": function_id,
        "semanticSha256": "b" * 64,
        "source": {"path": "src/refund.sem.ts", "line": 16, "column": 10, "sourceSha256": "c" * 64},
        "definition": {
            "template": [{"kind": "text", "text": "Decide the refund."}],
            "examples": examples
            if examples is not None
            else [
                {
                    "inputs": {
                        "customer": {"priorRefunds": 0, "tier": "enterprise"},
                        "order": {"ageDays": 45, "status": "paid", "total": 129.5},
                    },
                    "output": "approve",
                }
            ],
            "constraints": complete_constraints() if constraints is None else constraints,
        },
        "inputs": [
            {"name": "customer", "index": 0, "tsType": "Customer", "type": CUSTOMER},
            {"name": "order", "index": 1, "tsType": "Order", "type": ORDER},
        ],
        "output": {
            "kind": "scalar",
            "tsType": "RefundDecision",
            "head": {
                "kind": "nominal",
                "sourceKind": "string-union",
                "support": ["approve", "deny", "review"],
            },
        },
        "model": {"encoder": "encoder.test", "adapter": "adapter.test.refund"},
    }


def incomplete_ir() -> dict[str, Any]:
    """examples/refund.sem.ts: only the 90-day deny and the fraudulent never-approve."""

    return refund_ir(
        [
            rule("never", op("===", STATUS, lit("fraudulent")), "approve"),
            rule("always", op(">", AGE, lit(90)), "deny"),
        ]
    )


def expected_label(inputs: dict[str, Any]) -> str:
    order, customer = inputs["order"], inputs["customer"]
    if order["ageDays"] > 90:
        return "deny"
    if order["status"] == "fraudulent":
        return "review"
    window = 60 if customer["tier"] == "enterprise" else 30
    if order["ageDays"] > window or customer["priorRefunds"] > 2:
        return "review"
    return "approve"


def changed_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        found: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                found.append(f"{prefix}/{key}")
            else:
                found.extend(changed_paths(left[key], right[key], f"{prefix}/{key}"))
        return found
    return [] if left == right and type(left) is type(right) else [prefix]


# --- a fake language-model fallback ----------------------------------------------


class FakeFallback:
    """Labels every input with a fixed rule and counts its requests."""

    def __init__(self) -> None:
        self.descriptor = TeacherDescriptor("fake-llm", "fake-model", "f" * 64)
        self.generated: list[int] = []
        self.boundaries: list[int] = []
        self.counterfactuals: list[GeneratedCase] = []

    def _grid(self) -> list[dict[str, Any]]:
        return [
            {
                "customer": {"priorRefunds": prior, "tier": tier},
                "order": {"ageDays": age, "status": status, "total": 50},
            }
            for age in (10, 40, 70, 91, 120)
            for status in ("paid", "fraudulent")
            for tier in ("enterprise", "standard")
            for prior in (0, 3)
        ]

    def generate(self, ir: dict[str, Any], n: int, /) -> tuple[GeneratedCase, ...]:
        self.generated.append(n)
        # A deliberately wrong label on decided inputs ("review" for ageDays > 90)
        # shows that the constraints overrule the fallback where they decide.
        grid = self._grid()
        return tuple(
            GeneratedCase(grid[index % len(grid)], "review")
            if grid[index % len(grid)]["order"]["ageDays"] > 90
            else GeneratedCase(grid[index % len(grid)], expected_label(grid[index % len(grid)]))
            for index in range(n)
        )

    def generate_boundary_pair(self, ir: dict[str, Any], index: int, /) -> Any:
        from semantscript_trainer import BoundaryPairProposal

        self.boundaries.append(index)
        base = {
            "customer": {"priorRefunds": 0, "tier": "standard"},
            "order": {"ageDays": 90, "status": "paid", "total": 50},
        }
        edited = deepcopy(base)
        edited["order"]["ageDays"] = 91
        return BoundaryPairProposal(
            predicate_false=GeneratedCase(base, "review"),
            predicate_true=GeneratedCase(edited, "review"),
        )

    def generate_counterfactual(self, ir: dict[str, Any], anchor: GeneratedCase, /) -> Any:
        from semantscript_trainer import CounterfactualProposal

        self.counterfactuals.append(anchor)
        twin = deepcopy(anchor.inputs)
        twin["order"]["ageDays"] = 10 if anchor.inputs["order"]["ageDays"] > 90 else 120
        return CounterfactualProposal(
            twin=GeneratedCase(twin, expected_label(twin)), reason="crossing 90 days"
        )


# --- identity and configuration ----------------------------------------------------


def test_descriptor_records_the_constraints_teacher_and_its_sampling_digest() -> None:
    teacher = ConstraintsTeacher()
    config = ConstraintsTeacherConfig()

    assert isinstance(teacher, Teacher) and isinstance(teacher, AdversarialTeacher)
    assert teacher.descriptor.provider == "constraints"
    assert teacher.descriptor.model == "compiled-constraints-v2"
    assert teacher.descriptor.configuration_sha256 == config.sampling_sha256
    assert ConstraintsTeacher().descriptor == teacher.descriptor
    for changed in (
        ConstraintsTeacherConfig(seed=2),
        ConstraintsTeacherConfig(twin_filter=False),
        ConstraintsTeacherConfig(ranges={"total": {"low": 0, "high": 10}}),
    ):
        assert ConstraintsTeacher(changed).descriptor.configuration_sha256 != (
            teacher.descriptor.configuration_sha256
        )

    mixed = ConstraintsTeacher(fallback=FakeFallback())
    assert mixed.descriptor.provider == "constraints+fake-llm"
    assert mixed.descriptor.model == "fake-model"
    assert mixed.descriptor.configuration_sha256 != teacher.descriptor.configuration_sha256


def test_loads_the_keyword_and_a_closed_constraints_table(tmp_path: Path) -> None:
    assert load_teacher_config("constraints") == ConstraintsTeacherConfig()

    path = tmp_path / "teacher.toml"
    path.write_text(
        """[teacher]
backend = "constraints"
seed = 7
twin_filter = false

[teacher.ranges]
total = { low = 0, high = 8000, distribution = "log", decimals = 1 }
"order.ageDays" = { low = 0, high = 120 }

[teacher.fallback]
backend = "ollama"
model = "qwen2.5:7b"
""",
        encoding="utf-8",
    )
    config = load_teacher_config(path)
    assert isinstance(config, ConstraintsTeacherConfig)
    assert config.seed == 7 and config.twin_filter is False
    assert config.range_for("order.total") == NumberRange(0, 8000, "log", 1)
    assert config.range_for("order.ageDays") == NumberRange(0, 120)
    assert config.range_for("customer.priorRefunds") is None
    assert isinstance(config.fallback, TeacherConfig)
    assert config.fallback.model == "qwen2.5:7b"
    assert config.public_projection()["fallback"]["backend"] == "ollama"

    teacher = create_teacher(config)
    assert isinstance(teacher, ConstraintsTeacher)
    assert teacher.descriptor.provider == "constraints+ollama"
    assert teacher.descriptor.model == "qwen2.5:7b"
    assert type(teacher.fallback).__name__ == "OllamaTeacher"

    # A file named like the keyword is a file, not the keyword.
    keyword_file = tmp_path / "constraints"
    keyword_file.write_text('[teacher]\nbackend = "constraints"\nseed = 3\n', encoding="utf-8")
    loaded = load_teacher_config(keyword_file)
    assert isinstance(loaded, ConstraintsTeacherConfig) and loaded.seed == 3


def test_a_directory_named_like_the_keyword_does_not_shadow_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "constraints").mkdir()
    monkeypatch.chdir(tmp_path)
    assert load_teacher_config("constraints") == ConstraintsTeacherConfig()


def test_an_unknown_backend_lists_the_valid_ones() -> None:
    for value in ({"backend": "constraint"}, {"backend": "constraint", "model": "x"}):
        with pytest.raises(TeacherConfigurationError, match="expected anthropic, ollama or"):
            teacher_config_from_mapping(value)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"backend": "constraints", "model": "x"}, "unknown constraints teacher"),
        ({"backend": "constraints", "seed": "1"}, "seed must be an integer"),
        ({"backend": "constraints", "ranges": {"total": {"low": 5, "high": 1}}}, "below high"),
        (
            {"backend": "constraints", "ranges": {"total": {"low": 0, "high": 9, "step": 1}}},
            "unknown keys in range 'total'",
        ),
        (
            {
                "backend": "constraints",
                "ranges": {"total": {"low": 0, "high": 9, "distribution": "x"}},
            },
            "uniform, log or count",
        ),
        (
            {"backend": "constraints", "fallback": {"backend": "constraints"}},
            "language-model backend",
        ),
        ({"backend": "constraints", "fallback": {"backend": "ollama"}}, "invalid teacher"),
        ({"backend": "constraints", "near_threshold_share": 2}, "from 0 to 1"),
    ],
)
def test_rejects_invalid_constraints_tables(value: dict[str, Any], message: str) -> None:
    with pytest.raises(TeacherConfigurationError, match=message):
        teacher_config_from_mapping(value)


# --- sampling -----------------------------------------------------------------------


def test_infers_threshold_aware_ranges_from_the_predicates_and_examples() -> None:
    sampler = ConstraintSampler(refund_ir(), ConstraintsTeacherConfig())

    assert sampler.thresholds_for("order.ageDays") == [30.0, 60.0, 90.0]
    assert sampler.range_for("order.ageDays") == NumberRange(0, 180, "uniform", 0)
    assert sampler.thresholds_for("customer.priorRefunds") == [2.0]
    assert sampler.range_for("customer.priorRefunds") == NumberRange(0, 10, "count", 0)
    # Not compared by any predicate: a default range widened by the gold example,
    # with the example's one decimal place.
    assert sampler.thresholds_for("order.total") == []
    assert sampler.range_for("order.total") == NumberRange(0, 130, "uniform", 1)
    assert sampler.facts["order.status"].strings == {"fraudulent", "paid"}

    configured = ConstraintSampler(
        refund_ir(),
        ConstraintsTeacherConfig(ranges={"ageDays": {"low": 0, "high": 120}}),
    )
    assert configured.range_for("order.ageDays") == NumberRange(0, 120)


def test_generates_distinct_decided_cases_dense_at_the_thresholds() -> None:
    ir = refund_ir()
    teacher = ConstraintsTeacher()

    cases = teacher.generate(deepcopy(ir), 300)

    assert len(cases) == 300
    assert len({json.dumps(case.inputs, sort_keys=True) for case in cases}) == 300
    for case in cases:
        validate_case(ir, case)
        validate_case_constraints(ir, case)
        assert case.output == expected_label(case.inputs)
    ages = [case.inputs["order"]["ageDays"] for case in cases]
    assert {29, 30, 31, 59, 60, 61, 89, 90, 91} & set(ages)
    assert sum(age in (29, 30, 31, 59, 60, 61, 89, 90, 91) for age in ages) > 30
    assert {case.output for case in cases} == {"approve", "deny", "review"}
    # Reproducible: a fresh teacher with the same seed yields the same corpus.
    assert ConstraintsTeacher().generate(deepcopy(ir), 300) == cases
    assert ConstraintsTeacher(ConstraintsTeacherConfig(seed=2)).generate(ir, 300) != cases


def test_every_pure_mode_case_has_a_single_field_twin() -> None:
    ir = refund_ir()
    teacher = ConstraintsTeacher()
    for case in teacher.generate(ir, 60):
        proposal = teacher.generate_counterfactual(ir, case)
        assert proposal.twin.output != case.output
        assert proposal.twin.output == expected_label(proposal.twin.inputs)
        assert len(changed_paths(case.inputs, proposal.twin.inputs)) == 1
        assert "moves the constraints' answer to" in proposal.reason


def test_boundary_pairs_sit_one_field_apart_on_either_side_of_each_predicate() -> None:
    ir = refund_ir()
    teacher = ConstraintsTeacher()
    sampler = ConstraintSampler(ir, ConstraintsTeacherConfig())
    for index in range(len(ir["definition"]["constraints"])):
        pair = teacher.generate_boundary_pair(ir, index)
        assert sampler.predicate(index, pair.predicate_false.inputs) is False
        assert sampler.predicate(index, pair.predicate_true.inputs) is True
        assert len(changed_paths(pair.predicate_false.inputs, pair.predicate_true.inputs)) == 1
        for case in (pair.predicate_false, pair.predicate_true):
            assert case.output == expected_label(case.inputs)


def test_feeds_the_production_dataset_and_adversarial_generators(tmp_path: Path) -> None:
    ir = refund_ir()
    teacher = ConstraintsTeacher()

    base = SyntheticDatasetGenerator(teacher, tmp_path).generate(deepcopy(ir), 80)
    adversarial = AdversarialDatasetGenerator(teacher, tmp_path).generate(deepcopy(ir), base)

    assert base.gold_count == 1 and len(base.cases) == 80
    assert base.teacher.provider == "constraints"
    assert adversarial.boundary_count == 2 * len(ir["definition"]["constraints"])
    assert len(adversarial.pairs) == 79
    for case in adversarial.cases:
        assert case.output == expected_label(case.inputs)


def test_samples_every_input_kind_the_case_contract_accepts() -> None:
    ir = refund_ir(
        [
            rule(
                "always",
                op(
                    "||",
                    op("===", path("ticket", "channel"), lit("phone")),
                    op(
                        ">",
                        {
                            "node": "property",
                            "object": path("ticket", "tags"),
                            "property": "length",
                        },
                        lit(2),
                    ),
                ),
                True,
            ),
            rule(
                "always",
                op(
                    "&&",
                    op("!==", path("ticket", "channel"), lit("phone")),
                    op(
                        "<=",
                        {
                            "node": "property",
                            "object": path("ticket", "tags"),
                            "property": "length",
                        },
                        lit(2),
                    ),
                ),
                False,
            ),
        ],
        examples=[],
    )
    ir["inputs"] = [
        {
            "name": "ticket",
            "index": 0,
            "tsType": "Ticket",
            "type": {
                "kind": "object",
                "fields": [
                    {"name": "channel", "optional": False, "type": {"kind": "string"}},
                    {
                        "name": "tags",
                        "optional": False,
                        "type": {"kind": "array", "items": {"kind": "string"}},
                    },
                    {"name": "note", "optional": True, "type": {"kind": "string"}},
                    {
                        "name": "level",
                        "optional": False,
                        "type": {"kind": "enum", "values": ["a", "b"]},
                    },
                    {
                        "name": "pair",
                        "optional": False,
                        "type": {
                            "kind": "tuple",
                            "items": [{"kind": "boolean"}, {"kind": "null"}],
                        },
                    },
                ],
            },
        }
    ]
    ir["output"]["head"] = {"kind": "nominal", "sourceKind": "boolean", "support": [False, True]}

    cases = ConstraintsTeacher().generate(ir, 120)

    assert len(cases) == 120
    for case in cases:
        validate_case(ir, case)
        validate_case_constraints(ir, case)
    channels = {case.inputs["ticket"]["channel"] for case in cases}
    assert "phone" in channels and channels - {"phone"}
    assert any("note" not in case.inputs["ticket"] for case in cases)
    assert any(len(case.inputs["ticket"]["tags"]) == 3 for case in cases)


def single_input_ir(
    name: str,
    input_type: dict[str, Any],
    constraints: list[dict[str, Any]],
    support: list[Any],
    example: tuple[Any, Any],
) -> dict[str, Any]:
    """An expression over one input with a scalar output and one gold example."""

    ir = refund_ir(constraints, examples=[{"inputs": {name: example[0]}, "output": example[1]}])
    ir["inputs"] = [{"name": name, "index": 0, "tsType": "T", "type": input_type}]
    source_kind = "boolean" if all(isinstance(value, bool) for value in support) else "string-union"
    ir["output"]["head"] = {"kind": "nominal", "sourceKind": source_kind, "support": support}
    return ir


def number_fields(*names: str) -> dict[str, Any]:
    return {
        "kind": "object",
        "fields": [{"name": name, "optional": False, "type": {"kind": "number"}} for name in names],
    }


def length(*names: str) -> dict[str, Any]:
    return {"node": "property", "object": path(*names), "property": "length"}


def train_adversarial(ir: dict[str, Any], teacher: ConstraintsTeacher, root: Path, n: int) -> Any:
    base = SyntheticDatasetGenerator(teacher, root).generate(deepcopy(ir), n)
    adversarial = AdversarialDatasetGenerator(teacher, root).generate(deepcopy(ir), base)
    return base, adversarial


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_trains_a_comparison_between_two_inputs(tmp_path: Path, seed: int) -> None:
    # p.x > p.y has no literal threshold: the edits move one field to the other's
    # value (and one step either side), so every kept anchor has a twin and the
    # production counterfactual stage finds it.
    x, y = path("p", "x"), path("p", "y")
    ir = single_input_ir(
        "p",
        number_fields("x", "y"),
        [rule("always", op(">", x, y), "hi"), rule("always", op("<=", x, y), "lo")],
        ["hi", "lo"],
        ({"x": 5, "y": 3}, "hi"),
    )
    teacher = ConstraintsTeacher(ConstraintsTeacherConfig(seed=seed))

    base, adversarial = train_adversarial(ir, teacher, tmp_path, 400)

    assert len(adversarial.pairs) == 399
    assert ConstraintSampler(ir, ConstraintsTeacherConfig()).facts["p.x"].peers == {"p.y"}
    outputs = [case.output for case in base.cases]
    assert 0.3 < outputs.count("hi") / len(outputs) < 0.7
    for case in adversarial.cases:
        assert case.output == ("hi" if case.inputs["p"]["x"] > case.inputs["p"]["y"] else "lo")


def test_counterfactual_stage_repeats_the_twin_filter_search() -> None:
    x, y = path("p", "x"), path("p", "y")
    ir = single_input_ir(
        "p",
        number_fields("x", "y"),
        [rule("always", op(">", x, y), "hi"), rule("always", op("<=", x, y), "lo")],
        ["hi", "lo"],
        ({"x": 5, "y": 3}, "hi"),
    )
    cases = ConstraintsTeacher().generate(deepcopy(ir), 200)
    # A fresh teacher (as after a cached base dataset) finds the same twin for
    # every kept anchor on its first try.
    first, second = ConstraintsTeacher(), ConstraintsTeacher()
    for case in cases:
        twin = first.generate_counterfactual(ir, case)
        assert second.generate_counterfactual(ir, case) == twin
        assert len(changed_paths(case.inputs, twin.twin.inputs)) == 1


def test_trains_an_array_length_threshold_by_appending_or_removing_one_item(
    tmp_path: Path,
) -> None:
    items = {
        "kind": "object",
        "fields": [
            {
                "name": "items",
                "optional": False,
                "type": {"kind": "array", "items": {"kind": "string"}},
            }
        ],
    }
    ir = single_input_ir(
        "c",
        items,
        [
            rule("always", op(">", length("c", "items"), lit(3)), True),
            rule("always", op("<=", length("c", "items"), lit(3)), False),
        ],
        [False, True],
        ({"items": ["a", "b", "c", "d"]}, True),
    )
    teacher = ConstraintsTeacher()

    base, adversarial = train_adversarial(ir, teacher, tmp_path, 200)

    outputs = [case.output for case in base.cases]
    assert 0.2 < outputs.count(True) / len(outputs) < 0.8
    assert adversarial.boundary_count == 4 and len(adversarial.pairs) == 199
    for case in adversarial.cases:
        assert case.output == (len(case.inputs["c"]["items"]) > 3)
    for index in range(2):
        pair = teacher.generate_boundary_pair(ir, index)
        before = pair.predicate_false.inputs["c"]["items"]
        after = pair.predicate_true.inputs["c"]["items"]
        assert abs(len(before) - len(after)) == 1
        shorter, longer = sorted((before, after), key=len)
        assert longer[: len(shorter)] == shorter


def test_fractional_thresholds_get_a_range_that_balances_the_labels() -> None:
    score = path("s", "score")
    ir = single_input_ir(
        "s",
        number_fields("score"),
        [
            rule("always", op("<", score, lit(0.25)), "low"),
            rule("always", op("&&", op(">=", score, lit(0.25)), op("<", score, lit(0.75))), "mid"),
            rule("always", op(">=", score, lit(0.75)), "high"),
        ],
        ["high", "low", "mid"],
        ({"score": 0.5}, "mid"),
    )
    sampler = ConstraintSampler(ir, ConstraintsTeacherConfig())
    assert sampler.range_for("s.score") == NumberRange(0, 1.5, "uniform", 2)

    outputs = [case.output for case in ConstraintsTeacher().generate(ir, 300)]
    for label in ("low", "mid", "high"):
        assert outputs.count(label) / len(outputs) > 0.12

    celsius = path("t", "celsius")
    negative = single_input_ir(
        "t",
        number_fields("celsius"),
        [
            rule("always", op("<", celsius, lit(-2)), True),
            rule("always", op(">=", celsius, lit(-2)), False),
        ],
        [False, True],
        ({"celsius": -3}, True),
    )
    assert ConstraintSampler(negative, ConstraintsTeacherConfig()).range_for(
        "t.celsius"
    ) == NumberRange(-4, 4, "uniform", 0)


# --- incomplete constraints -----------------------------------------------------


def test_pure_mode_names_an_input_the_constraints_do_not_decide() -> None:
    with pytest.raises(TeacherConfigurationError) as raised:
        ConstraintsTeacher().generate(incomplete_ir(), 50)

    message = str(raised.value)
    assert message.startswith("src/refund.sem.ts:16 (nf_aaaaaaaa): the constraints do not decide")
    assert 'For the input {"customer":' in message
    assert "the constraints admit " in message
    assert '"review"' in message
    assert "[teacher.fallback]" in message


def test_contradictory_constraints_admit_no_output() -> None:
    ir = refund_ir(
        [
            rule("always", op(">", AGE, lit(-1)), "deny"),
            rule("never", op(">", AGE, lit(-1)), "deny"),
        ]
    )
    with pytest.raises(TeacherConfigurationError, match="admit no output"):
        ConstraintsTeacher().generate(ir, 5)


def test_pure_mode_rejects_expressions_without_constraints_or_with_object_outputs() -> None:
    with pytest.raises(TeacherConfigurationError, match="declares no constraints"):
        ConstraintsTeacher().generate(refund_ir([]), 5)
    record = refund_ir()
    record["output"] = {"kind": "object", "fields": []}
    with pytest.raises(TeacherConfigurationError, match="output is an object"):
        ConstraintsTeacher().generate(record, 5)


# --- mixed mode -------------------------------------------------------------------


def test_mixed_mode_labels_decided_inputs_and_delegates_the_rest() -> None:
    ir = incomplete_ir()
    fallback = FakeFallback()
    teacher = ConstraintsTeacher(fallback=fallback)

    share = teacher.decided_share(ir)
    cases = teacher.generate(ir, 100)

    assert 0.3 < share < 0.7
    assert len(cases) == 100
    assert fallback.generated == [100 - round(share * 100)]
    for case in cases:
        validate_case_constraints(ir, case)
        if case.inputs["order"]["ageDays"] > 90:
            # The fallback said "review"; the constraints decide these and win.
            assert case.output == "deny"
    assert len({json.dumps(case.inputs, sort_keys=True) for case in cases}) == 100


def test_mixed_mode_uses_the_fallback_for_pairs_the_constraints_cannot_decide() -> None:
    ir = incomplete_ir()
    fallback = FakeFallback()
    teacher = ConstraintsTeacher(fallback=fallback)

    # never(fraudulent -> approve): flipping the status beyond 90 days stays decided.
    status_pair = teacher.generate_boundary_pair(ir, 0)
    assert status_pair.predicate_false.output == status_pair.predicate_true.output == "deny"
    assert fallback.boundaries == []
    # always(ageDays > 90 -> deny): the other side is never decided, so the fallback
    # proposes the pair and the constraints relabel the side they decide.
    age_pair = teacher.generate_boundary_pair(ir, 1)
    assert fallback.boundaries == [1]
    assert age_pair.predicate_true.output == "deny"
    assert age_pair.predicate_false.output == "review"

    anchor = GeneratedCase(
        {
            "customer": {"priorRefunds": 0, "tier": "standard"},
            "order": {"ageDays": 120, "status": "paid", "total": 50},
        },
        "deny",
    )
    twin = teacher.generate_counterfactual(ir, anchor)
    assert fallback.counterfactuals == [anchor]
    assert twin.twin.inputs["order"]["ageDays"] == 10


def test_mixed_mode_sends_no_request_when_the_constraints_decide_everything() -> None:
    fallback = FakeFallback()
    teacher = ConstraintsTeacher(fallback=fallback)
    ir = refund_ir()

    assert teacher.decided_share(ir) == 1.0
    cases = teacher.generate(ir, 50)
    teacher.generate_boundary_pair(ir, 0)
    teacher.generate_counterfactual(ir, cases[0])

    assert len(cases) == 50
    assert fallback.generated == fallback.boundaries == fallback.counterfactuals == []


def test_mixed_mode_delegates_an_expression_without_constraints() -> None:
    fallback = FakeFallback()
    teacher = ConstraintsTeacher(fallback=fallback)

    cases = teacher.generate(refund_ir([]), 12)

    assert fallback.generated == [12] and len(cases) == 12
    assert teacher.decided_share(refund_ir([])) == 0.0


def test_pure_mode_without_a_twin_raises_a_skippable_generation_error() -> None:
    teacher = ConstraintsTeacher()
    ir = refund_ir([rule("always", op(">", AGE, lit(-1)), "deny")])
    anchor = GeneratedCase(
        {
            "customer": {"priorRefunds": 0, "tier": "standard"},
            "order": {"ageDays": 5, "status": "paid", "total": 1},
        },
        "deny",
    )
    with pytest.raises(AdversarialGenerationError, match="no single-field edit"):
        teacher.generate_counterfactual(ir, anchor)


# --- held-out sampling -------------------------------------------------------------


def test_sample_decided_draws_a_separate_stream_and_honours_exclusions() -> None:
    ir = refund_ir()
    teacher = ConstraintsTeacher()
    training = teacher.generate(ir, 200)

    heldout = teacher.sample_decided(ir, 100, exclude=[case.inputs for case in training])

    trained = {json.dumps(case.inputs, sort_keys=True) for case in training}
    assert len(heldout) == 100
    assert not trained & {json.dumps(case.inputs, sort_keys=True) for case in heldout}
    for case in heldout:
        assert case.output == expected_label(case.inputs)
    with pytest.raises(TeacherConfigurationError, match="reserved"):
        teacher.sample_decided(ir, 5, stream="train")
    # Undecided inputs are skipped rather than failing a held-out draw.
    assert len(teacher.sample_decided(incomplete_ir(), 20)) == 20
