"""The held-out constraint sample: coverage, determinism, disjointness and budget.

CPU only and model-free: the sampler draws inputs from the IR's types and
predicates and evaluates the predicates; the verifier's use of the sample is
covered in test_verification.py.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from semantscript_trainer.canonical_input import serialize_canonical_inputs
from semantscript_trainer.constraints import (
    MAXIMUM_CONSTRAINT_EVALUATION_STEPS,
    compile_constraints,
)
from semantscript_trainer.held_out import (
    DEFAULT_HELD_OUT_SAMPLES,
    HeldOutSampleError,
    held_out_evaluation_allowance,
    sample_held_out_inputs,
)

# --- IR helpers (the constraints teacher's refund policy, self-contained) -------


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


def union(*values: str) -> dict[str, Any]:
    return {"kind": "union", "variants": [{"kind": "literal", "value": v} for v in values]}


CUSTOMER = {
    "kind": "object",
    "name": "Customer",
    "fields": [
        {"name": "priorRefunds", "optional": False, "type": {"kind": "number"}},
        {"name": "tier", "optional": False, "type": union("enterprise", "standard")},
    ],
}
ORDER = {
    "kind": "object",
    "name": "Order",
    "fields": [
        {"name": "ageDays", "optional": False, "type": {"kind": "number"}},
        {"name": "status", "optional": False, "type": union("fraudulent", "paid")},
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

    paid_in_window = op("&&", op("===", STATUS, lit("paid")), within_window())
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
        rule("always", op("&&", paid_in_window, op("<=", PRIOR, lit(2))), "approve", "in window"),
        rule("always", op("&&", paid_in_window, op(">", PRIOR, lit(2))), "review", "repeat"),
        rule("never", op(">", PRIOR, lit(2)), "approve", "customer.priorRefunds > 2"),
    ]


def refund_ir(
    constraints: list[dict[str, Any]] | None = None,
    *,
    examples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "semantscript.neural-function",
        "irVersion": 1,
        "stage": "source",
        "id": "nf_" + "a" * 64,
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
            {"name": "customer", "index": 0, "tsType": "Customer", "type": deepcopy(CUSTOMER)},
            {"name": "order", "index": 1, "tsType": "Order", "type": deepcopy(ORDER)},
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


def sample(ir: dict[str, Any], **kwargs: Any) -> Any:
    kwargs.setdefault("seed", 3)
    return sample_held_out_inputs(ir, compile_constraints(ir), **kwargs)


def encode(ir: dict[str, Any], inputs: dict[str, Any]) -> bytes:
    return serialize_canonical_inputs(ir["inputs"], inputs, version=2)


def test_every_constraint_is_sampled_on_both_sides_of_its_boundary_and_inside() -> None:
    ir = refund_ir()
    compiled = compile_constraints(ir)
    drawn = sample(ir, canonical_input_version=2)

    assert DEFAULT_HELD_OUT_SAMPLES == 512
    assert len(drawn.inputs) == 512 and drawn.requested == 512
    assert 0 < drawn.uniform < 512
    assert [coverage.index for coverage in drawn.coverage] == list(range(len(compiled)))
    for coverage in drawn.coverage:
        assert coverage.shortfalls == (), coverage
        assert coverage.boundary_true and coverage.boundary_false
        assert coverage.interior_true and coverage.interior_false
    # The counts are real: each constraint's predicate holds on at least as many
    # sampled inputs as its true-side categories claim, and fails on as many.
    for coverage in drawn.coverage:
        sides = [compiled.evaluate(coverage.index, inputs) for inputs in drawn.inputs]
        assert sides.count(True) >= coverage.boundary_true + coverage.interior_true
        assert sides.count(False) >= coverage.boundary_false + coverage.interior_false
    # Interior draws sit away from the thresholds: some stale order is well past 90
    # days and some fresh one well inside every window.
    ages = [inputs["order"]["ageDays"] for inputs in drawn.inputs]
    assert any(age > 100 for age in ages) and any(age < 25 for age in ages)
    # No input repeats.
    assert len({encode(ir, inputs) for inputs in drawn.inputs}) == 512


def test_the_same_seed_draws_the_same_sample_and_another_seed_another() -> None:
    ir = refund_ir()
    first = sample(ir, seed=5)
    assert sample(ir, seed=5).inputs == first.inputs
    assert sample(deepcopy(ir), seed=5).coverage == first.coverage
    other = sample(ir, seed=6)
    assert other.seed == 6 and other.inputs != first.inputs
    # The size bounds the sample, and a smaller one is still deterministic.
    small = sample(ir, seed=5, size=40)
    assert len(small.inputs) == 40 and sample(ir, seed=5, size=40).inputs == small.inputs


def test_no_sampled_input_is_a_training_input() -> None:
    ir = refund_ir()
    first = sample(ir, canonical_input_version=2)
    corpus = {encode(ir, inputs) for inputs in first.inputs[::2]}
    # The gold example is part of the corpus too.
    corpus.add(encode(ir, ir["definition"]["examples"][0]["inputs"]))
    excluded = sample(ir, canonical_input_version=2, exclude=corpus)
    assert len(excluded.inputs) == 512
    assert not {encode(ir, inputs) for inputs in excluded.inputs} & corpus


def test_a_small_input_space_the_corpus_covers_yields_what_is_left() -> None:
    # One boolean input: two possible inputs, one of them a training input.
    ir = refund_ir(
        [rule("always", {"node": "input", "name": "flag"}, "deny", "flag")],
        examples=[{"inputs": {"flag": True}, "output": "deny"}],
    )
    ir["inputs"] = [{"name": "flag", "index": 0, "tsType": "boolean", "type": {"kind": "boolean"}}]
    corpus = {encode(ir, {"flag": True})}
    drawn = sample(ir, canonical_input_version=2, exclude=corpus, size=16)
    assert drawn.inputs == ({"flag": False},)
    (coverage,) = drawn.coverage
    assert "boundary-true" in coverage.shortfalls and "interior-true" in coverage.shortfalls


def test_a_function_without_constraints_has_an_empty_sample() -> None:
    empty = sample(refund_ir([]))
    assert empty.inputs == () and empty.coverage == () and empty.evaluations == 0


def test_the_sample_stays_within_the_constraint_evaluation_budget() -> None:
    ir = refund_ir()
    compiled = compile_constraints(ir)
    allowance = held_out_evaluation_allowance(compiled, 512)
    assert compiled.node_count * (allowance + 512) <= MAXIMUM_CONSTRAINT_EVALUATION_STEPS
    drawn = sample(ir)
    assert 0 < drawn.evaluations <= allowance

    # Many large constraints: the budget cannot carry the largest sample, and the
    # refusal names the flag and the size that fits.
    heavy = refund_ir(
        complete_constraints() * 20
        + [rule("never", op("===", STATUS, lit("fraudulent")), "approve", "fraud")]
    )
    heavy_compiled = compile_constraints(heavy)
    with pytest.raises(HeldOutSampleError, match=r"rerun with --held-out-samples \d+ or fewer"):
        held_out_evaluation_allowance(heavy_compiled, 50_000)
    largest = MAXIMUM_CONSTRAINT_EVALUATION_STEPS // heavy_compiled.node_count // 2
    assert held_out_evaluation_allowance(heavy_compiled, largest) >= largest

    for size in (0, 50_001, True, 1.5):
        with pytest.raises(HeldOutSampleError, match="sample size"):
            sample(ir, size=size)
    with pytest.raises(HeldOutSampleError, match="seed"):
        sample(ir, seed=-1)
