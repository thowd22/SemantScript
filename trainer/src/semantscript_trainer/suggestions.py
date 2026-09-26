"""The ``next:`` suggestion under each failed verification gate.

Pure and torch-free: :func:`semantscript_trainer.verification.evaluate_training_result`
collects the evidence (the missed gold cases, the corpus rows with their labels
and predictions, the constraint violations, the calibration numbers) and these
helpers turn it into one remedy from ``diagnostics/remedies.json`` per failure.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from semantscript_trainer.remedies import remedy

# Below this many calibration rows the ECE is noisy, so more cases come first.
MINIMUM_CALIBRATION_ROWS = 200
# At or above this accuracy the head fits; more epochs would not help the ECE.
FITTED_ACCURACY = 0.95
_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_MAXIMUM_LITERAL_CHARACTERS = 600

type Scalar = str | bool | int | float
type RequiredOutputs = Callable[[Mapping[str, Any]], Sequence[Any]]
"""The outputs the expression's active ``always`` constraints require for one input."""


@dataclass(frozen=True, slots=True)
class LabelledCase:
    """One verification record: its inputs, its label and the model's prediction."""

    case_id: str
    inputs: Mapping[str, Any]
    expected: Any
    predicted: Any
    teacher_labelled: bool


@dataclass(frozen=True, slots=True)
class Violation:
    """One record whose raw prediction broke constraint ``index``."""

    index: int
    source: str
    case: LabelledCase


def gold_miss_suggestion(
    misses: Sequence[LabelledCase],
    corpus: Sequence[LabelledCase],
    constraints: Sequence[Mapping[str, Any]],
    required: RequiredOutputs | None = None,
) -> str:
    """The constraint, example or check a set of missed gold or attested cases calls for."""

    if not misses:
        raise ValueError("a gold miss suggestion needs at least one miss")
    rule = shared_rule(misses, corpus, constraints, required)
    if rule is not None:
        predicate, output, count = rule
        return remedy(
            "gold-rule-constraint", predicate=predicate, output=_literal(output), count=count
        )
    best: tuple[LabelledCase, list[LabelledCase]] | None = None
    for miss in misses:
        repeats = [
            row
            for row in corpus
            if row.teacher_labelled
            and _same(row.expected, miss.expected)
            and _same(row.predicted, miss.predicted)
        ]
        if repeats and (best is None or len(repeats) > len(best[1])):
            best = (miss, repeats)
    if best is not None:
        miss, repeats = best
        chosen = min(repeats, key=lambda row: row.case_id)
        return remedy(
            "gold-repeated-miss-example",
            example=example_literal(chosen.inputs, chosen.expected),
            count=len(repeats),
            expected=_literal(miss.expected),
            predicted=_literal(miss.predicted),
        )
    first = misses[0]
    return remedy(
        "gold-check-example",
        case=first.case_id,
        expected=_literal(first.expected),
        predicted=_literal(first.predicted),
    )


def shared_rule(
    misses: Sequence[LabelledCase],
    corpus: Sequence[LabelledCase],
    constraints: Sequence[Mapping[str, Any]],
    required: RequiredOutputs | None = None,
) -> tuple[str, Any, int] | None:
    """A one-field predicate two or more misses share that the labels imply.

    The misses must expect the same scalar output (the compiler accepts
    constraints only on a scalar output) and all satisfy the predicate (one
    string or boolean value, or every number at or above, or at or below, a
    bound); every corpus row where it holds must carry that output, at least
    one teacher-labelled row must be among them, and no constraint may state
    it already: neither in the same source text nor, when ``required`` is
    given, as an active ``always`` that requires that output on every missed
    and covered row. Returns ``(predicate source, output, missed count)`` for
    the predicate that covers the most teacher-labelled rows, else None.
    """

    groups: dict[str, list[LabelledCase]] = {}
    for miss in misses:
        groups.setdefault(_key(miss.expected), []).append(miss)
    stated = {
        _normalize(str(item.get("source", ""))) for item in constraints if isinstance(item, Mapping)
    }
    flattened = [(row, _flatten(row.inputs)) for row in corpus]
    already: dict[str, bool] = {}

    def stated_for(row: LabelledCase, output: Any) -> bool:
        # Whether an active always() already requires ``output`` for this row.
        key = f"{_key(row.inputs)}\u0000{_key(output)}"
        if key not in already:
            assert required is not None
            already[key] = any(_same(value, output) for value in required(row.inputs))
        return already[key]

    for group in sorted(groups.values(), key=len, reverse=True):
        if len(group) < 2:
            break
        output = group[0].expected
        if not _is_scalar(output):
            continue
        fields = [_flatten(miss.inputs) for miss in group]
        shared = set(fields[0]).intersection(*fields[1:])
        best: tuple[int, int, str] | None = None
        for path in sorted(shared):
            for rank, source, holds in _candidates(path, [field[path] for field in fields]):
                if _normalize(source) in stated:
                    continue
                covered = [
                    row for row, values in flattened if path in values and holds(values[path])
                ]
                if any(not _same(row.expected, output) for row in covered):
                    continue
                support = sum(row.teacher_labelled for row in covered)
                if support == 0:
                    continue
                if required is not None and all(
                    stated_for(row, output) for row in (*group, *covered)
                ):
                    continue
                if best is None or (support, -rank) > (best[0], -best[1]):
                    best = (support, rank, source)
        if best is not None:
            return best[2], output, len(group)
    return None


def violation_suggestion(violations: Sequence[Violation], current_cases: int) -> str:
    """An example for the constraint broken most often, or denser data for isolated misses."""

    if not violations:
        raise ValueError("a violation suggestion needs at least one violation")
    by_index: dict[int, list[Violation]] = {}
    for violation in violations:
        by_index.setdefault(violation.index, []).append(violation)
    index, group = max(by_index.items(), key=lambda item: (len(item[1]), -item[0]))
    records = {violation.case.case_id: violation for violation in group}
    if len(records) >= 2:
        chosen = records[min(records)]
        return remedy(
            "violation-example",
            example=example_literal(chosen.case.inputs, chosen.case.expected),
            index=index,
            source=" ".join(chosen.source.split()) or "no source",
            count=len(records),
        )
    return remedy("violation-denser-data", cases=max(2 * current_cases, 1), current=current_cases)


def calibration_suggestion(
    *, ece: float, rows: int, accuracy: float, current_cases: int, epochs: int
) -> str:
    """More cases while the calibration split is small or the head fits; else more epochs."""

    if rows < MINIMUM_CALIBRATION_ROWS or accuracy >= FITTED_ACCURACY:
        return remedy(
            "calibration-more-cases",
            cases=max(2 * current_cases, 1),
            current=current_cases,
            ece=f"{ece:.4f}",
            rows=rows,
        )
    return remedy(
        "calibration-more-epochs",
        epochs=max(epochs + 1, math.ceil(epochs * 1.5)),
        current=epochs,
        accuracy=f"{accuracy:.4f}",
    )


def type_error_suggestion() -> str:
    return remedy("type-no-cache")


def seed_retry_suggestion(
    *, attempts: int, first_seed: int, last_seed: int, maximum_seed: int
) -> str | None:
    """The retry to run after a narrow failure: turn it on, or continue from the next seed."""

    if attempts <= 1:
        return remedy("seed-retry-attempts", attempts=3)
    if last_seed + 1 > maximum_seed:
        return None
    return remedy("seed-retry-next-seed", seed=last_seed + 1, first=first_seed, last=last_seed)


def example_literal(inputs: Mapping[str, Any], output: Any) -> str:
    """``{ inputs: {...}, output: ... }``, ready to paste into a sema call's examples."""

    return f"{{ inputs: {_literal(dict(inputs))}, output: {_literal(output)} }}"


def _candidates(path: str, values: list[Scalar]) -> list[tuple[int, str, Any]]:
    """(rank, source, test) predicates over one field that every value satisfies."""

    candidates: list[tuple[int, str, Any]] = []
    first = values[0]
    if isinstance(first, (str, bool)) and all(
        type(value) is type(first) and value == first for value in values
    ):
        candidates.append(
            (
                0,
                f"{path} === {_literal(first)}",
                lambda value, first=first: type(value) is type(first) and value == first,
            )
        )
    elif all(_numeric(value) for value in values):
        low = min(values)
        high = max(values)
        candidates.append(
            (
                1,
                f"{path} >= {_literal(low)}",
                lambda value, low=low: _numeric(value) and value >= low,
            )
        )
        candidates.append(
            (
                1,
                f"{path} <= {_literal(high)}",
                lambda value, high=high: _numeric(value) and value <= high,
            )
        )
    return candidates


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Scalar]:
    """Every scalar leaf under identifier keys, as ``order.status`` style paths."""

    leaves: dict[str, Scalar] = {}
    for key, item in value.items():
        if not isinstance(key, str) or _IDENTIFIER.match(key) is None:
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            leaves.update(_flatten(item, path))
        elif isinstance(item, (str, bool)) or _numeric(item):
            leaves[path] = item
    return leaves


def _is_scalar(value: object) -> bool:
    return isinstance(value, (str, bool)) or _numeric(value)


def _numeric(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _literal(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    if len(text) > _MAXIMUM_LITERAL_CHARACTERS:
        return text[: _MAXIMUM_LITERAL_CHARACTERS - 1] + "…"
    return text


def _key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _same(left: object, right: object) -> bool:
    return _key(left) == _key(right)


def _normalize(source: str) -> str:
    return re.sub(r"\s+", "", source)
