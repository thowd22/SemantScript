"""The held-out constraint sample: inputs the model never trained on, drawn from the IR.

The corpus constraint check scores every record the model trained on, so a model
that memorised its boundary pairs passes it while breaking a rule a few steps
away from them. The verifier therefore also scores each constraint on a sample
generated from the function's own input types and constraint predicates:

* **boundary** draws: an input with every number on or beside a predicate
  threshold, and a single-field edit of it that flips the constraint's
  predicate; both sides of the boundary are kept;
* **interior** draws: an input on one side of the predicate (true, then false)
  whose numbers are all more than one step from every threshold;
* **uniform** draws: inputs sampled evenly over the inferred ranges.

Every stream is seeded from ``seed``, the function id, the constraint and the
category, so the same seed gives the same sample. No input is a training input:
each candidate's canonical encoding is checked against the ``exclude`` set (the
corpus rows, gold examples and attested cases) and repeats are dropped. No
teacher and no benchmark data are involved.

The work is bounded by the constraint evaluation budget: the draws, predicate
evaluations and the verifier's own contract evaluation of the sample together
stay within ``MAXIMUM_CONSTRAINT_EVALUATION_STEPS``; a sample size the budget
cannot carry is refused, naming ``--held-out-samples``. A category no draw
reached within its share of the work (a predicate that holds nowhere, an input
space the corpus already covers) is recorded as a coverage shortfall, not an
error.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from semantscript_trainer.canonical_input import CanonicalInputError, serialize_canonical_inputs
from semantscript_trainer.constraints import (
    MAXIMUM_CONSTRAINT_EVALUATION_STEPS,
    CompiledConstraints,
    ConstraintConfigurationError,
    ConstraintEvaluationBudget,
    ConstraintEvaluationError,
)
from semantscript_trainer.teacher import JsonValue, NeuralFunctionIr, TeacherConfigurationError

DEFAULT_HELD_OUT_SAMPLES = 512
MAXIMUM_HELD_OUT_SAMPLES = 50_000
# Work units (a draw or one predicate evaluation) allowed per requested input.
HELD_OUT_WORK_FACTOR = 64
# Single-field edits one boundary anchor may try before the next anchor is drawn.
_EDITS_PER_ANCHOR = 64
_DOMAIN = "semantscript.held-out/v1"
_MAXIMUM_ROW_BYTES = 1 * 1024 * 1024


class HeldOutSampleError(ValueError):
    """The held-out sample cannot be drawn within its configured bounds."""


@dataclass(frozen=True, slots=True)
class ConstraintCoverage:
    """How many held-out inputs landed in each category of one constraint."""

    index: int
    boundary_true: int
    boundary_false: int
    interior_true: int
    interior_false: int

    @property
    def shortfalls(self) -> tuple[str, ...]:
        """The categories no draw reached: ``boundary-true``, ``interior-false``, ..."""

        return tuple(
            name
            for name, count in (
                ("boundary-true", self.boundary_true),
                ("boundary-false", self.boundary_false),
                ("interior-true", self.interior_true),
                ("interior-false", self.interior_false),
            )
            if count == 0
        )


@dataclass(frozen=True, slots=True)
class HeldOutSample:
    """Distinct inputs outside the training corpus, in draw order, and their provenance."""

    inputs: tuple[dict[str, JsonValue], ...]
    seed: int
    requested: int
    uniform: int
    coverage: tuple[ConstraintCoverage, ...]
    # Predicate evaluations the sampling spent (at most its work allowance).
    evaluations: int = 0


def held_out_evaluation_allowance(compiled: CompiledConstraints, size: int) -> int:
    """The sampler's work allowance for ``size`` inputs, or raise when none fits.

    The sampling work plus the verifier's ``size`` whole-contract evaluations
    must stay within the aggregate constraint evaluation step budget.
    """

    _check_size(size)
    per_case = max(compiled.node_count, 1)
    affordable = MAXIMUM_CONSTRAINT_EVALUATION_STEPS // per_case
    allowance = min(size * HELD_OUT_WORK_FACTOR, affordable - size)
    if allowance < size:
        largest = max(affordable // 2, 0)
        raise HeldOutSampleError(
            f"a held-out sample of {size} inputs exceeds the constraint evaluation budget "
            f"({MAXIMUM_CONSTRAINT_EVALUATION_STEPS} steps over {per_case} constraint "
            f"nodes); rerun with --held-out-samples {largest} or fewer"
        )
    try:
        compiled.ensure_evaluation_budget(allowance + size)
    except ConstraintConfigurationError as error:  # pragma: no cover - guarded above
        raise HeldOutSampleError(str(error)) from error
    return allowance


def sample_held_out_inputs(
    ir: NeuralFunctionIr,
    compiled: CompiledConstraints,
    *,
    size: int = DEFAULT_HELD_OUT_SAMPLES,
    seed: int,
    exclude: Collection[bytes] = frozenset(),
    canonical_input_version: int = 1,
) -> HeldOutSample:
    """Draw up to ``size`` distinct inputs for the held-out constraint check.

    ``exclude`` holds the canonical encodings (``canonical_input_version``) of
    every input the model trained or was verified on; no returned input encodes
    to one of them. Fewer than ``size`` inputs come back only when the input
    space runs out within the work allowance.
    """

    from semantscript_trainer.teacher_config import ConstraintsTeacherConfig
    from semantscript_trainer.teachers.constraints import ConstraintSampler, _Unsupported

    _check_size(size)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise HeldOutSampleError("held-out seed must be a non-negative integer")
    schema = ir.get("inputs")
    if not isinstance(schema, list):
        raise HeldOutSampleError("IR inputs must be an array")
    function_id = str(ir.get("id", ""))
    constraint_count = len(compiled)
    if constraint_count == 0:
        return HeldOutSample((), seed, size, 0, ())
    allowance = held_out_evaluation_allowance(compiled, size)
    try:
        near = ConstraintSampler(
            ir, ConstraintsTeacherConfig(seed=seed, near_threshold_share=1.0, twin_filter=False)
        )
        spread = ConstraintSampler(
            ir, ConstraintsTeacherConfig(seed=seed, near_threshold_share=0.0, twin_filter=False)
        )
    except (TeacherConfigurationError, _Unsupported) as error:
        raise HeldOutSampleError(f"held-out inputs cannot be sampled: {error}") from error

    state = _State(
        schema=schema,
        version=canonical_input_version,
        excluded=exclude,
        budget=ConstraintEvaluationBudget(),
        compiled=compiled,
    )
    # Half the inputs and half the work go to the per-constraint categories, the
    # rest (and whatever the categories leave) to uniform draws.
    structured_target = size // 2
    per_constraint = max(4, structured_target // constraint_count)
    boundary_quota = max(2, per_constraint // 2)
    interior_quota = max(1, per_constraint // 4)
    task_work = max(1, (allowance // 2) // (3 * constraint_count))
    structured_cap = max(size - size // 4, 1)
    coverage: list[ConstraintCoverage] = []
    for index in range(constraint_count):
        counts = {"boundary-true": 0, "boundary-false": 0, "interior-true": 0, "interior-false": 0}
        if len(state.inputs) < structured_cap:
            rng = _rng(seed, function_id, index, "boundary")
            state.task_work = 0
            for side, inputs in _boundary(near, state, index, rng, task_work):
                if len(state.inputs) >= structured_cap or (
                    counts["boundary-true"] + counts["boundary-false"] >= boundary_quota
                ):
                    break
                if state.accept(inputs):
                    counts[f"boundary-{'true' if side else 'false'}"] += 1
            for wanted in (True, False):
                label = f"interior-{'true' if wanted else 'false'}"
                rng = _rng(seed, function_id, index, label)
                state.task_work = 0
                while (
                    counts[label] < interior_quota
                    and len(state.inputs) < structured_cap
                    and state.task_work < task_work
                ):
                    inputs = _draw(spread, rng)
                    state.task_work += 1
                    if inputs is None or state.seen(inputs):
                        continue
                    side = state.predicate(index, inputs)
                    if side is not wanted or _near_threshold(spread, inputs):
                        continue
                    if state.accept(inputs):
                        counts[label] += 1
        coverage.append(
            ConstraintCoverage(
                index,
                counts["boundary-true"],
                counts["boundary-false"],
                counts["interior-true"],
                counts["interior-false"],
            )
        )
    structured = len(state.inputs)
    # Uniform draws evaluate no predicate; their count only bounds the time spent
    # on an input space the corpus has nearly exhausted.
    rng = _rng(seed, function_id, None, "uniform")
    draws = 0
    while len(state.inputs) < size and draws < size * HELD_OUT_WORK_FACTOR:
        inputs = _draw(spread, rng)
        draws += 1
        if inputs is not None:
            state.accept(inputs)
    return HeldOutSample(
        inputs=tuple(state.inputs),
        seed=seed,
        requested=size,
        uniform=len(state.inputs) - structured,
        coverage=tuple(coverage),
        evaluations=state.evaluations,
    )


@dataclass(slots=True)
class _State:
    schema: Sequence[Mapping[str, Any]]
    version: int
    excluded: Collection[bytes]
    budget: ConstraintEvaluationBudget
    compiled: CompiledConstraints
    inputs: list[dict[str, JsonValue]] = field(default_factory=list)
    keys: set[bytes] = field(default_factory=set)
    # Draws and predicate evaluations of the current category, and of all of them.
    task_work: int = 0
    evaluations: int = 0

    def encode(self, inputs: Mapping[str, JsonValue]) -> bytes | None:
        """The canonical encoding, or None for an input over the row byte limit.

        Any other encoding failure means the sampler and the input schema
        disagree, which would silently empty the sample, so it is raised.
        """

        try:
            return serialize_canonical_inputs(
                self.schema, inputs, version=self.version, maximum_bytes=_MAXIMUM_ROW_BYTES
            )
        except CanonicalInputError as error:
            if error.reason == "limit":
                return None
            raise HeldOutSampleError(
                f"a sampled held-out input cannot be canonically encoded: {error}"
            ) from error

    def seen(self, inputs: Mapping[str, JsonValue]) -> bool:
        encoded = self.encode(inputs)
        return encoded is None or encoded in self.keys or encoded in self.excluded

    def accept(self, inputs: dict[str, JsonValue]) -> bool:
        encoded = self.encode(inputs)
        if encoded is None or encoded in self.keys or encoded in self.excluded:
            return False
        self.keys.add(encoded)
        self.inputs.append(inputs)
        return True

    def predicate(self, index: int, inputs: Mapping[str, JsonValue]) -> bool | None:
        self.task_work += 1
        self.evaluations += 1
        try:
            return self.compiled.evaluate(index, inputs, budget=self.budget)
        except ConstraintEvaluationError:
            return None


def _boundary(
    sampler: Any, state: _State, index: int, rng: random.Random, work_cap: int
) -> Iterator[tuple[bool, dict[str, JsonValue]]]:
    """Both members of boundary pairs for constraint ``index``: (predicate side, input)."""

    while state.task_work < work_cap:
        anchor = _draw(sampler, rng)
        state.task_work += 1
        if anchor is None:
            continue
        side = state.predicate(index, anchor)
        if side is None:
            continue
        tried = 0
        for _leaf, _old, _new, twin in sampler.edits(anchor, rng):
            if tried >= _EDITS_PER_ANCHOR or state.task_work >= work_cap:
                break
            tried += 1
            flipped = state.predicate(index, twin)
            if flipped is None or flipped == side:
                continue
            yield side, anchor
            yield flipped, twin
            break


def _draw(sampler: Any, rng: random.Random) -> dict[str, JsonValue] | None:
    try:
        return sampler.sample(rng)
    except TeacherConfigurationError as error:
        raise HeldOutSampleError(f"held-out inputs cannot be sampled: {error}") from error


def _near_threshold(sampler: Any, inputs: Mapping[str, JsonValue]) -> bool:
    """Whether any number in ``inputs`` is on or one step beside a predicate threshold."""

    from semantscript_trainer.teachers.constraints import _get, _step

    for leaf in sampler.leaves(inputs):
        if not leaf.present or leaf.spec.get("kind") != "number":
            continue
        value = _get(inputs, leaf.segments)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        decimals = sampler.range_for(leaf.key).decimals
        for threshold in sampler.thresholds_for(leaf.key):
            if abs(float(value) - threshold) <= _step(threshold, decimals) + 1e-12:
                return True
    return False


def _rng(seed: int, function_id: str, index: int | None, category: str) -> random.Random:
    material = f"{_DOMAIN}\0{seed}\0{function_id}\0{'' if index is None else index}\0{category}"
    return random.Random(int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16))


def _check_size(size: object) -> None:
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or not 1 <= size <= MAXIMUM_HELD_OUT_SAMPLES
    ):
        raise HeldOutSampleError(
            f"held-out sample size must be an integer from 1 through {MAXIMUM_HELD_OUT_SAMPLES}"
        )


__all__ = [
    "DEFAULT_HELD_OUT_SAMPLES",
    "HELD_OUT_WORK_FACTOR",
    "MAXIMUM_HELD_OUT_SAMPLES",
    "ConstraintCoverage",
    "HeldOutSample",
    "HeldOutSampleError",
    "held_out_evaluation_allowance",
    "sample_held_out_inputs",
]
