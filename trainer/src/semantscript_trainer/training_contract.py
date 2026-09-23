"""Typed training rows, output-label mapping, and deterministic held-out splits."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Literal, cast

from semantscript_trainer.adversarial import AdversarialDataset
from semantscript_trainer.case_contract import build_case_schema, validate_case
from semantscript_trainer.dataset import TrainingDataset
from semantscript_trainer.teacher import GeneratedCase, JsonValue, NeuralFunctionIr

MAXIMUM_TRAINING_ROW_COUNT = 50_000
MAXIMUM_SPLIT_SEED = 2**63 - 1

_FUNCTION_ID = re.compile(r"^nf_[a-f0-9]{64}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_CANONICAL_DECIMAL = re.compile(r"^(?!-0$)-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$")

type HeadKind = Literal["nominal", "ordinal"]
type HeadParameterization = Literal["binary-sigmoid", "categorical-softmax"]
type TrainingOrigin = Literal["gold", "synthetic", "constraint-boundary", "counterfactual"]


class TrainingContractError(ValueError):
    """An IR, dataset, label, or split cannot form a safe training contract."""


@dataclass(frozen=True, slots=True, init=False)
class TrainingHeadContract:
    """Closed scalar-head support and its model-ABI parameterization."""

    kind: HeadKind
    source_kind: str
    support: tuple[JsonValue, ...]
    ordinal: bool
    parameterization: HeadParameterization
    logit_count: int
    _indices: Mapping[tuple[str, object], int] = field(repr=False)

    def __init__(
        self,
        *,
        kind: HeadKind,
        source_kind: str,
        support: Sequence[JsonValue],
    ) -> None:
        if kind not in ("nominal", "ordinal"):
            raise TrainingContractError("training head kind must be nominal or ordinal")
        if not isinstance(source_kind, str) or not source_kind:
            raise TrainingContractError("training head source kind must be nonempty")
        resolved = tuple(support)
        if len(resolved) < 2:
            raise TrainingContractError("training head support must contain at least two labels")
        _validate_head_family(kind, source_kind, resolved)
        indices: dict[tuple[str, object], int] = {}
        for index, value in enumerate(resolved):
            key = _semantic_scalar_key(value, context=f"support[{index}]")
            if key in indices:
                raise TrainingContractError(
                    "training head support contains duplicate semantic labels"
                )
            indices[key] = index
        boolean = source_kind == "boolean"
        if boolean and not (len(resolved) == 2 and resolved[0] is False and resolved[1] is True):
            raise TrainingContractError("boolean training support must be [false, true]")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "support", resolved)
        object.__setattr__(self, "ordinal", kind == "ordinal")
        object.__setattr__(
            self,
            "parameterization",
            "binary-sigmoid" if boolean else "categorical-softmax",
        )
        object.__setattr__(self, "logit_count", 1 if boolean else len(resolved))
        object.__setattr__(self, "_indices", indices)

    def label_index(self, value: JsonValue, /) -> int:
        """Map one semantic scalar label to its stable support index."""

        key = _semantic_scalar_key(value, context="training label")
        index = self._indices.get(key)
        if index is None:
            raise TrainingContractError("training label is outside the head support")
        return index


@dataclass(frozen=True, slots=True, init=False)
class TrainingRow:
    """One immutable labeled row plus its leakage-prevention split group."""

    row_id: str
    group_id: str
    origin: TrainingOrigin
    label_index: int
    _inputs_json: bytes = field(repr=False)

    def __init__(
        self,
        *,
        row_id: str,
        group_id: str,
        origin: TrainingOrigin,
        inputs: dict[str, JsonValue],
        label_index: int,
    ) -> None:
        if not isinstance(row_id, str) or not row_id:
            raise TrainingContractError("training row ID must be nonempty")
        if not isinstance(group_id, str) or not group_id:
            raise TrainingContractError("training row group ID must be nonempty")
        if origin not in ("gold", "synthetic", "constraint-boundary", "counterfactual"):
            raise TrainingContractError("training row origin is invalid")
        if isinstance(label_index, bool) or not isinstance(label_index, int) or label_index < 0:
            raise TrainingContractError("training row label index is invalid")
        if not isinstance(inputs, dict):
            raise TrainingContractError("training row inputs must be an object")
        try:
            encoded = _canonical_json_bytes(inputs)
        except (TypeError, ValueError, OverflowError, UnicodeEncodeError, RecursionError) as error:
            raise TrainingContractError(
                f"training row inputs are not canonical JSON: {error}"
            ) from error
        object.__setattr__(self, "row_id", row_id)
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "label_index", label_index)
        object.__setattr__(self, "_inputs_json", encoded)

    @property
    def inputs(self) -> dict[str, JsonValue]:
        value = json.loads(self._inputs_json)
        if not isinstance(value, dict):
            raise AssertionError("canonical training inputs are not an object")
        return cast(dict[str, JsonValue], value)


@dataclass(frozen=True, slots=True)
class TrainingCorpus:
    """Identity-bound base and adversarial rows for one scalar function."""

    function_id: str
    semantic_sha256: str
    base_dataset_sha256: str
    adversarial_dataset_sha256: str | None
    head: TrainingHeadContract
    rows: tuple[TrainingRow, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.function_id, str)
            or _FUNCTION_ID.fullmatch(self.function_id) is None
        ):
            raise TrainingContractError("training corpus function ID is invalid")
        for name in ("semantic_sha256", "base_dataset_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise TrainingContractError(f"training corpus {name} is invalid")
        if self.adversarial_dataset_sha256 is not None and (
            not isinstance(self.adversarial_dataset_sha256, str)
            or _SHA256.fullmatch(self.adversarial_dataset_sha256) is None
        ):
            raise TrainingContractError("training corpus adversarial dataset digest is invalid")
        if not isinstance(self.head, TrainingHeadContract):
            raise TrainingContractError("training corpus head is invalid")
        if not isinstance(self.rows, tuple) or not self.rows:
            raise TrainingContractError("training corpus must contain at least one row")
        if len(self.rows) > MAXIMUM_TRAINING_ROW_COUNT:
            raise TrainingContractError(
                f"training corpus exceeds maximum row count {MAXIMUM_TRAINING_ROW_COUNT}"
            )
        if any(not isinstance(row, TrainingRow) for row in self.rows):
            raise TrainingContractError("training corpus contains an invalid row")
        row_ids = [row.row_id for row in self.rows]
        if len(set(row_ids)) != len(row_ids):
            raise TrainingContractError("training corpus contains duplicate row IDs")
        if any(row.label_index >= len(self.head.support) for row in self.rows):
            raise TrainingContractError("training corpus contains an out-of-range label index")


@dataclass(frozen=True, slots=True)
class HeldOutSplitConfig:
    """Deterministic group-aware held-out split controls."""

    evaluation_ratio: float = 0.2
    seed: int = 1

    def __post_init__(self) -> None:
        ratio = self.evaluation_ratio
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
            or not 0 < ratio < 1
        ):
            raise TrainingContractError("evaluation_ratio must be a finite number between 0 and 1")
        object.__setattr__(self, "evaluation_ratio", float(ratio))
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= MAXIMUM_SPLIT_SEED
        ):
            raise TrainingContractError(
                f"split seed must be an integer from 0 through {MAXIMUM_SPLIT_SEED}"
            )


@dataclass(frozen=True, slots=True)
class TrainingSplit:
    """Rows partitioned without splitting any linked semantic group."""

    training: tuple[TrainingRow, ...]
    evaluation: tuple[TrainingRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.training, tuple) or not isinstance(self.evaluation, tuple):
            raise TrainingContractError("training split partitions must be tuples")
        if not self.training:
            raise TrainingContractError("training split must retain at least one training row")
        all_rows = self.training + self.evaluation
        if any(not isinstance(row, TrainingRow) for row in all_rows):
            raise TrainingContractError("training split contains an invalid row")
        if len({row.row_id for row in all_rows}) != len(all_rows):
            raise TrainingContractError("training split contains duplicate row IDs")
        training_groups = {row.group_id for row in self.training}
        evaluation_groups = {row.group_id for row in self.evaluation}
        if training_groups & evaluation_groups:
            raise TrainingContractError("training split divides a linked row group")


def derive_training_head(ir: NeuralFunctionIr, /) -> TrainingHeadContract:
    """Derive the scalar training head and stable support order from compiler IR."""

    if not isinstance(ir, dict):
        raise TrainingContractError("IR must be an object")
    try:
        build_case_schema(ir)
    except Exception as error:
        raise TrainingContractError(f"IR case contract is invalid: {error}") from error
    output = ir.get("output")
    if not isinstance(output, Mapping) or output.get("kind") != "scalar":
        raise TrainingContractError("TASK-5.6 training supports only scalar outputs")
    head = output.get("head")
    if not isinstance(head, Mapping):
        raise TrainingContractError("IR scalar output head must be an object")
    kind = head.get("kind")
    source_kind = head.get("sourceKind")
    if kind not in ("nominal", "ordinal") or not isinstance(source_kind, str):
        raise TrainingContractError("IR scalar output head kind is invalid")
    if source_kind in ("bounded-int", "bounded-number"):
        if kind != "ordinal":
            raise TrainingContractError("bounded numeric support must be ordinal")
        support = _decimal_support(head)
    else:
        raw_support = head.get("support")
        if not isinstance(raw_support, list):
            raise TrainingContractError("IR scalar output support must be an array")
        support = tuple(cast(JsonValue, value) for value in raw_support)
    _validate_head_family(cast(HeadKind, kind), source_kind, support)
    return TrainingHeadContract(
        kind=cast(HeadKind, kind),
        source_kind=source_kind,
        support=support,
    )


def assemble_training_corpus(
    ir: NeuralFunctionIr,
    base: TrainingDataset,
    adversarial: AdversarialDataset | None = None,
    /,
) -> TrainingCorpus:
    """Combine identity-matched base and adversarial rows into one corpus."""

    if not isinstance(base, TrainingDataset):
        raise TrainingContractError("base must be a TrainingDataset")
    if adversarial is not None and not isinstance(adversarial, AdversarialDataset):
        raise TrainingContractError("adversarial must be an AdversarialDataset")
    head = derive_training_head(ir)
    function_id = ir.get("id")
    semantic_sha256 = ir.get("semanticSha256")
    if function_id != base.function_id or semantic_sha256 != base.semantic_sha256:
        raise TrainingContractError("base dataset identity does not match the IR")
    if adversarial is not None and (
        adversarial.function_id != base.function_id
        or adversarial.base_dataset_sha256 != base.dataset_sha256
    ):
        raise TrainingContractError("adversarial dataset identity does not match the base dataset")
    combined_count = len(base.cases) + (0 if adversarial is None else len(adversarial.cases))
    if combined_count > MAXIMUM_TRAINING_ROW_COUNT:
        raise TrainingContractError(
            f"training corpus exceeds maximum row count {MAXIMUM_TRAINING_ROW_COUNT}"
        )

    source_pairs: dict[int, str] = {}
    if adversarial is not None:
        for pair in adversarial.pairs:
            source_index = pair.source_case_index
            if not 0 <= source_index < len(base.cases):
                raise TrainingContractError("counterfactual pair source index is out of range")
            if base.cases[source_index].origin != "synthetic":
                raise TrainingContractError("counterfactual pair source must be synthetic")
            if source_index in source_pairs:
                raise TrainingContractError(
                    "a base row cannot source more than one counterfactual pair"
                )
            source_pairs[source_index] = pair.pair_id

    rows: list[TrainingRow] = []
    for index, case in enumerate(base.cases):
        _validate_row_case(ir, case.inputs, case.output, f"base case {index}")
        group_id = source_pairs.get(index, f"base:{index}")
        rows.append(
            TrainingRow(
                row_id=f"base:{index}",
                group_id=group_id,
                origin=case.origin,
                inputs=case.inputs,
                label_index=head.label_index(case.output),
            )
        )

    if adversarial is not None:
        _append_adversarial_rows(ir, base, adversarial, head, rows)
    if len(rows) > MAXIMUM_TRAINING_ROW_COUNT:
        raise TrainingContractError(
            f"training corpus exceeds maximum row count {MAXIMUM_TRAINING_ROW_COUNT}"
        )
    return TrainingCorpus(
        function_id=base.function_id,
        semantic_sha256=base.semantic_sha256,
        base_dataset_sha256=base.dataset_sha256,
        adversarial_dataset_sha256=(None if adversarial is None else adversarial.dataset_sha256),
        head=head,
        rows=tuple(rows),
    )


def split_training_corpus(
    corpus: TrainingCorpus,
    config: HeldOutSplitConfig | None = None,
    /,
) -> TrainingSplit:
    """Deterministically split complete groups while preserving source row order."""

    if not isinstance(corpus, TrainingCorpus):
        raise TrainingContractError("split requires a TrainingCorpus")
    if config is None:
        config = HeldOutSplitConfig()
    if not isinstance(config, HeldOutSplitConfig):
        raise TrainingContractError("split config must be a HeldOutSplitConfig")
    groups: dict[str, list[TrainingRow]] = defaultdict(list)
    for row in corpus.rows:
        groups[row.group_id].append(row)
    if len(groups) < 2:
        return TrainingSplit(training=corpus.rows, evaluation=())

    # SPEC 4.1 requires every source example to be used unchanged in both
    # training and verification. Gold groups are therefore never calibration-only.
    mandatory_training_groups = {
        group_id for group_id, rows in groups.items() if any(row.origin == "gold" for row in rows)
    }

    seed_bytes = config.seed.to_bytes(8, byteorder="big", signed=False)
    ranked = sorted(
        (group_id for group_id in groups if group_id not in mandatory_training_groups),
        key=lambda group_id: (
            hashlib.sha256(seed_bytes + b"\0" + group_id.encode("utf-8")).digest(),
            group_id,
        ),
    )
    maximum_selected_groups = len(ranked) if mandatory_training_groups else len(ranked) - 1
    if maximum_selected_groups < 1:
        return TrainingSplit(training=corpus.rows, evaluation=())
    target = max(
        1, min(len(corpus.rows) - 1, math.floor(len(corpus.rows) * config.evaluation_ratio + 0.5))
    )
    prefix_rows = 0
    candidates: list[tuple[int, int]] = []
    for count, group_id in enumerate(ranked[:maximum_selected_groups], start=1):
        prefix_rows += len(groups[group_id])
        candidates.append((abs(prefix_rows - target), count))
    selected_count = min(candidates)[1]
    evaluation_groups = set(ranked[:selected_count])
    training = tuple(row for row in corpus.rows if row.group_id not in evaluation_groups)
    evaluation = tuple(row for row in corpus.rows if row.group_id in evaluation_groups)
    return TrainingSplit(training=training, evaluation=evaluation)


def _append_adversarial_rows(
    ir: NeuralFunctionIr,
    base: TrainingDataset,
    adversarial: AdversarialDataset,
    head: TrainingHeadContract,
    rows: list[TrainingRow],
) -> None:
    by_id = {case.case_id: case for case in adversarial.cases}
    pair_by_id = {pair.pair_id: pair for pair in adversarial.pairs}
    boundary_groups: dict[int, set[bool]] = defaultdict(set)
    constraint_count = _constraint_count(ir)

    for case in adversarial.cases:
        _validate_row_case(ir, case.inputs, case.output, f"adversarial case {case.case_id}")
        if case.tag == "constraint-boundary":
            index = case.constraint_index
            if index is None or not 0 <= index < constraint_count:
                raise TrainingContractError("constraint-boundary row index is out of range")
            if case.predicate_result in boundary_groups[index]:
                raise TrainingContractError("constraint-boundary group repeats a predicate side")
            boundary_groups[index].add(cast(bool, case.predicate_result))
            group_id = f"boundary:{index}"
        else:
            if case.pair_id not in pair_by_id:
                raise TrainingContractError("counterfactual row has no matching pair")
            group_id = cast(str, case.pair_id)
        rows.append(
            TrainingRow(
                row_id=case.case_id,
                group_id=group_id,
                origin=case.tag,
                inputs=case.inputs,
                label_index=head.label_index(case.output),
            )
        )

    if set(boundary_groups) != set(range(constraint_count)) or any(
        sides != {False, True} for sides in boundary_groups.values()
    ):
        raise TrainingContractError("adversarial boundary rows do not cover both predicate sides")
    for pair in adversarial.pairs:
        anchor = by_id.get(pair.anchor_case_id)
        twin = by_id.get(pair.twin_case_id)
        if (
            anchor is None
            or twin is None
            or anchor.pair_id != pair.pair_id
            or twin.pair_id != pair.pair_id
            or anchor.pair_role != "anchor"
            or twin.pair_role != "twin"
        ):
            raise TrainingContractError("counterfactual pair links are inconsistent")
        source = base.cases[pair.source_case_index]
        if not _case_equals(anchor.inputs, anchor.output, source.inputs, source.output):
            raise TrainingContractError("counterfactual anchor does not match its base source row")


def _validate_row_case(
    ir: NeuralFunctionIr,
    inputs: dict[str, JsonValue],
    output: JsonValue,
    context: str,
) -> None:
    try:
        validate_case(ir, GeneratedCase(inputs=inputs, output=output))
    except Exception as error:
        raise TrainingContractError(f"{context} is invalid for the IR: {error}") from error


def _validate_head_family(
    kind: HeadKind,
    source_kind: str,
    support: tuple[JsonValue, ...],
) -> None:
    if source_kind == "boolean":
        if kind != "nominal" or not (
            len(support) == 2 and support[0] is False and support[1] is True
        ):
            raise TrainingContractError("boolean head must be nominal with support [false, true]")
        return
    if source_kind in ("string-union", "string-enum"):
        if kind != "nominal" or any(not isinstance(value, str) for value in support):
            raise TrainingContractError("nominal string head has invalid support")
        return
    if source_kind == "number-enum":
        if kind != "nominal" or any(not _is_number(value) for value in support):
            raise TrainingContractError("nominal number head has invalid support")
        return
    if source_kind == "ordinal-string":
        if kind != "ordinal" or any(not isinstance(value, str) for value in support):
            raise TrainingContractError("ordinal string head has invalid support")
        return
    if source_kind in ("bounded-int", "bounded-number"):
        if kind != "ordinal" or any(not _is_number(value) for value in support):
            raise TrainingContractError("ordinal number head has invalid support")
        return
    raise TrainingContractError(f"unsupported training head source kind {source_kind!r}")


def _decimal_support(head: Mapping[str, Any]) -> tuple[JsonValue, ...]:
    raw = head.get("supportDecimal")
    if not isinstance(raw, list):
        raise TrainingContractError("ordinal numeric supportDecimal must be an array")
    integral = head.get("sourceKind") == "bounded-int"
    result: list[JsonValue] = []
    for index, token in enumerate(raw):
        if not isinstance(token, str) or _CANONICAL_DECIMAL.fullmatch(token) is None:
            raise TrainingContractError(f"supportDecimal[{index}] is not canonical decimal text")
        try:
            decimal = Decimal(token)
            if integral and decimal != decimal.to_integral_value():
                raise TrainingContractError(
                    f"supportDecimal[{index}] must be integral for bounded-int"
                )
            value: int | float = int(decimal) if integral else float(decimal)
        except (InvalidOperation, OverflowError, ValueError) as error:
            raise TrainingContractError(f"supportDecimal[{index}] is invalid") from error
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            raise TrainingContractError(f"supportDecimal[{index}] is not finite binary64")
        result.append(value)
    return tuple(result)


def _semantic_scalar_key(value: JsonValue, *, context: str) -> tuple[str, object]:
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise TrainingContractError(f"{context} contains invalid Unicode") from error
        return ("string", value)
    if _is_number(value):
        try:
            number = float(value)
        except (OverflowError, ValueError) as error:
            raise TrainingContractError(f"{context} is not finite binary64") from error
        if not math.isfinite(number):
            raise TrainingContractError(f"{context} is not finite binary64")
        return ("number", struct.pack(">d", number))
    raise TrainingContractError(f"{context} must be a boolean, string, or finite number")


def _case_equals(
    left_inputs: dict[str, JsonValue],
    left_output: JsonValue,
    right_inputs: dict[str, JsonValue],
    right_output: JsonValue,
) -> bool:
    return _canonical_json_bytes(
        {"inputs": left_inputs, "output": left_output}
    ) == _canonical_json_bytes({"inputs": right_inputs, "output": right_output})


def _constraint_count(ir: NeuralFunctionIr) -> int:
    definition = ir.get("definition")
    constraints = definition.get("constraints") if isinstance(definition, Mapping) else None
    if not isinstance(constraints, list):
        raise TrainingContractError("IR definition constraints must be an array")
    return len(constraints)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="strict")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


__all__ = [
    "MAXIMUM_SPLIT_SEED",
    "MAXIMUM_TRAINING_ROW_COUNT",
    "HeadKind",
    "HeadParameterization",
    "HeldOutSplitConfig",
    "TrainingContractError",
    "TrainingCorpus",
    "TrainingHeadContract",
    "TrainingOrigin",
    "TrainingRow",
    "TrainingSplit",
    "assemble_training_corpus",
    "derive_training_head",
    "split_training_corpus",
]
