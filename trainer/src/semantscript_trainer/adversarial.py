"""Constraint-boundary and counterfactual adversarial dataset sidecars."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from semantscript_trainer.case_contract import build_case_schema, validate_case
from semantscript_trainer.constraints import (
    CompiledConstraints,
    ConstraintConfigurationError,
    ConstraintEvaluationBudget,
    ConstraintEvaluationError,
    ConstraintViolationError,
    compile_constraints,
    predicate_depends_on_inputs,
)
from semantscript_trainer.dataset import (
    MAXIMUM_DATASET_CACHE_BYTES,
    DatasetCacheError,
    TrainingDataset,
    _cache_entry_exists,
    _canonical_document_bytes,
    _canonical_json_bytes,
    _exclusive_lock,
    _fsync_directory,
    _loads_dataset_json,
    _read_bounded_regular_file,
)
from semantscript_trainer.teacher import (
    AdversarialTeacher,
    BoundaryPairProposal,
    CounterfactualProposal,
    GeneratedCase,
    JsonValue,
    NeuralFunctionIr,
    TeacherConfigurationError,
    TeacherDescriptor,
    TeacherResponseError,
    TeacherTransportError,
)

ADVERSARIAL_DATASET_KIND = "semantscript.adversarial-dataset"
ADVERSARIAL_DATASET_VERSION = 1
MAXIMUM_ADVERSARIAL_CASE_COUNT = 50_000
MAXIMUM_COMBINED_CASE_COUNT = 50_000
_log = logging.getLogger(__name__)
MAXIMUM_GENERATION_ATTEMPTS = 10

_REQUEST_DIGEST_DOMAIN = b"semantscript.adversarial-request/v1\0"
_PAYLOAD_DIGEST_DOMAIN = b"semantscript.adversarial-payload/v1\0"
_CASE_ID_DOMAIN = b"semantscript.adversarial-case/v1\0"
_PAIR_ID_DOMAIN = b"semantscript.counterfactual-pair/v1\0"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_FUNCTION_ID = re.compile(r"^nf_[a-f0-9]{64}$")
_CASE_ID = re.compile(r"^ac_[a-f0-9]{64}$")
_PAIR_ID = re.compile(r"^cf_[a-f0-9]{64}$")

type AdversarialTag = Literal["constraint-boundary", "counterfactual"]
type BoundarySide = Literal["predicate-false", "predicate-true"]
type PairRole = Literal["anchor", "twin"]


class AdversarialDatasetError(RuntimeError):
    """Base class for adversarial generation and cache failures."""


class AdversarialConfigurationError(AdversarialDatasetError, ValueError):
    """The generation request or source dataset is invalid."""


class AdversarialGenerationError(AdversarialDatasetError):
    """A teacher did not produce a locally valid adversarial case."""


class UnsynthesizableConstraintError(AdversarialGenerationError):
    """A constraint has no locally demonstrable pair of predicate sides."""


class AdversarialCacheError(AdversarialDatasetError):
    """An adversarial sidecar cannot be read, validated, or published safely."""


@dataclass(frozen=True, slots=True)
class AdversarialGenerationConfig:
    """Cost and retry controls for adversarial augmentation."""

    counterfactual_ratio: float = 1.0
    maximum_attempts: int = 3

    def __post_init__(self) -> None:
        ratio = self.counterfactual_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise AdversarialConfigurationError(
                "counterfactual_ratio must be a finite number from 0 through 1"
            )
        if not math.isfinite(float(ratio)) or ratio < 0 or ratio > 1:
            raise AdversarialConfigurationError(
                "counterfactual_ratio must be a finite number from 0 through 1"
            )
        object.__setattr__(
            self,
            "counterfactual_ratio",
            0.0 if float(ratio) == 0 else float(ratio),
        )
        attempts = self.maximum_attempts
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 1
            or attempts > MAXIMUM_GENERATION_ATTEMPTS
        ):
            raise AdversarialConfigurationError(
                f"maximum_attempts must be an integer from 1 through {MAXIMUM_GENERATION_ATTEMPTS}"
            )

    def document(self) -> dict[str, int | float]:
        return {
            "counterfactualRatio": float(self.counterfactual_ratio),
            "maximumAttempts": self.maximum_attempts,
        }


@dataclass(frozen=True, slots=True, init=False)
class AdversarialCase:
    """One immutable tagged row in the adversarial sidecar."""

    case_id: str
    tag: AdversarialTag
    constraint_index: int | None
    predicate_result: bool | None
    pair_id: str | None
    pair_role: PairRole | None
    _inputs_json: bytes = field(repr=False)
    _output_json: bytes = field(repr=False)

    def __init__(
        self,
        *,
        case_id: str,
        inputs: dict[str, JsonValue],
        output: JsonValue,
        tag: AdversarialTag,
        constraint_index: int | None = None,
        predicate_result: bool | None = None,
        pair_id: str | None = None,
        pair_role: PairRole | None = None,
    ) -> None:
        if not isinstance(case_id, str) or _CASE_ID.fullmatch(case_id) is None:
            raise AdversarialConfigurationError("adversarial case_id is invalid")
        if not isinstance(inputs, dict):
            raise AdversarialConfigurationError("adversarial case inputs must be an object")
        if tag not in ("constraint-boundary", "counterfactual"):
            raise AdversarialConfigurationError("adversarial case tag is invalid")
        if tag == "constraint-boundary":
            if (
                isinstance(constraint_index, bool)
                or not isinstance(constraint_index, int)
                or constraint_index < 0
                or not isinstance(predicate_result, bool)
                or pair_id is not None
                or pair_role is not None
            ):
                raise AdversarialConfigurationError("constraint-boundary metadata is invalid")
        else:
            if (
                constraint_index is not None
                or predicate_result is not None
                or not isinstance(pair_id, str)
                or _PAIR_ID.fullmatch(pair_id) is None
                or pair_role not in ("anchor", "twin")
            ):
                raise AdversarialConfigurationError("counterfactual case metadata is invalid")
        try:
            inputs_json = _canonical_json_bytes(inputs)
            output_json = _canonical_json_bytes(output)
        except (TypeError, ValueError, OverflowError, UnicodeEncodeError, RecursionError) as error:
            raise AdversarialConfigurationError(
                f"adversarial case cannot be snapshotted as canonical JSON: {error}"
            ) from error
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "tag", tag)
        object.__setattr__(self, "constraint_index", constraint_index)
        object.__setattr__(self, "predicate_result", predicate_result)
        object.__setattr__(self, "pair_id", pair_id)
        object.__setattr__(self, "pair_role", pair_role)
        object.__setattr__(self, "_inputs_json", inputs_json)
        object.__setattr__(self, "_output_json", output_json)

    @property
    def inputs(self) -> dict[str, JsonValue]:
        value = json.loads(self._inputs_json)
        if not isinstance(value, dict):
            raise AssertionError("canonical adversarial inputs are not an object")
        return cast(dict[str, JsonValue], value)

    @property
    def output(self) -> JsonValue:
        return cast(JsonValue, json.loads(self._output_json))


@dataclass(frozen=True, slots=True)
class CounterfactualPair:
    """Stable linkage and explanation for one anchor/twin pair."""

    pair_id: str
    anchor_case_id: str
    twin_case_id: str
    source_case_index: int
    changed_path: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or _PAIR_ID.fullmatch(self.pair_id) is None:
            raise AdversarialConfigurationError("counterfactual pair_id is invalid")
        if (
            not isinstance(self.anchor_case_id, str)
            or _CASE_ID.fullmatch(self.anchor_case_id) is None
            or not isinstance(self.twin_case_id, str)
            or _CASE_ID.fullmatch(self.twin_case_id) is None
            or self.anchor_case_id == self.twin_case_id
        ):
            raise AdversarialConfigurationError("counterfactual case references are invalid")
        if (
            isinstance(self.source_case_index, bool)
            or not isinstance(self.source_case_index, int)
            or self.source_case_index < 0
        ):
            raise AdversarialConfigurationError("counterfactual source index is invalid")
        if not isinstance(self.changed_path, str) or not self.changed_path.startswith("/"):
            raise AdversarialConfigurationError("counterfactual changed_path is invalid")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise AdversarialConfigurationError("counterfactual reason must be nonempty")


@dataclass(frozen=True, slots=True)
class AdversarialDataset:
    """A verified, cached adversarial supplement to one base training dataset."""

    function_id: str
    base_dataset_sha256: str
    teacher: TeacherDescriptor
    config: AdversarialGenerationConfig
    cases: tuple[AdversarialCase, ...]
    pairs: tuple[CounterfactualPair, ...]
    cache_key_sha256: str
    payload_sha256: str
    dataset_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.function_id, str)
            or _FUNCTION_ID.fullmatch(self.function_id) is None
        ):
            raise AdversarialConfigurationError("adversarial function_id is invalid")
        for name in (
            "base_dataset_sha256",
            "cache_key_sha256",
            "payload_sha256",
            "dataset_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise AdversarialConfigurationError(f"adversarial {name} is invalid")
        if not isinstance(self.teacher, TeacherDescriptor):
            raise AdversarialConfigurationError("adversarial teacher identity is invalid")
        if not isinstance(self.config, AdversarialGenerationConfig):
            raise AdversarialConfigurationError("adversarial config is invalid")
        if not isinstance(self.cases, tuple) or len(self.cases) > MAXIMUM_ADVERSARIAL_CASE_COUNT:
            raise AdversarialConfigurationError("adversarial cases are invalid or exceed the cap")
        if not isinstance(self.pairs, tuple):
            raise AdversarialConfigurationError("counterfactual pairs must be a tuple")
        if any(not isinstance(case, AdversarialCase) for case in self.cases):
            raise AdversarialConfigurationError("adversarial cases contain invalid values")
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise AdversarialConfigurationError(
                "adversarial cases contain invalid or duplicate IDs"
            )
        if any(not isinstance(pair, CounterfactualPair) for pair in self.pairs):
            raise AdversarialConfigurationError("counterfactual pairs contain invalid values")
        pair_ids = [pair.pair_id for pair in self.pairs]
        if len(set(pair_ids)) != len(pair_ids):
            raise AdversarialConfigurationError(
                "counterfactual pairs contain invalid or duplicate IDs"
            )
        by_id = {case.case_id: case for case in self.cases}
        linked_case_ids: set[str] = set()
        for pair in self.pairs:
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
                raise AdversarialConfigurationError(
                    "counterfactual pair has missing or inconsistent case links"
                )
            if pair.anchor_case_id in linked_case_ids or pair.twin_case_id in linked_case_ids:
                raise AdversarialConfigurationError(
                    "a counterfactual case cannot belong to more than one pair"
                )
            linked_case_ids.update((pair.anchor_case_id, pair.twin_case_id))
        counterfactual_case_ids = {
            case.case_id for case in self.cases if case.tag == "counterfactual"
        }
        if linked_case_ids != counterfactual_case_ids:
            raise AdversarialConfigurationError(
                "counterfactual cases and pair links must form an exact bijection"
            )

    @property
    def boundary_count(self) -> int:
        return sum(case.tag == "constraint-boundary" for case in self.cases)

    @property
    def counterfactual_case_count(self) -> int:
        return len(self.cases) - self.boundary_count


@dataclass(frozen=True, slots=True)
class _AdversarialRequest:
    ir: NeuralFunctionIr
    base: TrainingDataset
    descriptor: TeacherDescriptor
    config: AdversarialGenerationConfig
    constraints: CompiledConstraints
    constraint_budget: ConstraintEvaluationBudget
    selected_indices: tuple[int, ...]
    # Every synthetic index in deterministic ranking order; the first
    # ``len(selected_indices)`` are the preferred anchors and the rest are spares
    # taken, in order, for anchors whose counterfactual the teacher cannot produce.
    candidate_indices: tuple[int, ...]
    expected_case_count: int
    cache_key_sha256: str


class AdversarialDatasetGenerator:
    """Generate and cache a locally verified adversarial sidecar."""

    __slots__ = ("_cache_directory", "_config", "_teacher")

    def __init__(
        self,
        teacher: object,
        cache_directory: str | os.PathLike[str],
        *,
        config: AdversarialGenerationConfig | None = None,
    ) -> None:
        if config is None:
            config = AdversarialGenerationConfig()
        if not isinstance(config, AdversarialGenerationConfig):
            raise AdversarialConfigurationError("config must be an AdversarialGenerationConfig")
        descriptor = getattr(teacher, "descriptor", None)
        if not isinstance(descriptor, TeacherDescriptor):
            raise AdversarialConfigurationError(
                "adversarial teacher must expose a TeacherDescriptor"
            )
        try:
            self._cache_directory = Path(cache_directory)
        except TypeError as error:
            raise AdversarialConfigurationError("cache_directory must be path-like") from error
        self._teacher = teacher
        self._config = config

    @property
    def descriptor(self) -> TeacherDescriptor:
        descriptor = self._teacher.descriptor
        if not isinstance(descriptor, TeacherDescriptor):
            raise AdversarialConfigurationError("teacher descriptor changed type")
        return descriptor

    @property
    def config(self) -> AdversarialGenerationConfig:
        return self._config

    def cache_path(self, ir: NeuralFunctionIr, base: TrainingDataset, /) -> Path:
        request = self._prepare_request(ir, base)
        return self._path_for_request(request)

    def generate(self, ir: NeuralFunctionIr, base: TrainingDataset, /) -> AdversarialDataset:
        request = self._prepare_request(ir, base)
        path = self._path_for_request(request)
        self._ensure_cache_directory(path.parent)
        lock_path = path.with_suffix(".lock")
        try:
            with _exclusive_lock(lock_path):
                if _cache_entry_exists(path):
                    return _load_adversarial_dataset(path, request)
                cases, pairs = self._generate_cases(request)
                document, payload_sha256 = _build_document(request, cases, pairs)
                encoded = _canonical_document_bytes(document)
                if len(encoded) > MAXIMUM_DATASET_CACHE_BYTES:
                    raise AdversarialCacheError(
                        f"adversarial cache entry exceeds maximum byte length "
                        f"{MAXIMUM_DATASET_CACHE_BYTES}"
                    )
                self._publish(path, encoded, request)
                loaded = _load_adversarial_dataset(path, request)
                if not hmac.compare_digest(loaded.payload_sha256, payload_sha256):
                    raise AdversarialCacheError(
                        "published adversarial payload digest changed unexpectedly"
                    )
                return loaded
        except DatasetCacheError as error:
            raise AdversarialCacheError(
                f"adversarial cache lock or entry is unsafe: {error}"
            ) from error

    def _prepare_request(
        self,
        ir: NeuralFunctionIr,
        base: TrainingDataset,
    ) -> _AdversarialRequest:
        if not isinstance(ir, dict):
            raise AdversarialConfigurationError("IR must be an object")
        if not isinstance(base, TrainingDataset):
            raise AdversarialConfigurationError("base must be a TrainingDataset")
        try:
            snapshot = deepcopy(ir)
            build_case_schema(snapshot)
            constraints = compile_constraints(snapshot)
        except (
            TypeError,
            ValueError,
            RecursionError,
            TeacherConfigurationError,
            ConstraintConfigurationError,
        ) as error:
            raise AdversarialConfigurationError(
                f"IR adversarial contract is invalid: {error}"
            ) from error
        if (
            snapshot.get("id") != base.function_id
            or snapshot.get("semanticSha256") != base.semantic_sha256
        ):
            raise AdversarialConfigurationError(
                "base dataset function identity does not match the IR"
            )
        synthetic_indices = tuple(
            index for index, case in enumerate(base.cases) if case.origin == "synthetic"
        )
        selected_count = _ratio_count(
            len(synthetic_indices),
            float(self._config.counterfactual_ratio),
        )
        candidate_indices = _rank_indices(base, synthetic_indices)
        selected_indices = candidate_indices[:selected_count]
        expected_case_count = len(constraints) * 2 + len(selected_indices) * 2
        if (
            expected_case_count > MAXIMUM_ADVERSARIAL_CASE_COUNT
            or len(base.cases) + expected_case_count > MAXIMUM_COMBINED_CASE_COUNT
        ):
            raise AdversarialConfigurationError(
                "base plus adversarial cases exceed the configured bounded row count"
            )
        try:
            constraints.ensure_evaluation_budget(len(base.cases) + expected_case_count)
        except ConstraintConfigurationError as error:
            raise AdversarialConfigurationError(str(error)) from error
        constraint_budget = ConstraintEvaluationBudget()
        for index, case in enumerate(base.cases):
            generated = GeneratedCase(inputs=case.inputs, output=case.output)
            try:
                validate_case(snapshot, generated)
                constraints.validate_case(generated, budget=constraint_budget)
            except (
                TeacherConfigurationError,
                TeacherResponseError,
                ConstraintConfigurationError,
                ConstraintEvaluationError,
                ConstraintViolationError,
            ) as error:
                raise AdversarialConfigurationError(
                    f"base dataset case {index} violates the IR: {error}"
                ) from error
        descriptor = self.descriptor
        cache_key = _request_digest(
            snapshot,
            base,
            descriptor,
            self._config,
        )
        return _AdversarialRequest(
            ir=snapshot,
            base=base,
            descriptor=descriptor,
            config=self._config,
            constraints=constraints,
            constraint_budget=constraint_budget,
            selected_indices=selected_indices,
            candidate_indices=candidate_indices,
            expected_case_count=expected_case_count,
            cache_key_sha256=cache_key,
        )

    def _generate_cases(
        self,
        request: _AdversarialRequest,
    ) -> tuple[tuple[AdversarialCase, ...], tuple[CounterfactualPair, ...]]:
        if request.expected_case_count == 0:
            return (), ()
        if not isinstance(self._teacher, AdversarialTeacher):
            raise AdversarialConfigurationError(
                "this request requires a teacher with adversarial generation capability"
            )
        cases: list[AdversarialCase] = []
        pairs: list[CounterfactualPair] = []
        approximate_bytes = 0
        # Every lifecycle input keeps one label: a proposal whose inputs repeat a base
        # or earlier adversarial input under a different label is rejected and retried,
        # because the training corpus refuses conflicting labels.
        known_labels: dict[bytes, JsonValue] = {
            _canonical_json_bytes(case.inputs): case.output for case in request.base.cases
        }

        for constraint_index, constraint in enumerate(request.constraints):
            if not predicate_depends_on_inputs(constraint):
                result = request.constraints.evaluate(
                    constraint_index,
                    {},
                    budget=request.constraint_budget,
                )
                raise UnsynthesizableConstraintError(
                    f"constraint {constraint_index} is constant {str(result).lower()} and has "
                    "no opposite predicate side"
                )
            proposal = self._boundary_proposal(request, constraint_index, known_labels)
            for predicate_result, generated in (
                (False, proposal.predicate_false),
                (True, proposal.predicate_true),
            ):
                ordinal = len(cases)
                document = {
                    "tag": "constraint-boundary",
                    "inputs": generated.inputs,
                    "output": generated.output,
                    "constraintIndex": constraint_index,
                    "predicateResult": predicate_result,
                    "pairId": None,
                    "pairRole": None,
                }
                case_id = _case_id(request.cache_key_sha256, ordinal, document)
                approximate_bytes = _bounded_growth(
                    approximate_bytes,
                    {"caseId": case_id, **document},
                )
                cases.append(
                    AdversarialCase(
                        case_id=case_id,
                        inputs=generated.inputs,
                        output=generated.output,
                        tag="constraint-boundary",
                        constraint_index=constraint_index,
                        predicate_result=predicate_result,
                    )
                )

        wanted = len(request.selected_indices)
        skipped = 0
        last_skip: AdversarialGenerationError | None = None
        for source_index in request.candidate_indices:
            pair_ordinal = len(pairs)
            if pair_ordinal >= wanted:
                break
            source = request.base.cases[source_index]
            anchor = GeneratedCase(inputs=source.inputs, output=source.output)
            try:
                proposal, changed_path = self._counterfactual_proposal(
                    request,
                    source_index,
                    anchor,
                    known_labels,
                )
            except AdversarialGenerationError as error:
                # An anchor with no single-field twin the teacher can find (or that
                # it keeps getting wrong) is not worth the build: take the next
                # ranked spare instead, up to as many skips as pairs wanted.
                skipped += 1
                last_skip = error
                _log.warning(
                    "counterfactual anchor %d skipped after %d attempt(s); trying the next candidate",
                    source_index,
                    request.config.maximum_attempts,
                )
                if skipped > wanted:
                    raise
                continue
            pair_id = _pair_id(request.cache_key_sha256, pair_ordinal, source_index)
            anchor_ordinal = len(cases)
            anchor_document = {
                "tag": "counterfactual",
                "inputs": anchor.inputs,
                "output": anchor.output,
                "constraintIndex": None,
                "predicateResult": None,
                "pairId": pair_id,
                "pairRole": "anchor",
            }
            anchor_id = _case_id(
                request.cache_key_sha256,
                anchor_ordinal,
                anchor_document,
            )
            twin_ordinal = anchor_ordinal + 1
            twin_document = {
                "tag": "counterfactual",
                "inputs": proposal.twin.inputs,
                "output": proposal.twin.output,
                "constraintIndex": None,
                "predicateResult": None,
                "pairId": pair_id,
                "pairRole": "twin",
            }
            twin_id = _case_id(request.cache_key_sha256, twin_ordinal, twin_document)
            pair = CounterfactualPair(
                pair_id=pair_id,
                anchor_case_id=anchor_id,
                twin_case_id=twin_id,
                source_case_index=source_index,
                changed_path=changed_path,
                reason=proposal.reason.strip(),
            )
            approximate_bytes = _bounded_growth(
                approximate_bytes,
                {"caseId": anchor_id, **anchor_document},
                {"caseId": twin_id, **twin_document},
                _pair_document(pair),
            )
            cases.extend(
                (
                    AdversarialCase(
                        case_id=anchor_id,
                        inputs=anchor.inputs,
                        output=anchor.output,
                        tag="counterfactual",
                        pair_id=pair_id,
                        pair_role="anchor",
                    ),
                    AdversarialCase(
                        case_id=twin_id,
                        inputs=proposal.twin.inputs,
                        output=proposal.twin.output,
                        tag="counterfactual",
                        pair_id=pair_id,
                        pair_role="twin",
                    ),
                )
            )
            pairs.append(pair)
        if len(pairs) < wanted:
            raise AdversarialGenerationError(
                f"only {len(pairs)} of {wanted} counterfactual pairs could be generated: {last_skip}"
            ) from last_skip
        return tuple(cases), tuple(pairs)

    def _boundary_proposal(
        self,
        request: _AdversarialRequest,
        constraint_index: int,
        known_labels: dict[bytes, JsonValue],
    ) -> BoundaryPairProposal:
        last_error: Exception | None = None
        for _ in range(request.config.maximum_attempts):
            teacher_ir = deepcopy(request.ir)
            try:
                proposal = cast(AdversarialTeacher, self._teacher).generate_boundary_pair(
                    teacher_ir,
                    constraint_index,
                )
                _assert_ir_unchanged(request.ir, teacher_ir)
                _validate_boundary_proposal(
                    request.ir,
                    request.constraints,
                    constraint_index,
                    proposal,
                    request.constraint_budget,
                )
                _reject_label_conflicts(
                    known_labels,
                    (proposal.predicate_false, proposal.predicate_true),
                    context=f"constraint {constraint_index} boundary pair",
                )
                return proposal
            except TeacherTransportError:
                raise
            except TeacherConfigurationError:
                raise
            except (
                TeacherResponseError,
                ConstraintConfigurationError,
                ConstraintEvaluationError,
                ConstraintViolationError,
                AdversarialGenerationError,
            ) as error:
                last_error = error
        raise UnsynthesizableConstraintError(
            f"constraint {constraint_index} did not yield a valid two-sided boundary pair "
            f"within {request.config.maximum_attempts} attempts: {last_error}"
        ) from last_error

    def _counterfactual_proposal(
        self,
        request: _AdversarialRequest,
        source_index: int,
        anchor: GeneratedCase,
        known_labels: dict[bytes, JsonValue],
    ) -> tuple[CounterfactualProposal, str]:
        last_error: Exception | None = None
        for _ in range(request.config.maximum_attempts):
            teacher_ir = deepcopy(request.ir)
            teacher_anchor = GeneratedCase(
                inputs=deepcopy(anchor.inputs),
                output=deepcopy(anchor.output),
            )
            try:
                proposal = cast(AdversarialTeacher, self._teacher).generate_counterfactual(
                    teacher_ir,
                    teacher_anchor,
                )
                _assert_ir_unchanged(request.ir, teacher_ir)
                if _canonical_json_bytes(
                    {"inputs": teacher_anchor.inputs, "output": teacher_anchor.output}
                ) != _canonical_json_bytes({"inputs": anchor.inputs, "output": anchor.output}):
                    raise TeacherResponseError("teacher mutated the counterfactual anchor")
                changed_path = _validate_counterfactual(
                    request.ir,
                    request.constraints,
                    anchor,
                    proposal,
                    request.constraint_budget,
                )
                _reject_label_conflicts(
                    known_labels,
                    (proposal.twin,),
                    context=f"base synthetic case {source_index} counterfactual twin",
                )
                return proposal, changed_path
            except TeacherTransportError:
                raise
            except TeacherConfigurationError:
                raise
            except (
                TeacherResponseError,
                ConstraintConfigurationError,
                ConstraintEvaluationError,
                ConstraintViolationError,
                AdversarialGenerationError,
            ) as error:
                last_error = error
        raise AdversarialGenerationError(
            f"base synthetic case {source_index} did not yield a valid counterfactual "
            f"within {request.config.maximum_attempts} attempts: {last_error}"
        ) from last_error

    def _path_for_request(self, request: _AdversarialRequest) -> Path:
        key = request.cache_key_sha256
        return (
            self._cache_directory
            / "adversarial-datasets"
            / f"v{ADVERSARIAL_DATASET_VERSION}"
            / key[:2]
            / f"{key}.json"
        )

    @staticmethod
    def _ensure_cache_directory(directory: Path) -> None:
        try:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as error:
            raise AdversarialCacheError(
                f"could not create adversarial cache directory: {error}"
            ) from error
        if not directory.is_dir():
            raise AdversarialCacheError("adversarial cache path is not a directory")

    @staticmethod
    def _publish(path: Path, encoded: bytes, request: _AdversarialRequest) -> None:
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{request.cache_key_sha256}.",
                suffix=".tmp",
                dir=path.parent,
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            _load_adversarial_dataset(temporary_path, request)
            os.replace(temporary_path, path)
            temporary_path = None
            _fsync_directory(path.parent)
        except AdversarialCacheError:
            raise
        except OSError as error:
            raise AdversarialCacheError(
                f"could not publish adversarial cache entry: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass


def _reject_label_conflicts(
    known_labels: dict[bytes, JsonValue],
    cases: Sequence[GeneratedCase],
    *,
    context: str,
) -> None:
    """Reject cases whose inputs already carry a different lifecycle label, then record them."""

    for case in cases:
        key = _canonical_json_bytes(case.inputs)
        existing = known_labels.get(key)
        if existing is not None and not _json_equal(existing, case.output):
            raise AdversarialGenerationError(
                f"{context} repeats an existing lifecycle input under a different label"
            )
    for case in cases:
        known_labels.setdefault(_canonical_json_bytes(case.inputs), case.output)


def _validate_boundary_proposal(
    ir: NeuralFunctionIr,
    constraints: CompiledConstraints,
    constraint_index: int,
    proposal: BoundaryPairProposal,
    constraint_budget: ConstraintEvaluationBudget,
) -> None:
    if not isinstance(proposal, BoundaryPairProposal):
        raise TeacherResponseError("teacher boundary result must be a BoundaryPairProposal")
    if constraint_index < 0 or constraint_index >= len(constraints):
        raise AdversarialConfigurationError("constraint index is out of range")
    for expected, case in (
        (False, proposal.predicate_false),
        (True, proposal.predicate_true),
    ):
        validate_case(ir, case)
        constraints.validate_case(case, budget=constraint_budget)
        actual = constraints.evaluate(
            constraint_index,
            case.inputs,
            budget=constraint_budget,
        )
        if actual is not expected:
            raise AdversarialGenerationError(
                f"constraint {constraint_index} boundary proposal did not reach predicate "
                f"{str(expected).lower()}"
            )
    _single_changed_path(
        proposal.predicate_false.inputs,
        proposal.predicate_true.inputs,
        context=f"constraint {constraint_index} boundary pair",
    )


def _validate_counterfactual(
    ir: NeuralFunctionIr,
    constraints: CompiledConstraints,
    anchor: GeneratedCase,
    proposal: CounterfactualProposal,
    constraint_budget: ConstraintEvaluationBudget,
) -> str:
    if not isinstance(proposal, CounterfactualProposal):
        raise TeacherResponseError("teacher counterfactual result must be a CounterfactualProposal")
    validate_case(ir, anchor)
    constraints.validate_case(anchor, budget=constraint_budget)
    validate_case(ir, proposal.twin)
    constraints.validate_case(proposal.twin, budget=constraint_budget)
    if _json_equal(anchor.output, proposal.twin.output):
        raise AdversarialGenerationError("counterfactual output label did not change")
    return _single_changed_path(
        anchor.inputs,
        proposal.twin.inputs,
        context="counterfactual pair",
    )


def _single_changed_path(left: JsonValue, right: JsonValue, *, context: str) -> str:
    changes: list[str] = []
    _collect_changes(left, right, "", changes)
    if len(changes) != 1:
        raise AdversarialGenerationError(
            f"{context} must change exactly one JSON path; found {len(changes)}"
        )
    return changes[0] or "/"


def _collect_changes(left: Any, right: Any, path: str, changes: list[str]) -> None:
    if len(changes) > 1 or _json_equal(left, right):
        return
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}/{_pointer_token(key)}"
            if key not in left or key not in right:
                changes.append(child_path)
            else:
                _collect_changes(left[key], right[key], child_path, changes)
            if len(changes) > 1:
                return
        return
    if isinstance(left, list) and isinstance(right, list):
        limit = max(len(left), len(right))
        for index in range(limit):
            child_path = f"{path}/{index}"
            if index >= len(left) or index >= len(right):
                changes.append(child_path)
            else:
                _collect_changes(left[index], right[index], child_path, changes)
            if len(changes) > 1:
                return
        return
    changes.append(path or "/")


def _json_equal(left: Any, right: Any) -> bool:
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        try:
            return float(left) == float(right)
        except (OverflowError, ValueError):
            return False
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(first, second) for first, second in zip(left, right, strict=True)
        )
    return bool(left == right)


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _assert_ir_unchanged(expected: NeuralFunctionIr, actual: NeuralFunctionIr) -> None:
    try:
        unchanged = _canonical_json_bytes(expected) == _canonical_json_bytes(actual)
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError, RecursionError) as error:
        raise TeacherResponseError(
            f"teacher mutated the adversarial IR into invalid JSON: {error}"
        ) from error
    if not unchanged:
        raise TeacherResponseError("teacher mutated the adversarial IR request")


def _ratio_count(total: int, ratio: float) -> int:
    # Nearest row count, with exact half rounded upward. Endpoints remain exact.
    return min(total, math.floor(total * ratio + 0.5))


def _rank_indices(base: TrainingDataset, candidates: tuple[int, ...]) -> tuple[int, ...]:
    """Every candidate index in the deterministic order anchors are taken from."""

    ranked: list[tuple[str, int]] = []
    for index in candidates:
        case = base.cases[index]
        encoded = _canonical_json_bytes(
            {"index": index, "inputs": case.inputs, "output": case.output}
        )
        ranked.append((hashlib.sha256(encoded).hexdigest(), index))
    ranked.sort()
    return tuple(index for _, index in ranked)


def _bounded_growth(current: int, *documents: Mapping[str, object]) -> int:
    try:
        updated = current + sum(len(_canonical_json_bytes(document)) for document in documents)
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError, RecursionError) as error:
        raise AdversarialGenerationError(
            f"adversarial cases cannot be serialized safely: {error}"
        ) from error
    if updated > MAXIMUM_DATASET_CACHE_BYTES:
        raise AdversarialGenerationError(
            f"adversarial cases exceed maximum byte length {MAXIMUM_DATASET_CACHE_BYTES}"
        )
    return updated


def _request_digest(
    ir: NeuralFunctionIr,
    base: TrainingDataset,
    descriptor: TeacherDescriptor,
    config: AdversarialGenerationConfig,
) -> str:
    try:
        projection: dict[str, object] = {
            "kind": "semantscript.adversarial-request",
            "version": 1,
            "ir": {
                "kind": ir["kind"],
                "irVersion": ir["irVersion"],
                "id": ir["id"],
                "semanticSha256": ir["semanticSha256"],
                "definition": ir["definition"],
                "inputs": ir["inputs"],
                "output": ir["output"],
            },
            "baseDataset": {
                "datasetSha256": base.dataset_sha256,
                "payloadSha256": base.payload_sha256,
            },
            "teacher": _descriptor_document(descriptor),
            "generation": {
                **config.document(),
                "datasetFormatVersion": ADVERSARIAL_DATASET_VERSION,
                "generatorContractVersion": 2,
                "promptContractVersion": 1,
                "pairContractVersion": 1,
                "constraintEvaluatorVersion": 1,
            },
        }
        encoded = _canonical_json_bytes(projection)
    except KeyError as error:
        raise AdversarialConfigurationError(
            f"IR is missing required adversarial field {error.args[0]!r}"
        ) from error
    except (
        TypeError,
        ValueError,
        OverflowError,
        UnicodeEncodeError,
        RecursionError,
    ) as error:
        raise AdversarialConfigurationError(
            f"adversarial request cannot be canonicalized: {error}"
        ) from error
    return hashlib.sha256(_REQUEST_DIGEST_DOMAIN + encoded).hexdigest()


def _case_id(request_sha256: str, ordinal: int, document: Mapping[str, object]) -> str:
    encoded = _canonical_json_bytes(
        {"requestSha256": request_sha256, "ordinal": ordinal, "case": document}
    )
    return "ac_" + hashlib.sha256(_CASE_ID_DOMAIN + encoded).hexdigest()


def _pair_id(request_sha256: str, ordinal: int, source_index: int) -> str:
    encoded = _canonical_json_bytes(
        {
            "requestSha256": request_sha256,
            "ordinal": ordinal,
            "sourceCaseIndex": source_index,
        }
    )
    return "cf_" + hashlib.sha256(_PAIR_ID_DOMAIN + encoded).hexdigest()


def _build_document(
    request: _AdversarialRequest,
    cases: tuple[AdversarialCase, ...],
    pairs: tuple[CounterfactualPair, ...],
) -> tuple[dict[str, object], str]:
    boundary_count = len(request.constraints) * 2
    payload: dict[str, object] = {
        "requestSha256": request.cache_key_sha256,
        "function": {
            "id": request.base.function_id,
            "semanticSha256": request.base.semantic_sha256,
        },
        "baseDataset": {
            "datasetSha256": request.base.dataset_sha256,
            "payloadSha256": request.base.payload_sha256,
        },
        "teacher": _descriptor_document(request.descriptor),
        "config": request.config.document(),
        "counts": {
            "total": len(cases),
            "constraintBoundary": boundary_count,
            "counterfactual": len(cases) - boundary_count,
            "pairs": len(pairs),
        },
        "cases": [_case_document(case) for case in cases],
        "pairs": [_pair_document(pair) for pair in pairs],
    }
    payload_sha256 = hashlib.sha256(
        _PAYLOAD_DIGEST_DOMAIN + _canonical_json_bytes(payload)
    ).hexdigest()
    return (
        {
            "kind": ADVERSARIAL_DATASET_KIND,
            "datasetVersion": ADVERSARIAL_DATASET_VERSION,
            "payloadSha256": payload_sha256,
            "payload": payload,
        },
        payload_sha256,
    )


def _case_document(case: AdversarialCase) -> dict[str, object]:
    return {
        "caseId": case.case_id,
        "tag": case.tag,
        "inputs": case.inputs,
        "output": case.output,
        "constraintIndex": case.constraint_index,
        "predicateResult": case.predicate_result,
        "pairId": case.pair_id,
        "pairRole": case.pair_role,
    }


def _case_identity_document(raw: Mapping[str, object]) -> dict[str, object]:
    return {
        "tag": raw["tag"],
        "inputs": raw["inputs"],
        "output": raw["output"],
        "constraintIndex": raw["constraintIndex"],
        "predicateResult": raw["predicateResult"],
        "pairId": raw["pairId"],
        "pairRole": raw["pairRole"],
    }


def _pair_document(pair: CounterfactualPair) -> dict[str, object]:
    return {
        "pairId": pair.pair_id,
        "anchorCaseId": pair.anchor_case_id,
        "twinCaseId": pair.twin_case_id,
        "sourceCaseIndex": pair.source_case_index,
        "changedPath": pair.changed_path,
        "reason": pair.reason,
    }


def _descriptor_document(descriptor: TeacherDescriptor) -> dict[str, str]:
    return {
        "provider": descriptor.provider,
        "model": descriptor.model,
        "configurationSha256": descriptor.configuration_sha256,
    }


def _load_adversarial_dataset(
    path: Path,
    request: _AdversarialRequest,
) -> AdversarialDataset:
    try:
        encoded = _read_bounded_regular_file(path)
        document = _loads_dataset_json(encoded)
    except DatasetCacheError as error:
        raise AdversarialCacheError(f"invalid adversarial cache entry: {error}") from error
    if not isinstance(document, dict) or set(document) != {
        "kind",
        "datasetVersion",
        "payloadSha256",
        "payload",
    }:
        _bad_cache("adversarial envelope has unknown or missing fields")
    if (
        document["kind"] != ADVERSARIAL_DATASET_KIND
        or type(document["datasetVersion"]) is not int
        or document["datasetVersion"] != ADVERSARIAL_DATASET_VERSION
    ):
        _bad_cache("adversarial dataset kind or version is unsupported")
    payload_sha256 = document["payloadSha256"]
    payload = document["payload"]
    if not isinstance(payload_sha256, str) or _SHA256.fullmatch(payload_sha256) is None:
        _bad_cache("adversarial payloadSha256 is invalid")
    if not isinstance(payload, dict):
        _bad_cache("adversarial payload must be an object")
    actual_payload_sha256 = hashlib.sha256(
        _PAYLOAD_DIGEST_DOMAIN + _canonical_json_bytes(payload)
    ).hexdigest()
    if not hmac.compare_digest(payload_sha256, actual_payload_sha256):
        _bad_cache("adversarial payload digest does not match its content")
    try:
        canonical = _canonical_document_bytes(document)
    except DatasetCacheError as error:
        raise AdversarialCacheError(f"invalid adversarial cache entry: {error}") from error
    if canonical != encoded:
        _bad_cache("adversarial cache entry is not canonical JSON")

    if set(payload) != {
        "requestSha256",
        "function",
        "baseDataset",
        "teacher",
        "config",
        "counts",
        "cases",
        "pairs",
    }:
        _bad_cache("adversarial payload has unknown or missing fields")
    request_sha256 = payload["requestSha256"]
    if not isinstance(request_sha256, str) or not hmac.compare_digest(
        request_sha256,
        request.cache_key_sha256,
    ):
        _bad_cache("adversarial request digest does not match this request")
    _validate_payload_identity(payload, request)

    raw_cases = payload["cases"]
    raw_pairs = payload["pairs"]
    if not isinstance(raw_cases, list) or len(raw_cases) != request.expected_case_count:
        _bad_cache("adversarial cases do not match the expected count")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != len(request.selected_indices):
        _bad_cache("counterfactual pairs do not match the configured ratio")

    cases = tuple(
        _parse_cached_case(request, raw, ordinal) for ordinal, raw in enumerate(raw_cases)
    )
    pairs = tuple(
        _parse_cached_pair(request, raw, ordinal) for ordinal, raw in enumerate(raw_pairs)
    )
    _validate_cached_semantics(request, cases, pairs)

    try:
        return AdversarialDataset(
            function_id=request.base.function_id,
            base_dataset_sha256=request.base.dataset_sha256,
            teacher=request.descriptor,
            config=request.config,
            cases=cases,
            pairs=pairs,
            cache_key_sha256=request.cache_key_sha256,
            payload_sha256=payload_sha256,
            dataset_sha256=hashlib.sha256(encoded).hexdigest(),
        )
    except AdversarialConfigurationError as error:
        raise AdversarialCacheError(f"adversarial dataset invariants failed: {error}") from error


def _validate_payload_identity(
    payload: Mapping[str, object],
    request: _AdversarialRequest,
) -> None:
    function = payload["function"]
    if not isinstance(function, dict) or set(function) != {"id", "semanticSha256"}:
        _bad_cache("adversarial function identity is invalid")
    if (
        function["id"] != request.base.function_id
        or function["semanticSha256"] != request.base.semantic_sha256
    ):
        _bad_cache("adversarial function identity does not match the IR")
    base = payload["baseDataset"]
    if not isinstance(base, dict) or set(base) != {"datasetSha256", "payloadSha256"}:
        _bad_cache("adversarial base dataset identity is invalid")
    if (
        base["datasetSha256"] != request.base.dataset_sha256
        or base["payloadSha256"] != request.base.payload_sha256
    ):
        _bad_cache("adversarial base dataset identity does not match")
    teacher = payload["teacher"]
    if (
        not isinstance(teacher, dict)
        or set(teacher) != {"provider", "model", "configurationSha256"}
        or any(not isinstance(value, str) for value in teacher.values())
        or teacher != _descriptor_document(request.descriptor)
    ):
        _bad_cache("adversarial teacher identity does not match")
    config = payload["config"]
    if (
        not isinstance(config, dict)
        or set(config) != {"counterfactualRatio", "maximumAttempts"}
        or type(config["counterfactualRatio"]) is not float
        or not math.isfinite(config["counterfactualRatio"])
        or type(config["maximumAttempts"]) is not int
        or config != request.config.document()
    ):
        _bad_cache("adversarial config does not match")
    counts = payload["counts"]
    boundary_count = len(request.constraints) * 2
    expected_counts = {
        "total": request.expected_case_count,
        "constraintBoundary": boundary_count,
        "counterfactual": request.expected_case_count - boundary_count,
        "pairs": len(request.selected_indices),
    }
    if (
        not isinstance(counts, dict)
        or set(counts) != set(expected_counts)
        or any(type(value) is not int for value in counts.values())
        or counts != expected_counts
    ):
        _bad_cache("adversarial counts do not match the request")


def _parse_cached_case(
    request: _AdversarialRequest,
    raw: object,
    ordinal: int,
) -> AdversarialCase:
    expected_keys = {
        "caseId",
        "tag",
        "inputs",
        "output",
        "constraintIndex",
        "predicateResult",
        "pairId",
        "pairRole",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        _bad_cache(f"adversarial case {ordinal} has unknown or missing fields")
    case_id = raw["caseId"]
    expected_id = _case_id(
        request.cache_key_sha256,
        ordinal,
        _case_identity_document(raw),
    )
    if not isinstance(case_id, str) or not hmac.compare_digest(case_id, expected_id):
        _bad_cache(f"adversarial case {ordinal} ID is invalid")
    inputs = raw["inputs"]
    if not isinstance(inputs, dict):
        _bad_cache(f"adversarial case {ordinal} inputs must be an object")
    try:
        case = AdversarialCase(
            case_id=case_id,
            inputs=inputs,
            output=cast(JsonValue, raw["output"]),
            tag=cast(AdversarialTag, raw["tag"]),
            constraint_index=cast(int | None, raw["constraintIndex"]),
            predicate_result=cast(bool | None, raw["predicateResult"]),
            pair_id=cast(str | None, raw["pairId"]),
            pair_role=cast(PairRole | None, raw["pairRole"]),
        )
        generated = GeneratedCase(inputs=case.inputs, output=case.output)
        validate_case(request.ir, generated)
        request.constraints.validate_case(generated, budget=request.constraint_budget)
    except (
        AdversarialConfigurationError,
        TeacherConfigurationError,
        TeacherResponseError,
        ConstraintConfigurationError,
        ConstraintEvaluationError,
        ConstraintViolationError,
    ) as error:
        raise AdversarialCacheError(f"adversarial case {ordinal} is invalid: {error}") from error
    return case


def _parse_cached_pair(
    request: _AdversarialRequest,
    raw: object,
    ordinal: int,
) -> CounterfactualPair:
    if not isinstance(raw, dict) or set(raw) != {
        "pairId",
        "anchorCaseId",
        "twinCaseId",
        "sourceCaseIndex",
        "changedPath",
        "reason",
    }:
        _bad_cache(f"counterfactual pair {ordinal} has unknown or missing fields")
    source_index = raw["sourceCaseIndex"]
    if type(source_index) is not int:
        _bad_cache(f"counterfactual pair {ordinal} source index is invalid")
    expected_pair_id = _pair_id(request.cache_key_sha256, ordinal, source_index)
    if raw["pairId"] != expected_pair_id:
        _bad_cache(f"counterfactual pair {ordinal} ID is invalid")
    try:
        return CounterfactualPair(
            pair_id=cast(str, raw["pairId"]),
            anchor_case_id=cast(str, raw["anchorCaseId"]),
            twin_case_id=cast(str, raw["twinCaseId"]),
            source_case_index=source_index,
            changed_path=cast(str, raw["changedPath"]),
            reason=cast(str, raw["reason"]),
        )
    except AdversarialConfigurationError as error:
        raise AdversarialCacheError(f"counterfactual pair {ordinal} is invalid: {error}") from error


def _validate_cached_semantics(
    request: _AdversarialRequest,
    cases: tuple[AdversarialCase, ...],
    pairs: tuple[CounterfactualPair, ...],
) -> None:
    boundary_count = len(request.constraints) * 2
    for ordinal in range(boundary_count):
        case = cases[ordinal]
        constraint_index = ordinal // 2
        predicate_result = bool(ordinal % 2)
        if (
            case.tag != "constraint-boundary"
            or case.constraint_index != constraint_index
            or case.predicate_result is not predicate_result
        ):
            _bad_cache(f"boundary case {ordinal} metadata is out of order")
        actual = request.constraints.evaluate(
            constraint_index,
            case.inputs,
            budget=request.constraint_budget,
        )
        if actual is not predicate_result:
            _bad_cache(f"boundary case {ordinal} is on the wrong predicate side")
        if ordinal % 2 == 1:
            previous = cases[ordinal - 1]
            try:
                _single_changed_path(
                    previous.inputs,
                    case.inputs,
                    context=f"constraint {constraint_index} boundary pair",
                )
            except AdversarialGenerationError as error:
                raise AdversarialCacheError(str(error)) from error

    by_id = {case.case_id: case for case in cases}
    sources = tuple(pair.source_case_index for pair in pairs)
    rank = {index: position for position, index in enumerate(request.candidate_indices)}
    if (
        len(sources) != len(request.selected_indices)
        or len(set(sources)) != len(sources)
        or any(index not in rank for index in sources)
        or list(sources) != sorted(sources, key=lambda index: rank[index])
    ):
        _bad_cache("counterfactual source selection does not follow the ranked candidates")
    for pair_ordinal, pair in enumerate(pairs):
        anchor = by_id.get(pair.anchor_case_id)
        twin = by_id.get(pair.twin_case_id)
        expected_anchor_ordinal = boundary_count + pair_ordinal * 2
        expected_twin_ordinal = expected_anchor_ordinal + 1
        if (
            anchor is None
            or twin is None
            or cases[expected_anchor_ordinal] != anchor
            or cases[expected_twin_ordinal] != twin
            or anchor.tag != "counterfactual"
            or twin.tag != "counterfactual"
            or anchor.pair_id != pair.pair_id
            or twin.pair_id != pair.pair_id
            or anchor.pair_role != "anchor"
            or twin.pair_role != "twin"
        ):
            _bad_cache(f"counterfactual pair {pair_ordinal} links are inconsistent")
        source = request.base.cases[pair.source_case_index]
        if not _json_equal(anchor.inputs, source.inputs) or not _json_equal(
            anchor.output,
            source.output,
        ):
            _bad_cache(f"counterfactual pair {pair_ordinal} anchor changed from its base row")
        try:
            changed_path = _validate_counterfactual(
                request.ir,
                request.constraints,
                GeneratedCase(inputs=anchor.inputs, output=anchor.output),
                CounterfactualProposal(
                    twin=GeneratedCase(inputs=twin.inputs, output=twin.output),
                    reason=pair.reason,
                ),
                request.constraint_budget,
            )
        except (
            TeacherResponseError,
            ConstraintConfigurationError,
            ConstraintEvaluationError,
            ConstraintViolationError,
            AdversarialGenerationError,
        ) as error:
            raise AdversarialCacheError(
                f"counterfactual pair {pair_ordinal} is invalid: {error}"
            ) from error
        if changed_path != pair.changed_path:
            _bad_cache(f"counterfactual pair {pair_ordinal} changed path is incorrect")


def _bad_cache(message: str) -> None:
    raise AdversarialCacheError(message)


__all__ = [
    "ADVERSARIAL_DATASET_KIND",
    "ADVERSARIAL_DATASET_VERSION",
    "MAXIMUM_ADVERSARIAL_CASE_COUNT",
    "MAXIMUM_COMBINED_CASE_COUNT",
    "AdversarialCacheError",
    "AdversarialCase",
    "AdversarialConfigurationError",
    "AdversarialDataset",
    "AdversarialDatasetError",
    "AdversarialDatasetGenerator",
    "AdversarialGenerationConfig",
    "AdversarialGenerationError",
    "AdversarialTag",
    "BoundarySide",
    "CounterfactualPair",
    "PairRole",
    "UnsynthesizableConstraintError",
]
