"""Backend-independent teacher contracts and case-generation orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type NeuralFunctionIr = dict[str, JsonValue]

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
MAXIMUM_TEACHER_RESPONSE_BYTES = 64 * 1024 * 1024
MAXIMUM_COUNTERFACTUAL_REASON_BYTES = 16 * 1024


class TeacherError(RuntimeError):
    """Base class for failures at the teacher boundary."""


class TeacherConfigurationError(TeacherError, ValueError):
    """The teacher or requested generation is not configured correctly."""


class TeacherTransportError(TeacherError):
    """A teacher request could not be transported successfully."""


class TeacherResponseError(TeacherError):
    """A teacher response is malformed or violates the case contract."""


class TeacherBatchError(TeacherError):
    """A submitted teacher batch is incomplete or failed."""


class TeacherBatchTimeout(TeacherBatchError):
    """A teacher batch did not complete before its configured deadline."""


@dataclass(frozen=True, slots=True)
class GeneratedCase:
    """One schema-validated synthetic input/output example."""

    inputs: dict[str, JsonValue]
    output: JsonValue


@dataclass(frozen=True, slots=True)
class BoundaryPairProposal:
    """Teacher-proposed cases on the false and true sides of one predicate."""

    predicate_false: GeneratedCase
    predicate_true: GeneratedCase

    def __post_init__(self) -> None:
        if not isinstance(self.predicate_false, GeneratedCase) or not isinstance(
            self.predicate_true, GeneratedCase
        ):
            raise TeacherResponseError("boundary proposals must contain two GeneratedCase values")


@dataclass(frozen=True, slots=True)
class CounterfactualProposal:
    """A teacher-proposed minimally edited twin and its explanation."""

    twin: GeneratedCase
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.twin, GeneratedCase):
            raise TeacherResponseError("counterfactual twin must be a GeneratedCase")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise TeacherResponseError("counterfactual reason must be a nonempty string")
        try:
            encoded = self.reason.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise TeacherResponseError(
                "counterfactual reason must contain valid Unicode scalar values"
            ) from error
        if len(encoded) > MAXIMUM_COUNTERFACTUAL_REASON_BYTES:
            raise TeacherResponseError(
                "counterfactual reason exceeds maximum byte length "
                f"{MAXIMUM_COUNTERFACTUAL_REASON_BYTES}"
            )


@dataclass(frozen=True, slots=True)
class TeacherDescriptor:
    """Stable, secret-free identity used in provenance and cache keys."""

    provider: str
    model: str
    configuration_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider:
            raise TeacherConfigurationError("teacher provider must be nonempty")
        if not isinstance(self.model, str) or not self.model:
            raise TeacherConfigurationError("teacher model must be nonempty")
        if (
            not isinstance(self.configuration_sha256, str)
            or _SHA256.fullmatch(self.configuration_sha256) is None
        ):
            raise TeacherConfigurationError(
                "teacher configuration_sha256 must be 64 lowercase hexadecimal characters"
            )


@runtime_checkable
class Teacher(Protocol):
    """A pluggable source of labeled cases for one neural-function IR record."""

    @property
    def descriptor(self) -> TeacherDescriptor: ...

    def generate(self, ir: NeuralFunctionIr, n: int, /) -> tuple[GeneratedCase, ...]: ...


@runtime_checkable
class AdversarialTeacher(Protocol):
    """Optional capability for boundary and counterfactual generation."""

    @property
    def descriptor(self) -> TeacherDescriptor: ...

    def generate_boundary_pair(
        self,
        ir: NeuralFunctionIr,
        constraint_index: int,
        /,
    ) -> BoundaryPairProposal: ...

    def generate_counterfactual(
        self,
        ir: NeuralFunctionIr,
        anchor: GeneratedCase,
        /,
    ) -> CounterfactualProposal: ...


class CaseGenerator:
    """Backend-neutral generation boundary that distrusts teacher responses."""

    __slots__ = ("_teacher",)

    def __init__(self, teacher: Teacher) -> None:
        if not isinstance(teacher, Teacher):
            raise TeacherConfigurationError("teacher does not implement the Teacher protocol")
        self._teacher = teacher

    @property
    def descriptor(self) -> TeacherDescriptor:
        return self._teacher.descriptor

    def generate(self, ir: NeuralFunctionIr, n: int, /) -> tuple[GeneratedCase, ...]:
        # Local import keeps the data contract usable without a module import cycle.
        from .case_contract import validate_case_count, validate_generated_cases

        expected_count = validate_case_count(n)
        if expected_count == 0:
            return ()

        return validate_generated_cases(
            ir,
            self._teacher.generate(ir, expected_count),
            expected_count=expected_count,
        )


__all__ = [
    "MAXIMUM_COUNTERFACTUAL_REASON_BYTES",
    "MAXIMUM_TEACHER_RESPONSE_BYTES",
    "AdversarialTeacher",
    "BoundaryPairProposal",
    "CaseGenerator",
    "CounterfactualProposal",
    "GeneratedCase",
    "JsonValue",
    "NeuralFunctionIr",
    "Teacher",
    "TeacherBatchError",
    "TeacherBatchTimeout",
    "TeacherConfigurationError",
    "TeacherDescriptor",
    "TeacherError",
    "TeacherResponseError",
    "TeacherTransportError",
]
