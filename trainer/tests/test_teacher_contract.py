from dataclasses import FrozenInstanceError

import pytest

from semantscript_trainer.case_contract import MAXIMUM_CASE_COUNT
from semantscript_trainer.teacher import (
    CaseGenerator,
    GeneratedCase,
    Teacher,
    TeacherBatchError,
    TeacherBatchTimeout,
    TeacherConfigurationError,
    TeacherDescriptor,
    TeacherError,
    TeacherResponseError,
    TeacherTransportError,
)


class FakeTeacher:
    def __init__(self, cases: tuple[GeneratedCase, ...]) -> None:
        self._cases = cases
        self.calls: list[tuple[dict, int]] = []
        self._descriptor = TeacherDescriptor(
            provider="fake",
            model="fake-v1",
            configuration_sha256="a" * 64,
        )

    @property
    def descriptor(self) -> TeacherDescriptor:
        return self._descriptor

    def generate(self, ir: dict, n: int, /) -> tuple[GeneratedCase, ...]:
        self.calls.append((ir, n))
        return self._cases


def test_runtime_protocol_and_descriptor_are_backend_neutral() -> None:
    teacher = FakeTeacher(())

    assert isinstance(teacher, Teacher)
    assert CaseGenerator(teacher).descriptor == teacher.descriptor
    assert not hasattr(teacher.descriptor, "__dict__")
    with pytest.raises(FrozenInstanceError):
        teacher.descriptor.model = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "arguments",
    [
        {"provider": "", "model": "m", "configuration_sha256": "a" * 64},
        {"provider": "p", "model": "", "configuration_sha256": "a" * 64},
        {"provider": "p", "model": "m", "configuration_sha256": "A" * 64},
        {"provider": "p", "model": "m", "configuration_sha256": "a" * 63},
    ],
)
def test_descriptor_rejects_invalid_identity(arguments: dict[str, str]) -> None:
    with pytest.raises(TeacherConfigurationError):
        TeacherDescriptor(**arguments)


def test_case_generator_revalidates_exact_count_and_contract() -> None:
    ir = simple_ir()
    valid = GeneratedCase(inputs={"value": "x"}, output="yes")
    teacher = FakeTeacher((valid,))

    assert CaseGenerator(teacher).generate(ir, 1) == (valid,)
    assert teacher.calls == [(ir, 1)]

    with pytest.raises(TeacherResponseError, match="expected 2"):
        CaseGenerator(teacher).generate(ir, 2)
    invalid = FakeTeacher((GeneratedCase(inputs={"wrong": "x"}, output="yes"),))
    with pytest.raises(TeacherResponseError, match="input names"):
        CaseGenerator(invalid).generate(ir, 1)


def test_zero_count_does_not_call_teacher() -> None:
    teacher = FakeTeacher(())

    assert CaseGenerator(teacher).generate(simple_ir(), 0) == ()
    assert teacher.calls == []


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_case_generator_rejects_invalid_counts_before_transport(count) -> None:
    teacher = FakeTeacher(())

    with pytest.raises(TeacherConfigurationError, match="case count"):
        CaseGenerator(teacher).generate(simple_ir(), count)
    assert teacher.calls == []


def test_case_generator_enforces_count_cap_before_transport() -> None:
    teacher = FakeTeacher(())

    with pytest.raises(TeacherConfigurationError, match="maximum"):
        CaseGenerator(teacher).generate(simple_ir(), MAXIMUM_CASE_COUNT + 1)
    assert teacher.calls == []


def test_case_generator_rejects_non_teacher() -> None:
    with pytest.raises(TeacherConfigurationError, match="Teacher protocol"):
        CaseGenerator(object())  # type: ignore[arg-type]


def test_typed_errors_share_the_teacher_base() -> None:
    assert issubclass(TeacherConfigurationError, TeacherError)
    assert issubclass(TeacherTransportError, TeacherError)
    assert issubclass(TeacherResponseError, TeacherError)
    assert issubclass(TeacherBatchError, TeacherError)
    assert issubclass(TeacherBatchTimeout, TeacherBatchError)


def simple_ir() -> dict:
    return {
        "inputs": [{"name": "value", "index": 0, "tsType": "string", "type": {"kind": "string"}}],
        "output": {
            "kind": "scalar",
            "tsType": '"yes" | "no"',
            "head": {
                "kind": "nominal",
                "sourceKind": "string-union",
                "support": ["yes", "no"],
            },
        },
    }
