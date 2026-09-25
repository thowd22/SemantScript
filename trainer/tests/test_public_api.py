from __future__ import annotations

import pytest

import semantscript_trainer
import semantscript_trainer.artifact as artifact_module
import semantscript_trainer.canonical_input as canonical_input_module
import semantscript_trainer.lifecycle as lifecycle_module
import semantscript_trainer.training as training_module
import semantscript_trainer.training_contract as training_contract_module
import semantscript_trainer.verification as verification_module
from semantscript_trainer import (
    AdversarialDataset,
    AdversarialDatasetGenerator,
    AdversarialGenerationConfig,
    AdversarialTeacher,
    BoundaryPairProposal,
    CaseGenerator,
    ConstraintsTeacherConfig,
    DatasetCacheError,
    DatasetCase,
    DatasetConfigurationError,
    GeneratedCase,
    StrictJsonError,
    StrictJsonLimits,
    SyntheticDatasetGenerator,
    Teacher,
    TeacherConfig,
    TeacherConfigurationError,
    TeacherDescriptor,
    TeacherResponseError,
    TeacherTransportError,
    TrainingDataset,
    create_teacher,
    load_teacher_config,
    loads_strict_json,
    validate_case_constraints,
)
from semantscript_trainer.adversarial import (
    AdversarialDataset as DefinedAdversarialDataset,
)
from semantscript_trainer.adversarial import (
    AdversarialDatasetGenerator as DefinedAdversarialDatasetGenerator,
)
from semantscript_trainer.adversarial import (
    AdversarialGenerationConfig as DefinedAdversarialGenerationConfig,
)
from semantscript_trainer.constraints import (
    validate_case_constraints as defined_validate_case_constraints,
)
from semantscript_trainer.dataset import (
    DatasetCacheError as DefinedDatasetCacheError,
)
from semantscript_trainer.dataset import DatasetCase as DefinedDatasetCase
from semantscript_trainer.dataset import (
    DatasetConfigurationError as DefinedDatasetConfigurationError,
)
from semantscript_trainer.dataset import (
    SyntheticDatasetGenerator as DefinedSyntheticDatasetGenerator,
)
from semantscript_trainer.dataset import TrainingDataset as DefinedTrainingDataset
from semantscript_trainer.strict_json import StrictJsonError as DefinedStrictJsonError
from semantscript_trainer.strict_json import StrictJsonLimits as DefinedStrictJsonLimits
from semantscript_trainer.strict_json import loads_strict_json as defined_loads_strict_json
from semantscript_trainer.teacher import (
    AdversarialTeacher as DefinedAdversarialTeacher,
)
from semantscript_trainer.teacher import (
    BoundaryPairProposal as DefinedBoundaryPairProposal,
)
from semantscript_trainer.teacher import (
    CaseGenerator as DefinedCaseGenerator,
)
from semantscript_trainer.teacher import (
    GeneratedCase as DefinedGeneratedCase,
)
from semantscript_trainer.teacher import (
    Teacher as DefinedTeacher,
)
from semantscript_trainer.teacher_config import (
    ConstraintsTeacherConfig as DefinedConstraintsTeacherConfig,
)
from semantscript_trainer.teacher_config import TeacherConfig as DefinedTeacherConfig
from semantscript_trainer.teachers import AnthropicTeacher, ConstraintsTeacher, OllamaTeacher


def test_top_level_package_exports_teacher_public_api() -> None:
    assert CaseGenerator is DefinedCaseGenerator
    assert GeneratedCase is DefinedGeneratedCase
    assert Teacher is DefinedTeacher
    assert TeacherConfig is DefinedTeacherConfig
    assert ConstraintsTeacherConfig is DefinedConstraintsTeacherConfig
    assert isinstance(create_teacher({"backend": "constraints"}), ConstraintsTeacher)
    assert semantscript_trainer.create_teacher is create_teacher
    assert semantscript_trainer.load_teacher_config is load_teacher_config
    assert issubclass(TeacherConfigurationError, RuntimeError)
    assert issubclass(TeacherResponseError, RuntimeError)
    assert issubclass(TeacherTransportError, RuntimeError)
    assert TeacherDescriptor(provider="test", model="test", configuration_sha256="a" * 64)


@pytest.mark.parametrize(
    "values",
    [
        {"provider": 1, "model": "test", "configuration_sha256": "a" * 64},
        {"provider": "test", "model": 2, "configuration_sha256": "a" * 64},
        {"provider": "test", "model": "test", "configuration_sha256": 3},
    ],
)
def test_teacher_descriptor_requires_string_identity_fields(values: dict[str, object]) -> None:
    with pytest.raises(TeacherConfigurationError):
        TeacherDescriptor(**values)


def test_top_level_package_exports_dataset_public_api() -> None:
    assert DatasetCacheError is DefinedDatasetCacheError
    assert DatasetCase is DefinedDatasetCase
    assert DatasetConfigurationError is DefinedDatasetConfigurationError
    assert StrictJsonError is DefinedStrictJsonError
    assert StrictJsonLimits is DefinedStrictJsonLimits
    assert SyntheticDatasetGenerator is DefinedSyntheticDatasetGenerator
    assert TrainingDataset is DefinedTrainingDataset
    assert loads_strict_json is defined_loads_strict_json


def test_top_level_package_exports_adversarial_public_api() -> None:
    assert AdversarialDataset is DefinedAdversarialDataset
    assert AdversarialDatasetGenerator is DefinedAdversarialDatasetGenerator
    assert AdversarialGenerationConfig is DefinedAdversarialGenerationConfig
    assert AdversarialTeacher is DefinedAdversarialTeacher
    assert BoundaryPairProposal is DefinedBoundaryPairProposal
    assert validate_case_constraints is defined_validate_case_constraints


def test_top_level_package_exports_training_public_api() -> None:
    modules = (
        artifact_module,
        canonical_input_module,
        lifecycle_module,
        training_contract_module,
        training_module,
        verification_module,
    )

    for module in modules:
        for name in module.__all__:
            assert name in semantscript_trainer.__all__
            assert getattr(semantscript_trainer, name) is getattr(module, name)


def test_factory_selects_anthropic_with_injected_client() -> None:
    client = object()

    teacher = create_teacher(
        {"backend": "anthropic", "model": "claude-sonnet-5", "mode": "direct"},
        client=client,
    )

    assert isinstance(teacher, AnthropicTeacher)
    assert teacher.descriptor.provider == "anthropic"
    assert teacher.descriptor.model == "claude-sonnet-5"
    assert teacher._client is client


def test_factory_selects_ollama_with_injected_client() -> None:
    client = object()

    teacher = create_teacher(
        {"backend": "ollama", "model": "qwen3:14b-q4_K_M"},
        client=client,
    )

    assert isinstance(teacher, OllamaTeacher)
    assert teacher.descriptor.provider == "ollama"
    assert teacher.descriptor.model == "qwen3:14b-q4_K_M"
    assert teacher._client is client
