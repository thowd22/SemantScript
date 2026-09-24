"""Build exact verified-stage IR from compiler, training, and verification records."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from semantscript_trainer.artifact import (
    ArtifactConfigurationError,
    validate_verified_ir_binding,
)
from semantscript_trainer.canonical_input import CANONICAL_INPUT_ENCODINGS
from semantscript_trainer.teacher import JsonValue, NeuralFunctionIr, TeacherDescriptor
from semantscript_trainer.training import TrainingConfig, TrainingResult
from semantscript_trainer.training_contract import TrainingRow, TrainingSplit
from semantscript_trainer.verification import (
    VerificationResult,
    require_passing_verification,
)

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_IMMUTABLE_REVISION = re.compile(r"^[a-f0-9]{40}$")
_TRAINER_COMMIT = re.compile(r"^[a-f0-9]{7,64}$")
_RFC3339 = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:"
    r"[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)
_MAXIMUM_SAFE_INTEGER = 2**53 - 1
_TOP_LEVEL_FIELDS = {
    "kind",
    "irVersion",
    "stage",
    "id",
    "semanticSha256",
    "source",
    "definition",
    "inputs",
    "output",
    "model",
    "runtime",
    "trainingProvenance",
    "verification",
}


class VerifiedIrBuildError(ValueError):
    """Compiler, training, verification, and provenance records cannot be bound."""


@dataclass(frozen=True, slots=True)
class TrainingProvenanceCounts:
    """Complete, caller-declared lifecycle counts checked against measured records."""

    examples: int
    synthetic: int
    adversarial: int
    calibration: int
    verification: int
    attested_verification: int

    def __post_init__(self) -> None:
        for name in (
            "examples",
            "synthetic",
            "adversarial",
            "calibration",
            "verification",
            "attested_verification",
        ):
            value = getattr(self, name)
            minimum = 1 if name in ("calibration", "verification", "attested_verification") else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise VerifiedIrBuildError(
                    f"provenance count {name} must be an integer of at least {minimum}"
                )

    def to_document(self) -> dict[str, JsonValue]:
        return {
            "examples": self.examples,
            "synthetic": self.synthetic,
            "adversarial": self.adversarial,
            "calibration": self.calibration,
            "verification": self.verification,
            "attestedVerification": self.attested_verification,
        }


@dataclass(frozen=True, slots=True)
class VerifiedIrProvenance:
    """Explicit identities needed to complete source-stage training provenance."""

    teacher: TeacherDescriptor
    base_model_name: str
    base_model_revision: str
    base_model_weights_sha256: str
    dataset_sha256: str
    counts: TrainingProvenanceCounts
    seed: int
    trainer_version: str
    trainer_commit: str
    trained_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.teacher, TeacherDescriptor):
            raise VerifiedIrBuildError("provenance teacher must be a TeacherDescriptor")
        if not isinstance(self.base_model_name, str) or not self.base_model_name:
            raise VerifiedIrBuildError("provenance base_model_name must be nonempty")
        if (
            not isinstance(self.base_model_revision, str)
            or _IMMUTABLE_REVISION.fullmatch(self.base_model_revision) is None
        ):
            raise VerifiedIrBuildError(
                "provenance base_model_revision must be an immutable 40-character commit SHA"
            )
        _require_sha256("base_model_weights_sha256", self.base_model_weights_sha256)
        _require_sha256("dataset_sha256", self.dataset_sha256)
        if not isinstance(self.counts, TrainingProvenanceCounts):
            raise VerifiedIrBuildError("provenance counts must be TrainingProvenanceCounts")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise VerifiedIrBuildError("provenance seed must be a non-negative integer")
        if not isinstance(self.trainer_version, str) or not self.trainer_version:
            raise VerifiedIrBuildError("provenance trainer_version must be nonempty")
        if (
            not isinstance(self.trainer_commit, str)
            or _TRAINER_COMMIT.fullmatch(self.trainer_commit) is None
        ):
            raise VerifiedIrBuildError(
                "provenance trainer_commit must be 7 to 64 lowercase hexadecimal characters"
            )
        _parse_timestamp("provenance trained_at", self.trained_at)

    def to_document(self) -> dict[str, JsonValue]:
        return {
            "status": "complete",
            "teacher": {
                "provider": self.teacher.provider,
                "model": self.teacher.model,
                "configurationSha256": self.teacher.configuration_sha256,
            },
            "baseModel": {
                "name": self.base_model_name,
                "revision": self.base_model_revision,
                "weightsSha256": self.base_model_weights_sha256,
            },
            "datasetSha256": self.dataset_sha256,
            "counts": self.counts.to_document(),
            "seed": self.seed,
            "trainer": {
                "version": self.trainer_version,
                "commit": self.trainer_commit,
            },
            "trainedAt": self.trained_at,
        }


@dataclass(frozen=True, slots=True)
class BuiltVerifiedIr:
    """Immutable exact bytes for one validated verified-stage IR document."""

    source_ir_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.source_ir_bytes, bytes) or not self.source_ir_bytes:
            raise VerifiedIrBuildError("source_ir_bytes must be nonempty bytes")

    @property
    def document(self) -> NeuralFunctionIr:
        """Return a fresh mutable document suitable for artifact export."""

        value = json.loads(self.source_ir_bytes)
        if not isinstance(value, dict):
            raise AssertionError("validated verified IR bytes must encode an object")
        return cast(NeuralFunctionIr, value)


def build_verified_ir(
    source_ir: NeuralFunctionIr,
    training: TrainingResult,
    verification: VerificationResult,
    provenance: VerifiedIrProvenance,
    /,
) -> BuiltVerifiedIr:
    """Advance one exact compiler source record to a validated verified record."""

    _validate_source_lifecycle(source_ir)
    if not isinstance(training, TrainingResult):
        raise VerifiedIrBuildError("training must be a TrainingResult")
    if not isinstance(verification, VerificationResult):
        raise VerifiedIrBuildError("verification must be a VerificationResult")
    if not isinstance(provenance, VerifiedIrProvenance):
        raise VerifiedIrBuildError("provenance must be a VerifiedIrProvenance")
    try:
        require_passing_verification(verification)
    except (TypeError, ValueError, RuntimeError) as error:
        raise VerifiedIrBuildError(f"verification must be passing: {error}") from error

    function_id = source_ir.get("id")
    semantic_sha256 = source_ir.get("semanticSha256")
    if not (
        function_id == training.function_id == verification.function_id
        and semantic_sha256 == training.semantic_sha256 == verification.semantic_sha256
    ):
        raise VerifiedIrBuildError(
            "source IR, training, and verification identities must match exactly"
        )
    _validate_training_provenance(source_ir, training, verification, provenance)

    try:
        document = deepcopy(source_ir)
    except (TypeError, ValueError, RecursionError) as error:
        raise VerifiedIrBuildError(f"source IR cannot be snapshotted: {error}") from error
    restore_integral_numbers(document)
    document["stage"] = "verified"
    training_provenance = provenance.to_document()
    # The encoding the rows were serialized with; the artifact and runtime follow it.
    training_provenance["canonicalInput"] = CANONICAL_INPUT_ENCODINGS[
        training.config.canonical_input_version
    ]
    document["trainingProvenance"] = training_provenance
    document["verification"] = verification.to_ir_document()
    encoded = _serialize_exact_json(document)
    try:
        validate_verified_ir_binding(document, encoded, training, verification)
    except ArtifactConfigurationError as error:
        raise VerifiedIrBuildError(f"constructed verified IR is invalid: {error}") from error
    return BuiltVerifiedIr(source_ir_bytes=encoded)


def _validate_source_lifecycle(source_ir: NeuralFunctionIr) -> None:
    if not isinstance(source_ir, dict) or set(source_ir) != _TOP_LEVEL_FIELDS:
        raise VerifiedIrBuildError("source IR must contain exactly the neural-function v1 fields")
    ir_version = source_ir.get("irVersion")
    # Compiler bundles read through the strict JSON reader carry every number
    # as binary64, so the version may arrive as the integral float 1.0.
    if (
        source_ir.get("kind") != "semantscript.neural-function"
        or isinstance(ir_version, bool)
        or not isinstance(ir_version, (int, float))
        or ir_version != 1
    ):
        raise VerifiedIrBuildError("source IR must be neural-function IR v1")
    if source_ir.get("stage") != "source":
        raise VerifiedIrBuildError("lifecycle builder requires source-stage IR")
    if source_ir.get("trainingProvenance") != {"status": "pending"}:
        raise VerifiedIrBuildError("source IR trainingProvenance must be exactly pending")
    if source_ir.get("verification") != {"status": "pending"}:
        raise VerifiedIrBuildError("source IR verification must be exactly pending")


def _validate_training_provenance(
    source_ir: NeuralFunctionIr,
    training: TrainingResult,
    verification: VerificationResult,
    provenance: VerifiedIrProvenance,
) -> None:
    if not isinstance(training.config, TrainingConfig):
        raise VerifiedIrBuildError("training config is invalid")
    if not isinstance(training.split, TrainingSplit):
        raise VerifiedIrBuildError("training split is invalid")
    if (
        provenance.base_model_name != training.config.encoder_name
        or provenance.base_model_revision != training.config.encoder_revision
    ):
        raise VerifiedIrBuildError(
            "provenance base model must match the trained encoder and immutable revision"
        )
    if provenance.dataset_sha256 != training.base_dataset_sha256:
        raise VerifiedIrBuildError("provenance dataset digest must match the training dataset")
    if provenance.seed != training.config.seed:
        raise VerifiedIrBuildError("provenance seed must match the training split seed")

    rows = training.split.training + training.split.evaluation
    if any(not isinstance(row, TrainingRow) for row in rows):
        raise VerifiedIrBuildError("training split contains an invalid row")
    origins = {
        name: sum(row.origin == name for row in rows)
        for name in ("gold", "synthetic", "constraint-boundary", "counterfactual")
    }
    definition = source_ir.get("definition")
    examples = definition.get("examples") if isinstance(definition, dict) else None
    if not isinstance(examples, list):
        raise VerifiedIrBuildError("source IR definition examples must be an array")
    expected_examples = len(examples)
    if origins["gold"] != expected_examples:
        raise VerifiedIrBuildError(
            "training gold-row count must equal the compiler source example count"
        )
    if verification.attested_cases < origins["gold"]:
        raise VerifiedIrBuildError(
            "verification attested count cannot be smaller than the training gold count"
        )
    calibration_count = len(training.split.evaluation)
    verification_count = len(rows) + verification.attested_cases - origins["gold"]
    expected = TrainingProvenanceCounts(
        examples=expected_examples,
        synthetic=origins["synthetic"],
        adversarial=origins["constraint-boundary"] + origins["counterfactual"],
        calibration=calibration_count,
        verification=verification_count,
        attested_verification=verification.attested_cases,
    )
    if provenance.counts != expected:
        raise VerifiedIrBuildError(
            f"provenance counts do not match training and verification evidence; expected {expected}"
        )
    calibration = verification.metrics.heads[0].calibration
    if calibration.sample_count != calibration_count:
        raise VerifiedIrBuildError(
            "verification calibration sample count must equal the held-out row count"
        )
    trained_at = _parse_timestamp("provenance trained_at", provenance.trained_at)
    verified_at = _parse_timestamp("verification verified_at", verification.verified_at)
    if trained_at > verified_at:
        raise VerifiedIrBuildError("provenance trained_at cannot be later than verified_at")


def restore_integral_numbers(document: NeuralFunctionIr) -> None:
    """Write integral binary64 values back as integers, as the compiler emitted them.

    The compiler serializes IR with JavaScript number formatting, which never
    writes a fractional part for an integral value, while the trainer's strict
    JSON reader rounds every token to binary64. Both spell the same semantic
    value, so the exact verified bytes restore the compiler's spelling for
    every safe integer (including signed zero, which JavaScript writes as 0).
    """

    work: list[dict[str, Any] | list[Any]] = [document]
    while work:
        current = work.pop()
        items = current.items() if isinstance(current, dict) else enumerate(current)
        for key, value in items:
            if isinstance(value, (dict, list)):
                work.append(value)
            elif (
                isinstance(value, float)
                and value.is_integer()
                and abs(value) <= _MAXIMUM_SAFE_INTEGER
            ):
                current[key] = int(value)  # type: ignore[index]


def _serialize_exact_json(document: NeuralFunctionIr) -> bytes:
    try:
        return (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError) as error:
        raise VerifiedIrBuildError(
            f"verified IR cannot be encoded as exact UTF-8 JSON: {error}"
        ) from error


def _require_sha256(name: str, value: Any) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise VerifiedIrBuildError(f"provenance {name} must be 64 lowercase hexadecimal characters")


def _parse_timestamp(name: str, value: Any) -> datetime:
    if not isinstance(value, str) or _RFC3339.fullmatch(value) is None:
        raise VerifiedIrBuildError(f"{name} must be an RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as error:
        raise VerifiedIrBuildError(f"{name} must be an RFC 3339 timestamp") from error
    if parsed.tzinfo is None:
        raise VerifiedIrBuildError(f"{name} must be an RFC 3339 timestamp")
    return parsed


__all__ = [
    "BuiltVerifiedIr",
    "TrainingProvenanceCounts",
    "VerifiedIrBuildError",
    "VerifiedIrProvenance",
    "build_verified_ir",
]
