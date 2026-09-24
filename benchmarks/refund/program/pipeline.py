"""Bounded compiler-to-Node lifecycle bridge for the live refund benchmark.

This module never accepts final benchmark cases. It parses a distinct, attested
release-verification record for artifact gating and accepts only the final
benchmark dataset's opaque identity so the two lifecycles can be proven distinct.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from semantscript_trainer import (
    AdversarialDataset,
    ArtifactExportConfig,
    ArtifactProvenance,
    BuiltVerifiedIr,
    ExportedArtifact,
    GeneratedCase,
    TrainingConfig,
    TrainingDataset,
    TrainingResult,
    TrainingRow,
    VerificationConfig,
    VerificationResult,
    VerifiedIrProvenance,
    build_verified_ir,
    export_application_artifact,
    loads_strict_json,
    semantic_json_sha256,
    train_classifier,
    verify_training_result,
)
from semantscript_trainer.case_contract import validate_case
from semantscript_trainer.strict_json import StrictJsonLimits

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_FUNCTION_ID = re.compile(r"^nf_[a-f0-9]{64}$")
_SOURCE_FILE = re.compile(r"^[a-z][a-z0-9-]*\.sem\.ts$")
_MAXIMUM_COMPILER_OUTPUT_BYTES = 8 * 1024 * 1024
_MAXIMUM_RUNTIME_INPUT_BYTES = 1024 * 1024
_MAXIMUM_SUBPROCESS_ERROR_BYTES = 64 * 1024
_MAXIMUM_RELEASE_RECORD_BYTES = 8 * 1024 * 1024
_MAXIMUM_RELEASE_CASES = 10_000
_MAXIMUM_LEDGER_INPUTS = 50_000
_REFUND_SUPPORT = ("approve", "deny", "review")
_REFUND_FUNCTION_ID = "nf_65e347f7dd8736c55d82e539be7ad005cedb3ca396dfa2ab89de619f77adfc9c"
_REFUND_FUNCTION_SEMANTIC_SHA256 = (
    "f7efe891ae5e62b2f0dcec517e118482c995468aaafd813f263c4179a99e545f"
)
REFUND_ARTIFACT_TRAINING_KEY_KIND = "semantscript.refund-artifact-training-key.v1"
_RECORD_ID = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,198}[a-z0-9])?$")
_RFC3339_UTC = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,3})?Z$"
)
_HUMAN_ATTESTATION_DECLARATION = (
    "The listed cases are human-authored, were not generated or rewritten by any model or "
    "teacher, and are licensed or de-identified for this benchmark."
)
_JUDGE_ATTESTATION_DECLARATION = (
    "The listed cases have expected outputs adjudicated case by case by the named independent "
    "model judge under the referenced rubric with a recorded rationale per case; their inputs "
    "derive from real, de-identified transactions; neither inputs nor labels were produced by "
    "any teacher model used for training, and the judge is not a training teacher for this "
    "benchmark."
)
_MAXIMUM_ATTESTATION_TEXT = 500
_RELEASE_RECORD_LIMITS = StrictJsonLimits(
    maximum_bytes=_MAXIMUM_RELEASE_RECORD_BYTES,
    maximum_depth=64,
    maximum_nodes=500_000,
)
_PROGRAM_DIRECTORY = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _PROGRAM_DIRECTORY.parents[2]


class RefundPipelineError(RuntimeError):
    """The benchmark lifecycle could not produce a verified runtime result."""


@dataclass(frozen=True, slots=True)
class CompiledRefundProgram:
    """Exact compiler output and the sole source-stage refund IR record."""

    output_directory: Path
    bundle_path: Path
    source_ir: dict[str, Any]
    function_id: str
    semantic_sha256: str


@dataclass(frozen=True, slots=True)
class FinalBenchmarkDatasetIdentity:
    """The opaque final-test identity; no final-test cases enter model lifecycle code."""

    function_id: str
    semantic_sha256: str
    dataset_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.function_id, str)
            or _FUNCTION_ID.fullmatch(self.function_id) is None
        ):
            raise RefundPipelineError("final benchmark function ID is invalid")
        for name in ("semantic_sha256", "dataset_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise RefundPipelineError(f"final benchmark {name} is invalid")


@dataclass(frozen=True, slots=True)
class JudgeAttestationInputs:
    """Identity of an independent model judge that adjudicated release cases."""

    provider: str
    model: str
    interface: str
    session_reference: str
    rubric_sha256: str

    def __post_init__(self) -> None:
        for name in ("provider", "model", "interface", "session_reference"):
            value = getattr(self, name)
            if not isinstance(value, str) or not 1 <= len(value) <= _MAXIMUM_ATTESTATION_TEXT:
                raise RefundPipelineError(f"judge {name} must be 1 through 500 characters")
        _require_sha256(self.rubric_sha256, "judge rubric digest")

    def document(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "interface": self.interface,
            "sessionReference": self.session_reference,
        }


@dataclass(frozen=True, slots=True, init=False)
class ReleaseVerificationRecord:
    """Closed attested release-gate record parsed from semantic-JSON-compatible data."""

    function_id: str
    semantic_sha256: str
    created_at: str
    payload_sha256: str
    attestation_sha256: str
    case_ids: tuple[str, ...]
    input_sha256s: tuple[str, ...]
    _document_json: bytes

    def __init__(self, document: Mapping[str, Any]) -> None:
        encoded = _exact_json_bytes(document, "release verification record")
        decoded = loads_strict_json(encoded, limits=_RELEASE_RECORD_LIMITS)
        if not isinstance(decoded, dict):
            raise RefundPipelineError("release verification record must be an object")
        parsed = _validate_release_verification_document(decoded)
        object.__setattr__(self, "function_id", cast(str, parsed["function_id"]))
        object.__setattr__(self, "semantic_sha256", cast(str, parsed["semantic_sha256"]))
        object.__setattr__(self, "created_at", cast(str, parsed["created_at"]))
        object.__setattr__(self, "payload_sha256", cast(str, parsed["payload_sha256"]))
        object.__setattr__(self, "attestation_sha256", cast(str, parsed["attestation_sha256"]))
        object.__setattr__(self, "case_ids", cast(tuple[str, ...], parsed["case_ids"]))
        object.__setattr__(
            self,
            "input_sha256s",
            cast(tuple[str, ...], parsed["input_sha256s"]),
        )
        object.__setattr__(
            self,
            "_document_json",
            _exact_json_bytes(decoded, "release verification record"),
        )

    @property
    def document(self) -> dict[str, Any]:
        value = json.loads(self._document_json)
        if not isinstance(value, dict):
            raise AssertionError("validated release record is not an object")
        return value

    @property
    def generated_cases(self) -> tuple[GeneratedCase, ...]:
        cases = self.document["cases"]
        return tuple(
            GeneratedCase(inputs=case["inputs"], output=case["expected"]) for case in cases
        )

    def validate_against(self, source_ir: dict[str, Any]) -> None:
        if (
            source_ir.get("id") != self.function_id
            or source_ir.get("semanticSha256") != self.semantic_sha256
        ):
            raise RefundPipelineError("release verification record binds a different function")
        for index, case in enumerate(self.generated_cases):
            try:
                validate_case(source_ir, case)
            except (RuntimeError, TypeError, ValueError) as error:
                raise RefundPipelineError(
                    f"release verification case {index} violates the compiled contract: {error}"
                ) from error


@dataclass(frozen=True, slots=True)
class RefundPipelineResult:
    """In-memory evidence from one completed live lifecycle run."""

    compiled: CompiledRefundProgram
    training: TrainingResult
    verification: VerificationResult
    built_verified_ir: BuiltVerifiedIr
    exported: ExportedArtifact
    runtime_diagnostic: dict[str, Any]
    training_ledger: dict[str, Any]
    final_benchmark_dataset_sha256: str
    release_verification_payload_sha256: str
    release_verification_attestation_sha256: str


def compile_refund_program(
    output_directory: str | Path,
    /,
    *,
    timeout_seconds: int = 60,
    source_file: str | None = None,
    support: Sequence[str] | None = None,
) -> CompiledRefundProgram:
    """Compile the committed diagnostic refund source into one exact source IR.

    ``source_file`` names another single-function program in the program
    directory (the companion risk function of the shared-encoder experiment)
    with its own output ``support``; such a program is not held to the
    canonical refund identity.
    """

    timeout = _bounded_timeout(timeout_seconds)
    if source_file is not None and (
        not isinstance(source_file, str) or _SOURCE_FILE.fullmatch(source_file) is None
    ):
        raise RefundPipelineError(
            "source_file must name a .sem.ts program in the program directory"
        )
    expected_support = _REFUND_SUPPORT if support is None else tuple(support)
    try:
        output = Path(output_directory)
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise RefundPipelineError("compiler output directory must be absent or empty")
        output.mkdir(parents=True, exist_ok=True)
        resolved_output = output.resolve(strict=True)
    except (OSError, TypeError) as error:
        raise RefundPipelineError(f"compiler output directory is invalid: {error}") from error
    command = ["node", str(_PROGRAM_DIRECTORY / "compile-program.mjs"), str(resolved_output)]
    if source_file is not None:
        command.append(source_file)
    completed = _run_bounded_process(
        command,
        cwd=_REPOSITORY_ROOT,
        timeout_seconds=timeout,
        stdout_limit=_MAXIMUM_COMPILER_OUTPUT_BYTES,
        stderr_limit=_MAXIMUM_SUBPROCESS_ERROR_BYTES,
        context="refund compiler",
    )
    if completed.returncode != 0:
        raise RefundPipelineError(
            "refund compilation failed: "
            + _bounded_error(completed.stderr.decode("utf-8", "replace"))
        )
    try:
        summary = loads_strict_json(completed.stdout)
    except (TypeError, ValueError) as error:
        raise RefundPipelineError("compiler returned an invalid summary") from error
    if not isinstance(summary, dict) or set(summary) != {
        "bundlePath",
        "functionId",
        "semanticSha256",
    }:
        raise RefundPipelineError("compiler returned an unexpected summary shape")
    if not all(isinstance(summary[name], str) for name in summary):
        raise RefundPipelineError("compiler summary values must be strings")
    bundle_path = Path(summary["bundlePath"]).resolve()
    if bundle_path != resolved_output / "semantscript.ir.v1.json":
        raise RefundPipelineError("compiler returned an unexpected IR bundle path")
    try:
        bundle_bytes = bundle_path.read_bytes()
    except OSError as error:
        raise RefundPipelineError(f"could not read compiler IR bundle: {error}") from error
    if len(bundle_bytes) > _MAXIMUM_COMPILER_OUTPUT_BYTES:
        raise RefundPipelineError("compiler IR bundle exceeded its byte limit")
    try:
        bundle = loads_strict_json(bundle_bytes)
    except (TypeError, ValueError) as error:
        raise RefundPipelineError(f"compiler IR bundle is invalid: {error}") from error
    source_ir = _source_record(bundle, expected_support)
    function_id = source_ir.get("id")
    semantic_sha256 = source_ir.get("semanticSha256")
    if (
        not isinstance(function_id, str)
        or _FUNCTION_ID.fullmatch(function_id) is None
        or function_id != summary["functionId"]
    ):
        raise RefundPipelineError("compiler function identity is inconsistent")
    if (
        not isinstance(semantic_sha256, str)
        or _SHA256.fullmatch(semantic_sha256) is None
        or semantic_sha256 != summary["semanticSha256"]
    ):
        raise RefundPipelineError("compiler semantic identity is inconsistent")
    if source_file is None and (
        function_id != _REFUND_FUNCTION_ID or semantic_sha256 != _REFUND_FUNCTION_SEMANTIC_SHA256
    ):
        raise RefundPipelineError("compiler output is not the canonical refund function")
    return CompiledRefundProgram(
        output_directory=resolved_output,
        bundle_path=bundle_path,
        source_ir=source_ir,
        function_id=function_id,
        semantic_sha256=semantic_sha256,
    )


def build_release_verification_record(
    source_ir: dict[str, Any],
    cases: Sequence[tuple[str, GeneratedCase]],
    /,
    *,
    created_at: str,
    attested_at: str,
    evidence_sha256: str,
    attestor: str | None = None,
    judge: JudgeAttestationInputs | None = None,
) -> ReleaseVerificationRecord:
    """Build and re-parse one closed release-only attested verification record.

    Exactly one of ``attestor`` (a human) or ``judge`` (an independent model
    judge) must be given; the record then carries that attestation and ``null``
    for the other.
    """

    if (attestor is None) == (judge is None):
        raise RefundPipelineError(
            "release verification needs exactly one of a human attestor or a judge"
        )
    if judge is not None and not isinstance(judge, JudgeAttestationInputs):
        raise RefundPipelineError("release verification judge must be JudgeAttestationInputs")

    if not isinstance(source_ir, dict):
        raise RefundPipelineError("release verification source IR must be an object")
    function_id = source_ir.get("id")
    semantic_sha256 = source_ir.get("semanticSha256")
    if not isinstance(function_id, str) or _FUNCTION_ID.fullmatch(function_id) is None:
        raise RefundPipelineError("release verification source function ID is invalid")
    if not isinstance(semantic_sha256, str) or _SHA256.fullmatch(semantic_sha256) is None:
        raise RefundPipelineError("release verification source semantic digest is invalid")
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise RefundPipelineError("release verification cases must be a sequence")
    if not 1 <= len(cases) <= _MAXIMUM_RELEASE_CASES:
        raise RefundPipelineError(
            f"release verification must contain 1 through {_MAXIMUM_RELEASE_CASES} cases"
        )
    case_documents: list[dict[str, Any]] = []
    for index, value in enumerate(cases):
        if not isinstance(value, tuple) or len(value) != 2:
            raise RefundPipelineError(
                f"release verification case {index} must be a (case_id, GeneratedCase) tuple"
            )
        case_id, generated = value
        if not isinstance(case_id, str) or _RECORD_ID.fullmatch(case_id) is None:
            raise RefundPipelineError(f"release verification case {index} ID is invalid")
        if not isinstance(generated, GeneratedCase):
            raise RefundPipelineError(
                f"release verification case {index} must contain a GeneratedCase"
            )
        try:
            validate_case(source_ir, generated)
            inputs = deepcopy(generated.inputs)
            expected = deepcopy(generated.output)
        except (RuntimeError, TypeError, ValueError, RecursionError) as error:
            raise RefundPipelineError(
                f"release verification case {index} violates the compiled contract: {error}"
            ) from error
        case_documents.append(
            {
                "id": case_id,
                "inputs": inputs,
                "inputSha256": semantic_json_sha256(inputs),
                "expected": expected,
            }
        )
    case_documents.sort(key=lambda value: cast(str, value["id"]).encode("utf-8"))
    case_ids = [case["id"] for case in case_documents]
    human_attestation: dict[str, Any] | None = None
    judge_attestation: dict[str, Any] | None = None
    if attestor is not None:
        human_attestation = {
            "attestor": attestor,
            "attestedAt": attested_at,
            "caseIds": case_ids,
            "declaration": _HUMAN_ATTESTATION_DECLARATION,
            "evidenceSha256": evidence_sha256,
        }
        attestation: dict[str, Any] = human_attestation
    else:
        assert judge is not None
        judge_attestation = {
            "judge": judge.document(),
            "rubricSha256": judge.rubric_sha256,
            "attestedAt": attested_at,
            "caseIds": case_ids,
            "declaration": _JUDGE_ATTESTATION_DECLARATION,
            "evidenceSha256": evidence_sha256,
        }
        attestation = judge_attestation
    payload: dict[str, Any] = {
        "kind": "semantscript.refund-release-verification-record",
        "recordVersion": 1,
        "benchmark": "refund-decision",
        "split": "release-verification-only",
        "createdAt": created_at,
        "function": {"id": function_id, "semanticSha256": semantic_sha256},
        "support": list(_REFUND_SUPPORT),
        "cases": case_documents,
        "humanAttestation": human_attestation,
        "judgeAttestation": judge_attestation,
        "attestationSha256": semantic_json_sha256(attestation),
    }
    record = ReleaseVerificationRecord({**payload, "payloadSha256": semantic_json_sha256(payload)})
    record.validate_against(source_ir)
    return record


def parse_release_verification_record(
    value: str | bytes | bytearray | Mapping[str, Any],
    source_ir: dict[str, Any],
    /,
) -> ReleaseVerificationRecord:
    """Strictly parse external JSON and validate every case against compiled IR."""

    if isinstance(value, Mapping):
        record = ReleaseVerificationRecord(value)
    else:
        try:
            decoded = loads_strict_json(value, limits=_RELEASE_RECORD_LIMITS)
        except (TypeError, ValueError) as error:
            raise RefundPipelineError(
                f"release verification record is invalid JSON: {error}"
            ) from error
        if not isinstance(decoded, dict):
            raise RefundPipelineError("release verification record must be an object")
        record = ReleaseVerificationRecord(decoded)
    record.validate_against(source_ir)
    return record


def derive_training_input_ledger(
    source_ir: dict[str, Any],
    base_dataset: TrainingDataset,
    training: TrainingResult,
    release_verification: ReleaseVerificationRecord,
    /,
    *,
    created_at: str,
    adversarial_dataset: AdversarialDataset | None = None,
) -> dict[str, Any]:
    """Derive—not accept—the complete input-use ledger from lifecycle objects."""

    _validate_rfc3339(created_at, "ledger createdAt")
    if not isinstance(base_dataset, TrainingDataset):
        raise RefundPipelineError("ledger base dataset must be a TrainingDataset")
    if not isinstance(training, TrainingResult):
        raise RefundPipelineError("ledger training must be a TrainingResult")
    if not isinstance(release_verification, ReleaseVerificationRecord):
        raise RefundPipelineError("ledger release verification must be a ReleaseVerificationRecord")
    if adversarial_dataset is not None and not isinstance(adversarial_dataset, AdversarialDataset):
        raise RefundPipelineError("ledger adversarial dataset is invalid")
    function_id = source_ir.get("id")
    semantic_sha256 = source_ir.get("semanticSha256")
    if not (
        function_id
        == base_dataset.function_id
        == training.function_id
        == release_verification.function_id
        and semantic_sha256
        == base_dataset.semantic_sha256
        == training.semantic_sha256
        == release_verification.semantic_sha256
    ):
        raise RefundPipelineError("ledger lifecycle objects bind different functions")
    if training.base_dataset_sha256 != base_dataset.dataset_sha256:
        raise RefundPipelineError("ledger training result contradicts the base dataset")
    expected_adversarial_sha256 = (
        None if adversarial_dataset is None else adversarial_dataset.dataset_sha256
    )
    if training.adversarial_dataset_sha256 != expected_adversarial_sha256:
        raise RefundPipelineError("ledger training result contradicts the adversarial dataset")
    if adversarial_dataset is not None and (
        adversarial_dataset.function_id != function_id
        or adversarial_dataset.base_dataset_sha256 != base_dataset.dataset_sha256
    ):
        raise RefundPipelineError("ledger adversarial dataset binding is invalid")

    definition = source_ir.get("definition")
    examples = definition.get("examples") if isinstance(definition, dict) else None
    if not isinstance(examples, list):
        raise RefundPipelineError("ledger source IR examples must be an array")
    example_inputs: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        if not isinstance(example, dict) or set(example) != {"inputs", "output"}:
            raise RefundPipelineError(f"ledger source example {index} is malformed")
        inputs = example.get("inputs")
        if not isinstance(inputs, dict):
            raise RefundPipelineError(f"ledger source example {index} inputs must be an object")
        example_inputs.append(inputs)
    synthetic_inputs = [case.inputs for case in base_dataset.cases if case.origin == "synthetic"]
    gold_inputs = [case.inputs for case in base_dataset.cases if case.origin == "gold"]
    adversarial_inputs = (
        [] if adversarial_dataset is None else [case.inputs for case in adversarial_dataset.cases]
    )
    calibration_inputs: list[dict[str, Any]] = []
    for row in training.split.evaluation:
        if not isinstance(row, TrainingRow):
            raise RefundPipelineError("ledger calibration split contains an invalid row")
        calibration_inputs.append(row.inputs)
    verification_inputs = [case.inputs for case in release_verification.generated_cases]

    if _input_digest_counts(example_inputs) != _input_digest_counts(gold_inputs):
        raise RefundPipelineError("ledger base gold cases do not match source IR examples")
    lifecycle_inputs = gold_inputs + synthetic_inputs + adversarial_inputs
    split_inputs = [row.inputs for row in training.split.training + training.split.evaluation]
    if _input_digest_counts(lifecycle_inputs) != _input_digest_counts(split_inputs):
        raise RefundPipelineError("ledger training split does not match source datasets")
    lifecycle_digests = set(_input_digest_counts(lifecycle_inputs))
    calibration_digests = set(_input_digest_counts(calibration_inputs))
    if not calibration_digests <= lifecycle_digests:
        raise RefundPipelineError("ledger calibration contains a non-lifecycle input")
    if lifecycle_digests & set(release_verification.input_sha256s):
        raise RefundPipelineError("ledger release verification overlaps training inputs")

    partitions = [
        _ledger_partition("examples", example_inputs),
        _ledger_partition("synthetic", synthetic_inputs),
        _ledger_partition("adversarial", adversarial_inputs),
        _ledger_partition("calibration", calibration_inputs),
        _ledger_partition("verification", verification_inputs),
    ]
    payload: dict[str, Any] = {
        "kind": "semantscript.refund-training-input-ledger",
        "ledgerVersion": 1,
        "benchmark": "refund-decision",
        "createdAt": created_at,
        "function": {"id": function_id, "semanticSha256": semantic_sha256},
        "sources": {
            "baseDatasetSha256": base_dataset.dataset_sha256,
            "adversarialDatasetSha256": expected_adversarial_sha256,
            "releaseVerificationPayloadSha256": release_verification.payload_sha256,
            "releaseVerificationAttestationSha256": release_verification.attestation_sha256,
        },
        "partitions": partitions,
    }
    return {**payload, "payloadSha256": semantic_json_sha256(payload)}


def derive_refund_artifact_training_key_sha256(
    sources: Mapping[str, Any],
    /,
) -> str:
    """Derive the manifest key from the canonical function and closed ledger sources."""

    if not isinstance(sources, Mapping):
        raise RefundPipelineError("artifact training-key sources must be an object")
    expected = {
        "baseDatasetSha256",
        "adversarialDatasetSha256",
        "releaseVerificationPayloadSha256",
        "releaseVerificationAttestationSha256",
    }
    if set(sources) != expected:
        raise RefundPipelineError(
            "artifact training-key sources must contain exactly: " + ", ".join(sorted(expected))
        )
    base_dataset_sha256 = _require_sha256(
        sources["baseDatasetSha256"], "artifact training-key base dataset digest"
    )
    adversarial_dataset_sha256 = sources["adversarialDatasetSha256"]
    if adversarial_dataset_sha256 is not None:
        adversarial_dataset_sha256 = _require_sha256(
            adversarial_dataset_sha256,
            "artifact training-key adversarial dataset digest",
        )
    release_payload_sha256 = _require_sha256(
        sources["releaseVerificationPayloadSha256"],
        "artifact training-key release payload digest",
    )
    release_attestation_sha256 = _require_sha256(
        sources["releaseVerificationAttestationSha256"],
        "artifact training-key release attestation digest",
    )
    projection = {
        "kind": REFUND_ARTIFACT_TRAINING_KEY_KIND,
        "function": {
            "id": _REFUND_FUNCTION_ID,
            "semanticSha256": _REFUND_FUNCTION_SEMANTIC_SHA256,
        },
        "sources": {
            "baseDatasetSha256": base_dataset_sha256,
            "adversarialDatasetSha256": adversarial_dataset_sha256,
            "releaseVerificationPayloadSha256": release_payload_sha256,
            "releaseVerificationAttestationSha256": release_attestation_sha256,
        },
    }
    return semantic_json_sha256(cast(Any, projection))


def run_refund_pipeline(
    compiler_output_directory: str | Path,
    artifact_root: str | Path,
    base_dataset: TrainingDataset,
    release_verification: ReleaseVerificationRecord,
    final_benchmark: FinalBenchmarkDatasetIdentity,
    /,
    *,
    verified_ir_provenance: VerifiedIrProvenance,
    artifact_provenance: ArtifactProvenance,
    tokenizer: Any,
    encoder: Any,
    tokenizer_json: bytes,
    parity_input_ids: Any,
    parity_attention_mask: Any,
    runtime_inputs: dict[str, Any],
    verified_at: str,
    ledger_created_at: str,
    adversarial_dataset: AdversarialDataset | None = None,
    training_config: TrainingConfig | None = None,
    verification_config: VerificationConfig | None = None,
    export_config: ArtifactExportConfig | None = None,
    subprocess_timeout_seconds: int = 60,
) -> RefundPipelineResult:
    """Run the real compile, train, verify, bind, export, and Node runtime APIs.

    Only the opaque final-benchmark identity crosses this boundary. Its cases and
    labels never enter training, calibration, release verification, IR binding,
    export, or the diagnostic runtime invocation.
    """

    if not isinstance(base_dataset, TrainingDataset):
        raise RefundPipelineError("base_dataset must be a TrainingDataset")
    if not isinstance(release_verification, ReleaseVerificationRecord):
        raise RefundPipelineError("release_verification must be a ReleaseVerificationRecord")
    if not isinstance(final_benchmark, FinalBenchmarkDatasetIdentity):
        raise RefundPipelineError("final_benchmark must be a FinalBenchmarkDatasetIdentity")
    if not isinstance(verified_ir_provenance, VerifiedIrProvenance):
        raise RefundPipelineError("verified_ir_provenance must be a VerifiedIrProvenance")
    if not isinstance(artifact_provenance, ArtifactProvenance):
        raise RefundPipelineError("artifact_provenance must be an ArtifactProvenance")
    if verified_ir_provenance.teacher != base_dataset.teacher:
        raise RefundPipelineError("verified IR teacher provenance contradicts the training dataset")
    if verified_ir_provenance.dataset_sha256 != base_dataset.dataset_sha256:
        raise RefundPipelineError("verified IR dataset provenance contradicts the training dataset")
    compiled = compile_refund_program(
        compiler_output_directory,
        timeout_seconds=subprocess_timeout_seconds,
    )
    release_verification.validate_against(compiled.source_ir)
    if (
        final_benchmark.function_id != compiled.function_id
        or final_benchmark.semantic_sha256 != compiled.semantic_sha256
    ):
        raise RefundPipelineError("final benchmark identity binds a different function")
    if final_benchmark.dataset_sha256 == release_verification.payload_sha256:
        raise RefundPipelineError(
            "final benchmark dataset must differ from release verification evidence"
        )
    if final_benchmark.dataset_sha256 == base_dataset.dataset_sha256 or (
        adversarial_dataset is not None
        and final_benchmark.dataset_sha256 == adversarial_dataset.dataset_sha256
    ):
        raise RefundPipelineError("final benchmark dataset must differ from training datasets")
    training = train_classifier(
        compiled.source_ir,
        base_dataset,
        adversarial_dataset,
        config=training_config,
        tokenizer=tokenizer,
        encoder=encoder,
    )
    verification = verify_training_result(
        compiled.source_ir,
        training,
        base_dataset,
        adversarial_dataset,
        tokenizer=tokenizer,
        attested_verification=release_verification.generated_cases,
        config=verification_config,
        verified_at=verified_at,
    )
    built = build_verified_ir(
        compiled.source_ir,
        training,
        verification,
        verified_ir_provenance,
    )
    training_ledger = derive_training_input_ledger(
        compiled.source_ir,
        base_dataset,
        training,
        release_verification,
        created_at=ledger_created_at,
        adversarial_dataset=adversarial_dataset,
    )
    expected_training_key_sha256 = derive_refund_artifact_training_key_sha256(
        cast(dict[str, Any], training_ledger["sources"])
    )
    if artifact_provenance.training_key_sha256 != expected_training_key_sha256:
        raise RefundPipelineError(
            "artifact provenance training key does not bind the training/release ledger sources"
        )
    export_arguments: dict[str, Any] = {
        "tokenizer_json": tokenizer_json,
        "source_ir_bytes": built.source_ir_bytes,
        "provenance": artifact_provenance,
        "input_ids": parity_input_ids,
        "attention_mask": parity_attention_mask,
    }
    if export_config is not None:
        export_arguments["config"] = export_config
    exported = export_application_artifact(
        artifact_root,
        built.document,
        training,
        verification,
        **export_arguments,
    )
    runtime_diagnostic = run_refund_runtime(
        exported.artifact_root,
        compiled.function_id,
        runtime_inputs,
        timeout_seconds=subprocess_timeout_seconds,
    )
    return RefundPipelineResult(
        compiled=compiled,
        training=training,
        verification=verification,
        built_verified_ir=built,
        exported=exported,
        runtime_diagnostic=runtime_diagnostic,
        training_ledger=training_ledger,
        final_benchmark_dataset_sha256=final_benchmark.dataset_sha256,
        release_verification_payload_sha256=release_verification.payload_sha256,
        release_verification_attestation_sha256=release_verification.attestation_sha256,
    )


def run_refund_runtime(
    artifact_root: str | Path,
    function_id: str,
    inputs: dict[str, Any],
    /,
    *,
    timeout_seconds: int = 60,
    support: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Execute the exported diagnostic function in the deployed Node runtime.

    ``support`` names the function's output support in artifact order when it
    is not the canonical refund decision (the companion risk function).
    """

    timeout = _bounded_timeout(timeout_seconds)
    if not isinstance(function_id, str) or _FUNCTION_ID.fullmatch(function_id) is None:
        raise RefundPipelineError("runtime function ID is invalid")
    expected_support = _REFUND_SUPPORT if support is None else tuple(support)
    if (
        len(expected_support) < 2
        or any(not isinstance(v, str) or not v or "," in v for v in expected_support)
        or len(set(expected_support)) != len(expected_support)
    ):
        raise RefundPipelineError("runtime support must be distinct nonempty strings")
    try:
        encoded = json.dumps(
            inputs,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise RefundPipelineError(f"runtime inputs are not strict JSON: {error}") from error
    if len(encoded) > _MAXIMUM_RUNTIME_INPUT_BYTES:
        raise RefundPipelineError("runtime inputs exceeded their byte limit")
    completed = _run_bounded_process(
        [
            "node",
            str(_PROGRAM_DIRECTORY / "run-runtime.mjs"),
            str(artifact_root),
            function_id,
            ",".join(expected_support),
        ],
        cwd=_REPOSITORY_ROOT,
        stdin=encoded,
        timeout_seconds=timeout,
        stdout_limit=_MAXIMUM_RUNTIME_INPUT_BYTES,
        stderr_limit=_MAXIMUM_SUBPROCESS_ERROR_BYTES,
        context="refund runtime",
    )
    if completed.returncode != 0:
        raise RefundPipelineError(
            "refund runtime failed: " + _bounded_error(completed.stderr.decode("utf-8", "replace"))
        )
    try:
        value = loads_strict_json(completed.stdout)
    except (TypeError, ValueError) as error:
        raise RefundPipelineError("runtime returned invalid JSON") from error
    _validate_runtime_diagnostic(value, expected_support)
    return value


def _refund_source_record(bundle: Any) -> dict[str, Any]:
    return _source_record(bundle, _REFUND_SUPPORT)


def _source_record(bundle: Any, support: Sequence[str]) -> dict[str, Any]:
    if not isinstance(bundle, dict) or bundle.get("kind") != "semantscript.ir-bundle":
        raise RefundPipelineError("compiler output is not a SemantScript IR bundle")
    functions = bundle.get("functions")
    if not isinstance(functions, list) or len(functions) != 1 or not isinstance(functions[0], dict):
        raise RefundPipelineError("refund source must compile to exactly one function")
    record = functions[0]
    if record.get("stage") != "source":
        raise RefundPipelineError("compiler refund function is not source-stage IR")
    runtime = record.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("resultMode") != "diagnostic":
        raise RefundPipelineError("refund source must compile in diagnostic result mode")
    output = record.get("output")
    head = output.get("head") if isinstance(output, dict) else None
    record_support = head.get("support") if isinstance(head, dict) else None
    if not isinstance(record_support, list) or tuple(record_support) != tuple(support):
        raise RefundPipelineError("program output support does not match the expected support")
    definition = record.get("definition")
    if not isinstance(definition, dict) or definition.get("examples") != []:
        raise RefundPipelineError("benchmark held-out cases must not be compiler examples")
    return record


def _validate_release_verification_document(document: dict[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        document,
        {
            "kind",
            "recordVersion",
            "benchmark",
            "split",
            "createdAt",
            "function",
            "support",
            "cases",
            "humanAttestation",
            "judgeAttestation",
            "attestationSha256",
            "payloadSha256",
        },
        "release verification record",
    )
    if document["kind"] != "semantscript.refund-release-verification-record":
        raise RefundPipelineError("release verification record kind is invalid")
    if type(document["recordVersion"]) not in (int, float) or document["recordVersion"] != 1:
        raise RefundPipelineError("release verification record version must be 1")
    if document["benchmark"] != "refund-decision":
        raise RefundPipelineError("release verification benchmark is invalid")
    if document["split"] != "release-verification-only":
        raise RefundPipelineError("release verification split is invalid")
    created_at = _validate_rfc3339(document["createdAt"], "release verification createdAt")
    function = document["function"]
    if not isinstance(function, dict):
        raise RefundPipelineError("release verification function must be an object")
    _require_exact_keys(function, {"id", "semanticSha256"}, "release verification function")
    function_id = function["id"]
    semantic_sha256 = function["semanticSha256"]
    if not isinstance(function_id, str) or _FUNCTION_ID.fullmatch(function_id) is None:
        raise RefundPipelineError("release verification function ID is invalid")
    if not isinstance(semantic_sha256, str) or _SHA256.fullmatch(semantic_sha256) is None:
        raise RefundPipelineError("release verification semantic digest is invalid")
    if document["support"] != list(_REFUND_SUPPORT):
        raise RefundPipelineError("release verification support is not canonical")

    raw_cases = document["cases"]
    if not isinstance(raw_cases, list) or not 1 <= len(raw_cases) <= _MAXIMUM_RELEASE_CASES:
        raise RefundPipelineError(
            f"release verification cases must contain 1 through {_MAXIMUM_RELEASE_CASES} entries"
        )
    case_ids: list[str] = []
    input_sha256s: list[str] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise RefundPipelineError(f"release verification case {index} must be an object")
        _require_exact_keys(
            raw_case,
            {"id", "inputs", "inputSha256", "expected"},
            f"release verification case {index}",
        )
        case_id = raw_case["id"]
        if not isinstance(case_id, str) or _RECORD_ID.fullmatch(case_id) is None:
            raise RefundPipelineError(f"release verification case {index} ID is invalid")
        inputs = raw_case["inputs"]
        if not isinstance(inputs, dict):
            raise RefundPipelineError(f"release verification case {index} inputs must be an object")
        input_sha256 = raw_case["inputSha256"]
        if not isinstance(input_sha256, str) or _SHA256.fullmatch(input_sha256) is None:
            raise RefundPipelineError(f"release verification case {index} input digest is invalid")
        if semantic_json_sha256(cast(Any, inputs)) != input_sha256:
            raise RefundPipelineError(
                f"release verification case {index} input digest does not match inputs"
            )
        if raw_case["expected"] not in _REFUND_SUPPORT:
            raise RefundPipelineError(f"release verification case {index} label is invalid")
        case_ids.append(case_id)
        input_sha256s.append(input_sha256)
    if case_ids != sorted(set(case_ids), key=lambda value: value.encode("utf-8")):
        raise RefundPipelineError("release verification case IDs must be sorted and unique")
    if len(set(input_sha256s)) != len(input_sha256s):
        raise RefundPipelineError("release verification inputs must be unique")

    human = document["humanAttestation"]
    judge = document["judgeAttestation"]
    if (human is None) == (judge is None):
        raise RefundPipelineError("release verification must carry exactly one attestation")
    if human is not None:
        attestation = _validate_human_attestation(human, case_ids, created_at)
        attestation_kind = "human-authored"
    else:
        attestation = _validate_judge_attestation(judge, case_ids, created_at)
        attestation_kind = "independent-judge"
    attestation_sha256 = _require_sha256(
        document["attestationSha256"], "release verification attestation digest"
    )
    if semantic_json_sha256(cast(Any, attestation)) != attestation_sha256:
        raise RefundPipelineError("release verification attestation digest does not match")
    payload_sha256 = _require_sha256(
        document["payloadSha256"], "release verification payload digest"
    )
    payload = {name: value for name, value in document.items() if name != "payloadSha256"}
    if semantic_json_sha256(cast(Any, payload)) != payload_sha256:
        raise RefundPipelineError("release verification payload digest does not match")
    return {
        "function_id": function_id,
        "semantic_sha256": semantic_sha256,
        "created_at": created_at,
        "payload_sha256": payload_sha256,
        "attestation_sha256": attestation_sha256,
        "case_ids": tuple(case_ids),
        "input_sha256s": tuple(input_sha256s),
        "attestation_kind": attestation_kind,
    }


def _validate_attestation_common(
    attestation: Any,
    case_ids: Sequence[str],
    created_at: str,
    declaration: str,
    keys: set[str],
) -> dict[str, Any]:
    if not isinstance(attestation, dict):
        raise RefundPipelineError("release verification attestation must be an object")
    _require_exact_keys(attestation, keys, "release verification attestation")
    attested_at = _validate_rfc3339(attestation["attestedAt"], "release verification attestedAt")
    if _parse_rfc3339(attested_at) > _parse_rfc3339(created_at):
        raise RefundPipelineError("release verification attestation cannot postdate the record")
    if attestation["caseIds"] != list(case_ids):
        raise RefundPipelineError("release verification attestation must list every case ID")
    if attestation["declaration"] != declaration:
        raise RefundPipelineError("release verification attestation declaration is invalid")
    _require_sha256(attestation["evidenceSha256"], "release verification evidence digest")
    return attestation


def _validate_human_attestation(
    attestation: Any,
    case_ids: Sequence[str],
    created_at: str,
) -> dict[str, Any]:
    validated = _validate_attestation_common(
        attestation,
        case_ids,
        created_at,
        _HUMAN_ATTESTATION_DECLARATION,
        {"attestor", "attestedAt", "caseIds", "declaration", "evidenceSha256"},
    )
    attestor = validated["attestor"]
    if not isinstance(attestor, str) or not 1 <= len(attestor) <= _MAXIMUM_ATTESTATION_TEXT:
        raise RefundPipelineError("release verification attestor must be 1 through 500 characters")
    return validated


def _validate_judge_attestation(
    attestation: Any,
    case_ids: Sequence[str],
    created_at: str,
) -> dict[str, Any]:
    validated = _validate_attestation_common(
        attestation,
        case_ids,
        created_at,
        _JUDGE_ATTESTATION_DECLARATION,
        {"judge", "rubricSha256", "attestedAt", "caseIds", "declaration", "evidenceSha256"},
    )
    judge = validated["judge"]
    if not isinstance(judge, dict):
        raise RefundPipelineError("release verification judge must be an object")
    _require_exact_keys(
        judge, {"provider", "model", "interface", "sessionReference"}, "release verification judge"
    )
    for name, value in judge.items():
        if not isinstance(value, str) or not 1 <= len(value) <= _MAXIMUM_ATTESTATION_TEXT:
            raise RefundPipelineError(
                f"release verification judge {name} must be 1 through 500 characters"
            )
    _require_sha256(validated["rubricSha256"], "release verification rubric digest")
    return validated


def _ledger_partition(name: str, inputs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(inputs) > _MAXIMUM_LEDGER_INPUTS:
        raise RefundPipelineError(
            f"ledger {name} partition exceeds {_MAXIMUM_LEDGER_INPUTS} lifecycle inputs"
        )
    try:
        digests = sorted({semantic_json_sha256(cast(Any, value)) for value in inputs})
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError) as error:
        raise RefundPipelineError(
            f"ledger {name} inputs are invalid semantic JSON: {error}"
        ) from error
    return {"name": name, "inputSha256s": digests}


def _input_digest_counts(inputs: Sequence[dict[str, Any]]) -> Counter[str]:
    try:
        return Counter(semantic_json_sha256(cast(Any, value)) for value in inputs)
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError) as error:
        raise RefundPipelineError(f"lifecycle inputs are invalid semantic JSON: {error}") from error


def _require_exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise RefundPipelineError(f"{context} must contain exactly: {', '.join(sorted(expected))}")


def _require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RefundPipelineError(f"{context} must be 64 lowercase hexadecimal characters")
    return value


def _validate_rfc3339(value: Any, context: str) -> str:
    if not isinstance(value, str) or _RFC3339_UTC.fullmatch(value) is None:
        raise RefundPipelineError(f"{context} must be an RFC 3339 UTC timestamp")
    try:
        _parse_rfc3339(value)
    except ValueError as error:
        raise RefundPipelineError(f"{context} must be a real calendar timestamp") from error
    return value


def _parse_rfc3339(value: str) -> datetime:
    return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(UTC)


def _exact_json_bytes(value: Mapping[str, Any], context: str) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, OverflowError, UnicodeError, RecursionError) as error:
        raise RefundPipelineError(f"{context} is not strict JSON: {error}") from error
    if len(encoded) > _MAXIMUM_RELEASE_RECORD_BYTES:
        raise RefundPipelineError(
            f"{context} exceeds maximum byte length {_MAXIMUM_RELEASE_RECORD_BYTES}"
        )
    return encoded


def _run_bounded_process(
    command: Sequence[str],
    /,
    *,
    cwd: Path,
    timeout_seconds: int,
    stdout_limit: int,
    stderr_limit: int,
    context: str,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Capture a process incrementally and terminate its group at hard bounds."""

    timeout = _bounded_timeout(timeout_seconds)
    for name, value in (("stdout_limit", stdout_limit), ("stderr_limit", stderr_limit)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RefundPipelineError(f"{context} {name} must be a positive integer")
    if not isinstance(context, str) or not context:
        raise RefundPipelineError("bounded subprocess context must be nonempty")
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)) or not command:
        raise RefundPipelineError(f"{context} command must be a nonempty string sequence")
    if any(not isinstance(argument, str) or not argument for argument in command):
        raise RefundPipelineError(f"{context} command arguments must be nonempty strings")
    if stdin is not None and not isinstance(stdin, bytes):
        raise RefundPipelineError(f"{context} stdin must be bytes")

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    capture_lock = threading.Lock()
    capture_failure: list[str] = []
    stdout_done = threading.Event()
    stderr_done = threading.Event()
    started = time.monotonic()

    def capture(
        stream: Any,
        chunks: list[bytes],
        limit: int,
        name: str,
        done: threading.Event,
    ) -> None:
        size = 0
        try:
            while True:
                chunk = os.read(stream.fileno(), 64 * 1024)
                if not chunk:
                    return
                size += len(chunk)
                if size > limit:
                    with capture_lock:
                        if not capture_failure:
                            capture_failure.append(f"{context} {name} exceeded its byte limit")
                    return
                chunks.append(chunk)
        except (OSError, ValueError) as error:
            with capture_lock:
                if not capture_failure:
                    capture_failure.append(f"{context} {name} could not be read: {error}")
        finally:
            done.set()

    with tempfile.TemporaryFile() as input_file:
        if stdin is not None:
            input_file.write(stdin)
            input_file.seek(0)
        popen_arguments: dict[str, Any] = {
            "cwd": cwd,
            "stdin": input_file if stdin is not None else subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        if os.name == "posix":
            popen_arguments["start_new_session"] = True
        elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            popen_arguments["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            process = subprocess.Popen(list(command), **popen_arguments)
        except OSError as error:
            raise RefundPipelineError(f"could not run the {context}: {error}") from error
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(
            target=capture,
            args=(process.stdout, stdout_chunks, stdout_limit, "stdout", stdout_done),
            daemon=True,
            name=f"{context}-stdout",
        )
        stderr_thread = threading.Thread(
            target=capture,
            args=(process.stderr, stderr_chunks, stderr_limit, "stderr", stderr_done),
            daemon=True,
            name=f"{context}-stderr",
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            while True:
                with capture_lock:
                    failure = capture_failure[0] if capture_failure else None
                if failure is not None:
                    _terminate_process_group(process)
                    raise RefundPipelineError(failure)
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    _terminate_process_group(process)
                    raise RefundPipelineError(f"{context} exceeded its {timeout}-second timeout")
                if process.poll() is not None and stdout_done.is_set() and stderr_done.is_set():
                    returncode = process.returncode
                    assert returncode is not None
                    break
                time.sleep(min(remaining, 0.01))
        finally:
            # The command is not allowed to leave background descendants behind,
            # even when the direct parent has already exited successfully.
            _terminate_process_group(process)
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)
            process.stdout.close()
            process.stderr.close()
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=returncode,
        stdout=b"".join(stdout_chunks),
        stderr=b"".join(stderr_chunks),
    )


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            if process.poll() is None:
                process.wait(timeout=1)
            return
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.wait(timeout=1)
        return

    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=1)
            return
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)


def _validate_runtime_diagnostic(value: Any, support: Sequence[str] = _REFUND_SUPPORT) -> None:
    if not isinstance(value, dict) or set(value) != {
        "value",
        "confidence",
        "uncertainty",
        "distribution",
        "expectedValue",
    }:
        raise RefundPipelineError("runtime returned an unexpected diagnostic shape")
    if value["value"] not in support:
        raise RefundPipelineError("runtime value is outside the canonical support")
    for name in ("confidence", "uncertainty"):
        measurement = value[name]
        if (
            isinstance(measurement, bool)
            or not isinstance(measurement, (int, float))
            or not math.isfinite(measurement)
            or not 0 <= measurement <= 1
        ):
            raise RefundPipelineError(f"runtime {name} is invalid")
    if value["expectedValue"] is not None:
        raise RefundPipelineError("nominal refund output must have a null expected value")
    distribution = value.get("distribution")
    if not isinstance(distribution, list) or tuple(
        item.get("value") if isinstance(item, dict) else None for item in distribution
    ) != tuple(support):
        raise RefundPipelineError("runtime distribution is not in canonical support order")
    if any(set(item) != {"value", "probability"} for item in distribution):
        raise RefundPipelineError("runtime distribution entries have an unexpected shape")
    probabilities = [item.get("probability") for item in distribution]
    if any(
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
        for probability in probabilities
    ):
        raise RefundPipelineError("runtime distribution contains an invalid probability")
    if abs(sum(probabilities) - 1.0) > 1e-12:
        raise RefundPipelineError("runtime probabilities do not sum to one")
    maximum = max(probabilities)
    if abs(value["confidence"] - maximum) > 8 * math.ulp(1.0):
        raise RefundPipelineError("runtime confidence is not the top probability")
    expected_value = tuple(support)[probabilities.index(maximum)]
    if value["value"] != expected_value:
        raise RefundPipelineError("runtime value is not the stable top-probability decision")


def _bounded_timeout(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 300:
        raise RefundPipelineError("subprocess timeout must be an integer from 1 through 300")
    return value


def _bounded_error(value: str) -> str:
    return value[-4096:].strip() or "unknown subprocess failure"


__all__ = [
    "CompiledRefundProgram",
    "FinalBenchmarkDatasetIdentity",
    "RefundPipelineError",
    "RefundPipelineResult",
    "ReleaseVerificationRecord",
    "build_release_verification_record",
    "compile_refund_program",
    "derive_training_input_ledger",
    "parse_release_verification_record",
    "run_refund_pipeline",
    "run_refund_runtime",
]
