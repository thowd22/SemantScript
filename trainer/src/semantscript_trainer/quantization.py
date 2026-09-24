"""Derive a quantized release from a published float32 artifact.

The trained PyTorch model is not persisted, so the verified float32 ONNX chain
in a published release is the reference. Quantization produces a new
content-addressed release whose encoder graph is dynamically quantized while
every other resource and every evidence block is copied unchanged; before the
release is published the quantized chain is run against the source chain over
caller-supplied records and refused when decisions or calibration move beyond
the configured tolerances. The tolerances are written into the manifest so a
consumer can see what the graph was allowed to change.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import time
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from semantscript_trainer.artifact import (
    ArtifactConfigurationError,
    ArtifactExportConfig,
    ArtifactExportError,
    ArtifactPublicationError,
    _create_staging_directory,
    _descriptor_directory_path,
    _file_identity,
    _fsync_tree,
    _json_bytes,
    _prepare_artifact_root,
    _prepare_child_directory,
    _publish_pointer,
    _publish_release,
    _remove_owned_staging,
    _require_directory_identity,
    _resource_paths,
    _safe_existing_release_file,
    _validate_manifest_document,
    _validate_rfc3339,
    _write_exclusive,
)
from semantscript_trainer.canonical_input import (
    canonical_input_version,
    serialize_canonical_inputs,
)
from semantscript_trainer.lifecycle import restore_integral_numbers
from semantscript_trainer.strict_json import StrictJsonError, StrictJsonLimits, loads_strict_json
from semantscript_trainer.teacher import JsonValue

DEFAULT_QUANTIZED_ECE_THRESHOLD = 0.1
DEFAULT_QUANTIZED_ECE_BINS = 15
MAXIMUM_QUANTIZATION_RECORDS = 1_000_000
QUANTIZED_PRECISION = "int8-dynamic"

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_RESOURCE_ROLES = ("tokenizer", "encoder", "adapter", "head")


class QuantizationGateError(ArtifactExportError):
    """The quantized graph moved decisions or calibration beyond the tolerances."""

    def __init__(self, message: str, report: QuantizationReport) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True, slots=True)
class QuantizationRecord:
    """One input the quantized chain must decide like the source chain.

    ``label_index`` is the expected output's index in the head support and is
    needed for calibration; ``attested`` marks acceptance-test cases, which
    have their own (default zero) disagreement tolerance.
    """

    inputs: dict[str, JsonValue]
    label_index: int | None = None
    attested: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, dict):
            raise ArtifactConfigurationError("quantization record inputs must be an object")
        if self.label_index is not None and (
            isinstance(self.label_index, bool)
            or not isinstance(self.label_index, int)
            or self.label_index < 0
        ):
            raise ArtifactConfigurationError(
                "quantization record label_index must be a non-negative integer or None"
            )
        if not isinstance(self.attested, bool):
            raise ArtifactConfigurationError("quantization record attested must be a boolean")


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """Quantizer settings and the gate the derived graph must pass."""

    weight_type: Literal["int8", "uint8"] = "int8"
    per_channel: bool = False
    reduce_range: bool = False
    maximum_argmax_disagreement_rate: float = 0.0
    maximum_attested_disagreements: int = 0
    ece_threshold: float = DEFAULT_QUANTIZED_ECE_THRESHOLD
    ece_bins: int = DEFAULT_QUANTIZED_ECE_BINS

    def __post_init__(self) -> None:
        if self.weight_type not in ("int8", "uint8"):
            raise ArtifactConfigurationError('weight_type must be "int8" or "uint8"')
        for name in ("per_channel", "reduce_range"):
            if not isinstance(getattr(self, name), bool):
                raise ArtifactConfigurationError(f"{name} must be a boolean")
        for name in ("maximum_argmax_disagreement_rate", "ece_threshold"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0 <= value <= 1
            ):
                raise ArtifactConfigurationError(f"{name} must be a finite number within [0, 1]")
        attested = self.maximum_attested_disagreements
        if isinstance(attested, bool) or not isinstance(attested, int) or attested < 0:
            raise ArtifactConfigurationError(
                "maximum_attested_disagreements must be a non-negative integer"
            )
        bins = self.ece_bins
        if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 1024:
            raise ArtifactConfigurationError("ece_bins must be an integer within [2, 1024]")

    def to_manifest_document(self, source_manifest_sha256: str) -> dict[str, JsonValue]:
        return {
            "method": "dynamic",
            "weightType": self.weight_type,
            "perChannel": self.per_channel,
            "reduceRange": self.reduce_range,
            "argmaxDisagreementTolerance": float(self.maximum_argmax_disagreement_rate),
            "attestedDisagreementTolerance": self.maximum_attested_disagreements,
            "eceThreshold": float(self.ece_threshold),
            "sourceManifestSha256": source_manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class QuantizationReport:
    """What the quantized chain did on the supplied records, gate or no gate."""

    source_manifest_sha256: str
    method: str
    weight_type: str
    per_channel: bool
    reduce_range: bool
    records_checked: int
    labeled_records: int
    attested_records: int
    argmax_disagreements: int
    attested_disagreements: int
    argmax_disagreement_rate: float
    source_accuracy: float
    quantized_accuracy: float
    source_ece: float
    quantized_ece: float
    temperature: float
    ece_bins: int
    source_encoder_byte_length: int
    quantized_encoder_byte_length: int
    quantized_matmul_count: int
    elapsed_seconds: float

    def to_document(self) -> dict[str, JsonValue]:
        return {
            "sourceManifestSha256": self.source_manifest_sha256,
            "method": self.method,
            "weightType": self.weight_type,
            "perChannel": self.per_channel,
            "reduceRange": self.reduce_range,
            "recordsChecked": self.records_checked,
            "labeledRecords": self.labeled_records,
            "attestedRecords": self.attested_records,
            "argmaxDisagreements": self.argmax_disagreements,
            "attestedDisagreements": self.attested_disagreements,
            "argmaxDisagreementRate": self.argmax_disagreement_rate,
            "sourceAccuracy": self.source_accuracy,
            "quantizedAccuracy": self.quantized_accuracy,
            "sourceEce": self.source_ece,
            "quantizedEce": self.quantized_ece,
            "temperature": self.temperature,
            "eceBins": self.ece_bins,
            "sourceEncoderByteLength": self.source_encoder_byte_length,
            "quantizedEncoderByteLength": self.quantized_encoder_byte_length,
            "quantizedMatMulCount": self.quantized_matmul_count,
            "elapsedSeconds": self.elapsed_seconds,
        }


@dataclass(frozen=True, slots=True)
class QuantizedArtifact:
    """Paths and identity of one published quantized release plus its report."""

    artifact_root: Path
    release_directory: Path
    manifest_path: Path
    manifest_sha256: str
    pointer_path: Path | None
    manifest: dict[str, JsonValue]
    report: QuantizationReport


def quantize_release_artifact(
    source_artifact_root: str | os.PathLike[str],
    source_manifest_sha256: str,
    /,
    *,
    records: Sequence[QuantizationRecord],
    quantization: QuantizationConfig | None = None,
    config: ArtifactExportConfig | None = None,
    artifact_root: str | os.PathLike[str] | None = None,
    created_at: str | None = None,
) -> QuantizedArtifact:
    """Publish a quantized copy of one released artifact after verifying it.

    The source release is found under ``source_artifact_root`` by manifest
    digest and every resource is integrity-checked before use. The new release
    is published under ``artifact_root`` (the source root when omitted) with
    the same staging, exclusive-write and content-addressing rules as a fresh
    export, and its pointer is advanced when the export config says so.
    """

    resolved = ArtifactExportConfig() if config is None else config
    if not isinstance(resolved, ArtifactExportConfig):
        raise ArtifactConfigurationError("config must be an ArtifactExportConfig")
    settings = QuantizationConfig() if quantization is None else quantization
    if not isinstance(settings, QuantizationConfig):
        raise ArtifactConfigurationError("quantization must be a QuantizationConfig")
    checked_records = _validate_records(records)
    if (
        not isinstance(source_manifest_sha256, str)
        or _SHA256.fullmatch(source_manifest_sha256) is None
    ):
        raise ArtifactConfigurationError(
            "source_manifest_sha256 must be 64 lowercase hexadecimal characters"
        )
    if created_at is not None:
        _validate_rfc3339("created_at", created_at)

    source_root = Path(os.path.abspath(Path(source_artifact_root)))
    source_release = source_root / "releases" / f"sha256-{source_manifest_sha256}"
    manifest = _load_source_manifest(source_release, source_manifest_sha256, resolved)
    sources = _source_resources(manifest, source_release, resolved)
    function = cast(dict[str, Any], cast(list[Any], manifest["functions"])[0])
    function_id = cast(str, function["id"])
    encoder_onnx = cast(dict[str, Any], sources["encoder"][0]["onnx"])
    if encoder_onnx.get("precision", "float32") != "float32":
        raise ArtifactConfigurationError("source release encoder is already quantized")
    head_outputs = cast(list[Any], cast(dict[str, Any], sources["head"][0]["onnx"])["outputs"])
    logit_count = _dimension(cast(dict[str, Any], head_outputs[0])["shape"][1])
    maximum_sequence_length = _positive_integer(
        sources["tokenizer"][0].get("maximumSequenceLength"), "tokenizer maximumSequenceLength"
    )
    temperature = _temperature(function)
    compatibility = manifest.get("compatibility")
    encoding = compatibility.get("canonicalInput") if isinstance(compatibility, dict) else None
    input_version = canonical_input_version(encoding)
    if input_version is None:
        raise ArtifactConfigurationError("source manifest declares an unsupported canonical input")
    for record in checked_records:
        if record.label_index is not None and record.label_index >= logit_count:
            raise ArtifactConfigurationError(
                "quantization record label_index exceeds the head support"
            )

    root, root_descriptor = _prepare_artifact_root(
        source_artifact_root if artifact_root is None else artifact_root
    )
    releases: Path | None = None
    releases_descriptor: int | None = None
    staging: Path | None = None
    staging_descriptor: int | None = None
    pointer_path: Path | None = None
    started = time.monotonic()
    try:
        releases, releases_descriptor = _prepare_child_directory(root, "releases", root_descriptor)
        staging, staging_descriptor = _create_staging_directory(releases, releases_descriptor)
        _require_directory_identity(root, root_descriptor, "artifact_root")
        _require_directory_identity(releases, releases_descriptor, "artifact releases")
        _require_directory_identity(staging, staging_descriptor, "artifact staging")
        staging_io_root = _descriptor_directory_path(staging, staging_descriptor)
        paths = _resource_paths(staging_io_root, function_id)
        copied: dict[str, bytes] = {}
        for role in ("tokenizer", "adapter", "head"):
            copied[role] = _read_resource(sources[role][1], resolved.maximum_resource_bytes)
            _write_exclusive(paths[role], copied[role])
        component = _quantize_encoder(sources["encoder"][1], paths["encoder"], settings, resolved)
        report = _verify_quantized_chain(
            source_encoder=sources["encoder"][1],
            quantized_encoder=paths["encoder"],
            adapter=paths["adapter"],
            head=paths["head"],
            tokenizer_json=copied["tokenizer"],
            input_schema=cast(list[Any], function["inputs"]),
            input_version=input_version,
            maximum_sequence_length=maximum_sequence_length,
            logit_count=logit_count,
            temperature=temperature,
            records=checked_records,
            settings=settings,
            source_manifest_sha256=source_manifest_sha256,
            component=component,
            started=started,
        )
        resources = _derived_resources(manifest, paths, settings, source_manifest_sha256, resolved)
        derived = deepcopy(manifest)
        restore_integral_numbers(derived)
        derived["resources"] = cast(JsonValue, resources)
        build = cast(dict[str, JsonValue], derived["build"])
        build["createdAt"] = _utc_now() if created_at is None else created_at
        _validate_manifest_document(derived)
        manifest_bytes = _json_bytes(derived)
        if len(manifest_bytes) > resolved.maximum_manifest_bytes:
            raise ArtifactConfigurationError(
                f"manifest exceeds maximum {resolved.maximum_manifest_bytes} bytes"
            )
        _write_exclusive(staging_io_root / "manifest.json", manifest_bytes)
        _fsync_tree(staging_io_root)
        _require_directory_identity(root, root_descriptor, "artifact_root")
        _require_directory_identity(releases, releases_descriptor, "artifact releases")
        _require_directory_identity(staging, staging_descriptor, "artifact staging")
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        release = _publish_release(
            staging,
            releases / f"sha256-{manifest_sha256}",
            manifest_bytes,
            resources,
            resolved,
            releases_descriptor,
            staging_descriptor,
        )
        if resolved.publish_pointer:
            pointer_path = _publish_pointer(root, manifest_sha256, root_descriptor)
        _require_directory_identity(root, root_descriptor, "artifact_root")
        _require_directory_identity(releases, releases_descriptor, "artifact releases")
        return QuantizedArtifact(
            artifact_root=root,
            release_directory=release,
            manifest_path=release / "manifest.json",
            manifest_sha256=manifest_sha256,
            pointer_path=pointer_path,
            manifest=derived,
            report=report,
        )
    except ArtifactExportError:
        if staging is not None and releases is not None:
            _remove_owned_staging(staging, releases, releases_descriptor, staging_descriptor)
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        if staging is not None and releases is not None:
            _remove_owned_staging(staging, releases, releases_descriptor, staging_descriptor)
        raise ArtifactPublicationError(f"quantized release failed: {error}") from error
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if releases_descriptor is not None:
            os.close(releases_descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)


def _validate_records(records: Sequence[QuantizationRecord]) -> tuple[QuantizationRecord, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ArtifactConfigurationError("records must be a sequence of QuantizationRecord")
    checked = tuple(records)
    if not 1 <= len(checked) <= MAXIMUM_QUANTIZATION_RECORDS:
        raise ArtifactConfigurationError(
            f"records must contain between 1 and {MAXIMUM_QUANTIZATION_RECORDS} entries"
        )
    if any(not isinstance(record, QuantizationRecord) for record in checked):
        raise ArtifactConfigurationError("records must be QuantizationRecord instances")
    if not any(record.label_index is not None for record in checked):
        raise ArtifactConfigurationError(
            "quantization verification needs at least one labeled record for calibration"
        )
    return checked


def _load_source_manifest(
    release: Path, expected_sha256: str, config: ArtifactExportConfig
) -> dict[str, JsonValue]:
    info = _lstat(release, "source release")
    if not stat.S_ISDIR(info.st_mode):
        raise ArtifactConfigurationError("source release is not a directory")
    manifest_bytes = _read_resource(release / "manifest.json", config.maximum_manifest_bytes)
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_sha256:
        raise ArtifactConfigurationError("source release manifest does not match its digest")
    try:
        manifest = loads_strict_json(
            manifest_bytes, limits=StrictJsonLimits(maximum_bytes=config.maximum_manifest_bytes)
        )
    except StrictJsonError as error:
        raise ArtifactConfigurationError(f"source manifest is invalid: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("kind") != "semantscript.application-artifact"
        or manifest.get("artifactVersion") != 1
    ):
        raise ArtifactConfigurationError("source manifest is not an application artifact v1")
    functions = manifest.get("functions")
    if not isinstance(functions, list) or len(functions) != 1 or not isinstance(functions[0], dict):
        raise ArtifactConfigurationError("source manifest must contain exactly one function")
    return manifest


def _source_resources(
    manifest: dict[str, JsonValue], release: Path, config: ArtifactExportConfig
) -> dict[str, tuple[dict[str, Any], Path]]:
    resources = manifest.get("resources")
    if not isinstance(resources, list) or len(resources) != 4:
        raise ArtifactConfigurationError("source manifest must contain exactly four resources")
    by_role: dict[str, tuple[dict[str, Any], Path]] = {}
    for resource in resources:
        if not isinstance(resource, dict):
            raise ArtifactConfigurationError("source manifest resources must be objects")
        role = resource.get("role")
        relative = resource.get("path")
        if role not in _RESOURCE_ROLES or role in by_role or not isinstance(relative, str):
            raise ArtifactConfigurationError("source manifest resource roles are invalid")
        path = _safe_existing_release_file(release, relative)
        byte_length, digest = _file_identity(path, config.maximum_resource_bytes)
        if byte_length != resource.get("byteLength") or digest != resource.get("sha256"):
            raise ArtifactConfigurationError(f"source release resource {relative} is not intact")
        if role != "tokenizer" and not isinstance(resource.get("onnx"), dict):
            raise ArtifactConfigurationError(
                f"source release resource {relative} lacks ONNX metadata"
            )
        by_role[role] = (resource, path)
    if set(by_role) != set(_RESOURCE_ROLES):
        raise ArtifactConfigurationError("source manifest must cover every resource role once")
    return by_role


def _quantize_encoder(
    source: Path, destination: Path, settings: QuantizationConfig, config: ArtifactExportConfig
) -> Any:
    try:
        from semantscript_model.export import quantize_onnx_encoder
    except ImportError as error:
        raise ArtifactConfigurationError(
            "artifact quantization requires the optional ONNX training dependencies"
        ) from error
    try:
        return quantize_onnx_encoder(
            source,
            destination,
            weight_type=settings.weight_type,
            per_channel=settings.per_channel,
            reduce_range=settings.reduce_range,
            maximum_component_bytes=config.maximum_resource_bytes,
        )
    except ImportError as error:
        raise ArtifactConfigurationError(str(error)) from error


def _verify_quantized_chain(
    *,
    source_encoder: Path,
    quantized_encoder: Path,
    adapter: Path,
    head: Path,
    tokenizer_json: bytes,
    input_schema: list[Any],
    input_version: int,
    maximum_sequence_length: int,
    logit_count: int,
    temperature: float,
    records: tuple[QuantizationRecord, ...],
    settings: QuantizationConfig,
    source_manifest_sha256: str,
    component: Any,
    started: float,
) -> QuantizationReport:
    try:
        import numpy
        import onnxruntime
        import torch
        from tokenizers import Tokenizer

        from semantscript_model.calibration import calibration_metrics
    except ImportError as error:
        raise ArtifactConfigurationError(
            "quantized-graph verification requires numpy, onnxruntime, tokenizers, torch and "
            "the SemantScript model package"
        ) from error

    def session(path: Path) -> Any:
        return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    try:
        tokenizer = Tokenizer.from_str(tokenizer_json.decode("utf-8"))
        tokenizer.enable_truncation(maximum_sequence_length)
        source_session = session(source_encoder)
        quantized_session = session(quantized_encoder)
        adapter_session = session(adapter)
        head_session = session(head)
    except Exception as error:
        raise ArtifactConfigurationError(
            f"quantized-graph verification could not start: {error}"
        ) from error

    def chain(encoder_session: Any, ids: Any, mask: Any) -> Any:
        embedding = encoder_session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        function_embedding = adapter_session.run(None, {"sentence_embedding": embedding})[0]
        logits = head_session.run(None, {"function_embedding": function_embedding})[0]
        if logits.shape != (1, logit_count) or not bool(numpy.isfinite(logits).all()):
            raise ArtifactConfigurationError(
                "chain produced logits of the wrong shape or non-finite"
            )
        return logits[0]

    source_logits: list[Any] = []
    quantized_logits: list[Any] = []
    labels: list[int] = []
    disagreements = 0
    attested_disagreements = 0
    attested_records = 0
    for index, record in enumerate(records):
        try:
            text = serialize_canonical_inputs(
                input_schema, record.inputs, version=input_version
            ).decode("utf-8")
        except Exception as error:
            raise ArtifactConfigurationError(
                f"quantization record {index} is not a valid function input: {error}"
            ) from error
        encoding = tokenizer.encode(text)
        if len(encoding.ids) == 0:
            raise ArtifactConfigurationError(f"quantization record {index} tokenized to nothing")
        ids = numpy.asarray([encoding.ids], dtype=numpy.int64)
        mask = numpy.asarray([encoding.attention_mask], dtype=numpy.int64)
        source = chain(source_session, ids, mask)
        quantized = chain(quantized_session, ids, mask)
        if int(source.argmax()) != int(quantized.argmax()):
            disagreements += 1
            if record.attested:
                attested_disagreements += 1
        if record.attested:
            attested_records += 1
        if record.label_index is not None:
            source_logits.append(source)
            quantized_logits.append(quantized)
            labels.append(record.label_index)

    targets = torch.tensor(labels, dtype=torch.int64)
    source_metrics = calibration_metrics(
        torch.tensor(numpy.stack(source_logits), dtype=torch.float32),
        targets,
        temperature=temperature,
        bin_count=settings.ece_bins,
    )
    quantized_metrics = calibration_metrics(
        torch.tensor(numpy.stack(quantized_logits), dtype=torch.float32),
        targets,
        temperature=temperature,
        bin_count=settings.ece_bins,
    )
    report = QuantizationReport(
        source_manifest_sha256=source_manifest_sha256,
        method=component.method,
        weight_type=component.weight_type,
        per_channel=component.per_channel,
        reduce_range=component.reduce_range,
        records_checked=len(records),
        labeled_records=len(labels),
        attested_records=attested_records,
        argmax_disagreements=disagreements,
        attested_disagreements=attested_disagreements,
        argmax_disagreement_rate=disagreements / len(records),
        source_accuracy=float(source_metrics.accuracy),
        quantized_accuracy=float(quantized_metrics.accuracy),
        source_ece=float(source_metrics.ece),
        quantized_ece=float(quantized_metrics.ece),
        temperature=temperature,
        ece_bins=settings.ece_bins,
        source_encoder_byte_length=component.source_byte_length,
        quantized_encoder_byte_length=component.byte_length,
        quantized_matmul_count=component.quantized_matmul_count,
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    failures: list[str] = []
    if attested_disagreements > settings.maximum_attested_disagreements:
        failures.append(
            f"{attested_disagreements} attested record(s) changed decision "
            f"(tolerance {settings.maximum_attested_disagreements})"
        )
    if report.argmax_disagreement_rate > settings.maximum_argmax_disagreement_rate:
        failures.append(
            f"{disagreements} of {len(records)} records changed decision "
            f"(rate {report.argmax_disagreement_rate:.6f}, "
            f"tolerance {settings.maximum_argmax_disagreement_rate})"
        )
    if report.quantized_ece > settings.ece_threshold:
        failures.append(
            f"quantized ECE {report.quantized_ece:.6f} exceeds threshold {settings.ece_threshold}"
        )
    if failures:
        raise QuantizationGateError("quantization gate failed: " + "; ".join(failures), report)
    return report


def _derived_resources(
    manifest: dict[str, JsonValue],
    paths: dict[str, Path],
    settings: QuantizationConfig,
    source_manifest_sha256: str,
    config: ArtifactExportConfig,
) -> list[dict[str, JsonValue]]:
    resources: list[dict[str, JsonValue]] = []
    for original in cast(list[Any], manifest["resources"]):
        resource = cast(dict[str, JsonValue], deepcopy(original))
        restore_integral_numbers(resource)
        role = cast(str, resource["role"])
        byte_length, digest = _file_identity(paths[role], config.maximum_resource_bytes)
        if role == "encoder":
            onnx = cast(dict[str, JsonValue], resource["onnx"])
            onnx["precision"] = QUANTIZED_PRECISION
            onnx["quantization"] = cast(
                JsonValue, settings.to_manifest_document(source_manifest_sha256)
            )
            resource["byteLength"] = byte_length
            resource["sha256"] = digest
        elif byte_length != resource["byteLength"] or digest != resource["sha256"]:
            raise ArtifactPublicationError(f"staged {role} resource differs from the source")
        resources.append(resource)
    return resources


def _read_resource(path: Path, maximum_bytes: int) -> bytes:
    info = _lstat(path, "release resource")
    if not stat.S_ISREG(info.st_mode) or info.st_size < 1 or info.st_size > maximum_bytes:
        raise ArtifactConfigurationError(f"release resource is not a bounded regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(maximum_bytes + 1)
    finally:
        os.close(descriptor)
    if len(data) != info.st_size:
        raise ArtifactPublicationError(f"release resource changed while reading: {path}")
    return data


def _lstat(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise ArtifactConfigurationError(f"{label} is missing: {path}") from error
    if stat.S_ISLNK(info.st_mode):
        raise ArtifactConfigurationError(f"{label} must not be a symlink: {path}")
    return info


def _temperature(function: dict[str, Any]) -> float:
    heads = function.get("heads")
    if not isinstance(heads, list) or len(heads) != 1 or not isinstance(heads[0], dict):
        raise ArtifactConfigurationError("source manifest must contain exactly one head")
    calibration = heads[0].get("calibration")
    value = calibration.get("temperature") if isinstance(calibration, dict) else None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ArtifactConfigurationError("source manifest head calibration temperature is invalid")
    return float(value)


def _dimension(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise ArtifactConfigurationError("head logits width must be a concrete integer dimension")
    return _positive_integer(int(value), "head logits width")


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise ArtifactConfigurationError(f"{label} must be a positive integer")
    if int(value) < 1:
        raise ArtifactConfigurationError(f"{label} must be a positive integer")
    return int(value)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "DEFAULT_QUANTIZED_ECE_BINS",
    "DEFAULT_QUANTIZED_ECE_THRESHOLD",
    "MAXIMUM_QUANTIZATION_RECORDS",
    "QUANTIZED_PRECISION",
    "QuantizationConfig",
    "QuantizationGateError",
    "QuantizationRecord",
    "QuantizationReport",
    "QuantizedArtifact",
    "quantize_release_artifact",
]
