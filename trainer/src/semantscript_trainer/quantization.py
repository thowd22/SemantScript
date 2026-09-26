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
import logging
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
    have their own (default zero) disagreement tolerance. ``function_id``
    names the function the record belongs to and is required when the release
    has more than one. A function with several heads (an object output) takes
    one label per head, in manifest order, through ``label_indices`` instead
    of ``label_index``; ``None`` leaves that head unlabeled.
    """

    inputs: dict[str, JsonValue]
    label_index: int | None = None
    attested: bool = False
    function_id: str | None = None
    label_indices: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, dict):
            raise ArtifactConfigurationError("quantization record inputs must be an object")
        if self.label_index is not None and not _is_label(self.label_index):
            raise ArtifactConfigurationError(
                "quantization record label_index must be a non-negative integer or None"
            )
        if not isinstance(self.attested, bool):
            raise ArtifactConfigurationError("quantization record attested must be a boolean")
        if self.function_id is not None and (
            not isinstance(self.function_id, str) or not self.function_id
        ):
            raise ArtifactConfigurationError(
                "quantization record function_id must be a non-empty string or None"
            )
        if self.label_indices is not None:
            if self.label_index is not None:
                raise ArtifactConfigurationError(
                    "quantization record takes label_index or label_indices, not both"
                )
            if (
                not isinstance(self.label_indices, tuple)
                or not self.label_indices
                or any(label is not None and not _is_label(label) for label in self.label_indices)
            ):
                raise ArtifactConfigurationError(
                    "quantization record label_indices must be a non-empty tuple of "
                    "non-negative integers or None"
                )

    def labels(self) -> tuple[int | None, ...]:
        """One label per head: ``label_indices``, or ``label_index`` for a single head."""

        return self.label_indices if self.label_indices is not None else (self.label_index,)


def _is_label(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


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
class QuantizationFunctionReport:
    """One function's share of the quantized-graph check.

    Accuracy and ECE are over the function's labeled records, the worst head
    for ECE; they are ``None`` when none of its records carries a label.
    """

    function_id: str
    records_checked: int
    labeled_records: int
    attested_records: int
    argmax_disagreements: int
    attested_disagreements: int
    source_accuracy: float | None
    quantized_accuracy: float | None
    source_ece: float | None
    quantized_ece: float | None

    def to_document(self) -> dict[str, JsonValue]:
        return {
            "id": self.function_id,
            "recordsChecked": self.records_checked,
            "labeledRecords": self.labeled_records,
            "attestedRecords": self.attested_records,
            "argmaxDisagreements": self.argmax_disagreements,
            "attestedDisagreements": self.attested_disagreements,
            "sourceAccuracy": self.source_accuracy,
            "quantizedAccuracy": self.quantized_accuracy,
            "sourceEce": self.source_ece,
            "quantizedEce": self.quantized_ece,
        }


@dataclass(frozen=True, slots=True)
class QuantizationReport:
    """What the quantized chain did on the supplied records, gate or no gate.

    For a release with several functions or heads, accuracy is over every
    labeled head decision, the ECE figures are the worst head's, the
    temperature is the first head's and the encoder sizes add up every
    encoder graph; ``functions`` carries each function's own figures.
    """

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
    functions: tuple[QuantizationFunctionReport, ...] = ()

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
            "functions": [function.to_document() for function in self.functions],
        }

    def to_manifest_verification(
        self, record_sources: Sequence[dict[str, JsonValue]]
    ) -> dict[str, JsonValue]:
        """The figures a derived manifest records next to its tolerances."""

        return {
            "recordsChecked": self.records_checked,
            "attestedRecords": self.attested_records,
            "decisionChanges": self.argmax_disagreements,
            "attestedDecisionChanges": self.attested_disagreements,
            "decisionChangeRate": float(self.argmax_disagreement_rate),
            "sourceEce": float(self.source_ece),
            "quantizedEce": float(self.quantized_ece),
            "recordSources": [dict(source) for source in record_sources],
            "functions": [
                {
                    "id": function.function_id,
                    "recordsChecked": function.records_checked,
                    "attestedRecords": function.attested_records,
                    "decisionChanges": function.argmax_disagreements,
                    "attestedDecisionChanges": function.attested_disagreements,
                    "sourceEce": function.source_ece,
                    "quantizedEce": function.quantized_ece,
                }
                for function in self.functions
            ],
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
    record_sources: Sequence[dict[str, JsonValue]] | None = None,
) -> QuantizedArtifact:
    """Publish a quantized copy of one released artifact after verifying it.

    The source release is found under ``source_artifact_root`` by manifest
    digest and every resource is integrity-checked before use. Every encoder
    graph is quantized (a depth-routed release has one per depth) and every
    other resource is copied unchanged. Each record runs through its
    function's chain (its encoder, adapter and heads) on both graphs, and a
    decision is the tuple of the heads' answers. The new release is published
    under ``artifact_root`` (the source root when omitted) with the same
    staging, exclusive-write and content-addressing rules as a fresh export,
    and its pointer is advanced when the export config says so.

    ``record_sources`` names what the records were drawn from (dataset
    digests and counts); when given, each quantized encoder's manifest entry
    also records the figures the gate measured (``quantization.verification``).
    """

    resolved = ArtifactExportConfig() if config is None else config
    if not isinstance(resolved, ArtifactExportConfig):
        raise ArtifactConfigurationError("config must be an ArtifactExportConfig")
    settings = QuantizationConfig() if quantization is None else quantization
    if not isinstance(settings, QuantizationConfig):
        raise ArtifactConfigurationError("quantization must be a QuantizationConfig")
    checked_records = _validate_records(records)
    sources_document = _validate_record_sources(record_sources)
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
    chains = _function_chains(manifest, sources)
    tokenizer_ref = _model_ref(manifest, "tokenizerRef")
    tokenizer_resource = sources.get(tokenizer_ref)
    if tokenizer_resource is None or tokenizer_resource[0].get("role") != "tokenizer":
        raise ArtifactConfigurationError("source manifest model.tokenizerRef names no tokenizer")
    maximum_sequence_length = _positive_integer(
        tokenizer_resource[0].get("maximumSequenceLength"), "tokenizer maximumSequenceLength"
    )
    encoder_refs = [ref for ref, (resource, _) in sources.items() if resource["role"] == "encoder"]
    for ref in encoder_refs:
        onnx = cast(dict[str, Any], sources[ref][0]["onnx"])
        if onnx.get("precision", "float32") != "float32":
            raise ArtifactConfigurationError("source release encoder is already quantized")
    compatibility = manifest.get("compatibility")
    encoding = compatibility.get("canonicalInput") if isinstance(compatibility, dict) else None
    input_version = canonical_input_version(encoding)
    if input_version is None:
        raise ArtifactConfigurationError("source manifest declares an unsupported canonical input")
    _assign_records(checked_records, chains)

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
        paths = {
            ref: staging_io_root / cast(str, resource["path"])
            for ref, (resource, _) in sources.items()
        }
        tokenizer_json = b""
        for ref, (resource, path) in sources.items():
            if resource["role"] == "encoder":
                continue
            data = _read_resource(path, resolved.maximum_resource_bytes)
            if ref == tokenizer_ref:
                tokenizer_json = data
            _write_exclusive(paths[ref], data)
        components = {
            ref: _quantize_encoder(sources[ref][1], paths[ref], settings, resolved)
            for ref in encoder_refs
        }
        report = _verify_quantized_chain(
            source_encoders={ref: sources[ref][1] for ref in encoder_refs},
            quantized_encoders={ref: paths[ref] for ref in encoder_refs},
            resource_paths=paths,
            chains=chains,
            tokenizer_json=tokenizer_json,
            input_version=input_version,
            maximum_sequence_length=maximum_sequence_length,
            records=checked_records,
            settings=settings,
            source_manifest_sha256=source_manifest_sha256,
            components=[components[ref] for ref in encoder_refs],
            started=started,
        )
        verification = (
            None if sources_document is None else report.to_manifest_verification(sources_document)
        )
        resources = _derived_resources(
            manifest, paths, settings, source_manifest_sha256, resolved, verification
        )
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


@dataclass(frozen=True, slots=True)
class _Head:
    ref: str
    width: int
    temperature: float


@dataclass(frozen=True, slots=True)
class _Chain:
    """One function's path through the release: its encoder, adapter and heads."""

    function_id: str
    input_schema: list[Any]
    encoder_ref: str
    adapter_ref: str
    heads: tuple[_Head, ...]


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
    if not any(any(label is not None for label in record.labels()) for record in checked):
        raise ArtifactConfigurationError(
            "quantization verification needs at least one labeled record for calibration"
        )
    return checked


_RECORD_SOURCE_KINDS = ("training-dataset", "adversarial-dataset", "held-out")


def _validate_record_sources(
    record_sources: Sequence[dict[str, JsonValue]] | None,
) -> list[dict[str, JsonValue]] | None:
    if record_sources is None:
        return None
    if isinstance(record_sources, (str, bytes)) or not isinstance(record_sources, Sequence):
        raise ArtifactConfigurationError("record_sources must be a sequence of objects")
    checked: list[dict[str, JsonValue]] = []
    for index, source in enumerate(record_sources):
        if (
            not isinstance(source, dict)
            or set(source) != {"kind", "functionId", "sha256", "records"}
            or source.get("kind") not in _RECORD_SOURCE_KINDS
            or not isinstance(source.get("functionId"), str)
            or not isinstance(source.get("sha256"), str)
            or _SHA256.fullmatch(cast(str, source["sha256"])) is None
            or not _is_label(source.get("records"))
        ):
            raise ArtifactConfigurationError(
                f"record_sources[{index}] must be {{kind, functionId, sha256, records}} with "
                f"kind one of {', '.join(_RECORD_SOURCE_KINDS)}"
            )
        checked.append(dict(source))
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
    if (
        not isinstance(functions, list)
        or not functions
        or not all(isinstance(function, dict) for function in functions)
    ):
        raise ArtifactConfigurationError("source manifest must contain at least one function")
    return manifest


def _source_resources(
    manifest: dict[str, JsonValue], release: Path, config: ArtifactExportConfig
) -> dict[str, tuple[dict[str, Any], Path]]:
    """Every resource by ref, each integrity-checked against the manifest."""

    resources = manifest.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ArtifactConfigurationError("source manifest resources must be a non-empty array")
    by_ref: dict[str, tuple[dict[str, Any], Path]] = {}
    for resource in resources:
        if not isinstance(resource, dict):
            raise ArtifactConfigurationError("source manifest resources must be objects")
        ref = resource.get("ref")
        role = resource.get("role")
        relative = resource.get("path")
        if (
            not isinstance(ref, str)
            or ref in by_ref
            or role not in _RESOURCE_ROLES
            or not isinstance(relative, str)
        ):
            raise ArtifactConfigurationError("source manifest resource roles are invalid")
        path = _safe_existing_release_file(release, relative)
        byte_length, digest = _file_identity(path, config.maximum_resource_bytes)
        if byte_length != resource.get("byteLength") or digest != resource.get("sha256"):
            raise ArtifactConfigurationError(f"source release resource {relative} is not intact")
        if role != "tokenizer" and not isinstance(resource.get("onnx"), dict):
            raise ArtifactConfigurationError(
                f"source release resource {relative} lacks ONNX metadata"
            )
        by_ref[ref] = (resource, path)
    roles = {resource["role"] for resource, _ in by_ref.values()}
    if roles != set(_RESOURCE_ROLES):
        raise ArtifactConfigurationError("source manifest must cover every resource role once")
    if sum(1 for resource, _ in by_ref.values() if resource["role"] == "tokenizer") != 1:
        raise ArtifactConfigurationError("source manifest must contain exactly one tokenizer")
    return by_ref


def _model_ref(manifest: dict[str, JsonValue], key: str) -> str:
    model = manifest.get("model")
    ref = model.get(key) if isinstance(model, dict) else None
    if not isinstance(ref, str):
        raise ArtifactConfigurationError(f"source manifest model.{key} is missing")
    return ref


def _function_chains(
    manifest: dict[str, JsonValue], sources: dict[str, tuple[dict[str, Any], Path]]
) -> dict[str, _Chain]:
    model_encoder = _model_ref(manifest, "encoderRef")
    model_adapter = _model_ref(manifest, "adapterRef")

    def resource(ref: object, role: str, label: str) -> dict[str, Any]:
        entry = sources.get(ref) if isinstance(ref, str) else None
        if entry is None or entry[0]["role"] != role:
            raise ArtifactConfigurationError(f"{label} names no {role} resource")
        return entry[0]

    chains: dict[str, _Chain] = {}
    for function in cast(list[dict[str, Any]], manifest["functions"]):
        function_id = function.get("id")
        if not isinstance(function_id, str) or function_id in chains:
            raise ArtifactConfigurationError("source manifest function ids are invalid")
        encoder_ref = function.get("encoderRef", model_encoder)
        adapter_ref = function.get("adapterRef", model_adapter)
        resource(encoder_ref, "encoder", f"function {function_id} encoderRef")
        resource(adapter_ref, "adapter", f"function {function_id} adapterRef")
        heads = function.get("heads")
        if not isinstance(heads, list) or not heads:
            raise ArtifactConfigurationError(f"function {function_id} must have at least one head")
        chain_heads: list[_Head] = []
        for head in heads:
            if not isinstance(head, dict):
                raise ArtifactConfigurationError(f"function {function_id} heads must be objects")
            head_resource = resource(head.get("headRef"), "head", f"function {function_id} head")
            outputs = cast(list[Any], cast(dict[str, Any], head_resource["onnx"])["outputs"])
            chain_heads.append(
                _Head(
                    ref=cast(str, head["headRef"]),
                    width=_dimension(cast(dict[str, Any], outputs[0])["shape"][1]),
                    temperature=_temperature(head),
                )
            )
        inputs = function.get("inputs")
        if not isinstance(inputs, list):
            raise ArtifactConfigurationError(f"function {function_id} inputs must be an array")
        chains[function_id] = _Chain(
            function_id=function_id,
            input_schema=inputs,
            encoder_ref=cast(str, encoder_ref),
            adapter_ref=cast(str, adapter_ref),
            heads=tuple(chain_heads),
        )
    return chains


def _assign_records(records: tuple[QuantizationRecord, ...], chains: dict[str, _Chain]) -> None:
    """Every record names a function of the release (implicit for one) and fits its heads."""

    only = next(iter(chains)) if len(chains) == 1 else None
    for index, record in enumerate(records):
        function_id = record.function_id or only
        if function_id is None:
            raise ArtifactConfigurationError(
                f"quantization record {index} needs a function_id: the release has "
                f"{len(chains)} functions"
            )
        chain = chains.get(function_id)
        if chain is None:
            raise ArtifactConfigurationError(
                f"quantization record {index} names function {function_id}, which the "
                "release does not contain"
            )
        labels = record.labels()
        if len(labels) != len(chain.heads):
            raise ArtifactConfigurationError(
                f"quantization record {index} has {len(labels)} label(s) for "
                f"{len(chain.heads)} head(s)"
            )
        for label, head in zip(labels, chain.heads, strict=True):
            if label is not None and label >= max(head.width, 2):
                raise ArtifactConfigurationError(
                    "quantization record label_index exceeds the head support"
                )


def _quantize_encoder(
    source: Path, destination: Path, settings: QuantizationConfig, config: ArtifactExportConfig
) -> Any:
    try:
        from semantscript_model.export import quantize_onnx_encoder
    except ImportError as error:
        raise ArtifactConfigurationError(
            "artifact quantization requires the optional ONNX training dependencies"
        ) from error
    root_logger = logging.getLogger()
    root_logger.addFilter(_quiet_preprocessing_advice)
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
    finally:
        root_logger.removeFilter(_quiet_preprocessing_advice)


def _quiet_preprocessing_advice(record: logging.LogRecord) -> bool:
    """Drop onnxruntime's advice to run its static-quantization preprocessing first.

    Dynamic quantization of an exported encoder does not need it, and the gate
    below is what decides whether the quantized graph is acceptable.
    """

    return not record.getMessage().startswith("Please consider to run pre-processing")


class _Tally:
    """Decision and calibration counts for one function."""

    def __init__(self, chain: _Chain) -> None:
        self.chain = chain
        self.records = 0
        self.labeled = 0
        self.attested = 0
        self.disagreements = 0
        self.attested_disagreements = 0
        self.source_logits: list[list[Any]] = [[] for _ in chain.heads]
        self.quantized_logits: list[list[Any]] = [[] for _ in chain.heads]
        self.labels: list[list[int]] = [[] for _ in chain.heads]


def _verify_quantized_chain(
    *,
    source_encoders: dict[str, Path],
    quantized_encoders: dict[str, Path],
    resource_paths: dict[str, Path],
    chains: dict[str, _Chain],
    tokenizer_json: bytes,
    input_version: int,
    maximum_sequence_length: int,
    records: tuple[QuantizationRecord, ...],
    settings: QuantizationConfig,
    source_manifest_sha256: str,
    components: Sequence[Any],
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

    used_encoders = {chain.encoder_ref for chain in chains.values()}
    try:
        tokenizer = Tokenizer.from_str(tokenizer_json.decode("utf-8"))
        tokenizer.enable_truncation(maximum_sequence_length)
        source_sessions = {ref: session(source_encoders[ref]) for ref in used_encoders}
        quantized_sessions = {ref: session(quantized_encoders[ref]) for ref in used_encoders}
        adapter_sessions = {
            ref: session(resource_paths[ref]) for ref in {c.adapter_ref for c in chains.values()}
        }
        head_sessions = {
            head.ref: session(resource_paths[head.ref])
            for chain in chains.values()
            for head in chain.heads
        }
    except Exception as error:
        raise ArtifactConfigurationError(
            f"quantized-graph verification could not start: {error}"
        ) from error

    def chain_logits(encoder_session: Any, chain: _Chain, ids: Any, mask: Any) -> list[Any]:
        embedding = encoder_session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        function_embedding = adapter_sessions[chain.adapter_ref].run(
            None, {"sentence_embedding": embedding}
        )[0]
        outputs: list[Any] = []
        for head in chain.heads:
            logits = head_sessions[head.ref].run(None, {"function_embedding": function_embedding})[
                0
            ]
            if logits.shape != (1, head.width) or not bool(numpy.isfinite(logits).all()):
                raise ArtifactConfigurationError(
                    "chain produced logits of the wrong shape or non-finite"
                )
            outputs.append(logits[0])
        return outputs

    def decision(logits: Any) -> int:
        # A single-logit head is a sigmoid: the decision is whether it is positive.
        return int(logits[0] > 0) if logits.shape[0] == 1 else int(logits.argmax())

    only = next(iter(chains)) if len(chains) == 1 else None
    tallies = {function_id: _Tally(chain) for function_id, chain in chains.items()}
    for index, record in enumerate(records):
        tally = tallies[cast(str, record.function_id or only)]
        chain = tally.chain
        try:
            text = serialize_canonical_inputs(
                chain.input_schema, record.inputs, version=input_version
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
        source = chain_logits(source_sessions[chain.encoder_ref], chain, ids, mask)
        quantized = chain_logits(quantized_sessions[chain.encoder_ref], chain, ids, mask)
        tally.records += 1
        changed = any(
            decision(left) != decision(right) for left, right in zip(source, quantized, strict=True)
        )
        if changed:
            tally.disagreements += 1
            if record.attested:
                tally.attested_disagreements += 1
        if record.attested:
            tally.attested += 1
        labels = record.labels()
        if any(label is not None for label in labels):
            tally.labeled += 1
        for position, label in enumerate(labels):
            if label is not None:
                tally.source_logits[position].append(source[position])
                tally.quantized_logits[position].append(quantized[position])
                tally.labels[position].append(label)

    function_reports: list[QuantizationFunctionReport] = []
    head_eces: list[tuple[str, float, float]] = []
    correct_source = 0
    correct_quantized = 0
    labeled_decisions = 0
    for function_id, tally in tallies.items():
        function_source_ece: float | None = None
        function_quantized_ece: float | None = None
        function_correct_source = 0
        function_correct_quantized = 0
        function_decisions = 0
        for position, head in enumerate(tally.chain.heads):
            labels = tally.labels[position]
            if not labels:
                continue
            targets = torch.tensor(labels, dtype=torch.int64)
            source_metrics = calibration_metrics(
                torch.tensor(numpy.stack(tally.source_logits[position]), dtype=torch.float32),
                targets,
                temperature=head.temperature,
                bin_count=settings.ece_bins,
            )
            quantized_metrics = calibration_metrics(
                torch.tensor(numpy.stack(tally.quantized_logits[position]), dtype=torch.float32),
                targets,
                temperature=head.temperature,
                bin_count=settings.ece_bins,
            )
            head_eces.append((function_id, float(source_metrics.ece), float(quantized_metrics.ece)))
            function_source_ece = max(function_source_ece or 0.0, float(source_metrics.ece))
            function_quantized_ece = max(
                function_quantized_ece or 0.0, float(quantized_metrics.ece)
            )
            function_correct_source += round(float(source_metrics.accuracy) * len(labels))
            function_correct_quantized += round(float(quantized_metrics.accuracy) * len(labels))
            function_decisions += len(labels)
        correct_source += function_correct_source
        correct_quantized += function_correct_quantized
        labeled_decisions += function_decisions
        function_reports.append(
            QuantizationFunctionReport(
                function_id=function_id,
                records_checked=tally.records,
                labeled_records=tally.labeled,
                attested_records=tally.attested,
                argmax_disagreements=tally.disagreements,
                attested_disagreements=tally.attested_disagreements,
                source_accuracy=(
                    function_correct_source / function_decisions if function_decisions else None
                ),
                quantized_accuracy=(
                    function_correct_quantized / function_decisions if function_decisions else None
                ),
                source_ece=function_source_ece,
                quantized_ece=function_quantized_ece,
            )
        )

    disagreements = sum(tally.disagreements for tally in tallies.values())
    attested_disagreements = sum(tally.attested_disagreements for tally in tallies.values())
    first = components[0]
    first_chain = next(iter(chains.values()))
    report = QuantizationReport(
        source_manifest_sha256=source_manifest_sha256,
        method=first.method,
        weight_type=first.weight_type,
        per_channel=first.per_channel,
        reduce_range=first.reduce_range,
        records_checked=len(records),
        labeled_records=sum(tally.labeled for tally in tallies.values()),
        attested_records=sum(tally.attested for tally in tallies.values()),
        argmax_disagreements=disagreements,
        attested_disagreements=attested_disagreements,
        argmax_disagreement_rate=disagreements / len(records),
        source_accuracy=correct_source / labeled_decisions,
        quantized_accuracy=correct_quantized / labeled_decisions,
        source_ece=max(ece for _, ece, _ in head_eces),
        quantized_ece=max(ece for _, _, ece in head_eces),
        temperature=first_chain.heads[0].temperature,
        ece_bins=settings.ece_bins,
        source_encoder_byte_length=sum(component.source_byte_length for component in components),
        quantized_encoder_byte_length=sum(component.byte_length for component in components),
        quantized_matmul_count=sum(component.quantized_matmul_count for component in components),
        elapsed_seconds=round(time.monotonic() - started, 3),
        functions=tuple(function_reports),
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
    for function_id, _, quantized_ece in head_eces:
        if quantized_ece > settings.ece_threshold:
            where = "" if len(chains) == 1 else f" for {function_id}"
            failures.append(
                f"quantized ECE {quantized_ece:.6f}{where} exceeds threshold "
                f"{settings.ece_threshold}"
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
    verification: dict[str, JsonValue] | None = None,
) -> list[dict[str, JsonValue]]:
    resources: list[dict[str, JsonValue]] = []
    for original in cast(list[Any], manifest["resources"]):
        resource = cast(dict[str, JsonValue], deepcopy(original))
        restore_integral_numbers(resource)
        role = cast(str, resource["role"])
        byte_length, digest = _file_identity(
            paths[cast(str, resource["ref"])], config.maximum_resource_bytes
        )
        if role == "encoder":
            onnx = cast(dict[str, JsonValue], resource["onnx"])
            onnx["precision"] = QUANTIZED_PRECISION
            document = settings.to_manifest_document(source_manifest_sha256)
            if verification is not None:
                document["verification"] = cast(JsonValue, deepcopy(verification))
            onnx["quantization"] = cast(JsonValue, document)
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


def _temperature(head: dict[str, Any]) -> float:
    calibration = head.get("calibration")
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
    "QuantizationFunctionReport",
    "QuantizationGateError",
    "QuantizationRecord",
    "QuantizationReport",
    "QuantizedArtifact",
    "quantize_release_artifact",
]
