"""Verified scalar-model export and atomic application-artifact publication."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
from collections.abc import Sequence
from copy import deepcopy
from ctypes import CDLL, c_char_p, c_int, c_uint, get_errno
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path
from struct import pack
from typing import Any, cast

from semantscript_trainer.canonical_input import (
    CANONICAL_INPUT_ENCODINGS,
    CANONICAL_INPUT_V1,
    canonical_input_version,
)
from semantscript_trainer.case_contract import validate_case
from semantscript_trainer.constraints import ConstraintConfigurationError, compile_constraints
from semantscript_trainer.semantic_json import semantic_json_bytes, semantic_json_sha256
from semantscript_trainer.strict_json import StrictJsonError, StrictJsonLimits, loads_strict_json
from semantscript_trainer.teacher import GeneratedCase, JsonValue, NeuralFunctionIr
from semantscript_trainer.training import TrainingResult
from semantscript_trainer.training_contract import (
    TrainingContractError,
    TrainingHeadContract,
    derive_output_heads,
)
from semantscript_trainer.verification import (
    VerificationResult,
    model_state_sha256,
    require_passing_verification,
)

ARTIFACT_VERSION = 1
MODEL_ABI_VERSION = 1
RUNTIME_ABI_VERSION = 1
ONNX_OPSET_VERSION = 17
MAXIMUM_MANIFEST_BYTES = 8 * 1024 * 1024
MAXIMUM_TOKENIZER_BYTES = 64 * 1024 * 1024
MAXIMUM_RESOURCE_BYTES = 1024 * 1024 * 1024
MAXIMUM_AGGREGATE_RESOURCE_BYTES = 2 * 1024 * 1024 * 1024
MAXIMUM_PARITY_RELATIVE_TOLERANCE = 1e-3
MAXIMUM_PARITY_ABSOLUTE_TOLERANCE = 1e-3

_APPLICATION_ID = re.compile(r"^[a-z][a-z0-9._-]{1,127}$")
_FUNCTION_ID = re.compile(r"^nf_[a-f0-9]{64}$")
_LOGICAL_REF = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$")
_SEMVER = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"\.(?:0|[1-9][0-9]*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_CANONICAL_DECIMAL = re.compile(r"^(?!-0$)-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?$")
_RFC3339_UTC = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:"
    r"[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)
_RFC3339 = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:"
    r"[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)

_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_ANCHORED_DIRECTORY_OPERATIONS = os.name == "posix" and all(
    operation in os.supports_dir_fd
    for operation in (os.open, os.mkdir, os.rename, os.stat, os.unlink)
)
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004


class ArtifactExportError(RuntimeError):
    """Base class for artifact export and publication failures."""


class ArtifactConfigurationError(ArtifactExportError, ValueError):
    """The export request is incomplete, inconsistent, or unsafe."""


class ArtifactPublicationError(ArtifactExportError):
    """A validated artifact could not be published atomically."""


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Explicit application/build identities not retained by training."""

    application_id: str
    application_version: str
    compiler_version: str
    trainer_version: str
    created_at: str
    training_key_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.application_id, str)
            or _APPLICATION_ID.fullmatch(self.application_id) is None
        ):
            raise ArtifactConfigurationError("application_id is invalid")
        for name in ("application_version", "compiler_version", "trainer_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SEMVER.fullmatch(value) is None:
                raise ArtifactConfigurationError(f"{name} must be a semantic version")
        if not isinstance(self.created_at, str) or _RFC3339_UTC.fullmatch(self.created_at) is None:
            raise ArtifactConfigurationError("created_at must be an RFC 3339 UTC timestamp")
        try:
            parsed_created_at = datetime.fromisoformat(self.created_at[:-1] + "+00:00")
        except ValueError as error:
            raise ArtifactConfigurationError(
                "created_at must be an RFC 3339 UTC timestamp"
            ) from error
        if parsed_created_at.tzinfo != UTC:
            raise ArtifactConfigurationError("created_at must be an RFC 3339 UTC timestamp")
        _require_sha256("training_key_sha256", self.training_key_sha256)


@dataclass(frozen=True, slots=True)
class ArtifactExportConfig:
    """Resource, parity, and pointer controls for one bounded export."""

    publish_pointer: bool = True
    parity_relative_tolerance: float = 1e-4
    parity_absolute_tolerance: float = 1e-5
    maximum_manifest_bytes: int = MAXIMUM_MANIFEST_BYTES
    maximum_tokenizer_bytes: int = MAXIMUM_TOKENIZER_BYTES
    maximum_resource_bytes: int = MAXIMUM_RESOURCE_BYTES
    maximum_aggregate_resource_bytes: int = MAXIMUM_AGGREGATE_RESOURCE_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.publish_pointer, bool):
            raise ArtifactConfigurationError("publish_pointer must be a boolean")
        parity_limits = {
            "parity_relative_tolerance": MAXIMUM_PARITY_RELATIVE_TOLERANCE,
            "parity_absolute_tolerance": MAXIMUM_PARITY_ABSOLUTE_TOLERANCE,
        }
        for name, maximum in parity_limits.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ArtifactConfigurationError(
                    f"{name} must be a finite number between 0 and {maximum}"
                )
            if not math.isfinite(float(value)) or not 0 <= value <= maximum:
                raise ArtifactConfigurationError(
                    f"{name} must be a finite number between 0 and {maximum}"
                )
        for name in (
            "maximum_manifest_bytes",
            "maximum_tokenizer_bytes",
            "maximum_resource_bytes",
            "maximum_aggregate_resource_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ArtifactConfigurationError(f"{name} must be a positive integer")
        if self.maximum_tokenizer_bytes > self.maximum_resource_bytes:
            raise ArtifactConfigurationError(
                "maximum_tokenizer_bytes cannot exceed maximum_resource_bytes"
            )
        if self.maximum_resource_bytes > self.maximum_aggregate_resource_bytes:
            raise ArtifactConfigurationError(
                "maximum_resource_bytes cannot exceed maximum_aggregate_resource_bytes"
            )
        hard_limits = {
            "maximum_manifest_bytes": MAXIMUM_MANIFEST_BYTES,
            "maximum_tokenizer_bytes": MAXIMUM_TOKENIZER_BYTES,
            "maximum_resource_bytes": MAXIMUM_RESOURCE_BYTES,
            "maximum_aggregate_resource_bytes": MAXIMUM_AGGREGATE_RESOURCE_BYTES,
        }
        for name, hard_limit in hard_limits.items():
            if getattr(self, name) > hard_limit:
                raise ArtifactConfigurationError(
                    f"{name} cannot exceed the runtime limit {hard_limit}"
                )


@dataclass(frozen=True, slots=True)
class ExportedArtifact:
    """Paths and exact identity of one immutable published release."""

    artifact_root: Path
    release_directory: Path
    manifest_path: Path
    manifest_sha256: str
    pointer_path: Path | None
    manifest: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ArtifactFunction:
    """One verified function of an application: its IR, training view and evidence."""

    ir: NeuralFunctionIr
    training: TrainingResult
    verification: VerificationResult
    source_ir_bytes: bytes


def export_application_artifact(
    artifact_root: str | os.PathLike[str],
    ir: NeuralFunctionIr,
    training: TrainingResult,
    verification: VerificationResult,
    /,
    *,
    tokenizer_json: bytes,
    source_ir_bytes: bytes,
    provenance: ArtifactProvenance,
    input_ids: Any,
    attention_mask: Any,
    config: ArtifactExportConfig | None = None,
) -> ExportedArtifact:
    """Export, validate, and atomically publish one verified scalar function."""

    return export_multi_function_artifact(
        artifact_root,
        [ArtifactFunction(ir, training, verification, source_ir_bytes)],
        tokenizer_json=tokenizer_json,
        provenance=provenance,
        input_ids=input_ids,
        attention_mask=attention_mask,
        config=config,
    )


def export_multi_function_artifact(
    artifact_root: str | os.PathLike[str],
    functions: Sequence[ArtifactFunction],
    /,
    *,
    tokenizer_json: bytes,
    provenance: ArtifactProvenance,
    input_ids: Any,
    attention_mask: Any,
    config: ArtifactExportConfig | None = None,
) -> ExportedArtifact:
    """Export, validate, and atomically publish every verified function of one application.

    All functions must share one tokenizer, one encoder module, one adapter
    module (absent means the identity adapter), one canonical input encoding
    and the application-level encoder and adapter refs; each brings its own
    head, IR, verification and source IR bytes. The manifest lists every
    function and one head resource per function.
    """

    resolved = ArtifactExportConfig() if config is None else config
    if not isinstance(resolved, ArtifactExportConfig):
        raise ArtifactConfigurationError("config must be an ArtifactExportConfig")
    if not isinstance(provenance, ArtifactProvenance):
        raise ArtifactConfigurationError("provenance must be an ArtifactProvenance")
    entries = _validate_functions(functions)
    tokenizer_bytes = _validate_tokenizer(tokenizer_json, entries[0].verification, resolved)
    for entry in entries[1:]:
        if entry.verification.tokenizer_sha256 != entries[0].verification.tokenizer_sha256:
            raise ArtifactConfigurationError("every function must be verified with one tokenizer")
    routing = _shared_modules(entries)
    encoding = _canonical_input_encoding(entries[0].ir, entries[0].training)
    for entry in entries[1:]:
        if _canonical_input_encoding(entry.ir, entry.training) != encoding:
            raise ArtifactConfigurationError(
                "every function must share one canonical input encoding"
            )
    function_heads = [(cast(str, entry.ir["id"]), len(entry.training.heads)) for entry in entries]

    root, root_descriptor = _prepare_artifact_root(artifact_root)
    releases: Path | None = None
    releases_descriptor: int | None = None
    staging: Path | None = None
    staging_descriptor: int | None = None
    pointer_path: Path | None = None
    try:
        releases, releases_descriptor = _prepare_child_directory(root, "releases", root_descriptor)
        staging, staging_descriptor = _create_staging_directory(releases, releases_descriptor)
        _require_directory_identity(root, root_descriptor, "artifact_root")
        _require_directory_identity(releases, releases_descriptor, "artifact releases")
        _require_directory_identity(staging, staging_descriptor, "artifact staging")
        staging_io_root = _descriptor_directory_path(staging, staging_descriptor)
        paths = _application_resource_paths(
            staging_io_root,
            function_heads,
            encoder_keys=tuple(routing.encoder_refs),
            adapter_refs=tuple(routing.adapters),
        )
        _write_exclusive(paths["tokenizer"], tokenizer_bytes)
        _export_application_onnx(entries, routing, input_ids, attention_mask, paths, resolved)
        for entry in entries:
            if model_state_sha256(entry.training.model) != entry.verification.model_state_sha256:
                raise ArtifactConfigurationError(
                    "trained model state changed during export; run verification again"
                )
        resources = _resource_documents(entries, paths, resolved, routing)
        manifest = _manifest_document(entries, provenance, encoding, resources, routing)
        _validate_manifest_document(manifest)
        manifest_bytes = _json_bytes(manifest)
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
        release = releases / f"sha256-{manifest_sha256}"
        release = _publish_release(
            staging,
            release,
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
        return ExportedArtifact(
            artifact_root=root,
            release_directory=release,
            manifest_path=release / "manifest.json",
            manifest_sha256=manifest_sha256,
            pointer_path=pointer_path,
            manifest=manifest,
        )
    except ArtifactExportError:
        if staging is not None and releases is not None:
            _remove_owned_staging(staging, releases, releases_descriptor, staging_descriptor)
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        if staging is not None and releases is not None:
            _remove_owned_staging(staging, releases, releases_descriptor, staging_descriptor)
        raise ArtifactPublicationError(f"artifact export failed: {error}") from error
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if releases_descriptor is not None:
            os.close(releases_descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)


def _validate_functions(functions: Sequence[ArtifactFunction]) -> tuple[ArtifactFunction, ...]:
    if isinstance(functions, (str, bytes)) or not isinstance(functions, Sequence) or not functions:
        raise ArtifactConfigurationError(
            "functions must be a nonempty sequence of ArtifactFunction"
        )
    entries = tuple(functions)
    seen_ids: set[str] = set()
    seen_heads: set[str] = set()
    for entry in entries:
        if not isinstance(entry, ArtifactFunction):
            raise ArtifactConfigurationError("functions must be ArtifactFunction instances")
        if not isinstance(entry.training, TrainingResult):
            raise ArtifactConfigurationError("training must be a TrainingResult")
        if not isinstance(entry.verification, VerificationResult):
            raise ArtifactConfigurationError("verification must be a VerificationResult")
        require_passing_verification(entry.verification)
        validate_verified_ir_binding(
            entry.ir, entry.source_ir_bytes, entry.training, entry.verification
        )
        _validate_model_contract(entry.training)
        if model_state_sha256(entry.training.model) != entry.verification.model_state_sha256:
            raise ArtifactConfigurationError(
                "trained model state changed after verification; run verification again"
            )
        function_id = cast(str, entry.ir["id"])
        if function_id in seen_ids:
            raise ArtifactConfigurationError(f"function {function_id} appears more than once")
        seen_ids.add(function_id)
        model = cast(dict[str, JsonValue], entry.ir["model"])
        for name in ("encoder", "adapter"):
            _require_logical_ref(f"IR model.{name}", model.get(name))
        depth = model.get("encoderDepth")
        if depth is not None and (
            isinstance(depth, bool) or not isinstance(depth, int) or depth < 1
        ):
            raise ArtifactConfigurationError("IR model.encoderDepth must be a positive integer")
        for binding in cast(list[Any], model["heads"]):
            head_ref = cast(str, cast(dict[str, JsonValue], binding)["ref"])
            if head_ref in seen_heads:
                raise ArtifactConfigurationError(
                    f"head ref {head_ref} is bound by more than one head"
                )
            seen_heads.add(head_ref)
    return entries


@dataclass(frozen=True, slots=True)
class _Routing:
    """The application's shared modules as the functions bind them."""

    encoder: Any
    adapters: dict[str, Any]
    adapter_depths: dict[str, int | None]
    encoder_refs: dict[str, str]  # depth key -> IR encoder ref
    function_adapters: dict[str, str]

    def encoder_key_of(self, function_id: str) -> str:
        return _depth_key(self.adapter_depths[self.function_adapters[function_id]])


def _shared_modules(entries: Sequence[ArtifactFunction]) -> _Routing:
    encoder = entries[0].training.model.encoder
    adapters: dict[str, Any] = {}
    depths: dict[str, int | None] = {}
    encoder_refs: dict[str, str] = {}
    function_adapters: dict[str, str] = {}
    for entry in entries:
        if entry.training.model.encoder is not encoder:
            raise ArtifactConfigurationError("every function must share one encoder module")
        model = cast(dict[str, JsonValue], entry.ir["model"])
        adapter_ref = cast(str, model["adapter"])
        # A classifier without an adapter module exports the identity adapter.
        adapter = getattr(entry.training.model, "adapter", None)
        depth = cast(int | None, getattr(entry.training.model, "depth", None))
        declared = model.get("encoderDepth")
        normalized = encoder.validate_depth(cast(int | None, declared))
        if normalized != depth:
            raise ArtifactConfigurationError(
                f"function {model['adapter']} declares encoderDepth {declared!r} "
                f"but trained at {depth!r}"
            )
        if adapter_ref in adapters:
            if adapters[adapter_ref] is not adapter:
                raise ArtifactConfigurationError(
                    f"functions of domain {adapter_ref} must share one adapter module"
                )
            if depths[adapter_ref] != depth:
                raise ArtifactConfigurationError(
                    f"functions of domain {adapter_ref} must share one encoder depth"
                )
        else:
            if adapter is not None and any(existing is adapter for existing in adapters.values()):
                raise ArtifactConfigurationError(
                    f"one adapter module is bound under more than one adapter ref ({adapter_ref})"
                )
            adapters[adapter_ref] = adapter
            depths[adapter_ref] = depth
        key = _depth_key(depth)
        encoder_ref = cast(str, model["encoder"])
        if encoder_refs.setdefault(key, encoder_ref) != encoder_ref:
            raise ArtifactConfigurationError(
                f"functions at encoder depth {key} must share one encoder ref"
            )
        function_adapters[cast(str, entry.ir["id"])] = adapter_ref
    if len(set(encoder_refs.values())) != len(encoder_refs):
        raise ArtifactConfigurationError("each encoder depth needs its own encoder ref")
    return _Routing(encoder, adapters, depths, encoder_refs, function_adapters)


def _depth_key(depth: int | None) -> str:
    return "full" if depth is None else f"depth-{depth:03d}"


def _safe_ref(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", ref)


def validate_verified_ir_binding(
    ir: NeuralFunctionIr,
    source_ir_bytes: bytes,
    training: TrainingResult,
    verification: VerificationResult,
) -> None:
    """Validate exact verified IR bytes against their training and verification evidence."""

    if not isinstance(training, TrainingResult):
        raise ArtifactConfigurationError("training must be a TrainingResult")
    if not isinstance(verification, VerificationResult):
        raise ArtifactConfigurationError("verification must be a VerificationResult")
    require_passing_verification(verification)
    if not isinstance(ir, dict):
        raise ArtifactConfigurationError("IR must be an object")
    if not isinstance(source_ir_bytes, bytes) or not source_ir_bytes:
        raise ArtifactConfigurationError("source_ir_bytes must be nonempty bytes")
    try:
        parsed = loads_strict_json(source_ir_bytes)
    except StrictJsonError as error:
        raise ArtifactConfigurationError(f"source IR is invalid: {error}") from error
    if not _semantic_equal(parsed, ir):
        raise ArtifactConfigurationError("source_ir_bytes do not encode the supplied IR")
    ir_version = ir.get("irVersion")
    if (
        ir.get("kind") != "semantscript.neural-function"
        or isinstance(ir_version, bool)
        or not isinstance(ir_version, int)
        or ir_version != 1
    ):
        raise ArtifactConfigurationError("only neural-function IR v1 can be exported")
    if ir.get("stage") != "verified":
        raise ArtifactConfigurationError("artifact export requires verified-stage IR")
    _validate_closed_ir_shape(ir)
    function_id = ir.get("id")
    semantic_sha256 = ir.get("semanticSha256")
    if not isinstance(function_id, str) or _FUNCTION_ID.fullmatch(function_id) is None:
        raise ArtifactConfigurationError("IR function ID is invalid")
    _require_sha256("IR semanticSha256", semantic_sha256)
    semantic_projection = {
        "irVersion": ir.get("irVersion"),
        "definition": ir.get("definition"),
        "inputs": ir.get("inputs"),
        "output": ir.get("output"),
        "runtime": ir.get("runtime"),
    }
    try:
        computed_semantic_sha256 = semantic_json_sha256(cast(JsonValue, semantic_projection))
    except (TypeError, ValueError, OverflowError) as error:
        raise ArtifactConfigurationError(f"IR semantic projection is invalid: {error}") from error
    if computed_semantic_sha256 != semantic_sha256:
        raise ArtifactConfigurationError("IR semanticSha256 does not match its semantic fields")
    identities = (
        training.function_id,
        verification.function_id,
        function_id,
        training.semantic_sha256,
        verification.semantic_sha256,
        semantic_sha256,
    )
    if identities[0] != identities[1] or identities[1] != identities[2]:
        raise ArtifactConfigurationError("IR, training, and verification function IDs differ")
    if identities[3] != identities[4] or identities[4] != identities[5]:
        raise ArtifactConfigurationError("IR, training, and verification semantic digests differ")
    if not _semantic_equal(ir.get("verification"), verification.to_ir_document()):
        raise ArtifactConfigurationError(
            "verified IR does not contain the supplied verification evidence"
        )
    try:
        ir_heads = derive_output_heads(ir)
    except TrainingContractError as error:
        raise ArtifactConfigurationError(f"IR training contract is invalid: {error}") from error
    if ir_heads != training.heads:
        raise ArtifactConfigurationError("IR output heads differ from the trained heads")
    _validate_manifest_inputs(ir.get("inputs"))
    for head_spec, item in zip(_ir_head_specs(ir), training.heads, strict=True):
        _validate_manifest_head_type(
            _runtime_head_type_for(head_spec, item.contract),
            item.contract.parameterization,
        )
    _runtime_policy(ir)
    definition = cast(dict[str, JsonValue], ir["definition"])
    try:
        for example in cast(list[dict[str, JsonValue]], definition["examples"]):
            validate_case(
                ir,
                GeneratedCase(
                    inputs=cast(dict[str, JsonValue], example["inputs"]),
                    output=example["output"],
                ),
            )
        compile_constraints(ir)
    except (ConstraintConfigurationError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactConfigurationError(f"IR definition contract is invalid: {error}") from error
    model = ir.get("model")
    if not isinstance(model, dict):
        raise ArtifactConfigurationError("IR model binding must be an object")
    for name in ("encoder", "adapter"):
        _require_logical_ref(f"IR model {name}", model.get(name))
    heads = model.get("heads")
    if (
        not isinstance(heads, list)
        or len(heads) != len(training.heads)
        or any(not isinstance(head, dict) for head in heads)
    ):
        raise ArtifactConfigurationError("IR model heads must bind every trained output head")
    for head, item in zip(heads, training.heads, strict=True):
        if head.get("outputPath") != item.output_path:
            raise ArtifactConfigurationError(
                "IR model head outputPath differs from its trained head"
            )
        _require_logical_ref("IR head ref", head.get("ref"))
    resource_refs = {
        "tokenizer.main",
        cast(str, model.get("encoder")),
        cast(str, model.get("adapter")),
        *(cast(str, head.get("ref")) for head in heads),
    }
    if len(resource_refs) != 3 + len(heads):
        raise ArtifactConfigurationError("tokenizer, encoder, adapter, and head refs must differ")


def _validate_closed_ir_shape(ir: NeuralFunctionIr) -> None:
    required = {
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
    if set(ir) != required:
        raise ArtifactConfigurationError("verified IR must contain exactly the v1 fields")
    source = ir.get("source")
    if not isinstance(source, dict) or set(source) != {"path", "line", "column", "sourceSha256"}:
        raise ArtifactConfigurationError("IR source location is invalid")
    path = source.get("path")
    if (
        not isinstance(path, str)
        or not 1 <= len(path) <= 500
        or path.startswith("/")
        or "\\" in path
        or any(segment in ("", ".", "..") for segment in path.split("/"))
    ):
        raise ArtifactConfigurationError("IR source path is not a portable relative path")
    for name in ("line", "column"):
        value = source.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ArtifactConfigurationError(f"IR source {name} must be a positive integer")
    _require_sha256("IR sourceSha256", source.get("sourceSha256"))

    definition = ir.get("definition")
    if not isinstance(definition, dict) or set(definition) != {
        "template",
        "examples",
        "constraints",
    }:
        raise ArtifactConfigurationError("IR definition is invalid")
    template = definition.get("template")
    examples = definition.get("examples")
    constraints = definition.get("constraints")
    if not isinstance(template, list) or not template:
        raise ArtifactConfigurationError("IR definition template must be nonempty")
    for part in template:
        if not isinstance(part, dict):
            raise ArtifactConfigurationError("IR template parts must be objects")
        if part.get("kind") == "text":
            if set(part) != {"kind", "text"} or not isinstance(part.get("text"), str):
                raise ArtifactConfigurationError("IR text template part is invalid")
        elif part.get("kind") == "input":
            name = part.get("name")
            if (
                set(part) != {"kind", "name"}
                or not isinstance(name, str)
                or _IDENTIFIER.fullmatch(name) is None
            ):
                raise ArtifactConfigurationError("IR input template part is invalid")
        else:
            raise ArtifactConfigurationError("IR template part kind is invalid")
    if not isinstance(examples, list) or any(
        not isinstance(example, dict)
        or set(example) != {"inputs", "output"}
        or not isinstance(example.get("inputs"), dict)
        for example in examples
    ):
        raise ArtifactConfigurationError("IR definition examples are invalid")
    if not isinstance(constraints, list):
        raise ArtifactConfigurationError("IR definition constraints must be an array")

    model = ir.get("model")
    runtime = ir.get("runtime")
    if not isinstance(model, dict) or not {"encoder", "adapter", "heads"} <= set(model) <= {
        "encoder",
        "adapter",
        "heads",
        "encoderDepth",
    }:
        raise ArtifactConfigurationError("IR model binding is not closed")
    if not isinstance(runtime, dict) or set(runtime) != {
        "resultMode",
        "confidenceThreshold",
        "fallbackRef",
        "synchronous",
    }:
        raise ArtifactConfigurationError("IR runtime policy is not closed")
    if runtime.get("synchronous") is not True:
        raise ArtifactConfigurationError("IR runtime synchronous must be true")
    _validate_completed_provenance(ir.get("trainingProvenance"))


def _validate_completed_provenance(value: Any) -> None:
    required = {
        "status",
        "teacher",
        "baseModel",
        "datasetSha256",
        "counts",
        "seed",
        "trainer",
        "trainedAt",
    }
    if (
        not isinstance(value, dict)
        or set(value) - {"canonicalInput"} != required
        or value.get("status") != "complete"
    ):
        raise ArtifactConfigurationError("verified IR trainingProvenance is invalid")
    # Absent means the exact-JSON envelope: every IR verified before the compact encoding.
    if "canonicalInput" in value and canonical_input_version(value["canonicalInput"]) is None:
        raise ArtifactConfigurationError(
            "verified IR trainingProvenance canonicalInput is unsupported"
        )
    teacher = value.get("teacher")
    base = value.get("baseModel")
    if not isinstance(teacher, dict) or set(teacher) != {
        "provider",
        "model",
        "configurationSha256",
    }:
        raise ArtifactConfigurationError("IR teacher provenance is invalid")
    _nonempty("IR teacher provider", teacher.get("provider"))
    _nonempty("IR teacher model", teacher.get("model"))
    _require_sha256("IR teacher configurationSha256", teacher.get("configurationSha256"))
    if not isinstance(base, dict) or set(base) != {"name", "revision", "weightsSha256"}:
        raise ArtifactConfigurationError("IR base-model provenance is invalid")
    _nonempty("IR base model name", base.get("name"))
    _nonempty("IR base model revision", base.get("revision"))
    _require_sha256("IR base model weightsSha256", base.get("weightsSha256"))
    _require_sha256("IR training datasetSha256", value.get("datasetSha256"))

    counts = value.get("counts")
    count_names = {
        "examples",
        "synthetic",
        "adversarial",
        "calibration",
        "verification",
        "attestedVerification",
    }
    if not isinstance(counts, dict) or set(counts) != count_names:
        raise ArtifactConfigurationError("IR training provenance counts are invalid")
    for name in count_names:
        count = counts.get(name)
        minimum = 1 if name in ("calibration", "verification", "attestedVerification") else 0
        if isinstance(count, bool) or not isinstance(count, int) or count < minimum:
            raise ArtifactConfigurationError(f"IR training provenance count {name} is invalid")
    seed = value.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ArtifactConfigurationError("IR training provenance seed is invalid")
    trainer = value.get("trainer")
    if not isinstance(trainer, dict) or set(trainer) != {"version", "commit"}:
        raise ArtifactConfigurationError("IR trainer provenance is invalid")
    _nonempty("IR trainer version", trainer.get("version"))
    commit = trainer.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[a-f0-9]{7,64}", commit) is None:
        raise ArtifactConfigurationError("IR trainer commit is invalid")
    _validate_rfc3339("IR trainedAt", value.get("trainedAt"))


def _validate_tokenizer(
    tokenizer_json: bytes,
    verification: VerificationResult,
    config: ArtifactExportConfig,
) -> bytes:
    if not isinstance(tokenizer_json, bytes) or not tokenizer_json:
        raise ArtifactConfigurationError("tokenizer_json must be nonempty bytes")
    if len(tokenizer_json) > config.maximum_tokenizer_bytes:
        raise ArtifactConfigurationError(
            f"tokenizer_json exceeds maximum {config.maximum_tokenizer_bytes} bytes"
        )
    digest = hashlib.sha256(tokenizer_json).hexdigest()
    if digest != verification.tokenizer_sha256:
        raise ArtifactConfigurationError(
            "tokenizer_json differs from the tokenizer used for verification"
        )
    try:
        parsed = loads_strict_json(
            tokenizer_json,
            limits=StrictJsonLimits(
                maximum_bytes=config.maximum_tokenizer_bytes,
                maximum_depth=128,
                maximum_nodes=2_000_000,
            ),
        )
    except StrictJsonError as error:
        raise ArtifactConfigurationError(f"tokenizer_json is invalid: {error}") from error
    if not isinstance(parsed, dict):
        raise ArtifactConfigurationError("tokenizer_json must encode an object")
    try:
        from tokenizers import Tokenizer

        Tokenizer.from_str(tokenizer_json.decode("utf-8", errors="strict"))
    except ModuleNotFoundError as error:
        raise ArtifactConfigurationError(
            "artifact export requires the optional tokenizers training dependency"
        ) from error
    except (TypeError, ValueError, UnicodeError) as error:
        raise ArtifactConfigurationError(
            f"tokenizer_json is not a loadable tokenizer model: {error}"
        ) from error
    return tokenizer_json


def _validate_model_contract(training: TrainingResult) -> None:
    model = training.model
    encoder = getattr(model, "encoder", None)
    hidden_size = getattr(encoder, "hidden_size", None)
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise ArtifactConfigurationError("trained encoder hidden_size is invalid")
    for module, item in zip(_head_modules(training), training.heads, strict=True):
        head_config = getattr(module, "config", None)
        if getattr(head_config, "input_size", None) != hidden_size:
            raise ArtifactConfigurationError("trained head input width does not match the encoder")
        if getattr(head_config, "output_size", None) != item.contract.logit_count:
            raise ArtifactConfigurationError("trained head logit width does not match its contract")
        if getattr(head_config, "kind", None) != item.contract.parameterization:
            raise ArtifactConfigurationError(
                "trained head parameterization does not match its contract"
            )


def _head_modules(training: TrainingResult) -> tuple[Any, ...]:
    """One exportable head module per output head; a FieldHeads module is unpacked."""

    head = getattr(training.model, "head", None)
    if head is None:
        raise ArtifactConfigurationError("trained model has no head module")
    submodules = getattr(head, "heads", None)
    modules = tuple(submodules) if submodules is not None and hasattr(head, "slices") else (head,)
    if len(modules) != len(training.heads):
        raise ArtifactConfigurationError("trained head modules do not match the output heads")
    return modules


def _ir_head_specs(ir: NeuralFunctionIr) -> tuple[dict[str, Any], ...]:
    """The IR head spec of every output head: the scalar head or one per object field."""

    output = ir.get("output")
    if not isinstance(output, dict):
        raise ArtifactConfigurationError("IR output must be an object")
    if output.get("kind") == "scalar":
        head = output.get("head")
        if not isinstance(head, dict):
            raise ArtifactConfigurationError("IR scalar output head must be an object")
        return (head,)
    if output.get("kind") != "object":
        raise ArtifactConfigurationError("IR output kind must be scalar or object")
    fields = output.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ArtifactConfigurationError("IR object output requires fields")
    specs: list[dict[str, Any]] = []
    for field in fields:
        head = field.get("head") if isinstance(field, dict) else None
        if not isinstance(head, dict):
            raise ArtifactConfigurationError("IR object output field head must be an object")
        specs.append(head)
    return tuple(specs)


def _application_resource_paths(
    staging: Path,
    functions: Sequence[tuple[str, int]],
    *,
    encoder_keys: Sequence[str] = ("full",),
    adapter_refs: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Resource paths: the full encoder is ``model.onnx`` and a prefix ``depth-NNN.onnx``;
    a single adapter keeps ``application.onnx`` and several are named by their refs."""

    paths = {"tokenizer": staging / "tokenizer" / "tokenizer.json"}
    for key in encoder_keys:
        paths[f"encoder:{key}"] = (
            staging / "models" / "encoder" / ("model.onnx" if key == "full" else f"{key}.onnx")
        )
    if adapter_refs is None or len(adapter_refs) <= 1:
        paths["adapter"] = staging / "models" / "adapters" / "application.onnx"
        if adapter_refs:
            paths[f"adapter:{adapter_refs[0]}"] = paths["adapter"]
    else:
        for ref in adapter_refs:
            paths[f"adapter:{ref}"] = staging / "models" / "adapters" / f"{_safe_ref(ref)}.onnx"
    if "encoder:full" in paths:
        paths["encoder"] = paths["encoder:full"]
    for function_id, head_count in functions:
        for index in range(head_count):
            paths[f"head:{function_id}:{index}"] = (
                staging / "models" / "heads" / function_id / f"head-{index:03d}.onnx"
            )
    return paths


def _resource_paths(staging: Path, function_id: str) -> dict[str, Path]:
    paths = _application_resource_paths(staging, [(function_id, 1)])
    paths["head"] = paths.pop(f"head:{function_id}:0")
    return paths


def _export_application_onnx(
    entries: Sequence[ArtifactFunction],
    routing: _Routing,
    input_ids: Any,
    attention_mask: Any,
    paths: dict[str, Path],
    config: ArtifactExportConfig,
) -> None:
    try:
        from semantscript_model.export import export_routed_application_components
    except ImportError as error:
        raise ArtifactConfigurationError(
            "artifact export requires the optional ONNX training dependencies"
        ) from error
    head_modules: dict[str, Any] = {}
    head_paths: dict[str, Path] = {}
    head_adapters: dict[str, str] = {}
    for entry in entries:
        function_id = cast(str, entry.ir["id"])
        for index, module in enumerate(_head_modules(entry.training)):
            head_modules[f"{function_id}:{index}"] = module
            head_paths[f"{function_id}:{index}"] = paths[f"head:{function_id}:{index}"]
            head_adapters[f"{function_id}:{index}"] = routing.function_adapters[function_id]
    export_routed_application_components(
        routing.encoder,
        routing.adapters,
        routing.adapter_depths,
        head_modules,
        head_adapters,
        input_ids,
        attention_mask,
        encoder_paths={key: paths[f"encoder:{key}"] for key in routing.encoder_refs},
        adapter_paths={ref: paths[f"adapter:{ref}"] for ref in routing.adapters},
        head_paths=head_paths,
        relative_tolerance=config.parity_relative_tolerance,
        absolute_tolerance=config.parity_absolute_tolerance,
        maximum_component_bytes=config.maximum_resource_bytes,
    )


def _resource_documents(
    entries: Sequence[ArtifactFunction],
    paths: dict[str, Path],
    config: ArtifactExportConfig,
    routing: _Routing,
) -> list[dict[str, JsonValue]]:
    first = entries[0]
    hidden_size = getattr(first.training.model.encoder, "hidden_size", None)
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise ArtifactConfigurationError("trained encoder hidden_size is invalid")
    embedding = {"name": "sentence_embedding", "dtype": "float32", "shape": ["BATCH", hidden_size]}
    function_embedding = {
        "name": "function_embedding",
        "dtype": "float32",
        "shape": ["BATCH", hidden_size],
    }
    definitions: list[tuple[str, str, str, str, dict[str, Any] | None]] = [
        ("tokenizer", "tokenizer.main", "tokenizer", "tokenizer/tokenizer.json", None),
    ]
    staging = paths["tokenizer"].parent.parent
    for key, encoder_ref in routing.encoder_refs.items():
        definitions.append(
            (
                f"encoder:{key}",
                encoder_ref,
                "encoder",
                paths[f"encoder:{key}"].relative_to(staging).as_posix(),
                {
                    "opset": ONNX_OPSET_VERSION,
                    "inputs": [
                        {"name": "input_ids", "dtype": "int64", "shape": ["BATCH", "SEQUENCE"]},
                        {
                            "name": "attention_mask",
                            "dtype": "int64",
                            "shape": ["BATCH", "SEQUENCE"],
                        },
                    ],
                    "outputs": [dict(embedding)],
                    "externalData": False,
                },
            )
        )
    for adapter_ref in routing.adapters:
        definitions.append(
            (
                f"adapter:{adapter_ref}",
                adapter_ref,
                "adapter",
                paths[f"adapter:{adapter_ref}"].relative_to(staging).as_posix(),
                {
                    "opset": ONNX_OPSET_VERSION,
                    "inputs": [dict(embedding)],
                    "outputs": [dict(function_embedding)],
                    "externalData": False,
                },
            )
        )
    for entry in entries:
        function_id = cast(str, entry.ir["id"])
        bindings = cast(list[Any], cast(dict[str, Any], entry.ir["model"])["heads"])
        for index, item in enumerate(entry.training.heads):
            binding = cast(dict[str, JsonValue], bindings[index])
            definitions.append(
                (
                    f"head:{function_id}:{index}",
                    cast(str, binding["ref"]),
                    "head",
                    f"models/heads/{function_id}/head-{index:03d}.onnx",
                    {
                        "opset": ONNX_OPSET_VERSION,
                        "inputs": [dict(function_embedding)],
                        "outputs": [
                            {
                                "name": "logits",
                                "dtype": "float32",
                                "shape": ["BATCH", item.contract.logit_count],
                            }
                        ],
                        "externalData": False,
                    },
                )
            )
    resources: list[dict[str, JsonValue]] = []
    total = 0
    for key, reference, role, relative_path, onnx in definitions:
        byte_length, digest = _file_identity(paths[key], config.maximum_resource_bytes)
        total += byte_length
        if total > config.maximum_aggregate_resource_bytes:
            raise ArtifactConfigurationError(
                "artifact resources exceed maximum aggregate byte length "
                f"{config.maximum_aggregate_resource_bytes}"
            )
        resource: dict[str, JsonValue] = {
            "ref": reference,
            "role": role,
            "format": "tokenizer-json" if role == "tokenizer" else "onnx",
            "formatVersion": 1,
            "path": relative_path,
            "byteLength": byte_length,
            "sha256": digest,
        }
        if role == "tokenizer":
            resource["maximumSequenceLength"] = first.training.config.maximum_sequence_length
        else:
            resource["onnx"] = cast(JsonValue, onnx)
        resources.append(resource)
    return resources


def _manifest_document(
    entries: Sequence[ArtifactFunction],
    provenance: ArtifactProvenance,
    encoding: str,
    resources: list[dict[str, JsonValue]],
    routing: _Routing,
) -> dict[str, JsonValue]:
    # The model's encoder is the full stack when any domain runs it, else the deepest prefix.
    application_encoder = routing.encoder_refs.get(
        "full", routing.encoder_refs[max(routing.encoder_refs)]
    )
    functions: list[JsonValue] = []
    for entry in entries:
        ir, training, verification = entry.ir, entry.training, entry.verification
        inputs = ir.get("inputs")
        output = ir.get("output")
        model = cast(dict[str, JsonValue], ir["model"])
        bindings = cast(list[JsonValue], model["heads"])
        if not isinstance(inputs, list) or not isinstance(output, dict):
            raise ArtifactConfigurationError("IR input/output schemas are invalid")
        head_documents: list[JsonValue] = []
        for index, (item, head_spec) in enumerate(
            zip(training.heads, _ir_head_specs(ir), strict=True)
        ):
            head_metadata = verification.to_manifest_head_metadata(index)
            field_name = item.field_name
            head_documents.append(
                {
                    "outputPath": [] if field_name is None else [field_name],
                    "headRef": cast(str, cast(dict[str, JsonValue], bindings[index])["ref"]),
                    "type": _runtime_head_type_for(head_spec, item.contract),
                    "parameterization": item.contract.parameterization,
                    "calibration": head_metadata["calibration"],
                    "verification": head_metadata["verification"],
                }
            )
        functions.append(
            {
                "id": cast(str, ir["id"]),
                "semanticSha256": cast(str, ir["semanticSha256"]),
                "inputs": deepcopy(inputs),
                "inputSchemaSha256": semantic_json_sha256(inputs),
                "outputSchemaSha256": semantic_json_sha256(output),
                "adapterRef": cast(str, model["adapter"]),
                **(
                    {}
                    if cast(str, model["encoder"]) == application_encoder
                    else {"encoderRef": cast(str, model["encoder"])}
                ),
                "heads": head_documents,
                "runtime": _runtime_policy(ir),
                "verification": verification.to_manifest_function_verification(),
                "trainingProvenance": _training_provenance(ir, training, provenance),
            }
        )
    first_model = cast(dict[str, JsonValue], entries[0].ir["model"])
    return {
        "kind": "semantscript.application-artifact",
        "artifactVersion": ARTIFACT_VERSION,
        "irVersion": 1,
        "compatibility": {
            "runtimeAbiVersion": RUNTIME_ABI_VERSION,
            "modelAbiVersion": MODEL_ABI_VERSION,
            "canonicalInput": encoding,
            "minimumRuntimeVersion": "0.0.0",
            "requiredCapabilities": [],
        },
        "application": {
            "id": provenance.application_id,
            "version": provenance.application_version,
        },
        "build": {
            "createdAt": provenance.created_at,
            "compilerVersion": provenance.compiler_version,
            "trainerVersion": provenance.trainer_version,
            "sourceIrSha256": _source_ir_sha256(entries),
        },
        "resources": cast(JsonValue, resources),
        "model": {
            "tokenizerRef": "tokenizer.main",
            "encoderRef": application_encoder,
            "adapterRef": cast(str, first_model["adapter"]),
        },
        "functions": functions,
    }


def _source_ir_sha256(entries: Sequence[ArtifactFunction]) -> str:
    """One function: the digest of its source IR bytes; several: the digest of their
    per-function digests joined by newlines in function order."""

    digests = [hashlib.sha256(entry.source_ir_bytes).hexdigest() for entry in entries]
    if len(digests) == 1:
        return digests[0]
    return hashlib.sha256("\n".join(digests).encode("ascii")).hexdigest()


def _canonical_input_encoding(ir: NeuralFunctionIr, training: TrainingResult) -> str:
    """The encoding the model was trained with, cross-checked against the verified IR."""

    encoding = CANONICAL_INPUT_ENCODINGS[training.config.canonical_input_version]
    raw = ir.get("trainingProvenance")
    declared = raw.get("canonicalInput", CANONICAL_INPUT_V1) if isinstance(raw, dict) else None
    if declared != encoding:
        raise ArtifactConfigurationError(
            "IR trainingProvenance canonicalInput differs from the trained encoding"
        )
    return encoding


def _training_provenance(
    ir: NeuralFunctionIr,
    training: TrainingResult,
    export: ArtifactProvenance,
) -> dict[str, JsonValue]:
    raw = ir.get("trainingProvenance")
    if not isinstance(raw, dict) or raw.get("status") != "complete":
        raise ArtifactConfigurationError("verified IR requires complete trainingProvenance")
    dataset = raw.get("datasetSha256")
    _require_sha256("IR training datasetSha256", dataset)
    if dataset != training.base_dataset_sha256:
        raise ArtifactConfigurationError("IR datasetSha256 differs from the trained dataset")
    teacher = raw.get("teacher")
    base_model = raw.get("baseModel")
    if not isinstance(teacher, dict) or not isinstance(base_model, dict):
        raise ArtifactConfigurationError("IR teacher/baseModel provenance is invalid")
    provider = _nonempty("IR teacher provider", teacher.get("provider"))
    teacher_model = _nonempty("IR teacher model", teacher.get("model"))
    teacher_config = teacher.get("configurationSha256")
    _require_sha256("IR teacher configurationSha256", teacher_config)
    base_name = _nonempty("IR base model name", base_model.get("name"))
    base_revision = _nonempty("IR base model revision", base_model.get("revision"))
    weights_sha256 = base_model.get("weightsSha256")
    _require_sha256("IR base model weightsSha256", weights_sha256)
    if (
        base_name != training.config.encoder_name
        or base_revision != training.config.encoder_revision
    ):
        raise ArtifactConfigurationError(
            "IR base model does not match the trained encoder revision"
        )
    return {
        "datasetSha256": cast(str, dataset),
        "trainingKeySha256": export.training_key_sha256,
        "teacher": f"{provider}/{teacher_model}@sha256:{teacher_config}",
        "baseModel": f"{base_name}@{base_revision}#sha256:{weights_sha256}",
    }


def _runtime_head_type(output: dict[str, JsonValue], training: TrainingResult) -> JsonValue:
    head = output.get("head")
    if output.get("kind") != "scalar" or not isinstance(head, dict):
        raise ArtifactConfigurationError("scalar head type requires a scalar output")
    return _runtime_head_type_for(head, training.head)


def _runtime_head_type_for(head: dict[str, Any], contract: TrainingHeadContract) -> JsonValue:
    source_kind = contract.source_kind
    support = list(contract.support)
    if source_kind == "boolean":
        return {"kind": "boolean", "support": [False, True]}
    if source_kind in ("bounded-int", "bounded-number"):
        result: dict[str, JsonValue] = {
            "kind": "ordinal-number",
            "sourceKind": source_kind,
        }
        for key in ("minimum", "maximum", "step", "supportDecimal"):
            value = head.get(key)
            if key == "supportDecimal":
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    raise ArtifactConfigurationError("bounded output supportDecimal is invalid")
            elif not isinstance(value, str):
                raise ArtifactConfigurationError(f"bounded output {key} is invalid")
            result[key] = deepcopy(value)
        return result
    if all(isinstance(value, str) for value in support):
        kind = "ordinal-string" if contract.ordinal else "nominal-string"
        return {"kind": kind, "support": support}
    if not contract.ordinal and all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in support
    ):
        return {"kind": "nominal-number", "support": support}
    raise ArtifactConfigurationError("training head cannot be represented by model ABI v1")


def _runtime_policy(ir: NeuralFunctionIr) -> dict[str, JsonValue]:
    output = ir.get("output")
    scalar = isinstance(output, dict) and output.get("kind") == "scalar"
    raw = ir.get("runtime")
    if not isinstance(raw, dict):
        raise ArtifactConfigurationError("IR runtime policy must be an object")
    mode = raw.get("resultMode")
    threshold = raw.get("confidenceThreshold")
    fallback = raw.get("fallbackRef")
    if mode not in ("value", "diagnostic"):
        raise ArtifactConfigurationError("IR resultMode is invalid")
    if threshold is not None and (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0 <= threshold <= 1
    ):
        raise ArtifactConfigurationError("IR confidenceThreshold is invalid")
    if fallback is not None and (not isinstance(fallback, str) or not fallback):
        raise ArtifactConfigurationError("IR fallbackRef is invalid")
    if threshold is None:
        if fallback is not None:
            raise ArtifactConfigurationError("unthresholded runtime policy cannot set fallbackRef")
        policy = "none"
    else:
        policy = "scalar-top1" if scalar else "all-fields"
        if mode == "diagnostic" and fallback is not None:
            raise ArtifactConfigurationError("diagnostic runtime policy cannot set fallbackRef")
    return {
        "resultMode": mode,
        "confidenceThreshold": threshold,
        "policy": policy,
        "fallbackRef": fallback,
    }


def _semantic_equal(left: Any, right: Any) -> bool:
    """Compare JSON values with the compiler's typed binary64 semantics."""

    try:
        return semantic_json_bytes(cast(JsonValue, left)) == semantic_json_bytes(
            cast(JsonValue, right)
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ArtifactConfigurationError(f"IR contains invalid semantic JSON: {error}") from error


def _validate_manifest_document(manifest: dict[str, JsonValue]) -> None:
    """Validate untrusted IR projections before the immutable release is published."""

    resources = manifest.get("resources")
    functions = manifest.get("functions")
    if not isinstance(functions, list) or not functions:
        raise ArtifactConfigurationError("manifest must contain at least one function")
    head_total = sum(
        len(function.get("heads", [])) if isinstance(function, dict) else 0
        for function in functions
    )
    if not isinstance(resources, list) or len(resources) < 3 + head_total:
        raise ArtifactConfigurationError(
            "manifest must contain a tokenizer, an encoder, an adapter and one resource per head"
        )
    refs: set[str] = set()
    paths: set[str] = set()
    roles: dict[str, int] = {}
    for resource in resources:
        if not isinstance(resource, dict):
            raise ArtifactConfigurationError("manifest resources must be objects")
        reference = resource.get("ref")
        relative_path = resource.get("path")
        _require_logical_ref("manifest resource ref", reference)
        if not isinstance(relative_path, str):
            raise ArtifactConfigurationError("manifest resource path is invalid")
        if reference in refs or relative_path in paths:
            raise ArtifactConfigurationError("manifest resource refs and paths must be unique")
        refs.add(reference)
        paths.add(relative_path)
        role = resource.get("role")
        if not isinstance(role, str):
            raise ArtifactConfigurationError("manifest resource role is invalid")
        roles[role] = roles.get(role, 0) + 1
        if resource.get("format") == "onnx":
            _validate_onnx_precision(resource.get("onnx"), f"manifest resource {reference}")
    if (
        roles.get("tokenizer") != 1
        or roles.get("encoder", 0) < 1
        or roles.get("adapter", 0) < 1
        or roles.get("head") != head_total
        or set(roles) - {"tokenizer", "encoder", "adapter", "head"}
    ):
        raise ArtifactConfigurationError(
            "manifest resources must be one tokenizer, at least one encoder and adapter, "
            "and one per head"
        )

    function_ids: set[str] = set()
    head_refs: set[str] = set()
    for function in functions:
        if not isinstance(function, dict):
            raise ArtifactConfigurationError("manifest functions must be objects")
        function_id = function.get("id")
        if not isinstance(function_id, str) or function_id in function_ids:
            raise ArtifactConfigurationError("manifest function ids must be unique strings")
        function_ids.add(function_id)
        inputs = function.get("inputs")
        _validate_manifest_inputs(inputs)
        if not isinstance(inputs, list):
            raise AssertionError("validated manifest inputs must be an array")
        if function.get("inputSchemaSha256") != semantic_json_sha256(cast(JsonValue, inputs)):
            raise ArtifactConfigurationError("manifest inputSchemaSha256 does not match its inputs")

        heads = function.get("heads")
        if not isinstance(heads, list) or not heads or any(not isinstance(h, dict) for h in heads):
            raise ArtifactConfigurationError("manifest function must contain at least one head")
        scalar = len(heads) == 1 and heads[0].get("outputPath") == []
        field_names: set[str] = set()
        for head in heads:
            output_path = head.get("outputPath")
            if scalar:
                if output_path != []:
                    raise ArtifactConfigurationError(
                        "scalar manifest head outputPath must be empty"
                    )
            elif (
                not isinstance(output_path, list)
                or len(output_path) != 1
                or not isinstance(output_path[0], str)
                or output_path[0] in field_names
            ):
                raise ArtifactConfigurationError(
                    "flat object manifest heads must name distinct single-segment output paths"
                )
            else:
                field_names.add(output_path[0])
            head_ref = head.get("headRef")
            if not isinstance(head_ref, str) or head_ref in head_refs or head_ref not in refs:
                raise ArtifactConfigurationError(
                    "manifest head refs must name distinct head resources"
                )
            head_refs.add(head_ref)
            _validate_manifest_head_type(head.get("type"), head.get("parameterization"))
        runtime = function.get("runtime")
        if not isinstance(runtime, dict):
            raise ArtifactConfigurationError("manifest runtime policy must be an object")
        threshold = runtime.get("confidenceThreshold")
        if threshold is None and (
            runtime.get("policy") != "none" or runtime.get("fallbackRef") is not None
        ):
            raise ArtifactConfigurationError(
                "unthresholded manifest runtime policy is inconsistent"
            )
        expected_policy = "scalar-top1" if scalar else "all-fields"
        if threshold is not None and runtime.get("policy") != expected_policy:
            raise ArtifactConfigurationError(
                f"thresholded manifest policy must be {expected_policy}"
            )
        if runtime.get("resultMode") == "diagnostic" and runtime.get("fallbackRef") is not None:
            raise ArtifactConfigurationError("diagnostic manifest policy cannot set fallbackRef")


ONNX_PRECISIONS = ("float32", "int8-dynamic")
_ONNX_QUANTIZATION_FIELDS = {
    "method",
    "weightType",
    "perChannel",
    "reduceRange",
    "argmaxDisagreementTolerance",
    "attestedDisagreementTolerance",
    "eceThreshold",
    "sourceManifestSha256",
}


def _validate_onnx_precision(onnx: Any, label: str) -> None:
    """Precision is optional (float32 when absent); quantization goes with a quantized precision."""

    if not isinstance(onnx, dict):
        raise ArtifactConfigurationError(f"{label} onnx block must be an object")
    precision = onnx.get("precision", "float32")
    if precision not in ONNX_PRECISIONS:
        raise ArtifactConfigurationError(f"{label} onnx precision is unsupported")
    quantization = onnx.get("quantization")
    if precision == "float32":
        if "quantization" in onnx:
            raise ArtifactConfigurationError(f"{label} float32 graph cannot carry quantization")
        return
    if not isinstance(quantization, dict) or set(quantization) != _ONNX_QUANTIZATION_FIELDS:
        raise ArtifactConfigurationError(
            f"{label} quantized graph must carry exactly the quantization fields"
        )
    if quantization.get("method") != "dynamic":
        raise ArtifactConfigurationError(f"{label} quantization method is unsupported")
    if quantization.get("weightType") not in ("int8", "uint8"):
        raise ArtifactConfigurationError(f"{label} quantization weightType is unsupported")
    for name in ("perChannel", "reduceRange"):
        if not isinstance(quantization.get(name), bool):
            raise ArtifactConfigurationError(f"{label} quantization {name} must be a boolean")
    for name in ("argmaxDisagreementTolerance", "eceThreshold"):
        value = quantization.get(name)
        if not _is_finite_number(value) or not 0 <= value <= 1:
            raise ArtifactConfigurationError(f"{label} quantization {name} must be within [0, 1]")
    attested = quantization.get("attestedDisagreementTolerance")
    if (
        isinstance(attested, bool)
        or not isinstance(attested, (int, float))
        or attested < 0
        or attested != int(attested)
    ):
        raise ArtifactConfigurationError(
            f"{label} quantization attestedDisagreementTolerance must be a non-negative integer"
        )
    _require_sha256(
        f"{label} quantization sourceManifestSha256", quantization.get("sourceManifestSha256")
    )


def _validate_manifest_inputs(inputs: Any) -> None:
    if not isinstance(inputs, list):
        raise ArtifactConfigurationError("manifest function inputs must be an array")
    budget = [0]
    names: set[str] = set()
    for index, entry in enumerate(inputs):
        if not isinstance(entry, dict) or set(entry) != {"name", "index", "tsType", "type"}:
            raise ArtifactConfigurationError(f"manifest input {index} has invalid fields")
        name = entry.get("name")
        ts_type = entry.get("tsType")
        if not isinstance(name, str) or _IDENTIFIER.fullmatch(name) is None or name in names:
            raise ArtifactConfigurationError(f"manifest input {index} has an invalid name")
        entry_index = entry.get("index")
        if (
            isinstance(entry_index, bool)
            or not isinstance(entry_index, int)
            or entry_index != index
        ):
            raise ArtifactConfigurationError("manifest input indices must be dense and ordered")
        if not isinstance(ts_type, str) or not ts_type:
            raise ArtifactConfigurationError(f"manifest input {index} has an invalid tsType")
        names.add(name)
        _validate_manifest_input_type(entry.get("type"), depth=0, budget=budget)


def _validate_manifest_input_type(value: Any, *, depth: int, budget: list[int]) -> None:
    budget[0] += 1
    if depth > 64 or budget[0] > 100_000:
        raise ArtifactConfigurationError("manifest input types exceed runtime complexity limits")
    if not isinstance(value, dict):
        raise ArtifactConfigurationError("manifest input type must be an object")
    kind = value.get("kind")
    if kind in ("string", "boolean", "number", "null"):
        if set(value) != {"kind"}:
            raise ArtifactConfigurationError(f"manifest {kind} input type has extra fields")
        return
    if kind == "literal":
        if set(value) != {"kind", "value"} or not _is_json_scalar(value.get("value")):
            raise ArtifactConfigurationError("manifest literal input type is invalid")
        return
    if kind == "enum":
        if set(value) != {"kind", "name", "base", "values"}:
            raise ArtifactConfigurationError("manifest enum input type has invalid fields")
        name = value.get("name")
        base = value.get("base")
        values = value.get("values")
        if not isinstance(name, str) or not name or base not in ("string", "number"):
            raise ArtifactConfigurationError("manifest enum input type is invalid")
        if not isinstance(values, list) or not values:
            raise ArtifactConfigurationError("manifest enum values must be nonempty")
        if base == "string" and not all(isinstance(item, str) for item in values):
            raise ArtifactConfigurationError("manifest string enum contains a non-string")
        if base == "number" and not all(_is_finite_number(item) for item in values):
            raise ArtifactConfigurationError("manifest number enum contains a non-number")
        if len({semantic_json_bytes(cast(JsonValue, item)) for item in values}) != len(values):
            raise ArtifactConfigurationError("manifest enum values must be semantically unique")
        return
    if kind == "array":
        if set(value) != {"kind", "items"}:
            raise ArtifactConfigurationError("manifest array input type has invalid fields")
        _validate_manifest_input_type(value.get("items"), depth=depth + 1, budget=budget)
        return
    if kind == "tuple":
        items = value.get("items")
        if set(value) != {"kind", "items"} or not isinstance(items, list):
            raise ArtifactConfigurationError("manifest tuple input type is invalid")
        for item in items:
            _validate_manifest_input_type(item, depth=depth + 1, budget=budget)
        return
    if kind == "object":
        fields = value.get("fields")
        name = value.get("name")
        if (
            set(value) != {"kind", "name", "fields"}
            or not isinstance(name, str)
            or not name
            or not isinstance(fields, list)
        ):
            raise ArtifactConfigurationError("manifest object input type is invalid")
        field_names: set[str] = set()
        for field in fields:
            if not isinstance(field, dict) or set(field) != {"name", "optional", "type"}:
                raise ArtifactConfigurationError("manifest object field is invalid")
            field_name = field.get("name")
            if not isinstance(field_name, str) or field_name in field_names:
                raise ArtifactConfigurationError("manifest object field names must be unique")
            if not isinstance(field.get("optional"), bool):
                raise ArtifactConfigurationError("manifest object field optional must be boolean")
            field_names.add(field_name)
            _validate_manifest_input_type(field.get("type"), depth=depth + 1, budget=budget)
        return
    if kind == "union":
        variants = value.get("variants")
        if (
            set(value) != {"kind", "variants"}
            or not isinstance(variants, list)
            or len(variants) < 2
        ):
            raise ArtifactConfigurationError("manifest union input type is invalid")
        keys: list[bytes] = []
        for variant in variants:
            _validate_manifest_input_type(variant, depth=depth + 1, budget=budget)
            keys.append(semantic_json_bytes(cast(JsonValue, variant)))
        if any(left >= right for left, right in pairwise(keys)):
            raise ArtifactConfigurationError(
                "manifest union variants must be in strict semantic byte order"
            )
        return
    raise ArtifactConfigurationError(f"manifest input type kind is unsupported: {kind!r}")


def _validate_manifest_head_type(value: Any, parameterization: Any) -> None:
    if not isinstance(value, dict):
        raise ArtifactConfigurationError("manifest head type must be an object")
    kind = value.get("kind")
    if kind == "boolean":
        if set(value) != {"kind", "support"} or value.get("support") != [False, True]:
            raise ArtifactConfigurationError("manifest boolean head type is invalid")
        if parameterization != "binary-sigmoid":
            raise ArtifactConfigurationError("manifest boolean head must use binary-sigmoid")
        return
    if kind in ("nominal-string", "ordinal-string", "nominal-number"):
        support = value.get("support")
        if set(value) != {"kind", "support"} or not isinstance(support, list) or len(support) < 2:
            raise ArtifactConfigurationError("manifest categorical head type is invalid")
        valid = (
            all(isinstance(item, str) for item in support)
            if kind != "nominal-number"
            else all(_is_finite_number(item) for item in support)
        )
        if not valid or len(
            {semantic_json_bytes(cast(JsonValue, item)) for item in support}
        ) != len(support):
            raise ArtifactConfigurationError("manifest categorical support is invalid")
        if parameterization != "categorical-softmax":
            raise ArtifactConfigurationError("manifest categorical head must use softmax")
        return
    if kind != "ordinal-number" or set(value) != {
        "kind",
        "sourceKind",
        "minimum",
        "maximum",
        "step",
        "supportDecimal",
    }:
        raise ArtifactConfigurationError("manifest ordinal head type is invalid")
    _validate_decimal_support(value)
    if parameterization != "categorical-softmax":
        raise ArtifactConfigurationError("manifest ordinal head must use softmax")


def _validate_decimal_support(value: dict[str, Any]) -> None:
    source_kind = value.get("sourceKind")
    tokens = [value.get("minimum"), value.get("maximum"), value.get("step")]
    support = value.get("supportDecimal")
    if source_kind not in ("bounded-int", "bounded-number") or not isinstance(support, list):
        raise ArtifactConfigurationError("manifest ordinal-number source is invalid")
    tokens.extend(support)
    if len(support) < 2 or any(
        not isinstance(token, str)
        or len(token) > 128
        or _CANONICAL_DECIMAL.fullmatch(token) is None
        for token in tokens
    ):
        raise ArtifactConfigurationError("manifest ordinal-number decimals are invalid")
    minimum_text, maximum_text, step_text = cast(list[str], tokens[:3])
    support_text = cast(list[str], tokens[3:])
    try:
        minimum, maximum, step = map(Decimal, (minimum_text, maximum_text, step_text))
        support_values = [Decimal(token) for token in support_text]
    except InvalidOperation as error:
        raise ArtifactConfigurationError("manifest ordinal-number decimals are invalid") from error
    if step <= 0 or minimum >= maximum:
        raise ArtifactConfigurationError("manifest ordinal-number range or step is invalid")
    if source_kind == "bounded-int" and (
        step_text != "1"
        or any("." in token for token in [minimum_text, maximum_text, *support_text])
    ):
        raise ArtifactConfigurationError("bounded-int manifest support must be integral step 1")
    expected = [minimum + step * index for index in range(len(support_values))]
    if support_values != expected or support_values[-1] != maximum:
        raise ArtifactConfigurationError(
            "manifest supportDecimal is not the exact minimum/maximum/step sequence"
        )
    binary64: set[bytes] = set()
    for token in support_text:
        converted = float(token)
        if not math.isfinite(converted):
            raise ArtifactConfigurationError("manifest ordinal support is not finite binary64")
        binary64.add(pack(">d", converted))
    if len(binary64) != len(support_text):
        raise ArtifactConfigurationError("manifest ordinal support collides in binary64")


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool)) or _is_finite_number(value)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _prepare_artifact_root(value: str | os.PathLike[str]) -> tuple[Path, int | None]:
    try:
        root = Path(value)
    except TypeError as error:
        raise ArtifactConfigurationError("artifact_root must be path-like") from error
    if not root.name:
        raise ArtifactConfigurationError("artifact_root must name a directory")
    root = Path(os.path.abspath(root))
    parent = root.parent
    if _ANCHORED_DIRECTORY_OPERATIONS:
        try:
            parent_descriptor = os.open(parent, _DIRECTORY_OPEN_FLAGS)
        except OSError as error:
            raise ArtifactConfigurationError(
                "artifact_root parent must be a non-symlink directory"
            ) from error
        try:
            created = False
            try:
                os.mkdir(root.name, mode=0o700, dir_fd=parent_descriptor)
                created = True
            except FileExistsError:
                pass
            try:
                root_descriptor = os.open(
                    root.name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_descriptor
                )
            except OSError as error:
                raise ArtifactConfigurationError(
                    "artifact_root must be a non-symlink directory"
                ) from error
            if created:
                _fsync_directory_descriptor(parent_descriptor)
            return root, root_descriptor
        finally:
            os.close(parent_descriptor)

    if not parent.exists() or parent.is_symlink() or not parent.is_dir():
        raise ArtifactConfigurationError("artifact_root parent must be a non-symlink directory")
    if root.exists() or root.is_symlink():
        info = root.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ArtifactConfigurationError("artifact_root must be a non-symlink directory")
    else:
        root.mkdir(mode=0o700)
        _fsync_directory(parent)
    return root, None


def _prepare_child_directory(
    parent: Path, name: str, parent_descriptor: int | None
) -> tuple[Path, int | None]:
    child = parent / name
    if parent_descriptor is not None:
        created = False
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
            created = True
        except FileExistsError:
            pass
        try:
            child_descriptor = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_descriptor)
        except OSError as error:
            raise ArtifactConfigurationError(
                f"artifact {name} must be a non-symlink directory"
            ) from error
        if created:
            _fsync_directory_descriptor(parent_descriptor)
        return child, child_descriptor

    if child.exists() or child.is_symlink():
        info = child.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ArtifactConfigurationError(f"artifact {name} must be a non-symlink directory")
    else:
        child.mkdir(mode=0o700)
        _fsync_directory(parent)
    return child, None


def _create_staging_directory(
    parent: Path, parent_descriptor: int | None
) -> tuple[Path, int | None]:
    for _ in range(128):
        name = f".staging-{os.urandom(16).hex()}"
        try:
            if parent_descriptor is None:
                (parent / name).mkdir(mode=0o700)
                return parent / name, None
            else:
                os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
                try:
                    descriptor = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_descriptor)
                except OSError:
                    os.rmdir(name, dir_fd=parent_descriptor)
                    raise
                return parent / name, descriptor
        except FileExistsError:
            continue
    raise ArtifactPublicationError("could not allocate a unique staging directory")


def _descriptor_directory_path(path: Path, descriptor: int | None) -> Path:
    """Return a path whose traversal stays bound to an open directory."""

    if descriptor is None:
        return path
    if sys.platform.startswith("linux"):
        candidates = (Path("/proc/self/fd") / str(descriptor),)
    elif sys.platform == "darwin":
        candidates = (Path("/dev/fd") / str(descriptor),)
    else:
        candidates = ()
    expected = os.fstat(descriptor)
    for candidate in candidates:
        try:
            actual = candidate.stat()
        except OSError:
            continue
        if stat.S_ISDIR(actual.st_mode) and (actual.st_dev, actual.st_ino) == (
            expected.st_dev,
            expected.st_ino,
        ):
            return candidate
    raise ArtifactPublicationError(
        "opened staging directory has no safe descriptor-backed pathname"
    )


def _require_directory_identity(path: Path, descriptor: int | None, label: str) -> None:
    if descriptor is None:
        return
    expected = os.fstat(descriptor)
    try:
        actual = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ArtifactPublicationError(f"{label} path changed during export") from error
    if not stat.S_ISDIR(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
        expected.st_dev,
        expected.st_ino,
    ):
        raise ArtifactPublicationError(f"{label} path changed during export")


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _file_identity(path: Path, maximum_bytes: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactConfigurationError(f"artifact resource is not a regular file: {path}")
        if info.st_size < 1 or info.st_size > maximum_bytes:
            raise ArtifactConfigurationError(
                f"artifact resource byte length must be within 1..{maximum_bytes}: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while block := handle.read(1024 * 1024):
                size += len(block)
                if size > maximum_bytes:
                    raise ArtifactConfigurationError(
                        f"artifact resource exceeds maximum {maximum_bytes} bytes: {path}"
                    )
                digest.update(block)
            after = os.fstat(handle.fileno())
    finally:
        os.close(descriptor)
    if size != info.st_size or (info.st_dev, info.st_ino) != (after.st_dev, after.st_ino):
        raise ArtifactPublicationError(f"artifact resource changed while hashing: {path}")
    return size, digest.hexdigest()


def _publish_release(
    staging: Path,
    release: Path,
    manifest_bytes: bytes,
    resources: list[dict[str, JsonValue]],
    config: ArtifactExportConfig,
    releases_descriptor: int | None,
    staging_descriptor: int | None,
) -> Path:
    try:
        _rename_directory_no_replace(staging, release, releases_descriptor)
    except FileExistsError:
        _verify_existing_release(release, manifest_bytes, resources, config, releases_descriptor)
        _remove_owned_staging(
            staging,
            release.parent,
            releases_descriptor,
            staging_descriptor,
        )
    except OSError as error:
        raise ArtifactPublicationError(f"could not publish immutable release: {error}") from error
    if releases_descriptor is None:
        _fsync_directory(release.parent)
    else:
        _fsync_directory_descriptor(releases_descriptor)
    return release


def _rename_directory_no_replace(
    source: Path,
    destination: Path,
    parent_descriptor: int | None,
) -> None:
    """Atomically rename one sibling directory without replacing any entry."""

    if source.parent != destination.parent:
        raise ArtifactPublicationError("release staging must share its destination directory")
    if parent_descriptor is None:
        # Python's Windows rename maps to MoveFile and fails when the destination exists.
        if os.name == "nt":
            os.rename(source, destination)
            return
        raise ArtifactPublicationError(
            "atomic no-clobber directory publication is unavailable on this platform"
        )

    library = CDLL(None, use_errno=True)
    source_name = os.fsencode(source.name)
    destination_name = os.fsencode(destination.name)
    if sys.platform.startswith("linux"):
        try:
            renameat2 = library.renameat2
        except AttributeError as error:
            raise ArtifactPublicationError(
                "libc does not provide atomic renameat2 publication"
            ) from error
        renameat2.argtypes = [c_int, c_char_p, c_int, c_char_p, c_uint]
        renameat2.restype = c_int
        result = renameat2(
            parent_descriptor,
            source_name,
            parent_descriptor,
            destination_name,
            _RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin":
        try:
            renameatx_np = library.renameatx_np
        except AttributeError as error:
            raise ArtifactPublicationError(
                "libc does not provide atomic renameatx_np publication"
            ) from error
        renameatx_np.argtypes = [c_int, c_char_p, c_int, c_char_p, c_uint]
        renameatx_np.restype = c_int
        result = renameatx_np(
            parent_descriptor,
            source_name,
            parent_descriptor,
            destination_name,
            _RENAME_EXCL,
        )
    else:
        raise ArtifactPublicationError(
            "atomic no-clobber directory publication is unavailable on this platform"
        )
    if result == 0:
        return
    error_number = get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _verify_existing_release(
    release: Path,
    manifest_bytes: bytes,
    resources: list[dict[str, JsonValue]],
    config: ArtifactExportConfig,
    releases_descriptor: int | None,
) -> None:
    if releases_descriptor is not None:
        try:
            release_descriptor = os.open(
                release.name,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=releases_descriptor,
            )
        except OSError as error:
            raise ArtifactPublicationError(
                "content-addressed release path is not a safe directory"
            ) from error
        try:
            size, digest, existing_manifest_bytes = _relative_file_identity(
                release_descriptor,
                "manifest.json",
                config.maximum_manifest_bytes,
                include_bytes=True,
            )
            if size != len(manifest_bytes) or digest != hashlib.sha256(manifest_bytes).hexdigest():
                raise ArtifactPublicationError(
                    "existing content-addressed release manifest is inconsistent"
                )
            if existing_manifest_bytes != manifest_bytes:
                raise ArtifactPublicationError(
                    "existing content-addressed release manifest bytes differ"
                )
            for resource in resources:
                relative = resource["path"]
                if not isinstance(relative, str):
                    raise AssertionError("resource path must be a string")
                length, resource_digest, _ = _relative_file_identity(
                    release_descriptor,
                    relative,
                    config.maximum_resource_bytes,
                    include_bytes=False,
                )
                if length != resource["byteLength"] or resource_digest != resource["sha256"]:
                    raise ArtifactPublicationError(
                        f"existing content-addressed release resource is inconsistent: {relative}"
                    )
            return
        finally:
            os.close(release_descriptor)

    info = release.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ArtifactPublicationError("content-addressed release path is not a safe directory")
    existing_manifest = release / "manifest.json"
    size, digest = _file_identity(existing_manifest, config.maximum_manifest_bytes)
    if size != len(manifest_bytes) or digest != hashlib.sha256(manifest_bytes).hexdigest():
        raise ArtifactPublicationError(
            "existing content-addressed release manifest is inconsistent"
        )
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    descriptor = os.open(existing_manifest, flags)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            if handle.read() != manifest_bytes:
                raise ArtifactPublicationError(
                    "existing content-addressed release manifest bytes differ"
                )
    finally:
        os.close(descriptor)
    for resource in resources:
        relative = resource["path"]
        if not isinstance(relative, str):
            raise AssertionError("resource path must be a string")
        target = _safe_existing_release_file(release, relative)
        length, resource_digest = _file_identity(target, config.maximum_resource_bytes)
        if length != resource["byteLength"] or resource_digest != resource["sha256"]:
            raise ArtifactPublicationError(
                f"existing content-addressed release resource is inconsistent: {relative}"
            )


def _relative_file_identity(
    root_descriptor: int,
    relative: str,
    maximum_bytes: int,
    *,
    include_bytes: bool,
) -> tuple[int, str, bytes | None]:
    segments = relative.split("/")
    if not segments or any(segment in {"", ".", ".."} for segment in segments):
        raise ArtifactPublicationError(f"existing release resource path is unsafe: {relative}")
    opened_directories: list[int] = []
    current_descriptor = root_descriptor
    try:
        for segment in segments[:-1]:
            try:
                current_descriptor = os.open(
                    segment,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=current_descriptor,
                )
            except OSError as error:
                raise ArtifactPublicationError(
                    f"existing release resource has an unsafe parent: {relative}"
                ) from error
            opened_directories.append(current_descriptor)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(segments[-1], flags, dir_fd=current_descriptor)
        except OSError as error:
            raise ArtifactPublicationError(
                f"existing release resource is not a safe regular file: {relative}"
            ) from error
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size < 1 or info.st_size > maximum_bytes:
                raise ArtifactPublicationError(
                    f"existing release resource is not a bounded regular file: {relative}"
                )
            digest = hashlib.sha256()
            contents = bytearray() if include_bytes else None
            size = 0
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                while block := handle.read(1024 * 1024):
                    size += len(block)
                    if size > maximum_bytes:
                        raise ArtifactPublicationError(
                            f"existing release resource exceeds maximum bytes: {relative}"
                        )
                    digest.update(block)
                    if contents is not None:
                        contents.extend(block)
            if size != info.st_size:
                raise ArtifactPublicationError(
                    f"existing release resource changed while hashing: {relative}"
                )
            return size, digest.hexdigest(), bytes(contents) if contents is not None else None
        finally:
            os.close(descriptor)
    finally:
        for descriptor in reversed(opened_directories):
            os.close(descriptor)


def _safe_existing_release_file(release: Path, relative: str) -> Path:
    segments = relative.split("/")
    current = release
    for segment in segments[:-1]:
        current /= segment
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ArtifactPublicationError(
                f"existing release resource has an unsafe parent: {relative}"
            )
    return current / segments[-1]


def _publish_pointer(root: Path, manifest_sha256: str, root_descriptor: int | None) -> Path:
    pointer = {
        "kind": "semantscript.artifact-pointer",
        "pointerVersion": 1,
        "release": f"releases/sha256-{manifest_sha256}",
        "manifestSha256": manifest_sha256,
    }
    destination = root / "current.json"
    if root_descriptor is not None:
        try:
            info = os.stat("current.json", dir_fd=root_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(info.st_mode):
                raise ArtifactPublicationError("current.json must be a regular file or absent")
        temporary_name = f".current-{os.getpid()}-{os.urandom(8).hex()}.tmp"
        try:
            _write_exclusive_at(root_descriptor, temporary_name, _json_bytes(pointer))
            os.rename(
                temporary_name,
                "current.json",
                src_dir_fd=root_descriptor,
                dst_dir_fd=root_descriptor,
            )
            _fsync_directory_descriptor(root_descriptor)
        finally:
            try:
                os.unlink(temporary_name, dir_fd=root_descriptor)
            except FileNotFoundError:
                pass
        return destination

    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ArtifactPublicationError("current.json must be a regular file or absent")
    temporary = root / f".current-{os.getpid()}-{os.urandom(8).hex()}.tmp"
    try:
        _write_exclusive(temporary, _json_bytes(pointer))
        os.replace(temporary, destination)
        _fsync_directory(root)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def _write_exclusive_at(directory_descriptor: int, name: str, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ArtifactConfigurationError(
            f"artifact metadata is not strict JSON: {error}"
        ) from error


def _fsync_tree(root: Path) -> None:
    for directory, _, filenames in os.walk(root, topdown=False, followlinks=False):
        current = Path(directory)
        for name in filenames:
            target = current / name
            info = target.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ArtifactPublicationError(f"staging contains a non-regular file: {target}")
            descriptor = os.open(target, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _fsync_directory(current)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        _fsync_directory_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_descriptor(descriptor: int) -> None:
    os.fsync(descriptor)


def _remove_owned_staging(
    staging: Path,
    releases: Path,
    releases_descriptor: int | None,
    staging_descriptor: int | None,
) -> None:
    try:
        if staging.parent != releases or not staging.name.startswith(".staging-"):
            raise ArtifactPublicationError("refusing to clean an unowned staging path")
        if releases_descriptor is not None:
            if staging_descriptor is None:
                raise ArtifactPublicationError(
                    "refusing to clean staging without its directory descriptor"
                )
            expected = os.fstat(staging_descriptor)
            actual = os.stat(
                staging.name,
                dir_fd=releases_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
                expected.st_dev,
                expected.st_ino,
            ):
                raise ArtifactPublicationError("refusing to clean a replaced staging directory")
            shutil.rmtree(staging.name, dir_fd=releases_descriptor)
        elif staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
    except FileNotFoundError:
        pass


def _require_sha256(name: str, value: Any) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ArtifactConfigurationError(f"{name} must be 64 lowercase hexadecimal characters")


def _require_logical_ref(name: str, value: Any) -> None:
    if not isinstance(value, str) or len(value) > 200 or _LOGICAL_REF.fullmatch(value) is None:
        raise ArtifactConfigurationError(f"{name} is invalid")


def _nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactConfigurationError(f"{name} must be nonempty")
    return value


def _validate_rfc3339(name: str, value: Any) -> None:
    if not isinstance(value, str) or _RFC3339.fullmatch(value) is None:
        raise ArtifactConfigurationError(f"{name} must be an RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as error:
        raise ArtifactConfigurationError(f"{name} must be an RFC 3339 timestamp") from error
    if parsed.tzinfo is None:
        raise ArtifactConfigurationError(f"{name} must be an RFC 3339 timestamp")


__all__ = [
    "ARTIFACT_VERSION",
    "MAXIMUM_AGGREGATE_RESOURCE_BYTES",
    "MAXIMUM_MANIFEST_BYTES",
    "MAXIMUM_PARITY_ABSOLUTE_TOLERANCE",
    "MAXIMUM_PARITY_RELATIVE_TOLERANCE",
    "MAXIMUM_RESOURCE_BYTES",
    "MAXIMUM_TOKENIZER_BYTES",
    "MODEL_ABI_VERSION",
    "ONNX_OPSET_VERSION",
    "RUNTIME_ABI_VERSION",
    "ArtifactConfigurationError",
    "ArtifactExportConfig",
    "ArtifactExportError",
    "ArtifactProvenance",
    "ArtifactPublicationError",
    "ExportedArtifact",
    "export_application_artifact",
    "validate_verified_ir_binding",
]
