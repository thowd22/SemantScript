"""Content-addressed build cache behind ``semantscript train`` (TASK-7.2).

A rebuild must not retrain what did not change. Function ids are derived from
the semantic identity of an expression, so they are the cache key; the value is
everything the exporter needs to publish that function again without training:
the exact encoder, adapter and head weights (safetensors), the verified IR bytes,
the verification record and the digests that bind them. A cached function is
reused only when its id and semantic digest match the bundle and it was built
under the same recipe (encoder, canonical input version, adapter size, training,
verification and adversarial settings and the teacher); any other difference is
a miss. Every file is digest-checked on load and a mismatch is a miss too.

Layout under the trainer cache directory::

    applications/<application-id>/
      application.json          recipe digest, last release digest, function index
      shared.safetensors        encoder and adapter state, exact bits
      functions/<function-id>/
        function.json           digests, metrics, verification record, file digests
        verified-ir.json        exact verified IR bytes
        head.safetensors        the function's head state
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from semantscript_trainer.adversarial import AdversarialGenerationConfig
from semantscript_trainer.semantic_json import semantic_json_sha256
from semantscript_trainer.teacher import JsonValue, TeacherDescriptor
from semantscript_trainer.training import EpochMetrics, TrainingConfig, TrainingResult
from semantscript_trainer.verification import (
    CalibrationRecordV1,
    HeadVerificationV1,
    VerificationConfig,
    VerificationMetricsV1,
    VerificationResult,
)

CACHE_VERSION = 2
APPLICATION_KIND = "semantscript.build-cache-application"
FUNCTION_KIND = "semantscript.build-cache-function"
_RECIPE_KIND = "semantscript.build-cache-recipe"
_SHARED_FILE = "shared.safetensors"
_ADAPTERS_DIRECTORY = "adapters"
_HEAD_FILE = "head.safetensors"
_IR_FILE = "verified-ir.json"
_RECORD_FILE = "function.json"
_INDEX_FILE = "application.json"


class BuildCacheError(RuntimeError):
    """The cache cannot be read or written consistently."""


@dataclass(frozen=True, slots=True)
class CacheRecipe:
    """Everything besides the expression itself that decides a trained head."""

    encoder_name: str
    encoder_revision: str
    canonical_input_version: int
    adapter_bottleneck_size: int
    training: dict[str, Any]
    verification: dict[str, Any]
    adversarial: dict[str, Any]
    teacher: dict[str, str]

    @property
    def projection(self) -> dict[str, JsonValue]:
        return cast(
            dict[str, JsonValue],
            {
                "kind": _RECIPE_KIND,
                "cacheVersion": CACHE_VERSION,
                "encoderName": self.encoder_name,
                "encoderRevision": self.encoder_revision,
                "canonicalInputVersion": self.canonical_input_version,
                "adapterBottleneckSize": self.adapter_bottleneck_size,
                "training": self.training,
                "verification": self.verification,
                "adversarial": self.adversarial,
                "teacher": self.teacher,
            },
        )

    @property
    def sha256(self) -> str:
        return semantic_json_sha256(self.projection)


def cache_recipe(
    training: TrainingConfig,
    verification: VerificationConfig,
    adversarial: AdversarialGenerationConfig,
    adapter_bottleneck_size: int,
    teacher: TeacherDescriptor,
) -> CacheRecipe:
    return CacheRecipe(
        encoder_name=training.encoder_name,
        encoder_revision=training.encoder_revision,
        canonical_input_version=training.canonical_input_version,
        adapter_bottleneck_size=adapter_bottleneck_size,
        training=asdict(training),
        verification=asdict(verification),
        adversarial=asdict(adversarial),
        teacher={
            "provider": teacher.provider,
            "model": teacher.model,
            "configurationSha256": teacher.configuration_sha256,
        },
    )


@dataclass(frozen=True, slots=True)
class CachedFunction:
    """One function as it was last trained, verified and published."""

    function_id: str
    semantic_sha256: str
    recipe_sha256: str
    shared_state_sha256: str
    adapter_ref: str
    model_sha256: str
    source_path: str | None
    dataset_sha256: str
    dataset_cases: int
    dataset_gold: int
    adversarial_dataset_sha256: str | None
    adversarial_cases: int
    tokenizer_sha256: str
    model_state_sha256: str
    device: str
    metrics: tuple[EpochMetrics, ...]
    selected_epoch: int | None
    verification: VerificationResult
    verified_ir_bytes: bytes
    head_state_path: Path


class ApplicationCache:
    """The cache of one application under the trainer's cache directory."""

    def __init__(self, cache_directory: str | os.PathLike[str], application_id: str) -> None:
        if not isinstance(application_id, str) or not application_id:
            raise BuildCacheError("application_id must be a nonempty string")
        self.root = Path(cache_directory) / "applications" / application_id
        self.application_id = application_id

    # ---- reading -------------------------------------------------------

    def index(self) -> dict[str, Any] | None:
        path = self.root / _INDEX_FILE
        if not path.is_file():
            return None
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (
            not isinstance(document, dict)
            or document.get("kind") != APPLICATION_KIND
            or document.get("cacheVersion") != CACHE_VERSION
        ):
            return None
        return document

    def release_sha256(self) -> str | None:
        index = self.index()
        release = index.get("release") if index is not None else None
        digest = release.get("manifestSha256") if isinstance(release, dict) else None
        return digest if isinstance(digest, str) else None

    def cached_function(self, function_id: str) -> CachedFunction | None:
        """The function's record when every file is intact, else None (a miss)."""

        directory = self.root / "functions" / function_id
        record_path = directory / _RECORD_FILE
        if not record_path.is_file():
            return None
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (
            not isinstance(record, dict)
            or record.get("kind") != FUNCTION_KIND
            or record.get("cacheVersion") != CACHE_VERSION
            or record.get("functionId") != function_id
        ):
            return None
        try:
            files = record["files"]
            ir_bytes = (directory / _IR_FILE).read_bytes()
            head_path = directory / _HEAD_FILE
            if _sha256(ir_bytes) != files[_IR_FILE] or _file_sha256(head_path) != files[_HEAD_FILE]:
                return None
            verification = _verification_from_record(record["verification"])
            metrics = tuple(
                EpochMetrics(
                    epoch=int(entry["epoch"]),
                    mean_training_loss=float(entry["meanTrainingLoss"]),
                    held_out_accuracy=float(entry["heldOutAccuracy"]),
                    held_out_field_accuracy=(
                        None
                        if entry.get("heldOutFieldAccuracy") is None
                        else tuple(float(v) for v in entry["heldOutFieldAccuracy"])
                    ),
                )
                for entry in record["metrics"]
            )
            return CachedFunction(
                function_id=function_id,
                semantic_sha256=str(record["semanticSha256"]),
                recipe_sha256=str(record["recipeSha256"]),
                shared_state_sha256=str(record["sharedStateSha256"]),
                adapter_ref=str(record["adapterRef"]),
                model_sha256=str(record["modelSha256"]),
                source_path=record.get("sourcePath"),
                dataset_sha256=str(record["datasetSha256"]),
                dataset_cases=int(record["datasetCases"]),
                dataset_gold=int(record["datasetGold"]),
                adversarial_dataset_sha256=record.get("adversarialDatasetSha256"),
                adversarial_cases=int(record.get("adversarialCases", 0)),
                tokenizer_sha256=str(record["tokenizerSha256"]),
                model_state_sha256=str(record["modelStateSha256"]),
                device=str(record["device"]),
                metrics=metrics,
                selected_epoch=record.get("selectedEpoch"),
                verification=verification,
                verified_ir_bytes=ir_bytes,
                head_state_path=head_path,
            )
        except (KeyError, TypeError, ValueError, OSError, RuntimeError):
            return None

    def shared_state_intact(self) -> bool:
        """The encoder file and every adapter file match the digests the index recorded."""

        index = self.index()
        if index is None:
            return False
        path = self.root / _SHARED_FILE
        if not path.is_file() or _file_sha256(path) != index.get("sharedStateSha256"):
            return False
        adapters = index.get("adapterStateSha256")
        if not isinstance(adapters, dict):
            return False
        for ref, digest in adapters.items():
            adapter_path = self._adapter_path(str(ref))
            if not adapter_path.is_file() or _file_sha256(adapter_path) != digest:
                return False
        return True

    def cached_adapter_refs(self) -> tuple[str, ...]:
        index = self.index()
        adapters = index.get("adapterStateSha256") if index is not None else None
        return tuple(sorted(adapters)) if isinstance(adapters, dict) else ()

    def function_shared_sha256(self, adapter_ref: str) -> str | None:
        """The digest a function's record must carry: its encoder and its domain adapter."""

        index = self.index()
        if index is None:
            return None
        encoder = index.get("sharedStateSha256")
        adapters = index.get("adapterStateSha256")
        adapter = adapters.get(adapter_ref) if isinstance(adapters, dict) else None
        if not isinstance(encoder, str) or not isinstance(adapter, str):
            return None
        return combined_shared_sha256(encoder, adapter)

    def load_shared_state(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """The encoder state and every cached adapter's state by ref, exactly as stored."""

        if not self.shared_state_intact():
            raise BuildCacheError("cached shared encoder state is missing or corrupt")
        encoder = _load_safetensors(self.root / _SHARED_FILE)
        adapters = {
            ref: _load_safetensors(self._adapter_path(ref)) for ref in self.cached_adapter_refs()
        }
        return encoder, adapters

    def load_head_state(self, cached: CachedFunction) -> dict[str, Any]:
        return _load_safetensors(cached.head_state_path)

    # ---- writing -------------------------------------------------------

    def store_shared_state(self, encoder: Any, adapters: Mapping[str, Any]) -> SharedStateDigests:
        """Persist the exact encoder and every adapter's weights; returns their digests.

        Every previously cached adapter is dropped: this is the fresh joint build.
        """

        self.root.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.root / _ADAPTERS_DIRECTORY, ignore_errors=True)
        _save_safetensors(dict(encoder.state_dict()), self.root / _SHARED_FILE)
        digests = {ref: self.store_adapter_state(ref, module) for ref, module in adapters.items()}
        return SharedStateDigests(_file_sha256(self.root / _SHARED_FILE), digests)

    def store_adapter_state(self, ref: str, adapter: Any) -> str:
        """Persist one domain adapter's exact weights (a new domain on a cached encoder)."""

        path = self._adapter_path(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_safetensors(dict(adapter.state_dict()), path)
        return _file_sha256(path)

    def _adapter_path(self, ref: str) -> Path:
        return self.root / _ADAPTERS_DIRECTORY / f"{ref.replace('.', '__')}.safetensors"

    def store_function(
        self,
        *,
        ir: Mapping[str, Any],
        recipe_sha256: str,
        shared_state_sha256: str,
        training: TrainingResult,
        verification: VerificationResult,
        verified_ir_bytes: bytes,
        dataset_cases: int,
        dataset_gold: int,
        adversarial_cases: int,
    ) -> None:
        function_id = str(ir["id"])
        model = ir.get("model")
        if model is None:
            model = {"encoder": "encoder.main", "adapter": "adapter.application"}
        if not isinstance(model, Mapping) or not isinstance(model.get("adapter"), str):
            raise BuildCacheError("function IR must name its adapter under model.adapter")
        directory = self.root / "functions" / function_id
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        head = getattr(training.model, "head", None)
        if head is None:
            raise BuildCacheError("training result model has no head to cache")
        head_path = directory / _HEAD_FILE
        _save_safetensors(dict(head.state_dict()), head_path)
        (directory / _IR_FILE).write_bytes(verified_ir_bytes)
        source = ir.get("source")
        record = {
            "kind": FUNCTION_KIND,
            "cacheVersion": CACHE_VERSION,
            "functionId": function_id,
            "semanticSha256": ir["semanticSha256"],
            "recipeSha256": recipe_sha256,
            "sharedStateSha256": shared_state_sha256,
            "adapterRef": model["adapter"],
            "modelSha256": semantic_json_sha256(cast(JsonValue, dict(model))),
            "sourcePath": source.get("path") if isinstance(source, Mapping) else None,
            "datasetSha256": training.base_dataset_sha256,
            "datasetCases": dataset_cases,
            "datasetGold": dataset_gold,
            "adversarialDatasetSha256": training.adversarial_dataset_sha256,
            "adversarialCases": adversarial_cases,
            "tokenizerSha256": verification.tokenizer_sha256,
            "modelStateSha256": verification.model_state_sha256,
            "device": training.device,
            "selectedEpoch": training.selected_epoch,
            "metrics": [
                {
                    "epoch": entry.epoch,
                    "meanTrainingLoss": entry.mean_training_loss,
                    "heldOutAccuracy": entry.held_out_accuracy,
                    "heldOutFieldAccuracy": (
                        None
                        if entry.held_out_field_accuracy is None
                        else list(entry.held_out_field_accuracy)
                    ),
                }
                for entry in training.metrics
            ],
            "verification": _verification_record(verification),
            "files": {_IR_FILE: _sha256(verified_ir_bytes), _HEAD_FILE: _file_sha256(head_path)},
            "storedAt": _utc_now(),
        }
        (directory / _RECORD_FILE).write_text(_dump(record), encoding="utf-8")

    def store_index(
        self,
        *,
        recipe: CacheRecipe,
        function_ids: Sequence[str],
        release_sha256: str,
        shared_state_sha256: str,
        tokenizer_sha256: str,
        adapter_state_sha256: Mapping[str, str],
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        document = {
            "kind": APPLICATION_KIND,
            "cacheVersion": CACHE_VERSION,
            "applicationId": self.application_id,
            "recipeSha256": recipe.sha256,
            "recipe": recipe.projection,
            "sharedStateSha256": shared_state_sha256,
            "adapterStateSha256": dict(sorted(adapter_state_sha256.items())),
            "tokenizerSha256": tokenizer_sha256,
            "release": {"manifestSha256": release_sha256},
            "functions": sorted(function_ids),
            "updatedAt": _utc_now(),
        }
        (self.root / _INDEX_FILE).write_text(_dump(document), encoding="utf-8")

    def shared_state_sha256(self) -> str | None:
        index = self.index()
        digest = index.get("sharedStateSha256") if index is not None else None
        return digest if isinstance(digest, str) else None


@dataclass(frozen=True, slots=True)
class SharedStateDigests:
    """Digests of the cached encoder file and of each adapter file by ref."""

    encoder_sha256: str
    adapter_sha256: dict[str, str]

    def for_function(self, adapter_ref: str) -> str:
        return combined_shared_sha256(self.encoder_sha256, self.adapter_sha256[adapter_ref])


def combined_shared_sha256(encoder_sha256: str, adapter_sha256: str) -> str:
    """The shared-state digest of one function: its encoder and its domain adapter."""

    return hashlib.sha256(f"{encoder_sha256}\n{adapter_sha256}".encode("ascii")).hexdigest()


# ---- verification records ----------------------------------------------


def _verification_record(verification: VerificationResult) -> dict[str, Any]:
    return {
        "functionId": verification.function_id,
        "semanticSha256": verification.semantic_sha256,
        "modelStateSha256": verification.model_state_sha256,
        "tokenizerSha256": verification.tokenizer_sha256,
        "status": verification.status,
        "verifiedAt": verification.verified_at,
        "attestedCases": verification.attested_cases,
        "pairCount": verification.pair_count,
        "failures": list(verification.failures),
        "metrics": verification.metrics.to_ir_document(),
    }


def _verification_from_record(record: Mapping[str, Any]) -> VerificationResult:
    metrics = record["metrics"]
    heads = tuple(
        HeadVerificationV1(
            output_path=str(head["outputPath"]),
            accuracy=float(head["accuracy"]),
            pair_consistency=float(head["pairConsistency"]),
            calibration=CalibrationRecordV1(
                temperature=float(head["calibration"]["temperature"]),
                ece=float(head["calibration"]["ece"]),
                brier=float(head["calibration"]["brier"]),
                sample_count=int(head["calibration"]["sampleCount"]),
                split_sha256=str(head["calibration"]["splitSha256"]),
                ece_bins=int(head["calibration"]["eceBins"]),
            ),
        )
        for head in metrics["heads"]
    )
    return VerificationResult(
        function_id=str(record["functionId"]),
        semantic_sha256=str(record["semanticSha256"]),
        model_state_sha256=str(record["modelStateSha256"]),
        tokenizer_sha256=str(record["tokenizerSha256"]),
        status=record["status"],
        verified_at=str(record["verifiedAt"]),
        metrics=VerificationMetricsV1(
            accuracy=float(metrics["accuracy"]),
            ece=float(metrics["ece"]),
            brier=float(metrics["brier"]),
            pair_consistency=float(metrics["pairConsistency"]),
            heads=heads,
            example_failures=int(metrics["exampleFailures"]),
            constraint_violations=int(metrics["constraintViolations"]),
            type_errors=int(metrics.get("typeErrors", 0)),
        ),
        attested_cases=int(record["attestedCases"]),
        pair_count=int(record["pairCount"]),
        failures=tuple(str(item) for item in record["failures"]),
    )


# ---- files ----------------------------------------------------------------


def _save_safetensors(state: Mapping[str, Any], path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - training extra absent
        raise BuildCacheError(
            "the build cache needs safetensors; install the training extra"
        ) from error
    tensors = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in state.items()}
    temporary = path.with_name(path.name + ".tmp")
    try:
        save_file(tensors, str(temporary))
        os.replace(temporary, path)
    except (OSError, RuntimeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise BuildCacheError(f"could not write {path.name}: {error}") from error


def _load_safetensors(path: Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - training extra absent
        raise BuildCacheError(
            "the build cache needs safetensors; install the training extra"
        ) from error
    try:
        return dict(load_file(str(path)))
    except (OSError, RuntimeError, ValueError) as error:
        raise BuildCacheError(f"could not read {path.name}: {error}") from error


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "APPLICATION_KIND",
    "CACHE_VERSION",
    "FUNCTION_KIND",
    "ApplicationCache",
    "BuildCacheError",
    "CacheRecipe",
    "CachedFunction",
    "cache_recipe",
]
