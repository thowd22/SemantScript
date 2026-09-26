"""Derive an int8 release of an application from one of its published releases.

``semantscript releases derive --int8`` runs this through
``python -m semantscript_trainer.cli derive-int8``. The release's own
verification records come from the build cache, found strictly by digest:

* the training dataset whose file SHA-256 is the manifest's
  ``trainingProvenance.datasetSha256`` for each function; its gold cases (the
  sema call's examples) are the attested records;
* the adversarial sidecar the build cache's ``function.json`` pins
  (``adversarialDatasetSha256``), or, when the cache has no such record for
  that dataset, the one sidecar whose ``baseDataset.datasetSha256`` is that
  dataset.

A missing or ambiguous file refuses the derivation instead of verifying on
records the release was not trained on. Held-out records join through
:func:`held_out_records` once the release gate records a held-out set; until
then it contributes none and the report says so. The quantized release is
published beside the source without moving ``current.json``: promoting it is
a separate, reversible step (``semantscript releases promote``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

from semantscript_trainer.artifact import ArtifactExportConfig
from semantscript_trainer.quantization import (
    QUANTIZED_PRECISION,
    QuantizationConfig,
    QuantizationGateError,
    QuantizationRecord,
    QuantizationReport,
    quantize_release_artifact,
)
from semantscript_trainer.remedies import remedy
from semantscript_trainer.teacher import JsonValue

DERIVE_REPORT_KIND = "semantscript.derive-report"
DERIVE_REPORT_VERSION = 1
HELD_OUT_PENDING = (
    "no held-out set is recorded for this release: the quantized graph is verified on the "
    "release's training, gold and adversarial records; derive does not re-check the "
    "release gate's held-out inputs yet"
)

_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_RELEASE_SPEC = re.compile(r"^(?:releases/)?(?:sha256-)?([a-f0-9]{7,64})/*$")


class DeriveError(ValueError):
    """The derivation cannot start: no such release, or its records are not cached."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.fix = fix


@dataclass(frozen=True, slots=True)
class RecordSource:
    """One file of records the derivation checks the quantized graph on."""

    kind: str
    function_id: str
    sha256: str
    path: Path
    records: int

    def to_manifest_document(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind,
            "functionId": self.function_id,
            "sha256": self.sha256,
            "records": self.records,
        }

    def to_report_document(self) -> dict[str, JsonValue]:
        return {**self.to_manifest_document(), "path": str(self.path)}


@dataclass(frozen=True, slots=True)
class HeldOutRecords:
    """Records a release gate held out from training, for one function."""

    source: RecordSource
    records: tuple[QuantizationRecord, ...]


HeldOutProvider = Callable[[Mapping[str, Any], Path], Sequence[HeldOutRecords]]


def held_out_records(manifest: Mapping[str, Any], cache_directory: Path) -> list[HeldOutRecords]:
    """The release's held-out records; none until the release gate records a held-out set.

    The hook the derivation calls for them: when a release records its
    held-out set, this returns it (labeled, not attested) and the derivation
    checks the quantized graph on it too and names it in the manifest's
    ``recordSources`` with kind ``held-out``.
    """

    del manifest, cache_directory
    return []


@dataclass(frozen=True, slots=True)
class DeriveResult:
    """What one derivation did; ``report`` is the ``semantscript.derive-report`` document."""

    status: str
    report: dict[str, Any]


def resolve_release(artifact_root: Path, release: str | None) -> str:
    """The manifest digest of the named release, or of the current one when none is named."""

    releases = artifact_root / "releases"
    if release is None:
        pointer_path = artifact_root / "current.json"
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise DeriveError(
                f"{pointer_path} does not exist, so there is no current release to derive from",
                remedy("artifact-missing", path=artifact_root),
            ) from error
        except (OSError, ValueError) as error:
            raise DeriveError(f"{pointer_path} is unreadable: {error}") from error
        digest = pointer.get("manifestSha256") if isinstance(pointer, dict) else None
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise DeriveError(f"{pointer_path} names no release")
        return digest
    match = _RELEASE_SPEC.fullmatch(release)
    if match is None:
        raise DeriveError(
            f"{release} is not a release: give its manifest digest (at least 7 lowercase hex "
            "characters), sha256-<digest> or releases/sha256-<digest>"
        )
    prefix = match.group(1)
    try:
        names = sorted(path.name for path in releases.iterdir())
    except OSError as error:
        raise DeriveError(f"{releases} is not readable: {error}") from error
    matches = [
        name[len("sha256-") :]
        for name in names
        if name.startswith(f"sha256-{prefix}") and _DIGEST.fullmatch(name[len("sha256-") :])
    ]
    if not matches:
        raise DeriveError(f"no release under {releases} matches {release}")
    if len(matches) > 1:
        raise DeriveError(f"{release} matches {len(matches)} releases; give more of the digest")
    return matches[0]


def verification_records(
    manifest: Mapping[str, Any],
    cache_directory: Path,
    *,
    held_out: HeldOutProvider = held_out_records,
) -> tuple[list[QuantizationRecord], list[RecordSource]]:
    """The release's records from the build cache, strictly by digest (see the module docs)."""

    functions = cast(list[dict[str, Any]], manifest.get("functions") or [])
    application = manifest.get("application")
    application_id = application.get("id") if isinstance(application, dict) else None
    datasets = _documents(cache_directory / "datasets" / "v1", "semantscript.training-dataset")
    sidecars = _documents(
        cache_directory / "adversarial-datasets" / "v1", "semantscript.adversarial-dataset"
    )
    records: list[QuantizationRecord] = []
    sources: list[RecordSource] = []
    for function in functions:
        function_id = cast(str, function["id"])
        provenance = function.get("trainingProvenance")
        dataset_sha256 = provenance.get("datasetSha256") if isinstance(provenance, dict) else None
        if not isinstance(dataset_sha256, str) or _DIGEST.fullmatch(dataset_sha256) is None:
            raise DeriveError(
                f"release function {function_id} records no training dataset digest",
                _records_missing(cache_directory, function_id, "(none recorded)"),
            )
        dataset = datasets.get(dataset_sha256)
        if dataset is None or _function_id(dataset[1]) != function_id:
            raise DeriveError(
                f"the build cache {cache_directory} has no training dataset "
                f"{dataset_sha256} for {function_id}",
                _records_missing(cache_directory, function_id, dataset_sha256),
            )
        heads = cast(list[dict[str, Any]], function["heads"])
        cases = _cases(dataset[1])
        records.extend(
            QuantizationRecord(
                dict(cast(dict[str, JsonValue], case["inputs"])),
                attested=case.get("origin") == "gold",
                function_id=function_id,
                label_indices=_labels(heads, case.get("output")),
            )
            for case in cases
        )
        sources.append(
            RecordSource("training-dataset", function_id, dataset_sha256, dataset[0], len(cases))
        )
        sidecar = _adversarial_sidecar(
            cache_directory, application_id, function_id, dataset_sha256, sidecars
        )
        if sidecar is not None:
            sidecar_sha256, (path, payload) = sidecar
            adversarial = _cases(payload)
            records.extend(
                QuantizationRecord(
                    dict(cast(dict[str, JsonValue], case["inputs"])),
                    function_id=function_id,
                    label_indices=_labels(heads, case.get("output")),
                )
                for case in adversarial
            )
            sources.append(
                RecordSource(
                    "adversarial-dataset", function_id, sidecar_sha256, path, len(adversarial)
                )
            )
    for entry in held_out(manifest, cache_directory):
        records.extend(entry.records)
        sources.append(entry.source)
    return records, sources


def derive_int8_release(
    artifact_root: str | os.PathLike[str],
    *,
    cache_directory: str | os.PathLike[str],
    release: str | None = None,
    quantization: QuantizationConfig | None = None,
    created_at: str | None = None,
    held_out: HeldOutProvider = held_out_records,
    log: Callable[[str], None] = lambda message: None,
    command_flags: str = "",
) -> DeriveResult:
    """Derive, verify and publish an int8 release; refusals return a ``refused`` report.

    Raises :class:`DeriveError` when the release or its records cannot be
    found; the error's ``fix`` names the next command. ``command_flags`` are
    the location flags the caller was given (``--artifact``, ``--cache-dir``,
    ``--python`` ...), already shell-quoted with a leading space; every
    ``semantscript releases derive`` command the report names repeats them
    after the source release, so the command runs as printed.
    """

    started = time.monotonic()
    root = Path(os.path.abspath(artifact_root))
    cache = Path(os.path.abspath(cache_directory))
    settings = QuantizationConfig() if quantization is None else quantization
    source_sha256 = resolve_release(root, release)
    command = f"semantscript releases derive --int8 {source_sha256[:12]}{command_flags}"
    source_directory = root / "releases" / f"sha256-{source_sha256}"
    try:
        manifest = json.loads((source_directory / "manifest.json").read_bytes())
    except (OSError, ValueError) as error:
        raise DeriveError(
            f"releases/sha256-{source_sha256} has no readable manifest: {error}",
            remedy("artifact-corrupt"),
        ) from error
    if not isinstance(manifest, dict):
        raise DeriveError(f"releases/sha256-{source_sha256} manifest is not an object")
    for resource in cast(list[Any], manifest.get("resources") or []):
        onnx = resource.get("onnx") if isinstance(resource, dict) else None
        if isinstance(onnx, dict) and onnx.get("precision", "float32") != "float32":
            origin = (onnx.get("quantization") or {}).get("sourceManifestSha256")
            known = isinstance(origin, str) and _DIGEST.fullmatch(origin) is not None
            raise DeriveError(
                f"releases/sha256-{source_sha256[:12]} is already {onnx['precision']}; derive "
                "from the float32 release it was derived from"
                + (f" (releases/sha256-{origin[:12]})" if known else ""),
                (
                    f"semantscript releases derive --int8 {origin[:12]}{command_flags}"
                    if known
                    else "semantscript releases to find the float32 release, then "
                    f"semantscript releases derive --int8 <release>{command_flags}"
                ),
            )
    records, sources = verification_records(manifest, cache, held_out=held_out)
    held_out_count = sum(source.records for source in sources if source.kind == "held-out")
    log(
        f"verifying the int8 graph of releases/sha256-{source_sha256[:12]} on {len(records)} "
        f"records ({sum(1 for record in records if record.attested)} attested) from "
        f"{len(sources)} cached file(s); weight type {settings.weight_type}, per-channel "
        f"{_yes(settings.per_channel)}, reduce-range {_yes(settings.reduce_range)}"
    )
    report: dict[str, Any] = {
        "kind": DERIVE_REPORT_KIND,
        "reportVersion": DERIVE_REPORT_VERSION,
        "status": "failed",
        "precision": QUANTIZED_PRECISION,
        "artifactRoot": str(root),
        "cacheDirectory": str(cache),
        "source": {
            "manifestSha256": source_sha256,
            "release": f"releases/sha256-{source_sha256}",
            "bytes": _tree_bytes(source_directory),
        },
        "derived": None,
        "settings": settings.to_manifest_document(source_sha256),
        "recordSources": [source.to_report_document() for source in sources],
        "heldOut": {
            "records": held_out_count,
            "note": None if held_out_count else HELD_OUT_PENDING,
        },
        "quantization": None,
        "failures": [],
        "next": None,
    }
    try:
        derived = quantize_release_artifact(
            root,
            source_sha256,
            records=records,
            quantization=settings,
            config=ArtifactExportConfig(publish_pointer=False),
            created_at=created_at,
            record_sources=[source.to_manifest_document() for source in sources],
        )
    except QuantizationGateError as error:
        failures = str(error).removeprefix("quantization gate failed: ").split("; ")
        report.update(
            status="refused",
            quantization=error.report.to_document(),
            failures=failures,
            next=remedy(
                "int8-gate-refused",
                alternative=(
                    f", or try {command} {_settings_flags(settings, per_channel=True)}, which "
                    "changes fewer decisions on most encoders"
                    if not settings.per_channel and _decisions_failed(error.report, settings)
                    else ""
                ),
                command=command,
                tolerance=_tolerance_flags(error.report, settings),
            ),
        )
        report["elapsedSeconds"] = round(time.monotonic() - started, 3)
        return DeriveResult("refused", report)
    report.update(
        status="published",
        derived={
            "manifestSha256": derived.manifest_sha256,
            "release": f"releases/sha256-{derived.manifest_sha256}",
            "bytes": _tree_bytes(derived.release_directory),
        },
        quantization=derived.report.to_document(),
        next=f"semantscript releases promote {derived.manifest_sha256[:12]}",
    )
    report["elapsedSeconds"] = round(time.monotonic() - started, 3)
    return DeriveResult("published", report)


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def _decisions_failed(report: QuantizationReport, settings: QuantizationConfig) -> bool:
    """Whether the refusal involved changed decisions (not only calibration)."""

    return (
        report.attested_disagreements > settings.maximum_attested_disagreements
        or report.argmax_disagreement_rate > settings.maximum_argmax_disagreement_rate
    )


def _quantization_flags(settings: QuantizationConfig, *, per_channel: bool) -> list[str]:
    return [
        flag
        for flag, used in (
            (f"--weight-type {settings.weight_type}", settings.weight_type != "int8"),
            ("--per-channel", per_channel),
            ("--reduce-range", settings.reduce_range),
        )
        if used
    ]


def _settings_flags(settings: QuantizationConfig, *, per_channel: bool) -> str:
    """The flags that repeat these settings, with ``per_channel`` in place of the given one."""

    defaults = QuantizationConfig()
    flags = _quantization_flags(settings, per_channel=per_channel)
    if settings.maximum_attested_disagreements != defaults.maximum_attested_disagreements:
        flags.append(f"--max-attested-disagreements {settings.maximum_attested_disagreements}")
    if settings.maximum_argmax_disagreement_rate != defaults.maximum_argmax_disagreement_rate:
        flags.append(f"--max-decision-change-rate {settings.maximum_argmax_disagreement_rate:g}")
    if settings.ece_threshold != defaults.ece_threshold:
        flags.append(f"--ece-threshold {settings.ece_threshold:g}")
    return " ".join(flags)


def _tolerance_flags(report: QuantizationReport, settings: QuantizationConfig) -> str:
    """The flags that repeat these settings under a tolerance admitting exactly these figures.

    A tolerance the caller already widened is kept when it is wider than the
    figures need, so the suggested command does not refuse on a check that
    passed.
    """

    flags = _quantization_flags(settings, per_channel=settings.per_channel)
    flags += [
        "--max-attested-disagreements "
        f"{max(report.attested_disagreements, settings.maximum_attested_disagreements, 0)}",
        "--max-decision-change-rate "
        f"{_ceil(max(report.argmax_disagreement_rate, settings.maximum_argmax_disagreement_rate))}",
    ]
    if report.quantized_ece > settings.ece_threshold:
        flags.append(f"--ece-threshold {_ceil(report.quantized_ece)}")
    elif settings.ece_threshold != QuantizationConfig().ece_threshold:
        flags.append(f"--ece-threshold {settings.ece_threshold:g}")
    return " ".join(flags)


def _ceil(value: float) -> str:
    """``value`` rounded up to four decimals, so the printed tolerance admits it."""

    scaled = int(-(-value * 10_000 // 1))
    return f"{min(scaled, 10_000) / 10_000:g}"


def _records_missing(cache: Path, function_id: str, dataset: str) -> str:
    return remedy("int8-records-missing", cache=cache, function=function_id, dataset=dataset)


def _documents(directory: Path, kind: str) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Cached documents of one kind by file SHA-256; a torn or foreign file is skipped."""

    documents: dict[str, tuple[Path, dict[str, Any]]] = {}
    try:
        shards = sorted(directory.iterdir())
    except OSError:
        return documents
    for shard in shards:
        if not shard.is_dir() or shard.is_symlink():
            continue
        for path in sorted(shard.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                data = path.read_bytes()
                document = json.loads(data)
            except (OSError, ValueError):
                continue
            if not isinstance(document, dict):
                continue
            payload = document.get("payload")
            if document.get("kind") != kind or not isinstance(payload, dict):
                continue
            documents[hashlib.sha256(data).hexdigest()] = (path, payload)
    return documents


def _adversarial_sidecar(
    cache: Path,
    application_id: object,
    function_id: str,
    dataset_sha256: str,
    sidecars: dict[str, tuple[Path, dict[str, Any]]],
) -> tuple[str, tuple[Path, dict[str, Any]]] | None:
    """The sidecar the build cache pins for this dataset, else the unique one built on it."""

    if isinstance(application_id, str):
        record_path = (
            cache / "applications" / application_id / "functions" / function_id / "function.json"
        )
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = None
        if isinstance(record, dict) and record.get("datasetSha256") == dataset_sha256:
            if "adversarialDatasetSha256" in record:
                pinned = record["adversarialDatasetSha256"]
                if pinned is None:
                    return None
                sidecar = sidecars.get(pinned) if isinstance(pinned, str) else None
                if sidecar is None:
                    raise DeriveError(
                        f"{record_path} pins adversarial dataset {pinned}, which the build "
                        "cache no longer holds",
                        _records_missing(cache, function_id, str(pinned)),
                    )
                return pinned, sidecar
    candidates = [
        (digest, entry)
        for digest, entry in sidecars.items()
        if _function_id(entry[1]) == function_id
        and isinstance(entry[1].get("baseDataset"), dict)
        and entry[1]["baseDataset"].get("datasetSha256") == dataset_sha256
    ]
    if len(candidates) > 1:
        raise DeriveError(
            f"{len(candidates)} adversarial datasets in {cache} are built on {dataset_sha256} "
            f"and no build-cache record says which one {function_id} was trained on",
            _records_missing(cache, function_id, dataset_sha256),
        )
    return candidates[0] if candidates else None


def _function_id(payload: Mapping[str, Any]) -> object:
    function = payload.get("function")
    return function.get("id") if isinstance(function, dict) else None


def _cases(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = payload.get("cases")
    if not isinstance(cases, list) or not all(
        isinstance(case, dict) and isinstance(case.get("inputs"), dict) for case in cases
    ):
        raise DeriveError("a cached dataset's cases are not {inputs, output} objects")
    return cast(list[dict[str, Any]], cases)


def _labels(heads: Sequence[Mapping[str, Any]], output: object) -> tuple[int | None, ...]:
    """Each head's label index for one case's output; None where the output is off its support."""

    labels: list[int | None] = []
    for head in heads:
        path = head.get("outputPath") or []
        value = output
        for field in path:
            value = value.get(field) if isinstance(value, dict) else None
        head_type = head.get("type")
        support = head_type.get("support") if isinstance(head_type, dict) else None
        decimals = head_type.get("supportDecimal") if isinstance(head_type, dict) else None
        label: int | None = None
        if isinstance(decimals, list):
            label = _decimal_label(decimals, value)
        elif isinstance(support, list):
            for index, member in enumerate(support):
                if type(member) is type(value) or (
                    isinstance(member, (int, float))
                    and isinstance(value, (int, float))
                    and not isinstance(member, bool)
                    and not isinstance(value, bool)
                ):
                    if member == value:
                        label = index
                        break
        labels.append(label)
    return tuple(labels)


def _decimal_label(decimals: Sequence[object], value: object) -> int | None:
    """The index of ``value`` in an ordinal-number head's canonical decimal support."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        target = Decimal(repr(value)) if isinstance(value, float) else Decimal(value)
    except (InvalidOperation, ValueError):
        return None
    for index, token in enumerate(decimals):
        if not isinstance(token, str):
            continue
        try:
            if Decimal(token) == target:
                return index
        except InvalidOperation:
            continue
    return None


def _tree_bytes(path: Path) -> int:
    total = 0
    for directory, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(directory) / name).lstat().st_size
            except OSError:
                pass
    return total


__all__ = [
    "DERIVE_REPORT_KIND",
    "DERIVE_REPORT_VERSION",
    "HELD_OUT_PENDING",
    "DeriveError",
    "DeriveResult",
    "HeldOutProvider",
    "HeldOutRecords",
    "RecordSource",
    "derive_int8_release",
    "held_out_records",
    "resolve_release",
    "verification_records",
]
