"""``derive-int8``: derive an int8 release from an application release's cached records."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

from semantscript_trainer import cli
from semantscript_trainer import quantization as quantization_module
from semantscript_trainer.artifact import ArtifactConfigurationError
from semantscript_trainer.derive import (
    HELD_OUT_PENDING,
    DeriveError,
    HeldOutRecords,
    RecordSource,
    derive_int8_release,
    resolve_release,
    verification_records,
)
from semantscript_trainer.quantization import (
    QuantizationConfig,
    QuantizationRecord,
    quantize_release_artifact,
)

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "runtime" / "test" / "fixtures"
FIRST = f"nf_{'1' * 64}"
SECOND = f"nf_{'8' * 64}"
APPLICATION = "runtime-fixture"
SUPPORT = ["approve", "deny", "review"]

if shutil.which("node") is None:  # pragma: no cover - the CI images carry node
    pytest.skip(
        "the fixture artifact is written by runtime/test/fixtures/artifact.mjs",
        allow_module_level=True,
    )


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _cases(count: int, *, gold: int, output: str = "review") -> list[dict[str, Any]]:
    return [
        {
            "inputs": {"facts": {"a": index, "b": index * 3 + 1}},
            "origin": "gold" if index < gold else "synthetic",
            "output": output,
        }
        for index in range(count)
    ]


def write_dataset(cache: Path, function_id: str, cases: list[dict[str, Any]], tag: str) -> str:
    request = hashlib.sha256(f"{tag}:{function_id}".encode()).hexdigest()
    return _write_json(
        cache / "datasets" / "v1" / request[:2] / f"{request}.json",
        {
            "datasetVersion": 1,
            "kind": "semantscript.training-dataset",
            "payload": {
                "cases": cases,
                "function": {"id": function_id},
                "requestSha256": request,
                "teacher": {"provider": "fixture", "model": "fixture"},
            },
        },
    )


def write_sidecar(
    cache: Path, function_id: str, base_sha256: str, cases: list[dict[str, Any]], tag: str
) -> str:
    request = hashlib.sha256(f"{tag}:{function_id}".encode()).hexdigest()
    return _write_json(
        cache / "adversarial-datasets" / "v1" / request[:2] / f"{request}.json",
        {
            "kind": "semantscript.adversarial-dataset",
            "payload": {
                "baseDataset": {"datasetSha256": base_sha256},
                "cases": cases,
                "function": {"id": function_id},
            },
        },
    )


def fixture_release(root: Path, datasets: dict[str, str]) -> str:
    """A two-function release on the quantizable fixture encoder; returns its digest."""

    script = """
import process from 'node:process';
const [moduleUrl, root, second, datasets] = process.argv.slice(1);
const { createFixtureArtifact } = await import(moduleUrl);
const digests = JSON.parse(datasets);
const result = await createFixtureArtifact(root, {
  encoderSource: 'quantizable-encoder.onnx',
  extraFunctions: [{ id: second, headRef: 'head.second.value' }],
  transformManifest: (manifest) => {
    // The fixture's functions share one provenance object: replace it, per function.
    for (const fn of manifest.functions) {
      fn.trainingProvenance = { ...fn.trainingProvenance, datasetSha256: digests[fn.id] };
    }
  },
});
process.stdout.write(result.manifestSha256);
"""
    completed = subprocess.run(
        [
            "node",
            "--input-type=module",
            "--eval",
            script,
            (FIXTURES / "artifact.mjs").as_uri(),
            str(root),
            SECOND,
            json.dumps(datasets),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


@pytest.fixture
def application(tmp_path: Path) -> SimpleNamespace:
    """The release, its build cache and the digests of what the cache holds."""

    cache = tmp_path / "cache"
    first_dataset = write_dataset(cache, FIRST, _cases(20, gold=3), "base")
    second_dataset = write_dataset(cache, SECOND, _cases(16, gold=2), "base")
    # The first function's sidecar is pinned by the build cache record; a second
    # sidecar on the same dataset is left over from an earlier run.
    pinned = write_sidecar(cache, FIRST, first_dataset, _cases(6, gold=0), "adv")
    write_sidecar(cache, FIRST, first_dataset, _cases(9, gold=0), "adv-old")
    _write_json(
        cache / "applications" / APPLICATION / "functions" / FIRST / "function.json",
        {"datasetSha256": first_dataset, "adversarialDatasetSha256": pinned},
    )
    # The second function has no build cache record: its one sidecar is used.
    second_sidecar = write_sidecar(cache, SECOND, second_dataset, _cases(4, gold=0), "adv")
    digest = fixture_release(tmp_path / "artifact", {FIRST: first_dataset, SECOND: second_dataset})
    return SimpleNamespace(
        root=tmp_path / "artifact",
        cache=cache,
        digest=digest,
        first_dataset=first_dataset,
        second_dataset=second_dataset,
        pinned=pinned,
        second_sidecar=second_sidecar,
    )


def releases(root: Path) -> list[str]:
    return sorted(path.name for path in (root / "releases").iterdir())


def test_derives_publishes_and_records_provenance(application: SimpleNamespace) -> None:
    before = (application.root / "current.json").read_bytes()

    result = derive_int8_release(application.root, cache_directory=application.cache)

    assert result.status == "published"
    report = result.report
    assert report["kind"] == "semantscript.derive-report"
    assert report["source"]["manifestSha256"] == application.digest
    derived = report["derived"]["manifestSha256"]
    assert report["next"] == f"semantscript releases promote {derived[:12]}"
    # Published beside the source; the pointer is left for releases promote.
    assert (application.root / "current.json").read_bytes() == before
    assert releases(application.root) == sorted(
        [f"sha256-{application.digest}", f"sha256-{derived}"]
    )
    assert [
        (source["kind"], source["sha256"], source["records"]) for source in report["recordSources"]
    ] == [
        ("training-dataset", application.first_dataset, 20),
        ("adversarial-dataset", application.pinned, 6),
        ("training-dataset", application.second_dataset, 16),
        ("adversarial-dataset", application.second_sidecar, 4),
    ]
    assert report["heldOut"] == {"records": 0, "note": HELD_OUT_PENDING}
    figures = report["quantization"]
    assert figures["recordsChecked"] == 46 and figures["attestedRecords"] == 5
    assert figures["argmaxDisagreements"] == 0 and figures["attestedDisagreements"] == 0
    assert [entry["id"] for entry in figures["functions"]] == [FIRST, SECOND]
    assert [entry["recordsChecked"] for entry in figures["functions"]] == [26, 20]

    manifest_bytes = (
        application.root / "releases" / f"sha256-{derived}" / "manifest.json"
    ).read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == derived
    manifest = json.loads(manifest_bytes)
    encoder = next(r for r in manifest["resources"] if r["role"] == "encoder")
    assert encoder["onnx"]["precision"] == "int8-dynamic"
    quantization = encoder["onnx"]["quantization"]
    assert quantization["sourceManifestSha256"] == application.digest
    assert quantization["attestedDisagreementTolerance"] == 0
    assert quantization["argmaxDisagreementTolerance"] == 0.0
    verification = quantization["verification"]
    assert verification["recordsChecked"] == 46
    assert verification["attestedRecords"] == 5
    assert verification["decisionChanges"] == 0
    assert verification["attestedDecisionChanges"] == 0
    assert verification["recordSources"][0] == {
        "kind": "training-dataset",
        "functionId": FIRST,
        "sha256": application.first_dataset,
        "records": 20,
    }
    assert [entry["id"] for entry in verification["functions"]] == [FIRST, SECOND]
    # Every other resource is the source's, byte for byte.
    source = json.loads(
        (
            application.root / "releases" / f"sha256-{application.digest}" / "manifest.json"
        ).read_text()
    )
    assert [r for r in manifest["resources"] if r["role"] != "encoder"] == [
        r for r in source["resources"] if r["role"] != "encoder"
    ]

    runtime_entry = ROOT / "runtime" / "dist" / "index.js"
    if not runtime_entry.is_file():
        pytest.skip("runtime/dist is produced by the repository build gate")
    script = """
import process from 'node:process';
const [runtimeUrl, release, first, second] = process.argv.slice(1);
const runtime = await import(runtimeUrl);
await runtime.checkSemaArtifact(release);
"""
    checked = subprocess.run(
        [
            "node",
            "--input-type=module",
            "--eval",
            script,
            runtime_entry.as_uri(),
            str(application.root / "releases" / f"sha256-{derived}"),
            FIRST,
            SECOND,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert checked.returncode == 0, checked.stderr


def flipped_encoder(source: Path, destination: Path, settings: Any, config: Any) -> Any:
    """A 'quantized' encoder whose projection is negated: every decision changes."""

    import numpy
    import onnx

    model = onnx.load(str(source))
    for initializer in model.graph.initializer:
        if initializer.name == "projection":
            values = -onnx.numpy_helper.to_array(initializer)
            initializer.CopyFrom(
                onnx.numpy_helper.from_array(values.astype(numpy.float32), initializer.name)
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    data = destination.read_bytes()
    return SimpleNamespace(
        method="dynamic",
        weight_type="int8",
        per_channel=False,
        reduce_range=False,
        byte_length=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        source_byte_length=source.stat().st_size,
        quantized_matmul_count=1,
    )


def test_refuses_an_attested_disagreement_with_the_figures_and_the_remedy(
    application: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(quantization_module, "_quantize_encoder", flipped_encoder)
    before = releases(application.root)
    report_path = tmp_path / "derive-report.json"

    status = cli.main(
        [
            "derive-int8",
            "--artifact",
            str(application.root),
            "--cache-dir",
            str(application.cache),
            "--report",
            str(report_path),
            "--ece-threshold",
            "1",
        ]
    )

    assert status == 2
    report = json.loads(report_path.read_text())
    assert report["status"] == "refused" and report["derived"] is None
    assert report["quantization"]["attestedDisagreements"] == 5
    assert report["quantization"]["argmaxDisagreements"] == 46
    assert report["failures"][0] == "5 attested record(s) changed decision (tolerance 0)"
    assert report["failures"][1].startswith("46 of 46 records changed decision")
    assert report["next"].startswith("keep serving the float32 release")
    assert "--max-attested-disagreements 5 --max-decision-change-rate 1" in report["next"]
    assert "or try semantscript releases derive --int8 --per-channel" in report["next"]
    assert json.loads(capsys.readouterr().out)["status"] == "refused"
    # Nothing was published and the staging directory is gone.
    assert releases(application.root) == before

    # The tolerance the remedy names admits exactly these figures, and the
    # manifest records it.
    admitted = derive_int8_release(
        application.root,
        cache_directory=application.cache,
        quantization=QuantizationConfig(
            maximum_attested_disagreements=5,
            maximum_argmax_disagreement_rate=1.0,
            ece_threshold=1.0,
        ),
    )
    assert admitted.status == "published"

    # With --per-channel already on, the remedy repeats it and suggests nothing new.
    per_channel = derive_int8_release(
        application.root,
        cache_directory=application.cache,
        quantization=QuantizationConfig(per_channel=True, ece_threshold=1.0),
    )
    assert per_channel.status == "refused"
    assert "or try" not in per_channel.report["next"]
    assert (
        "derive --int8 --per-channel --max-attested-disagreements 5" in (per_channel.report["next"])
    )
    derived = admitted.report["derived"]["manifestSha256"]
    manifest = json.loads(
        (application.root / "releases" / f"sha256-{derived}" / "manifest.json").read_text()
    )
    encoder = next(r for r in manifest["resources"] if r["role"] == "encoder")
    assert encoder["onnx"]["quantization"]["attestedDisagreementTolerance"] == 5
    assert encoder["onnx"]["quantization"]["verification"]["attestedDecisionChanges"] == 5


def test_missing_or_ambiguous_records_refuse_with_the_remedy(
    application: SimpleNamespace, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    empty = tmp_path / "empty-cache"
    with pytest.raises(DeriveError, match="has no training dataset") as raised:
        derive_int8_release(application.root, cache_directory=empty)
    assert raised.value.fix is not None
    assert raised.value.fix.startswith("pass --cache-dir the build cache the release was")
    assert application.first_dataset in raised.value.fix

    status = cli.main(
        ["derive-int8", "--artifact", str(application.root), "--cache-dir", str(empty)]
    )
    assert status == 1
    assert "; next: pass --cache-dir" in capsys.readouterr().err

    # Without the build cache record, two sidecars on one dataset are ambiguous.
    shutil.rmtree(application.cache / "applications")
    with pytest.raises(DeriveError, match="2 adversarial datasets"):
        derive_int8_release(application.root, cache_directory=application.cache)

    # A pinned sidecar the cache no longer holds is missing, not replaced.
    pinned_record = (
        application.cache / "applications" / APPLICATION / "functions" / FIRST / "function.json"
    )
    _write_json(
        pinned_record,
        {"datasetSha256": application.first_dataset, "adversarialDatasetSha256": "e" * 64},
    )
    with pytest.raises(DeriveError, match="no longer holds"):
        derive_int8_release(application.root, cache_directory=application.cache)


def test_held_out_records_join_through_the_hook(application: SimpleNamespace) -> None:
    manifest = json.loads(
        (
            application.root / "releases" / f"sha256-{application.digest}" / "manifest.json"
        ).read_text()
    )
    held = RecordSource("held-out", SECOND, "f" * 64, application.cache / "held-out.json", 2)

    def provider(release_manifest: Any, cache: Path) -> list[HeldOutRecords]:
        assert release_manifest["functions"][0]["id"] == FIRST and cache == application.cache
        return [
            HeldOutRecords(
                held,
                tuple(
                    QuantizationRecord(
                        {"facts": {"a": index, "b": 2}}, function_id=SECOND, label_indices=(2,)
                    )
                    for index in range(2)
                ),
            )
        ]

    records, sources = verification_records(manifest, application.cache, held_out=provider)
    assert len(records) == 48 and sources[-1] is held

    result = derive_int8_release(
        application.root, cache_directory=application.cache, held_out=provider
    )
    assert result.report["heldOut"] == {"records": 2, "note": None}
    assert result.report["recordSources"][-1]["kind"] == "held-out"
    derived = result.report["derived"]["manifestSha256"]
    manifest = json.loads(
        (application.root / "releases" / f"sha256-{derived}" / "manifest.json").read_text()
    )
    encoder = next(r for r in manifest["resources"] if r["role"] == "encoder")
    assert encoder["onnx"]["quantization"]["verification"]["recordSources"][-1] == {
        "kind": "held-out",
        "functionId": SECOND,
        "sha256": "f" * 64,
        "records": 2,
    }


def test_resolves_the_named_release_and_refuses_an_int8_source(
    application: SimpleNamespace,
) -> None:
    assert resolve_release(application.root, None) == application.digest
    assert resolve_release(application.root, application.digest[:9]) == application.digest
    assert (
        resolve_release(application.root, f"releases/sha256-{application.digest}")
        == application.digest
    )
    with pytest.raises(DeriveError, match="is not a release"):
        resolve_release(application.root, "xyz")
    with pytest.raises(DeriveError, match="no release under"):
        resolve_release(application.root, "0000000")

    derived = derive_int8_release(application.root, cache_directory=application.cache)
    digest = derived.report["derived"]["manifestSha256"]
    with pytest.raises(DeriveError, match="already int8-dynamic"):
        derive_int8_release(application.root, cache_directory=application.cache, release=digest)


def test_multi_function_records_must_name_their_function(application: SimpleNamespace) -> None:
    with pytest.raises(ArtifactConfigurationError, match="needs a function_id"):
        quantize_release_artifact(
            application.root,
            application.digest,
            records=[QuantizationRecord({"facts": {"a": 1, "b": 2}}, 2)],
        )
    with pytest.raises(ArtifactConfigurationError, match="does not contain"):
        quantize_release_artifact(
            application.root,
            application.digest,
            records=[QuantizationRecord({"facts": {"a": 1, "b": 2}}, 2, function_id="nf_x")],
        )
    with pytest.raises(ArtifactConfigurationError, match=r"2 label\(s\) for 1 head"):
        quantize_release_artifact(
            application.root,
            application.digest,
            records=[
                QuantizationRecord(
                    {"facts": {"a": 1, "b": 2}}, function_id=FIRST, label_indices=(2, 0)
                )
            ],
        )
    with pytest.raises(ArtifactConfigurationError, match="not both"):
        QuantizationRecord({}, 1, label_indices=(1,))
    with pytest.raises(ArtifactConfigurationError, match="record_sources"):
        quantize_release_artifact(
            application.root,
            application.digest,
            records=[QuantizationRecord({"facts": {"a": 1, "b": 2}}, 2, function_id=FIRST)],
            record_sources=[{"kind": "teacher"}],
        )


def test_the_validator_checks_the_recorded_verification_figures() -> None:
    from semantscript_trainer.artifact import _validate_onnx_precision

    verification = {
        "recordsChecked": 4,
        "attestedRecords": 1,
        "decisionChanges": 0,
        "attestedDecisionChanges": 0,
        "decisionChangeRate": 0.0,
        "sourceEce": 0.1,
        "quantizedEce": 0.1,
        "recordSources": [
            {"kind": "training-dataset", "functionId": FIRST, "sha256": "a" * 64, "records": 4}
        ],
        "functions": [
            {
                "id": FIRST,
                "recordsChecked": 4,
                "attestedRecords": 1,
                "decisionChanges": 0,
                "attestedDecisionChanges": 0,
                "sourceEce": None,
                "quantizedEce": None,
            }
        ],
    }
    settings = QuantizationConfig().to_manifest_document("b" * 64)

    def onnx(**changes: Any) -> dict[str, Any]:
        return {
            "precision": "int8-dynamic",
            "quantization": {**settings, "verification": {**verification, **changes}},
        }

    _validate_onnx_precision(onnx(), "encoder")
    # Releases derived before the figures were recorded stay valid.
    _validate_onnx_precision({"precision": "int8-dynamic", "quantization": settings}, "encoder")
    for changes, match in (
        ({"attestedDecisionChanges": 2}, "exceeds attestedRecords"),
        ({"decisionChanges": 5}, "exceeds recordsChecked"),
        ({"quantizedEce": 1.5}, "quantizedEce"),
        ({"recordSources": [{"kind": "teacher"}]}, "recordSources"),
        ({"functions": [{"id": FIRST}]}, "function fields"),
        ({"extra": 1}, "exactly the verification fields"),
    ):
        with pytest.raises(ArtifactConfigurationError, match=match):
            _validate_onnx_precision(onnx(**changes), "encoder")
