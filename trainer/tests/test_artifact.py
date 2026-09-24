from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from semantscript_model import (  # noqa: E402
    ClassificationHead,
    HeadConfig,
    SemanticClassifier,
    SentenceEncoder,
)
from semantscript_trainer import artifact as artifact_module  # noqa: E402
from semantscript_trainer.artifact import (  # noqa: E402
    ArtifactConfigurationError,
    ArtifactExportConfig,
    ArtifactProvenance,
    ArtifactPublicationError,
    export_application_artifact,
)
from semantscript_trainer.semantic_json import (  # noqa: E402
    semantic_json_bytes,
    semantic_json_sha256,
)
from semantscript_trainer.training import (  # noqa: E402
    EpochMetrics,
    TrainingConfig,
    TrainingResult,
)
from semantscript_trainer.training_contract import (  # noqa: E402
    TrainingHeadContract,
    TrainingRow,
    TrainingSplit,
)
from semantscript_trainer.verification import (  # noqa: E402
    CalibrationRecordV1,
    HeadVerificationV1,
    VerificationGateError,
    VerificationMetricsV1,
    VerificationResult,
    model_state_sha256,
)

ROOT = Path(__file__).parents[2]
TOKENIZER_JSON = (ROOT / "runtime" / "test" / "fixtures" / "tokenizer.json").read_bytes()
FUNCTION_ID = f"nf_{'1' * 64}"
DATASET_SHA256 = "3" * 64
WEIGHTS_SHA256 = "4" * 64
TRAINING_KEY_SHA256 = "5" * 64
SPLIT_SHA256 = "6" * 64
REVISION = "7" * 40


class TinyTokenEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = torch.nn.Embedding(8, 4)

    def forward(self, *, input_ids, attention_mask, return_dict):
        del attention_mask
        assert return_dict is True
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_exports_content_addressed_artifact_and_node_loads_without_python(tmp_path: Path) -> None:
    training, verification, ir, source = fixture()

    exported = export_application_artifact(
        tmp_path / "artifact",
        ir,
        training,
        verification,
        tokenizer_json=TOKENIZER_JSON,
        source_ir_bytes=source,
        provenance=provenance(),
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.int64),
    )

    assert exported.release_directory.name == f"sha256-{exported.manifest_sha256}"
    assert hashlib.sha256(exported.manifest_path.read_bytes()).hexdigest() == (
        exported.manifest_sha256
    )
    expected = {
        "manifest.json",
        "tokenizer/tokenizer.json",
        "models/encoder/model.onnx",
        "models/adapters/application.onnx",
        f"models/heads/{FUNCTION_ID}/head-000.onnx",
    }
    assert {
        path.relative_to(exported.release_directory).as_posix()
        for path in exported.release_directory.rglob("*")
        if path.is_file()
    } == expected
    manifest = exported.manifest
    function = manifest["functions"][0]
    assert (
        function["heads"][0]["calibration"]
        == (verification.to_manifest_head_metadata()["calibration"])
    )
    assert function["verification"] == verification.to_manifest_function_verification()
    for resource in manifest["resources"]:
        resource_path = exported.release_directory.joinpath(*resource["path"].split("/"))
        resource_bytes = resource_path.read_bytes()
        assert len(resource_bytes) == resource["byteLength"]
        assert hashlib.sha256(resource_bytes).hexdigest() == resource["sha256"]

    pointer = json.loads((exported.artifact_root / "current.json").read_text())
    assert pointer == {
        "kind": "semantscript.artifact-pointer",
        "pointerVersion": 1,
        "release": f"releases/sha256-{exported.manifest_sha256}",
        "manifestSha256": exported.manifest_sha256,
    }

    runtime_entry = ROOT / "runtime" / "dist" / "index.js"
    if not runtime_entry.is_file():
        pytest.skip("runtime/dist is produced by the repository build gate")
    script = """
import process from 'node:process';
const [runtimeUrl, artifactRoot, functionId] = process.argv.slice(1);
const runtime = await import(runtimeUrl);
const handle = await runtime.loadSemaArtifact(artifactRoot);
if (!handle.functionIds.has(functionId)) throw new Error('exported function missing');
const value = runtime.__sema.call(functionId, { text: 'semantscript input true' });
if (value !== 'accept' && value !== 'reject') throw new Error('unexpected inference value');
await handle.close();
"""
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "--eval",
            script,
            runtime_entry.as_uri(),
            str(exported.artifact_root),
            FUNCTION_ID,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_rejects_stale_model_tokenizer_failed_evidence_and_wrong_source(tmp_path: Path) -> None:
    training, verification, ir, source = fixture()
    arguments: dict[str, Any] = {
        "tokenizer_json": TOKENIZER_JSON,
        "source_ir_bytes": source,
        "provenance": provenance(),
        "input_ids": torch.tensor([[1]], dtype=torch.int64),
        "attention_mask": torch.tensor([[1]], dtype=torch.int64),
    }

    with torch.no_grad():
        next(training.model.parameters()).add_(1)
    with pytest.raises(ArtifactConfigurationError, match="changed after verification"):
        export_application_artifact(
            tmp_path / "stale-model", ir, training, verification, **arguments
        )

    training, verification, ir, source = fixture()
    arguments["source_ir_bytes"] = source
    with pytest.raises(ArtifactConfigurationError, match="differs from the tokenizer"):
        export_application_artifact(
            tmp_path / "wrong-tokenizer",
            ir,
            training,
            verification,
            **{**arguments, "tokenizer_json": b"{}"},
        )

    failed = replace(verification, status="failed", failures=("gate failed",))
    with pytest.raises(VerificationGateError, match="gate failed"):
        export_application_artifact(
            tmp_path / "failed-verification", ir, training, failed, **arguments
        )

    with pytest.raises(ArtifactConfigurationError, match="do not encode"):
        export_application_artifact(
            tmp_path / "wrong-source",
            ir,
            training,
            verification,
            **{**arguments, "source_ir_bytes": b"{}"},
        )

    typed_source = source.replace(b'"seed":1', b'"seed":true')
    assert typed_source != source
    with pytest.raises(ArtifactConfigurationError, match="do not encode"):
        export_application_artifact(
            tmp_path / "typed-source",
            ir,
            training,
            verification,
            **{**arguments, "source_ir_bytes": typed_source},
        )

    stale_ir = deepcopy(ir)
    stale_ir["inputs"][0]["type"] = {"kind": "boolean"}
    stale_source = json.dumps(stale_ir, separators=(",", ":")).encode()
    with pytest.raises(ArtifactConfigurationError, match="semanticSha256 does not match"):
        export_application_artifact(
            tmp_path / "stale-semantic",
            stale_ir,
            training,
            verification,
            **{**arguments, "source_ir_bytes": stale_source},
        )

    boolean_version_ir = deepcopy(ir)
    boolean_version_ir["irVersion"] = True
    boolean_version_training, boolean_version_verification, boolean_version_source = (
        rebind_semantic(training, verification, boolean_version_ir)
    )
    with pytest.raises(ArtifactConfigurationError, match="neural-function IR v1"):
        export_application_artifact(
            tmp_path / "boolean-version",
            boolean_version_ir,
            boolean_version_training,
            boolean_version_verification,
            **{**arguments, "source_ir_bytes": boolean_version_source},
        )

    boolean_index_ir = deepcopy(ir)
    boolean_index_ir["inputs"][0]["index"] = False
    boolean_index_training, boolean_index_verification, boolean_index_source = rebind_semantic(
        training, verification, boolean_index_ir
    )
    with pytest.raises(ArtifactConfigurationError, match="dense and ordered"):
        export_application_artifact(
            tmp_path / "boolean-index",
            boolean_index_ir,
            boolean_index_training,
            boolean_index_verification,
            **{**arguments, "source_ir_bytes": boolean_index_source},
        )
    assert not any(tmp_path.rglob("current.json"))


@pytest.mark.parametrize(
    ("field", "value"),
    (("parity_relative_tolerance", 1e300), ("parity_absolute_tolerance", 0.002)),
)
def test_rejects_parity_tolerance_above_hard_limit(field: str, value: float) -> None:
    with pytest.raises(ArtifactConfigurationError, match=r"between 0 and 0\.001"):
        ArtifactExportConfig(**{field: value})


def test_refuses_to_reuse_tampered_immutable_release(tmp_path: Path) -> None:
    training, verification, ir, source = fixture()
    arguments = {
        "tokenizer_json": TOKENIZER_JSON,
        "source_ir_bytes": source,
        "provenance": provenance(),
        "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.int64),
    }
    exported = export_application_artifact(
        tmp_path / "artifact", ir, training, verification, **arguments
    )
    pointer_before = (exported.artifact_root / "current.json").read_bytes()
    head_path = exported.release_directory / f"models/heads/{FUNCTION_ID}/head-000.onnx"
    with head_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ArtifactPublicationError, match="resource is inconsistent"):
        export_application_artifact(tmp_path / "artifact", ir, training, verification, **arguments)

    assert (exported.artifact_root / "current.json").read_bytes() == pointer_before
    assert not list((exported.artifact_root / "releases").glob(".staging-*"))


def test_concurrent_empty_release_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training, verification, ir, source = fixture()
    original_rename = artifact_module._rename_directory_no_replace
    raced_destination: list[Path] = []

    def create_competing_release(
        staging: Path,
        destination: Path,
        parent_descriptor: int | None,
    ) -> None:
        destination.mkdir()
        raced_destination.append(destination)
        original_rename(staging, destination, parent_descriptor)

    monkeypatch.setattr(
        artifact_module,
        "_rename_directory_no_replace",
        create_competing_release,
    )
    artifact_root = tmp_path / "artifact"
    with pytest.raises(ArtifactPublicationError):
        export_application_artifact(
            artifact_root,
            ir,
            training,
            verification,
            tokenizer_json=TOKENIZER_JSON,
            source_ir_bytes=source,
            provenance=provenance(),
            input_ids=torch.tensor([[1, 2]], dtype=torch.int64),
            attention_mask=torch.tensor([[1, 1]], dtype=torch.int64),
        )

    assert len(raced_destination) == 1
    assert raced_destination[0].is_dir()
    assert list(raced_destination[0].iterdir()) == []
    assert not (artifact_root / "current.json").exists()
    assert not list((artifact_root / "releases").glob(".staging-*"))


@pytest.mark.skipif(not artifact_module._ANCHORED_DIRECTORY_OPERATIONS, reason="needs dir_fd")
def test_rejects_artifact_root_path_swap_without_redirecting_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training, verification, ir, source = fixture()
    artifact_root = tmp_path / "artifact"
    moved_root = tmp_path / "moved-artifact"
    redirect = tmp_path / "redirect"
    redirect.mkdir()
    original_create = artifact_module._create_staging_directory

    def swap_root_after_staging(parent: Path, parent_descriptor: int | None):
        staging, staging_descriptor = original_create(parent, parent_descriptor)
        artifact_root.rename(moved_root)
        artifact_root.symlink_to(redirect, target_is_directory=True)
        return staging, staging_descriptor

    monkeypatch.setattr(
        artifact_module,
        "_create_staging_directory",
        swap_root_after_staging,
    )
    with pytest.raises(ArtifactPublicationError, match="artifact_root path changed"):
        export_application_artifact(
            artifact_root,
            ir,
            training,
            verification,
            tokenizer_json=TOKENIZER_JSON,
            source_ir_bytes=source,
            provenance=provenance(),
            input_ids=torch.tensor([[1, 2]], dtype=torch.int64),
            attention_mask=torch.tensor([[1, 1]], dtype=torch.int64),
        )

    assert not any(path.is_file() for path in redirect.rglob("*"))
    assert not list((moved_root / "releases").glob(".staging-*"))
    assert not (moved_root / "current.json").exists()


def test_requires_strict_semantic_order_for_union_inputs(tmp_path: Path) -> None:
    training, verification, ir, _ = fixture()
    variants = sorted(
        ({"kind": "string"}, {"kind": "boolean"}),
        key=semantic_json_bytes,
    )
    valid_ir = deepcopy(ir)
    valid_ir["inputs"][0]["type"] = {"kind": "union", "variants": variants}
    valid_training, valid_verification, valid_source = rebind_semantic(
        training, verification, valid_ir
    )
    export_application_artifact(
        tmp_path / "valid-union",
        valid_ir,
        valid_training,
        valid_verification,
        tokenizer_json=TOKENIZER_JSON,
        source_ir_bytes=valid_source,
        provenance=provenance(),
        input_ids=torch.tensor([[1]], dtype=torch.int64),
        attention_mask=torch.tensor([[1]], dtype=torch.int64),
    )

    reversed_ir = deepcopy(valid_ir)
    reversed_ir["inputs"][0]["type"]["variants"].reverse()
    reversed_training, reversed_verification, reversed_source = rebind_semantic(
        training, verification, reversed_ir
    )
    with pytest.raises(ArtifactConfigurationError, match="strict semantic byte order"):
        export_application_artifact(
            tmp_path / "reversed-union",
            reversed_ir,
            reversed_training,
            reversed_verification,
            tokenizer_json=TOKENIZER_JSON,
            source_ir_bytes=reversed_source,
            provenance=provenance(),
            input_ids=torch.tensor([[1]], dtype=torch.int64),
            attention_mask=torch.tensor([[1]], dtype=torch.int64),
        )


def fixture() -> tuple[TrainingResult, VerificationResult, dict[str, Any], bytes]:
    torch.manual_seed(1)
    definition = {
        "template": [
            {"kind": "text", "text": "Classify "},
            {"kind": "input", "name": "text"},
        ],
        "examples": [],
        "constraints": [],
    }
    inputs = [{"name": "text", "index": 0, "tsType": "string", "type": {"kind": "string"}}]
    output = {
        "kind": "scalar",
        "tsType": '"accept" | "reject"',
        "head": {
            "kind": "nominal",
            "sourceKind": "string-union",
            "support": ["accept", "reject"],
        },
    }
    runtime = {
        "resultMode": "value",
        "confidenceThreshold": None,
        "fallbackRef": None,
        "synchronous": True,
    }
    semantic_sha256 = semantic_json_sha256(
        {
            "irVersion": 1,
            "definition": definition,
            "inputs": inputs,
            "output": output,
            "runtime": runtime,
        }
    )
    contract = TrainingHeadContract(
        kind="nominal",
        source_kind="string-union",
        support=("accept", "reject"),
    )
    encoder = SentenceEncoder(encoder=TinyTokenEncoder())
    head = ClassificationHead(HeadConfig(input_size=4, kind="categorical-softmax", cardinality=2))
    model = SemanticClassifier(encoder, head)
    training_row = TrainingRow(
        row_id="train:0",
        group_id="train:0",
        origin="gold",
        inputs={"text": "semantscript"},
        label_index=0,
    )
    evaluation_row = TrainingRow(
        row_id="eval:0",
        group_id="eval:0",
        origin="synthetic",
        inputs={"text": "input"},
        label_index=1,
    )
    training = TrainingResult(
        model=model,
        head=contract,
        split=TrainingSplit(training=(training_row,), evaluation=(evaluation_row,)),
        config=TrainingConfig(
            encoder_name="fixture/encoder",
            encoder_revision=REVISION,
            maximum_sequence_length=128,
        ),
        device="cpu",
        metrics=(EpochMetrics(epoch=1, mean_training_loss=0.1, held_out_accuracy=1.0),),
        function_id=FUNCTION_ID,
        semantic_sha256=semantic_sha256,
        base_dataset_sha256=DATASET_SHA256,
        adversarial_dataset_sha256=None,
    )
    calibration = CalibrationRecordV1(
        temperature=1.0,
        ece=0.0,
        brier=0.0,
        sample_count=1,
        split_sha256=SPLIT_SHA256,
        ece_bins=2,
    )
    verified_head = HeadVerificationV1(
        output_path="",
        accuracy=1.0,
        pair_consistency=1.0,
        calibration=calibration,
    )
    metrics = VerificationMetricsV1(
        accuracy=1.0,
        ece=0.0,
        brier=0.0,
        pair_consistency=1.0,
        heads=(verified_head,),
        example_failures=0,
        constraint_violations=0,
        type_errors=0,
    )
    verification = VerificationResult(
        function_id=FUNCTION_ID,
        semantic_sha256=semantic_sha256,
        model_state_sha256=model_state_sha256(model),
        tokenizer_sha256=hashlib.sha256(TOKENIZER_JSON).hexdigest(),
        status="passed",
        verified_at="2026-09-23T00:00:00Z",
        metrics=metrics,
        attested_cases=1,
        pair_count=0,
        failures=(),
    )
    ir: dict[str, Any] = {
        "kind": "semantscript.neural-function",
        "irVersion": 1,
        "stage": "verified",
        "id": FUNCTION_ID,
        "semanticSha256": semantic_sha256,
        "source": {
            "path": "src/fixture.sem.ts",
            "line": 1,
            "column": 1,
            "sourceSha256": "9" * 64,
        },
        "definition": definition,
        "inputs": inputs,
        "output": output,
        "model": {
            "encoder": "encoder.main",
            "adapter": "adapter.application",
            "heads": [{"outputPath": "", "ref": "head.fixture.value"}],
        },
        "runtime": runtime,
        "trainingProvenance": {
            "status": "complete",
            "teacher": {
                "provider": "fixture",
                "model": "deterministic",
                "configurationSha256": "8" * 64,
            },
            "baseModel": {
                "name": "fixture/encoder",
                "revision": REVISION,
                "weightsSha256": WEIGHTS_SHA256,
            },
            "datasetSha256": DATASET_SHA256,
            "counts": {
                "examples": 0,
                "synthetic": 1,
                "adversarial": 0,
                "calibration": 1,
                "verification": 2,
                "attestedVerification": 1,
            },
            "seed": 1,
            "trainer": {"version": "0.0.0", "commit": "abcdef0"},
            "trainedAt": "2026-09-23T00:00:00Z",
            "canonicalInput": "semantscript.canonical-input/v2",
        },
        "verification": verification.to_ir_document(),
    }
    source = json.dumps(ir, ensure_ascii=False, separators=(",", ":")).encode()
    return training, verification, ir, source


def provenance() -> ArtifactProvenance:
    return ArtifactProvenance(
        application_id="artifact-fixture",
        application_version="0.0.0",
        compiler_version="0.0.0",
        trainer_version="0.0.0",
        created_at="2026-09-23T00:00:00Z",
        training_key_sha256=TRAINING_KEY_SHA256,
    )


def rebind_semantic(
    training: TrainingResult,
    verification: VerificationResult,
    ir: dict[str, Any],
) -> tuple[TrainingResult, VerificationResult, bytes]:
    digest = semantic_json_sha256(
        {
            "irVersion": ir["irVersion"],
            "definition": ir["definition"],
            "inputs": ir["inputs"],
            "output": ir["output"],
            "runtime": ir["runtime"],
        }
    )
    ir["semanticSha256"] = digest
    rebound_training = replace(training, semantic_sha256=digest)
    rebound_verification = replace(verification, semantic_sha256=digest)
    ir["verification"] = rebound_verification.to_ir_document()
    source = json.dumps(ir, ensure_ascii=False, separators=(",", ":")).encode()
    return rebound_training, rebound_verification, source
