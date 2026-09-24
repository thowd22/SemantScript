from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

from semantscript_model import (  # noqa: E402
    ClassificationHead,
    HeadConfig,
    SemanticClassifier,
    SentenceEncoder,
)
from semantscript_trainer import quantization as quantization_module  # noqa: E402
from semantscript_trainer.artifact import (  # noqa: E402
    ArtifactConfigurationError,
    ArtifactExportConfig,
    ArtifactProvenance,
    export_application_artifact,
)
from semantscript_trainer.quantization import (  # noqa: E402
    QuantizationConfig,
    QuantizationGateError,
    QuantizationRecord,
    QuantizedArtifact,
    quantize_release_artifact,
)
from semantscript_trainer.semantic_json import semantic_json_sha256  # noqa: E402
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
    VerificationMetricsV1,
    VerificationResult,
    model_state_sha256,
)

ROOT = Path(__file__).parents[2]
TOKENIZER_JSON = (ROOT / "runtime" / "test" / "fixtures" / "tokenizer.json").read_bytes()
FUNCTION_ID = f"nf_{'2' * 64}"
REVISION = "7" * 40
INPUT_SCHEMA = [{"name": "text", "index": 0, "tsType": "string", "type": {"kind": "string"}}]


class LinearTokenEncoder(torch.nn.Module):
    """Embedding plus a Linear so dynamic quantization has a MatMul to convert."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = torch.nn.Embedding(64, 8)
        self.projection = torch.nn.Linear(8, 8)

    def forward(self, *, input_ids, attention_mask, return_dict):
        del attention_mask
        assert return_dict is True
        return SimpleNamespace(last_hidden_state=self.projection(self.embedding(input_ids)))


def deterministic_model() -> Any:
    torch.manual_seed(5)
    inner = LinearTokenEncoder()
    with torch.no_grad():
        # Well separated weights keep decision margins far above int8 rounding.
        for row in range(64):
            inner.embedding.weight[row].zero_()
            inner.embedding.weight[row, row % 8] = 3.0
        inner.projection.weight.copy_(2.0 * torch.eye(8))
        inner.projection.bias.zero_()
    head = ClassificationHead(HeadConfig(input_size=8, kind="categorical-softmax", cardinality=3))
    with torch.no_grad():
        for parameter in head.parameters():
            if parameter.ndim == 2:
                parameter.zero_()
                for row in range(parameter.shape[0]):
                    parameter[row, row % parameter.shape[1]] = 4.0
            else:
                parameter.zero_()
    return SemanticClassifier(SentenceEncoder(encoder=inner), head)


def published_float32_release(root: Path) -> tuple[Any, dict[str, Any]]:
    model = deterministic_model()
    definition = {
        "template": [{"kind": "text", "text": "Classify "}, {"kind": "input", "name": "text"}],
        "examples": [],
        "constraints": [],
    }
    output = {
        "kind": "scalar",
        "tsType": '"a" | "b" | "c"',
        "head": {"kind": "nominal", "sourceKind": "string-union", "support": ["a", "b", "c"]},
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
            "inputs": INPUT_SCHEMA,
            "output": output,
            "runtime": runtime,
        }
    )
    contract = TrainingHeadContract(
        kind="nominal", source_kind="string-union", support=("a", "b", "c")
    )
    rows = (
        TrainingRow(
            row_id="t:0",
            group_id="t:0",
            origin="synthetic",
            inputs={"text": "alpha"},
            label_index=0,
        ),
        TrainingRow(
            row_id="e:0", group_id="e:0", origin="synthetic", inputs={"text": "beta"}, label_index=1
        ),
    )
    training = TrainingResult(
        model=model,
        head=contract,
        split=TrainingSplit(training=(rows[0],), evaluation=(rows[1],)),
        config=TrainingConfig(
            encoder_name="fixture/encoder", encoder_revision=REVISION, maximum_sequence_length=128
        ),
        device="cpu",
        metrics=(EpochMetrics(epoch=1, mean_training_loss=0.1, held_out_accuracy=1.0),),
        function_id=FUNCTION_ID,
        semantic_sha256=semantic_sha256,
        base_dataset_sha256="3" * 64,
        adversarial_dataset_sha256=None,
    )
    calibration = CalibrationRecordV1(
        temperature=1.5, ece=0.0, brier=0.0, sample_count=1, split_sha256="6" * 64, ece_bins=2
    )
    metrics = VerificationMetricsV1(
        accuracy=1.0,
        ece=0.0,
        brier=0.0,
        pair_consistency=1.0,
        heads=(
            HeadVerificationV1(
                output_path="", accuracy=1.0, pair_consistency=1.0, calibration=calibration
            ),
        ),
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
        verified_at="2026-09-24T00:00:00Z",
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
        "source": {"path": "src/fixture.sem.ts", "line": 1, "column": 1, "sourceSha256": "9" * 64},
        "definition": definition,
        "inputs": INPUT_SCHEMA,
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
                "weightsSha256": "4" * 64,
            },
            "datasetSha256": "3" * 64,
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
        },
        "verification": verification.to_ir_document(),
    }
    exported = export_application_artifact(
        root,
        ir,
        training,
        verification,
        tokenizer_json=TOKENIZER_JSON,
        source_ir_bytes=json.dumps(ir, ensure_ascii=False, separators=(",", ":")).encode(),
        provenance=ArtifactProvenance(
            application_id="artifact-fixture",
            application_version="0.0.0",
            compiler_version="0.0.0",
            trainer_version="0.0.0",
            created_at="2026-09-24T00:00:00Z",
            training_key_sha256="5" * 64,
        ),
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.int64),
    )
    return exported, exported.manifest


TEXTS = (
    "semantscript input true",
    "input",
    "semantscript",
    "true input",
    "input semantscript",
    "true",
)


def labeled_records(exported: Any, attested_count: int = 2) -> list[QuantizationRecord]:
    """Label every record with the float32 chain's own decision so accuracy is exactly one."""

    import numpy
    import onnxruntime
    from tokenizers import Tokenizer

    from semantscript_trainer.canonical_input import serialize_canonical_inputs

    release = exported.release_directory
    tokenizer = Tokenizer.from_str(TOKENIZER_JSON.decode("utf-8"))
    sessions = [
        onnxruntime.InferenceSession(str(release / path), providers=["CPUExecutionProvider"])
        for path in (
            "models/encoder/model.onnx",
            "models/adapters/application.onnx",
            f"models/heads/{FUNCTION_ID}/head-000.onnx",
        )
    ]
    records = []
    for index, text in enumerate(TEXTS):
        encoding = tokenizer.encode(
            serialize_canonical_inputs(INPUT_SCHEMA, {"text": text}).decode()
        )
        ids = numpy.asarray([encoding.ids], dtype=numpy.int64)
        mask = numpy.asarray([encoding.attention_mask], dtype=numpy.int64)
        embedding = sessions[0].run(None, {"input_ids": ids, "attention_mask": mask})[0]
        function_embedding = sessions[1].run(None, {"sentence_embedding": embedding})[0]
        logits = sessions[2].run(None, {"function_embedding": function_embedding})[0]
        records.append(
            QuantizationRecord(
                {"text": text}, int(logits[0].argmax()), attested=index < attested_count
            )
        )
    return records


def test_derives_verified_int8_release_and_node_runtime_loads_it(tmp_path: Path) -> None:
    exported, source_manifest = published_float32_release(tmp_path / "artifact")
    records = labeled_records(exported)

    quantized = quantize_release_artifact(
        tmp_path / "artifact",
        exported.manifest_sha256,
        records=records,
        quantization=QuantizationConfig(ece_threshold=1.0),
        created_at="2026-09-24T12:00:00Z",
    )

    assert isinstance(quantized, QuantizedArtifact)
    assert quantized.manifest_sha256 != exported.manifest_sha256
    assert quantized.release_directory.name == f"sha256-{quantized.manifest_sha256}"
    assert (
        hashlib.sha256(quantized.manifest_path.read_bytes()).hexdigest()
        == quantized.manifest_sha256
    )
    pointer = json.loads((tmp_path / "artifact" / "current.json").read_text())
    assert pointer["manifestSha256"] == quantized.manifest_sha256
    assert not any(
        path.name.startswith(".staging-") for path in (tmp_path / "artifact" / "releases").iterdir()
    )

    manifest = json.loads(quantized.manifest_path.read_text())
    encoder = next(resource for resource in manifest["resources"] if resource["role"] == "encoder")
    assert encoder["onnx"]["precision"] == "int8-dynamic"
    assert encoder["onnx"]["quantization"] == {
        "method": "dynamic",
        "weightType": "int8",
        "perChannel": False,
        "reduceRange": False,
        "argmaxDisagreementTolerance": 0.0,
        "attestedDisagreementTolerance": 0,
        "eceThreshold": 1.0,
        "sourceManifestSha256": exported.manifest_sha256,
    }
    source_encoder = next(r for r in source_manifest["resources"] if r["role"] == "encoder")
    assert encoder["sha256"] != source_encoder["sha256"]
    assert (
        encoder["byteLength"]
        == (quantized.release_directory / "models/encoder/model.onnx").stat().st_size
    )
    for role in ("tokenizer", "adapter", "head"):
        theirs = next(r for r in source_manifest["resources"] if r["role"] == role)
        ours = next(r for r in manifest["resources"] if r["role"] == role)
        assert ours == theirs
        assert (quantized.release_directory / ours["path"]).read_bytes() == (
            exported.release_directory / theirs["path"]
        ).read_bytes()
    expected = deepcopy(source_manifest)
    expected["build"]["createdAt"] = "2026-09-24T12:00:00Z"
    expected["resources"] = manifest["resources"]
    assert manifest == expected
    text = quantized.manifest_path.read_text()
    assert '"opset": 17,' in text and '"formatVersion": 1,' in text
    assert '"opset": 17.0' not in text and '"formatVersion": 1.0' not in text
    assert f'"byteLength": {encoder["byteLength"]},' in text

    report = quantized.report
    assert report.records_checked == len(TEXTS)
    assert report.labeled_records == len(TEXTS) and report.attested_records == 2
    assert report.argmax_disagreements == 0 and report.attested_disagreements == 0
    assert report.source_accuracy == 1.0 and report.quantized_accuracy == 1.0
    assert report.temperature == 1.5 and report.ece_bins == 15
    assert report.quantized_matmul_count >= 1
    assert report.source_encoder_byte_length == source_encoder["byteLength"]
    assert report.to_document()["sourceManifestSha256"] == exported.manifest_sha256

    with pytest.raises(ArtifactConfigurationError, match="already quantized"):
        quantize_release_artifact(tmp_path / "artifact", quantized.manifest_sha256, records=records)

    runtime_entry = ROOT / "runtime" / "dist" / "index.js"
    if not runtime_entry.is_file():
        pytest.skip("runtime/dist is produced by the repository build gate")
    script = """
import process from 'node:process';
const [runtimeUrl, artifactRoot, functionId] = process.argv.slice(1);
const runtime = await import(runtimeUrl);
const handle = await runtime.loadSemaArtifact(artifactRoot);
if (!handle.functionIds.has(functionId)) throw new Error('quantized function missing');
const value = runtime.__sema.call(functionId, { text: 'semantscript input true' });
if (!['a', 'b', 'c'].includes(value)) throw new Error('unexpected inference value');
await handle.close();
"""
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "--eval",
            script,
            runtime_entry.as_uri(),
            str(tmp_path / "artifact"),
            FUNCTION_ID,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_gate_refuses_changed_decisions_and_removes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy
    import onnx

    exported, _ = published_float32_release(tmp_path / "artifact")
    records = labeled_records(exported)

    def flipped_encoder(source: Path, destination: Path, settings: Any, config: Any) -> Any:
        model = onnx.load(str(source))
        for initializer in model.graph.initializer:
            if list(initializer.dims) == [8, 8]:
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

    monkeypatch.setattr(quantization_module, "_quantize_encoder", flipped_encoder)
    before = sorted(path.name for path in (tmp_path / "artifact" / "releases").iterdir())
    with pytest.raises(QuantizationGateError, match="attested record") as raised:
        quantize_release_artifact(
            tmp_path / "artifact",
            exported.manifest_sha256,
            records=records,
            quantization=QuantizationConfig(ece_threshold=1.0),
        )
    report = raised.value.report
    assert report.attested_disagreements >= 1
    assert report.argmax_disagreements >= report.attested_disagreements
    assert sorted(path.name for path in (tmp_path / "artifact" / "releases").iterdir()) == before
    pointer = json.loads((tmp_path / "artifact" / "current.json").read_text())
    assert pointer["manifestSha256"] == exported.manifest_sha256

    # The same change passes when the artifact records a tolerance that admits it.
    relaxed = quantize_release_artifact(
        tmp_path / "artifact",
        exported.manifest_sha256,
        records=records,
        quantization=QuantizationConfig(
            maximum_argmax_disagreement_rate=1.0,
            maximum_attested_disagreements=len(records),
            ece_threshold=1.0,
        ),
    )
    encoder = next(r for r in relaxed.manifest["resources"] if r["role"] == "encoder")
    assert encoder["onnx"]["quantization"]["argmaxDisagreementTolerance"] == 1.0
    assert encoder["onnx"]["quantization"]["attestedDisagreementTolerance"] == len(records)


def test_gate_refuses_calibration_drift(tmp_path: Path) -> None:
    exported, _ = published_float32_release(tmp_path / "artifact")
    records = labeled_records(exported)
    with pytest.raises(QuantizationGateError, match="ECE") as raised:
        quantize_release_artifact(
            tmp_path / "artifact",
            exported.manifest_sha256,
            records=records,
            quantization=QuantizationConfig(ece_threshold=0.0),
        )
    assert raised.value.report.quantized_ece > 0.0
    assert not any(
        path.name.startswith(".staging-") for path in (tmp_path / "artifact" / "releases").iterdir()
    )


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"weight_type": "int4"}, "weight_type"),
        ({"per_channel": 1}, "per_channel"),
        ({"maximum_argmax_disagreement_rate": 1.5}, "maximum_argmax_disagreement_rate"),
        ({"maximum_argmax_disagreement_rate": float("nan")}, "maximum_argmax_disagreement_rate"),
        ({"maximum_attested_disagreements": -1}, "maximum_attested_disagreements"),
        ({"maximum_attested_disagreements": True}, "maximum_attested_disagreements"),
        ({"ece_threshold": -0.1}, "ece_threshold"),
        ({"ece_bins": 1}, "ece_bins"),
    ],
)
def test_rejects_invalid_quantization_settings(arguments: dict[str, Any], match: str) -> None:
    with pytest.raises(ArtifactConfigurationError, match=match):
        QuantizationConfig(**arguments)


def test_rejects_unusable_records_and_sources(tmp_path: Path) -> None:
    exported, _ = published_float32_release(tmp_path / "artifact")
    with pytest.raises(ArtifactConfigurationError, match="label_index"):
        QuantizationRecord({"text": "x"}, -1)
    with pytest.raises(ArtifactConfigurationError, match="attested"):
        QuantizationRecord({"text": "x"}, 0, attested="yes")  # type: ignore[arg-type]
    with pytest.raises(ArtifactConfigurationError, match="at least one labeled"):
        quantize_release_artifact(
            tmp_path / "artifact",
            exported.manifest_sha256,
            records=[QuantizationRecord({"text": "x"})],
        )
    with pytest.raises(ArtifactConfigurationError, match="exceeds the head support"):
        quantize_release_artifact(
            tmp_path / "artifact",
            exported.manifest_sha256,
            records=[QuantizationRecord({"text": "x"}, 3)],
        )
    with pytest.raises(ArtifactConfigurationError, match="64 lowercase hexadecimal"):
        quantize_release_artifact(
            tmp_path / "artifact", "nope", records=[QuantizationRecord({"text": "x"}, 0)]
        )
    with pytest.raises(ArtifactConfigurationError, match="source release"):
        quantize_release_artifact(
            tmp_path / "artifact", "0" * 64, records=[QuantizationRecord({"text": "x"}, 0)]
        )
    with pytest.raises(ArtifactConfigurationError, match="not a valid function input"):
        quantize_release_artifact(
            tmp_path / "artifact",
            exported.manifest_sha256,
            records=[QuantizationRecord({"other": 1}, 0)],
        )
    assert not any(
        path.name.startswith(".staging-") for path in (tmp_path / "artifact" / "releases").iterdir()
    )
    with pytest.raises(ArtifactConfigurationError, match="config must be"):
        quantize_release_artifact(
            tmp_path / "artifact",
            exported.manifest_sha256,
            records=[QuantizationRecord({"text": "x"}, 0)],
            config=object(),  # type: ignore[arg-type]
        )
    tampered = exported.release_directory / "models" / "adapters" / "application.onnx"
    tampered.write_bytes(tampered.read_bytes() + b"\0")
    with pytest.raises(ArtifactConfigurationError, match="not intact"):
        quantize_release_artifact(
            tmp_path / "artifact",
            exported.manifest_sha256,
            records=[QuantizationRecord({"text": "x"}, 0)],
        )
    assert ArtifactExportConfig().publish_pointer is True
