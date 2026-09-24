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

from semantscript_model import (  # noqa: E402
    AdapterConfig,
    ApplicationAdapter,
    ClassificationHead,
    HeadConfig,
    SentenceEncoder,
    SharedEncoderApplication,
)
from semantscript_trainer.artifact import (  # noqa: E402
    ArtifactConfigurationError,
    ArtifactFunction,
    ArtifactProvenance,
    export_multi_function_artifact,
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
REVISION = "7" * 40
INPUTS = [{"name": "text", "index": 0, "tsType": "string", "type": {"kind": "string"}}]
FUNCTIONS = {
    "nf_" + "a" * 64: ("a", "b", "c"),
    "nf_" + "b" * 64: ("no", "yes"),
}


class TokenEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = torch.nn.Embedding(64, 8)

    def forward(self, *, input_ids, attention_mask, return_dict):
        del attention_mask
        assert return_dict is True
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def build_application() -> Any:
    torch.manual_seed(3)
    adapter = ApplicationAdapter(AdapterConfig(hidden_size=8, bottleneck_size=4))
    with torch.no_grad():
        adapter.up.weight.normal_()
    return SharedEncoderApplication(
        SentenceEncoder(encoder=TokenEncoder()),
        adapter,
        {
            function_id: ClassificationHead(
                HeadConfig(input_size=8, kind="categorical-softmax", cardinality=len(support))
            )
            for function_id, support in FUNCTIONS.items()
        },
    )


def function_records(
    application: Any, adapter_ref: str = "adapter.application"
) -> list[ArtifactFunction]:
    entries = []
    for function_id, support in FUNCTIONS.items():
        view = application.function_model(function_id)
        definition = {
            "template": [
                {"kind": "text", "text": f"Classify {function_id[:6]} "},
                {"kind": "input", "name": "text"},
            ],
            "examples": [],
            "constraints": [],
        }
        output = {
            "kind": "scalar",
            "tsType": " | ".join(f'"{value}"' for value in support),
            "head": {"kind": "nominal", "sourceKind": "string-union", "support": list(support)},
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
                "inputs": INPUTS,
                "output": output,
                "runtime": runtime,
            }
        )
        contract = TrainingHeadContract(kind="nominal", source_kind="string-union", support=support)
        rows = (
            TrainingRow(
                row_id="t:0",
                group_id="t:0",
                origin="synthetic",
                inputs={"text": "alpha"},
                label_index=0,
            ),
            TrainingRow(
                row_id="e:0",
                group_id="e:0",
                origin="synthetic",
                inputs={"text": "beta"},
                label_index=1,
            ),
        )
        training = TrainingResult(
            model=view,
            head=contract,
            split=TrainingSplit(training=(rows[0],), evaluation=(rows[1],)),
            config=TrainingConfig(
                encoder_name="fixture/encoder",
                encoder_revision=REVISION,
                maximum_sequence_length=128,
            ),
            device="cpu",
            metrics=(EpochMetrics(epoch=1, mean_training_loss=0.1, held_out_accuracy=1.0),),
            function_id=function_id,
            semantic_sha256=semantic_sha256,
            base_dataset_sha256="3" * 64,
            adversarial_dataset_sha256=None,
        )
        calibration = CalibrationRecordV1(
            temperature=1.0, ece=0.0, brier=0.0, sample_count=1, split_sha256="6" * 64, ece_bins=2
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
            function_id=function_id,
            semantic_sha256=semantic_sha256,
            model_state_sha256=model_state_sha256(view),
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
            "id": function_id,
            "semanticSha256": semantic_sha256,
            "source": {"path": "src/app.sem.ts", "line": 1, "column": 1, "sourceSha256": "9" * 64},
            "definition": definition,
            "inputs": INPUTS,
            "output": output,
            "model": {
                "encoder": "encoder.main",
                "adapter": adapter_ref,
                "heads": [{"outputPath": "", "ref": f"head.{function_id[3:11]}.value"}],
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
                "canonicalInput": "semantscript.canonical-input/v2",
            },
            "verification": verification.to_ir_document(),
        }
        source = json.dumps(ir, ensure_ascii=False, separators=(",", ":")).encode()
        entries.append(ArtifactFunction(ir, training, verification, source))
    return entries


def provenance() -> ArtifactProvenance:
    return ArtifactProvenance(
        application_id="application-fixture",
        application_version="0.0.0",
        compiler_version="0.0.0",
        trainer_version="0.0.0",
        created_at="2026-09-24T00:00:00Z",
        training_key_sha256="5" * 64,
    )


def export(tmp_path: Path, entries: list[ArtifactFunction]):
    return export_multi_function_artifact(
        tmp_path / "artifact",
        entries,
        tokenizer_json=TOKENIZER_JSON,
        provenance=provenance(),
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.int64),
    )


def test_exports_two_functions_over_one_encoder_and_adapter_and_node_calls_both(
    tmp_path: Path,
) -> None:
    application = build_application()
    entries = function_records(application)

    exported = export(tmp_path, entries)

    manifest = exported.manifest
    assert [fn["id"] for fn in manifest["functions"]] == list(FUNCTIONS)
    assert [resource["role"] for resource in manifest["resources"]] == [
        "tokenizer",
        "encoder",
        "adapter",
        "head",
        "head",
    ]
    assert manifest["model"] == {
        "tokenizerRef": "tokenizer.main",
        "encoderRef": "encoder.main",
        "adapterRef": "adapter.application",
    }
    assert {fn["adapterRef"] for fn in manifest["functions"]} == {"adapter.application"}
    assert manifest["compatibility"]["canonicalInput"] == "semantscript.canonical-input/v2"
    digests = [hashlib.sha256(entry.source_ir_bytes).hexdigest() for entry in entries]
    assert (
        manifest["build"]["sourceIrSha256"]
        == hashlib.sha256("\n".join(digests).encode()).hexdigest()
    )
    files = {
        path.relative_to(exported.release_directory).as_posix()
        for path in exported.release_directory.rglob("*")
        if path.is_file()
    }
    assert files == {
        "manifest.json",
        "tokenizer/tokenizer.json",
        "models/encoder/model.onnx",
        "models/adapters/application.onnx",
        *(f"models/heads/{function_id}/head-000.onnx" for function_id in FUNCTIONS),
    }
    for resource in manifest["resources"]:
        path = exported.release_directory.joinpath(*resource["path"].split("/"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == resource["sha256"]
    pointer = json.loads((tmp_path / "artifact" / "current.json").read_text())
    assert pointer["manifestSha256"] == exported.manifest_sha256

    runtime_entry = ROOT / "runtime" / "dist" / "index.js"
    if not runtime_entry.is_file():
        pytest.skip("runtime/dist is produced by the repository build gate")
    script = """
import process from 'node:process';
const [runtimeUrl, artifactRoot, ...functionIds] = process.argv.slice(1);
const runtime = await import(runtimeUrl);
const handle = await runtime.loadSemaArtifact(artifactRoot);
if (handle.functionIds.size !== functionIds.length) throw new Error('unexpected function count');
for (const functionId of functionIds) {
  if (!handle.functionIds.has(functionId)) throw new Error(`missing ${functionId}`);
  const value = runtime.__sema.call(functionId, { text: 'semantscript input true' });
  if (typeof value !== 'string') throw new Error(`unexpected value for ${functionId}`);
}
// Both functions as one stage: one encoder pass, one adapter pass, two heads.
const stage = handle.callStage(functionIds.map((functionId) => ({ functionId, inputs: { text: 'semantscript input true' } })));
if (stage.passes.encoder !== 1 || stage.passes.adapter !== 1 || stage.passes.head !== functionIds.length) {
  throw new Error(`unexpected stage passes ${JSON.stringify(stage.passes)}`);
}
for (const [index, functionId] of functionIds.entries()) {
  if (stage.results[index] !== runtime.__sema.call(functionId, { text: 'semantscript input true' })) {
    throw new Error(`stage result differs from a single call for ${functionId}`);
  }
}
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
            *FUNCTIONS,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_rejects_functions_that_do_not_share_the_application(tmp_path: Path) -> None:
    application = build_application()
    entries = function_records(application)

    other = build_application()
    foreign = function_records(other)[1]
    with pytest.raises(ArtifactConfigurationError, match="share one encoder"):
        export(tmp_path / "foreign", [entries[0], foreign])

    with pytest.raises(ArtifactConfigurationError, match="more than once"):
        export(tmp_path / "twice", [entries[0], entries[0]])

    mismatched = function_records(application, adapter_ref="adapter.other")[1]
    with pytest.raises(ArtifactConfigurationError, match="encoder and adapter refs"):
        export(tmp_path / "refs", [entries[0], mismatched])

    same_head = deepcopy(entries[1].ir)
    same_head["model"]["heads"][0]["ref"] = entries[0].ir["model"]["heads"][0]["ref"]
    with pytest.raises(ArtifactConfigurationError, match="head ref"):
        export(
            tmp_path / "heads",
            [
                entries[0],
                ArtifactFunction(
                    same_head,
                    entries[1].training,
                    entries[1].verification,
                    json.dumps(same_head).encode(),
                ),
            ],
        )
    with pytest.raises(ArtifactConfigurationError, match="nonempty sequence"):
        export(tmp_path / "empty", [])
    assert not (tmp_path / "foreign" / "artifact" / "releases").exists() or not any(
        (tmp_path / "foreign" / "artifact" / "releases").iterdir()
    )
