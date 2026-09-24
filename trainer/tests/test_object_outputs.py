"""Flat interface outputs: one head per field from contract to release artifact."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from semantscript_trainer.canonical_input import serialize_canonical_inputs_string  # noqa: E402
from semantscript_trainer.dataset import DatasetCase, TrainingDataset  # noqa: E402
from semantscript_trainer.semantic_json import semantic_json_sha256  # noqa: E402
from semantscript_trainer.teacher import TeacherDescriptor  # noqa: E402
from semantscript_trainer.training import (  # noqa: E402
    TrainingConfig,
    train_corpus,
)
from semantscript_trainer.training_contract import (  # noqa: E402
    OutputHead,
    TrainingContractError,
    TrainingHeadContract,
    assemble_training_corpus,
    derive_output_heads,
    derive_training_head,
    output_labels,
)
from semantscript_trainer.verification import (  # noqa: E402
    HeadVerificationV1,
    verify_training_result,
)

ROOT = Path(__file__).parents[2]
TOKENIZER_JSON = (ROOT / "runtime" / "test" / "fixtures" / "tokenizer.json").read_bytes()
FUNCTION_ID = "nf_" + "c" * 64
VERIFIED_AT = "2026-09-24T00:00:00Z"
LABELS = ("low", "high")


def object_ir(
    *, examples: list[dict[str, Any]] | None = None, threshold: float | None = None
) -> dict[str, Any]:
    document = {
        "kind": "semantscript.neural-function",
        "irVersion": 1,
        "stage": "lowered",
        "id": FUNCTION_ID,
        "semanticSha256": "",
        "source": {
            "path": "object.sem.ts",
            "line": 1,
            "column": 1,
            "sourceSha256": "3" * 64,
        },
        "definition": {
            "template": [
                {"kind": "text", "text": "Rate "},
                {"kind": "input", "name": "text"},
            ],
            "examples": examples or [],
            "constraints": [],
        },
        "inputs": [{"name": "text", "index": 0, "tsType": "string", "type": {"kind": "string"}}],
        "output": {
            "kind": "object",
            "tsType": "Rating",
            "fields": [
                {
                    "name": "label",
                    "tsType": '"low" | "high"',
                    "head": {
                        "kind": "nominal",
                        "sourceKind": "string-union",
                        "support": list(LABELS),
                    },
                },
                {
                    "name": "flag",
                    "tsType": "boolean",
                    "head": {"kind": "nominal", "sourceKind": "boolean", "support": [False, True]},
                },
            ],
        },
        "runtime": {
            "resultMode": "value",
            "confidenceThreshold": threshold,
            "fallbackRef": None,
            "synchronous": True,
        },
    }
    document["semanticSha256"] = semantic_json_sha256(
        {key: document[key] for key in ("irVersion", "definition", "inputs", "output", "runtime")}
    )
    return document


def rating(text: str) -> dict[str, Any]:
    """The fixture's ground truth: ``label`` follows the prefix, ``flag`` the suffix."""

    prefix, suffix = text.split("-", 1)
    return {"label": "high" if prefix == "hot" else "low", "flag": suffix.startswith("odd")}


def rating_dataset(ir: dict[str, Any], texts: list[str], *, gold: int = 1) -> TrainingDataset:
    cases = tuple(
        DatasetCase({"text": text}, rating(text), "gold" if index < gold else "synthetic")
        for index, text in enumerate(texts)
    )
    return TrainingDataset(
        function_id=FUNCTION_ID,
        semantic_sha256=ir["semanticSha256"],
        requested_case_count=len(cases),
        teacher=TeacherDescriptor("fake", "base", "4" * 64),
        cases=cases,
        cache_key_sha256="5" * 64,
        payload_sha256="6" * 64,
        dataset_sha256="7" * 64,
    )


def test_derives_one_head_per_field_and_labels_every_head() -> None:
    ir = object_ir()

    heads = derive_output_heads(ir)

    assert heads == (
        OutputHead(
            "/label",
            TrainingHeadContract(kind="nominal", source_kind="string-union", support=LABELS),
        ),
        OutputHead(
            "/flag",
            TrainingHeadContract(kind="nominal", source_kind="boolean", support=(False, True)),
        ),
    )
    assert [head.field_name for head in heads] == ["label", "flag"]
    assert output_labels(heads, {"label": "high", "flag": False}) == (1, 0)
    with pytest.raises(TrainingContractError, match="missing field"):
        output_labels(heads, {"label": "high"})
    with pytest.raises(TrainingContractError, match="object output case must be an object"):
        output_labels(heads, "high")
    with pytest.raises(TrainingContractError, match="derive_output_heads"):
        derive_training_head(ir)

    texts = ["hot-odd", "cold-even", "hot-even", "cold-odd"]
    corpus = assemble_training_corpus(ir, rating_dataset(ir, texts, gold=0))

    assert corpus.heads == heads
    assert corpus.head == heads[0].contract
    assert corpus.logit_count == 3
    assert [row.label_indices for row in corpus.rows] == [(1, 1), (0, 0), (1, 0), (0, 1)]
    assert [row.label_index for row in corpus.rows] == [1, 0, 1, 0]


class PrefixSuffixTokenizer:
    """Two tokens per text: the prefix decides ``label``, the suffix decides ``flag``."""

    def __init__(self) -> None:
        # The release binds the tokenizer the verifier saw; publish the runtime fixture.
        self.semantscript_tokenizer_json = TOKENIZER_JSON
        self.texts: list[str] = []

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, Any]:
        assert add_special_tokens is True and padding is True and truncation is True
        assert max_length >= 2 and return_tensors == "pt"
        self.texts.extend(texts)
        ids = []
        for text in texts:
            value = text.split("=", 1)[1]
            prefix, suffix = value.split("-", 1)
            ids.append([1 if prefix == "hot" else 2, 3 if suffix.startswith("odd") else 4])
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
        }


class PrefixSuffixEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=6)
        # Wide enough for the runtime fixture tokenizer's ids as well as the test tokens.
        self.embedding = torch.nn.Embedding(64, 6)

    def forward(self, *, input_ids, attention_mask, return_dict):
        del attention_mask
        assert return_dict is True
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def fixture_texts() -> list[str]:
    return [
        f"{prefix}-{suffix}{index}"
        for index in range(4)
        for prefix in ("hot", "cold")
        for suffix in ("odd", "even")
    ]


def train_fixture(
    seed: int = 11, *, threshold: float | None = None
) -> tuple[dict[str, Any], TrainingDataset, Any, Any]:
    ir = object_ir(
        examples=[{"inputs": {"text": "hot-odd0"}, "output": rating("hot-odd0")}],
        threshold=threshold,
    )
    base = rating_dataset(ir, fixture_texts())
    corpus = assemble_training_corpus(ir, base)
    tokenizer = PrefixSuffixTokenizer()
    result = train_corpus(
        ir,
        corpus,
        config=TrainingConfig(
            epochs=40,
            batch_size=4,
            learning_rate=0.1,
            weight_decay=0,
            maximum_sequence_length=8,
            evaluation_ratio=0.25,
            seed=seed,
            device="cpu",
        ),
        tokenizer=tokenizer,
        encoder=PrefixSuffixEncoder(),
    )
    return ir, base, result, tokenizer


def test_trains_field_heads_offline_and_reports_per_field_accuracy() -> None:
    ir, _, result, tokenizer = train_fixture()

    assert [head.output_path for head in result.heads] == ["/label", "/flag"]
    assert result.logit_count == 3
    assert result.model.head.slices == ((0, 2), (2, 3))
    assert [head.config.output_size for head in result.model.head.heads] == [2, 1]
    assert result.training_row_count == 12
    assert result.held_out_row_count == 4
    assert result.held_out_accuracy == 1.0
    last = result.metrics[-1]
    assert last.held_out_field_accuracy == (1.0, 1.0)
    assert all(
        epoch.held_out_field_accuracy is not None
        and min(epoch.held_out_field_accuracy) >= epoch.held_out_accuracy
        for epoch in result.metrics
    )
    expected_texts = {
        serialize_canonical_inputs_string(ir["inputs"], {"text": text}, version=2)
        for text in fixture_texts()
    }
    assert set(tokenizer.texts) == expected_texts


def test_verifies_every_field_head_and_projects_object_metrics() -> None:
    ir, base, result, tokenizer = train_fixture()

    verification = verify_training_result(
        ir,
        result,
        base,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
    )

    heads = verification.metrics.heads
    assert [head.output_path for head in heads] == ["/label", "/flag"]
    assert all(isinstance(head, HeadVerificationV1) for head in heads)
    assert verification.status == "passed"
    assert verification.attested_cases == 1
    assert verification.metrics.accuracy == 1.0
    assert verification.metrics.ece == max(head.calibration.ece for head in heads)
    assert verification.metrics.brier == max(head.calibration.brier for head in heads)
    assert verification.metrics.pair_consistency == 1.0
    document = verification.to_ir_document()
    assert [head["outputPath"] for head in document["metrics"]["heads"]] == ["/label", "/flag"]
    assert verification.to_manifest_head_metadata(1) == heads[1].to_manifest_metadata()


def test_exports_object_function_and_node_returns_the_typed_object(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from semantscript_trainer.artifact import ArtifactProvenance, export_application_artifact

    provenance = ArtifactProvenance(
        application_id="object-fixture",
        application_version="0.0.0",
        compiler_version="0.0.0",
        trainer_version="0.0.0",
        created_at=VERIFIED_AT,
        training_key_sha256="5" * 64,
    )

    def export(root: Path, threshold: float | None):
        ir, base, result, tokenizer = train_fixture(threshold=threshold)
        verification = verify_training_result(
            ir, result, base, tokenizer=tokenizer, verified_at=VERIFIED_AT
        )
        verified = verified_ir(ir, result, verification)
        exported = export_application_artifact(
            root,
            verified,
            result,
            verification,
            tokenizer_json=TOKENIZER_JSON,
            source_ir_bytes=json.dumps(
                verified, ensure_ascii=False, separators=(",", ":")
            ).encode(),
            provenance=provenance,
            input_ids=torch.tensor([[1, 3]], dtype=torch.int64),
            attention_mask=torch.tensor([[1, 1]], dtype=torch.int64),
        )
        return exported, verification

    exported, verification = export(tmp_path / "artifact", None)

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
        f"models/heads/{FUNCTION_ID}/head-000.onnx",
        f"models/heads/{FUNCTION_ID}/head-001.onnx",
    }
    function = exported.manifest["functions"][0]
    assert [head["outputPath"] for head in function["heads"]] == [["label"], ["flag"]]
    assert [head["headRef"] for head in function["heads"]] == [
        "head.object.label",
        "head.object.flag",
    ]
    assert [head["type"] for head in function["heads"]] == [
        {"kind": "nominal-string", "support": ["low", "high"]},
        {"kind": "boolean", "support": [False, True]},
    ]
    assert [head["parameterization"] for head in function["heads"]] == [
        "categorical-softmax",
        "binary-sigmoid",
    ]
    for index, head in enumerate(function["heads"]):
        assert head["calibration"] == verification.to_manifest_head_metadata(index)["calibration"]
    assert function["runtime"]["policy"] == "none"
    resources = {resource["ref"]: resource for resource in exported.manifest["resources"]}
    assert resources["head.object.label"]["onnx"]["outputs"][0]["shape"] == ["BATCH", 2]
    assert resources["head.object.flag"]["onnx"]["outputs"][0]["shape"] == ["BATCH", 1]
    for resource in exported.manifest["resources"]:
        path = exported.release_directory.joinpath(*resource["path"].split("/"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == resource["sha256"]

    # A thresholded object function is published with the runtime's all-fields policy.
    thresholded, _ = export(tmp_path / "thresholded", 0.5)
    assert thresholded.manifest["functions"][0]["runtime"] == {
        "resultMode": "value",
        "confidenceThreshold": 0.5,
        "policy": "all-fields",
        "fallbackRef": None,
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
const value = runtime.__sema.call(functionId, { text: 'hot-odd9' });
const keys = Object.keys(value).sort().join(',');
if (keys !== 'flag,label') throw new Error(`unexpected object shape ${keys}`);
if (value.label !== 'low' && value.label !== 'high') throw new Error('label outside support');
if (typeof value.flag !== 'boolean') throw new Error('flag is not a boolean');
const stage = handle.callStage([{ functionId, inputs: { text: 'hot-odd9' } }]);
if (stage.passes.head !== 2) throw new Error(`expected two head passes, saw ${stage.passes.head}`);
if (JSON.stringify(stage.results[0]) !== JSON.stringify(value)) throw new Error('stage differs');
await handle.close();
"""
    completed = subprocess.run(
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
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr


def verified_ir(ir: dict[str, Any], result: Any, verification: Any) -> dict[str, Any]:
    document = json.loads(json.dumps(ir))
    document["stage"] = "verified"
    document["model"] = {
        "encoder": "encoder.main",
        "adapter": "adapter.application",
        "heads": [
            {"outputPath": "/label", "ref": "head.object.label"},
            {"outputPath": "/flag", "ref": "head.object.flag"},
        ],
    }
    document["trainingProvenance"] = {
        "status": "complete",
        "teacher": {"provider": "fake", "model": "base", "configurationSha256": "4" * 64},
        "baseModel": {
            "name": result.config.encoder_name,
            "revision": result.config.encoder_revision,
            "weightsSha256": "8" * 64,
        },
        "datasetSha256": result.base_dataset_sha256,
        "counts": {
            "examples": 1,
            "synthetic": 15,
            "adversarial": 0,
            "calibration": 4,
            "verification": 16,
            "attestedVerification": 1,
        },
        "seed": 1,
        "trainer": {"version": "0.0.0", "commit": "abcdef0"},
        "trainedAt": VERIFIED_AT,
        "canonicalInput": "semantscript.canonical-input/v2",
    }
    document["verification"] = verification.to_ir_document()
    return document
