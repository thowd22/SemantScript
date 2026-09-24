from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from derive_multihead_artifact import (
    MAXIMUM_FUNCTIONS,
    DerivationError,
    derive_multihead_artifact,
)

SOURCE_FUNCTION_ID = "nf_" + "5" * 64
CREATED_AT = "2026-09-24T00:00:00Z"


def onnx_descriptor(inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {"opset": 17, "inputs": inputs, "outputs": outputs, "externalData": False}


def write_source_release(root: Path, *, functions: list[dict[str, Any]] | None = None) -> str:
    release_files = {
        "tokenizer/tokenizer.json": b'{"version":"1.0"}',
        "models/encoder/model.onnx": b"encoder-bytes" * 100,
        "models/adapters/application.onnx": b"adapter-bytes",
        f"models/heads/{SOURCE_FUNCTION_ID}/head-000.onnx": b"head-bytes",
    }
    staging = root / "staging"
    for relative, content in release_files.items():
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def resource(relative: str, ref: str, role: str, onnx: dict[str, Any] | None) -> dict[str, Any]:
        content = release_files[relative]
        entry: dict[str, Any] = {
            "ref": ref,
            "role": role,
            "format": "tokenizer-json" if onnx is None else "onnx",
            "formatVersion": 1,
            "path": relative,
            "byteLength": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        if onnx is None:
            entry["maximumSequenceLength"] = 128
        else:
            entry["onnx"] = onnx
        return entry

    embedding = {"name": "sentence_embedding", "dtype": "float32", "shape": ["BATCH", 4]}
    function_embedding = {"name": "function_embedding", "dtype": "float32", "shape": ["BATCH", 4]}
    manifest = {
        "kind": "semantscript.application-artifact",
        "artifactVersion": 1,
        "irVersion": 1,
        "compatibility": {
            "runtimeAbiVersion": 1,
            "modelAbiVersion": 1,
            "canonicalInput": "semantscript.canonical-input/v2",
            "minimumRuntimeVersion": "0.0.0",
            "requiredCapabilities": [],
        },
        "application": {"id": "fixture", "version": "1.0.0"},
        "build": {
            "createdAt": "2026-09-23T00:00:00Z",
            "compilerVersion": "0.0.0",
            "trainerVersion": "0.0.0",
            "sourceIrSha256": "9" * 64,
        },
        "resources": [
            resource("tokenizer/tokenizer.json", "tokenizer.main", "tokenizer", None),
            resource(
                "models/encoder/model.onnx",
                "encoder.main",
                "encoder",
                onnx_descriptor(
                    [
                        {"name": "input_ids", "dtype": "int64", "shape": ["BATCH", "SEQUENCE"]},
                        {
                            "name": "attention_mask",
                            "dtype": "int64",
                            "shape": ["BATCH", "SEQUENCE"],
                        },
                    ],
                    [embedding],
                ),
            ),
            resource(
                "models/adapters/application.onnx",
                "adapter.application",
                "adapter",
                onnx_descriptor([embedding], [function_embedding]),
            ),
            resource(
                f"models/heads/{SOURCE_FUNCTION_ID}/head-000.onnx",
                "head.fixture.000",
                "head",
                onnx_descriptor(
                    [function_embedding],
                    [{"name": "logits", "dtype": "float32", "shape": ["BATCH", 3]}],
                ),
            ),
        ],
        "model": {
            "tokenizerRef": "tokenizer.main",
            "encoderRef": "encoder.main",
            "adapterRef": "adapter.application",
        },
        "functions": functions
        if functions is not None
        else [
            {
                "id": SOURCE_FUNCTION_ID,
                "semanticSha256": "6" * 64,
                "inputs": [],
                "inputSchemaSha256": "7" * 64,
                "outputSchemaSha256": "8" * 64,
                "adapterRef": "adapter.application",
                "heads": [{"outputPath": [], "headRef": "head.fixture.000"}],
                "runtime": {
                    "resultMode": "value",
                    "confidenceThreshold": None,
                    "policy": "none",
                    "fallbackRef": None,
                },
                "verification": {"status": "passed"},
                "trainingProvenance": {"datasetSha256": "1" * 64},
            }
        ],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    (staging / "manifest.json").write_bytes(manifest_bytes)
    (root / "releases").mkdir()
    staging.rename(root / "releases" / f"sha256-{digest}")
    (root / "current.json").write_text(
        json.dumps(
            {
                "kind": "semantscript.artifact-pointer",
                "pointerVersion": 1,
                "release": f"releases/sha256-{digest}",
                "manifestSha256": digest,
            }
        )
    )
    return digest


def test_clones_the_verified_function_over_shared_resources(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source_digest = write_source_release(source)
    output = tmp_path / "derived"

    record = derive_multihead_artifact(source, output, functions=3, created_at=CREATED_AT)

    pointer = json.loads((output / "current.json").read_text())
    release = output / pointer["release"]
    manifest_bytes = (release / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == pointer["manifestSha256"]
    assert record["manifestSha256"] == pointer["manifestSha256"]
    assert record["sourceManifestSha256"] == source_digest
    assert record["functions"] == 3
    manifest = json.loads(manifest_bytes)
    assert manifest["application"] == {"id": "fixture-multihead", "version": "1.0.0"}
    assert manifest["build"]["createdAt"] == CREATED_AT
    assert manifest["build"]["sourceIrSha256"] == "9" * 64
    assert len(manifest["functions"]) == 3
    ids = [fn["id"] for fn in manifest["functions"]]
    assert ids == record["functionIds"]
    assert len(set(ids)) == 3 and SOURCE_FUNCTION_ID not in ids
    head_refs = [fn["heads"][0]["headRef"] for fn in manifest["functions"]]
    assert len(set(head_refs)) == 3
    for fn in manifest["functions"]:
        assert fn["semanticSha256"] == "6" * 64
        assert fn["adapterRef"] == "adapter.application"
        assert fn["heads"][0]["outputPath"] == []
    roles = [resource["role"] for resource in manifest["resources"]]
    assert roles == ["tokenizer", "encoder", "adapter", "head", "head", "head"]
    assert [r["ref"] for r in manifest["resources"] if r["role"] == "head"] == head_refs
    for resource in manifest["resources"]:
        content = (release / resource["path"]).read_bytes()
        assert len(content) == resource["byteLength"]
        assert hashlib.sha256(content).hexdigest() == resource["sha256"]
    head_paths = [r["path"] for r in manifest["resources"] if r["role"] == "head"]
    assert head_paths == [f"models/heads/{function_id}/head-000.onnx" for function_id in ids]
    # Shared resources are the same bytes as the source (linked or copied).
    source_release = source / json.loads((source / "current.json").read_text())["release"]
    for relative in ("tokenizer/tokenizer.json", "models/encoder/model.onnx"):
        assert (release / relative).read_bytes() == (source_release / relative).read_bytes()
    assert (output / "staging").exists() is False

    # A second derivation with the same inputs reproduces the manifest digest.
    again = derive_multihead_artifact(
        source, tmp_path / "again", functions=3, created_at=CREATED_AT
    )
    assert again["manifestSha256"] == record["manifestSha256"]


@pytest.mark.parametrize("functions", [0, -1, MAXIMUM_FUNCTIONS + 1, True])
def test_rejects_unusable_function_counts(tmp_path: Path, functions: Any) -> None:
    write_source_release(tmp_path / "source")
    with pytest.raises(DerivationError, match="functions must be"):
        derive_multihead_artifact(tmp_path / "source", tmp_path / "derived", functions=functions)


def test_refuses_existing_output_multi_function_and_object_sources(tmp_path: Path) -> None:
    write_source_release(tmp_path / "source")
    (tmp_path / "taken").mkdir()
    with pytest.raises(DerivationError, match="already exists"):
        derive_multihead_artifact(tmp_path / "source", tmp_path / "taken", functions=1)

    base = json.loads(
        (
            tmp_path
            / "source"
            / json.loads((tmp_path / "source" / "current.json").read_text())["release"]
            / "manifest.json"
        ).read_bytes()
    )["functions"][0]
    second = json.loads(json.dumps(base))
    second["id"] = "nf_" + "a" * 64
    write_source_release(tmp_path / "two", functions=[base, second])
    with pytest.raises(DerivationError, match="single-function"):
        derive_multihead_artifact(tmp_path / "two", tmp_path / "derived-two", functions=2)

    flat = json.loads(json.dumps(base))
    flat["heads"][0]["outputPath"] = ["field"]
    write_source_release(tmp_path / "flat", functions=[flat])
    with pytest.raises(DerivationError, match="scalar"):
        derive_multihead_artifact(tmp_path / "flat", tmp_path / "derived-flat", functions=2)
    assert not (tmp_path / "derived-flat").exists()


def test_refuses_a_pointer_that_does_not_match_its_manifest(tmp_path: Path) -> None:
    write_source_release(tmp_path / "source")
    pointer_path = tmp_path / "source" / "current.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["manifestSha256"] = "0" * 64
    pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(DerivationError, match="digest"):
        derive_multihead_artifact(tmp_path / "source", tmp_path / "derived", functions=1)
