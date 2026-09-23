from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from semantscript_trainer.artifact import validate_verified_ir_binding
from semantscript_trainer.lifecycle import (
    BuiltVerifiedIr,
    TrainingProvenanceCounts,
    VerifiedIrBuildError,
    VerifiedIrProvenance,
    build_verified_ir,
)
from semantscript_trainer.semantic_json import semantic_json_sha256
from semantscript_trainer.teacher import TeacherDescriptor
from semantscript_trainer.training import EpochMetrics, TrainingConfig, TrainingResult
from semantscript_trainer.training_contract import (
    TrainingHeadContract,
    TrainingRow,
    TrainingSplit,
)
from semantscript_trainer.verification import (
    CalibrationRecordV1,
    HeadVerificationV1,
    VerificationMetricsV1,
    VerificationResult,
)

FUNCTION_ID = f"nf_{'1' * 64}"
DATASET_SHA256 = "2" * 64
MODEL_SHA256 = "3" * 64
TOKENIZER_SHA256 = "4" * 64
WEIGHTS_SHA256 = "5" * 64
TEACHER_SHA256 = "6" * 64
SPLIT_SHA256 = "7" * 64
REVISION = "8" * 40


def test_builds_exact_export_ready_verified_ir_without_mutating_source() -> None:
    source, training, verification, provenance = fixture()
    before = deepcopy(source)

    built = build_verified_ir(source, training, verification, provenance)

    assert isinstance(built, BuiltVerifiedIr)
    assert source == before
    assert built.source_ir_bytes.endswith(b"\n")
    assert (
        built.source_ir_bytes
        == (
            json.dumps(
                built.document,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
    )
    document = built.document
    assert document["stage"] == "verified"
    assert document["trainingProvenance"] == provenance.to_document()
    assert document["verification"] == verification.to_ir_document()
    assert built.document is not document
    validate_verified_ir_binding(
        document,
        built.source_ir_bytes,
        training,
        verification,
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(stage="verified"), "source-stage"),
        (lambda value: value.update(irVersion=2), "IR v1"),
        (
            lambda value: value.update(trainingProvenance={"status": "pending", "extra": 1}),
            "exactly pending",
        ),
        (
            lambda value: value["model"].update(encoder="unsafe/path"),
            "model encoder",
        ),
    ],
)
def test_rejects_invalid_source_lifecycle_or_model_refs(mutation, match: str) -> None:
    source, training, verification, provenance = fixture()
    mutation(source)

    with pytest.raises(VerifiedIrBuildError, match=match):
        build_verified_ir(source, training, verification, provenance)


def test_rejects_mismatched_identity_and_failed_verification() -> None:
    source, training, verification, provenance = fixture()
    wrong_training = replace(training, function_id=f"nf_{'9' * 64}")
    with pytest.raises(VerifiedIrBuildError, match="identities must match"):
        build_verified_ir(source, wrong_training, verification, provenance)

    failed = replace(verification, status="failed", failures=("accuracy gate failed",))
    with pytest.raises(VerifiedIrBuildError, match="verification must be passing"):
        build_verified_ir(source, training, failed, provenance)


def test_rejects_provenance_that_disagrees_with_measured_records() -> None:
    source, training, verification, provenance = fixture()

    wrong_counts = replace(
        provenance,
        counts=replace(provenance.counts, synthetic=2),
    )
    with pytest.raises(VerifiedIrBuildError, match="counts do not match"):
        build_verified_ir(source, training, verification, wrong_counts)

    wrong_base = replace(provenance, base_model_name="other/encoder")
    with pytest.raises(VerifiedIrBuildError, match="base model must match"):
        build_verified_ir(source, training, verification, wrong_base)

    wrong_dataset = replace(provenance, dataset_sha256="a" * 64)
    with pytest.raises(VerifiedIrBuildError, match="dataset digest must match"):
        build_verified_ir(source, training, verification, wrong_dataset)

    wrong_seed = replace(provenance, seed=2)
    with pytest.raises(VerifiedIrBuildError, match="seed must match"):
        build_verified_ir(source, training, verification, wrong_seed)


def test_rejects_mutable_revision_bad_versions_and_timestamp_order() -> None:
    source, training, verification, provenance = fixture()
    with pytest.raises(VerifiedIrBuildError, match="immutable 40-character"):
        replace(provenance, base_model_revision="main")
    with pytest.raises(VerifiedIrBuildError, match="trainer_version must be nonempty"):
        replace(provenance, trainer_version="")
    with pytest.raises(VerifiedIrBuildError, match="trained_at cannot be later"):
        build_verified_ir(
            source,
            training,
            verification,
            replace(provenance, trained_at="2026-09-23T00:00:01Z"),
        )


def test_rejects_calibration_count_not_bound_to_held_out_split() -> None:
    source, training, verification, provenance = fixture()
    wrong_calibration = replace(
        verification.metrics.heads[0].calibration,
        sample_count=2,
    )
    wrong_head = replace(verification.metrics.heads[0], calibration=wrong_calibration)
    wrong_metrics = replace(
        verification.metrics,
        heads=(wrong_head,),
        ece=wrong_calibration.ece,
        brier=wrong_calibration.brier,
    )
    wrong_verification = replace(verification, metrics=wrong_metrics)

    with pytest.raises(VerifiedIrBuildError, match="calibration sample count"):
        build_verified_ir(source, training, wrong_verification, provenance)


def fixture() -> tuple[
    dict[str, Any],
    TrainingResult,
    VerificationResult,
    VerifiedIrProvenance,
]:
    definition = {
        "template": [
            {"kind": "text", "text": "Classify "},
            {"kind": "input", "name": "text"},
        ],
        "examples": [{"inputs": {"text": "gold"}, "output": "accept"}],
        "constraints": [],
    }
    inputs = [
        {
            "name": "text",
            "index": 0,
            "tsType": "string",
            "type": {"kind": "string"},
        }
    ]
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
    source: dict[str, Any] = {
        "kind": "semantscript.neural-function",
        "irVersion": 1,
        "stage": "source",
        "id": FUNCTION_ID,
        "semanticSha256": semantic_sha256,
        "source": {
            "path": "src/fixture.sem.ts",
            "line": 1,
            "column": 1,
            "sourceSha256": "a" * 64,
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
        "trainingProvenance": {"status": "pending"},
        "verification": {"status": "pending"},
    }
    contract = TrainingHeadContract(
        kind="nominal",
        source_kind="string-union",
        support=("accept", "reject"),
    )
    training = TrainingResult(
        model=object(),
        head=contract,
        split=TrainingSplit(
            training=(
                TrainingRow(
                    row_id="gold:0",
                    group_id="gold:0",
                    origin="gold",
                    inputs={"text": "gold"},
                    label_index=0,
                ),
            ),
            evaluation=(
                TrainingRow(
                    row_id="synthetic:0",
                    group_id="synthetic:0",
                    origin="synthetic",
                    inputs={"text": "generated"},
                    label_index=1,
                ),
            ),
        ),
        config=TrainingConfig(
            encoder_name="fixture/encoder",
            encoder_revision=REVISION,
            seed=1,
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
    head = HeadVerificationV1(
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
        heads=(head,),
        example_failures=0,
        constraint_violations=0,
        type_errors=0,
    )
    verification = VerificationResult(
        function_id=FUNCTION_ID,
        semantic_sha256=semantic_sha256,
        model_state_sha256=MODEL_SHA256,
        tokenizer_sha256=TOKENIZER_SHA256,
        status="passed",
        verified_at="2026-09-23T00:00:00Z",
        metrics=metrics,
        attested_cases=1,
        pair_count=0,
        failures=(),
    )
    provenance = VerifiedIrProvenance(
        teacher=TeacherDescriptor(
            provider="fixture",
            model="teacher-v1",
            configuration_sha256=TEACHER_SHA256,
        ),
        base_model_name="fixture/encoder",
        base_model_revision=REVISION,
        base_model_weights_sha256=WEIGHTS_SHA256,
        dataset_sha256=DATASET_SHA256,
        counts=TrainingProvenanceCounts(
            examples=1,
            synthetic=1,
            adversarial=0,
            calibration=1,
            verification=2,
            attested_verification=1,
        ),
        seed=1,
        trainer_version="0.1.0",
        trainer_commit="abcdef0",
        trained_at="2026-09-22T23:59:59Z",
    )
    return source, training, verification, provenance
