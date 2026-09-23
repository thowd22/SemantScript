from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from semantscript_trainer import (
    AdversarialCase,
    AdversarialDataset,
    AdversarialGenerationConfig,
    ArtifactProvenance,
    DatasetCase,
    EpochMetrics,
    GeneratedCase,
    TeacherDescriptor,
    TrainingConfig,
    TrainingDataset,
    TrainingHeadContract,
    TrainingProvenanceCounts,
    TrainingResult,
    TrainingRow,
    TrainingSplit,
    VerifiedIrProvenance,
    semantic_json_sha256,
)

from . import pipeline


@pytest.fixture(scope="module")
def compiled_program(tmp_path_factory: pytest.TempPathFactory) -> pipeline.CompiledRefundProgram:
    return pipeline.compile_refund_program(tmp_path_factory.mktemp("refund-compiler"))


def test_real_compiler_bridge_emits_one_diagnostic_source_record(
    compiled_program: pipeline.CompiledRefundProgram,
) -> None:
    assert compiled_program.bundle_path.is_file()
    assert compiled_program.source_ir["stage"] == "source"
    assert compiled_program.source_ir["runtime"]["resultMode"] == "diagnostic"
    assert compiled_program.source_ir["definition"]["examples"] == []
    assert compiled_program.source_ir["output"]["head"]["support"] == [
        "approve",
        "deny",
        "review",
    ]


def test_compile_bridge_refuses_nonempty_output_and_unbounded_timeout(tmp_path: Path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(pipeline.RefundPipelineError, match="absent or empty"):
        pipeline.compile_refund_program(occupied)
    with pytest.raises(pipeline.RefundPipelineError, match="1 through 300"):
        pipeline.compile_refund_program(tmp_path / "unused", timeout_seconds=301)


def test_release_record_builder_derives_cases_and_every_digest(
    compiled_program: pipeline.CompiledRefundProgram,
) -> None:
    record = release_record(compiled_program.source_ir)

    assert record.function_id == compiled_program.function_id
    assert record.case_ids == ("release-01",)
    assert record.input_sha256s == (semantic_json_sha256(refund_inputs(3, 14, 75)),)
    assert record.payload_sha256 == record.document["payloadSha256"]
    assert record.attestation_sha256 == record.document["attestationSha256"]
    parsed = pipeline.parse_release_verification_record(
        json.dumps(record.document, separators=(",", ":")).encode(),
        compiled_program.source_ir,
    )
    assert parsed == record

    mutable_case = parsed.generated_cases[0]
    mutable_case.inputs["order"]["total"] = 999
    assert parsed.generated_cases[0].inputs["order"]["total"] == 75


def test_release_record_accepts_an_independent_judge_attestation(
    compiled_program: pipeline.CompiledRefundProgram,
) -> None:
    judge = pipeline.JudgeAttestationInputs(
        provider="unit-test",
        model="fixture-judge-not-a-real-model",
        interface="unit-test",
        session_reference="unit-test-session",
        rubric_sha256="5" * 64,
    )
    record = pipeline.build_release_verification_record(
        compiled_program.source_ir,
        (("release-01", GeneratedCase(inputs=refund_inputs(3, 14, 75), output="review")),),
        created_at="2026-09-23T11:00:00Z",
        attested_at="2026-09-23T10:59:00Z",
        evidence_sha256="6" * 64,
        judge=judge,
    )
    document = record.document
    assert document["humanAttestation"] is None
    assert document["judgeAttestation"]["judge"]["model"] == "fixture-judge-not-a-real-model"
    assert document["judgeAttestation"]["rubricSha256"] == "5" * 64
    assert document["judgeAttestation"]["caseIds"] == ["release-01"]
    assert record.attestation_sha256 == semantic_json_sha256(document["judgeAttestation"])
    assert (
        pipeline.parse_release_verification_record(
            json.dumps(document).encode("utf-8"), compiled_program.source_ir
        ).payload_sha256
        == record.payload_sha256
    )

    with pytest.raises(pipeline.RefundPipelineError, match="exactly one of"):
        pipeline.build_release_verification_record(
            compiled_program.source_ir,
            (("release-01", GeneratedCase(inputs=refund_inputs(3, 14, 75), output="review")),),
            created_at="2026-09-23T11:00:00Z",
            attested_at="2026-09-23T10:59:00Z",
            evidence_sha256="6" * 64,
            attestor="someone",
            judge=judge,
        )

    both = dict(document)
    both["humanAttestation"] = {
        "attestor": "someone",
        "attestedAt": "2026-09-23T10:59:00Z",
        "caseIds": ["release-01"],
        "declaration": pipeline._HUMAN_ATTESTATION_DECLARATION,
        "evidenceSha256": "6" * 64,
    }
    with pytest.raises(pipeline.RefundPipelineError, match="exactly one attestation"):
        pipeline.ReleaseVerificationRecord(both)

    wrong_declaration = json.loads(json.dumps(document))
    wrong_declaration["judgeAttestation"]["declaration"] = pipeline._HUMAN_ATTESTATION_DECLARATION
    wrong_declaration["attestationSha256"] = semantic_json_sha256(
        wrong_declaration["judgeAttestation"]
    )
    with pytest.raises(pipeline.RefundPipelineError, match="declaration is invalid"):
        pipeline.ReleaseVerificationRecord(wrong_declaration)


def test_release_record_rejects_tampering_and_wrong_function(
    compiled_program: pipeline.CompiledRefundProgram,
) -> None:
    record = release_record(compiled_program.source_ir)
    tampered = record.document
    tampered["cases"][0]["expected"] = "approve"
    with pytest.raises(pipeline.RefundPipelineError, match="payload digest does not match"):
        pipeline.parse_release_verification_record(tampered, compiled_program.source_ir)

    wrong_source = {**compiled_program.source_ir, "id": "nf_" + "9" * 64}
    with pytest.raises(pipeline.RefundPipelineError, match="different function"):
        pipeline.parse_release_verification_record(record.document, wrong_source)


def test_ledger_is_derived_and_allows_calibration_reuse(
    compiled_program: pipeline.CompiledRefundProgram,
) -> None:
    base, training = training_fixture(compiled_program)
    release = release_record(compiled_program.source_ir)

    ledger = pipeline.derive_training_input_ledger(
        compiled_program.source_ir,
        base,
        training,
        release,
        created_at="2026-09-23T12:00:00Z",
    )

    by_name = {partition["name"]: partition["inputSha256s"] for partition in ledger["partitions"]}
    assert by_name["examples"] == []
    assert len(by_name["synthetic"]) == 2
    assert by_name["calibration"] == [semantic_json_sha256(training.split.evaluation[0].inputs)]
    assert by_name["calibration"][0] in by_name["synthetic"]
    assert by_name["verification"] == list(release.input_sha256s)
    assert ledger["sources"] == {
        "baseDatasetSha256": base.dataset_sha256,
        "adversarialDatasetSha256": None,
        "releaseVerificationPayloadSha256": release.payload_sha256,
        "releaseVerificationAttestationSha256": release.attestation_sha256,
    }
    payload = {key: value for key, value in ledger.items() if key != "payloadSha256"}
    assert ledger["payloadSha256"] == semantic_json_sha256(payload)


def test_artifact_training_key_matches_typescript_canonical_vector() -> None:
    assert (
        pipeline.derive_refund_artifact_training_key_sha256(
            {
                "baseDatasetSha256": "a" * 64,
                "adversarialDatasetSha256": None,
                "releaseVerificationPayloadSha256": "c" * 64,
                "releaseVerificationAttestationSha256": "d" * 64,
            }
        )
        == "04f065ef6bdd717115ec6179da818f42c73d17ca37b24ad30e33ad885ac212b0"
    )


def test_ledger_retains_actual_adversarial_dataset_digest(
    compiled_program: pipeline.CompiledRefundProgram,
) -> None:
    base, training = training_fixture(compiled_program)
    adversarial_case = AdversarialCase(
        case_id="ac_" + "5" * 64,
        inputs=refund_inputs(2, 25, 90),
        output="review",
        tag="constraint-boundary",
        constraint_index=0,
        predicate_result=True,
    )
    adversarial = AdversarialDataset(
        function_id=compiled_program.function_id,
        base_dataset_sha256=base.dataset_sha256,
        teacher=base.teacher,
        config=AdversarialGenerationConfig(),
        cases=(adversarial_case,),
        pairs=(),
        cache_key_sha256="5" * 64,
        payload_sha256="6" * 64,
        dataset_sha256="7" * 64,
    )
    adversarial_row = TrainingRow(
        row_id="adversarial:0",
        group_id="adversarial:0",
        origin="constraint-boundary",
        inputs=adversarial_case.inputs,
        label_index=2,
    )
    training = replace(
        training,
        split=TrainingSplit(
            training=(*training.split.training, adversarial_row),
            evaluation=training.split.evaluation,
        ),
        adversarial_dataset_sha256=adversarial.dataset_sha256,
    )

    ledger = pipeline.derive_training_input_ledger(
        compiled_program.source_ir,
        base,
        training,
        release_record(compiled_program.source_ir),
        created_at="2026-09-23T12:00:00Z",
        adversarial_dataset=adversarial,
    )

    assert ledger["sources"]["adversarialDatasetSha256"] == adversarial.dataset_sha256
    assert ledger["partitions"][2]["inputSha256s"] == [
        semantic_json_sha256(adversarial_case.inputs)
    ]


def test_ledger_rejects_substitution_and_release_leakage(
    compiled_program: pipeline.CompiledRefundProgram,
) -> None:
    base, training = training_fixture(compiled_program)
    release = release_record(compiled_program.source_ir)
    substituted = replace(
        training,
        split=TrainingSplit(
            training=training.split.training,
            evaluation=(
                TrainingRow(
                    row_id="synthetic:substitute",
                    group_id="synthetic:substitute",
                    origin="synthetic",
                    inputs=refund_inputs(8, 8, 8),
                    label_index=0,
                ),
            ),
        ),
    )
    with pytest.raises(pipeline.RefundPipelineError, match="does not match source datasets"):
        pipeline.derive_training_input_ledger(
            compiled_program.source_ir,
            base,
            substituted,
            release,
            created_at="2026-09-23T12:00:00Z",
        )

    overlapping_release = release_record(
        compiled_program.source_ir,
        inputs=training.split.training[0].inputs,
    )
    with pytest.raises(pipeline.RefundPipelineError, match="overlaps training"):
        pipeline.derive_training_input_ledger(
            compiled_program.source_ir,
            base,
            training,
            overlapping_release,
            created_at="2026-09-23T12:00:00Z",
        )


def test_pipeline_keeps_final_benchmark_out_of_release_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    compiled_program: pipeline.CompiledRefundProgram,
) -> None:
    events: list[str] = []
    base, training = training_fixture(compiled_program)
    release = release_record(compiled_program.source_ir)
    verification = object()
    built = type("Built", (), {"source_ir_bytes": b"{}", "document": {}})()
    exported = type("Exported", (), {"artifact_root": tmp_path / "artifact"})()
    diagnostic = {"value": "review"}

    monkeypatch.setattr(
        pipeline,
        "compile_refund_program",
        lambda *args, **kwargs: events.append("compile") or compiled_program,
    )
    monkeypatch.setattr(
        pipeline,
        "train_classifier",
        lambda *args, **kwargs: events.append("train") or training,
    )

    def verify(*args, **kwargs):
        events.append("verify")
        assert kwargs["human_verification"] == release.generated_cases
        return verification

    monkeypatch.setattr(pipeline, "verify_training_result", verify)
    monkeypatch.setattr(
        pipeline,
        "build_verified_ir",
        lambda *args, **kwargs: events.append("bind") or built,
    )
    monkeypatch.setattr(
        pipeline,
        "export_application_artifact",
        lambda *args, **kwargs: events.append("export") or exported,
    )
    monkeypatch.setattr(
        pipeline,
        "run_refund_runtime",
        lambda *args, **kwargs: events.append("runtime") or diagnostic,
    )
    final = pipeline.FinalBenchmarkDatasetIdentity(
        compiled_program.function_id,
        compiled_program.semantic_sha256,
        "f" * 64,
    )
    result = pipeline.run_refund_pipeline(
        tmp_path / "compiler",
        tmp_path / "artifact",
        base,
        release,
        final,
        verified_ir_provenance=provenance_fixture(base),
        artifact_provenance=artifact_provenance_fixture(base, release),
        tokenizer=object(),
        encoder=object(),
        tokenizer_json=b"{}",
        parity_input_ids=object(),
        parity_attention_mask=object(),
        runtime_inputs={},
        verified_at="2026-09-23T12:30:00Z",
        ledger_created_at="2026-09-23T12:31:00Z",
    )

    assert events == ["compile", "train", "verify", "bind", "export", "runtime"]
    assert result.final_benchmark_dataset_sha256 == "f" * 64
    assert result.release_verification_payload_sha256 == release.payload_sha256
    assert result.training_ledger["sources"]["baseDatasetSha256"] == base.dataset_sha256

    events.clear()
    wrong_artifact_provenance = replace(
        artifact_provenance_fixture(base, release), training_key_sha256="e" * 64
    )
    with pytest.raises(pipeline.RefundPipelineError, match="does not bind"):
        pipeline.run_refund_pipeline(
            tmp_path / "compiler-wrong-key",
            tmp_path / "artifact-wrong-key",
            base,
            release,
            final,
            verified_ir_provenance=provenance_fixture(base),
            artifact_provenance=wrong_artifact_provenance,
            tokenizer=object(),
            encoder=object(),
            tokenizer_json=b"{}",
            parity_input_ids=object(),
            parity_attention_mask=object(),
            runtime_inputs={},
            verified_at="2026-09-23T12:30:00Z",
            ledger_created_at="2026-09-23T12:31:00Z",
        )
    assert events == ["compile", "train", "verify", "bind"]


def test_pipeline_rejects_same_final_and_release_before_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    compiled_program: pipeline.CompiledRefundProgram,
) -> None:
    base, _ = training_fixture(compiled_program)
    release = release_record(compiled_program.source_ir)
    monkeypatch.setattr(
        pipeline, "compile_refund_program", lambda *args, **kwargs: compiled_program
    )
    monkeypatch.setattr(
        pipeline,
        "train_classifier",
        lambda *args, **kwargs: pytest.fail("training must not start"),
    )
    final = pipeline.FinalBenchmarkDatasetIdentity(
        compiled_program.function_id,
        compiled_program.semantic_sha256,
        release.payload_sha256,
    )
    with pytest.raises(pipeline.RefundPipelineError, match="must differ"):
        pipeline.run_refund_pipeline(
            tmp_path / "compiler",
            tmp_path / "artifact",
            base,
            release,
            final,
            **pipeline_arguments(base, release),
        )


def test_pipeline_rejects_contradictory_provenance(
    tmp_path: Path,
    compiled_program: pipeline.CompiledRefundProgram,
) -> None:
    base, _ = training_fixture(compiled_program)
    release = release_record(compiled_program.source_ir)
    final = pipeline.FinalBenchmarkDatasetIdentity(
        compiled_program.function_id,
        compiled_program.semantic_sha256,
        "f" * 64,
    )
    arguments = pipeline_arguments(base, release)
    arguments["verified_ir_provenance"] = replace(
        arguments["verified_ir_provenance"], dataset_sha256="e" * 64
    )
    with pytest.raises(pipeline.RefundPipelineError, match="dataset provenance contradicts"):
        pipeline.run_refund_pipeline(
            tmp_path / "compiler",
            tmp_path / "artifact",
            base,
            release,
            final,
            **arguments,
        )


def test_runtime_bridge_is_bounded_and_requires_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = runtime_diagnostic()

    def completed(value: dict[str, Any]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess([], 0, stdout=json.dumps(value).encode(), stderr=b"")

    monkeypatch.setattr(pipeline, "_run_bounded_process", lambda *args, **kwargs: completed(valid))
    assert pipeline.run_refund_runtime("artifact", "nf_" + "1" * 64, {"x": 1}) == valid

    reversed_support = {**valid, "distribution": list(reversed(valid["distribution"]))}
    monkeypatch.setattr(
        pipeline,
        "_run_bounded_process",
        lambda *args, **kwargs: completed(reversed_support),
    )
    with pytest.raises(pipeline.RefundPipelineError, match="support order"):
        pipeline.run_refund_runtime("artifact", "nf_" + "1" * 64, {"x": 1})


def test_bounded_process_stops_output_flood_and_timeout(tmp_path: Path) -> None:
    with pytest.raises(pipeline.RefundPipelineError, match="stdout exceeded"):
        pipeline._run_bounded_process(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'x' * 200000)"],
            cwd=tmp_path,
            timeout_seconds=5,
            stdout_limit=1024,
            stderr_limit=1024,
            context="flood fixture",
        )
    with pytest.raises(pipeline.RefundPipelineError, match="timeout"):
        pipeline._run_bounded_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            timeout_seconds=1,
            stdout_limit=1024,
            stderr_limit=1024,
            context="timeout fixture",
        )

    descendant_pid_path = tmp_path / "descendant.pid"
    descendant_holds_pipes = (
        "import pathlib,subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))"
    )
    with pytest.raises(pipeline.RefundPipelineError, match="timeout"):
        pipeline._run_bounded_process(
            [sys.executable, "-c", descendant_holds_pipes, str(descendant_pid_path)],
            cwd=tmp_path,
            timeout_seconds=1,
            stdout_limit=1024,
            stderr_limit=1024,
            context="descendant pipe fixture",
        )
    descendant_pid = int(descendant_pid_path.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and _process_exists(descendant_pid):
        time.sleep(0.02)
    assert not _process_exists(descendant_pid)


def release_record(
    source_ir: dict[str, Any],
    *,
    inputs: dict[str, Any] | None = None,
) -> pipeline.ReleaseVerificationRecord:
    return pipeline.build_release_verification_record(
        source_ir,
        (
            (
                "release-01",
                GeneratedCase(inputs=inputs or refund_inputs(3, 14, 75), output="review"),
            ),
        ),
        created_at="2026-09-23T11:00:00Z",
        attestor="unit-test-human-not-a-production-attestation",
        attested_at="2026-09-23T10:59:00Z",
        evidence_sha256="a" * 64,
    )


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status_path = Path(f"/proc/{pid}/status")
    if status_path.exists() and "State:\tZ" in status_path.read_text():
        return False
    return True


def training_fixture(
    compiled: pipeline.CompiledRefundProgram,
) -> tuple[TrainingDataset, TrainingResult]:
    teacher = TeacherDescriptor("fixture", "teacher", "1" * 64)
    cases = (
        DatasetCase(refund_inputs(0, 10, 20), "approve", "synthetic"),
        DatasetCase(refund_inputs(1, 100, 40), "deny", "synthetic"),
    )
    base = TrainingDataset(
        function_id=compiled.function_id,
        semantic_sha256=compiled.semantic_sha256,
        requested_case_count=2,
        teacher=teacher,
        cases=cases,
        cache_key_sha256="2" * 64,
        payload_sha256="3" * 64,
        dataset_sha256="4" * 64,
    )
    head = TrainingHeadContract(
        kind="nominal",
        source_kind="string-union",
        support=("approve", "deny", "review"),
    )
    rows = (
        TrainingRow(
            row_id="base:0",
            group_id="base:0",
            origin="synthetic",
            inputs=cases[0].inputs,
            label_index=0,
        ),
        TrainingRow(
            row_id="base:1",
            group_id="base:1",
            origin="synthetic",
            inputs=cases[1].inputs,
            label_index=1,
        ),
    )
    training = TrainingResult(
        model=object(),
        head=head,
        split=TrainingSplit(training=(rows[0],), evaluation=(rows[1],)),
        config=TrainingConfig(seed=1),
        device="cpu",
        metrics=(EpochMetrics(epoch=1, mean_training_loss=0.1, held_out_accuracy=1.0),),
        function_id=compiled.function_id,
        semantic_sha256=compiled.semantic_sha256,
        base_dataset_sha256=base.dataset_sha256,
        adversarial_dataset_sha256=None,
    )
    return base, training


def provenance_fixture(base: TrainingDataset) -> VerifiedIrProvenance:
    return VerifiedIrProvenance(
        teacher=base.teacher,
        base_model_name="answerdotai/ModernBERT-base",
        base_model_revision="8" * 40,
        base_model_weights_sha256="9" * 64,
        dataset_sha256=base.dataset_sha256,
        counts=TrainingProvenanceCounts(
            examples=0,
            synthetic=2,
            adversarial=0,
            calibration=1,
            verification=3,
            human_authored_verification=1,
        ),
        seed=1,
        trainer_version="test",
        trainer_commit="abcdef0",
        trained_at="2026-09-23T10:00:00Z",
    )


def artifact_provenance_fixture(
    base: TrainingDataset,
    release: pipeline.ReleaseVerificationRecord,
) -> ArtifactProvenance:
    training_key_sha256 = pipeline.derive_refund_artifact_training_key_sha256(
        {
            "baseDatasetSha256": base.dataset_sha256,
            "adversarialDatasetSha256": None,
            "releaseVerificationPayloadSha256": release.payload_sha256,
            "releaseVerificationAttestationSha256": release.attestation_sha256,
        }
    )
    return ArtifactProvenance(
        application_id="refund-benchmark",
        application_version="1.0.0",
        compiler_version="1.0.0",
        trainer_version="1.0.0",
        created_at="2026-09-23T12:30:00Z",
        training_key_sha256=training_key_sha256,
    )


def pipeline_arguments(
    base: TrainingDataset,
    release: pipeline.ReleaseVerificationRecord,
) -> dict[str, Any]:
    return {
        "verified_ir_provenance": provenance_fixture(base),
        "artifact_provenance": artifact_provenance_fixture(base, release),
        "tokenizer": object(),
        "encoder": object(),
        "tokenizer_json": b"{}",
        "parity_input_ids": object(),
        "parity_attention_mask": object(),
        "runtime_inputs": {},
        "verified_at": "2026-09-23T12:30:00Z",
        "ledger_created_at": "2026-09-23T12:31:00Z",
    }


def refund_inputs(prior_refunds: int, age_days: int, total: int) -> dict[str, Any]:
    return {
        "customer": {"priorRefunds": prior_refunds, "tier": "standard"},
        "order": {"ageDays": age_days, "status": "paid", "total": total},
    }


def runtime_diagnostic() -> dict[str, Any]:
    return {
        "value": "review",
        "confidence": 0.5,
        "uncertainty": 0.5,
        "distribution": [
            {"value": "approve", "probability": 0.2},
            {"value": "deny", "probability": 0.3},
            {"value": "review", "probability": 0.5},
        ],
        "expectedValue": None,
    }
