"""Train, verify, bind, export and smoke the refund function from frozen data.

Replays the frozen Sonnet 5 corpus from its caches (no teacher call is allowed:
a cache miss aborts), compiles the canonical refund program, fine-tunes the
pinned ModernBERT encoder on the GPU, verifies against the attested release
record, binds verified IR, derives the training-input ledger, exports the
immutable artifact and runs one diagnostic call through the Node runtime.

Usage::

    PYTHONPATH=.:trainer/src:model/src:.python-packages python -m \\
        benchmarks.refund.program.run_release_pipeline \\
        --corpus-manifest benchmarks/refund/data/sonnet-pilot-2026-09-23/manifest.json \\
        --heldout-dir benchmarks/refund/data/heldout-uci-2026-09-23 \\
        --output-dir benchmarks/refund/data/release-2026-09-23
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.refund.program.claude_cli_teacher import (
    ClaudeCliTeacherConfig,
    ClaudeCliTrainingTeacher,
)
from benchmarks.refund.program.pipeline import (
    FinalBenchmarkDatasetIdentity,
    compile_refund_program,
    derive_refund_artifact_training_key_sha256,
    derive_training_input_ledger,
    parse_release_verification_record,
    run_refund_runtime,
)
from benchmarks.refund.program.pooled_corpus import (
    POOLED_CONFIG_KIND,
    pooled_teacher_from_projection,
)

from semantscript_trainer.adversarial import (
    AdversarialDatasetGenerator,
    AdversarialGenerationConfig,
)
from semantscript_trainer.artifact import ArtifactProvenance, export_application_artifact
from semantscript_trainer.canonical_input import serialize_canonical_inputs
from semantscript_trainer.dataset import SyntheticDatasetGenerator
from semantscript_trainer.lifecycle import (
    TrainingProvenanceCounts,
    VerifiedIrProvenance,
    build_verified_ir,
)
from semantscript_trainer.training import (
    DEFAULT_ENCODER_NAME,
    DEFAULT_ENCODER_REVISION,
    TrainingConfig,
    train_classifier,
)
from semantscript_trainer.verification import (
    VerificationConfig,
    evaluate_training_result,
    require_passing_verification,
    tokenizer_json_bytes,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class ReleasePipelineError(RuntimeError):
    """The release pipeline could not complete."""


def _refuse_teacher_calls(command: Sequence[str], **kwargs: Any) -> Any:
    raise RuntimeError("teacher requests are forbidden during pipeline replay")


def _teacher_from_manifest(manifest: dict[str, Any]) -> Any:
    configuration = manifest["teacher"]["configuration"]
    if configuration.get("kind") == POOLED_CONFIG_KIND:
        pooled = pooled_teacher_from_projection(configuration, process_runner=_refuse_teacher_calls)
        if pooled.descriptor.configuration_sha256 != manifest["teacher"]["configurationSha256"]:
            raise ReleasePipelineError(
                "pooled corpus teacher configuration cannot be reconstructed"
            )
        return pooled
    config = ClaudeCliTeacherConfig(
        executable=configuration["cli"]["executable"],
        timeout_seconds=configuration["limits"]["timeoutSeconds"],
        stdout_limit_bytes=configuration["limits"]["stdoutBytes"],
        stderr_limit_bytes=configuration["limits"]["stderrBytes"],
        max_budget_usd=configuration["request"]["maxBudgetUsd"],
        concurrency=configuration["generation"]["concurrency"],
        maximum_case_attempts=configuration["generation"]["maximumCaseAttempts"],
        cli_version=configuration["cli"]["requiredVersion"],
        model=configuration["request"]["model"],
    )
    teacher = ClaudeCliTrainingTeacher(config, process_runner=_refuse_teacher_calls)
    if teacher.descriptor.configuration_sha256 != manifest["teacher"]["configurationSha256"]:
        raise ReleasePipelineError("corpus teacher configuration cannot be reconstructed")
    return teacher


def run_release_pipeline(
    corpus_manifest_path: str | Path,
    heldout_directory: str | Path,
    output_directory: str | Path,
    *,
    training_config: TrainingConfig,
    verification_config: VerificationConfig | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    corpus_manifest_path = Path(corpus_manifest_path)
    corpus_root = corpus_manifest_path.parent
    heldout = Path(heldout_directory)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise ReleasePipelineError("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    log = _Logger(output / "pipeline.log")

    log("compiling canonical refund program")
    compiled = compile_refund_program(output / "compiler")
    ir = compiled.source_ir

    log("replaying frozen corpus from cache")
    teacher = _teacher_from_manifest(manifest)
    base = SyntheticDatasetGenerator(teacher, corpus_root / "synthetic").generate(
        ir, manifest["synthetic"]["requestedCaseCount"]
    )
    if base.dataset_sha256 != manifest["synthetic"]["datasetSha256"]:
        raise ReleasePipelineError("replayed synthetic dataset digest differs from the manifest")
    adversarial_config = AdversarialGenerationConfig(
        counterfactual_ratio=manifest["adversarial"]["config"]["counterfactualRatio"],
        maximum_attempts=manifest["adversarial"]["config"]["maximumAttempts"],
    )
    adversarial = AdversarialDatasetGenerator(
        teacher, corpus_root / "adversarial", config=adversarial_config
    ).generate(ir, base)
    if adversarial.dataset_sha256 != manifest["adversarial"]["datasetSha256"]:
        raise ReleasePipelineError("replayed adversarial dataset digest differs from the manifest")

    log("loading attested release record and final dataset identity")
    release = parse_release_verification_record(
        (heldout / "release-verification.json").read_bytes(), ir
    )
    final_document = json.loads((heldout / "final-benchmark-dataset.json").read_text("utf-8"))
    final = FinalBenchmarkDatasetIdentity(
        compiled.function_id, compiled.semantic_sha256, final_document["payloadSha256"]
    )
    if final.dataset_sha256 in (
        release.payload_sha256,
        base.dataset_sha256,
        adversarial.dataset_sha256,
    ):
        raise ReleasePipelineError("final benchmark dataset collides with a lifecycle record")

    log(
        f"training on {training_config.device}: {base.synthetic_count} synthetic + "
        f"{len(adversarial.cases)} adversarial rows, {training_config.epochs} epochs"
    )
    trained_at = _utc_now()
    training_started = time.monotonic()
    training = train_classifier(ir, base, adversarial, config=training_config)
    training_seconds = time.monotonic() - training_started
    for metric in training.metrics:
        log(
            f"epoch {metric.epoch}: mean loss {metric.mean_training_loss:.4f}, "
            f"held-out accuracy {metric.held_out_accuracy:.4f}"
        )

    log(f"verifying against {len(release.case_ids)} attested release cases")
    verified_at = _utc_now()
    verification_started = time.monotonic()
    verification = evaluate_training_result(
        ir,
        training,
        base,
        adversarial,
        attested_verification=release.generated_cases,
        config=verification_config,
        verified_at=verified_at,
    )
    verification_seconds = time.monotonic() - verification_started
    verification_record = {
        "status": verification.status,
        "gate": {
            "eceThreshold": (verification_config or VerificationConfig()).ece_threshold,
            "maximumConstraintViolationRate": (
                verification_config or VerificationConfig()
            ).maximum_constraint_violation_rate,
            "recordCount": len(training.split.training)
            + len(training.split.evaluation)
            + len(release.case_ids),
        },
        "verifiedAt": verification.verified_at,
        "attestedCases": verification.attested_cases,
        "pairCount": verification.pair_count,
        "failures": list(verification.failures),
        "metrics": verification.to_ir_document()["metrics"],
        "modelStateSha256": verification.model_state_sha256,
        "tokenizerSha256": verification.tokenizer_sha256,
        "elapsedSeconds": round(verification_seconds, 3),
    }
    (output / "verification.json").write_text(_dump(verification_record), encoding="utf-8")
    log(
        f"verification {verification.status}: accuracy {verification.metrics.accuracy:.4f}, "
        f"ece {verification.metrics.ece:.4f}, brier {verification.metrics.brier:.4f}"
    )
    require_passing_verification(verification)

    log("binding verified IR")
    rows = training.split.training + training.split.evaluation
    origins = {
        name: sum(row.origin == name for row in rows)
        for name in ("gold", "synthetic", "constraint-boundary", "counterfactual")
    }
    counts = TrainingProvenanceCounts(
        examples=len(ir["definition"]["examples"]),
        synthetic=origins["synthetic"],
        adversarial=origins["constraint-boundary"] + origins["counterfactual"],
        calibration=len(training.split.evaluation),
        verification=len(rows) + verification.attested_cases - origins["gold"],
        attested_verification=verification.attested_cases,
    )
    weights_path, _tokenizer_path = _pinned_encoder_files(training_config)
    provenance = VerifiedIrProvenance(
        teacher=base.teacher,
        base_model_name=training_config.encoder_name,
        base_model_revision=training_config.encoder_revision,
        base_model_weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        dataset_sha256=base.dataset_sha256,
        counts=counts,
        seed=training_config.seed,
        trainer_version=_package_version(),
        trainer_commit=_git_commit(),
        trained_at=trained_at,
    )
    built = build_verified_ir(ir, training, verification, provenance)
    (output / "verified-ir.json").write_bytes(built.source_ir_bytes)

    log("deriving training-input ledger")
    ledger_created_at = _utc_now()
    ledger = derive_training_input_ledger(
        ir, base, training, release, created_at=ledger_created_at, adversarial_dataset=adversarial
    )
    (output / "training-input-ledger.json").write_text(_dump(ledger), encoding="utf-8")
    training_key_sha256 = derive_refund_artifact_training_key_sha256(ledger["sources"])

    log("exporting immutable artifact")
    training.model.to("cpu")
    tokenizer = _load_tokenizer(training_config)
    sample_inputs = dict(base.cases[0].inputs)
    parity_text = serialize_canonical_inputs(ir["inputs"], sample_inputs).decode("utf-8")
    encoded = tokenizer(
        [parity_text],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=training_config.maximum_sequence_length,
        return_tensors="pt",
    )
    artifact_provenance = ArtifactProvenance(
        application_id="refund-benchmark",
        application_version="1.0.0",
        compiler_version=_package_version(),
        trainer_version=_package_version(),
        created_at=_utc_now(),
        training_key_sha256=training_key_sha256,
    )
    exported = export_application_artifact(
        output / "artifact",
        built.document,
        training,
        verification,
        # The artifact must carry exactly the tokenizer bytes verification hashed.
        tokenizer_json=tokenizer_json_bytes(tokenizer),
        source_ir_bytes=built.source_ir_bytes,
        provenance=artifact_provenance,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )

    log("running one diagnostic call through the Node runtime")
    diagnostic = run_refund_runtime(exported.artifact_root, compiled.function_id, sample_inputs)
    log(f"runtime diagnostic: {diagnostic['value']} (confidence {diagnostic['confidence']:.4f})")

    summary = {
        "kind": "semantscript.refund-release-pipeline-manifest",
        "manifestVersion": 1,
        "completedAt": _utc_now(),
        "elapsedSeconds": round(time.monotonic() - started, 3),
        "function": {"id": compiled.function_id, "semanticSha256": compiled.semantic_sha256},
        "corpus": {
            "manifestPath": str(corpus_manifest_path),
            "syntheticDatasetSha256": base.dataset_sha256,
            "adversarialDatasetSha256": adversarial.dataset_sha256,
            "teacher": manifest["teacher"]["configurationSha256"],
        },
        "release": {
            "payloadSha256": release.payload_sha256,
            "attestationSha256": release.attestation_sha256,
            "caseCount": len(release.case_ids),
        },
        "finalBenchmarkDatasetSha256": final.dataset_sha256,
        "training": {
            "config": {
                "encoderName": training_config.encoder_name,
                "encoderRevision": training_config.encoder_revision,
                "epochs": training_config.epochs,
                "batchSize": training_config.batch_size,
                "learningRate": training_config.learning_rate,
                "maximumSequenceLength": training_config.maximum_sequence_length,
                "evaluationRatio": training_config.evaluation_ratio,
                "seed": training_config.seed,
                "loss": training_config.loss,
                "headArchitecture": training_config.head_architecture,
            },
            "device": training.device,
            "trainedAt": trained_at,
            "selectedEpoch": training.selected_epoch,
            "selectBestEpoch": training_config.select_best_epoch,
            "trainingRows": training.training_row_count,
            "calibrationRows": len(training.split.evaluation),
            "epochs": [
                {
                    "epoch": m.epoch,
                    "meanTrainingLoss": m.mean_training_loss,
                    "heldOutAccuracy": m.held_out_accuracy,
                }
                for m in training.metrics
            ],
            "elapsedSeconds": round(training_seconds, 3),
            "baseModelWeightsSha256": provenance.base_model_weights_sha256,
        },
        "verification": verification_record,
        "provenanceCounts": counts.to_document(),
        "ledger": {
            "payloadSha256": ledger["payloadSha256"],
            "sources": ledger["sources"],
            "trainingKeySha256": training_key_sha256,
        },
        "artifact": {
            "root": str(exported.artifact_root),
            "releaseDirectory": str(exported.release_directory),
            "manifestSha256": exported.manifest_sha256,
        },
        "runtimeDiagnostic": diagnostic,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "trainerCommit": provenance.trainer_commit,
        },
    }
    (output / "pipeline-manifest.json").write_text(_dump(summary), encoding="utf-8")
    log(f"done in {summary['elapsedSeconds']:.0f}s")
    return summary


class _Logger:
    def __init__(self, path: Path) -> None:
        self._path = path

    def __call__(self, message: str) -> None:
        line = f"{_utc_now()} {message}"
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def _pinned_encoder_files(config: TrainingConfig) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(
        config.encoder_name,
        "model.safetensors",
        revision=config.encoder_revision,
        local_files_only=config.local_files_only,
    )
    tokenizer = hf_hub_download(
        config.encoder_name,
        "tokenizer.json",
        revision=config.encoder_revision,
        local_files_only=config.local_files_only,
    )
    return Path(weights), Path(tokenizer)


def _load_tokenizer(config: TrainingConfig) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.encoder_name,
        revision=config.encoder_revision,
        local_files_only=config.local_files_only,
        trust_remote_code=False,
        use_fast=True,
    )


def _package_version() -> str:
    text = (_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("version"):
            return line.split("=", 1)[1].strip().strip('"')
    return "0.0.0"


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else "0000000"


def _dump(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument("--heldout-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--evaluation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ece-threshold", type=float, default=0.1)
    parser.add_argument("--maximum-constraint-violation-rate", type=float, default=0.0)
    parser.add_argument("--select-best-epoch", action="store_true")
    arguments = parser.parse_args(argv)
    training_config = TrainingConfig(
        encoder_name=DEFAULT_ENCODER_NAME,
        encoder_revision=DEFAULT_ENCODER_REVISION,
        local_files_only=True,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        maximum_sequence_length=arguments.max_sequence_length,
        evaluation_ratio=arguments.evaluation_ratio,
        seed=arguments.seed,
        device=arguments.device,
        select_best_epoch=arguments.select_best_epoch,
    )
    try:
        run_release_pipeline(
            arguments.corpus_manifest,
            arguments.heldout_dir,
            arguments.output_dir,
            training_config=training_config,
            verification_config=VerificationConfig(
                ece_threshold=arguments.ece_threshold,
                maximum_constraint_violation_rate=arguments.maximum_constraint_violation_rate,
            ),
        )
    except Exception as error:
        sys.stderr.write(f"release pipeline failed: {type(error).__name__}: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
