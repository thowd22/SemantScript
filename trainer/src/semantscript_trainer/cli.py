"""Bundle-to-artifact driver behind ``semantscript train``.

``train_bundle`` takes one compiler IR bundle and turns every source-stage
function in it into a verified head of one content-addressed artifact: it
generates the synthetic dataset (and the adversarial sidecar when the function
has constraints), trains one classifier for a single function or the
shared-encoder application for several, verifies every function, binds verified
IR with derived provenance counts and exports the artifact. ``python -m
semantscript_trainer.cli train`` wires the driver to a teacher TOML, a cache
directory and the training, verification and adversarial settings, and writes a
per-function JSON report that the Node CLI renders.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from semantscript_trainer.adversarial import (
    AdversarialDataset,
    AdversarialDatasetGenerator,
    AdversarialGenerationConfig,
)
from semantscript_trainer.application import application_function, train_application
from semantscript_trainer.artifact import (
    ArtifactFunction,
    ArtifactProvenance,
    ExportedArtifact,
    export_multi_function_artifact,
)
from semantscript_trainer.canonical_input import serialize_canonical_inputs
from semantscript_trainer.dataset import SyntheticDatasetGenerator, TrainingDataset
from semantscript_trainer.lifecycle import (
    TrainingProvenanceCounts,
    VerifiedIrProvenance,
    build_verified_ir,
)
from semantscript_trainer.semantic_json import semantic_json_sha256
from semantscript_trainer.teacher import (
    AdversarialTeacher,
    JsonValue,
    NeuralFunctionIr,
    Teacher,
    TeacherDescriptor,
)
from semantscript_trainer.teacher_config import create_teacher, load_teacher_config
from semantscript_trainer.training import (
    TrainingConfig,
    TrainingResult,
    _load_tokenizer,
    train_classifier,
)
from semantscript_trainer.verification import (
    VerificationConfig,
    VerificationResult,
    evaluate_training_result,
    tokenizer_json_bytes,
)

BUNDLE_KIND = "semantscript.ir-bundle"
REPORT_KIND = "semantscript.train-report"
REPORT_VERSION = 1
DEFAULT_CASES = 64
TRAINER_VERSION = "0.0.0"
_TRAINING_KEY_KIND = "semantscript.training-key"


class TrainBundleError(RuntimeError):
    """The bundle, the teacher or the configuration cannot produce an artifact."""


class TrainBundleFailure(TrainBundleError):
    """Verification refused at least one function; the report records why."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True, slots=True)
class TrainedFunction:
    """One bundle function after generation, training and verification."""

    ir: NeuralFunctionIr
    base: TrainingDataset
    adversarial: AdversarialDataset | None
    training: TrainingResult
    verification: VerificationResult


@dataclass(frozen=True, slots=True)
class TrainBundleResult:
    """The per-function report and the published artifact."""

    report: dict[str, Any]
    exported: ExportedArtifact
    functions: tuple[TrainedFunction, ...]


def train_bundle(
    bundle: Mapping[str, Any],
    artifact_root: str | Path,
    *,
    teacher: Teacher,
    cache_directory: str | Path,
    cases: int = DEFAULT_CASES,
    training_config: TrainingConfig | None = None,
    verification_config: VerificationConfig | None = None,
    adversarial_config: AdversarialGenerationConfig | None = None,
    application_id: str | None = None,
    application_version: str = "0.0.0",
    compiler_version: str = "0.0.0",
    trainer_version: str = TRAINER_VERSION,
    trainer_commit: str | None = None,
    tokenizer: Any | None = None,
    encoder: Any | None = None,
    base_model_weights_sha256: str | None = None,
    log: Callable[[str], None] | None = None,
) -> TrainBundleResult:
    """Train, verify and export every function of ``bundle`` into ``artifact_root``.

    ``tokenizer``, ``encoder`` and ``base_model_weights_sha256`` default to the
    pinned Hugging Face encoder of ``training_config``; tests inject fakes.
    """

    say = log if log is not None else (lambda _message: None)
    functions = _bundle_functions(bundle)
    if isinstance(cases, bool) or not isinstance(cases, int) or cases < 0:
        raise TrainBundleError("cases must be a non-negative integer")
    resolved_training = training_config if training_config is not None else TrainingConfig()
    resolved_verification = (
        verification_config if verification_config is not None else VerificationConfig()
    )
    resolved_adversarial = (
        adversarial_config if adversarial_config is not None else AdversarialGenerationConfig()
    )
    descriptor = getattr(teacher, "descriptor", None)
    if not isinstance(descriptor, TeacherDescriptor):
        raise TrainBundleError("teacher must expose a TeacherDescriptor")
    resolved_application_id = (
        application_id if application_id is not None else _application_id(functions[0])
    )
    cache = Path(cache_directory)
    resolved_tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(resolved_training)

    datasets: list[tuple[NeuralFunctionIr, TrainingDataset, AdversarialDataset | None]] = []
    for ir in functions:
        function_id = cast(str, ir["id"])
        definition = cast(dict[str, Any], ir["definition"])
        gold = len(cast(list[Any], definition.get("examples", [])))
        total = max(cases, gold)
        say(f"{function_id}: generating {total} cases ({gold} gold)")
        base = SyntheticDatasetGenerator(teacher, cache).generate(ir, total)
        adversarial: AdversarialDataset | None = None
        if cast(list[Any], definition.get("constraints", [])):
            if not isinstance(teacher, AdversarialTeacher):
                raise TrainBundleError(
                    f"{function_id} declares constraints; the teacher must support "
                    "boundary and counterfactual generation"
                )
            say(f"{function_id}: generating adversarial cases around its constraints")
            adversarial = AdversarialDatasetGenerator(
                teacher, cache, config=resolved_adversarial
            ).generate(ir, base)
        datasets.append((ir, base, adversarial))

    trainings: dict[str, TrainingResult]
    shared_model: Any
    if len(datasets) == 1:
        ir, base, adversarial = datasets[0]
        say(f"{ir['id']}: training one classifier")
        training = train_classifier(
            ir,
            base,
            adversarial,
            config=resolved_training,
            tokenizer=resolved_tokenizer,
            encoder=encoder,
        )
        trainings = {cast(str, ir["id"]): training}
        shared_model = training.model
    else:
        say(f"training {len(datasets)} functions over one shared encoder and adapter")
        application = train_application(
            [application_function(ir, base, adversarial) for ir, base, adversarial in datasets],
            config=resolved_training,
            tokenizer=resolved_tokenizer,
            encoder=encoder,
        )
        trainings = dict(application.functions)
        shared_model = application.model
    trained_at = _utc_now()

    trained: list[TrainedFunction] = []
    for ir, base, adversarial in datasets:
        function_id = cast(str, ir["id"])
        training = trainings[function_id]
        verification = evaluate_training_result(
            ir,
            training,
            base,
            adversarial,
            tokenizer=resolved_tokenizer,
            config=resolved_verification,
            verified_at=trained_at,
        )
        say(
            f"{function_id}: verification {verification.status}, accuracy "
            f"{verification.metrics.accuracy:.4f}, ece {verification.metrics.ece:.4f}, "
            f"held-out accuracy {training.held_out_accuracy:.4f}"
        )
        trained.append(TrainedFunction(ir, base, adversarial, training, verification))

    training_key = _training_key_sha256(trained)
    report: dict[str, Any] = {
        "kind": REPORT_KIND,
        "reportVersion": REPORT_VERSION,
        "status": "passed",
        "trainedAt": trained_at,
        "application": {"id": resolved_application_id, "version": application_version},
        "teacher": {
            "provider": descriptor.provider,
            "model": descriptor.model,
            "configurationSha256": descriptor.configuration_sha256,
        },
        "trainingKeySha256": training_key,
        "artifact": None,
        "functions": [_function_report(entry) for entry in trained],
    }
    failed = [entry for entry in trained if entry.verification.status != "passed"]
    if failed:
        report["status"] = "failed"
        names = ", ".join(cast(str, entry.ir["id"]) for entry in failed)
        raise TrainBundleFailure(f"verification failed for {names}", report)

    weights_sha256 = (
        base_model_weights_sha256
        if base_model_weights_sha256 is not None
        else _pinned_weights_sha256(resolved_training)
    )
    commit = trainer_commit if trainer_commit is not None else _git_commit()
    artifact_functions: list[ArtifactFunction] = []
    for entry in trained:
        provenance = VerifiedIrProvenance(
            teacher=entry.base.teacher,
            base_model_name=resolved_training.encoder_name,
            base_model_revision=resolved_training.encoder_revision,
            base_model_weights_sha256=weights_sha256,
            dataset_sha256=entry.base.dataset_sha256,
            counts=_provenance_counts(entry),
            seed=resolved_training.seed,
            trainer_version=trainer_version,
            trainer_commit=commit,
            trained_at=trained_at,
        )
        built = build_verified_ir(entry.ir, entry.training, entry.verification, provenance)
        artifact_functions.append(
            ArtifactFunction(
                built.document, entry.training, entry.verification, built.source_ir_bytes
            )
        )

    first = trained[0]
    sample_inputs = dict(first.base.cases[0].inputs)
    parity_text = serialize_canonical_inputs(
        cast(list[Any], first.ir["inputs"]),
        sample_inputs,
        version=resolved_training.canonical_input_version,
    ).decode("utf-8")
    encoded = resolved_tokenizer(
        [parity_text],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=resolved_training.maximum_sequence_length,
        return_tensors="pt",
    )
    to_device = getattr(shared_model, "to", None)
    if callable(to_device):
        to_device("cpu")
    say(f"exporting {len(artifact_functions)} function(s) to {artifact_root}")
    exported = export_multi_function_artifact(
        artifact_root,
        artifact_functions,
        tokenizer_json=tokenizer_json_bytes(resolved_tokenizer),
        provenance=ArtifactProvenance(
            application_id=resolved_application_id,
            application_version=application_version,
            compiler_version=compiler_version,
            trainer_version=trainer_version,
            created_at=_utc_now(),
            training_key_sha256=training_key,
        ),
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )
    report["artifact"] = {
        "root": str(exported.artifact_root),
        "releaseDirectory": str(exported.release_directory),
        "manifestSha256": exported.manifest_sha256,
    }
    say(f"published release {exported.manifest_sha256}")
    return TrainBundleResult(report=report, exported=exported, functions=tuple(trained))


def _bundle_functions(bundle: Mapping[str, Any]) -> list[NeuralFunctionIr]:
    if not isinstance(bundle, Mapping):
        raise TrainBundleError("bundle must be a JSON object")
    if bundle.get("kind") != BUNDLE_KIND or bundle.get("bundleVersion") != 1:
        raise TrainBundleError(f"bundle must be a {BUNDLE_KIND} version 1 document")
    functions = bundle.get("functions")
    if not isinstance(functions, list) or not functions:
        raise TrainBundleError("bundle must contain at least one neural function")
    resolved: list[NeuralFunctionIr] = []
    refs: set[tuple[str, str]] = set()
    seen: set[str] = set()
    for raw in functions:
        if not isinstance(raw, Mapping):
            raise TrainBundleError("bundle functions must be objects")
        function_id = raw.get("id")
        if not isinstance(function_id, str) or function_id in seen:
            raise TrainBundleError("bundle functions must have unique string ids")
        seen.add(function_id)
        if raw.get("stage") != "source":
            raise TrainBundleError(f"{function_id} is not a source-stage record")
        model = raw.get("model")
        if not isinstance(model, Mapping):
            raise TrainBundleError(f"{function_id} has no model binding")
        encoder_ref = model.get("encoder")
        adapter_ref = model.get("adapter")
        if not isinstance(encoder_ref, str) or not isinstance(adapter_ref, str):
            raise TrainBundleError(f"{function_id} must bind encoder and adapter refs")
        refs.add((encoder_ref, adapter_ref))
        definition = raw.get("definition")
        if not isinstance(definition, Mapping):
            raise TrainBundleError(f"{function_id} has no definition")
        resolved.append(cast(NeuralFunctionIr, copy.deepcopy(dict(raw))))
    if len(refs) != 1:
        raise TrainBundleError("every bundle function must bind the same encoder and adapter refs")
    return resolved


def _application_id(ir: NeuralFunctionIr) -> str:
    encoder_ref = cast(str, cast(dict[str, Any], ir["model"])["encoder"])
    prefix = "encoder."
    return encoder_ref[len(prefix) :] if encoder_ref.startswith(prefix) else "application"


def _provenance_counts(entry: TrainedFunction) -> TrainingProvenanceCounts:
    rows = entry.training.split.training + entry.training.split.evaluation
    origins = {
        name: sum(row.origin == name for row in rows)
        for name in ("gold", "synthetic", "constraint-boundary", "counterfactual")
    }
    examples = cast(dict[str, Any], entry.ir["definition"]).get("examples", [])
    return TrainingProvenanceCounts(
        examples=len(cast(list[Any], examples)),
        synthetic=origins["synthetic"],
        adversarial=origins["constraint-boundary"] + origins["counterfactual"],
        calibration=len(entry.training.split.evaluation),
        verification=len(rows) + entry.verification.attested_cases - origins["gold"],
        attested_verification=entry.verification.attested_cases,
    )


def _function_report(entry: TrainedFunction) -> dict[str, Any]:
    training = entry.training
    epoch_index = (
        training.selected_epoch - 1
        if training.selected_epoch is not None
        else len(training.metrics) - 1
    )
    field_accuracy = training.metrics[epoch_index].held_out_field_accuracy
    verification = entry.verification
    source = cast(dict[str, Any], entry.ir.get("source", {}))
    adversarial: dict[str, Any] | None = None
    if entry.adversarial is not None:
        adversarial = {
            "sha256": entry.adversarial.dataset_sha256,
            "cases": len(entry.adversarial.cases),
        }
    return {
        "id": entry.ir["id"],
        "semanticSha256": entry.ir["semanticSha256"],
        "sourcePath": source.get("path"),
        "dataset": {
            "sha256": entry.base.dataset_sha256,
            "cases": len(entry.base.cases),
            "gold": entry.base.gold_count,
        },
        "adversarial": adversarial,
        "training": {
            "trainingRows": training.training_row_count,
            "heldOutRows": training.held_out_row_count,
            "epochs": len(training.metrics),
            "selectedEpoch": training.selected_epoch,
            "heldOutAccuracy": training.held_out_accuracy,
            "heldOutFieldAccuracy": None if field_accuracy is None else list(field_accuracy),
        },
        "verification": {
            "status": verification.status,
            "failures": list(verification.failures),
            "attestedCases": verification.attested_cases,
            "pairCount": verification.pair_count,
            "metrics": verification.to_ir_document()["metrics"],
        },
    }


def _training_key_sha256(trained: list[TrainedFunction]) -> str:
    projection: JsonValue = {
        "kind": _TRAINING_KEY_KIND,
        "keyVersion": 1,
        "functions": [
            {
                "id": cast(str, entry.ir["id"]),
                "semanticSha256": cast(str, entry.ir["semanticSha256"]),
                "datasetSha256": entry.base.dataset_sha256,
                "adversarialDatasetSha256": (
                    None if entry.adversarial is None else entry.adversarial.dataset_sha256
                ),
            }
            for entry in sorted(trained, key=lambda entry: cast(str, entry.ir["id"]))
        ],
    }
    return semantic_json_sha256(projection)


def _pinned_weights_sha256(config: TrainingConfig) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - training extra absent
        raise TrainBundleError(
            "provenance needs huggingface_hub; install the training extra"
        ) from error
    import hashlib

    weights = hf_hub_download(
        config.encoder_name,
        "model.safetensors",
        revision=config.encoder_revision,
        local_files_only=config.local_files_only,
    )
    return hashlib.sha256(Path(weights).read_bytes()).hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else "0000000"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _dump(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m semantscript_trainer.cli",
        description="Train, verify and export every function of a SemantScript IR bundle.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="produce an artifact from an IR bundle")
    train.add_argument("--bundle", required=True, type=Path, help="compiler IR bundle")
    train.add_argument("--artifact", required=True, type=Path, help="artifact root to publish")
    train.add_argument("--teacher", required=True, type=Path, help="TOML with a [teacher] table")
    train.add_argument("--cache-dir", type=Path, default=Path(".semantscript/cache"))
    train.add_argument("--report", type=Path, help="where to write the JSON report")
    train.add_argument("--cases", type=int, default=DEFAULT_CASES)
    train.add_argument("--application-id")
    train.add_argument("--application-version", default="0.0.0")
    train.add_argument("--compiler-version", default="0.0.0")
    train.add_argument("--encoder-name")
    train.add_argument("--encoder-revision")
    train.add_argument("--local-files-only", action="store_true")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--max-sequence-length", type=int)
    train.add_argument("--evaluation-ratio", type=float)
    train.add_argument("--seed", type=int)
    train.add_argument("--device")
    train.add_argument("--head-architecture")
    train.add_argument("--select-best-epoch", action="store_true")
    train.add_argument("--ece-threshold", type=float)
    train.add_argument("--max-constraint-violation-rate", type=float)
    train.add_argument("--counterfactual-ratio", type=float)
    return parser


def _training_config(arguments: argparse.Namespace) -> TrainingConfig:
    values = {
        "encoder_name": arguments.encoder_name,
        "encoder_revision": arguments.encoder_revision,
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "maximum_sequence_length": arguments.max_sequence_length,
        "evaluation_ratio": arguments.evaluation_ratio,
        "seed": arguments.seed,
        "device": arguments.device,
        "head_architecture": arguments.head_architecture,
    }
    kwargs: dict[str, Any] = {key: value for key, value in values.items() if value is not None}
    if arguments.local_files_only:
        kwargs["local_files_only"] = True
    if arguments.select_best_epoch:
        kwargs["select_best_epoch"] = True
    return TrainingConfig(**kwargs)


def _verification_config(arguments: argparse.Namespace) -> VerificationConfig:
    kwargs: dict[str, Any] = {}
    if arguments.ece_threshold is not None:
        kwargs["ece_threshold"] = arguments.ece_threshold
    if arguments.max_constraint_violation_rate is not None:
        kwargs["maximum_constraint_violation_rate"] = arguments.max_constraint_violation_rate
    return VerificationConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command != "train":  # pragma: no cover - argparse enforces the choice
        return 2

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    report: dict[str, Any] | None = None
    code = 0
    try:
        bundle = json.loads(Path(arguments.bundle).read_text(encoding="utf-8"))
        teacher = create_teacher(load_teacher_config(arguments.teacher))
        adversarial_config = (
            AdversarialGenerationConfig(counterfactual_ratio=arguments.counterfactual_ratio)
            if arguments.counterfactual_ratio is not None
            else None
        )
        result = train_bundle(
            bundle,
            arguments.artifact,
            teacher=teacher,
            cache_directory=arguments.cache_dir,
            cases=arguments.cases,
            training_config=_training_config(arguments),
            verification_config=_verification_config(arguments),
            adversarial_config=adversarial_config,
            application_id=arguments.application_id,
            application_version=arguments.application_version,
            compiler_version=arguments.compiler_version,
            log=log,
        )
        report = result.report
    except TrainBundleFailure as error:
        log(f"error: {error}")
        report = error.report
        code = 1
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        log(f"error: {error}")
        code = 1
    if report is not None:
        text = _dump(report)
        if arguments.report is not None:
            Path(arguments.report).parent.mkdir(parents=True, exist_ok=True)
            Path(arguments.report).write_text(text, encoding="utf-8")
        sys.stdout.write(text)
    return code


__all__ = [
    "BUNDLE_KIND",
    "DEFAULT_CASES",
    "REPORT_KIND",
    "REPORT_VERSION",
    "TrainBundleError",
    "TrainBundleFailure",
    "TrainBundleResult",
    "TrainedFunction",
    "main",
    "train_bundle",
]


if __name__ == "__main__":
    raise SystemExit(main())
