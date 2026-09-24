"""Train, score and time the typed-decisions workflows as SemantScript applications (TASK-6.6).

For each workflow the compiled bundle holds five functions over one ``state``
input. The train split replays as the training corpus (cases 0-279) with the
rest held back as attested verification cases, the five heads train jointly over
one shared encoder and adapter, every function is verified with the release
gate, and the test split (100 cases, 500 decisions per workflow) is scored:
accuracy against the discrete gold label, Brier score against the gold
distribution, and for score questions the expected-value error. Latency per
case (all five questions, one encoder pass) is measured on the GPU in PyTorch
and on CPU through ONNX Runtime; when every function of a workflow passes the
gate the application artifact is exported and the Node runtime's ``callStage``
path is timed as well.

Usage::

    PYTHONNOUSERSITE=1 HSA_ENABLE_DXG_DETECTION=1 \\
    PYTHONPATH=trainer/src:model/src:.python-packages python3 \\
        benchmarks/typed-decisions/program/run_typed_decisions.py \\
        --bundle benchmarks/typed-decisions/program/dist/semantscript.ir.v1.json \\
        --output-dir benchmarks/typed-decisions/data/results-2026-09-24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semantscript_trainer.application import application_function, train_application
from semantscript_trainer.artifact import (
    ArtifactFunction,
    ArtifactProvenance,
    export_multi_function_artifact,
)
from semantscript_trainer.canonical_input import serialize_canonical_inputs
from semantscript_trainer.dataset import SyntheticDatasetGenerator
from semantscript_trainer.lifecycle import (
    TrainingProvenanceCounts,
    VerifiedIrProvenance,
    build_verified_ir,
)
from semantscript_trainer.teacher import GeneratedCase, TeacherDescriptor
from semantscript_trainer.training import TrainingConfig, _load_tokenizer
from semantscript_trainer.verification import (
    VerificationConfig,
    evaluate_training_result,
    tokenizer_json_bytes,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET = "LocalLLaMA/typed-decisions"
DATASET_REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
WORKFLOWS = (
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
)
TRAINING_CASES = 280
ATTESTED_PER_QUESTION = 5
LATENCY_WARMUP = 10
RESULT_KIND = "semantscript.typed-decisions-result"
PUBLISHED = {
    "laya-typed-decisions (fine-tuned, Laya README)": 0.766,
    "TypeSafe Jev 1.13.0 (general, dataset card)": 0.727,
    "meraGPT Decider 1 (general, dataset card)": 0.768,
    "ModernBERT-base specialist (dataset card)": 0.646,
    "MiniLM-L6 specialist (dataset card)": 0.587,
    "teacher self-agreement ceiling (dataset card)": 0.735,
    "prior (dataset card)": 0.470,
}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def dataset_root() -> Path:
    root = (
        Path.home()
        / ".cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/snapshots"
        / DATASET_REVISION
    )
    if not root.is_dir():
        raise SystemExit(f"download {DATASET} at {DATASET_REVISION} first")
    return root


def load_split(root: Path, workflow: str, split: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(str(root / workflow / f"{split}-00000-of-00001.parquet")).to_pylist()
    return sorted(rows, key=lambda row: row["id"])


@dataclass(frozen=True, slots=True)
class Question:
    name: str
    kind: str
    support: tuple[Any, ...]


def question_of(ir: dict[str, Any]) -> str:
    for part in ir["definition"]["template"]:
        if part["kind"] == "text":
            for line in part["text"].splitlines():
                line = line.strip()
                if line.startswith("Question: "):
                    return line[len("Question: ") :].strip()
    raise ValueError(f"function {ir['id']} names no question")


def question_spec(ir: dict[str, Any]) -> Question:
    head = ir["output"]["head"]
    kind = head["sourceKind"]
    if kind == "boolean":
        return Question(question_of(ir), "noul", (False, True))
    if kind == "bounded-int":
        return Question(question_of(ir), "score", tuple(int(v) for v in head["supportDecimal"]))
    return Question(question_of(ir), "choice", tuple(head["support"]))


def gold_label(row: dict[str, Any], question: Question) -> Any:
    label = row[f"{question.name}__label"]
    if question.kind == "noul":
        return label == "true"
    if question.kind == "score":
        return int(label)
    return label


def gold_distribution(row: dict[str, Any], question: Question) -> list[float]:
    probabilities = json.loads(row[f"{question.name}__probabilities"])
    if question.kind == "noul":
        return [float(probabilities["false"]), float(probabilities["true"])]
    return [float(probabilities[str(value)]) for value in question.support]


class DatasetTeacher:
    """Replays the workflow's train split as generated cases; no language model."""

    def __init__(self, workflow: str, rows: Sequence[dict[str, Any]]) -> None:
        self._rows = tuple(rows)
        projection = {
            "kind": "semantscript.typed-decisions-train-teacher",
            "dataset": DATASET,
            "revision": DATASET_REVISION,
            "workflow": workflow,
            "caseIds": [row["id"] for row in self._rows],
        }
        encoded = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
        self.descriptor = TeacherDescriptor(
            provider="typed-decisions-train-split",
            model=f"{DATASET}@{DATASET_REVISION[:12]}/{workflow}",
            configuration_sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def generate(self, ir: dict[str, Any], n: int, /) -> tuple[GeneratedCase, ...]:
        question = question_spec(ir)
        if n > len(self._rows):
            raise RuntimeError(f"train split holds {len(self._rows)} cases, not {n}")
        return tuple(
            GeneratedCase({"state": row["state"]}, gold_label(row, question))
            for row in self._rows[:n]
        )


def attested_cases(rows: Sequence[dict[str, Any]], question: Question) -> list[GeneratedCase]:
    ranked = sorted(rows, key=lambda row: -float(row[f"{question.name}__confidence"]))
    return [
        GeneratedCase({"state": row["state"]}, gold_label(row, question))
        for row in ranked[:ATTESTED_PER_QUESTION]
    ]


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def summarize(samples: Sequence[float]) -> dict[str, float]:
    return {
        "p50Ms": round(percentile(samples, 0.5), 3),
        "p95Ms": round(percentile(samples, 0.95), 3),
        "meanMs": round(statistics.fmean(samples), 3),
        "count": len(samples),
    }


def state_text(ir: dict[str, Any], state: str, version: int) -> str:
    return serialize_canonical_inputs(ir["inputs"], {"state": state}, version=version).decode(
        "utf-8"
    )


def score_workflow(
    workflow: str,
    functions: list[dict[str, Any]],
    application: Any,
    verifications: dict[str, Any],
    test_rows: list[dict[str, Any]],
    tokenizer: Any,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Accuracy, Brier and expected-score error on the test split, per question."""

    import torch

    model = application.model
    model.eval()
    device = torch.device("cuda" if config.device == "cuda" else "cpu")
    model.to(device)
    per_question: dict[str, Any] = {}
    correct_total = 0
    decisions = 0
    with torch.no_grad():
        texts = [
            state_text(functions[0], row["state"], config.canonical_input_version)
            for row in test_rows
        ]
        embeddings = []
        for offset in range(0, len(texts), 8):
            encoded = tokenizer(
                texts[offset : offset + 8],
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=config.maximum_sequence_length,
                return_tensors="pt",
            )
            embeddings.append(
                model.embed(encoded["input_ids"].to(device), encoded["attention_mask"].to(device))
            )
        function_embedding = torch.cat(embeddings, dim=0)
        for ir in functions:
            question = question_spec(ir)
            head = model.heads[ir["id"]]
            logits = head(function_embedding).float()
            temperature = verifications[ir["id"]].metrics.heads[0].calibration.temperature
            if question.kind == "noul":
                p_true = torch.sigmoid(logits[:, 0] / temperature)
                probabilities = torch.stack([1 - p_true, p_true], dim=1)
            else:
                probabilities = torch.softmax(logits / temperature, dim=1)
            probabilities = probabilities.cpu()
            predicted = probabilities.argmax(dim=1).tolist()
            correct = 0
            brier = 0.0
            score_error = 0.0
            within_one = 0
            for index, row in enumerate(test_rows):
                gold = gold_label(row, question)
                gold_index = question.support.index(gold)
                if predicted[index] == gold_index:
                    correct += 1
                distribution = gold_distribution(row, question)
                brier += (
                    sum(
                        (float(probabilities[index][k]) - distribution[k]) ** 2
                        for k in range(len(distribution))
                    )
                    / 2
                )
                if question.kind == "score":
                    expected = float(
                        sum(
                            probabilities[index][k] * question.support[k]
                            for k in range(len(question.support))
                        )
                    )
                    score_error += abs(expected - float(row[f"{question.name}__score"]))
                    within_one += int(abs(predicted[index] - gold_index) <= 1)
            count = len(test_rows)
            entry: dict[str, Any] = {
                "functionId": ir["id"],
                "type": question.kind,
                "support": list(question.support),
                "accuracy": round(correct / count, 4),
                "brier": round(brier / count, 4),
                "temperature": round(temperature, 4),
                "gate": verifications[ir["id"]].status,
                "gateAccuracy": round(verifications[ir["id"]].metrics.accuracy, 4),
                "gateEce": round(verifications[ir["id"]].metrics.ece, 4),
            }
            if question.kind == "score":
                entry["scoreMae"] = round(score_error / count, 4)
                entry["withinOneLevel"] = round(within_one / count, 4)
            per_question[question.name] = entry
            correct_total += correct
            decisions += count
    return {
        "questions": per_question,
        "accuracy": round(correct_total / decisions, 4),
        "correct": correct_total,
        "decisions": decisions,
    }


def measure_gpu(
    functions: list[dict[str, Any]],
    application: Any,
    test_rows: list[dict[str, Any]],
    tokenizer: Any,
    config: TrainingConfig,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"available": False}
    model = application.model
    model.eval()
    model.to("cuda")
    ids = [ir["id"] for ir in functions]
    samples = []
    with torch.no_grad():
        for index, row in enumerate(test_rows):
            encoded = tokenizer(
                [state_text(functions[0], row["state"], config.canonical_input_version)],
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=config.maximum_sequence_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to("cuda")
            attention_mask = encoded["attention_mask"].to("cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            embedding = model.embed(input_ids, attention_mask)
            for function_id in ids:
                model.heads[function_id](embedding)
            torch.cuda.synchronize()
            if index >= LATENCY_WARMUP:
                samples.append((time.perf_counter() - started) * 1000)
    return {
        "available": True,
        "framework": "pytorch-eager",
        "questionsPerCase": len(ids),
        **summarize(samples),
    }


def measure_cpu_onnx(
    functions: list[dict[str, Any]],
    application: Any,
    test_rows: list[dict[str, Any]],
    tokenizer: Any,
    config: TrainingConfig,
) -> dict[str, Any]:
    import numpy
    import onnxruntime

    from semantscript_model.export import export_application_components

    model = application.model
    model.to("cpu")
    model.eval()
    ids = [ir["id"] for ir in functions]
    sample = tokenizer(
        [state_text(functions[0], test_rows[0]["state"], config.canonical_input_version)],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=config.maximum_sequence_length,
        return_tensors="pt",
    )
    with tempfile.TemporaryDirectory(prefix="semantscript-typed-decisions-") as temporary:
        directory = Path(temporary)
        export_application_components(
            model.encoder,
            model.adapter,
            {function_id: model.heads[function_id] for function_id in ids},
            sample["input_ids"],
            sample["attention_mask"],
            encoder_path=directory / "encoder.onnx",
            adapter_path=directory / "adapter.onnx",
            head_paths={function_id: directory / f"{function_id}.onnx" for function_id in ids},
        )
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        encoder = onnxruntime.InferenceSession(
            str(directory / "encoder.onnx"), options, providers=["CPUExecutionProvider"]
        )
        adapter = onnxruntime.InferenceSession(
            str(directory / "adapter.onnx"), options, providers=["CPUExecutionProvider"]
        )
        heads = [
            onnxruntime.InferenceSession(
                str(directory / f"{function_id}.onnx"), options, providers=["CPUExecutionProvider"]
            )
            for function_id in ids
        ]
        sizes = {
            "encoder": (directory / "encoder.onnx").stat().st_size,
            "adapter": (directory / "adapter.onnx").stat().st_size,
            "heads": sum((directory / f"{function_id}.onnx").stat().st_size for function_id in ids),
        }
        samples = []
        tokens = []
        for index, row in enumerate(test_rows):
            encoded = tokenizer(
                [state_text(functions[0], row["state"], config.canonical_input_version)],
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=config.maximum_sequence_length,
                return_tensors="pt",
            )
            feed = {
                "input_ids": encoded["input_ids"].numpy().astype(numpy.int64),
                "attention_mask": encoded["attention_mask"].numpy().astype(numpy.int64),
            }
            tokens.append(int(encoded["input_ids"].shape[1]))
            started = time.perf_counter()
            sentence = encoder.run(None, feed)[0]
            function = adapter.run(None, {"sentence_embedding": sentence})[0]
            for head in heads:
                head.run(None, {"function_embedding": function})
            if index >= LATENCY_WARMUP:
                samples.append((time.perf_counter() - started) * 1000)
    return {
        "provider": "CPUExecutionProvider",
        "questionsPerCase": len(ids),
        "meanTokens": round(statistics.fmean(tokens), 1),
        "maxTokens": max(tokens),
        "artifactBytes": sizes,
        **summarize(samples),
    }


def export_workflow(
    workflow: str,
    functions: list[dict[str, Any]],
    application: Any,
    trainings: dict[str, Any],
    verifications: dict[str, Any],
    bases: dict[str, Any],
    tokenizer: Any,
    config: TrainingConfig,
    output: Path,
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Publish the workflow's artifact and time the Node runtime's fused stage over the test split."""

    from huggingface_hub import hf_hub_download

    weights = Path(
        hf_hub_download(
            config.encoder_name,
            "model.safetensors",
            revision=config.encoder_revision,
            local_files_only=True,
        )
    )
    weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
    commit = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or "0000000"
    )
    trained_at = utc_now()
    entries = []
    for ir in functions:
        training = trainings[ir["id"]]
        verification = verifications[ir["id"]]
        base = bases[ir["id"]]
        rows = training.split.training + training.split.evaluation
        counts = TrainingProvenanceCounts(
            examples=0,
            synthetic=sum(row.origin == "synthetic" for row in rows),
            adversarial=0,
            calibration=len(training.split.evaluation),
            verification=len(rows) + verification.attested_cases,
            attested_verification=verification.attested_cases,
        )
        provenance = VerifiedIrProvenance(
            teacher=base.teacher,
            base_model_name=config.encoder_name,
            base_model_revision=config.encoder_revision,
            base_model_weights_sha256=weights_sha256,
            dataset_sha256=base.dataset_sha256,
            counts=counts,
            seed=config.seed,
            trainer_version="0.0.0",
            trainer_commit=commit,
            trained_at=trained_at,
        )
        built = build_verified_ir(ir, training, verification, provenance)
        entries.append(
            ArtifactFunction(built.document, training, verification, built.source_ir_bytes)
        )
    sample = tokenizer(
        [
            state_text(
                functions[0],
                bases[functions[0]["id"]].cases[0].inputs["state"],
                config.canonical_input_version,
            )
        ],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=config.maximum_sequence_length,
        return_tensors="pt",
    )
    application.model.to("cpu")
    key = hashlib.sha256(
        "\n".join(sorted(bases[ir["id"]].dataset_sha256 for ir in functions)).encode()
    ).hexdigest()
    artifact_root = output / "artifacts" / workflow
    exported = export_multi_function_artifact(
        artifact_root,
        entries,
        tokenizer_json=tokenizer_json_bytes(tokenizer),
        provenance=ArtifactProvenance(
            application_id=f"typed-decisions-{workflow.replace('_', '-')}",
            application_version="1.0.0",
            compiler_version="0.0.0",
            trainer_version="0.0.0",
            created_at=utc_now(),
            training_key_sha256=key,
        ),
        input_ids=sample["input_ids"],
        attention_mask=sample["attention_mask"],
    )
    states_path = output / f"{workflow}-test-states.json"
    states_path.write_text(json.dumps([row["state"] for row in test_rows]), encoding="utf-8")
    script = _REPOSITORY_ROOT / "benchmarks/typed-decisions/program/run-stage-latency.mjs"
    completed = subprocess.run(
        [
            "node",
            str(script),
            str(artifact_root),
            str(states_path),
            json.dumps([ir["id"] for ir in functions]),
            str(LATENCY_WARMUP),
        ],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    states_path.unlink()
    if completed.returncode != 0:
        raise RuntimeError(f"stage latency script failed: {completed.stderr[-500:]}")
    return {
        "manifestSha256": exported.manifest_sha256,
        "root": str(artifact_root.relative_to(_REPOSITORY_ROOT)),
        "nodeRuntime": json.loads(completed.stdout),
    }


def run_workflow(
    workflow: str,
    functions: list[dict[str, Any]],
    root: Path,
    cache: Path,
    output: Path,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    train_rows = load_split(root, workflow, "train")
    test_rows = load_split(root, workflow, "test")
    training_rows = train_rows[:TRAINING_CASES]
    held_rows = train_rows[TRAINING_CASES:]
    teacher = DatasetTeacher(workflow, training_rows)
    config = TrainingConfig(
        local_files_only=True,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=0.01,
        maximum_sequence_length=arguments.max_sequence_length,
        evaluation_ratio=0.1,
        seed=1,
        device=arguments.device,
        head_architecture="linear",
        select_best_epoch=True,
    )
    tokenizer = _load_tokenizer(config)
    bases = {}
    corpora = []
    for ir in functions:
        base = SyntheticDatasetGenerator(teacher, cache).generate(ir, TRAINING_CASES)
        bases[ir["id"]] = base
        corpora.append(application_function(ir, base))
    log(f"{workflow}: training {len(functions)} heads jointly on {TRAINING_CASES} cases")
    started = time.monotonic()
    application = train_application(corpora, config=config, tokenizer=tokenizer)
    train_seconds = time.monotonic() - started
    verifications = {}
    for ir in functions:
        question = question_spec(ir)
        verification = evaluate_training_result(
            ir,
            application.functions[ir["id"]],
            bases[ir["id"]],
            None,
            tokenizer=tokenizer,
            attested_verification=attested_cases(held_rows, question),
            config=VerificationConfig(),
            verified_at=utc_now(),
        )
        verifications[ir["id"]] = verification
        log(
            f"{workflow}/{question.name}: gate {verification.status} calibrated {verification.metrics.accuracy:.3f} ece {verification.metrics.ece:.3f} misses {verification.metrics.example_failures}"
        )
    record: dict[str, Any] = {
        "trainSeconds": round(train_seconds, 1),
        "selectedEpoch": application.selected_epoch,
        "heldOutCurve": [round(m.mean_held_out_accuracy, 4) for m in application.metrics],
        "trainingCases": TRAINING_CASES,
        "attestedPerQuestion": ATTESTED_PER_QUESTION,
        "testCases": len(test_rows),
        "score": score_workflow(
            workflow, functions, application, verifications, test_rows, tokenizer, config
        ),
    }
    log(
        f"{workflow}: test accuracy {record['score']['accuracy']} over {record['score']['decisions']} decisions"
    )
    record["gpu"] = measure_gpu(functions, application, test_rows, tokenizer, config)
    log(f"{workflow}: gpu {json.dumps(record['gpu'])}")
    record["cpuOnnx"] = measure_cpu_onnx(functions, application, test_rows, tokenizer, config)
    log(
        f"{workflow}: cpu onnx {json.dumps({k: v for k, v in record['cpuOnnx'].items() if k != 'artifactBytes'})}"
    )
    if all(v.status == "passed" for v in verifications.values()):
        try:
            record["artifact"] = export_workflow(
                workflow,
                functions,
                application,
                dict(application.functions),
                verifications,
                bases,
                tokenizer,
                config,
                output,
                test_rows,
            )
            log(
                f"{workflow}: artifact {record['artifact']['manifestSha256']} node stage {json.dumps(record['artifact']['nodeRuntime'])[:200]}"
            )
        except (RuntimeError, ValueError, TypeError, OSError) as error:
            record["artifact"] = {"status": "export-failed", "error": str(error)[:400]}
            log(f"{workflow}: export failed: {str(error)[:200]}")
    else:
        failed = [question_of(ir) for ir in functions if verifications[ir["id"]].status != "passed"]
        record["artifact"] = {"status": "gate-refused", "failedQuestions": failed}
    del application
    torch.cuda.empty_cache()
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--cache-dir", type=Path, default=_REPOSITORY_ROOT / "benchmarks/typed-decisions/data/cache"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-sequence-length", type=int, default=768)
    parser.add_argument("--workflows", default=",".join(WORKFLOWS))
    arguments = parser.parse_args(argv)
    root = dataset_root()
    bundle = json.loads(arguments.bundle.read_text(encoding="utf-8"))
    by_workflow: dict[str, list[dict[str, Any]]] = {}
    for ir in bundle["functions"]:
        by_workflow.setdefault(ir["source"]["path"].removesuffix(".sem.ts"), []).append(ir)
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    results = json.loads(results_path.read_text()) if results_path.is_file() else {}
    import torch

    results.update(
        {
            "kind": RESULT_KIND,
            "resultVersion": 1,
            "startedAt": results.get("startedAt", utc_now()),
            "dataset": {"name": DATASET, "revision": DATASET_REVISION},
            "bundleSha256": hashlib.sha256(arguments.bundle.read_bytes()).hexdigest(),
            "recipe": {
                "encoder": TrainingConfig().encoder_name,
                "epochs": arguments.epochs,
                "batchSize": arguments.batch_size,
                "learningRate": arguments.learning_rate,
                "maxSequenceLength": arguments.max_sequence_length,
                "head": "linear",
                "loss": "proper",
                "selectBestEpoch": True,
            },
            "published": PUBLISHED,
            "environment": {
                "capturedAt": utc_now(),
                "operatingSystem": f"{platform.system()} {platform.release()}",
                "accelerator": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__,
            },
        }
    )
    results.setdefault("workflows", {})
    for workflow in arguments.workflows.split(","):
        results["workflows"][workflow] = run_workflow(
            workflow, by_workflow[workflow], root, arguments.cache_dir, output, arguments
        )
        results["completedAt"] = utc_now()
        results_path.write_text(
            json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
    scored = [w for w in results["workflows"].values() if "score" in w]
    if scored:
        correct = sum(w["score"]["correct"] for w in scored)
        decisions = sum(w["score"]["decisions"] for w in scored)
        results["overall"] = {
            "accuracy": round(correct / decisions, 4),
            "correct": correct,
            "decisions": decisions,
            "workflows": len(scored),
        }
        results_path.write_text(
            json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        log(f"overall accuracy {results['overall']['accuracy']} over {decisions} decisions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
