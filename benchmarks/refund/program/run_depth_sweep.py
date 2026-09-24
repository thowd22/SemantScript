"""Depth routing on the refund function: accuracy and latency per encoder depth (TASK-6.7).

For each depth the canonical refund function is trained as a one-domain
application whose adapter reads the shared encoder after that many layers
(``model.encoderDepth`` in the IR, the full stack for the last point), with the
committed compact-release recipe on the frozen v4 corpus; it is verified against
the attested release cases under decision-8's tolerance, exported as a
depth-routed artifact (an encoder prefix graph, the adapter and the head), scored
on the frozen final set through the Node runtime with every call timed, and
timed once more as a raw ONNX Runtime chain on CPU.

Usage::

    PYTHONNOUSERSITE=1 HSA_ENABLE_DXG_DETECTION=1 \\
    PYTHONPATH=.:trainer/src:model/src:.python-packages python3 -m \\
        benchmarks.refund.program.run_depth_sweep \\
        --corpus-manifest benchmarks/refund/data/pooled-v4-2026-09-24/manifest.json \\
        --heldout-dir benchmarks/refund/data/heldout-uci-2026-09-23 \\
        --output-dir benchmarks/refund/data/results-depth-sweep-2026-09-24 \\
        --depths 4,6,8,12,22
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.refund.program.pipeline import (
    compile_refund_program,
    parse_release_verification_record,
)
from benchmarks.refund.program.run_encoder_sweep import percentile
from benchmarks.refund.program.run_release_pipeline import (
    _load_tokenizer,
    _teacher_from_manifest,
)

from semantscript_trainer.adversarial import (
    AdversarialDatasetGenerator,
    AdversarialGenerationConfig,
)
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
from semantscript_trainer.training import TrainingConfig, TrainingExecutionError
from semantscript_trainer.verification import (
    VerificationConfig,
    evaluate_training_result,
    tokenizer_json_bytes,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_FINAL_SET_SCRIPT = _REPOSITORY_ROOT / "benchmarks/refund/program/run-final-set.mjs"
RESULT_KIND = "semantscript.refund-depth-sweep"
RESULT_VERSION = 1
LATENCY_ITERATIONS = 200
WARMUP_ITERATIONS = 20
FULL_DEPTH = 22
APPLICATION_ID = "refund-benchmark"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def recipe(device: str, epochs: int, batch_size: int, learning_rate: float) -> TrainingConfig:
    """The committed compact-release recipe (release-compact-2026-09-24)."""

    return TrainingConfig(
        local_files_only=True,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        maximum_sequence_length=128,
        evaluation_ratio=0.1,
        seed=1,
        device=device,
        head_architecture="linear",
        select_best_epoch=True,
        canonical_input_version=2,
    )


def routed_ir(ir: dict[str, Any], depth: int) -> dict[str, Any]:
    """The refund IR bound to one domain at ``depth`` (the full stack keeps the compiled refs)."""

    routed = copy.deepcopy(ir)
    model = routed["model"]
    model["adapter"] = f"adapter.{APPLICATION_ID}.refund"
    if depth < FULL_DEPTH:
        model["encoder"] = f"encoder.{APPLICATION_ID}.depth-{depth:03d}"
        model["encoderDepth"] = depth
    else:
        model["encoder"] = f"encoder.{APPLICATION_ID}"
        model.pop("encoderDepth", None)
    return routed


def sample_encoding(ir: dict[str, Any], tokenizer: Any, config: TrainingConfig, inputs: Any) -> Any:
    text = serialize_canonical_inputs(
        ir["inputs"], inputs, version=config.canonical_input_version
    ).decode("utf-8")
    return tokenizer(
        [text],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=config.maximum_sequence_length,
        return_tensors="pt",
    )


def timed(run: Any, iterations: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        run()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        run()
        samples.append((time.perf_counter() - started) * 1000)
    return {
        "p50": round(percentile(samples, 0.5), 3),
        "p95": round(percentile(samples, 0.95), 3),
        "mean": round(statistics.fmean(samples), 3),
        "min": round(min(samples), 3),
    }


def measure_cpu_chain(
    release_directory: Path, manifest: dict[str, Any], encoded: Any
) -> dict[str, Any]:
    """Time the exported encoder-adapter-head chain of the artifact on CPU through ONNX Runtime."""

    import numpy
    import onnxruntime

    resources = {resource["ref"]: resource for resource in manifest["resources"]}
    function = manifest["functions"][0]
    encoder_ref = function.get("encoderRef", manifest["model"]["encoderRef"])
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3

    def session(ref: str) -> Any:
        return onnxruntime.InferenceSession(
            str(release_directory / resources[ref]["path"]),
            options,
            providers=["CPUExecutionProvider"],
        )

    encoder = session(encoder_ref)
    adapter = session(function["adapterRef"])
    heads = [session(head["headRef"]) for head in function["heads"]]
    feed = {
        "input_ids": encoded["input_ids"].numpy().astype(numpy.int64),
        "attention_mask": encoded["attention_mask"].numpy().astype(numpy.int64),
    }

    def chain() -> None:
        sentence = encoder.run(None, feed)[0]
        function_embedding = adapter.run(None, {"sentence_embedding": sentence})[0]
        for head in heads:
            head.run(None, {"function_embedding": function_embedding})

    def encoder_only() -> None:
        encoder.run(None, feed)

    return {
        "provider": "CPUExecutionProvider, default threads",
        "tokens": int(encoded["input_ids"].shape[1]),
        "chain": timed(chain, LATENCY_ITERATIONS, WARMUP_ITERATIONS),
        "encoderOnly": timed(encoder_only, LATENCY_ITERATIONS, WARMUP_ITERATIONS),
        "bytes": {
            "encoder": resources[encoder_ref]["byteLength"],
            "adapter": resources[function["adapterRef"]]["byteLength"],
            "heads": sum(resources[head["headRef"]]["byteLength"] for head in function["heads"]),
        },
    }


def measure_gpu(model: Any, encoded: Any) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"available": False}
    model.eval()
    model.to("cuda")
    ids = encoded["input_ids"].to("cuda")
    mask = encoded["attention_mask"].to("cuda")

    def run() -> None:
        with torch.no_grad():
            model(ids, mask)
        torch.cuda.synchronize()

    result = timed(run, LATENCY_ITERATIONS, WARMUP_ITERATIONS)
    model.to("cpu")
    return {"provider": "PyTorch eager, ROCm", **result}


def run_final_set(artifact_root: Path, function_id: str, dataset_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["node", str(_FINAL_SET_SCRIPT), str(artifact_root), function_id, str(dataset_path), "20"],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"final-set script failed: {completed.stderr[-600:]}")
    return json.loads(completed.stdout)


def run_depth(
    depth: int,
    ir: dict[str, Any],
    base: Any,
    adversarial: Any,
    release: Any,
    config: TrainingConfig,
    tokenizer: Any,
    weights_sha256: str,
    output: Path,
    final_dataset: Path,
) -> dict[str, Any]:
    import torch

    bound = routed_ir(ir, depth)
    record: dict[str, Any] = {
        "depth": depth,
        "encoderRef": bound["model"]["encoder"],
        "adapterRef": bound["model"]["adapter"],
    }
    corpus = application_function(bound, base, adversarial)
    log(f"depth {depth}: training {config.epochs} epoch(s) at batch {config.batch_size}")
    started = time.monotonic()
    try:
        application = train_application([corpus], config=config, tokenizer=tokenizer)
    except (TrainingExecutionError, RuntimeError) as error:
        record["status"] = "not-trained"
        record["error"] = str(error)[:300]
        log(f"depth {depth}: training failed: {str(error)[:200]}")
        torch.cuda.empty_cache()
        return record
    record["trainSeconds"] = round(time.monotonic() - started, 1)
    training = application.functions[bound["id"]]
    record["selectedEpoch"] = training.selected_epoch
    record["heldOutCurve"] = [round(m.held_out_accuracy, 4) for m in training.metrics]
    record["trainedDepth"] = training.model.depth
    verification = evaluate_training_result(
        bound,
        training,
        base,
        adversarial,
        tokenizer=tokenizer,
        attested_verification=release.generated_cases,
        config=VerificationConfig(maximum_constraint_violation_rate=0.01),
        verified_at=utc_now(),
    )
    metrics = verification.metrics
    record["verification"] = {
        "status": verification.status,
        "releaseMisses": metrics.example_failures,
        "releaseAccuracy": round(1 - metrics.example_failures / len(release.case_ids), 4),
        "calibratedAccuracy": round(metrics.accuracy, 4),
        "pairConsistency": round(metrics.pair_consistency, 4),
        "ece": round(metrics.ece, 4),
        "brier": round(metrics.brier, 4),
        "constraintViolations": metrics.constraint_violations,
        "temperature": round(metrics.heads[0].calibration.temperature, 4),
        "failures": list(verification.failures)[:3],
    }
    log(f"depth {depth}: verification {json.dumps(record['verification'])[:300]}")
    encoded = sample_encoding(bound, tokenizer, config, dict(base.cases[0].inputs))
    record["latencyTokens"] = int(encoded["input_ids"].shape[1])
    record["gpu"] = measure_gpu(training.model, encoded)
    log(f"depth {depth}: gpu {json.dumps(record['gpu'])}")
    if verification.status != "passed":
        record["status"] = "gate-refused"
        del application
        torch.cuda.empty_cache()
        return record

    rows = training.split.training + training.split.evaluation
    origins = {
        name: sum(row.origin == name for row in rows)
        for name in ("gold", "synthetic", "constraint-boundary", "counterfactual")
    }
    provenance = VerifiedIrProvenance(
        teacher=base.teacher,
        base_model_name=config.encoder_name,
        base_model_revision=config.encoder_revision,
        base_model_weights_sha256=weights_sha256,
        dataset_sha256=base.dataset_sha256,
        counts=TrainingProvenanceCounts(
            examples=len(bound["definition"]["examples"]),
            synthetic=origins["synthetic"],
            adversarial=origins["constraint-boundary"] + origins["counterfactual"],
            calibration=len(training.split.evaluation),
            verification=len(rows) + verification.attested_cases - origins["gold"],
            attested_verification=verification.attested_cases,
        ),
        seed=config.seed,
        trainer_version="0.0.0",
        trainer_commit=git_commit(),
        trained_at=utc_now(),
    )
    built = build_verified_ir(bound, training, verification, provenance)
    application.model.to("cpu")
    artifact_root = output / "artifacts" / f"depth-{depth:03d}"
    exported = export_multi_function_artifact(
        artifact_root,
        [ArtifactFunction(built.document, training, verification, built.source_ir_bytes)],
        tokenizer_json=tokenizer_json_bytes(tokenizer),
        provenance=ArtifactProvenance(
            application_id=APPLICATION_ID,
            application_version="1.0.0",
            compiler_version="0.0.0",
            trainer_version="0.0.0",
            created_at=utc_now(),
            training_key_sha256=hashlib.sha256(
                f"{base.dataset_sha256}\n{adversarial.dataset_sha256}\ndepth-{depth}".encode()
            ).hexdigest(),
        ),
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )
    manifest = exported.manifest
    record["artifact"] = {
        "root": str(artifact_root.relative_to(_REPOSITORY_ROOT)),
        "manifestSha256": exported.manifest_sha256,
        "bytes": sum(resource["byteLength"] for resource in manifest["resources"]),
        "encoderPath": next(
            r["path"] for r in manifest["resources"] if r["ref"] == bound["model"]["encoder"]
        ),
    }
    record["cpuOnnx"] = measure_cpu_chain(exported.release_directory, manifest, encoded)
    log(f"depth {depth}: cpu onnx chain {json.dumps(record['cpuOnnx']['chain'])}")
    record["finalSet"] = run_final_set(artifact_root, bound["id"], final_dataset)
    log(
        f"depth {depth}: final set accuracy {record['finalSet']['accuracy']} "
        f"p50 {record['finalSet']['latencyMs']['p50']} ms"
    )
    record["status"] = "measured"
    del application
    torch.cuda.empty_cache()
    return record


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "0000000"


def environment(device: str) -> dict[str, Any]:
    import onnxruntime
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "node": subprocess.run(
            ["node", "--version"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu": platform.processor() or platform.machine(),
        "commit": git_commit(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument("--heldout-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--depths", default="4,6,8,12,22")
    arguments = parser.parse_args(argv)
    depths = [int(value) for value in arguments.depths.split(",") if value.strip()]
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    results: dict[str, Any] = (
        json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else {}
    )

    manifest_path = Path(arguments.corpus_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_root = manifest_path.parent
    heldout = Path(arguments.heldout_dir)
    with tempfile.TemporaryDirectory(prefix="semantscript-depth-sweep-") as temporary:
        compiled = compile_refund_program(Path(temporary) / "compiler")
    ir = compiled.source_ir
    teacher = _teacher_from_manifest(manifest)
    base = SyntheticDatasetGenerator(teacher, corpus_root / "synthetic").generate(
        ir, manifest["synthetic"]["requestedCaseCount"]
    )
    adversarial = AdversarialDatasetGenerator(
        teacher,
        corpus_root / "adversarial",
        config=AdversarialGenerationConfig(
            counterfactual_ratio=manifest["adversarial"]["config"]["counterfactualRatio"],
            maximum_attempts=manifest["adversarial"]["config"]["maximumAttempts"],
        ),
    ).generate(ir, base)
    release = parse_release_verification_record(
        (heldout / "release-verification.json").read_bytes(), ir
    )
    final_dataset = heldout / "final-benchmark-dataset.json"
    config = recipe(
        arguments.device, arguments.epochs, arguments.batch_size, arguments.learning_rate
    )
    tokenizer = _load_tokenizer(config)
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
    log(
        f"corpus: {base.synthetic_count} synthetic + {len(adversarial.cases)} adversarial rows; "
        f"release cases: {len(release.case_ids)}; depths {depths}"
    )
    results.update(
        {
            "kind": RESULT_KIND,
            "resultVersion": RESULT_VERSION,
            "startedAt": results.get("startedAt", utc_now()),
            "function": {"id": compiled.function_id, "semanticSha256": compiled.semantic_sha256},
            "corpus": {
                "manifest": str(manifest_path.resolve().relative_to(_REPOSITORY_ROOT)),
                "manifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "baseDatasetSha256": base.dataset_sha256,
                "adversarialDatasetSha256": adversarial.dataset_sha256,
            },
            "recipe": {
                "epochs": config.epochs,
                "batchSize": config.batch_size,
                "learningRate": config.learning_rate,
                "canonicalInputVersion": config.canonical_input_version,
                "evaluationRatio": config.evaluation_ratio,
                "seed": config.seed,
                "gate": {"eceThreshold": 0.1, "maximumConstraintViolationRate": 0.01},
            },
            "releaseCases": len(release.case_ids),
            "finalSet": {
                "path": str(final_dataset.relative_to(_REPOSITORY_ROOT)),
                "sha256": hashlib.sha256(final_dataset.read_bytes()).hexdigest(),
            },
            "latency": {
                "iterations": LATENCY_ITERATIONS,
                "warmup": WARMUP_ITERATIONS,
                "batch": 1,
                "finalSet": "every case timed once through the Node runtime after 20 warm-up calls",
            },
            "environment": environment(arguments.device),
        }
    )
    results.setdefault("depths", {})
    for depth in depths:
        results["depths"][str(depth)] = run_depth(
            depth,
            ir,
            base,
            adversarial,
            release,
            config,
            tokenizer,
            weights_sha256,
            output,
            final_dataset,
        )
        results["completedAt"] = utc_now()
        results_path.write_text(
            json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        log(f"wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
