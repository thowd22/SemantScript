"""Encoder size and initialisation sweep on the refund function (TASK-5.14).

For every encoder point, replay the frozen corpus, train with one recipe, head
and proper loss, verify against the release-attested cases, export the ONNX
chain and measure batch-1 latency:

* on CPU with ONNX Runtime (the deployed runtime path): p50/p95 of the
  encoder-adapter-head chain, and one encoder pass followed by 1, 10 and 50
  head passes to show the amortisation;
* on the GPU in PyTorch eager mode (ONNX Runtime here has no GPU provider,
  which the write-up states).

Points: ModernBERT-base (the decision-3 default), ModernBERT-large initialised
from the Laya typed-decisions checkpoint's encoder (extracted by
``prepare_laya_encoder.py``), ModernBERT-large from its pretrained weights, and
DeBERTa-v2-xlarge (~0.9B) as the ~1B point. An out-of-memory failure is
recorded as a result, with a smaller batch retried once.

Usage::

    PYTHONNOUSERSITE=1 HSA_ENABLE_DXG_DETECTION=1 \\
    PYTHONPATH=.:trainer/src:model/src:.python-packages python3 -m \\
        benchmarks.refund.program.run_encoder_sweep \\
        --corpus-manifest benchmarks/refund/data/pooled-v4-2026-09-24/manifest.json \\
        --heldout-dir benchmarks/refund/data/heldout-uci-2026-09-23 \\
        --output-dir benchmarks/refund/data/results-encoder-sweep-2026-09-24
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.refund.program.pipeline import (
    compile_refund_program,
    parse_release_verification_record,
)
from benchmarks.refund.program.run_release_pipeline import _teacher_from_manifest, _utc_now

from semantscript_trainer.adversarial import (
    AdversarialDatasetGenerator,
    AdversarialGenerationConfig,
)
from semantscript_trainer.canonical_input import serialize_canonical_inputs
from semantscript_trainer.dataset import SyntheticDatasetGenerator
from semantscript_trainer.training import (
    DEFAULT_ENCODER_NAME,
    DEFAULT_ENCODER_REVISION,
    TrainingConfig,
    TrainingExecutionError,
    _load_tokenizer,
    train_classifier,
)
from semantscript_trainer.verification import VerificationConfig, evaluate_training_result

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RESULT_KIND = "semantscript.encoder-sweep"
RESULT_VERSION = 1
LATENCY_ITERATIONS = 200
WARMUP_ITERATIONS = 20
HEAD_COUNTS = (1, 10, 50)


@dataclass(frozen=True, slots=True)
class EncoderPoint:
    name: str
    encoder_name: str
    encoder_revision: str
    initialisation: str
    nominal_parameters: str


POINTS: tuple[EncoderPoint, ...] = (
    EncoderPoint(
        "modernbert-base",
        DEFAULT_ENCODER_NAME,
        DEFAULT_ENCODER_REVISION,
        "pretrained ModernBERT-base (decision-3 default)",
        "~150M",
    ),
    EncoderPoint(
        "modernbert-large-laya-init",
        str(_REPOSITORY_ROOT / "benchmarks/refund/data/encoders/laya-typed-decisions-encoder"),
        "dd079950600224fb459af2a0cb1d74e1e57ee9cf",
        "ModernBERT-large weights taken from the Laya typed-decisions checkpoint (decision-trained)",
        "~400M",
    ),
    EncoderPoint(
        "modernbert-large",
        "answerdotai/ModernBERT-large",
        "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
        "pretrained ModernBERT-large",
        "~400M",
    ),
    EncoderPoint(
        "deberta-v2-xlarge",
        "microsoft/deberta-v2-xlarge",
        "1d134961d4db8e7e8eb1bc1ab81cb370244c57f7",
        "pretrained DeBERTa-v2-xlarge (the ~1B point)",
        "~0.9B",
    ),
)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def recipe(point: EncoderPoint, device: str, epochs: int, batch_size: int) -> TrainingConfig:
    """The committed compact-release recipe with the encoder swapped in."""

    return TrainingConfig(
        encoder_name=point.encoder_name,
        encoder_revision=point.encoder_revision,
        local_files_only=True,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=3e-5,
        weight_decay=0.01,
        maximum_sequence_length=128,
        evaluation_ratio=0.1,
        seed=1,
        device=device,
        head_architecture="linear",
        select_best_epoch=True,
        maximum_trainable_parameters=1_500_000_000,
    )


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[rank]


def summarize(samples: Sequence[float]) -> dict[str, float]:
    return {
        "p50Ms": round(percentile(samples, 0.5), 3),
        "p95Ms": round(percentile(samples, 0.95), 3),
        "meanMs": round(statistics.fmean(samples), 3),
        "minMs": round(min(samples), 3),
        "count": len(samples),
    }


def timed(run: Any, iterations: int, warmup: int, synchronize: Any = None) -> list[float]:
    for _ in range(warmup):
        run()
        if synchronize:
            synchronize()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        run()
        if synchronize:
            synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def parity_inputs(
    ir: Any, tokenizer: Any, config: TrainingConfig, sample_inputs: dict[str, Any]
) -> Any:
    text = serialize_canonical_inputs(
        ir["inputs"], sample_inputs, version=config.canonical_input_version
    ).decode("utf-8")
    return tokenizer(
        [text],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=config.maximum_sequence_length,
        return_tensors="pt",
    )


def measure_gpu(model: Any, encoded: Any, torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    model.eval()
    device = torch.device("cuda")
    model.to(device)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        samples = timed(
            lambda: model(input_ids=input_ids, attention_mask=attention_mask),
            LATENCY_ITERATIONS,
            WARMUP_ITERATIONS,
            torch.cuda.synchronize,
        )
    model.to("cpu")
    torch.cuda.empty_cache()
    return {"available": True, "framework": "pytorch-eager", **summarize(samples)}


def measure_cpu_onnx(paths: dict[str, Path], encoded: Any) -> dict[str, Any]:
    import numpy
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    sessions = {
        name: onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        for name, path in paths.items()
    }
    feed = {
        "input_ids": encoded["input_ids"].numpy().astype(numpy.int64),
        "attention_mask": encoded["attention_mask"].numpy().astype(numpy.int64),
    }

    def chain() -> Any:
        sentence = sessions["encoder"].run(None, feed)[0]
        function = sessions["adapter"].run(None, {"sentence_embedding": sentence})[0]
        return sessions["head"].run(None, {"function_embedding": function})[0]

    result: dict[str, Any] = {
        "threads": "onnxruntime default",
        "chain": summarize(timed(chain, LATENCY_ITERATIONS, WARMUP_ITERATIONS)),
    }
    sentence = sessions["encoder"].run(None, feed)[0]
    function = sessions["adapter"].run(None, {"sentence_embedding": sentence})[0]

    def head_only() -> Any:
        return sessions["head"].run(None, {"function_embedding": function})[0]

    result["headOnly"] = summarize(timed(head_only, LATENCY_ITERATIONS, WARMUP_ITERATIONS))
    result["encoderOnly"] = summarize(
        timed(lambda: sessions["encoder"].run(None, feed), LATENCY_ITERATIONS, WARMUP_ITERATIONS)
    )
    heads = {}
    for count in HEAD_COUNTS:

        def stage(count: int = count) -> None:
            sentence = sessions["encoder"].run(None, feed)[0]
            function = sessions["adapter"].run(None, {"sentence_embedding": sentence})[0]
            for _ in range(count):
                sessions["head"].run(None, {"function_embedding": function})

        samples = timed(stage, max(40, LATENCY_ITERATIONS // count), WARMUP_ITERATIONS // 2)
        summary = summarize(samples)
        heads[str(count)] = {**summary, "perDecisionP50Ms": round(summary["p50Ms"] / count, 3)}
    result["headsPerStage"] = heads
    return result


def export_chain(training: Any, encoded: Any, directory: Path) -> dict[str, Path]:
    from semantscript_model.export import export_application_components

    model = training.model
    model.to("cpu")
    paths = {
        "encoder": directory / "encoder.onnx",
        "adapter": directory / "adapter.onnx",
        "head": directory / "head.onnx",
    }
    export_application_components(
        model.encoder,
        None,
        {"function": model.head},
        encoded["input_ids"],
        encoded["attention_mask"],
        encoder_path=paths["encoder"],
        adapter_path=paths["adapter"],
        head_paths={"function": paths["head"]},
    )
    return paths


def run_point(
    point: EncoderPoint,
    ir: Any,
    base: Any,
    adversarial: Any,
    release: Any,
    device: str,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    record: dict[str, Any] = {
        "point": asdict(point),
        "recipe": {
            "epochs": epochs,
            "batchSize": batch_size,
            "learningRate": 3e-5,
            "head": "linear",
            "loss": "proper",
        },
    }
    attempts = []
    training = None
    config = None
    for attempt_batch in (batch_size, max(1, batch_size // 4)):
        config = recipe(point, device, epochs, attempt_batch)
        log(f"{point.name}: training {epochs} epoch(s) at batch {attempt_batch}")
        started = time.monotonic()
        try:
            training = train_classifier(ir, base, adversarial, config=config)
        except (TrainingExecutionError, RuntimeError, torch.cuda.OutOfMemoryError) as error:
            message = str(error)
            attempts.append(
                {
                    "batchSize": attempt_batch,
                    "outcome": "failed",
                    "error": message[:300],
                    "seconds": round(time.monotonic() - started, 1),
                }
            )
            log(f"{point.name}: batch {attempt_batch} failed: {message[:200]}")
            torch.cuda.empty_cache()
            if "out of memory" not in message.lower() and "HIP" not in message:
                break
            continue
        attempts.append(
            {
                "batchSize": attempt_batch,
                "outcome": "trained",
                "seconds": round(time.monotonic() - started, 1),
            }
        )
        break
    record["trainingAttempts"] = attempts
    if training is None or config is None:
        record["status"] = "not-trained"
        return record
    record["parameters"] = sum(p.numel() for p in training.model.parameters())
    record["selectedEpoch"] = training.selected_epoch
    record["heldOutCurve"] = [round(m.held_out_accuracy, 4) for m in training.metrics]
    tokenizer = _load_tokenizer(config)
    verification = evaluate_training_result(
        ir,
        training,
        base,
        adversarial,
        tokenizer=tokenizer,
        attested_verification=release.generated_cases,
        config=VerificationConfig(maximum_constraint_violation_rate=0.01),
        verified_at=_utc_now(),
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
        "failures": list(verification.failures)[:3],
    }
    log(f"{point.name}: verification {json.dumps(record['verification'])[:300]}")

    sample_inputs = dict(base.cases[0].inputs)
    encoded = parity_inputs(ir, tokenizer, config, sample_inputs)
    record["latencyTokens"] = int(encoded["input_ids"].shape[1])
    record["gpu"] = measure_gpu(training.model, encoded, torch)
    log(f"{point.name}: gpu {json.dumps(record['gpu'])}")
    with tempfile.TemporaryDirectory(prefix=f"semantscript-sweep-{point.name}-") as temporary:
        paths = export_chain(training, encoded, Path(temporary))
        record["artifactBytes"] = {name: path.stat().st_size for name, path in paths.items()}
        record["artifactBytes"]["total"] = sum(record["artifactBytes"].values())
        record["cpuOnnx"] = measure_cpu_onnx(paths, encoded)
    log(f"{point.name}: cpu onnx chain {json.dumps(record['cpuOnnx']['chain'])}")
    record["status"] = "measured"
    del training
    torch.cuda.empty_cache()
    return record


def environment(device: str) -> dict[str, Any]:
    import onnxruntime
    import torch

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
    }
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover
        pass
    import os

    return {
        "capturedAt": _utc_now(),
        "operatingSystem": f"{platform.system()} {platform.release()}",
        "cpu": f"{platform.processor() or platform.machine()} x{os.cpu_count()}",
        "device": device,
        "accelerator": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpuMemoryBytes": torch.cuda.get_device_properties(0).total_memory
        if torch.cuda.is_available()
        else None,
        "onnxruntimeProviders": onnxruntime.get_available_providers(),
        "versions": versions,
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument("--heldout-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--points", default=",".join(point.name for point in POINTS))
    arguments = parser.parse_args(argv)
    selected = {name.strip() for name in arguments.points.split(",") if name.strip()}
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
    with tempfile.TemporaryDirectory(prefix="semantscript-encoder-sweep-") as temporary:
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
    log(
        f"corpus: {base.synthetic_count} synthetic + {len(adversarial.cases)} adversarial rows; "
        f"release cases: {len(release.case_ids)}"
    )
    results.update(
        {
            "kind": RESULT_KIND,
            "resultVersion": RESULT_VERSION,
            "startedAt": results.get("startedAt", _utc_now()),
            "function": {"id": compiled.function_id, "semanticSha256": compiled.semantic_sha256},
            "corpus": {
                "manifest": str(manifest_path.resolve().relative_to(_REPOSITORY_ROOT)),
                "manifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "baseDatasetSha256": base.dataset_sha256,
                "adversarialDatasetSha256": adversarial.dataset_sha256,
            },
            "releaseCases": len(release.case_ids),
            "latency": {
                "iterations": LATENCY_ITERATIONS,
                "warmup": WARMUP_ITERATIONS,
                "batch": 1,
                "headCounts": list(HEAD_COUNTS),
                "cpu": "onnxruntime CPUExecutionProvider, default threads",
                "gpu": "PyTorch eager (onnxruntime has no GPU provider on this machine)",
            },
            "environment": environment(arguments.device),
        }
    )
    results.setdefault("points", {})

    for point in POINTS:
        if point.name not in selected:
            continue
        results["points"][point.name] = run_point(
            point,
            ir,
            base,
            adversarial,
            release,
            arguments.device,
            arguments.epochs,
            arguments.batch_size,
        )
        results["completedAt"] = _utc_now()
        results_path.write_text(
            json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        log(f"wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
