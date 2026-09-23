"""Sweep training settings and report release-gate evidence without exporting.

For each configuration: replay the frozen corpus, train on the GPU, evaluate
against the attested release record (no gate is enforced) and print held-out
accuracy, release misses, constraint violations and ECE. Use it to choose the
settings for ``run_release_pipeline``.

Usage::

    PYTHONNOUSERSITE=1 PYTHONPATH=.:trainer/src:model/src:.python-packages python -m \\
        benchmarks.refund.program.sweep_training \\
        --corpus-manifest benchmarks/refund/data/opus-v2-2026-09-23/manifest.json \\
        --heldout-dir benchmarks/refund/data/heldout-uci-2026-09-23 \\
        --configs "epochs=4,lr=2e-5" "epochs=8,lr=3e-5"
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Sequence
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
from semantscript_trainer.dataset import SyntheticDatasetGenerator
from semantscript_trainer.training import (
    DEFAULT_ENCODER_NAME,
    DEFAULT_ENCODER_REVISION,
    TrainingConfig,
    train_classifier,
)
from semantscript_trainer.verification import VerificationConfig, evaluate_training_result


def parse_config(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for part in text.split(","):
        key, _, raw = part.partition("=")
        key = key.strip()
        if key in ("epochs", "batch", "seed", "maxlen"):
            values[key] = int(raw)
        elif key in ("lr", "wd", "evalratio"):
            values[key] = float(raw)
        elif key == "head":
            values[key] = raw.strip()
        else:
            raise ValueError(f"unknown setting {key!r}")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument("--heldout-dir", required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dump-release-predictions", default=None)
    arguments = parser.parse_args(argv)

    manifest_path = Path(arguments.corpus_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_root = manifest_path.parent
    heldout = Path(arguments.heldout_dir)
    with tempfile.TemporaryDirectory(prefix="semantscript-sweep-") as temporary:
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
    print(
        f"corpus: {base.synthetic_count} synthetic + {len(adversarial.cases)} adversarial; "
        f"release cases: {len(release.case_ids)}",
        flush=True,
    )

    for text in arguments.configs:
        settings = parse_config(text)
        config = TrainingConfig(
            encoder_name=DEFAULT_ENCODER_NAME,
            encoder_revision=DEFAULT_ENCODER_REVISION,
            local_files_only=True,
            epochs=settings.get("epochs", 3),
            batch_size=settings.get("batch", 16),
            learning_rate=settings.get("lr", 2e-5),
            weight_decay=settings.get("wd", 0.01),
            maximum_sequence_length=settings.get("maxlen", 128),
            evaluation_ratio=settings.get("evalratio", 0.2),
            seed=settings.get("seed", 1),
            device=arguments.device,
            head_architecture=settings.get("head", "linear"),
        )
        started = time.monotonic()
        training = train_classifier(ir, base, adversarial, config=config)
        verification = evaluate_training_result(
            ir,
            training,
            base,
            adversarial,
            attested_verification=release.generated_cases,
            config=VerificationConfig(),
            verified_at=_utc_now(),
        )
        metrics = verification.metrics
        curve = " ".join(f"{m.held_out_accuracy:.3f}" for m in training.metrics)
        print(
            json.dumps(
                {
                    "config": text,
                    "seconds": round(time.monotonic() - started, 1),
                    "heldOutCurve": curve,
                    "calibratedAccuracy": round(metrics.accuracy, 4),
                    "releaseMisses": metrics.example_failures,
                    "releaseAccuracy": round(
                        1 - metrics.example_failures / len(release.case_ids), 4
                    ),
                    "constraintViolations": metrics.constraint_violations,
                    "pairConsistency": round(metrics.pair_consistency, 4),
                    "ece": round(metrics.ece, 4),
                    "status": verification.status,
                }
            ),
            flush=True,
        )
        if arguments.dump_release_predictions:
            _dump_release_predictions(
                ir, training, config, release, Path(arguments.dump_release_predictions), text
            )
        del training
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    return 0


def _dump_release_predictions(
    ir: dict[str, Any],
    training: Any,
    config: TrainingConfig,
    release: Any,
    path: Path,
    label: str,
) -> None:
    """Write per-case release predictions (expected, predicted, probabilities)."""

    import torch
    from transformers import AutoTokenizer

    from semantscript_trainer.canonical_input import serialize_canonical_inputs

    tokenizer = AutoTokenizer.from_pretrained(
        config.encoder_name,
        revision=config.encoder_revision,
        local_files_only=config.local_files_only,
        trust_remote_code=False,
        use_fast=True,
    )
    model = training.model
    model.eval()
    device = next(model.parameters()).device
    support = list(training.head.support)
    rows = []
    document = release.document
    with torch.no_grad():
        for case in document["cases"]:
            text = serialize_canonical_inputs(ir["inputs"], case["inputs"]).decode("utf-8")
            encoded = tokenizer(
                [text],
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=config.maximum_sequence_length,
                return_tensors="pt",
            )
            logits = model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
            )
            probabilities = torch.softmax(logits[0].float(), dim=-1).tolist()
            predicted = support[max(range(len(support)), key=lambda i: probabilities[i])]
            rows.append(
                {
                    "id": case["id"],
                    "inputs": case["inputs"],
                    "expected": case["expected"],
                    "predicted": predicted,
                    "probabilities": dict(
                        zip(support, [round(p, 4) for p in probabilities], strict=True)
                    ),
                }
            )
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing[label] = rows
    path.write_text(json.dumps(existing, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
