"""Derive an int8 release from a passing refund release and verify it on the quantized graph.

Reads a release pipeline manifest, replays the frozen corpus and the attested
release record exactly as the release pipeline does (teacher calls forbidden),
runs the quantized encoder chain against the float32 chain over every one of
those records, and publishes the derived release under a separate artifact
root with its own pipeline manifest so the benchmark harness can measure it
unchanged. The quantization report is written whether or not the gate passes.

Usage:
  python -m benchmarks.refund.program.quantize_release \\
    --pipeline-manifest benchmarks/refund/data/release-2026-09-23/pipeline-manifest.json \\
    --heldout-dir benchmarks/refund/data/heldout-uci-2026-09-23 \\
    --output-dir benchmarks/refund/data/release-int8-2026-09-24 \\
    [--weight-type int8|uint8] [--per-channel] [--reduce-range] \\
    [--maximum-argmax-disagreement-rate 0.0] [--maximum-attested-disagreements 0] \\
    [--ece-threshold 0.1]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmarks.refund.program.pipeline import (
    compile_refund_program,
    parse_release_verification_record,
    run_refund_runtime,
)
from benchmarks.refund.program.run_release_pipeline import (
    ReleasePipelineError,
    _dump,
    _Logger,
    _teacher_from_manifest,
    _utc_now,
)

from semantscript_trainer import (
    AdversarialDatasetGenerator,
    AdversarialGenerationConfig,
    QuantizationConfig,
    QuantizationGateError,
    QuantizationRecord,
    quantize_release_artifact,
)
from semantscript_trainer.dataset import SyntheticDatasetGenerator

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def quantize_release(
    pipeline_manifest_path: str | Path,
    heldout_directory: str | Path,
    output_directory: str | Path,
    *,
    quantization: QuantizationConfig | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    pipeline_manifest_path = Path(pipeline_manifest_path)
    heldout = Path(heldout_directory)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise ReleasePipelineError("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
    if source.get("kind") != "semantscript.refund-release-pipeline-manifest":
        raise ReleasePipelineError("pipeline manifest is not a refund release pipeline manifest")
    settings = QuantizationConfig() if quantization is None else quantization
    log = _Logger(output / "pipeline.log")

    log("compiling canonical refund program")
    compiled = compile_refund_program(output / "compiler")
    if (
        compiled.function_id != source["function"]["id"]
        or compiled.semantic_sha256 != source["function"]["semanticSha256"]
    ):
        raise ReleasePipelineError("compiled function differs from the source pipeline manifest")
    ir = compiled.source_ir
    support = list(ir["output"]["head"]["support"])

    log("replaying frozen corpus from cache")
    corpus_manifest_path = Path(source["corpus"]["manifestPath"])
    corpus_root = corpus_manifest_path.parent
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    teacher = _teacher_from_manifest(corpus_manifest)
    base = SyntheticDatasetGenerator(teacher, corpus_root / "synthetic").generate(
        ir, corpus_manifest["synthetic"]["requestedCaseCount"]
    )
    if base.dataset_sha256 != source["corpus"]["syntheticDatasetSha256"]:
        raise ReleasePipelineError("replayed synthetic dataset digest differs from the manifest")
    adversarial = AdversarialDatasetGenerator(
        teacher,
        corpus_root / "adversarial",
        config=AdversarialGenerationConfig(
            counterfactual_ratio=corpus_manifest["adversarial"]["config"]["counterfactualRatio"],
            maximum_attempts=corpus_manifest["adversarial"]["config"]["maximumAttempts"],
        ),
    ).generate(ir, base)
    if adversarial.dataset_sha256 != source["corpus"]["adversarialDatasetSha256"]:
        raise ReleasePipelineError("replayed adversarial dataset digest differs from the manifest")

    log("loading attested release record")
    release = parse_release_verification_record(
        (heldout / "release-verification.json").read_bytes(), ir
    )
    if release.payload_sha256 != source["release"]["payloadSha256"]:
        raise ReleasePipelineError("attested release record differs from the source pipeline")

    records = [
        QuantizationRecord(dict(case.inputs), support.index(case.output))
        for case in (*base.cases, *adversarial.cases)
    ] + [
        QuantizationRecord(dict(case.inputs), support.index(case.output), attested=True)
        for case in release.generated_cases
    ]
    log(
        f"verifying the quantized graph on {len(records)} records "
        f"({len(release.generated_cases)} attested), weight type {settings.weight_type}, "
        f"per-channel {settings.per_channel}, reduce-range {settings.reduce_range}"
    )
    report_path = output / "quantization-report.json"
    try:
        quantized = quantize_release_artifact(
            source["artifact"]["root"],
            source["artifact"]["manifestSha256"],
            records=records,
            quantization=settings,
            artifact_root=output / "artifact",
        )
    except QuantizationGateError as error:
        report = {
            "kind": "semantscript.refund-quantization-report",
            "status": "failed",
            "reason": str(error),
            "settings": settings.to_manifest_document(source["artifact"]["manifestSha256"]),
            "report": error.report.to_document(),
        }
        report_path.write_text(_dump(report), encoding="utf-8")
        log(f"quantization gate failed: {error}")
        raise
    report = quantized.report
    log(
        f"quantized graph verified: {report.argmax_disagreements} of {report.records_checked} "
        f"decisions changed ({report.attested_disagreements} attested), quantized ECE "
        f"{report.quantized_ece:.4f} vs source {report.source_ece:.4f}, encoder "
        f"{report.source_encoder_byte_length} -> {report.quantized_encoder_byte_length} bytes"
    )

    log("running one diagnostic call through the Node runtime")
    sample_inputs = dict(base.cases[0].inputs)
    diagnostic = run_refund_runtime(quantized.artifact_root, compiled.function_id, sample_inputs)
    log(f"runtime diagnostic: {diagnostic['value']} (confidence {diagnostic['confidence']:.4f})")

    derived = deepcopy(source)
    derived["derivedFrom"] = {
        "pipelineManifestPath": str(pipeline_manifest_path),
        "artifactManifestSha256": source["artifact"]["manifestSha256"],
    }
    derived["artifact"] = {
        "root": str(quantized.artifact_root),
        "releaseDirectory": str(quantized.release_directory),
        "manifestSha256": quantized.manifest_sha256,
        "encoderPrecision": "int8-dynamic",
    }
    derived["quantization"] = {
        "settings": settings.to_manifest_document(source["artifact"]["manifestSha256"]),
        "report": report.to_document(),
    }
    derived["runtimeDiagnostic"] = diagnostic
    derived["completedAt"] = _utc_now()
    derived["elapsedSeconds"] = round(time.monotonic() - started, 3)
    (output / "pipeline-manifest.json").write_text(_dump(derived), encoding="utf-8")
    report_path.write_text(
        _dump(
            {
                "kind": "semantscript.refund-quantization-report",
                "status": "passed",
                "settings": derived["quantization"]["settings"],
                "report": derived["quantization"]["report"],
                "artifact": derived["artifact"],
            }
        ),
        encoding="utf-8",
    )
    log(f"done in {derived['elapsedSeconds']}s")
    return derived


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--pipeline-manifest", required=True)
    parser.add_argument("--heldout-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weight-type", choices=("int8", "uint8"), default="int8")
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--reduce-range", action="store_true")
    parser.add_argument("--maximum-argmax-disagreement-rate", type=float, default=0.0)
    parser.add_argument("--maximum-attested-disagreements", type=int, default=0)
    parser.add_argument("--ece-threshold", type=float, default=0.1)
    arguments = parser.parse_args(argv)
    settings = QuantizationConfig(
        weight_type=arguments.weight_type,
        per_channel=arguments.per_channel,
        reduce_range=arguments.reduce_range,
        maximum_argmax_disagreement_rate=arguments.maximum_argmax_disagreement_rate,
        maximum_attested_disagreements=arguments.maximum_attested_disagreements,
        ece_threshold=arguments.ece_threshold,
    )
    try:
        quantize_release(
            arguments.pipeline_manifest,
            arguments.heldout_dir,
            arguments.output_dir,
            quantization=settings,
        )
    except QuantizationGateError as error:
        print(f"quantization refused: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"quantization failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
