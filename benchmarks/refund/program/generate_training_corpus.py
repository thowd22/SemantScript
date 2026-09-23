"""Generate and freeze a synthetic-training-only refund corpus with the CLI teacher.

The corpus is model-generated training data only. It is never a benchmark set,
never human-authored, and never used for release verification or final
evaluation. The manifest records the teacher identity, the run report, and the
dataset digests so the corpus can be audited and bound into a training ledger.

Usage::

    PYTHONPATH=.:trainer/src:model/src:.python-packages python -m \\
        benchmarks.refund.program.generate_training_corpus \\
        --output-dir benchmarks/refund/data/<run> --synthetic-count 600 \\
        --counterfactual-ratio 0.25 --concurrency 4
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import platform
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.refund.program.claude_cli_teacher import (
    CLAUDE_CLI_DATA_CLASSIFICATION,
    ClaudeCliTeacherConfig,
    ClaudeCliTrainingTeacher,
)
from benchmarks.refund.program.pipeline import compile_refund_program

from semantscript_trainer import (
    AdversarialDatasetGenerator,
    AdversarialGenerationConfig,
    SyntheticDatasetGenerator,
)

MANIFEST_KIND = "semantscript.refund-training-corpus-manifest"
MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
NOTICE = (
    "Model-generated training data only. Not human-authored, not a benchmark or "
    "release-verification set, and never to be used as an evaluation target."
)


class CorpusGenerationError(RuntimeError):
    """The corpus could not be generated or frozen."""


def generate_training_corpus(
    output_directory: str | Path,
    *,
    synthetic_count: int,
    counterfactual_ratio: float,
    config: ClaudeCliTeacherConfig,
    teacher: ClaudeCliTrainingTeacher | None = None,
) -> dict[str, Any]:
    """Compile, generate synthetic and adversarial data, and write the manifest."""

    output = Path(output_directory)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise CorpusGenerationError("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve(strict=True)

    compiled = compile_refund_program(output / "compiler")
    bundle_sha256 = hashlib.sha256(compiled.bundle_path.read_bytes()).hexdigest()
    teacher = teacher if teacher is not None else ClaudeCliTrainingTeacher(config)
    teacher.verify_installation()
    started_at = _utc_now()

    synthetic_generator = SyntheticDatasetGenerator(teacher, output / "synthetic")
    synthetic_started = time.monotonic()
    synthetic = synthetic_generator.generate(compiled.source_ir, synthetic_count)
    synthetic_seconds = time.monotonic() - synthetic_started
    synthetic_report = teacher.last_run_report

    adversarial_config = AdversarialGenerationConfig(counterfactual_ratio=counterfactual_ratio)
    adversarial_generator = AdversarialDatasetGenerator(
        teacher,
        output / "adversarial",
        config=adversarial_config,
    )
    adversarial_started = time.monotonic()
    adversarial = adversarial_generator.generate(compiled.source_ir, synthetic)
    adversarial_seconds = time.monotonic() - adversarial_started

    provenance = teacher.provenance
    manifest: dict[str, Any] = {
        "kind": MANIFEST_KIND,
        "manifestVersion": MANIFEST_VERSION,
        "dataClassification": CLAUDE_CLI_DATA_CLASSIFICATION,
        "notice": NOTICE,
        "startedAt": started_at,
        "completedAt": _utc_now(),
        "function": {
            "id": compiled.function_id,
            "semanticSha256": compiled.semantic_sha256,
            "bundlePath": _relative(compiled.bundle_path, output),
            "bundleSha256": bundle_sha256,
        },
        "teacher": {
            "provider": provenance.provider,
            "model": provenance.model,
            "cliVersion": provenance.cli_version,
            "protocol": provenance.protocol,
            "dataClassification": provenance.data_classification,
            "configurationSha256": provenance.configuration_sha256,
            "configuration": teacher.configuration_projection,
        },
        "synthetic": {
            "requestedCaseCount": synthetic.requested_case_count,
            "goldCount": synthetic.gold_count,
            "syntheticCount": synthetic.synthetic_count,
            "cachePath": _relative(
                synthetic_generator.cache_path(compiled.source_ir, synthetic_count), output
            ),
            "cacheKeySha256": synthetic.cache_key_sha256,
            "payloadSha256": synthetic.payload_sha256,
            "datasetSha256": synthetic.dataset_sha256,
            "labelCounts": _label_counts(case.output for case in synthetic.cases),
            "uniqueInputCount": len({_canonical(case.inputs) for case in synthetic.cases}),
            "runReport": synthetic_report.document() if synthetic_report else None,
            "elapsedSeconds": round(synthetic_seconds, 3),
        },
        "adversarial": {
            "config": adversarial_config.document(),
            "cachePath": _relative(
                adversarial_generator.cache_path(compiled.source_ir, synthetic), output
            ),
            "cacheKeySha256": adversarial.cache_key_sha256,
            "payloadSha256": adversarial.payload_sha256,
            "datasetSha256": adversarial.dataset_sha256,
            "baseDatasetSha256": adversarial.base_dataset_sha256,
            "caseCount": len(adversarial.cases),
            "pairCount": len(adversarial.pairs),
            "caseCountsByTag": dict(
                sorted(collections.Counter(case.tag for case in adversarial.cases).items())
            ),
            "labelCounts": _label_counts(case.output for case in adversarial.cases),
            "elapsedSeconds": round(adversarial_seconds, 3),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    encoded = json.dumps(
        manifest, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    (output / MANIFEST_NAME).write_bytes(encoded + b"\n")
    return manifest


def _label_counts(outputs: Any) -> dict[str, int]:
    counter = collections.Counter(_canonical(output) for output in outputs)
    return dict(sorted(counter.items()))


def _canonical(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synthetic-count", type=int, required=True)
    parser.add_argument("--counterfactual-ratio", type=float, default=0.25)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--maximum-case-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--max-budget-usd", type=float, default=2.0)
    parser.add_argument("--executable", default="claude")
    parser.add_argument("--require-cli-version", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    config = ClaudeCliTeacherConfig(
        executable=arguments.executable,
        timeout_seconds=arguments.timeout_seconds,
        max_budget_usd=arguments.max_budget_usd,
        concurrency=arguments.concurrency,
        maximum_case_attempts=arguments.maximum_case_attempts,
        cli_version=arguments.require_cli_version,
    )
    try:
        manifest = generate_training_corpus(
            arguments.output_dir,
            synthetic_count=arguments.synthetic_count,
            counterfactual_ratio=arguments.counterfactual_ratio,
            config=config,
        )
    except Exception as error:
        sys.stderr.write(f"corpus generation failed: {type(error).__name__}: {error}\n")
        return 1
    summary = {
        "synthetic": {
            key: manifest["synthetic"][key]
            for key in (
                "syntheticCount",
                "uniqueInputCount",
                "labelCounts",
                "runReport",
                "elapsedSeconds",
            )
        },
        "adversarial": {
            key: manifest["adversarial"][key]
            for key in ("caseCount", "pairCount", "caseCountsByTag", "elapsedSeconds")
        },
        "manifest": str(Path(arguments.output_dir) / MANIFEST_NAME),
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
