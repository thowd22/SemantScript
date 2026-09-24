"""Freeze a corpus of real pool inputs (rule-labeled) plus replayed Opus cases.

Usage::

    PYTHONPATH=.:trainer/src:model/src:.python-packages python -m \\
        benchmarks.refund.program.generate_pooled_corpus \\
        --pool benchmarks/refund/data/uci-pool/candidates.json \\
        --replay-manifest benchmarks/refund/data/opus-v2-2026-09-23/manifest.json \\
        --heldout-dir benchmarks/refund/data/heldout-uci-2026-09-23 \\
        --output-dir benchmarks/refund/data/pooled-v3-2026-09-23 \\
        --counterfactual-ratio 0.03 --model claude-opus-5-5 --concurrency 6
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
from benchmarks.refund.program.pooled_corpus import PooledCorpusTeacher

from semantscript_trainer import (
    AdversarialDatasetGenerator,
    AdversarialGenerationConfig,
    SyntheticDatasetGenerator,
)

MANIFEST_KIND = "semantscript.refund-training-corpus-manifest"
NOTICE = (
    "Model-generated and rule-labeled training data only. Real inputs are de-identified "
    "public transactions with every held-out input excluded by digest; labels are the unique "
    "output admitted by the compiled constraints. Never a benchmark or release set."
)


def generate_pooled_corpus(
    output_directory: str | Path,
    *,
    pool_path: Path,
    replay_manifest_path: Path | None,
    heldout_directory: Path,
    fraud_percent: int,
    counterfactual_ratio: float,
    adversarial_attempts: int,
    cli_config: ClaudeCliTeacherConfig,
    case_limit: int | None = None,
    resume: bool = False,
    stale_status_twins: bool = False,
) -> dict[str, Any]:
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()) and not resume:
        raise RuntimeError("output directory must be absent or empty (use resume to continue)")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve(strict=True)
    compiler_directory = output / "compiler"
    if resume and compiler_directory.exists():
        import shutil

        shutil.rmtree(compiler_directory)
    compiled = compile_refund_program(compiler_directory)
    ir = compiled.source_ir

    replay_path: Path | None = None
    if replay_manifest_path is not None:
        replay_manifest = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
        replay_path = replay_manifest_path.parent / replay_manifest["synthetic"]["cachePath"]
    adversarial_teacher = ClaudeCliTrainingTeacher(cli_config)
    teacher = PooledCorpusTeacher(
        pool_path=pool_path,
        replay_path=replay_path,
        heldout_directory=heldout_directory,
        fraud_percent=fraud_percent,
        adversarial=adversarial_teacher,
        stale_status_twins=stale_status_twins,
    )
    adversarial_teacher.verify_installation()
    available = len(teacher.generate(ir, 0)) or len(teacher.labeled_pool(ir)) + (
        0 if replay_path is None else len(teacher._replay_cases)
    )
    requested = available if case_limit is None else min(case_limit, available)
    pool_report = teacher.last_pool_report

    started_at = _utc_now()
    synthetic_generator = SyntheticDatasetGenerator(teacher, output / "synthetic")
    synthetic_started = time.monotonic()
    synthetic = synthetic_generator.generate(ir, requested)
    synthetic_seconds = time.monotonic() - synthetic_started

    adversarial_config = AdversarialGenerationConfig(
        counterfactual_ratio=counterfactual_ratio, maximum_attempts=adversarial_attempts
    )
    adversarial_generator = AdversarialDatasetGenerator(
        teacher, output / "adversarial", config=adversarial_config
    )
    adversarial_started = time.monotonic()
    adversarial = adversarial_generator.generate(ir, synthetic)
    adversarial_seconds = time.monotonic() - adversarial_started

    descriptor = teacher.descriptor
    manifest: dict[str, Any] = {
        "kind": MANIFEST_KIND,
        "manifestVersion": 1,
        "dataClassification": CLAUDE_CLI_DATA_CLASSIFICATION,
        "notice": NOTICE,
        "resumed": resume,
        "startedAt": started_at,
        "completedAt": _utc_now(),
        "function": {
            "id": compiled.function_id,
            "semanticSha256": compiled.semantic_sha256,
            "bundlePath": _relative(compiled.bundle_path, output),
            "bundleSha256": hashlib.sha256(compiled.bundle_path.read_bytes()).hexdigest(),
        },
        "teacher": {
            "provider": descriptor.provider,
            "model": descriptor.model,
            "cliVersion": adversarial_teacher.provenance.cli_version,
            "protocol": adversarial_teacher.provenance.protocol,
            "dataClassification": CLAUDE_CLI_DATA_CLASSIFICATION,
            "configurationSha256": descriptor.configuration_sha256,
            "configuration": teacher.configuration_projection,
        },
        "synthetic": {
            "requestedCaseCount": synthetic.requested_case_count,
            "goldCount": synthetic.gold_count,
            "syntheticCount": synthetic.synthetic_count,
            "cachePath": _relative(synthetic_generator.cache_path(ir, requested), output),
            "cacheKeySha256": synthetic.cache_key_sha256,
            "payloadSha256": synthetic.payload_sha256,
            "datasetSha256": synthetic.dataset_sha256,
            "labelCounts": _label_counts(case.output for case in synthetic.cases),
            "uniqueInputCount": len({_canonical(case.inputs) for case in synthetic.cases}),
            "poolReport": None if pool_report is None else pool_report.document(),
            "replayedCaseCount": 0 if replay_path is None else len(teacher._replay_cases),
            "loadedFromCache": False,
            "runReport": None,
            "elapsedSeconds": round(synthetic_seconds, 3),
        },
        "adversarial": {
            "config": adversarial_config.document(),
            "cachePath": _relative(adversarial_generator.cache_path(ir, synthetic), output),
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
            "runReport": None
            if adversarial_teacher.last_run_report is None
            else adversarial_teacher.last_run_report.document(),
            "elapsedSeconds": round(adversarial_seconds, 3),
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _label_counts(outputs: Any) -> dict[str, int]:
    return dict(sorted(collections.Counter(_canonical(output) for output in outputs).items()))


def _canonical(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--pool", required=True)
    parser.add_argument("--replay-manifest", default=None)
    parser.add_argument("--heldout-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fraud-percent", type=int, default=15)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--counterfactual-ratio", type=float, default=0.03)
    parser.add_argument("--adversarial-attempts", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--maximum-case-attempts", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stale-status-twins", action="store_true")
    arguments = parser.parse_args(argv)
    cli_config = ClaudeCliTeacherConfig(
        timeout_seconds=arguments.timeout_seconds,
        concurrency=arguments.concurrency,
        maximum_case_attempts=arguments.maximum_case_attempts,
        model=arguments.model,
    )
    try:
        manifest = generate_pooled_corpus(
            arguments.output_dir,
            pool_path=Path(arguments.pool),
            replay_manifest_path=None
            if arguments.replay_manifest is None
            else Path(arguments.replay_manifest),
            heldout_directory=Path(arguments.heldout_dir),
            fraud_percent=arguments.fraud_percent,
            counterfactual_ratio=arguments.counterfactual_ratio,
            adversarial_attempts=arguments.adversarial_attempts,
            cli_config=cli_config,
            case_limit=arguments.case_limit,
            resume=arguments.resume,
            stale_status_twins=arguments.stale_status_twins,
        )
    except Exception as error:
        sys.stderr.write(f"pooled corpus generation failed: {type(error).__name__}: {error}\n")
        return 1
    summary = {
        "synthetic": {
            k: manifest["synthetic"][k]
            for k in (
                "syntheticCount",
                "uniqueInputCount",
                "labelCounts",
                "poolReport",
                "replayedCaseCount",
                "elapsedSeconds",
            )
        },
        "adversarial": {
            k: manifest["adversarial"][k]
            for k in ("caseCount", "pairCount", "caseCountsByTag", "runReport", "elapsedSeconds")
        },
    }
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
