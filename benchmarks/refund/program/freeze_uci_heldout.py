"""Freeze judge-adjudicated held-out sets into contract records.

Reads ``sample.json`` (from ``sample_uci_heldout``), ``RUBRIC.md`` and
``adjudications.json`` (one label and rationale per sampled case, written by
the judge) in one directory, then writes:

- ``release-verification.json``: the Python release-gate record with a judge
  attestation, validated against the compiled refund IR;
- ``final-benchmark-dataset.json``: the TypeScript benchmark dataset with
  ``independent-judge`` cases and a judge attestation, validated by the
  workspace's ``validateRefundDataset``;
- ``manifest.json``: source provenance, digests and label counts.

Labels are checked against the compiled constraints before anything is written.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.refund.program.pipeline import (
    JudgeAttestationInputs,
    build_release_verification_record,
    compile_refund_program,
)

from semantscript_trainer import GeneratedCase
from semantscript_trainer.case_contract import validate_case
from semantscript_trainer.constraints import compile_constraints
from semantscript_trainer.semantic_json import semantic_json_sha256

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_JUDGE_DECLARATION = (
    "The listed cases have expected outputs adjudicated case by case by the named independent "
    "model judge under the referenced rubric with a recorded rationale per case; their inputs "
    "derive from real, de-identified transactions; neither inputs nor labels were produced by "
    "any teacher model used for training, and the judge is not a training teacher for this "
    "benchmark."
)
_SUPPORT = ("approve", "deny", "review")


class FreezeError(RuntimeError):
    """The held-out sets could not be frozen."""


def freeze_heldout_sets(directory: str | Path, *, created_at: str | None = None) -> dict[str, Any]:
    root = Path(directory)
    sample = json.loads((root / "sample.json").read_text(encoding="utf-8"))
    rubric_bytes = (root / "RUBRIC.md").read_bytes()
    adjudication_bytes = (root / "adjudications.json").read_bytes()
    adjudications = json.loads(adjudication_bytes)
    rubric_sha256 = hashlib.sha256(rubric_bytes).hexdigest()
    evidence_sha256 = hashlib.sha256(adjudication_bytes).hexdigest()
    if adjudications.get("rubricSha256") != rubric_sha256:
        raise FreezeError("adjudications reference a different rubric digest")
    judge = JudgeAttestationInputs(
        provider=adjudications["judge"]["provider"],
        model=adjudications["judge"]["model"],
        interface=adjudications["judge"]["interface"],
        session_reference=adjudications["judge"]["sessionReference"],
        rubric_sha256=rubric_sha256,
    )
    attested_at = adjudications["adjudicatedAt"]
    created = created_at or _utc_now()

    with tempfile.TemporaryDirectory(prefix="semantscript-heldout-compile-") as temporary:
        compiled = compile_refund_program(Path(temporary) / "compiler")
        source_ir = compiled.source_ir
    constraints = compile_constraints(source_ir)

    labeled: dict[str, list[dict[str, Any]]] = {}
    for split in ("release", "final"):
        rows = []
        for case in sample[split]:
            verdict = adjudications["cases"].get(case["id"])
            if verdict is None:
                raise FreezeError(f"{split} case {case['id']} has no adjudication")
            expected = verdict["expected"]
            if expected not in _SUPPORT:
                raise FreezeError(f"{split} case {case['id']} has an invalid label")
            generated = GeneratedCase(inputs=case["inputs"], output=expected)
            validate_case(source_ir, generated)
            constraints.validate_case(generated)
            if semantic_json_sha256(case["inputs"]) != case["inputSha256"]:
                raise FreezeError(f"{split} case {case['id']} input digest mismatch")
            rows.append({**case, "expected": expected})
        rows.sort(key=lambda row: row["id"].encode("utf-8"))
        labeled[split] = rows
    extra = set(adjudications["cases"]) - {c["id"] for s in labeled.values() for c in s}
    if extra:
        raise FreezeError(f"adjudications cover unknown cases: {sorted(extra)[:3]}")
    release_digests = {c["inputSha256"] for c in labeled["release"]}
    final_digests = {c["inputSha256"] for c in labeled["final"]}
    if release_digests & final_digests:
        raise FreezeError("release and final inputs overlap")

    release = build_release_verification_record(
        source_ir,
        tuple(
            (row["id"], GeneratedCase(inputs=row["inputs"], output=row["expected"]))
            for row in labeled["release"]
        ),
        created_at=created,
        attested_at=attested_at,
        evidence_sha256=evidence_sha256,
        judge=judge,
    )
    release_document = release.document

    final_cases = [
        {
            "id": row["id"],
            "inputs": row["inputs"],
            "inputSha256": row["inputSha256"],
            "expected": row["expected"],
            "origin": "independent-judge",
        }
        for row in labeled["final"]
    ]
    final_payload: dict[str, Any] = {
        "kind": "semantscript.refund-benchmark-dataset",
        "datasetVersion": 1,
        "benchmark": "refund-decision",
        "split": "evaluation-only",
        "createdAt": created,
        "function": {"id": compiled.function_id, "semanticSha256": compiled.semantic_sha256},
        "support": list(_SUPPORT),
        "cases": final_cases,
        "humanAttestation": None,
        "judgeAttestation": {
            "judge": judge.document(),
            "rubricSha256": rubric_sha256,
            "attestedAt": attested_at,
            "caseIds": [row["id"] for row in final_cases],
            "declaration": _JUDGE_DECLARATION,
            "evidenceSha256": evidence_sha256,
        },
    }
    final_document = {**final_payload, "payloadSha256": semantic_json_sha256(final_payload)}
    _validate_with_typescript(final_document)

    (root / "release-verification.json").write_text(_dump(release_document), encoding="utf-8")
    (root / "final-benchmark-dataset.json").write_text(_dump(final_document), encoding="utf-8")
    manifest = {
        "kind": "semantscript.refund-benchmark-uci-heldout-manifest",
        "manifestVersion": 1,
        "createdAt": created,
        "source": sample["source"],
        "derivation": sample["derivation"],
        "sampling": {
            "samplingVersion": sample["samplingVersion"],
            "fraudPercent": sample["fraudPercent"],
            "quotas": sample["quotas"],
            "excludedTrainingInputCount": sample["excludedTrainingInputCount"],
            "skipped": sample["skipped"],
        },
        "judge": {
            **judge.document(),
            "rubricSha256": rubric_sha256,
            "adjudicatedAt": attested_at,
            "evidenceSha256": evidence_sha256,
        },
        "function": {"id": compiled.function_id, "semanticSha256": compiled.semantic_sha256},
        "release": {
            "caseCount": len(labeled["release"]),
            "labelCounts": _label_counts(labeled["release"]),
            "fraudulentCount": _fraud_count(labeled["release"]),
            "payloadSha256": release.payload_sha256,
            "attestationSha256": release.attestation_sha256,
        },
        "final": {
            "caseCount": len(labeled["final"]),
            "labelCounts": _label_counts(labeled["final"]),
            "fraudulentCount": _fraud_count(labeled["final"]),
            "payloadSha256": final_document["payloadSha256"],
        },
    }
    (root / "manifest.json").write_text(_dump(manifest), encoding="utf-8")
    return manifest


def _validate_with_typescript(document: dict[str, Any]) -> None:
    script = (
        "import { readFileSync } from 'node:fs';"
        "import { validateRefundDataset } from './benchmarks/refund/dist/index.js';"
        "const value = JSON.parse(readFileSync(process.env.SEMANTSCRIPT_DATASET_PATH, 'utf8'));"
        "const dataset = validateRefundDataset(value);"
        "process.stdout.write(dataset.payloadSha256);"
    )
    with tempfile.TemporaryDirectory(prefix="semantscript-heldout-validate-") as temporary:
        document_path = Path(temporary) / "dataset.json"
        document_path.write_text(_dump(document), encoding="utf-8")
        completed = subprocess.run(
            ["node", "--input-type=module", "-e", script],
            cwd=_REPOSITORY_ROOT,
            env={**os.environ, "SEMANTSCRIPT_DATASET_PATH": str(document_path)},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    if completed.returncode != 0:
        raise FreezeError(
            "TypeScript dataset validation failed: " + completed.stderr.strip()[-800:]
        )
    if completed.stdout.strip() != document["payloadSha256"]:
        raise FreezeError("TypeScript validation returned a different payload digest")


def _label_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(collections.Counter(row["expected"] for row in rows).items()))


def _fraud_count(rows: Sequence[dict[str, Any]]) -> int:
    return sum(row["inputs"]["order"]["status"] == "fraudulent" for row in rows)


def _dump(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--directory", required=True)
    arguments = parser.parse_args(argv)
    try:
        manifest = freeze_heldout_sets(arguments.directory)
    except Exception as error:
        sys.stderr.write(f"freeze failed: {type(error).__name__}: {error}\n")
        return 1
    sys.stdout.write(
        json.dumps({k: manifest[k] for k in ("release", "final")}, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
