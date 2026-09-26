"""Stand-in for ``python -m semantscript_trainer.cli`` in the CLI tests.

Records the argument vector, writes a report shaped like the real driver's and
exits with ``FAKE_TRAINER_EXIT`` (default 0). ``FAKE_TRAINER_SKIP_REPORT`` leaves
no report behind so the CLI's handling of a silent trainer can be checked.

``doctor`` prints a passing doctor report and records its arguments at
``FAKE_DOCTOR_ARGV_PATH``; ``FAKE_DOCTOR_FAIL=<check id>`` fails that check,
``FAKE_DOCTOR_MALFORMED`` prints text that is not a report and
``FAKE_DOCTOR_BAD_ID`` reports a check the contract does not know and
``FAKE_DOCTOR_RAW_TEACHER`` puts the ``--teacher`` path, unescaped, in the
teacher-config summary (as a non-ASCII checkout or user name would appear).

``train --estimate`` prints an estimate document; ``teacher probe`` prints a
probe result (``FAKE_PROBE_FAIL`` makes it a failed one). ``FAKE_TRAINER_SPEND``
adds a ``teacher.spend`` object to the report. ``FAKE_TRAINER_RETRY`` adds the
seed retry's ``seed``, ``attempts`` and ``retry`` fields: three attempts from
``--seed`` (default 1), the last passing, or with ``FAKE_TRAINER_EXIT`` set, two
failed attempts and a stop reason. A failed report's failure ends with a
``next:`` line, as the real driver's do.

``FAKE_TRAINER_RAISE`` makes ``train`` (and ``train --estimate``) print a progress
line and then fail the way a broken environment does: ``module:<name>`` raises
``ModuleNotFoundError`` for that module, ``syntax`` a ``SyntaxError``, ``crash``
a ``RuntimeError``, all as uncaught tracebacks, and ``launch`` prints the
interpreter's own ``No module named`` line and exits 1. ``signal:<NAME>`` kills
the process with that signal, as the out-of-memory killer or a native crash
does; ``signal:<NAME>:after-warning`` first logs a traceback it goes on from. ``FAKE_TRAINER_NOISE``
prints a traceback a library logged and went on from before the report, and
Python's ``Exception ignored in`` shutdown traceback after it.

``derive-int8`` records its arguments at ``FAKE_TRAINER_ARGV_PATH`` and writes a
``semantscript.derive-report`` to ``--report``: a published one that copies the
source release under a new digest with an int8 encoder entry (so ``releases
promote`` can verify it), or, with ``FAKE_DERIVE_REFUSE``, a refused one with
the gate's figures and exit status 2. ``FAKE_DERIVE_MISSING`` prints the
records-missing error line and exits 1 without a report.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DOCTOR_CHECK_IDS = (
    "python",
    "trainer",
    "model",
    "torch",
    "device",
    "onnxruntime",
    "platform-env",
    "teacher-config",
    "teacher-key",
    "teacher-probe",
)


def doctor(argv: list[str]) -> int:
    record_path = os.environ.get("FAKE_DOCTOR_ARGV_PATH")
    if record_path:
        Path(record_path).write_text(json.dumps({"argv": argv}), encoding="utf-8")
    if os.environ.get("FAKE_DOCTOR_MALFORMED"):
        print("this is not a report")
        return 0
    failing = os.environ.get("FAKE_DOCTOR_FAIL", "")
    checks = [
        {
            "id": identifier,
            "status": "fail" if identifier == failing else "pass",
            "summary": f"fake {identifier}",
            "fix": f"fake fix for {identifier}" if identifier == failing else None,
        }
        for identifier in DOCTOR_CHECK_IDS
    ]
    if os.environ.get("FAKE_DOCTOR_BAD_ID"):
        checks[0]["id"] = "gpu"
    report = {"kind": "semantscript.doctor-report", "reportVersion": 1, "checks": checks}
    if os.environ.get("FAKE_DOCTOR_RAW_TEACHER") and "--teacher" in argv:
        teacher = argv[argv.index("--teacher") + 1]
        checks[DOCTOR_CHECK_IDS.index("teacher-config")]["summary"] = f"teacher {teacher}"
        print(json.dumps(report, ensure_ascii=False))
        return 0
    print(json.dumps(report))
    return 1 if failing else 0


def estimate(values: dict[str, str | bool]) -> int:
    cases = int(values.get("cases", 64))
    row = {
        "id": "nf_" + "3" * 64,
        "sourcePath": "src/app.sem.ts",
        "cached": {"dataset": False, "adversarial": False},
        "plannedRequests": {"synthetic": cases - 1, "boundary": 2, "counterfactual": 10},
        "expectedRequests": 90,
        "maximumRequests": 300,
        "batchRequests": 0,
        "inputTokens": 180000,
        "cacheReadTokens": 150000,
        "outputTokens": 9000,
        "costUsd": 0.2,
        "maximumCostUsd": 0.66,
        "seconds": 360.0,
        "batchSeconds": 0.0,
        "maximumSeconds": 1200.0,
    }
    if os.environ.get("FAKE_ESTIMATE_BATCH"):
        row.update(batchRequests=60, seconds=3720.0, batchSeconds=3600.0, maximumSeconds=350000.0)
    cached = {
        **row,
        "id": "nf_" + "4" * 64,
        "sourcePath": "src/other.sem.ts",
        "cached": {"dataset": True, "adversarial": None},
        "plannedRequests": {"synthetic": 0, "boundary": 0, "counterfactual": 0},
        "expectedRequests": 0,
        "maximumRequests": 0,
        "inputTokens": 0,
        "cacheReadTokens": 0,
        "outputTokens": 0,
        "costUsd": 0.0,
        "maximumCostUsd": 0.0,
        "seconds": 0.0,
        "batchSeconds": 0.0,
        "maximumSeconds": 0.0,
    }
    document = {
        "kind": "semantscript.train-estimate",
        "estimateVersion": 1,
        "teacher": {"backend": "anthropic", "model": "claude-sonnet-5", "fallback": False},
        "price": {
            "model": "claude-sonnet-5",
            "inputUsdPerMillion": 2.0,
            "outputUsdPerMillion": 10.0,
            "cacheReadUsdPerMillion": 0.2,
            "cacheWriteUsdPerMillion": 2.5,
            "source": "pinned Anthropic list price 2026-09-25",
        },
        "secondsPerRequest": 4.0,
        "secondsSource": "pinned default",
        "charactersPerToken": 2.1,
        "cases": cases,
        "functions": [row, cached],
        "total": {key: row[key] for key in row if isinstance(row[key], (int, float))},
    }
    if "max-cost-usd" in values:
        document["maxCostUsd"] = float(values["max-cost-usd"])
    print(json.dumps(document))
    return 0


def teacher_probe(argv: list[str]) -> int:
    record_path = os.environ.get("FAKE_DOCTOR_ARGV_PATH")
    if record_path:
        Path(record_path).write_text(json.dumps({"argv": argv}), encoding="utf-8")
    failed = bool(os.environ.get("FAKE_PROBE_FAIL"))
    print(
        json.dumps(
            {
                "kind": "semantscript.teacher-probe",
                "probeVersion": 1,
                "ok": not failed,
                "backend": "anthropic",
                "model": "anthropic/claude-sonnet-5",
                "baseUrl": "https://openrouter.ai/api",
                "requestSent": True,
                "latencySeconds": 1.25,
                "inputTokens": 16,
                "outputTokens": 4,
                "costUsd": 0.000072,
                "priceSource": "OpenRouter price list (2026-09-25)",
                "summary": "one-request probe failed"
                if failed
                else "one request to anthropic/claude-sonnet-5 answered in 1.2 s",
                "fix": "check the network" if failed else None,
            }
        )
    )
    return 1 if failed else 0


def derive_int8(argv: list[str]) -> int:
    import hashlib
    import shutil

    values: dict[str, str | bool] = {}
    index = 0
    while index < len(argv):
        item = argv[index]
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            values[item[2:]] = argv[index + 1]
            index += 2
            continue
        values[item[2:]] = True
        index += 1
    record_path = os.environ.get("FAKE_TRAINER_ARGV_PATH")
    if record_path:
        Path(record_path).write_text(json.dumps({"argv": argv}), encoding="utf-8")
    if os.environ.get("FAKE_DERIVE_MISSING"):
        print(
            "error: the build cache has no training dataset; next: pass --cache-dir the build "
            "cache the release was trained with",
            file=sys.stderr,
        )
        return 1
    root = Path(str(values["artifact"]))
    source = str(values["release"])
    settings = {
        "method": "dynamic",
        "weightType": values.get("weight-type", "int8"),
        "perChannel": values.get("per-channel", False) is True,
        "reduceRange": values.get("reduce-range", False) is True,
        "argmaxDisagreementTolerance": float(values.get("max-decision-change-rate", 0)),
        "attestedDisagreementTolerance": int(values.get("max-attested-disagreements", 0)),
        "eceThreshold": float(values.get("ece-threshold", 0.1)),
        "sourceManifestSha256": source,
    }
    refused = bool(os.environ.get("FAKE_DERIVE_REFUSE"))
    function = {
        "id": "nf_" + "1" * 64,
        "recordsChecked": 10,
        "labeledRecords": 10,
        "attestedRecords": 2,
        "argmaxDisagreements": 1 if refused else 0,
        "attestedDisagreements": 1 if refused else 0,
        "sourceAccuracy": 1.0,
        "quantizedAccuracy": 0.9 if refused else 1.0,
        "sourceEce": 0.02,
        "quantizedEce": 0.03,
    }
    quantization = {
        "recordsChecked": 10,
        "attestedRecords": 2,
        "argmaxDisagreements": function["argmaxDisagreements"],
        "attestedDisagreements": function["attestedDisagreements"],
        "argmaxDisagreementRate": 0.1 if refused else 0.0,
        "sourceEce": 0.02,
        "quantizedEce": 0.03,
        "sourceEncoderByteLength": 4000,
        "quantizedEncoderByteLength": 1000,
        "functions": [function],
    }
    report: dict[str, object] = {
        "kind": "semantscript.derive-report",
        "reportVersion": 1,
        "precision": "int8-dynamic",
        "artifactRoot": str(root),
        "source": {
            "manifestSha256": source,
            "release": f"releases/sha256-{source}",
            "bytes": 5000,
        },
        "settings": settings,
        "recordSources": [
            {
                "kind": "training-dataset",
                "functionId": function["id"],
                "sha256": "d" * 64,
                "records": 10,
                "path": "/cache/datasets/v1/dd/x.json",
            }
        ],
        "heldOut": {"records": 0, "note": "no held-out set is recorded for this release"},
        "quantization": quantization,
    }
    status = 0
    if refused:
        report.update(
            status="refused",
            derived=None,
            failures=["1 attested record(s) changed decision (tolerance 0)"],
            next="keep serving the float32 release (current.json is unchanged), or try "
            "semantscript releases derive --int8 --per-channel",
        )
        status = 2
    else:
        source_release = root / "releases" / f"sha256-{source}"
        manifest = json.loads((source_release / "manifest.json").read_text(encoding="utf-8"))
        manifest["build"]["createdAt"] = "2099-01-01T00:00:00Z"
        for resource in manifest["resources"]:
            if resource["role"] == "encoder":
                resource["onnx"]["precision"] = "int8-dynamic"
                resource["onnx"]["quantization"] = {
                    **settings,
                    "verification": {
                        "recordsChecked": 10,
                        "attestedRecords": 2,
                        "decisionChanges": 0,
                        "attestedDecisionChanges": 0,
                        "decisionChangeRate": 0.0,
                        "sourceEce": 0.02,
                        "quantizedEce": 0.03,
                        "recordSources": [
                            {
                                "kind": "training-dataset",
                                "functionId": function["id"],
                                "sha256": "d" * 64,
                                "records": 10,
                            }
                        ],
                        "functions": [],
                    },
                }
        data = json.dumps(manifest, indent=2).encode()
        digest = hashlib.sha256(data).hexdigest()
        target = root / "releases" / f"sha256-{digest}"
        shutil.copytree(source_release, target)
        (target / "manifest.json").write_bytes(data)
        report.update(
            status="published",
            derived={
                "manifestSha256": digest,
                "release": f"releases/sha256-{digest}",
                "bytes": 2000,
            },
            failures=[],
            next=f"semantscript releases promote {digest[:12]}",
        )
    Path(str(values["report"])).write_text(json.dumps(report), encoding="utf-8")
    print(json.dumps(report))
    return status


def main(argv: list[str]) -> int:
    if argv[:1] == ["doctor"]:
        return doctor(argv[1:])
    if argv[:1] == ["derive-int8"]:
        return derive_int8(argv[1:])
    if argv[:2] == ["teacher", "probe"]:
        return teacher_probe(argv[2:])
    values: dict[str, str | bool] = {}
    index = 0
    while index < len(argv):
        item = argv[index]
        if item.startswith("--"):
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                values[item[2:]] = argv[index + 1]
                index += 2
                continue
            values[item[2:]] = True
        index += 1
    record_path = os.environ.get("FAKE_TRAINER_ARGV_PATH")
    if record_path:
        Path(record_path).write_text(
            json.dumps({"argv": argv, "pythonpath": os.environ.get("PYTHONPATH", "")}),
            encoding="utf-8",
        )
    failure = os.environ.get("FAKE_TRAINER_RAISE")
    if failure:
        print("generating 64 cases (1 gold)", file=sys.stderr, flush=True)
        if failure == "launch":
            print(f"{sys.executable}: No module named semantscript_trainer.cli", file=sys.stderr)
            return 1
        if failure.startswith("module:"):
            name = failure.split(":", 1)[1]
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        if failure == "syntax":
            raise SyntaxError("invalid syntax")
        if failure.startswith("signal:"):
            import signal
            import traceback

            if failure.endswith(":after-warning"):
                try:
                    raise ValueError("optional backend unavailable")
                except ValueError:
                    traceback.print_exc()
                print("continuing without it", file=sys.stderr, flush=True)
            name = failure.split(":")[1]
            os.kill(os.getpid(), getattr(signal, name))
        raise RuntimeError("the fake trainer crashed")
    if values.get("estimate") is True:
        return estimate(values)
    exit_code = int(os.environ.get("FAKE_TRAINER_EXIT", "0"))
    if os.environ.get("FAKE_TRAINER_SKIP_REPORT"):
        return exit_code
    report = {
        "kind": "semantscript.train-report",
        "reportVersion": 1,
        "status": "passed" if exit_code == 0 else "failed",
        "trainedAt": "2026-09-24T00:00:00Z",
        "application": {"id": values.get("application-id", "demo"), "version": "0.0.0"},
        "teacher": {"provider": "fake", "model": "fake", "configurationSha256": "0" * 64},
        "trainingKeySha256": "1" * 64,
        "cache": {
            "reused": 0 if "no-cache" in values else 1,
            "trained": 1,
            "directory": None if "no-cache" in values else str(values["cache-dir"]),
        },
        "artifact": None
        if exit_code
        else {
            "root": str(values["artifact"]),
            "releaseDirectory": str(values["artifact"]) + "/releases/sha256-" + "2" * 64,
            "manifestSha256": "2" * 64,
        },
        "functions": [
            {
                "id": "nf_" + "3" * 64,
                "semanticSha256": "4" * 64,
                "sourcePath": "src/app.sem.ts",
                "cache": "trained",
                "dataset": {"sha256": "5" * 64, "cases": int(values.get("cases", 64)), "gold": 1},
                "adversarial": {"sha256": "6" * 64, "cases": 4},
                "training": {
                    "trainingRows": 60,
                    "heldOutRows": 8,
                    "epochs": 3,
                    "selectedEpoch": None,
                    "heldOutAccuracy": 0.875,
                    "heldOutFieldAccuracy": None,
                },
                "verification": {
                    "status": "passed" if exit_code == 0 else "failed",
                    "failures": []
                    if exit_code == 0
                    else ["injected failure\n  next: rerun with --epochs 5 (now 3)"],
                    "suggestions": [] if exit_code == 0 else ["rerun with --epochs 5 (now 3)"],
                    "attestedCases": 1,
                    "pairCount": 2,
                    "metrics": {
                        "accuracy": 0.9,
                        "ece": 0.05,
                        "brier": 0.1,
                        "pairConsistency": 1.0,
                        "heads": [],
                        "exampleFailures": 0,
                        "constraintViolations": 0,
                        "typeErrors": 0,
                    },
                },
            }
        ],
    }
    if os.environ.get("FAKE_TRAINER_SPEND"):
        report["teacher"]["spend"] = {
            "requests": 40,
            "replayed": 12,
            "inputTokens": 90000,
            "cacheReadTokens": 70000,
            "cacheWriteTokens": 2300,
            "outputTokens": 2700,
            "costUsd": 0.0931,
            "maxCostUsd": float(values["max-cost-usd"]) if "max-cost-usd" in values else None,
            "priceSource": "pinned Anthropic list price 2026-09-25",
            "seconds": 150.0,
        }
    if os.environ.get("FAKE_TRAINER_RETRY"):
        first = int(values.get("seed", 1))
        count = 3 if exit_code == 0 else 2

        def attempt(index: int) -> dict[str, object]:
            passed = exit_code == 0 and index == count - 1
            violations = 2 if passed else (4 + 4 * index if exit_code else 4 + index)
            return {
                "attempt": index + 1,
                "seed": first + index,
                "status": "passed" if passed else "failed",
                "functions": [
                    {
                        "id": "nf_" + "3" * 64,
                        "status": "passed" if passed else "failed",
                        "accuracy": 0.97,
                        "ece": 0.04,
                        "constraintViolations": violations,
                        "records": 394,
                        "violationRate": violations / 394,
                        "failures": [] if passed else ["violation rate over tolerance"],
                    }
                ],
            }

        report["seed"] = first + count - 1 if exit_code == 0 else None
        report["attempts"] = [attempt(index) for index in range(count)]
        report["retry"] = {
            "attempts": int(values.get("seed-attempts", 3)),
            "margin": float(values.get("seed-retry-margin", 2)),
            "stopReason": None
            if exit_code == 0
            else "nf_" + "3" * 64 + ": violation rate 2.0305% (8 of 394) is outside the "
            "retry margin 2 x 0.01 = 2.0000%",
        }
    noisy = bool(os.environ.get("FAKE_TRAINER_NOISE"))
    if noisy:
        # A library logging a handled exception with its traceback, then going on.
        print(
            'Traceback (most recent call last):\n  File "lib.py", line 3, in load\n'
            "ValueError: optional backend unavailable\nprogress: continuing",
            file=sys.stderr,
            flush=True,
        )
    report_path = Path(values["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if noisy:
        # Python's shutdown noise: a traceback after the report, as the process ends.
        print(
            "Exception ignored in: <function Handle.__del__ at 0x1>\n"
            'Traceback (most recent call last):\n  File "h.py", line 9, in __del__\n'
            "OSError: handle closed",
            file=sys.stderr,
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
