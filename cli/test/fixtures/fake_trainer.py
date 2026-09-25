"""Stand-in for ``python -m semantscript_trainer.cli`` in the CLI tests.

Records the argument vector, writes a report shaped like the real driver's and
exits with ``FAKE_TRAINER_EXIT`` (default 0). ``FAKE_TRAINER_SKIP_REPORT`` leaves
no report behind so the CLI's handling of a silent trainer can be checked.

``doctor`` prints a passing doctor report and records its arguments at
``FAKE_DOCTOR_ARGV_PATH``; ``FAKE_DOCTOR_FAIL=<check id>`` fails that check,
``FAKE_DOCTOR_MALFORMED`` prints text that is not a report and
``FAKE_DOCTOR_BAD_ID`` reports a check the contract does not know.
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
    print(json.dumps({"kind": "semantscript.doctor-report", "reportVersion": 1, "checks": checks}))
    return 1 if failing else 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["doctor"]:
        return doctor(argv[1:])
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
                    "failures": [] if exit_code == 0 else ["injected failure"],
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
    report_path = Path(values["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
