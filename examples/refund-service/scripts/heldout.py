"""Write the held-out set `npm run measure` scores: fresh inputs labelled by the constraints.

The trainer's built-in constraints teacher (``semantscript train --teacher
constraints``) samples the training corpus from the ``train`` stream of its seed;
this script draws each expression's held-out inputs from the separate ``heldout``
stream of the same teacher configuration and, to be sure, drops any input that
appears in a cached training or adversarial dataset of that expression. The
labels are the constraints' own (every expression here has complete
constraints), so the set measures how well the model learned the policy on
inputs it never saw.

Usage (from examples/refund-service, after ``npm run train``)::

    python3 scripts/heldout.py [--cases 200] [--teacher constraints]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from semantscript_trainer.teacher_config import create_teacher, load_teacher_config
from semantscript_trainer.teachers import ConstraintsTeacher

HERE = Path(__file__).resolve().parent.parent
BUNDLE = HERE / "dist" / "semantscript.ir.v1.json"
CACHE = HERE / ".semantscript" / "cache"
HELDOUT = HERE / ".semantscript" / "heldout.json"
TEACHER_FILE = HERE / "semantscript.teacher.toml"


def trained_inputs(cache: Path, function_id: str) -> Iterator[Mapping[str, Any]]:
    """Every input a cached dataset or adversarial sidecar of ``function_id`` holds."""

    for kind in ("datasets", "adversarial-datasets"):
        for path in sorted((cache / kind).rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
            except (OSError, ValueError, KeyError, TypeError):
                continue
            function = payload.get("function") if isinstance(payload, dict) else None
            if not isinstance(function, dict) or function.get("id") != function_id:
                continue
            for case in payload.get("cases", []):
                if isinstance(case, dict) and isinstance(case.get("inputs"), dict):
                    yield case["inputs"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", type=int, default=200, help="held-out cases per expression")
    parser.add_argument(
        "--teacher",
        default=str(TEACHER_FILE) if TEACHER_FILE.exists() else "constraints",
        help="the teacher train used (the constraints keyword or a teacher TOML)",
    )
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=HELDOUT)
    arguments = parser.parse_args(argv)

    teacher = create_teacher(load_teacher_config(arguments.teacher))
    if not isinstance(teacher, ConstraintsTeacher):
        print("the held-out labels come from the constraints teacher", file=sys.stderr)
        return 2
    bundle = json.loads(arguments.bundle.read_text(encoding="utf-8"))
    functions = []
    for ir in bundle["functions"]:
        exclude = list(trained_inputs(arguments.cache_dir, ir["id"]))
        cases = teacher.sample_decided(ir, arguments.cases, stream="heldout", exclude=exclude)
        functions.append(
            {
                "id": ir["id"],
                "name": ir["definition"].get("name") or ir["output"]["tsType"],
                "adapterRef": ir["model"]["adapter"],
                "support": ir["output"]["head"]["support"],
                "excludedTrainingInputs": len(exclude),
                "cases": [{"inputs": case.inputs, "expected": case.output} for case in cases],
            }
        )
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(
            {
                "kind": "refund-service-heldout",
                "teacher": {
                    "provider": teacher.descriptor.provider,
                    "model": teacher.descriptor.model,
                    "configurationSha256": teacher.descriptor.configuration_sha256,
                },
                "functions": functions,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    total = sum(len(entry["cases"]) for entry in functions)
    print(f"wrote {arguments.out} ({total} cases over {len(functions)} expressions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
