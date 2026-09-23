"""Select disjoint release and final held-out samples from UCI refund candidates.

Selection is deterministic: candidates are ordered by their salted key digest,
stratified by order-age bucket and customer tier so every policy boundary
(30, 60 and 90 days) is represented, and split between the release-verification
set and the final evaluation set. A hash-selected share of selected rows has
``order.status`` set to ``fraudulent`` so the never-approve constraint is
exercised; the change is recorded per case. Rows whose inputs already appear in
the training corpus, or that duplicate an earlier selection, are skipped.

Usage::

    PYTHONPATH=.:trainer/src:model/src:.python-packages python -m \\
        benchmarks.refund.program.sample_uci_heldout --candidates candidates.json \\
        --training-manifest benchmarks/refund/data/sonnet-pilot-2026-09-23/manifest.json \\
        --output-dir benchmarks/refund/data/heldout-uci-2026-09-23
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semantscript_trainer.semantic_json import semantic_json_sha256

SAMPLING_VERSION = 1
FRAUD_PERCENT = 15
AGE_BUCKETS = (("0-30", 0, 30), ("31-60", 31, 60), ("61-90", 61, 90), ("91+", 91, None))
TIERS = ("enterprise", "standard")
# Per-set quotas by age bucket; each bucket is split as evenly as availability allows
# across the two tiers.
QUOTAS = {
    "release": {"0-30": 32, "31-60": 20, "61-90": 14, "91+": 14},
    "final": {"0-30": 64, "31-60": 40, "61-90": 28, "91+": 28},
}


def bucket_of(age_days: float) -> str:
    for name, low, high in AGE_BUCKETS:
        if age_days >= low and (high is None or age_days <= high):
            return name
    raise ValueError(f"age {age_days} has no bucket")


def training_input_digests(manifest_path: Path) -> set[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    digests: set[str] = set()
    for section in ("synthetic", "adversarial"):
        document = json.loads((root / manifest[section]["cachePath"]).read_text(encoding="utf-8"))
        for case in document["payload"]["cases"]:
            digests.add(semantic_json_sha256(case["inputs"]))
    return digests


def select_samples(
    candidates: Sequence[dict[str, Any]],
    excluded_digests: set[str],
) -> dict[str, Any]:
    ordered = sorted(candidates, key=lambda row: row["key"])
    pools: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in ordered:
        inputs = row["inputs"]
        pools[(bucket_of(inputs["order"]["ageDays"]), inputs["customer"]["tier"])].append(row)

    taken_digests: set[str] = set(excluded_digests)
    selections: dict[str, list[dict[str, Any]]] = {"release": [], "final": []}
    skipped = {"trainingOverlap": 0, "duplicateInputs": 0}
    cursors: dict[tuple[str, str], int] = collections.defaultdict(int)

    def take(split: str, bucket: str, tier: str) -> bool:
        pool = pools[(bucket, tier)]
        while cursors[(bucket, tier)] < len(pool):
            row = pool[cursors[(bucket, tier)]]
            cursors[(bucket, tier)] += 1
            inputs = copy.deepcopy(row["inputs"])
            fraud = int(row["key"][:8], 16) % 100 < FRAUD_PERCENT
            if fraud:
                inputs["order"]["status"] = "fraudulent"
            digest = semantic_json_sha256(inputs)
            if digest in excluded_digests:
                skipped["trainingOverlap"] += 1
                continue
            if digest in taken_digests:
                skipped["duplicateInputs"] += 1
                continue
            taken_digests.add(digest)
            selections[split].append(
                {
                    "id": f"uci-{split[0]}-{row['key']}",
                    "inputs": inputs,
                    "inputSha256": digest,
                    "derivation": {
                        "sourceKey": row["key"],
                        "refundedAmount": row["refundedAmount"],
                        "statusSetToFraudulentByJudge": fraud,
                    },
                }
            )
            return True
        return False

    for split in ("release", "final"):
        for bucket, quota in QUOTAS[split].items():
            for index in range(quota):
                tier = TIERS[index % len(TIERS)]
                if not take(split, bucket, tier) and not take(
                    split, bucket, TIERS[(index + 1) % 2]
                ):
                    raise RuntimeError(f"not enough candidates for {split} {bucket}")
    for split in selections:
        selections[split].sort(key=lambda case: case["id"].encode("utf-8"))
    return {
        "kind": "semantscript.refund-benchmark-uci-heldout-sample",
        "samplingVersion": SAMPLING_VERSION,
        "fraudPercent": FRAUD_PERCENT,
        "quotas": QUOTAS,
        "excludedTrainingInputCount": len(excluded_digests),
        "skipped": skipped,
        "release": selections["release"],
        "final": selections["final"],
    }


def render_sheet(cases: Sequence[dict[str, Any]]) -> str:
    lines = ["id | tier | priorRefunds | ageDays | status | total"]
    for case in cases:
        customer = case["inputs"]["customer"]
        order = case["inputs"]["order"]
        lines.append(
            f"{case['id']} | {customer['tier']} | {customer['priorRefunds']} | "
            f"{order['ageDays']} | {order['status']} | {order['total']}"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args(argv)
    candidates_document = json.loads(Path(arguments.candidates).read_text(encoding="utf-8"))
    excluded = training_input_digests(Path(arguments.training_manifest))
    sample = select_samples(candidates_document["candidates"], excluded)
    sample["source"] = candidates_document["source"]
    sample["derivation"] = candidates_document["derivation"]
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "sample.json").write_text(
        json.dumps(sample, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    for split in ("release", "final"):
        (output / f"sheet-{split}.txt").write_text(render_sheet(sample[split]), encoding="utf-8")
    summary = {
        split: {
            "count": len(sample[split]),
            "fraudulent": sum(
                c["inputs"]["order"]["status"] == "fraudulent" for c in sample[split]
            ),
            "tiers": dict(
                collections.Counter(c["inputs"]["customer"]["tier"] for c in sample[split])
            ),
            "ageBuckets": dict(
                collections.Counter(
                    bucket_of(c["inputs"]["order"]["ageDays"]) for c in sample[split]
                )
            ),
        }
        for split in ("release", "final")
    }
    summary["skipped"] = sample["skipped"]
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
