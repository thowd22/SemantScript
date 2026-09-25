"""Run Jev (TypeSafe's decision model on OpenRouter) on the frozen refund final set.

A diagnostic comparator for TASK-5.17, not a pinned benchmark system: every
case of the final set is sent to ``POST /api/alpha/decisions`` as one choice
question over the same policy text, hard constraints and canonical inputs the
generative baselines receive, sequentially, and the answers are scored with
the benchmark's own definitions (accuracy on all and on the attested slice,
15-bin ECE over top-1 probability, nearest-rank p50/p95 latency). Agreement is
also reported per adjudication step, and the measured usage gives the cost of
a 10k-case typed corpus.

The OpenRouter key is read from ``OPENROUTER_API_KEY`` and never written
anywhere. Raw answers are cached under the output directory so a rerun
re-spends nothing; ``--max-cost-usd`` aborts before any request that would
exceed the budget.

Usage::

    set -a; . ./.env; set +a
    PYTHONPATH=.:trainer/src:model/src:.python-packages python3 -m \\
        benchmarks.refund.program.run_jev_comparator \\
        --dataset benchmarks/refund/data/heldout-uci-2026-09-23/final-benchmark-dataset.json \\
        --adjudications benchmarks/refund/data/heldout-uci-2026-09-23/adjudications.json \\
        --output-dir benchmarks/refund/data/results-jev-2026-09-25
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL_ALIAS = "~typesafe/jev-latest"
ECE_BIN_COUNT = 15
SUPPORT = ("approve", "deny", "review")
PRICE_PER_INPUT_TOKEN_USD = 0.042 / 1_000_000  # published on the model page, output free
MAXIMUM_RESPONSE_BYTES = 65_536
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 520, 521, 522, 523, 524})

CRITERIA = {
    "approve": "The refund is approved and paid out",
    "deny": "The refund is denied",
    "review": "The request is held for manual review",
}


class JevComparatorError(RuntimeError):
    """The comparator run could not complete."""


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_policy() -> dict[str, Any]:
    """The policy text, hard constraints and task digest from the built benchmark package."""

    script = (
        "import { REFUND_BASELINE_POLICY, REFUND_TASK_SPEC_SHA256 } from "
        "'./benchmarks/refund/dist/policy.js'; "
        "console.log(JSON.stringify({ policy: REFUND_BASELINE_POLICY.policy, "
        "hardConstraints: REFUND_BASELINE_POLICY.hardConstraints, "
        "taskSpecSha256: REFUND_TASK_SPEC_SHA256 }))"
    )
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise JevComparatorError(f"could not read the benchmark policy: {completed.stderr[:300]}")
    return json.loads(completed.stdout)


def build_request(policy: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    """One choice question per case over the same facts the baselines see."""

    return {
        "model": MODEL_ALIAS,
        "state": {
            "policy": policy["policy"],
            "hardConstraints": list(policy["hardConstraints"]),
            "inputDomain": 'order.status is exactly "paid" or "fraudulent"',
            "customer": {
                "priorRefunds": inputs["customer"]["priorRefunds"],
                "tier": inputs["customer"]["tier"],
            },
            "order": {
                "ageDays": inputs["order"]["ageDays"],
                "status": inputs["order"]["status"],
                "total": inputs["order"]["total"],
            },
        },
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": (
                    "Apply the complete refund policy and hard constraints in the state to the "
                    "customer and order records. Which decision does the policy give?"
                ),
                "criteria": dict(CRITERIA),
            }
        },
    }


Transport = Callable[[dict[str, Any]], tuple[dict[str, Any], float]]


def http_transport(api_key: str, timeout_seconds: float = 120.0) -> Transport:
    def send(request: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
        body = json.dumps(request, separators=(",", ":")).encode("utf-8")
        last_error: str | None = None
        for attempt in range(5):
            http_request = urllib.request.Request(
                DECISIONS_URL,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/semantscript",
                    "X-Title": "SemantScript refund comparator",
                },
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(http_request, timeout=timeout_seconds) as response:
                    raw = response.read(MAXIMUM_RESPONSE_BYTES + 1)
                elapsed_ms = (time.perf_counter() - started) * 1000
            except urllib.error.HTTPError as error:
                detail = error.read(2048).decode("utf-8", "replace")
                if error.code in RETRY_STATUSES and attempt < 4:
                    last_error = f"HTTP {error.code}: {detail[:200]}"
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise JevComparatorError(
                    f"decisions request failed: HTTP {error.code}: {detail[:400]}"
                ) from None
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if attempt < 4:
                    last_error = str(error)
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise JevComparatorError(f"decisions request failed: {error}") from None
            if len(raw) > MAXIMUM_RESPONSE_BYTES:
                raise JevComparatorError("decisions response exceeded the byte limit")
            return json.loads(raw.decode("utf-8")), elapsed_ms
        raise JevComparatorError(f"decisions request failed after retries: {last_error}")

    return send


def parse_answer(
    response: Mapping[str, Any],
) -> tuple[str, dict[str, float], float, dict[str, Any]]:
    answers = response.get("answers")
    if not isinstance(answers, Mapping) or not isinstance(answers.get("decision"), Mapping):
        raise JevComparatorError(
            f"decisions response has no decision answer: {json.dumps(response)[:300]}"
        )
    answer = answers["decision"]
    choice = answer.get("choice")
    probabilities = answer.get("probabilities")
    if choice not in SUPPORT or not isinstance(probabilities, Mapping):
        raise JevComparatorError(f"decisions answer is malformed: {json.dumps(answer)[:300]}")
    distribution = {value: float(probabilities.get(value, 0.0)) for value in SUPPORT}
    usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
    return (
        str(choice),
        distribution,
        float(answer.get("confidence", max(distribution.values()))),
        dict(usage),
    )


def nearest_rank(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, math.ceil(quantile * len(sorted_values)) - 1))
    return sorted_values[index]


def score(
    cases: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The benchmark's metric definitions (metrics.ts) applied to a prediction list."""

    by_id = {case["id"]: case for case in cases}
    counts = [0] * ECE_BIN_COUNT
    correct_in_bin = [0] * ECE_BIN_COUNT
    confidence_in_bin = [0.0] * ECE_BIN_COUNT
    correct = 0
    brier = 0.0
    for prediction in predictions:
        case = by_id[prediction["caseId"]]
        hit = prediction["value"] == case["expected"]
        correct += int(hit)
        probabilities = {
            entry["value"]: entry["probability"] for entry in prediction["distribution"]
        }
        confidence = max(probabilities.values())
        bucket = min(ECE_BIN_COUNT - 1, math.floor(confidence * ECE_BIN_COUNT))
        counts[bucket] += 1
        correct_in_bin[bucket] += int(hit)
        confidence_in_bin[bucket] += confidence
        brier += sum(
            (probabilities[v] - (1.0 if v == case["expected"] else 0.0)) ** 2 for v in SUPPORT
        )
    ece = 0.0
    for bucket in range(ECE_BIN_COUNT):
        if counts[bucket]:
            ece += (counts[bucket] / len(predictions)) * abs(
                correct_in_bin[bucket] / counts[bucket] - confidence_in_bin[bucket] / counts[bucket]
            )
    latencies = sorted(p["latencyMs"] for p in predictions)
    return {
        "accuracy": {
            "caseCount": len(predictions),
            "correctCount": correct,
            "accuracy": correct / len(predictions),
            "attestedCaseCount": len(predictions),
            "attestedCorrectCount": correct,
            "attestedAccuracy": correct / len(predictions),
        },
        "calibration": {"binCount": ECE_BIN_COUNT, "expectedCalibrationError": ece},
        "brier": brier / len(predictions),
        "latency": {"p50Ms": nearest_rank(latencies, 0.5), "p95Ms": nearest_rank(latencies, 0.95)},
    }


def agreement_by_step(
    cases: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    adjudications: Mapping[str, Any],
) -> dict[str, Any]:
    by_id = {case["id"]: case for case in cases}
    steps: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        record = adjudications["cases"].get(prediction["caseId"], {})
        step = str(record.get("step", "?"))
        entry = steps.setdefault(step, {"cases": 0, "agree": 0, "confusions": {}})
        entry["cases"] += 1
        expected = by_id[prediction["caseId"]]["expected"]
        if prediction["value"] == expected:
            entry["agree"] += 1
        else:
            key = f"{expected}->{prediction['value']}"
            entry["confusions"][key] = entry["confusions"].get(key, 0) + 1
    for entry in steps.values():
        entry["agreement"] = round(entry["agree"] / entry["cases"], 4)
    return dict(sorted(steps.items()))


STEP_NAMES = {
    "1": "stale order (ageDays > 90): deny",
    "2": "fraudulent within 90 days: review",
    "3": "paid, outside the tier window: deny",
    "4": "paid, inside the window, suspicious history: review",
    "5": "paid, inside the window, otherwise: approve",
}


def run(
    *,
    dataset: Mapping[str, Any],
    adjudications: Mapping[str, Any],
    policy: Mapping[str, Any],
    transport: Transport,
    output_dir: Path,
    max_cost_usd: float,
    log: Callable[[str], None],
) -> dict[str, Any]:
    cases = list(dataset["cases"])
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw-answers.json"
    raw: dict[str, Any] = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    started_at = utc_now()
    run_started = time.perf_counter()
    spent = sum(float(entry.get("usage", {}).get("cost", 0.0)) for entry in raw.values())
    models: set[str] = set()
    for index, case in enumerate(cases):
        if case["id"] in raw:
            continue
        if spent > max_cost_usd:
            raise JevComparatorError(f"spend {spent:.4f} USD exceeds --max-cost-usd {max_cost_usd}")
        response, elapsed_ms = transport(build_request(policy, case["inputs"]))
        raw[case["id"]] = {
            "response": response,
            "latencyMs": elapsed_ms,
            "usage": response.get("usage", {}),
        }
        spent += float(response.get("usage", {}).get("cost", 0.0))
        if (index + 1) % 20 == 0 or index + 1 == len(cases):
            log(f"{index + 1}/{len(cases)} cases, {spent:.5f} USD so far")
            raw_path.write_text(json.dumps(raw, indent=1, sort_keys=True) + "\n")
    raw_path.write_text(json.dumps(raw, indent=1, sort_keys=True) + "\n")
    measured_ms = (time.perf_counter() - run_started) * 1000

    predictions = []
    input_tokens = 0
    output_tokens = 0
    for case in cases:
        entry = raw[case["id"]]
        choice, distribution, confidence, usage = parse_answer(entry["response"])
        models.add(str(entry["response"].get("model", "")))
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))
        predictions.append(
            {
                "caseId": case["id"],
                "inputSha256": case["inputSha256"],
                "value": choice,
                "confidence": confidence,
                "distribution": [{"value": v, "probability": distribution[v]} for v in SUPPORT],
                "latencyMs": entry["latencyMs"],
            }
        )
    metrics = score(cases, predictions)
    total_cost = sum(float(entry["usage"].get("cost", 0.0)) for entry in raw.values())
    per_case_tokens = input_tokens / len(cases)
    results = {
        "kind": "semantscript.refund-benchmark-diagnostic-comparator",
        "comparator": "jev",
        "startedAt": started_at,
        "finishedAt": utc_now(),
        "model": {
            "provider": "openrouter/typesafe",
            "alias": MODEL_ALIAS,
            "resolved": sorted(models),
            "endpoint": DECISIONS_URL,
        },
        "protocol": {
            "concurrency": 1,
            "questionsPerCase": 1,
            "questionType": "choice",
            "criteria": CRITERIA,
            "taskSpecSha256": policy["taskSpecSha256"],
            "policyText": policy["policy"],
            "hardConstraints": list(policy["hardConstraints"]),
            "measuredDurationMs": measured_ms,
            "note": "diagnostic comparator; not a pinned benchmark system role, so not a sealed prediction set",
        },
        "dataset": {
            "path": str(dataset.get("_path", "")),
            "payloadSha256": dataset.get("payloadSha256"),
            "caseCount": len(cases),
        },
        "metrics": metrics,
        "agreementByStep": {
            step: {"rule": STEP_NAMES.get(step, step), **entry}
            for step, entry in agreement_by_step(cases, predictions, adjudications).items()
        },
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "costUsd": total_cost,
            "costPerCaseUsd": total_cost / len(cases),
            "inputTokensPerCase": per_case_tokens,
            "publishedInputPriceUsdPerMillion": PRICE_PER_INPUT_TOKEN_USD * 1_000_000,
            "estimatedCostFor10kCasesUsd": total_cost / len(cases) * 10_000,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "client": "urllib over HTTPS from the development machine (WSL2); latency includes the network",
        },
        "predictions": predictions,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cost-usd", type=float, default=0.25)
    arguments = parser.parse_args(argv)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2
    dataset = json.loads(arguments.dataset.read_text())
    dataset["_path"] = str(arguments.dataset)
    adjudications = json.loads(arguments.adjudications.read_text())
    policy = load_policy()
    results = run(
        dataset=dataset,
        adjudications=adjudications,
        policy=policy,
        transport=http_transport(api_key),
        output_dir=arguments.output_dir,
        max_cost_usd=arguments.max_cost_usd,
        log=lambda message: print(message, flush=True),
    )
    m = results["metrics"]
    print(
        f"accuracy {m['accuracy']['accuracy']:.4f} ({m['accuracy']['correctCount']}/{m['accuracy']['caseCount']}), "
        f"ECE {m['calibration']['expectedCalibrationError']:.4f}, Brier {results['metrics']['brier']:.4f}, "
        f"p50 {m['latency']['p50Ms']:.1f} ms, p95 {m['latency']['p95Ms']:.1f} ms, "
        f"cost {results['usage']['costUsd']:.5f} USD"
    )
    for step, entry in results["agreementByStep"].items():
        print(
            f"  step {step} {entry['rule']}: {entry['agree']}/{entry['cases']} {entry['confusions']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
