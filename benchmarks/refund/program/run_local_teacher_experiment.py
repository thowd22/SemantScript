"""Local teacher (Qwen3-14B) against the Sonnet 5 reference on the refund task (TASK-5.13).

The same real refund inputs are labeled by ``claude-sonnet-5`` (through
OpenRouter's Anthropic-format route, JSON-schema structured output) and by
``qwen3:14b`` (Ollama, thinking off, temperature 0), with the policy text and
hard constraints the benchmark baselines receive. Label agreement is reported
between the two and against the compiled constraints' unique admissible label,
per rubric step. A Sonnet-labeled evaluation set is frozen and never trained
on. Two ModernBERT-base students are then trained with the committed compact
recipe on the same training inputs, one per label set, and scored on the
evaluation set (accuracy, 15-bin ECE, Brier) and, as a diagnostic, on the
judge-attested final set.

Every labeler call is cached per input under the output directory, so a rerun
re-spends nothing; ``--sonnet-max-cost-usd`` aborts before a request that
would exceed the budget. Keys are read from the environment and never written.

Usage::

    set -a; . ./.env; set +a
    PYTHONNOUSERSITE=1 HSA_ENABLE_DXG_DETECTION=1 \\
    PYTHONPATH=.:trainer/src:model/src:.python-packages python3 -m \\
        benchmarks.refund.program.run_local_teacher_experiment \\
        --output-dir benchmarks/refund/data/results-local-teacher-2026-09-25 \\
        --stage sample|label-qwen|label-sonnet|train|all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.refund.program.pipeline import compile_refund_program

from semantscript_trainer.canonical_input import serialize_canonical_inputs
from semantscript_trainer.constraints import compile_constraints
from semantscript_trainer.dataset import SyntheticDatasetGenerator
from semantscript_trainer.semantic_json import semantic_json_sha256
from semantscript_trainer.teacher import GeneratedCase, TeacherDescriptor
from semantscript_trainer.training import (
    DEFAULT_ENCODER_NAME,
    DEFAULT_ENCODER_REVISION,
    TrainingConfig,
    train_classifier,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA = _REPOSITORY_ROOT / "benchmarks" / "refund" / "data"
POOL = DATA / "uci-pool" / "candidates.json"
HELDOUT = DATA / "heldout-uci-2026-09-23"
SUPPORT = ("approve", "deny", "review")
SONNET_MODEL = "claude-sonnet-5"
SONNET_ROUTE = "https://openrouter.ai/api/v1/messages"
QWEN_MODEL = "qwen3:14b"
OLLAMA_CHAT = "http://127.0.0.1:11434/api/chat"
ECE_BIN_COUNT = 15
DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision"],
    "properties": {"decision": {"type": "string", "enum": list(SUPPORT)}},
}
STEP_NAMES = {
    "1": "stale order (ageDays > 90): deny",
    "2": "fraudulent within 90 days: review",
    "3": "paid, outside the tier window: deny",
    "4": "paid, inside the window, suspicious history: review",
    "5": "paid, inside the window, otherwise: approve",
}


class ExperimentError(RuntimeError):
    """The experiment could not complete."""


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(message: str) -> None:
    print(f"{utc_now()} {message}", flush=True)


# ---- policy and prompt (the benchmark baselines' text) --------------------------


def load_policy() -> dict[str, Any]:
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
        raise ExperimentError(f"could not read the benchmark policy: {completed.stderr[:300]}")
    return json.loads(completed.stdout)


def canonical_input(inputs: Mapping[str, Any]) -> str:
    return json.dumps(
        {
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
        separators=(",", ":"),
    )


def build_prompt(policy: Mapping[str, Any], inputs: Mapping[str, Any]) -> tuple[str, str]:
    system = (
        "Apply the complete refund policy and hard constraints supplied by the user. "
        "Return only the required structured decision."
    )
    user = "\n".join(
        [
            f"Task specification SHA-256: {policy['taskSpecSha256']}",
            f"Policy: {policy['policy']}.",
            "Hard constraints:",
            *(f"{index + 1}. {rule}" for index, rule in enumerate(policy["hardConstraints"])),
            'Input domain: order.status is exactly "paid" or "fraudulent".',
            f"Support order: {', '.join(SUPPORT)}",
            f"Refund input (canonical refund JSON): {canonical_input(inputs)}",
        ]
    )
    return system, user


# ---- inputs -----------------------------------------------------------------------


def held_out_digests() -> frozenset[str]:
    digests: set[str] = set()
    for name in ("final-benchmark-dataset.json", "release-verification.json"):
        document = json.loads((HELDOUT / name).read_text())
        cases = document.get("cases") or document.get("payload", {}).get("cases") or []
        for case in cases:
            digests.add(case["inputSha256"])
    return frozenset(digests)


def sample_inputs(
    candidates: Sequence[Mapping[str, Any]],
    excluded: frozenset[str],
    *,
    evaluation: int,
    training: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Distinct real inputs with every held-out digest removed, split by a seeded shuffle."""

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    excluded_count = 0
    duplicates = 0
    for candidate in candidates:
        inputs = json.loads(canonical_input(candidate["inputs"]))
        digest = semantic_json_sha256(inputs)
        if digest in excluded:
            excluded_count += 1
            continue
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        rows.append({"inputSha256": digest, "inputs": inputs})
    rng = random.Random(seed)
    rng.shuffle(rows)
    if len(rows) < evaluation + training:
        raise ExperimentError(
            f"pool holds {len(rows)} usable inputs, fewer than {evaluation + training}"
        )
    counts = {
        "poolRows": len(candidates),
        "excludedHeldOut": excluded_count,
        "duplicateInputs": duplicates,
        "usable": len(rows),
    }
    return rows[:evaluation], rows[evaluation : evaluation + training], counts


# ---- labelers -----------------------------------------------------------------------

Labeler = Callable[[str, str], tuple[str, dict[str, Any]]]


def sonnet_labeler(api_key: str, timeout_seconds: float = 120.0) -> Labeler:
    def label(system: str, user: str) -> tuple[str, dict[str, Any]]:
        body = json.dumps(
            {
                "model": f"anthropic/{SONNET_MODEL}",
                "max_tokens": 256,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "output_config": {"format": {"type": "json_schema", "schema": DECISION_SCHEMA}},
            }
        ).encode()
        last = ""
        for attempt in range(5):
            request = urllib.request.Request(
                SONNET_ROUTE,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    document = json.loads(response.read(65_536).decode())
            except urllib.error.HTTPError as error:
                detail = error.read(1024).decode("utf-8", "replace")
                if error.code in (429, 500, 502, 503, 504, 520, 522, 524) and attempt < 4:
                    last = f"HTTP {error.code}"
                    time.sleep(3.0 * (attempt + 1))
                    continue
                raise ExperimentError(
                    f"Sonnet request failed: HTTP {error.code}: {detail[:200]}"
                ) from None
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if attempt < 4:
                    last = str(error)
                    time.sleep(3.0 * (attempt + 1))
                    continue
                raise ExperimentError(f"Sonnet request failed: {error}") from None
            if document.get("stop_reason") != "end_turn":
                raise ExperimentError(f"Sonnet stopped with {document.get('stop_reason')!r}")
            texts = [
                block["text"]
                for block in document.get("content", [])
                if block.get("type") == "text"
            ]
            if len(texts) != 1:
                raise ExperimentError("Sonnet response must contain exactly one text block")
            decision = json.loads(texts[0])["decision"]
            usage = document.get("usage", {})
            return decision, {
                "model": document.get("model"),
                "provider": document.get("provider"),
                "inputTokens": usage.get("input_tokens"),
                "outputTokens": usage.get("output_tokens"),
                "costUsd": usage.get("cost"),
            }
        raise ExperimentError(f"Sonnet request failed after retries: {last}")

    return label


def qwen_labeler(timeout_seconds: float = 600.0) -> Labeler:
    def label(system: str, user: str) -> tuple[str, dict[str, Any]]:
        body = json.dumps(
            {
                "model": QWEN_MODEL,
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "seed": 1},
                "format": DECISION_SCHEMA,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode()
        request = urllib.request.Request(
            OLLAMA_CHAT, data=body, method="POST", headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                document = json.loads(response.read(65_536).decode())
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ExperimentError(f"Ollama request failed: {error}") from None
        decision = json.loads(document["message"]["content"])["decision"]
        return decision, {
            "model": document.get("model"),
            "evalCount": document.get("eval_count"),
            "totalDurationMs": document.get("total_duration", 0) / 1e6,
        }

    return label


def label_rows(
    rows: Sequence[Mapping[str, Any]],
    labeler: Labeler,
    policy: Mapping[str, Any],
    cache_path: Path,
    *,
    name: str,
    max_cost_usd: float | None = None,
    concurrency: int = 1,
) -> dict[str, dict[str, Any]]:
    """Label every row once, caching per input digest; ``concurrency`` requests in flight."""

    cache: dict[str, dict[str, Any]] = (
        json.loads(cache_path.read_text()) if cache_path.exists() else {}
    )
    spent = sum(float(entry.get("meta", {}).get("costUsd") or 0.0) for entry in cache.values())
    pending = [row for row in rows if row["inputSha256"] not in cache]
    log(f"{name}: {len(rows) - len(pending)} cached, {len(pending)} to label")

    def one(row: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        system, user = build_prompt(policy, row["inputs"])
        started = time.perf_counter()
        decision, meta = labeler(system, user)
        if decision not in SUPPORT:
            raise ExperimentError(f"{name}: label {decision!r} is outside the support")
        return row["inputSha256"], {
            "label": decision,
            "latencyMs": (time.perf_counter() - started) * 1000,
            "meta": meta,
        }

    from concurrent.futures import ThreadPoolExecutor

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        for offset in range(0, len(pending), max(1, concurrency)):
            if max_cost_usd is not None and spent > max_cost_usd:
                raise ExperimentError(
                    f"{name}: spend {spent:.4f} USD exceeds the budget {max_cost_usd}"
                )
            chunk = pending[offset : offset + max(1, concurrency)]
            for digest, entry in pool.map(one, chunk):
                cache[digest] = entry
                spent += float(entry["meta"].get("costUsd") or 0.0)
                done += 1
            if done % 25 < len(chunk) or done == len(pending):
                cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
                log(f"{name}: {done}/{len(pending)} labeled, {spent:.4f} USD")
    cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return cache


# ---- rule labels and agreement ------------------------------------------------------


class RuleLabels:
    def __init__(self, ir: Mapping[str, Any]) -> None:
        self._constraints = compile_constraints(json.loads(json.dumps(ir)))

    def label(self, inputs: Mapping[str, Any]) -> str | None:
        admissible = [
            value
            for value in SUPPORT
            if not self._constraints.evaluate_output_contract(inputs, value)[1]
        ]
        return admissible[0] if len(admissible) == 1 else None


def rubric_step(inputs: Mapping[str, Any]) -> str:
    """The rubric step that decides a case (RUBRIC.md, version 1.1)."""

    order, customer = inputs["order"], inputs["customer"]
    if order["ageDays"] > 90:
        return "1"
    if order["status"] == "fraudulent":
        return "2"
    window = 60 if customer["tier"] == "enterprise" else 30
    if order["ageDays"] > window:
        return "3"
    prior, total, age = customer["priorRefunds"], order["total"], order["ageDays"]
    if (
        prior >= 5
        or (prior >= 3 and total >= 1000)
        or total >= 5000
        or (age <= 1 and total >= 2000 and prior >= 2)
    ):
        return "4"
    return "5"


def agreement(
    rows: Sequence[Mapping[str, Any]],
    sonnet: Mapping[str, Mapping[str, Any]],
    qwen: Mapping[str, Mapping[str, Any]],
    rules: RuleLabels,
) -> dict[str, Any]:
    totals = {"rows": 0, "sonnetQwen": 0, "sonnetRule": 0, "qwenRule": 0, "ruleAmbiguous": 0}
    steps: dict[str, dict[str, Any]] = {}
    confusions: dict[str, int] = {}
    for row in rows:
        digest = row["inputSha256"]
        s, q = sonnet[digest]["label"], qwen[digest]["label"]
        r = rules.label(row["inputs"])
        step = rubric_step(row["inputs"])
        entry = steps.setdefault(
            step,
            {"rule": STEP_NAMES[step], "rows": 0, "sonnetQwen": 0, "sonnetRule": 0, "qwenRule": 0},
        )
        totals["rows"] += 1
        entry["rows"] += 1
        if s == q:
            totals["sonnetQwen"] += 1
            entry["sonnetQwen"] += 1
        else:
            key = f"sonnet {s} / qwen {q}"
            confusions[key] = confusions.get(key, 0) + 1
        if r is None:
            totals["ruleAmbiguous"] += 1
        else:
            if s == r:
                totals["sonnetRule"] += 1
                entry["sonnetRule"] += 1
            if q == r:
                totals["qwenRule"] += 1
                entry["qwenRule"] += 1
    for entry in steps.values():
        for key in ("sonnetQwen", "sonnetRule", "qwenRule"):
            entry[f"{key}Rate"] = round(entry[key] / entry["rows"], 4)
    rates = {
        f"{key}Rate": round(totals[key] / totals["rows"], 4)
        for key in ("sonnetQwen", "sonnetRule", "qwenRule")
    }
    return {
        "totals": {**totals, **rates},
        "bySteps": dict(sorted(steps.items())),
        "disagreements": confusions,
    }


# ---- students ----------------------------------------------------------------------------


class ReplayTeacher:
    """Replays fixed labeled inputs as generated cases; no language model at training time."""

    def __init__(self, name: str, cases: Sequence[GeneratedCase], labels_sha256: str) -> None:
        self._cases = tuple(cases)
        projection = {
            "kind": "semantscript.local-teacher-experiment-replay",
            "labeler": name,
            "labelsSha256": labels_sha256,
            "count": len(cases),
        }
        encoded = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
        self.descriptor = TeacherDescriptor(
            provider=f"replay-labels/{name}",
            model=name,
            configuration_sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def generate(self, ir: Mapping[str, Any], n: int, /) -> tuple[GeneratedCase, ...]:
        if n > len(self._cases):
            raise ExperimentError(f"replay teacher holds {len(self._cases)} cases, not {n}")
        return self._cases[:n]


def student_ir(ir: Mapping[str, Any]) -> dict[str, Any]:
    """The refund IR with its constraints removed and its semantic digest recomputed.

    The dataset generator rejects any case that violates an active constraint,
    which is right for a build and wrong for this experiment: the point is to
    train on each teacher's labels as given, including the local teacher's
    mistakes. Both students train on this same copy, so the comparison is fair;
    the students never see constraints anyway, only the serialized inputs.
    """

    copy = json.loads(json.dumps(ir))
    copy["definition"]["constraints"] = []
    projection = {
        key: copy[key] for key in ("irVersion", "definition", "inputs", "output", "runtime")
    }
    copy["semanticSha256"] = semantic_json_sha256(projection)
    return copy


def recipe(device: str, epochs: int) -> TrainingConfig:
    """The committed compact-release recipe, best epoch by held-out accuracy."""

    return TrainingConfig(
        encoder_name=DEFAULT_ENCODER_NAME,
        encoder_revision=DEFAULT_ENCODER_REVISION,
        local_files_only=True,
        epochs=epochs,
        batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        maximum_sequence_length=128,
        evaluation_ratio=0.1,
        seed=1,
        device=device,
        head_architecture="linear",
        select_best_epoch=True,
        canonical_input_version=2,
    )


def student_logits(
    ir: Mapping[str, Any], training: Any, tokenizer: Any, inputs: Sequence[Mapping[str, Any]]
) -> Any:
    """Raw logits of a trained student over inputs, serialized as the trainer serialized them."""

    import torch

    schema = ir["inputs"]
    texts = [
        serialize_canonical_inputs(
            schema, row, version=training.config.canonical_input_version
        ).decode()
        for row in inputs
    ]
    model = training.model
    model.eval()
    device = torch.device(training.device)
    chunks = []
    with torch.no_grad():
        for offset in range(0, len(texts), 32):
            batch = tokenizer(
                texts[offset : offset + 32],
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=training.config.maximum_sequence_length,
                return_tensors="pt",
            )
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            chunks.append(logits.detach().to("cpu", torch.float64))
    return torch.cat(chunks, dim=0)


def score_cases(
    ir: Mapping[str, Any],
    training: Any,
    tokenizer: Any,
    temperature: float,
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Accuracy, 15-bin top-1 ECE and Brier of a trained student on labeled cases."""

    import torch

    logits = student_logits(ir, training, tokenizer, [case["inputs"] for case in cases])
    probabilities = [row.tolist() for row in torch.softmax(logits / temperature, dim=-1)]
    counts = [0] * ECE_BIN_COUNT
    correct_in_bin = [0] * ECE_BIN_COUNT
    confidence_in_bin = [0.0] * ECE_BIN_COUNT
    correct = 0
    brier = 0.0
    misses: list[dict[str, Any]] = []
    for case, distribution in zip(cases, probabilities, strict=True):
        index = max(range(len(SUPPORT)), key=lambda i: distribution[i])
        predicted = SUPPORT[index]
        hit = predicted == case["label"]
        correct += int(hit)
        confidence = distribution[index]
        bucket = min(ECE_BIN_COUNT - 1, int(confidence * ECE_BIN_COUNT))
        counts[bucket] += 1
        correct_in_bin[bucket] += int(hit)
        confidence_in_bin[bucket] += confidence
        brier += sum(
            (distribution[i] - (1.0 if SUPPORT[i] == case["label"] else 0.0)) ** 2
            for i in range(len(SUPPORT))
        )
        if not hit:
            misses.append(
                {
                    "inputs": case["inputs"],
                    "label": case["label"],
                    "predicted": predicted,
                    "confidence": round(confidence, 4),
                }
            )
    ece = sum(
        (counts[b] / len(cases))
        * abs(correct_in_bin[b] / counts[b] - confidence_in_bin[b] / counts[b])
        for b in range(ECE_BIN_COUNT)
        if counts[b]
    )
    return {
        "cases": len(cases),
        "correct": correct,
        "accuracy": round(correct / len(cases), 4),
        "ece": round(ece, 4),
        "brier": round(brier / len(cases), 4),
        "misses": misses[:20],
        "missCount": len(misses),
    }


def train_student(
    name: str,
    ir: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    evaluation_cases: Sequence[Mapping[str, Any]],
    final_cases: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    device: str,
    epochs: int,
    cache_directory: Path,
) -> dict[str, Any]:
    import torch

    cases = [
        GeneratedCase(dict(row["inputs"]), labels[row["inputSha256"]]["label"])
        for row in training_rows
    ]
    labels_sha256 = hashlib.sha256(
        json.dumps(
            [[row["inputSha256"], labels[row["inputSha256"]]["label"]] for row in training_rows]
        ).encode()
    ).hexdigest()
    teacher = ReplayTeacher(name, cases, labels_sha256)
    unconstrained = student_ir(ir)
    base = SyntheticDatasetGenerator(teacher, cache_directory).generate(unconstrained, len(cases))
    log(f"student {name}: training {epochs} epochs on {len(cases)} rows")
    started = time.monotonic()
    training = train_classifier(
        unconstrained, base, None, config=recipe(device, epochs), tokenizer=tokenizer
    )
    train_seconds = time.monotonic() - started
    # The verifier refuses a constrained function without an adversarial sidecar,
    # so the temperature is fitted here on the trainer's own held-out split with
    # the model package's fitter (the same routine the verifier calls).
    from semantscript_model.calibration import fit_temperature

    calibration_rows = training.split.evaluation
    calibration_logits = student_logits(
        ir, training, tokenizer, [row.inputs for row in calibration_rows]
    )
    targets = torch.tensor([row.label_index for row in calibration_rows], dtype=torch.int64)
    temperature = float(fit_temperature(calibration_logits, targets))
    calibration_accuracy = float((calibration_logits.argmax(dim=-1) == targets).double().mean())
    result = {
        "student": name,
        "teacher": teacher.descriptor.provider,
        "trainedOnIr": {
            "semanticSha256": unconstrained["semanticSha256"],
            "constraintsStripped": True,
        },
        "trainingRows": training.training_row_count,
        "heldOutRows": training.held_out_row_count,
        "selectedEpoch": training.selected_epoch,
        "heldOutCurve": [round(m.held_out_accuracy, 4) for m in training.metrics],
        "trainSeconds": round(train_seconds, 1),
        "temperature": round(temperature, 4),
        "calibrationSplit": {
            "rows": len(calibration_rows),
            "accuracy": round(calibration_accuracy, 4),
        },
        "evaluationSet": score_cases(ir, training, tokenizer, temperature, evaluation_cases),
        "finalSet": score_cases(ir, training, tokenizer, temperature, final_cases),
    }
    del training
    torch.cuda.empty_cache()
    return result


# ---- driver --------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["sample", "label-qwen", "label-sonnet", "train", "all"], default="all"
    )
    parser.add_argument("--evaluation", type=int, default=300)
    parser.add_argument("--training", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sonnet-max-cost-usd", type=float, default=6.0)
    parser.add_argument(
        "--sonnet-limit", type=int, default=None, help="label only the first N rows (smoke test)"
    )
    parser.add_argument("--sonnet-concurrency", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    out = arguments.output_dir
    out.mkdir(parents=True, exist_ok=True)
    stage = arguments.stage

    policy = load_policy()
    sets_path = out / "inputs.json"
    if stage in ("sample", "all") or not sets_path.exists():
        candidates = json.loads(POOL.read_text())["candidates"]
        evaluation, training, counts = sample_inputs(
            candidates,
            held_out_digests(),
            evaluation=arguments.evaluation,
            training=arguments.training,
            seed=arguments.seed,
        )
        sets_path.write_text(
            json.dumps(
                {
                    "kind": "semantscript.local-teacher-experiment-inputs",
                    "seed": arguments.seed,
                    "counts": counts,
                    "evaluation": evaluation,
                    "training": training,
                },
                indent=1,
            )
            + "\n"
        )
        log(f"sampled {len(evaluation)} evaluation and {len(training)} training inputs ({counts})")
    sets = json.loads(sets_path.read_text())
    evaluation_rows, training_rows = sets["evaluation"], sets["training"]
    all_rows = evaluation_rows + training_rows

    if stage in ("label-qwen", "all"):
        label_rows(
            all_rows, qwen_labeler(), policy, out / "labels-qwen3-14b.json", name="qwen3:14b"
        )
    if stage in ("label-sonnet", "all"):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("OPENROUTER_API_KEY is not set", file=sys.stderr)
            return 2
        rows = all_rows if arguments.sonnet_limit is None else all_rows[: arguments.sonnet_limit]
        label_rows(
            rows,
            sonnet_labeler(api_key),
            policy,
            out / "labels-claude-sonnet-5.json",
            name="claude-sonnet-5",
            max_cost_usd=arguments.sonnet_max_cost_usd,
            concurrency=arguments.sonnet_concurrency,
        )
    if stage in ("sample", "label-qwen", "label-sonnet"):
        return 0

    sonnet = json.loads((out / "labels-claude-sonnet-5.json").read_text())
    qwen = json.loads((out / "labels-qwen3-14b.json").read_text())
    missing = [
        row["inputSha256"]
        for row in all_rows
        if row["inputSha256"] not in sonnet or row["inputSha256"] not in qwen
    ]
    if missing:
        raise ExperimentError(
            f"{len(missing)} inputs lack a label from both teachers; run the label stages first"
        )

    with tempfile.TemporaryDirectory(prefix="local-teacher-ir-") as directory:
        compiled = compile_refund_program(directory)
        ir = compiled.source_ir
    rules = RuleLabels(ir)
    report = {
        "kind": "semantscript.local-teacher-experiment",
        "startedAt": utc_now(),
        "taskSpecSha256": policy["taskSpecSha256"],
        "functionId": compiled.function_id,
        "semanticSha256": compiled.semantic_sha256,
        "inputs": sets["counts"]
        | {
            "evaluation": len(evaluation_rows),
            "training": len(training_rows),
            "seed": sets["seed"],
        },
        "labelers": {
            "claude-sonnet-5": {
                "route": SONNET_ROUTE,
                "model": f"anthropic/{SONNET_MODEL}",
                "structuredOutput": "json_schema",
                "resolved": sorted({str(e["meta"].get("model")) for e in sonnet.values()}),
                "providers": sorted({str(e["meta"].get("provider")) for e in sonnet.values()}),
                "costUsd": round(
                    sum(float(e["meta"].get("costUsd") or 0) for e in sonnet.values()), 5
                ),
                "inputTokens": sum(int(e["meta"].get("inputTokens") or 0) for e in sonnet.values()),
                "latencyMsP50": sorted(e["latencyMs"] for e in sonnet.values())[len(sonnet) // 2],
            },
            "qwen3:14b": {
                "route": OLLAMA_CHAT,
                "options": {"think": False, "temperature": 0, "seed": 1},
                "resolved": sorted({str(e["meta"].get("model")) for e in qwen.values()}),
                "latencyMsP50": sorted(e["latencyMs"] for e in qwen.values())[len(qwen) // 2],
            },
        },
        "agreement": {
            "trainingInputs": agreement(training_rows, sonnet, qwen, rules),
            "evaluationInputs": agreement(evaluation_rows, sonnet, qwen, rules),
        },
    }
    evaluation_cases = [
        {
            "inputs": row["inputs"],
            "label": sonnet[row["inputSha256"]]["label"],
            "inputSha256": row["inputSha256"],
        }
        for row in evaluation_rows
    ]
    evaluation_document = {
        "kind": "semantscript.local-teacher-experiment-evaluation-set",
        "labeler": "claude-sonnet-5 via OpenRouter (Anthropic-format route, json_schema)",
        "note": "model-labeled evaluation target for TASK-5.13; not human-authored, not judge-adjudicated, never trained on",
        "cases": evaluation_cases,
    }
    evaluation_document["payloadSha256"] = semantic_json_sha256(evaluation_document["cases"])
    (out / "evaluation-set.json").write_text(json.dumps(evaluation_document, indent=1) + "\n")
    report["evaluationSetSha256"] = evaluation_document["payloadSha256"]
    final = json.loads((HELDOUT / "final-benchmark-dataset.json").read_text())
    final_cases = [{"inputs": c["inputs"], "label": c["expected"]} for c in final["cases"]]
    report["finalSetSha256"] = final["payloadSha256"]

    from semantscript_trainer.cli import _load_tokenizer

    tokenizer = _load_tokenizer(recipe(arguments.device, arguments.epochs))
    cache_directory = out / "dataset-cache"
    students = []
    for name, labels in (("claude-sonnet-5", sonnet), ("qwen3-14b", qwen)):
        students.append(
            train_student(
                name,
                ir,
                training_rows,
                labels,
                evaluation_cases,
                final_cases,
                tokenizer=tokenizer,
                device=arguments.device,
                epochs=arguments.epochs,
                cache_directory=cache_directory,
            )
        )
        log(
            f"student {name}: evaluation {students[-1]['evaluationSet']['accuracy']} (ECE {students[-1]['evaluationSet']['ece']}), final set {students[-1]['finalSet']['accuracy']}"
        )
        (out / "results.json").write_text(
            json.dumps({**report, "students": students, "finishedAt": utc_now()}, indent=1) + "\n"
        )
    report["students"] = students
    report["finishedAt"] = utc_now()
    (out / "results.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
