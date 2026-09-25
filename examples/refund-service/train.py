"""Train the refund service's artifact from its own constraints, no LLM teacher.

Every sema expression in this service declares a complete constraint set: for
any input exactly one output satisfies every active `always` and `never`
predicate. That makes the constraints a labeling function, so the training
corpus is sampled structured inputs labeled by the constraints themselves, and
the adversarial cases (a boundary pair per constraint, single-field
counterfactual twins) come from the same rule. The trainer, verifier, build
cache and artifact export are the ordinary `train_bundle` path; only the
teacher is local.

Usage (from this directory, after `npm run build`)::

    npm run train                # GPU, publishes .semantscript/artifact
    npm run train -- --cases 200 --epochs 4 --device cpu

The held-out set written beside the artifact (`.semantscript/heldout.json`) is
sampled with a different seed than training and scored later by
`scripts/measure.mjs` through the Node runtime.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from semantscript_trainer.adversarial import AdversarialGenerationConfig
from semantscript_trainer.cli import TrainBundleFailure, train_bundle
from semantscript_trainer.constraints import CompiledConstraints, compile_constraints
from semantscript_trainer.teacher import (
    BoundaryPairProposal,
    CounterfactualProposal,
    GeneratedCase,
    TeacherDescriptor,
)
from semantscript_trainer.training import TrainingConfig
from semantscript_trainer.verification import VerificationConfig

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "dist" / "semantscript.ir.v1.json"
ARTIFACT = HERE / ".semantscript" / "artifact"
CACHE = HERE / ".semantscript" / "cache"
REPORT = HERE / ".semantscript" / "train-report.json"
HELDOUT = HERE / ".semantscript" / "heldout.json"

# Sampling ranges for numeric fields, by property name. Thresholds come from the
# constraints themselves (every numeric literal in a predicate, and its
# neighbours), so the corpus is dense where the policy turns.
NUMBER_RANGES: dict[str, tuple[int, int, str]] = {
    # name: (low, high, distribution): "uniform" over the range, "log" (log-uniform,
    # so small amounts are as common as large ones) or "count" (zero half the
    # time, else uniform).
    "total": (0, 8000, "log"),
    "ageDays": (0, 120, "uniform"),
    "priorRefunds": (0, 6, "count"),
    "impactedUsers": (0, 500, "log"),
    "ordersLastHour": (0, 8, "count"),
    "chargebacks": (0, 3, "count"),
}
DEFAULT_RANGE = (0, 100, "uniform")
MAXIMUM_SAMPLING_ATTEMPTS = 200_000


def numeric_literals(node: Any, into: set[float]) -> None:
    if isinstance(node, dict):
        if node.get("node") == "literal" and isinstance(node.get("value"), (int, float)):
            if not isinstance(node["value"], bool):
                into.add(float(node["value"]))
        for value in node.values():
            numeric_literals(value, into)
    elif isinstance(node, list):
        for value in node:
            numeric_literals(value, into)


class ConstraintRule:
    """Labels inputs of one function by its constraints; None when ambiguous."""

    def __init__(self, ir: Mapping[str, Any]) -> None:
        self.ir = ir
        self.constraints: CompiledConstraints = compile_constraints(copy.deepcopy(dict(ir)))
        self.support: list[Any] = list(ir["output"]["head"]["support"])
        self.inputs: list[Mapping[str, Any]] = list(ir["inputs"])
        literals: set[float] = set()
        numeric_literals(ir["definition"]["constraints"], literals)
        self.thresholds = sorted(literals)

    def label(self, inputs: Mapping[str, Any]) -> Any | None:
        satisfying = [
            value
            for value in self.support
            if not self.constraints.evaluate_output_contract(inputs, value)[1]
        ]
        return satisfying[0] if len(satisfying) == 1 else None

    def predicate(self, index: int, inputs: Mapping[str, Any]) -> bool:
        return self.constraints.evaluate(index, inputs)

    # Sampling -------------------------------------------------------------

    def sample(self, rng: random.Random) -> dict[str, Any]:
        return {
            entry["name"]: self.sample_type(entry["type"], entry["name"], rng)
            for entry in self.inputs
        }

    def sample_type(self, spec: Mapping[str, Any], name: str, rng: random.Random) -> Any:
        kind = spec["kind"]
        if kind == "object":
            return {
                field["name"]: self.sample_type(field["type"], field["name"], rng)
                for field in spec["fields"]
            }
        if kind == "boolean":
            return rng.random() < 0.5
        if kind == "number":
            return self.sample_number(name, rng)
        if kind == "literal":
            return spec["value"]
        if kind == "enum":
            return rng.choice(list(spec["values"]))
        if kind == "union":
            return self.sample_type(rng.choice(list(spec["variants"])), name, rng)
        if kind == "string":
            return rng.choice(["alpha", "bravo", "charlie", "delta"])
        if kind == "null":
            return None
        raise ValueError(f"unsupported input kind {kind!r} for {name}")

    def sample_number(self, name: str, rng: random.Random) -> float | int:
        low, high, distribution = NUMBER_RANGES.get(name, DEFAULT_RANGE)
        near = [t for t in self.thresholds if low <= t <= high]
        if near and rng.random() < 0.3:
            value = min(high, max(low, rng.choice(near) + rng.choice([-1, 0, 0, 1])))
        elif distribution == "count":
            value = 0 if rng.random() < 0.5 else rng.randint(max(low, 1), high)
        elif distribution == "log":
            # Half log-uniform (small amounts are common), half uniform (large
            # amounts are still seen often enough to learn their thresholds).
            decimals = 1 if name == "total" and rng.random() < 0.3 else 0
            if rng.random() < 0.5:
                value = round(10 ** rng.uniform(0, math.log10(high)), decimals)
            else:
                value = round(rng.uniform(low, high), decimals)
        else:
            value = rng.randint(low, high)
        return int(value) if float(value).is_integer() else float(value)

    def candidate_values(
        self, spec: Mapping[str, Any], name: str, current: Any, rng: random.Random
    ) -> list[Any]:
        kind = spec["kind"]
        if kind == "boolean":
            return [not current]
        if kind == "number":
            low, high, _ = NUMBER_RANGES.get(name, DEFAULT_RANGE)
            values: list[Any] = []
            for threshold in self.thresholds:
                for delta in (0, 1, -1):
                    candidate = threshold + delta
                    if low <= candidate <= high:
                        values.append(int(candidate))
            values.extend(rng.randint(low, high) for _ in range(4))
            return [value for value in dict.fromkeys(values) if value != current]
        if kind == "union":
            return [
                variant["value"]
                for variant in spec["variants"]
                if variant["kind"] == "literal" and variant["value"] != current
            ]
        if kind == "enum":
            return [value for value in spec["values"] if value != current]
        return []

    def leaves(self) -> Iterator[tuple[tuple[str, ...], Mapping[str, Any]]]:
        def walk(
            spec: Mapping[str, Any], path: tuple[str, ...]
        ) -> Iterator[tuple[tuple[str, ...], Mapping[str, Any]]]:
            if spec["kind"] == "object":
                for field in spec["fields"]:
                    yield from walk(field["type"], (*path, field["name"]))
            else:
                yield path, spec

        for entry in self.inputs:
            yield from walk(entry["type"], (entry["name"],))


def get_path(inputs: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = inputs
    for key in path:
        value = value[key]
    return value


def set_path(inputs: dict[str, Any], path: Sequence[str], value: Any) -> None:
    target: Any = inputs
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def key_of(inputs: Mapping[str, Any]) -> str:
    return json.dumps(inputs, sort_keys=True, separators=(",", ":"))


class ConstraintTeacher:
    """A rule teacher: samples inputs and labels them from the constraints."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        configuration = {
            "kind": "semantscript.example.constraint-teacher",
            "version": 3,
            "seed": seed,
            "ranges": NUMBER_RANGES,
        }
        encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
        self.descriptor = TeacherDescriptor(
            provider="constraint-rule",
            model="refund-service-constraints",
            configuration_sha256=hashlib.sha256(encoded).hexdigest(),
        )
        self._rules: dict[str, ConstraintRule] = {}
        self._attempts: dict[str, int] = {}

    def rule_for(self, ir: Mapping[str, Any]) -> ConstraintRule:
        rule = self._rules.get(ir["id"])
        if rule is None:
            rule = ConstraintRule(copy.deepcopy(dict(ir)))
            self._rules[ir["id"]] = rule
        return rule

    def rng(self, ir: Mapping[str, Any], purpose: str) -> random.Random:
        # Distinct, reproducible streams per function and purpose; a retry of the
        # same purpose continues the stream instead of repeating it.
        key = f"{ir['id']}:{purpose}"
        attempt = self._attempts.get(key, 0)
        self._attempts[key] = attempt + 1
        digest = hashlib.sha256(f"{self.seed}:{key}:{attempt}".encode()).hexdigest()
        return random.Random(int(digest[:16], 16))

    def labeled_samples(
        self, ir: Mapping[str, Any], n: int, purpose: str
    ) -> tuple[GeneratedCase, ...]:
        rule = self.rule_for(ir)
        rng = self.rng(ir, purpose)
        cases: dict[str, GeneratedCase] = {}
        attempts = 0
        while len(cases) < n:
            attempts += 1
            if attempts > MAXIMUM_SAMPLING_ATTEMPTS:
                raise RuntimeError(f"{ir['id']}: could not sample {n} distinct labeled inputs")
            inputs = rule.sample(rng)
            key = key_of(inputs)
            if key in cases:
                continue
            label = rule.label(inputs)
            if label is None:
                continue
            case = GeneratedCase(inputs, label)
            # Every training case may be picked as a counterfactual anchor, so the
            # corpus keeps only inputs that one field edit can move across the
            # policy; interior points several edits from any boundary are dropped.
            if purpose == "train" and self.counterfactual(rule, case, rng) is None:
                continue
            cases[key] = case
        return tuple(cases.values())

    def counterfactual(
        self, rule: ConstraintRule, anchor: GeneratedCase, rng: random.Random
    ) -> CounterfactualProposal | None:
        leaves = list(rule.leaves())
        rng.shuffle(leaves)
        for path, spec in leaves:
            current = get_path(anchor.inputs, path)
            for value in rule.candidate_values(spec, path[-1], current, rng):
                twin = copy.deepcopy(dict(anchor.inputs))
                set_path(twin, path, value)
                label = rule.label(twin)
                if label is None or label == anchor.output:
                    continue
                return CounterfactualProposal(
                    twin=GeneratedCase(twin, label),
                    reason=f"setting {'.'.join(path)} from {current!r} to {value!r} moves the answer to {label!r}",
                )
        return None

    def generate(self, ir: Mapping[str, Any], n: int, /) -> tuple[GeneratedCase, ...]:
        return self.labeled_samples(ir, n, "train")

    def generate_boundary_pair(self, ir: Mapping[str, Any], index: int, /) -> BoundaryPairProposal:
        # Two labeled inputs one field apart with the predicate false on one side
        # and true on the other.
        rule = self.rule_for(ir)
        rng = self.rng(ir, f"boundary:{index}")
        for _ in range(MAXIMUM_SAMPLING_ATTEMPTS):
            inputs = rule.sample(rng)
            label = rule.label(inputs)
            if label is None:
                continue
            side = rule.predicate(index, inputs)
            leaves = list(rule.leaves())
            rng.shuffle(leaves)
            for path, spec in leaves:
                current = get_path(inputs, path)
                for value in rule.candidate_values(spec, path[-1], current, rng):
                    twin = copy.deepcopy(inputs)
                    set_path(twin, path, value)
                    if rule.predicate(index, twin) == side:
                        continue
                    twin_label = rule.label(twin)
                    if twin_label is None:
                        continue
                    pair = {
                        side: GeneratedCase(inputs, label),
                        not side: GeneratedCase(twin, twin_label),
                    }
                    return BoundaryPairProposal(
                        predicate_false=pair[False], predicate_true=pair[True]
                    )
        raise RuntimeError(f"{ir['id']}: constraint {index} has no reachable two-sided boundary")

    def generate_counterfactual(
        self, ir: Mapping[str, Any], anchor: GeneratedCase, /
    ) -> CounterfactualProposal:
        proposal = self.counterfactual(self.rule_for(ir), anchor, self.rng(ir, "counterfactual"))
        if proposal is None:
            raise RuntimeError(
                f"{ir['id']}: no single-field edit changes the label of {anchor.inputs}"
            )
        return proposal


def recipe(arguments: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        local_files_only=arguments.local_files_only,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=0.01,
        maximum_sequence_length=128,
        evaluation_ratio=0.1,
        seed=arguments.seed,
        device=arguments.device,
        head_architecture="linear",
        select_best_epoch=True,
        canonical_input_version=2,
    )


def write_heldout(bundle: Mapping[str, Any], teacher: ConstraintTeacher, per_function: int) -> None:
    functions = []
    for ir in bundle["functions"]:
        cases = teacher.labeled_samples(ir, per_function, "heldout")
        functions.append(
            {
                "id": ir["id"],
                "name": ir["definition"].get("name") or ir["output"]["tsType"],
                "adapterRef": ir["model"]["adapter"],
                "support": ir["output"]["head"]["support"],
                "cases": [{"inputs": case.inputs, "expected": case.output} for case in cases],
            }
        )
    HELDOUT.parent.mkdir(parents=True, exist_ok=True)
    HELDOUT.write_text(
        json.dumps({"kind": "refund-service-heldout", "functions": functions}, indent=1) + "\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument(
        "--cases", type=int, default=800, help="sampled training cases per expression"
    )
    parser.add_argument("--heldout", type=int, default=200, help="held-out cases per expression")
    parser.add_argument("--counterfactual-ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument(
        "--max-constraint-violation-rate",
        type=float,
        default=0.0,
        help="fraction of raw predictions allowed to violate an active constraint (recorded in the report)",
    )
    parser.add_argument("--full", action="store_true", help="retrain everything, ignore the cache")
    arguments = parser.parse_args(argv)

    bundle = json.loads(arguments.bundle.read_text())
    teacher = ConstraintTeacher(arguments.seed)
    write_heldout(bundle, teacher, arguments.heldout)
    try:
        result = train_bundle(
            bundle,
            arguments.artifact,
            teacher=teacher,
            cache_directory=arguments.cache_dir,
            cases=arguments.cases,
            training_config=recipe(arguments),
            verification_config=VerificationConfig(
                maximum_constraint_violation_rate=arguments.max_constraint_violation_rate
            ),
            adversarial_config=AdversarialGenerationConfig(
                counterfactual_ratio=arguments.counterfactual_ratio
            ),
            application_id="refund-service",
            full=arguments.full,
            log=lambda message: print(message, flush=True),
        )
    except TrainBundleFailure as failure:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(failure.report, indent=1) + "\n")
        print(f"training failed: {failure}", file=sys.stderr)
        return 1
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(result.report, indent=1) + "\n")
    print(
        f"artifact {result.exported.artifact_root} release {result.exported.manifest_sha256[:12]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
