"""Training corpus from real inputs labeled by the compiled constraints.

The refund policy is fully determined by its six deterministic constraints, so
for any input exactly one output is admissible. This teacher takes real,
de-identified refund inputs from the UCI pool (every held-out input excluded by
semantic digest), assigns the unique admissible label, and replays the frozen
Opus synthetic cases in front of them so the corpus keeps the threshold-dense
teacher data. Boundary pairs and counterfactual twins are still proposed by the
Claude CLI teacher, now anchored on real inputs.

Provenance is explicit: the descriptor's provider names the rule labeling, and
the configuration projection binds the pool file digest, the replayed dataset
digest, the exclusion digest and the CLI configuration.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.refund.program.claude_cli_teacher import (
    ClaudeCliTeacherConfig,
    ClaudeCliTrainingTeacher,
)

from semantscript_trainer.case_contract import validate_case_count
from semantscript_trainer.constraints import compile_constraints
from semantscript_trainer.semantic_json import semantic_json_sha256
from semantscript_trainer.teacher import (
    BoundaryPairProposal,
    CounterfactualProposal,
    GeneratedCase,
    NeuralFunctionIr,
    TeacherConfigurationError,
    TeacherDescriptor,
)

POOLED_CONFIG_KIND = "semantscript.refund-pooled-corpus-teacher-config"
_PROVIDER = "real-input-pool-rule-labels+anthropic-claude-cli-adversarial"
_LABELING = "compiled-constraints-unique-admissible-output"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class PoolReport:
    pool_rows: int
    labeled: int
    excluded_held_out: int
    duplicate_inputs: int
    ambiguous: int
    label_counts: dict[str, int]

    def document(self) -> dict[str, Any]:
        return {
            "poolRows": self.pool_rows,
            "labeled": self.labeled,
            "excludedHeldOut": self.excluded_held_out,
            "duplicateInputs": self.duplicate_inputs,
            "ambiguous": self.ambiguous,
            "labelCounts": dict(sorted(self.label_counts.items())),
        }


class PooledCorpusTeacher:
    """Teacher that replays frozen synthetic cases then labels real pool inputs."""

    def __init__(
        self,
        *,
        pool_path: Path,
        replay_path: Path | None,
        heldout_directory: Path,
        fraud_percent: int,
        adversarial: ClaudeCliTrainingTeacher,
        stale_status_twins: bool = False,
    ) -> None:
        if not isinstance(fraud_percent, int) or not 0 <= fraud_percent <= 100:
            raise TeacherConfigurationError("fraud_percent must be an integer from 0 through 100")
        if not isinstance(stale_status_twins, bool):
            raise TeacherConfigurationError("stale_status_twins must be a boolean")
        self._stale_status_twins = stale_status_twins
        self._pool_path = pool_path.resolve()
        self._replay_path = replay_path.resolve() if replay_path is not None else None
        self._heldout_directory = heldout_directory.resolve()
        self._fraud_percent = fraud_percent
        self._adversarial = adversarial
        self._pool_bytes = self._pool_path.read_bytes()
        self._pool = json.loads(self._pool_bytes)
        self._replay_cases: tuple[GeneratedCase, ...] = ()
        self._replay_sha256: str | None = None
        if self._replay_path is not None:
            replay_document = json.loads(self._replay_path.read_text(encoding="utf-8"))
            self._replay_sha256 = replay_document["payloadSha256"]
            self._replay_cases = tuple(
                GeneratedCase(inputs=case["inputs"], output=case["output"])
                for case in replay_document["payload"]["cases"]
                if case.get("origin") == "synthetic"
            )
        self._excluded = self._held_out_digests()
        self._last_report: PoolReport | None = None

    def _held_out_digests(self) -> frozenset[str]:
        digests: set[str] = set()
        for name in ("release-verification.json", "final-benchmark-dataset.json"):
            document = json.loads((self._heldout_directory / name).read_text(encoding="utf-8"))
            for case in document["cases"]:
                digests.add(case["inputSha256"])
        return frozenset(digests)

    @property
    def configuration_projection(self) -> dict[str, Any]:
        return {
            "kind": POOLED_CONFIG_KIND,
            "configVersion": 1,
            "labeling": _LABELING,
            "pool": {
                "path": _relative(self._pool_path),
                "sha256": hashlib.sha256(self._pool_bytes).hexdigest(),
            },
            "replay": None
            if self._replay_path is None
            else {
                "path": _relative(self._replay_path),
                "payloadSha256": self._replay_sha256,
                "caseCount": len(self._replay_cases),
            },
            "exclusions": {
                "heldoutDirectory": _relative(self._heldout_directory),
                "sha256": hashlib.sha256("\n".join(sorted(self._excluded)).encode()).hexdigest(),
                "count": len(self._excluded),
            },
            "fraudPercent": self._fraud_percent,
            # For every real order older than 90 days, also emit the same order with the
            # other status so the stale rule's precedence over fraud is taught from real
            # inputs; both twins are rule-labeled like every other pool row.
            "staleStatusTwins": self._stale_status_twins,
            "adversarial": self._adversarial.configuration_projection,
        }

    @property
    def descriptor(self) -> TeacherDescriptor:
        encoded = json.dumps(
            self.configuration_projection, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return TeacherDescriptor(
            provider=_PROVIDER,
            model=self._adversarial.descriptor.model,
            configuration_sha256=hashlib.sha256(encoded).hexdigest(),
        )

    @property
    def last_pool_report(self) -> PoolReport | None:
        return self._last_report

    def labeled_pool(self, ir: NeuralFunctionIr) -> tuple[GeneratedCase, ...]:
        constraints = compile_constraints(ir)
        support = list(ir["output"]["head"]["support"])
        seen: set[str] = {semantic_json_sha256(case.inputs) for case in self._replay_cases}
        cases: list[GeneratedCase] = []
        excluded = duplicates = ambiguous = 0
        counts: dict[str, int] = {}
        candidates: list[dict[str, Any]] = []
        for row in sorted(self._pool["candidates"], key=lambda entry: entry["key"]):
            inputs = copy.deepcopy(row["inputs"])
            if int(row["key"][:8], 16) % 100 < self._fraud_percent:
                inputs["order"]["status"] = "fraudulent"
            candidates.append(inputs)
            if self._stale_status_twins and inputs["order"]["ageDays"] > 90:
                twin = copy.deepcopy(inputs)
                twin["order"]["status"] = (
                    "paid" if inputs["order"]["status"] == "fraudulent" else "fraudulent"
                )
                candidates.append(twin)
        for inputs in candidates:
            digest = semantic_json_sha256(inputs)
            if digest in self._excluded:
                excluded += 1
                continue
            if digest in seen:
                duplicates += 1
                continue
            admissible = [
                value
                for value in support
                if not constraints.evaluate_output_contract(inputs, value)[1]
            ]
            if len(admissible) != 1:
                ambiguous += 1
                continue
            seen.add(digest)
            label = admissible[0]
            counts[json.dumps(label)] = counts.get(json.dumps(label), 0) + 1
            cases.append(GeneratedCase(inputs=inputs, output=label))
        self._last_report = PoolReport(
            pool_rows=len(self._pool["candidates"]),
            labeled=len(cases),
            excluded_held_out=excluded,
            duplicate_inputs=duplicates,
            ambiguous=ambiguous,
            label_counts=counts,
        )
        return tuple(cases)

    def generate(self, ir: NeuralFunctionIr, n: int, /) -> tuple[GeneratedCase, ...]:
        expected = validate_case_count(n)
        available = self._replay_cases + self.labeled_pool(ir)
        if expected > len(available):
            raise TeacherConfigurationError(
                f"pooled teacher can supply {len(available)} cases, not {expected}"
            )
        return available[:expected]

    def generate_boundary_pair(
        self, ir: NeuralFunctionIr, constraint_index: int, /
    ) -> BoundaryPairProposal:
        return self._adversarial.generate_boundary_pair(ir, constraint_index)

    def generate_counterfactual(
        self, ir: NeuralFunctionIr, anchor: GeneratedCase, /
    ) -> CounterfactualProposal:
        return self._adversarial.generate_counterfactual(ir, anchor)


def cli_config_from_projection(projection: Mapping[str, Any]) -> ClaudeCliTeacherConfig:
    """Rebuild a Claude CLI teacher configuration from its public projection."""

    return ClaudeCliTeacherConfig(
        executable=projection["cli"]["executable"],
        timeout_seconds=projection["limits"]["timeoutSeconds"],
        stdout_limit_bytes=projection["limits"]["stdoutBytes"],
        stderr_limit_bytes=projection["limits"]["stderrBytes"],
        max_budget_usd=projection["request"]["maxBudgetUsd"],
        concurrency=projection["generation"]["concurrency"],
        maximum_case_attempts=projection["generation"]["maximumCaseAttempts"],
        cli_version=projection["cli"]["requiredVersion"],
        model=projection["request"]["model"],
    )


def pooled_teacher_from_projection(
    projection: Mapping[str, Any],
    *,
    process_runner: Any | None = None,
) -> PooledCorpusTeacher:
    """Rebuild a pooled teacher (with an optional injected CLI runner) from its projection."""

    if projection.get("kind") != POOLED_CONFIG_KIND:
        raise TeacherConfigurationError("projection is not a pooled corpus teacher configuration")
    adversarial = ClaudeCliTrainingTeacher(
        cli_config_from_projection(projection["adversarial"]),
        process_runner=process_runner,
    )
    replay = projection.get("replay")
    return PooledCorpusTeacher(
        pool_path=_REPOSITORY_ROOT / projection["pool"]["path"],
        replay_path=None if replay is None else _REPOSITORY_ROOT / replay["path"],
        heldout_directory=_REPOSITORY_ROOT / projection["exclusions"]["heldoutDirectory"],
        fraud_percent=int(projection["fraudPercent"]),
        adversarial=adversarial,
        stale_status_twins=bool(projection.get("staleStatusTwins", False)),
    )


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPOSITORY_ROOT).as_posix()
    except ValueError as error:
        raise TeacherConfigurationError(
            f"pooled corpus inputs must live inside the repository: {path}"
        ) from error


__all__ = [
    "POOLED_CONFIG_KIND",
    "PoolReport",
    "PooledCorpusTeacher",
    "cli_config_from_projection",
    "pooled_teacher_from_projection",
]
