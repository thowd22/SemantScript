"""Post-training calibration, contract verification, and manifest-ready records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast

from semantscript_trainer.adversarial import AdversarialDataset
from semantscript_trainer.canonical_input import CanonicalInputError, serialize_canonical_inputs
from semantscript_trainer.case_contract import validate_case
from semantscript_trainer.constraints import (
    MAXIMUM_CONSTRAINT_EVALUATION_STEPS,
    ConstraintConfigurationError,
    ConstraintEvaluationBudget,
    ConstraintEvaluationError,
    compile_constraints,
)
from semantscript_trainer.dataset import TrainingDataset
from semantscript_trainer.strict_json import StrictJsonError, StrictJsonLimits, loads_strict_json
from semantscript_trainer.suggestions import (
    LabelledCase,
    Violation,
    calibration_suggestion,
    gold_miss_suggestion,
    type_error_suggestion,
    underfit_suggestion,
    violation_suggestion,
)
from semantscript_trainer.teacher import GeneratedCase, JsonValue, NeuralFunctionIr
from semantscript_trainer.training import (
    MAXIMUM_BATCH_SIZE,
    MAXIMUM_BATCH_TOKENS,
    MAXIMUM_CANONICAL_ROW_BYTES,
    MAXIMUM_CORPUS_TEXT_BYTES,
    TrainingConfig,
    TrainingResult,
    head_slices,
    predict_indices,
)
from semantscript_trainer.training_contract import (
    MAXIMUM_TRAINING_ROW_COUNT,
    HeldOutSplitConfig,
    OutputHead,
    TrainingCorpus,
    TrainingRow,
    assemble_training_corpus,
    output_labels,
    split_training_corpus,
)

_MAXIMUM_EXTERNAL_HUMAN_VERIFICATION_CASE_COUNT = 10_000
# Keep import-time trainer use independent of the optional model/training stack.
# These values are the public v1 calibration bounds mirrored by the model module;
# `_calibrate` loads the implementation only when verification actually runs.
DEFAULT_ECE_BIN_COUNT = 15
MAXIMUM_ECE_BIN_COUNT = 1_000
DEFAULT_MINIMUM_TEMPERATURE = 0.05
DEFAULT_MAXIMUM_TEMPERATURE = 20.0
MINIMUM_TEMPERATURE = 0.001
MAXIMUM_TEMPERATURE = 1_000.0
DEFAULT_TEMPERATURE_ITERATIONS = 80
MAXIMUM_TEMPERATURE_ITERATIONS = 256
MAXIMUM_HUMAN_VERIFICATION_CASE_COUNT = (
    MAXIMUM_TRAINING_ROW_COUNT + _MAXIMUM_EXTERNAL_HUMAN_VERIFICATION_CASE_COUNT
)
MAXIMUM_VERIFICATION_CASE_COUNT = MAXIMUM_HUMAN_VERIFICATION_CASE_COUNT
MAXIMUM_CALIBRATION_LOGIT_VALUES = 4_000_000
DEFAULT_SEED_ATTEMPTS = 3
MAXIMUM_SEED_ATTEMPTS = 20
DEFAULT_SEED_RETRY_MARGIN = 2.0
MAXIMUM_SEED_RETRY_MARGIN = 10.0

_CALIBRATION_SPLIT_DOMAIN = b"semantscript.calibration-split/v1\0"
_MODEL_STATE_DOMAIN = b"semantscript.model-state/v1\0"
_MAXIMUM_MODEL_STATE_BYTES = 2 * 1024 * 1024 * 1024
_MAXIMUM_MODEL_STATE_TENSORS = 100_000
_MAXIMUM_MODEL_STATE_NAME_BYTES = 8 * 1024 * 1024
_MAXIMUM_TOKENIZER_JSON_BYTES = 64 * 1024 * 1024
_TOKENIZER_JSON_LIMITS = StrictJsonLimits(
    maximum_bytes=_MAXIMUM_TOKENIZER_JSON_BYTES,
    maximum_depth=100,
    maximum_nodes=2_000_000,
)
_FUNCTION_ID = re.compile(r"^nf_[a-f0-9]{64}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_RFC3339_UTC = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z$"
)

type VerificationStatus = Literal["passed", "failed"]


class VerificationError(RuntimeError):
    """Base class for calibration and post-training verification failures."""


class VerificationConfigurationError(VerificationError, ValueError):
    """Verification inputs cannot form the required closed evidence set."""


class VerificationExecutionError(VerificationError):
    """The tokenizer, model, or constraint evaluator failed during measurement."""


class VerificationGateError(VerificationError):
    """Measured evidence failed one or more release gates."""

    def __init__(self, result: VerificationResult) -> None:
        self.result = result
        super().__init__("verification failed: " + "; ".join(result.failures))


@dataclass(frozen=True, slots=True)
class VerificationConfig:
    """Bounded metric, calibration, and release-gate controls."""

    ece_threshold: float = 0.1
    ece_bins: int = DEFAULT_ECE_BIN_COUNT
    batch_size: int = 32
    # Fraction of verification records (corpus rows plus attested cases) whose raw
    # prediction may violate an active constraint before the gate fails. Zero keeps
    # the strict contract; a build that relaxes it must record the value it used.
    maximum_constraint_violation_rate: float = 0.0
    minimum_temperature: float = DEFAULT_MINIMUM_TEMPERATURE
    maximum_temperature: float = DEFAULT_MAXIMUM_TEMPERATURE
    maximum_temperature_iterations: int = DEFAULT_TEMPERATURE_ITERATIONS

    def __post_init__(self) -> None:
        _unit_interval("ece_threshold", self.ece_threshold)
        _unit_interval("maximum_constraint_violation_rate", self.maximum_constraint_violation_rate)
        _bounded_integer("ece_bins", self.ece_bins, minimum=2, maximum=MAXIMUM_ECE_BIN_COUNT)
        _bounded_integer("batch_size", self.batch_size, minimum=1, maximum=MAXIMUM_BATCH_SIZE)
        _positive_finite("minimum_temperature", self.minimum_temperature)
        _positive_finite("maximum_temperature", self.maximum_temperature)
        if (
            self.minimum_temperature < MINIMUM_TEMPERATURE
            or self.maximum_temperature > MAXIMUM_TEMPERATURE
            or not self.minimum_temperature < self.maximum_temperature
        ):
            raise VerificationConfigurationError(
                "temperature bounds must be increasing and within "
                f"[{MINIMUM_TEMPERATURE}, {MAXIMUM_TEMPERATURE}]"
            )
        _bounded_integer(
            "maximum_temperature_iterations",
            self.maximum_temperature_iterations,
            minimum=1,
            maximum=MAXIMUM_TEMPERATURE_ITERATIONS,
        )


@dataclass(frozen=True, slots=True)
class SeedRetryConfig:
    """When a narrowly failed release gate retrains with the next seed.

    ``attempts`` counts every training run, the first included, so 1 turns the
    retry off. ``margin`` bounds "narrowly": a failure retries only when the
    constraint-violation rate is at most ``margin`` times the configured
    tolerance and the ECE at most ``margin`` times the configured threshold, and
    nothing else failed. The retry settings are not part of the build-cache
    recipe: they decide how many seeds a build may try, not what a head is.
    """

    attempts: int = DEFAULT_SEED_ATTEMPTS
    margin: float = DEFAULT_SEED_RETRY_MARGIN

    def __post_init__(self) -> None:
        _bounded_integer("seed attempts", self.attempts, minimum=1, maximum=MAXIMUM_SEED_ATTEMPTS)
        if (
            isinstance(self.margin, bool)
            or not isinstance(self.margin, (int, float))
            or not math.isfinite(float(self.margin))
            or not 1 <= float(self.margin) <= MAXIMUM_SEED_RETRY_MARGIN
        ):
            raise VerificationConfigurationError(
                "seed retry margin must be a finite number from 1 through "
                f"{MAXIMUM_SEED_RETRY_MARGIN:g}"
            )


@dataclass(frozen=True, slots=True)
class SeedRetryDecision:
    """Whether a failed attempt may retrain with the next seed, and why."""

    retry: bool
    reason: str


def seed_retry_decision(
    results: Sequence[VerificationResult],
    config: VerificationConfig,
    retry: SeedRetryConfig,
    /,
) -> SeedRetryDecision:
    """Classify one attempt's failed gates as narrow (retry) or not (stop).

    Only the constraint-violation-rate and ECE gates are seed-sensitive in a way a
    retry can fix: a gold or human example miss, an output type error, a rate
    beyond ``margin`` times the tolerance (any violation under a zero tolerance)
    or an ECE beyond ``margin`` times the threshold stops the build.
    """

    if not isinstance(config, VerificationConfig):
        raise VerificationConfigurationError("config must be a VerificationConfig")
    if not isinstance(retry, SeedRetryConfig):
        raise VerificationConfigurationError("retry must be a SeedRetryConfig")
    failed = [result for result in results if result.status != "passed"]
    if not failed:
        return SeedRetryDecision(False, "verification passed")
    tolerance = config.maximum_constraint_violation_rate
    threshold = config.ece_threshold
    narrow: list[str] = []
    for result in failed:
        metrics = result.metrics
        name = result.function_id
        if metrics.example_failures:
            return SeedRetryDecision(
                False,
                f"{name}: {metrics.example_failures} gold/human example prediction(s) "
                "failed; a gold miss is not a seed effect",
            )
        if metrics.type_errors:
            return SeedRetryDecision(
                False, f"{name}: {metrics.type_errors} output type check(s) failed"
            )
        seen = False
        if metrics.constraint_violations:
            if result.record_count is None:
                return SeedRetryDecision(
                    False, f"{name}: the violation rate's record count is unknown"
                )
            rate = metrics.constraint_violations / result.record_count
            if rate > tolerance:
                seen = True
                if tolerance == 0:
                    return SeedRetryDecision(
                        False,
                        f"{name}: {metrics.constraint_violations} constraint violation(s) "
                        "under a zero tolerance; the margin only widens a nonzero "
                        "--max-constraint-violation-rate",
                    )
                if rate > retry.margin * tolerance:
                    return SeedRetryDecision(
                        False,
                        f"{name}: violation rate {rate:.4%} ({metrics.constraint_violations} "
                        f"of {result.record_count}) is outside the retry margin "
                        f"{retry.margin:g} x {tolerance:.4g} = {retry.margin * tolerance:.4%}",
                    )
                narrow.append(
                    f"{name}: violation rate {rate:.4%} ({metrics.constraint_violations} of "
                    f"{result.record_count}) within {retry.margin:g} x tolerance {tolerance:.4g}"
                )
        if metrics.ece > threshold:
            seen = True
            if metrics.ece > retry.margin * threshold:
                return SeedRetryDecision(
                    False,
                    f"{name}: ECE {metrics.ece:.4f} is outside the retry margin "
                    f"{retry.margin:g} x {threshold:.4g} = {retry.margin * threshold:.4f}",
                )
            narrow.append(
                f"{name}: ECE {metrics.ece:.4f} within {retry.margin:g} x threshold {threshold:.4g}"
            )
        if not seen:
            return SeedRetryDecision(False, f"{name}: failed a gate the retry does not classify")
    return SeedRetryDecision(True, "; ".join(narrow))


@dataclass(frozen=True, slots=True)
class CalibrationRecordV1:
    """Schema-exact per-head temperature-scaling evidence."""

    temperature: float
    ece: float
    brier: float
    sample_count: int
    split_sha256: str
    ece_bins: int

    def __post_init__(self) -> None:
        _positive_finite("temperature", self.temperature)
        _unit_interval("ece", self.ece)
        _unit_interval("brier", self.brier)
        _bounded_integer("sample_count", self.sample_count, minimum=1, maximum=2**63 - 1)
        if not isinstance(self.split_sha256, str) or _SHA256.fullmatch(self.split_sha256) is None:
            raise VerificationConfigurationError(
                "calibration split_sha256 must be 64 lowercase hexadecimal characters"
            )
        _bounded_integer(
            "ece_bins",
            self.ece_bins,
            minimum=2,
            maximum=MAXIMUM_ECE_BIN_COUNT,
        )

    def to_document(self) -> dict[str, JsonValue]:
        return {
            "method": "temperature-scaling",
            "temperature": self.temperature,
            "ece": self.ece,
            "brier": self.brier,
            "sampleCount": self.sample_count,
            "splitSha256": self.split_sha256,
            "eceBins": self.ece_bins,
        }


@dataclass(frozen=True, slots=True)
class HeadVerificationV1:
    """One scalar head's held-out and pair-consistency evidence."""

    output_path: str
    accuracy: float
    pair_consistency: float
    calibration: CalibrationRecordV1

    def __post_init__(self) -> None:
        if not isinstance(self.output_path, str) or (
            self.output_path != "" and not self.output_path.startswith("/")
        ):
            raise VerificationConfigurationError(
                "head output path must be empty or a JSON pointer to an output field"
            )
        _unit_interval("accuracy", self.accuracy)
        _unit_interval("pair_consistency", self.pair_consistency)
        if not isinstance(self.calibration, CalibrationRecordV1):
            raise VerificationConfigurationError("head calibration is invalid")

    def to_ir_document(self) -> dict[str, JsonValue]:
        return {
            "outputPath": self.output_path,
            "accuracy": self.accuracy,
            "pairConsistency": self.pair_consistency,
            "calibration": self.calibration.to_document(),
        }

    def to_manifest_metadata(self) -> dict[str, JsonValue]:
        return {
            "calibration": self.calibration.to_document(),
            "verification": {
                "accuracy": self.accuracy,
                "pairConsistency": self.pair_consistency,
            },
        }


@dataclass(frozen=True, slots=True)
class VerificationMetricsV1:
    """Function-level metrics stored in verified IR."""

    accuracy: float
    ece: float
    brier: float
    pair_consistency: float
    heads: tuple[HeadVerificationV1, ...]
    example_failures: int
    constraint_violations: int
    type_errors: int = 0

    def __post_init__(self) -> None:
        for name in ("accuracy", "ece", "brier", "pair_consistency"):
            _unit_interval(name, getattr(self, name))
        if not isinstance(self.heads, tuple) or not self.heads:
            raise VerificationConfigurationError("verification requires at least one head")
        if any(not isinstance(head, HeadVerificationV1) for head in self.heads):
            raise VerificationConfigurationError("verification heads are invalid")
        if len({head.output_path for head in self.heads}) != len(self.heads):
            raise VerificationConfigurationError("verification head output paths must be unique")
        if len(self.heads) == 1 and self.heads[0].output_path == "":
            head = self.heads[0]
            if (
                self.accuracy != head.accuracy
                or self.ece != head.calibration.ece
                or self.brier != head.calibration.brier
                or self.pair_consistency != head.pair_consistency
            ):
                raise VerificationConfigurationError(
                    "scalar function metrics must equal the sole head metrics"
                )
        else:
            # A flat object is right only when every field is right, and the gate
            # sees the worst head's calibration.
            if any(head.output_path == "" for head in self.heads):
                raise VerificationConfigurationError(
                    "object output heads must all name an output field"
                )
            if (
                self.accuracy > min(head.accuracy for head in self.heads) + 1e-12
                or self.ece != max(head.calibration.ece for head in self.heads)
                or self.brier != max(head.calibration.brier for head in self.heads)
                or self.pair_consistency != min(head.pair_consistency for head in self.heads)
            ):
                raise VerificationConfigurationError(
                    "object function metrics must be the exact-match accuracy, the maximum "
                    "head ECE and Brier, and the minimum head pair consistency"
                )
        for name in ("example_failures", "type_errors"):
            _bounded_integer(
                name,
                getattr(self, name),
                minimum=0,
                maximum=MAXIMUM_VERIFICATION_CASE_COUNT,
            )
        _bounded_integer(
            "constraint_violations",
            self.constraint_violations,
            minimum=0,
            maximum=MAXIMUM_CONSTRAINT_EVALUATION_STEPS,
        )

    def to_ir_document(self) -> dict[str, JsonValue]:
        return {
            "accuracy": self.accuracy,
            "ece": self.ece,
            "brier": self.brier,
            "pairConsistency": self.pair_consistency,
            "heads": [head.to_ir_document() for head in self.heads],
            "exampleFailures": self.example_failures,
            "constraintViolations": self.constraint_violations,
            "typeErrors": self.type_errors,
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Measured verification evidence, including failed-build diagnostics."""

    function_id: str
    semantic_sha256: str
    model_state_sha256: str
    tokenizer_sha256: str
    status: VerificationStatus
    verified_at: str
    metrics: VerificationMetricsV1
    attested_cases: int
    pair_count: int
    failures: tuple[str, ...]
    # Verification records (corpus rows plus external attested cases) the
    # constraint-violation rate is measured over. Measured evidence only: it is
    # not serialized, so a result restored from IR or the build cache has None.
    record_count: int | None = field(default=None, compare=False)
    # One `next:` suggestion per failure, in the same order (diagnostics/remedies.json):
    # derived from the evidence when verification ran, and like record_count not
    # serialized, so a result restored from IR or the build cache has none.
    suggestions: tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.function_id, str)
            or _FUNCTION_ID.fullmatch(self.function_id) is None
        ):
            raise VerificationConfigurationError("verification function_id is invalid")
        if (
            not isinstance(self.semantic_sha256, str)
            or _SHA256.fullmatch(self.semantic_sha256) is None
        ):
            raise VerificationConfigurationError("verification semantic_sha256 is invalid")
        for name in ("model_state_sha256", "tokenizer_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise VerificationConfigurationError(f"verification {name} is invalid")
        if self.status not in ("passed", "failed"):
            raise VerificationConfigurationError("verification status is invalid")
        _validate_verified_at(self.verified_at)
        if not isinstance(self.metrics, VerificationMetricsV1):
            raise VerificationConfigurationError("verification metrics are invalid")
        _bounded_integer(
            "attested_cases",
            self.attested_cases,
            minimum=1,
            maximum=MAXIMUM_HUMAN_VERIFICATION_CASE_COUNT,
        )
        _bounded_integer(
            "pair_count", self.pair_count, minimum=0, maximum=MAXIMUM_TRAINING_ROW_COUNT
        )
        if self.pair_count == 0 and self.metrics.pair_consistency != 1.0:
            raise VerificationConfigurationError(
                "verification without counterfactual pairs must have pair consistency 1"
            )
        if self.pair_count > 0 and not math.isclose(
            self.metrics.pair_consistency * self.pair_count,
            round(self.metrics.pair_consistency * self.pair_count),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise VerificationConfigurationError(
                "pair consistency must be a realizable fraction of pair_count"
            )
        if self.record_count is not None:
            _bounded_integer(
                "record_count",
                self.record_count,
                minimum=1,
                maximum=MAXIMUM_HUMAN_VERIFICATION_CASE_COUNT,
            )
        if not isinstance(self.failures, tuple) or any(
            not isinstance(failure, str) or not failure for failure in self.failures
        ):
            raise VerificationConfigurationError("verification failures are invalid")
        if (self.status == "passed") != (not self.failures):
            raise VerificationConfigurationError("verification status and failures disagree")
        # Constraint violations may remain on a passing result when the gate's
        # configured tolerance admitted their rate; misses and type errors may not.
        if self.status == "passed" and (self.metrics.example_failures or self.metrics.type_errors):
            raise VerificationConfigurationError(
                "passing verification cannot contain example or type failures"
            )
        if (
            not isinstance(self.suggestions, tuple)
            or any(not isinstance(item, str) or not item for item in self.suggestions)
            or len(self.suggestions) not in (0, len(self.failures))
        ):
            raise VerificationConfigurationError(
                "verification suggestions must be empty or one non-empty string per failure"
            )

    def to_ir_document(self) -> dict[str, JsonValue]:
        return {
            "status": self.status,
            "verifiedAt": self.verified_at,
            "metrics": self.metrics.to_ir_document(),
        }

    def to_manifest_head_metadata(self, index: int = 0) -> dict[str, JsonValue]:
        self._require_passed()
        return self.metrics.heads[index].to_manifest_metadata()

    def to_manifest_function_verification(self) -> dict[str, JsonValue]:
        self._require_passed()
        return {
            "status": "passed",
            "accuracy": self.metrics.accuracy,
            "ece": self.metrics.ece,
            "brier": self.metrics.brier,
            "pairConsistency": self.metrics.pair_consistency,
            "attestedCases": self.attested_cases,
            "exampleFailures": self.metrics.example_failures,
            "constraintViolations": self.metrics.constraint_violations,
            "typeErrors": self.metrics.type_errors,
        }

    def _require_passed(self) -> None:
        if self.status != "passed":
            raise VerificationGateError(self)


@dataclass(frozen=True, slots=True)
class _CaseRecord:
    case_id: str
    inputs: dict[str, JsonValue]
    label_indices: tuple[int, ...]
    human_authored: bool

    @property
    def label_index(self) -> int:
        return self.label_indices[0]


def evaluate_training_result(
    ir: NeuralFunctionIr,
    training: TrainingResult,
    base: TrainingDataset,
    adversarial: AdversarialDataset | None = None,
    /,
    *,
    tokenizer: Any | None = None,
    attested_verification: Sequence[GeneratedCase] = (),
    config: VerificationConfig | None = None,
    verified_at: str | None = None,
) -> VerificationResult:
    """Measure calibration and all release gates without discarding failed evidence."""

    resolved = VerificationConfig() if config is None else config
    if not isinstance(resolved, VerificationConfig):
        raise VerificationConfigurationError("config must be a VerificationConfig")
    if not isinstance(training, TrainingResult):
        raise VerificationConfigurationError("training must be a TrainingResult")
    if not isinstance(training.config, TrainingConfig):
        raise VerificationConfigurationError("training result config is invalid")
    if resolved.batch_size * training.config.maximum_sequence_length > MAXIMUM_BATCH_TOKENS:
        raise VerificationConfigurationError(
            "verification batch_size * maximum_sequence_length exceeds maximum token count "
            f"{MAXIMUM_BATCH_TOKENS}"
        )
    corpus = _reconstruct_corpus(ir, training, base, adversarial)
    heads = corpus.output_heads
    if training.heads != heads:
        raise VerificationConfigurationError("training result heads do not match the corpus")
    if resolved.batch_size * corpus.logit_count > MAXIMUM_CALIBRATION_LOGIT_VALUES:
        raise VerificationConfigurationError(
            "verification batch_size * logit_count exceeds maximum value count "
            f"{MAXIMUM_CALIBRATION_LOGIT_VALUES}"
        )
    external_human = _validate_attested_cases(
        ir, corpus, attested_verification, training.config.canonical_input_version
    )
    gold_rows = tuple(row for row in corpus.rows if row.origin == "gold")
    human_count = len(gold_rows) + len(external_human)
    if human_count < 1:
        raise VerificationConfigurationError(
            "verification requires at least one attested gold or external case"
        )
    if human_count > MAXIMUM_HUMAN_VERIFICATION_CASE_COUNT:
        raise VerificationConfigurationError(
            f"attested verification cases exceed maximum {MAXIMUM_HUMAN_VERIFICATION_CASE_COUNT}"
        )
    _validate_source_examples(ir, base, gold_rows, heads, training.config.canonical_input_version)

    calibration_rows = training.split.evaluation
    if not calibration_rows:
        raise VerificationConfigurationError(
            "temperature fitting requires held-out calibration rows"
        )
    if any(row.origin == "gold" for row in calibration_rows):
        raise VerificationConfigurationError("gold examples cannot be calibration-only rows")
    if len(calibration_rows) * corpus.logit_count > MAXIMUM_CALIBRATION_LOGIT_VALUES:
        raise VerificationConfigurationError(
            f"calibration logits exceed maximum value count {MAXIMUM_CALIBRATION_LOGIT_VALUES}"
        )

    records = _case_records(corpus, external_human)
    resolved_tokenizer = _load_tokenizer(training) if tokenizer is None else tokenizer
    tokenizer_sha256 = hashlib.sha256(tokenizer_json_bytes(resolved_tokenizer)).hexdigest()
    model_sha256 = model_state_sha256(training.model)
    predictions, calibration_logits = _collect_predictions(
        ir,
        training,
        records,
        calibration_rows,
        resolved_tokenizer,
        resolved,
    )
    torch = _require_torch()
    targets = torch.tensor(
        [list(row.label_indices) for row in calibration_rows],
        dtype=torch.long,
    )
    split_sha256 = calibration_split_sha256(training, calibration_rows)
    example_failures, example_details = _example_failures(
        gold_rows, external_human, predictions, heads
    )
    constraint_violations, constraint_details, violation_evidence = _constraint_violations(
        ir,
        corpus,
        adversarial,
        records,
        predictions,
    )
    pair_consistencies, pair_count = _pair_consistency(corpus, adversarial, predictions)
    head_records: list[HeadVerificationV1] = []
    for index, (head, (start, end)) in enumerate(zip(heads, head_slices(heads), strict=True)):
        try:
            calibration = _calibrate(
                calibration_logits[:, start:end],
                targets[:, index],
                bin_count=resolved.ece_bins,
                minimum_temperature=resolved.minimum_temperature,
                maximum_temperature=resolved.maximum_temperature,
                maximum_iterations=resolved.maximum_temperature_iterations,
            )
        except (RuntimeError, TypeError, ValueError) as error:
            raise VerificationExecutionError(
                f"temperature calibration failed for head {head.output_path!r}: {error}"
            ) from error
        calibrated = calibration.calibrated_metrics
        head_records.append(
            HeadVerificationV1(
                output_path=head.output_path,
                accuracy=calibrated.accuracy,
                pair_consistency=pair_consistencies[index],
                calibration=_calibration_record(calibration, split_sha256),
            )
        )
    if len(head_records) == 1 and head_records[0].output_path == "":
        sole = head_records[0]
        metrics = VerificationMetricsV1(
            accuracy=sole.accuracy,
            ece=sole.calibration.ece,
            brier=sole.calibration.brier,
            pair_consistency=sole.pair_consistency,
            heads=(sole,),
            example_failures=example_failures,
            constraint_violations=constraint_violations,
            type_errors=0,
        )
    else:
        exact = sum(predictions[row.row_id] == row.label_indices for row in calibration_rows) / len(
            calibration_rows
        )
        metrics = VerificationMetricsV1(
            accuracy=exact,
            ece=max(record.calibration.ece for record in head_records),
            brier=max(record.calibration.brier for record in head_records),
            pair_consistency=min(record.pair_consistency for record in head_records),
            heads=tuple(head_records),
            example_failures=example_failures,
            constraint_violations=constraint_violations,
            type_errors=0,
        )
    if model_state_sha256(training.model) != model_sha256:
        raise VerificationExecutionError("classifier state changed during verification")
    if hashlib.sha256(tokenizer_json_bytes(resolved_tokenizer)).hexdigest() != tokenizer_sha256:
        raise VerificationExecutionError("tokenizer JSON changed during verification")
    gates = _failed_gates(metrics, resolved, len(records), example_details, constraint_details)
    failures = tuple(text for _gate, text in gates)
    suggestions: tuple[str, ...] = ()
    if gates:
        suggestions = _gate_suggestions(
            [gate for gate, _text in gates],
            ir=ir,
            heads=heads,
            corpus=corpus,
            external_human=external_human,
            predictions=predictions,
            violations=violation_evidence,
            metrics=metrics,
            calibration_rows=len(calibration_rows),
            current_cases=len(base.cases),
            epochs=training.config.epochs,
        )
    return VerificationResult(
        function_id=training.function_id,
        semantic_sha256=training.semantic_sha256,
        model_state_sha256=model_sha256,
        tokenizer_sha256=tokenizer_sha256,
        status="passed" if not failures else "failed",
        verified_at=_resolved_verified_at(verified_at),
        metrics=metrics,
        attested_cases=human_count,
        pair_count=pair_count,
        failures=failures,
        record_count=len(records),
        suggestions=suggestions,
    )


def verify_training_result(
    ir: NeuralFunctionIr,
    training: TrainingResult,
    base: TrainingDataset,
    adversarial: AdversarialDataset | None = None,
    /,
    **kwargs: Any,
) -> VerificationResult:
    """Build-facing entry point: return passing evidence or raise with the failed report."""

    result = evaluate_training_result(ir, training, base, adversarial, **kwargs)
    return require_passing_verification(result)


def require_passing_verification(result: VerificationResult, /) -> VerificationResult:
    """Reject failed evidence while retaining it on the typed exception."""

    if not isinstance(result, VerificationResult):
        raise VerificationConfigurationError("result must be a VerificationResult")
    if result.status != "passed":
        raise VerificationGateError(result)
    return result


def calibration_split_sha256(
    training: TrainingResult,
    rows: Sequence[TrainingRow],
    /,
) -> str:
    """Bind the ordered held-out row identities to datasets and function semantics."""

    if not isinstance(training, TrainingResult):
        raise VerificationConfigurationError("training must be a TrainingResult")
    if not isinstance(rows, Sequence) or not rows:
        raise VerificationConfigurationError("calibration split rows must be nonempty")
    if len(rows) > MAXIMUM_TRAINING_ROW_COUNT:
        raise VerificationConfigurationError(
            f"calibration split exceeds maximum row count {MAXIMUM_TRAINING_ROW_COUNT}"
        )
    entries: list[dict[str, JsonValue]] = []
    for row in rows:
        if not isinstance(row, TrainingRow):
            raise VerificationConfigurationError("calibration split contains an invalid row")
        entry: dict[str, JsonValue] = {
            "rowId": row.row_id,
            "groupId": row.group_id,
            "labelIndex": row.label_index,
        }
        if len(row.label_indices) > 1:
            entry["labelIndices"] = list(row.label_indices)
        entries.append(entry)
    document: dict[str, JsonValue] = {
        "functionId": training.function_id,
        "semanticSha256": training.semantic_sha256,
        "baseDatasetSha256": training.base_dataset_sha256,
        "adversarialDatasetSha256": training.adversarial_dataset_sha256,
        "rows": cast(JsonValue, entries),
    }
    encoded = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8", errors="strict")
    return hashlib.sha256(_CALIBRATION_SPLIT_DOMAIN + encoded).hexdigest()


def model_state_sha256(model: Any, /) -> str:
    """Hash a model's sorted tensor state without pickle or device-dependent metadata."""

    torch = _require_torch()
    try:
        state_dict_method = getattr(model, "state_dict", None)
        if not callable(state_dict_method):
            raise TypeError("model does not implement state_dict()")
        state = state_dict_method()
        if not isinstance(state, Mapping):
            raise TypeError("model state_dict() must return a mapping")
        if len(state) > _MAXIMUM_MODEL_STATE_TENSORS:
            raise ValueError(
                f"model state exceeds maximum tensor count {_MAXIMUM_MODEL_STATE_TENSORS}"
            )

        names = list(state)
        if any(not isinstance(name, str) for name in names):
            raise TypeError("model state tensor names must be strings")
        encoded_names = {name: name.encode("utf-8", errors="strict") for name in names}
        if sum(len(name) for name in encoded_names.values()) > _MAXIMUM_MODEL_STATE_NAME_BYTES:
            raise ValueError(
                f"model state names exceed maximum {_MAXIMUM_MODEL_STATE_NAME_BYTES} bytes"
            )

        digest = hashlib.sha256()
        digest.update(_MODEL_STATE_DOMAIN)
        digest.update(len(names).to_bytes(8, byteorder="big", signed=False))
        total_bytes = 0
        for name in sorted(names):
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"model state entry {name!r} is not a tensor")
            if tensor.layout != torch.strided or tensor.device.type == "meta":
                raise TypeError(f"model state tensor {name!r} must be dense and materialized")

            dtype = str(tensor.dtype).encode("ascii", errors="strict")
            shape = tuple(int(dimension) for dimension in tensor.shape)
            if any(dimension < 0 or dimension >= 2**64 for dimension in shape):
                raise ValueError(f"model state tensor {name!r} has an invalid shape")
            byte_count = tensor.numel() * tensor.element_size()
            total_bytes += byte_count
            if total_bytes > _MAXIMUM_MODEL_STATE_BYTES:
                raise ValueError(f"model state exceeds maximum {_MAXIMUM_MODEL_STATE_BYTES} bytes")

            _update_framed_hash(digest, encoded_names[name])
            _update_framed_hash(digest, dtype)
            digest.update(len(shape).to_bytes(8, byteorder="big", signed=False))
            for dimension in shape:
                digest.update(dimension.to_bytes(8, byteorder="big", signed=False))
            digest.update(byte_count.to_bytes(8, byteorder="big", signed=False))

            contiguous = (
                tensor.detach()
                .resolve_conj()
                .resolve_neg()
                .to(device="cpu")
                .contiguous()
                .reshape(-1)
            )
            raw = contiguous.view(torch.uint8).numpy()
            if raw.nbytes != byte_count:
                raise ValueError(f"model state tensor {name!r} has an inconsistent byte length")
            digest.update(memoryview(raw))
        return digest.hexdigest()
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationExecutionError(
            f"classifier state could not be fingerprinted: {error}"
        ) from error


def tokenizer_json_bytes(tokenizer: Any, /) -> bytes:
    """Return exact tokenizer JSON used by verification for later artifact binding.

    Hugging Face fast tokenizers expose ``backend_tokenizer.to_str``. Tests and
    custom tokenizers may instead expose immutable ``semantscript_tokenizer_json``
    bytes; no broader implicit serialization protocol is accepted.
    """

    missing = object()
    try:
        custom = getattr(tokenizer, "semantscript_tokenizer_json", missing)
        if custom is not missing:
            if not isinstance(custom, bytes):
                raise TypeError("semantscript_tokenizer_json must be bytes")
            document = custom
        else:
            backend = getattr(tokenizer, "backend_tokenizer", None)
            to_str = getattr(backend, "to_str", None)
            if not callable(to_str):
                raise TypeError(
                    "tokenizer must expose backend_tokenizer.to_str(pretty=False) or "
                    "semantscript_tokenizer_json bytes"
                )
            serialized = to_str(pretty=False)
            if not isinstance(serialized, str):
                raise TypeError("backend_tokenizer.to_str() must return a string")
            document = _normalize_fast_tokenizer_json(serialized)

        if not document:
            raise ValueError("tokenizer JSON must not be empty")
        parsed = loads_strict_json(document, limits=_TOKENIZER_JSON_LIMITS)
        if not isinstance(parsed, dict):
            raise ValueError("tokenizer JSON root must be an object")
        return document
    except VerificationError:
        raise
    except (StrictJsonError, TypeError, ValueError, UnicodeError) as error:
        raise VerificationExecutionError(
            f"tokenizer JSON could not be serialized safely: {error}"
        ) from error
    except Exception as error:
        raise VerificationExecutionError(f"tokenizer JSON serialization failed: {error}") from error


def _normalize_fast_tokenizer_json(serialized: str) -> bytes:
    """Serialize a fast tokenizer without its per-call padding and truncation state.

    ``backend_tokenizer.to_str`` embeds the runtime padding and truncation
    configuration, which every ``tokenizer(...)`` call may rewrite. Those fields
    do not change the vocabulary, normalizer, pre-tokenizer or post-processor, so
    they are cleared before hashing; the standard JSON module is used so integer
    token ids stay integers.
    """

    parsed = json.loads(serialized)
    if not isinstance(parsed, dict):
        raise ValueError("tokenizer JSON root must be an object")
    parsed["padding"] = None
    parsed["truncation"] = None
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8", errors="strict"
    )


def _update_framed_hash(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def _reconstruct_corpus(
    ir: NeuralFunctionIr,
    training: TrainingResult,
    base: TrainingDataset,
    adversarial: AdversarialDataset | None,
) -> TrainingCorpus:
    if training.function_id != base.function_id or training.semantic_sha256 != base.semantic_sha256:
        raise VerificationConfigurationError(
            "training result identity does not match the base dataset"
        )
    if training.base_dataset_sha256 != base.dataset_sha256:
        raise VerificationConfigurationError("training result base dataset digest does not match")
    supplied_adversarial_digest = None if adversarial is None else adversarial.dataset_sha256
    if training.adversarial_dataset_sha256 != supplied_adversarial_digest:
        raise VerificationConfigurationError(
            "training result adversarial dataset digest does not match"
        )
    try:
        corpus = assemble_training_corpus(ir, base, adversarial)
    except (RuntimeError, TypeError, ValueError) as error:
        raise VerificationConfigurationError(
            f"could not reconstruct training corpus: {error}"
        ) from error
    if corpus.head != training.head:
        raise VerificationConfigurationError("training result head does not match reconstructed IR")
    try:
        expected_split = split_training_corpus(
            corpus,
            HeldOutSplitConfig(
                evaluation_ratio=training.config.evaluation_ratio,
                seed=training.config.seed,
            ),
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise VerificationConfigurationError(
            f"could not reconstruct the deterministic training split: {error}"
        ) from error
    if training.split != expected_split:
        raise VerificationConfigurationError(
            "training split does not match the deterministic corpus split"
        )
    constraints = _constraints_count(ir)
    if constraints and adversarial is None:
        raise VerificationConfigurationError(
            "functions with constraints require an adversarial verification sidecar"
        )
    return corpus


def _validate_attested_cases(
    ir: NeuralFunctionIr,
    corpus: TrainingCorpus,
    cases: Sequence[GeneratedCase],
    version: int = 1,
) -> tuple[_CaseRecord, ...]:
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise VerificationConfigurationError("attested_verification must be a sequence of cases")
    if len(cases) > _MAXIMUM_EXTERNAL_HUMAN_VERIFICATION_CASE_COUNT:
        raise VerificationConfigurationError(
            "human verification cases exceed maximum "
            f"{_MAXIMUM_EXTERNAL_HUMAN_VERIFICATION_CASE_COUNT}"
        )
    schema = ir.get("inputs")
    if not isinstance(schema, list):
        raise VerificationConfigurationError("IR inputs must be an array")
    training_inputs: set[bytes] = set()
    total_bytes = 0
    for row in corpus.rows:
        encoded = _canonical_input(schema, row.inputs, row.row_id, version)
        total_bytes += len(encoded)
        if total_bytes > MAXIMUM_CORPUS_TEXT_BYTES:
            raise VerificationConfigurationError(
                f"verification canonical text exceeds maximum {MAXIMUM_CORPUS_TEXT_BYTES} bytes"
            )
        training_inputs.add(encoded)
    observed: set[bytes] = set()
    result: list[_CaseRecord] = []
    for index, case in enumerate(cases):
        if not isinstance(case, GeneratedCase):
            raise VerificationConfigurationError(
                "attested_verification must contain GeneratedCase values"
            )
        try:
            validate_case(ir, case)
            label_indices = output_labels(corpus.output_heads, case.output)
        except (RuntimeError, TypeError, ValueError) as error:
            raise VerificationConfigurationError(
                f"human verification case {index} is invalid: {error}"
            ) from error
        encoded = _canonical_input(schema, case.inputs, f"human:{index}", version)
        total_bytes += len(encoded)
        if total_bytes > MAXIMUM_CORPUS_TEXT_BYTES:
            raise VerificationConfigurationError(
                f"verification canonical text exceeds maximum {MAXIMUM_CORPUS_TEXT_BYTES} bytes"
            )
        if encoded in training_inputs:
            raise VerificationConfigurationError(
                f"human verification case {index} duplicates a training input"
            )
        if encoded in observed:
            raise VerificationConfigurationError(
                f"human verification case {index} duplicates another human case"
            )
        observed.add(encoded)
        result.append(
            _CaseRecord(
                case_id=f"human:{index}",
                inputs=deepcopy(case.inputs),
                label_indices=label_indices,
                human_authored=True,
            )
        )
    return tuple(result)


def _validate_source_examples(
    ir: NeuralFunctionIr,
    base: TrainingDataset,
    gold_rows: tuple[TrainingRow, ...],
    heads: tuple[OutputHead, ...],
    version: int = 1,
) -> None:
    definition = ir.get("definition")
    examples = definition.get("examples") if isinstance(definition, Mapping) else None
    schema = ir.get("inputs")
    if not isinstance(examples, list) or not isinstance(schema, list):
        raise VerificationConfigurationError("IR definition examples and inputs must be arrays")
    if len(examples) != base.gold_count or len(gold_rows) != base.gold_count:
        raise VerificationConfigurationError("base dataset does not contain every IR gold example")
    for index, raw in enumerate(examples):
        if not isinstance(raw, Mapping) or set(raw) != {"inputs", "output"}:
            raise VerificationConfigurationError(f"IR example {index} is malformed")
        inputs = raw.get("inputs")
        if not isinstance(inputs, dict):
            raise VerificationConfigurationError(f"IR example {index} inputs are invalid")
        row = gold_rows[index]
        if _canonical_input(schema, inputs, f"example:{index}", version) != _canonical_input(
            schema, row.inputs, row.row_id, version
        ):
            raise VerificationConfigurationError(
                f"base gold row {index} does not match its IR example"
            )
        try:
            if row.label_indices != output_labels(heads, cast(JsonValue, raw.get("output"))):
                raise VerificationConfigurationError(
                    f"base gold row {index} output does not match its IR example"
                )
            if row.label_indices != output_labels(heads, base.cases[index].output):
                raise VerificationConfigurationError(
                    f"base gold row {index} output does not match the base dataset"
                )
        except (RuntimeError, TypeError, ValueError) as error:
            if isinstance(error, VerificationConfigurationError):
                raise
            raise VerificationConfigurationError(
                f"IR example {index} output is invalid: {error}"
            ) from error


def _case_records(
    corpus: TrainingCorpus,
    external_human: tuple[_CaseRecord, ...],
) -> tuple[_CaseRecord, ...]:
    records = [
        _CaseRecord(
            case_id=row.row_id,
            inputs=row.inputs,
            label_indices=row.label_indices,
            human_authored=row.origin == "gold",
        )
        for row in corpus.rows
    ]
    records.extend(external_human)
    if len(records) > MAXIMUM_VERIFICATION_CASE_COUNT:
        raise VerificationConfigurationError(
            f"verification cases exceed maximum {MAXIMUM_VERIFICATION_CASE_COUNT}"
        )
    return tuple(records)


def _collect_predictions(
    ir: NeuralFunctionIr,
    training: TrainingResult,
    records: tuple[_CaseRecord, ...],
    calibration_rows: tuple[TrainingRow, ...],
    tokenizer: Any | None,
    config: VerificationConfig,
) -> tuple[dict[str, tuple[int, ...]], Any]:
    torch = _require_torch()
    schema = ir.get("inputs")
    if not isinstance(schema, list):
        raise VerificationConfigurationError("IR inputs must be an array")
    heads = training.heads
    text_entries: dict[str, list[str]] = {}
    byte_count = 0
    for record in records:
        encoded = _canonical_input(
            schema, record.inputs, record.case_id, training.config.canonical_input_version
        )
        byte_count += len(encoded)
        if byte_count > MAXIMUM_CORPUS_TEXT_BYTES:
            raise VerificationConfigurationError(
                f"verification canonical text exceeds maximum {MAXIMUM_CORPUS_TEXT_BYTES} bytes"
            )
        text_entries.setdefault(encoded.decode("utf-8", errors="strict"), []).append(record.case_id)

    calibration_ids = {row.row_id for row in calibration_rows}
    calibration_by_id: dict[str, Any] = {}
    predictions: dict[str, tuple[int, ...]] = {}
    resolved_tokenizer = _load_tokenizer(training) if tokenizer is None else tokenizer
    model = training.model
    if not hasattr(model, "eval") or not callable(model.eval):
        raise VerificationConfigurationError("training model does not implement eval()")
    try:
        device = torch.device(training.device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise VerificationConfigurationError(
            f"training result device is invalid: {error}"
        ) from error
    unique = tuple(text_entries.items())
    try:
        model.eval()
    except Exception as error:
        raise VerificationExecutionError(
            f"classifier could not enter evaluation mode: {error}"
        ) from error
    with torch.no_grad():
        for offset in range(0, len(unique), config.batch_size):
            batch = unique[offset : offset + config.batch_size]
            texts = [text for text, _ in batch]
            input_ids, attention_mask = _tokenize(
                texts,
                resolved_tokenizer,
                torch,
                device,
                training.config.maximum_sequence_length,
            )
            try:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception as error:
                raise VerificationExecutionError(
                    f"classifier verification forward pass failed: {error}"
                ) from error
            try:
                _validate_logits(logits, torch, len(batch), training.logit_count)
                cpu_logits = logits.detach().to(device="cpu", dtype=torch.float64)
                batch_predictions = predict_indices(cpu_logits, heads, torch)
                for index, (_, case_ids) in enumerate(batch):
                    prediction = tuple(int(v) for v in batch_predictions[index].tolist())
                    for case_id in case_ids:
                        predictions[case_id] = prediction
                        if case_id in calibration_ids:
                            calibration_by_id[case_id] = cpu_logits[index].clone()
            except VerificationExecutionError:
                raise
            except Exception as error:
                raise VerificationExecutionError(
                    f"classifier logits could not be materialized: {error}"
                ) from error
    if set(calibration_by_id) != calibration_ids:
        raise VerificationExecutionError("not every calibration row produced logits")
    try:
        calibration_logits = torch.stack(
            [calibration_by_id[row.row_id] for row in calibration_rows],
            dim=0,
        )
    except Exception as error:
        raise VerificationExecutionError(
            f"calibration logits could not be assembled: {error}"
        ) from error
    return predictions, calibration_logits


def _load_tokenizer(training: TrainingResult) -> Any:
    try:
        from transformers import AutoTokenizer
    except (ImportError, OSError) as error:
        raise VerificationExecutionError(
            "verification requires Transformers or an injected tokenizer"
        ) from error
    try:
        return AutoTokenizer.from_pretrained(
            training.config.encoder_name,
            revision=training.config.encoder_revision,
            local_files_only=training.config.local_files_only,
            trust_remote_code=False,
            use_fast=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise VerificationExecutionError(
            f"could not load the configured tokenizer: {error}"
        ) from error


def _tokenize(
    texts: list[str],
    tokenizer: Any,
    torch: Any,
    device: Any,
    maximum_sequence_length: int,
) -> tuple[Any, Any]:
    try:
        tokenized = tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=maximum_sequence_length,
            return_tensors="pt",
        )
    except Exception as error:
        raise VerificationExecutionError(f"verification tokenization failed: {error}") from error
    try:
        if not isinstance(tokenized, Mapping):
            raise VerificationExecutionError("tokenizer must return a mapping of tensors")
        input_ids = tokenized.get("input_ids")
        attention_mask = tokenized.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            raise VerificationExecutionError(
                "tokenizer must return input_ids and attention_mask tensors"
            )
        if (
            input_ids.ndim != 2
            or input_ids.shape != attention_mask.shape
            or input_ids.shape[0] != len(texts)
            or input_ids.shape[1] == 0
            or input_ids.shape[1] > maximum_sequence_length
        ):
            raise VerificationExecutionError(
                "tokenizer tensors must have bounded matching [batch, sequence] shapes"
            )
        if input_ids.dtype != torch.long or attention_mask.dtype != torch.long:
            raise VerificationExecutionError(
                "tokenizer input_ids and attention_mask must use int64"
            )
        if bool(((attention_mask != 0) & (attention_mask != 1)).any().item()):
            raise VerificationExecutionError(
                "tokenizer attention_mask must contain only zero and one"
            )
        if bool((attention_mask.sum(dim=1) == 0).any().item()):
            raise VerificationExecutionError(
                "every tokenized row must contain at least one attended token"
            )
        return input_ids.to(device), attention_mask.to(device)
    except VerificationExecutionError:
        raise
    except Exception as error:
        raise VerificationExecutionError(
            f"tokenizer tensors could not be validated or moved to {device}: {error}"
        ) from error


def _validate_logits(logits: Any, torch: Any, batch_size: int, logit_count: int) -> None:
    if not isinstance(logits, torch.Tensor):
        raise VerificationExecutionError("classifier must return a tensor of logits")
    if logits.ndim != 2 or logits.shape != (batch_size, logit_count):
        raise VerificationExecutionError(
            f"classifier logits must have shape [{batch_size}, {logit_count}]"
        )
    if not torch.is_floating_point(logits):
        raise VerificationExecutionError("classifier logits must use a floating-point dtype")
    if not bool(torch.isfinite(logits).all().item()):
        raise VerificationExecutionError("classifier logits contain non-finite values")


def _calibration_record(result: Any, split_sha256: str) -> CalibrationRecordV1:
    metrics = result.calibrated_metrics
    return CalibrationRecordV1(
        temperature=result.temperature,
        ece=metrics.ece,
        brier=metrics.brier_score,
        sample_count=metrics.sample_count,
        split_sha256=split_sha256,
        ece_bins=metrics.bin_count,
    )


def _example_failures(
    gold_rows: tuple[TrainingRow, ...],
    external_human: tuple[_CaseRecord, ...],
    predictions: Mapping[str, tuple[int, ...]],
    heads: Sequence[OutputHead],
) -> tuple[int, tuple[str, ...]]:
    """Count missed attested examples and describe each one for the failure report."""

    details: list[str] = []
    for row in gold_rows:
        if predictions[row.row_id] != row.label_indices:
            details.append(
                _miss_detail(
                    "gold example",
                    row.row_id,
                    row.inputs,
                    row.label_indices,
                    predictions[row.row_id],
                    heads,
                )
            )
    for record in external_human:
        if predictions[record.case_id] != record.label_indices:
            details.append(
                _miss_detail(
                    "attested case",
                    record.case_id,
                    record.inputs,
                    record.label_indices,
                    predictions[record.case_id],
                    heads,
                )
            )
    return len(details), tuple(details)


def _miss_detail(
    kind: str,
    case_id: str,
    inputs: Mapping[str, JsonValue],
    expected: Sequence[int],
    predicted: Sequence[int],
    heads: Sequence[OutputHead],
) -> str:
    return (
        f"{kind} {case_id}: inputs {_brief_json(inputs)} expected "
        f"{_brief_json(predicted_output_value(heads, expected))}, predicted "
        f"{_brief_json(predicted_output_value(heads, predicted))}"
    )


_MAXIMUM_DETAIL_CHARACTERS = 400
_MAXIMUM_DETAILS_PER_FAILURE = 10


def _brief_json(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(text) > _MAXIMUM_DETAIL_CHARACTERS:
        return text[: _MAXIMUM_DETAIL_CHARACTERS - 1] + "…"
    return text


def _constraint_violations(
    ir: NeuralFunctionIr,
    corpus: TrainingCorpus,
    adversarial: AdversarialDataset | None,
    records: tuple[_CaseRecord, ...],
    predictions: Mapping[str, tuple[int, ...]],
) -> tuple[int, tuple[str, ...], tuple[Violation, ...]]:
    """Count constraint violations under the raw model and describe each one."""

    try:
        constraints = compile_constraints(ir)
    except (RuntimeError, TypeError, ValueError) as error:
        raise VerificationConfigurationError(f"IR constraints are invalid: {error}") from error
    if len(constraints) == 0:
        return 0, (), ()
    definition = ir.get("definition")
    raw_constraints = definition.get("constraints") if isinstance(definition, Mapping) else None
    sources = [
        str(item.get("source", "")) if isinstance(item, Mapping) else ""
        for item in (raw_constraints if isinstance(raw_constraints, list) else [])
    ]
    details: list[str] = []
    evidence: list[Violation] = []
    try:
        constraints.ensure_evaluation_budget(len(records))
    except ConstraintConfigurationError as error:
        raise VerificationConfigurationError(
            f"adversarial verification exceeds the constraint evaluation budget: {error}"
        ) from error
    budget = ConstraintEvaluationBudget()
    violations = 0
    adversarial_by_id = (
        {} if adversarial is None else {case.case_id: case for case in adversarial.cases}
    )
    for record in records:
        predicted_output = predicted_output_value(corpus.output_heads, predictions[record.case_id])
        try:
            active, case_violations = constraints.evaluate_output_contract(
                record.inputs,
                predicted_output,
                budget=budget,
            )
        except ConstraintEvaluationError as error:
            raise VerificationExecutionError(
                f"constraint evaluation failed for case {record.case_id}: {error}"
            ) from error
        case = adversarial_by_id.get(record.case_id)
        if case is not None and case.tag == "constraint-boundary":
            assert case.constraint_index is not None
            if (case.constraint_index in active) is not case.predicate_result:
                raise VerificationConfigurationError(
                    f"constraint-boundary case {case.case_id} predicate metadata is stale"
                )
        violations += len(case_violations)
        for index in case_violations:
            source = sources[index] if index < len(sources) else ""
            details.append(
                f"constraint {index}{f' ({source})' if source else ''} violated by case "
                f"{record.case_id}: inputs {_brief_json(record.inputs)} predicted "
                f"{_brief_json(predicted_output)}"
            )
            evidence.append(
                Violation(
                    index,
                    source,
                    LabelledCase(
                        record.case_id,
                        record.inputs,
                        predicted_output_value(corpus.output_heads, record.label_indices),
                        predicted_output,
                        not record.human_authored,
                    ),
                )
            )
    return violations, tuple(details), tuple(evidence)


def _pair_consistency(
    corpus: TrainingCorpus,
    adversarial: AdversarialDataset | None,
    predictions: Mapping[str, tuple[int, ...]],
) -> tuple[tuple[float, ...], int]:
    """Per-head pair consistency: both members of a pair right on that head."""

    head_count = len(corpus.output_heads)
    if adversarial is None or not adversarial.pairs:
        return tuple(1.0 for _ in range(head_count)), 0
    by_id = {row.row_id: row for row in corpus.rows}
    passed = [0] * head_count
    for pair in adversarial.pairs:
        anchor = by_id[pair.anchor_case_id]
        twin = by_id[pair.twin_case_id]
        for index in range(head_count):
            if (
                predictions[anchor.row_id][index] == anchor.label_indices[index]
                and predictions[twin.row_id][index] == twin.label_indices[index]
            ):
                passed[index] += 1
    return tuple(count / len(adversarial.pairs) for count in passed), len(adversarial.pairs)


def predicted_output_value(heads: Sequence[OutputHead], indices: Sequence[int]) -> JsonValue:
    """The output value a prediction denotes: a support member or a flat object of them."""

    if len(heads) == 1 and heads[0].output_path == "":
        return heads[0].contract.support[indices[0]]
    value: dict[str, JsonValue] = {}
    for head, index in zip(heads, indices, strict=True):
        name = head.field_name
        if name is None:
            raise VerificationConfigurationError("object output head has no field name")
        value[name] = head.contract.support[index]
    return value


type _Gate = Literal["examples", "constraints", "types", "ece"]


def _gate_failures(
    metrics: VerificationMetricsV1,
    config: VerificationConfig,
    record_count: int,
    example_details: Sequence[str] = (),
    constraint_details: Sequence[str] = (),
) -> tuple[str, ...]:
    return tuple(
        text
        for _gate, text in _failed_gates(
            metrics, config, record_count, example_details, constraint_details
        )
    )


def _failed_gates(
    metrics: VerificationMetricsV1,
    config: VerificationConfig,
    record_count: int,
    example_details: Sequence[str] = (),
    constraint_details: Sequence[str] = (),
) -> list[tuple[_Gate, str]]:
    failures: list[tuple[_Gate, str]] = []
    if metrics.example_failures:
        failures.append(
            (
                "examples",
                f"{metrics.example_failures} gold/human example prediction(s) failed"
                + _detail_suffix(example_details),
            )
        )
    if metrics.constraint_violations:
        rate = metrics.constraint_violations / max(record_count, 1)
        if rate > config.maximum_constraint_violation_rate:
            failures.append(
                (
                    "constraints",
                    f"{metrics.constraint_violations} adversarial constraint check(s) failed "
                    f"({rate:.6g} of {record_count} records exceeds the configured tolerance "
                    f"{config.maximum_constraint_violation_rate:.6g})"
                    + _detail_suffix(constraint_details),
                )
            )
    if metrics.type_errors:
        failures.append(("types", f"{metrics.type_errors} output type check(s) failed"))
    if metrics.ece > config.ece_threshold:
        failures.append(
            (
                "ece",
                f"ECE {metrics.ece:.12g} exceeds configured threshold {config.ece_threshold:.12g}",
            )
        )
    return failures


def _gate_suggestions(
    gates: Sequence[_Gate],
    *,
    ir: NeuralFunctionIr,
    heads: Sequence[OutputHead],
    corpus: TrainingCorpus,
    external_human: Sequence[_CaseRecord],
    predictions: Mapping[str, tuple[int, ...]],
    violations: Sequence[Violation],
    metrics: VerificationMetricsV1,
    calibration_rows: int,
    current_cases: int,
    epochs: int,
) -> tuple[str, ...]:
    """One `next:` suggestion per failed gate, derived from the measured evidence."""

    def labelled(case_id: str, inputs: Any, labels: Sequence[int], teacher: bool) -> LabelledCase:
        return LabelledCase(
            case_id,
            inputs,
            predicted_output_value(heads, labels),
            predicted_output_value(heads, predictions[case_id]),
            teacher,
        )

    rows = [
        labelled(row.row_id, row.inputs, row.label_indices, row.origin != "gold")
        for row in corpus.rows
    ]
    # An underfit head needs more training before any example or constraint.
    underfit = underfit_suggestion(rows, current_cases=current_cases, epochs=epochs)
    suggestions: list[str] = []
    for gate in gates:
        if underfit is not None and gate in ("examples", "constraints"):
            suggestions.append(underfit)
        elif gate == "examples":
            misses = [
                row
                for row, source in zip(rows, corpus.rows, strict=True)
                if source.origin == "gold" and predictions[source.row_id] != source.label_indices
            ] + [
                labelled(record.case_id, record.inputs, record.label_indices, False)
                for record in external_human
                if predictions[record.case_id] != record.label_indices
            ]
            definition = ir.get("definition")
            constraints = definition.get("constraints") if isinstance(definition, Mapping) else None
            suggestions.append(
                gold_miss_suggestion(
                    misses,
                    rows,
                    [item for item in constraints or [] if isinstance(item, Mapping)],
                    _required_outputs(ir) if constraints else None,
                )
            )
        elif gate == "constraints":
            suggestions.append(violation_suggestion(violations, current_cases))
        elif gate == "types":
            suggestions.append(type_error_suggestion())
        else:
            suggestions.append(
                calibration_suggestion(
                    ece=metrics.ece,
                    rows=calibration_rows,
                    accuracy=metrics.accuracy,
                    current_cases=current_cases,
                    epochs=epochs,
                )
            )
    return tuple(suggestions)


def _detail_suffix(details: Sequence[str]) -> str:
    """The first few failing cases, one per line, so a failed build names them."""

    if not details:
        return ""
    shown = list(details[:_MAXIMUM_DETAILS_PER_FAILURE])
    remaining = len(details) - len(shown)
    lines = [f"  - {item}" for item in shown]
    if remaining > 0:
        lines.append(f"  - and {remaining} more")
    return ":\n" + "\n".join(lines)


def _canonical_input(
    schema: Sequence[Mapping[str, Any]],
    inputs: dict[str, JsonValue],
    case_id: str,
    version: int = 1,
) -> bytes:
    try:
        return serialize_canonical_inputs(
            schema,
            inputs,
            version=version,
            maximum_bytes=MAXIMUM_CANONICAL_ROW_BYTES,
        )
    except (CanonicalInputError, TypeError, ValueError) as error:
        raise VerificationConfigurationError(
            f"verification case {case_id!r} cannot be canonically encoded: {error}"
        ) from error


def _required_outputs(ir: NeuralFunctionIr) -> Callable[[Mapping[str, Any]], tuple[Any, ...]]:
    """The outputs the active ``always`` constraints require for one input, for the
    suggestion that must not propose a rule a constraint already states."""

    try:
        compiled = compile_constraints(ir)
    except (RuntimeError, TypeError, ValueError):
        return lambda _inputs: ()

    def required(inputs: Mapping[str, Any]) -> tuple[Any, ...]:
        outputs: list[Any] = []
        for index, constraint in enumerate(compiled):
            if constraint["kind"] != "always":
                continue
            try:
                active = compiled.evaluate(index, inputs)
            except (RuntimeError, TypeError, ValueError):
                continue
            if active:
                outputs.append(constraint["output"])
        return tuple(outputs)

    return required


def _constraints_count(ir: NeuralFunctionIr) -> int:
    definition = ir.get("definition")
    constraints = definition.get("constraints") if isinstance(definition, Mapping) else None
    if not isinstance(constraints, list):
        raise VerificationConfigurationError("IR definition constraints must be an array")
    return len(constraints)


def _resolved_verified_at(value: str | None) -> str:
    if value is None:
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    _validate_verified_at(value)
    return value


def _validate_verified_at(value: object) -> None:
    if not isinstance(value, str) or _RFC3339_UTC.fullmatch(value) is None:
        raise VerificationConfigurationError("verified_at must be an RFC 3339 UTC timestamp")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise VerificationConfigurationError("verified_at must be a valid timestamp") from error


def _require_torch() -> Any:
    try:
        import torch
    except (ImportError, OSError) as error:
        raise VerificationExecutionError(
            "verification requires the optional PyTorch training dependency"
        ) from error
    return torch


def _calibrate(logits: Any, targets: Any, /, **kwargs: Any) -> Any:
    try:
        from semantscript_model.calibration import calibrate
    except (ImportError, OSError) as error:
        raise VerificationExecutionError(
            "verification calibration requires the SemantScript model package"
        ) from error
    return calibrate(logits, targets, **kwargs)


def _bounded_integer(name: str, value: object, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise VerificationConfigurationError(
            f"{name} must be an integer from {minimum} through {maximum}"
        )


def _positive_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationConfigurationError(f"{name} must be a positive finite number")
    try:
        resolved = float(value)
    except (OverflowError, ValueError) as error:
        raise VerificationConfigurationError(f"{name} must be a positive finite number") from error
    if not math.isfinite(resolved) or resolved <= 0:
        raise VerificationConfigurationError(f"{name} must be a positive finite number")


def _unit_interval(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationConfigurationError(f"{name} must be a finite number from 0 through 1")
    try:
        resolved = float(value)
    except (OverflowError, ValueError) as error:
        raise VerificationConfigurationError(
            f"{name} must be a finite number from 0 through 1"
        ) from error
    if not math.isfinite(resolved) or not 0 <= resolved <= 1:
        raise VerificationConfigurationError(f"{name} must be a finite number from 0 through 1")


__all__ = [
    "MAXIMUM_CALIBRATION_LOGIT_VALUES",
    "MAXIMUM_HUMAN_VERIFICATION_CASE_COUNT",
    "MAXIMUM_VERIFICATION_CASE_COUNT",
    "CalibrationRecordV1",
    "HeadVerificationV1",
    "SeedRetryConfig",
    "SeedRetryDecision",
    "VerificationConfig",
    "VerificationConfigurationError",
    "VerificationError",
    "VerificationExecutionError",
    "VerificationGateError",
    "VerificationMetricsV1",
    "VerificationResult",
    "VerificationStatus",
    "calibration_split_sha256",
    "evaluate_training_result",
    "model_state_sha256",
    "require_passing_verification",
    "seed_retry_decision",
    "tokenizer_json_bytes",
    "verify_training_result",
]
