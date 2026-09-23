"""Bounded post-training temperature scaling and calibration metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import Tensor
except (ImportError, OSError) as error:  # pragma: no cover - environment dependent
    torch = None
    Tensor = Any
    _TORCH_IMPORT_ERROR: Exception | None = error
else:
    _TORCH_IMPORT_ERROR = None

DEFAULT_ECE_BIN_COUNT = 15
MAXIMUM_ECE_BIN_COUNT = 1_000
DEFAULT_MINIMUM_TEMPERATURE = 0.05
DEFAULT_MAXIMUM_TEMPERATURE = 20.0
MINIMUM_TEMPERATURE = 0.001
MAXIMUM_TEMPERATURE = 1_000.0
DEFAULT_TEMPERATURE_ITERATIONS = 80
MAXIMUM_TEMPERATURE_ITERATIONS = 256

_GOLDEN_RATIO_CONJUGATE = (math.sqrt(5.0) - 1.0) / 2.0
_SEARCH_RELATIVE_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    """Held-out scalar-head metrics under one fixed temperature."""

    sample_count: int
    bin_count: int
    accuracy: float
    ece: float
    brier_score: float

    def __post_init__(self) -> None:
        _bounded_integer("sample_count", self.sample_count, minimum=1, maximum=2**63 - 1)
        _bounded_integer(
            "bin_count",
            self.bin_count,
            minimum=2,
            maximum=MAXIMUM_ECE_BIN_COUNT,
        )
        _unit_interval("accuracy", self.accuracy)
        _unit_interval("ece", self.ece)
        _unit_interval("brier_score", self.brier_score)


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """A bounded temperature fit and its before/after held-out metrics."""

    temperature: float
    optimization_iterations: int
    uncalibrated_metrics: CalibrationMetrics
    calibrated_metrics: CalibrationMetrics

    def __post_init__(self) -> None:
        _positive_finite("temperature", self.temperature)
        _bounded_integer(
            "optimization_iterations",
            self.optimization_iterations,
            minimum=1,
            maximum=MAXIMUM_TEMPERATURE_ITERATIONS,
        )
        if not isinstance(self.uncalibrated_metrics, CalibrationMetrics) or not isinstance(
            self.calibrated_metrics, CalibrationMetrics
        ):
            raise TypeError("calibration result metrics must be CalibrationMetrics values")
        if (
            self.uncalibrated_metrics.sample_count != self.calibrated_metrics.sample_count
            or self.uncalibrated_metrics.bin_count != self.calibrated_metrics.bin_count
        ):
            raise ValueError("calibration result metrics must describe the same held-out set")


def temperature_scaled_probabilities(
    logits: Tensor,
    /,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Return a float64 distribution over support values for each logit row.

    One-column model-ABI logits are expanded to ``[false, true]`` before
    temperature scaling. Categorical logits retain their stable support order.
    Float64 centered softmax math keeps extreme finite float16/float32 logits
    stable and matches the runtime's divide-then-sigmoid/softmax semantics.
    """

    _require_torch()
    _validate_logits(logits)
    resolved_temperature = _temperature(temperature)
    return _probabilities(logits, resolved_temperature)


def fit_temperature(
    logits: Tensor,
    targets: Tensor,
    /,
    *,
    minimum_temperature: float = DEFAULT_MINIMUM_TEMPERATURE,
    maximum_temperature: float = DEFAULT_MAXIMUM_TEMPERATURE,
    maximum_iterations: int = DEFAULT_TEMPERATURE_ITERATIONS,
) -> float:
    """Fit one positive scalar temperature by bounded held-out NLL minimization."""

    temperature, _ = _fit_temperature(
        logits,
        targets,
        minimum_temperature=minimum_temperature,
        maximum_temperature=maximum_temperature,
        maximum_iterations=maximum_iterations,
    )
    return temperature


def calibration_metrics(
    logits: Tensor,
    targets: Tensor,
    /,
    *,
    temperature: float = 1.0,
    bin_count: int = DEFAULT_ECE_BIN_COUNT,
) -> CalibrationMetrics:
    """Measure accuracy, equal-width top-1 ECE, and normalized Brier score."""

    _require_torch()
    _validate_calibration_inputs(logits, targets)
    resolved_temperature = _temperature(temperature)
    resolved_bins = _bin_count(bin_count)

    with torch.no_grad():
        probabilities = _probabilities(logits.detach(), resolved_temperature)
        # Positive temperature cannot change the raw-logit ordering. argmax
        # deliberately selects the earlier support member on an exact tie.
        predictions = torch.argmax(_categorical_logits(logits.detach()), dim=1)
        correct = predictions == targets
        accuracy = correct.to(dtype=torch.float64).mean()
        confidence = probabilities.gather(1, predictions[:, None]).squeeze(1)

        indices = torch.floor(confidence * resolved_bins).to(dtype=torch.long)
        indices = indices.clamp(min=0, max=resolved_bins - 1)
        counts = torch.bincount(indices, minlength=resolved_bins).to(dtype=torch.float64)
        confidence_sums = torch.zeros(
            resolved_bins,
            dtype=torch.float64,
            device=logits.device,
        )
        correctness_sums = torch.zeros_like(confidence_sums)
        confidence_sums.scatter_add_(0, indices, confidence)
        correctness_sums.scatter_add_(0, indices, correct.to(dtype=torch.float64))
        populated = counts > 0
        bin_confidence = confidence_sums[populated] / counts[populated]
        bin_accuracy = correctness_sums[populated] / counts[populated]
        ece = (torch.abs(bin_accuracy - bin_confidence) * counts[populated]).sum() / logits.shape[0]

        correct_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
        # ||p - one_hot(y)||^2 = ||p||^2 - 2 p_y + 1. Dividing by two is the
        # IR's normalized multiclass definition and keeps the score in [0, 1].
        per_case_brier = (probabilities.square().sum(dim=1) - 2.0 * correct_probability + 1.0) / 2.0
        brier = per_case_brier.mean().clamp(min=0.0, max=1.0)

    return CalibrationMetrics(
        sample_count=logits.shape[0],
        bin_count=resolved_bins,
        accuracy=_unit_float(accuracy),
        ece=_unit_float(ece),
        brier_score=_unit_float(brier),
    )


def calibrate(
    logits: Tensor,
    targets: Tensor,
    /,
    *,
    bin_count: int = DEFAULT_ECE_BIN_COUNT,
    minimum_temperature: float = DEFAULT_MINIMUM_TEMPERATURE,
    maximum_temperature: float = DEFAULT_MAXIMUM_TEMPERATURE,
    maximum_iterations: int = DEFAULT_TEMPERATURE_ITERATIONS,
) -> CalibrationResult:
    """Fit temperature and report uncalibrated and calibrated held-out metrics."""

    resolved_bins = _bin_count(bin_count)
    temperature, iterations = _fit_temperature(
        logits,
        targets,
        minimum_temperature=minimum_temperature,
        maximum_temperature=maximum_temperature,
        maximum_iterations=maximum_iterations,
    )
    return CalibrationResult(
        temperature=temperature,
        optimization_iterations=iterations,
        uncalibrated_metrics=calibration_metrics(
            logits,
            targets,
            temperature=1.0,
            bin_count=resolved_bins,
        ),
        calibrated_metrics=calibration_metrics(
            logits,
            targets,
            temperature=temperature,
            bin_count=resolved_bins,
        ),
    )


def _fit_temperature(
    logits: Tensor,
    targets: Tensor,
    *,
    minimum_temperature: float,
    maximum_temperature: float,
    maximum_iterations: int,
) -> tuple[float, int]:
    _require_torch()
    _validate_calibration_inputs(logits, targets)
    lower_temperature, upper_temperature = _temperature_range(
        minimum_temperature,
        maximum_temperature,
    )
    iterations_limit = _iterations(maximum_iterations)

    # NLL is convex in inverse temperature. Golden-section search therefore
    # provides deterministic bounded work without allocating optimizer state or
    # retaining an autograd graph.
    lower = 1.0 / upper_temperature
    upper = 1.0 / lower_temperature
    neutral_inverse_temperature = min(max(1.0, lower), upper)
    categorical = _categorical_logits(logits.detach()).to(dtype=torch.float64)
    categorical = categorical - categorical.amax(dim=1, keepdim=True)
    detached_targets = targets.detach()

    def objective(inverse_temperature: float) -> float:
        with torch.no_grad():
            loss = torch.nn.functional.cross_entropy(
                categorical * inverse_temperature,
                detached_targets,
                reduction="mean",
            )
        value = float(loss.detach().cpu().item())
        return value if math.isfinite(value) else math.inf

    left = upper - _GOLDEN_RATIO_CONJUGATE * (upper - lower)
    right = lower + _GOLDEN_RATIO_CONJUGATE * (upper - lower)
    left_loss = objective(left)
    right_loss = objective(right)
    iterations = 0
    while iterations < iterations_limit:
        width = upper - lower
        if width <= _SEARCH_RELATIVE_TOLERANCE * max(1.0, abs(lower), abs(upper)):
            break
        if left_loss <= right_loss:
            upper = right
            right = left
            right_loss = left_loss
            left = upper - _GOLDEN_RATIO_CONJUGATE * (upper - lower)
            left_loss = objective(left)
        else:
            lower = left
            left = right
            left_loss = right_loss
            right = lower + _GOLDEN_RATIO_CONJUGATE * (upper - lower)
            right_loss = objective(right)
        iterations += 1

    candidates = (lower, upper, left, right, neutral_inverse_temperature)
    evaluated = [(objective(candidate), candidate) for candidate in candidates]
    best_loss = min(loss for loss, _ in evaluated)
    if not math.isfinite(best_loss):
        raise ValueError("temperature objective is non-finite across the bounded search range")
    slack = max(1e-12, 16.0 * math.ulp(best_loss))
    eligible = [
        inverse_temperature for loss, inverse_temperature in evaluated if loss <= best_loss + slack
    ]
    best_inverse_temperature = min(
        eligible,
        key=lambda value: (abs(math.log(value)), value),
    )
    fitted = 1.0 / best_inverse_temperature
    fitted = min(max(fitted, lower_temperature), upper_temperature)
    return fitted, max(1, iterations)


def _probabilities(logits: Tensor, temperature: float) -> Tensor:
    categorical = _categorical_logits(logits).to(dtype=torch.float64)
    centered = categorical - categorical.amax(dim=1, keepdim=True)
    return torch.softmax(centered / temperature, dim=1)


def _categorical_logits(logits: Tensor) -> Tensor:
    work = logits.to(dtype=torch.float64)
    if work.shape[1] == 1:
        return torch.cat((torch.zeros_like(work), work), dim=1)
    return work


def _validate_calibration_inputs(logits: Tensor, targets: Tensor) -> int:
    _validate_logits(logits)
    if not isinstance(targets, Tensor):
        raise TypeError("targets must be a torch.Tensor")
    if targets.layout != torch.strided:
        raise TypeError("targets must be a dense strided tensor")
    if targets.device.type == "meta":
        raise ValueError("targets must be materialized")
    if targets.ndim != 1:
        raise ValueError("targets must have shape [batch]")
    if targets.shape[0] != logits.shape[0]:
        raise ValueError("logits and targets batch dimensions must match")
    if targets.dtype != torch.long:
        raise TypeError("targets must have dtype torch.long")
    if targets.device != logits.device:
        raise ValueError("logits and targets must be on the same device")
    class_count = 2 if logits.shape[1] == 1 else logits.shape[1]
    if bool(((targets < 0) | (targets >= class_count)).any().item()):
        raise ValueError("targets contain a class index outside the head support")
    return class_count


def _validate_logits(logits: Tensor) -> None:
    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.layout != torch.strided:
        raise TypeError("logits must be a dense strided tensor")
    if logits.device.type == "meta":
        raise ValueError("logits must be materialized")
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if logits.shape[0] == 0:
        raise ValueError("calibration requires a nonempty batch")
    if logits.shape[1] < 1:
        raise ValueError("logits must contain at least one output column")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must have a floating-point dtype")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must contain only finite values")


def _temperature(value: object) -> float:
    _positive_finite("temperature", value)
    return float(value)


def _temperature_range(minimum: object, maximum: object) -> tuple[float, float]:
    _positive_finite("minimum_temperature", minimum)
    _positive_finite("maximum_temperature", maximum)
    lower = float(minimum)
    upper = float(maximum)
    if lower < MINIMUM_TEMPERATURE or upper > MAXIMUM_TEMPERATURE or lower >= upper:
        raise ValueError(
            "temperature search range must be increasing and within "
            f"[{MINIMUM_TEMPERATURE}, {MAXIMUM_TEMPERATURE}]"
        )
    return lower, upper


def _iterations(value: object) -> int:
    _bounded_integer(
        "maximum_iterations",
        value,
        minimum=1,
        maximum=MAXIMUM_TEMPERATURE_ITERATIONS,
    )
    return int(value)


def _bin_count(value: object) -> int:
    _bounded_integer(
        "bin_count",
        value,
        minimum=2,
        maximum=MAXIMUM_ECE_BIN_COUNT,
    )
    return int(value)


def _bounded_integer(name: str, value: object, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")


def _positive_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    try:
        resolved = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _unit_interval(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number from 0 through 1")
    try:
        resolved = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number from 0 through 1") from error
    if not math.isfinite(resolved) or not 0 <= resolved <= 1:
        raise ValueError(f"{name} must be a finite number from 0 through 1")


def _unit_float(value: Tensor) -> float:
    resolved = float(value.detach().cpu().item())
    # Floating reduction roundoff may stray a few ulps outside the proven range.
    return min(1.0, max(0.0, resolved))


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "SemantScript calibration requires the optional PyTorch training dependency; "
            "install the project model/training extra before calibrating logits"
        ) from _TORCH_IMPORT_ERROR


__all__ = [
    "DEFAULT_ECE_BIN_COUNT",
    "DEFAULT_MAXIMUM_TEMPERATURE",
    "DEFAULT_MINIMUM_TEMPERATURE",
    "DEFAULT_TEMPERATURE_ITERATIONS",
    "MAXIMUM_ECE_BIN_COUNT",
    "MAXIMUM_TEMPERATURE",
    "MAXIMUM_TEMPERATURE_ITERATIONS",
    "MINIMUM_TEMPERATURE",
    "CalibrationMetrics",
    "CalibrationResult",
    "calibrate",
    "calibration_metrics",
    "fit_temperature",
    "temperature_scaled_probabilities",
]
