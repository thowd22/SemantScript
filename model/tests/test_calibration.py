from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from semantscript_model.calibration import (
    DEFAULT_TEMPERATURE_ITERATIONS,
    MAXIMUM_TEMPERATURE_ITERATIONS,
    calibrate,
    calibration_metrics,
    fit_temperature,
    temperature_scaled_probabilities,
)

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - dependency-free job
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch training extra is unavailable")


def test_import_is_safe_and_use_fails_lazily_without_pytorch() -> None:
    model_source = Path(__file__).parents[1] / "src"
    script = f"""
import builtins
import sys
sys.path.insert(0, {str(model_source)!r})
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] == 'torch':
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from semantscript_model.calibration import temperature_scaled_probabilities
try:
    temperature_scaled_probabilities(object())
except ImportError:
    pass
else:
    raise AssertionError('missing PyTorch did not fail lazily')
"""

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr


@requires_torch
def test_binary_one_logit_probabilities_match_temperature_scaled_sigmoid() -> None:
    logits = torch.tensor([[-2.0], [0.0], [2.0]], dtype=torch.float16)

    probabilities = temperature_scaled_probabilities(logits, temperature=2.0)

    expected_true = torch.sigmoid(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64))
    assert probabilities.dtype == torch.float64
    assert probabilities.shape == (3, 2)
    torch.testing.assert_close(probabilities[:, 1], expected_true)
    torch.testing.assert_close(probabilities[:, 0], 1.0 - expected_true)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(3, dtype=torch.float64))


@requires_torch
def test_categorical_probabilities_are_shift_invariant_and_extreme_stable() -> None:
    logits = torch.tensor([[1_000.0, 999.0, -1_000.0]], dtype=torch.float64)

    original = temperature_scaled_probabilities(logits, temperature=2.0)
    shifted = temperature_scaled_probabilities(logits + 10_000.0, temperature=2.0)

    torch.testing.assert_close(original, shifted)
    assert torch.isfinite(original).all()
    assert original.argmax(dim=1).item() == 0


@requires_torch
def test_metrics_match_equal_width_ece_and_normalized_brier_definitions() -> None:
    probabilities = torch.tensor([[0.8, 0.2], [0.4, 0.6]], dtype=torch.float64)
    targets = torch.tensor([0, 0], dtype=torch.long)

    metrics = calibration_metrics(
        probabilities.log(),
        targets,
        temperature=1.0,
        bin_count=2,
    )

    assert metrics.sample_count == 2
    assert metrics.bin_count == 2
    assert metrics.accuracy == 0.5
    assert metrics.ece == pytest.approx(0.2)
    assert metrics.brier_score == pytest.approx(0.2)


@requires_torch
def test_binary_tie_selects_earlier_false_support_member() -> None:
    logits = torch.zeros((2, 1), dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    metrics = calibration_metrics(logits, targets, bin_count=2)

    assert metrics.accuracy == 0.5
    assert metrics.ece == 0.0
    assert metrics.brier_score == pytest.approx(0.25)


@requires_torch
def test_fitted_temperature_minimizes_binary_nll_and_improves_calibration() -> None:
    logits = torch.full((4, 1), 4.0, dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([1, 1, 1, 0], dtype=torch.long)
    expected = 4.0 / math.log(3.0)

    temperature = fit_temperature(logits, targets)
    result = calibrate(logits, targets, bin_count=10)

    assert temperature == pytest.approx(expected, rel=1e-6)
    assert result.temperature == pytest.approx(expected, rel=1e-6)
    assert 1 <= result.optimization_iterations <= DEFAULT_TEMPERATURE_ITERATIONS
    assert result.calibrated_metrics.accuracy == result.uncalibrated_metrics.accuracy == 0.75
    assert result.calibrated_metrics.ece < 1e-7
    assert result.calibrated_metrics.ece < result.uncalibrated_metrics.ece
    assert result.calibrated_metrics.brier_score < result.uncalibrated_metrics.brier_score
    assert logits.grad is None


@requires_torch
def test_flat_temperature_objective_prefers_neutral_temperature() -> None:
    logits = torch.zeros((6, 3), dtype=torch.float64)
    targets = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

    assert fit_temperature(logits, targets) == pytest.approx(1.0)


@requires_torch
def test_rejects_malformed_calibration_tensors() -> None:
    cases = [
        (torch.zeros(2), torch.tensor([0]), ValueError, "shape"),
        (torch.zeros((0, 2)), torch.empty(0, dtype=torch.long), ValueError, "nonempty"),
        (torch.zeros((1, 0)), torch.tensor([0]), ValueError, "one output"),
        (torch.zeros((2, 2)), torch.tensor([0]), ValueError, "batch"),
        (torch.zeros((1, 2), dtype=torch.long), torch.tensor([0]), TypeError, "floating"),
        (torch.zeros((1, 2)), torch.tensor([[0]]), ValueError, "shape"),
        (torch.zeros((1, 2)), torch.tensor([0], dtype=torch.int32), TypeError, "torch.long"),
        (torch.tensor([[float("nan"), 0.0]]), torch.tensor([0]), ValueError, "finite"),
        (torch.zeros((1, 2)), torch.tensor([2]), ValueError, "outside"),
    ]
    for logits, targets, error, message in cases:
        with pytest.raises(error, match=message):
            calibration_metrics(logits, targets)


@requires_torch
def test_rejects_invalid_temperature_bins_range_and_iteration_bounds() -> None:
    logits = torch.zeros((2, 2))
    targets = torch.tensor([0, 1])

    for temperature in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises((TypeError, ValueError), match="temperature"):
            temperature_scaled_probabilities(logits, temperature=temperature)
    for bin_count in (0, 1):
        with pytest.raises(ValueError, match="bin_count"):
            calibration_metrics(logits, targets, bin_count=bin_count)
    with pytest.raises(ValueError, match="search range"):
        fit_temperature(logits, targets, minimum_temperature=2, maximum_temperature=1)
    with pytest.raises(ValueError, match="maximum_iterations"):
        fit_temperature(
            logits,
            targets,
            maximum_iterations=MAXIMUM_TEMPERATURE_ITERATIONS + 1,
        )
