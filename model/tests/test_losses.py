from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from semantscript_model.losses import (  # noqa: E402
    classification_loss,
    proper_scoring_loss,
)


def test_uniform_binary_proper_loss_matches_closed_form() -> None:
    logits = torch.zeros((1, 2), dtype=torch.float32)
    targets = torch.tensor([0], dtype=torch.long)

    actual = proper_scoring_loss(logits, targets)
    expected = math.log(2.0) + 0.5 * (1.0 - 1.0 / math.sqrt(2.0))

    assert actual.item() == pytest.approx(expected)


def test_cross_entropy_flag_is_the_plain_pytorch_baseline() -> None:
    logits = torch.tensor([[0.2, -0.3, 1.1], [-0.4, 0.8, 0.1]], requires_grad=True)
    targets = torch.tensor([2, 1], dtype=torch.long)

    actual = classification_loss(logits, targets, loss="cross_entropy", reduction="none")

    assert torch.equal(actual, F.cross_entropy(logits, targets, reduction="none"))


def test_binary_sigmoid_abi_matches_equivalent_two_class_logits() -> None:
    binary_logits = torch.tensor([[-2.0], [0.5], [3.0]], requires_grad=True)
    categorical_logits = torch.cat((torch.zeros_like(binary_logits), binary_logits), dim=1)
    targets = torch.tensor([0, 1, 1], dtype=torch.long)

    for loss_name in ("proper", "cross_entropy"):
        binary = classification_loss(
            binary_logits,
            targets,
            loss=loss_name,
            reduction="none",
        )
        categorical = classification_loss(
            categorical_logits,
            targets,
            loss=loss_name,
            reduction="none",
        )
        assert torch.allclose(binary, categorical)


def test_ordinal_loss_adds_normalized_ranked_probability_score() -> None:
    probabilities = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64)
    logits = probabilities.log()
    targets = torch.tensor([0], dtype=torch.long)

    nominal = proper_scoring_loss(logits, targets, ordinal=False)
    ordinal = proper_scoring_loss(logits, targets, ordinal=True)
    expected_rps = ((0.5 - 1.0) ** 2 + (0.8 - 1.0) ** 2) / 2.0

    assert (ordinal - nominal).item() == pytest.approx(expected_rps)


def test_rps_penalizes_distant_probability_mass_more_than_adjacent_mass() -> None:
    probabilities = torch.tensor(
        [[0.5, 0.49, 0.01], [0.5, 0.01, 0.49]],
        dtype=torch.float64,
    )
    logits = probabilities.log()
    targets = torch.tensor([0, 0], dtype=torch.long)

    nominal = proper_scoring_loss(logits, targets, ordinal=False, reduction="none")
    ordinal = proper_scoring_loss(logits, targets, ordinal=True, reduction="none")
    rps = ordinal - nominal

    assert rps[0].item() == pytest.approx(0.12505)
    assert rps[1].item() == pytest.approx(0.24505)
    assert rps[0] < rps[1]


def test_expected_proper_loss_is_minimized_by_the_true_distribution() -> None:
    truth = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    competing = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    targets = torch.arange(3, dtype=torch.long)
    truthful_logits = truth.log().repeat(3, 1)
    competing_logits = competing.log().repeat(3, 1)

    truthful = proper_scoring_loss(truthful_logits, targets, reduction="none")
    untruthful = proper_scoring_loss(competing_logits, targets, reduction="none")

    assert torch.dot(truth, truthful) < torch.dot(truth, untruthful)


def test_extreme_and_shifted_logits_have_finite_losses_and_gradients() -> None:
    logits = torch.tensor(
        [[1_000.0, -1_000.0], [-1_000.0, 1_000.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0, 0], dtype=torch.long)

    loss = proper_scoring_loss(logits, targets)
    shifted = proper_scoring_loss(logits + 10_000.0, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert shifted.item() == pytest.approx(loss.item())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_logits_use_float32_loss_math(dtype) -> None:
    logits = torch.tensor([[0.5, -0.25]], dtype=dtype, requires_grad=True)
    targets = torch.tensor([0], dtype=torch.long)

    loss = proper_scoring_loss(logits, targets)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    ("logits", "targets", "kwargs", "error", "message"),
    [
        (torch.zeros(2), torch.tensor([0]), {}, ValueError, "shape"),
        (torch.zeros((1, 2)), torch.tensor([[0]]), {}, ValueError, "shape"),
        (torch.zeros((0, 2)), torch.empty(0, dtype=torch.long), {}, ValueError, "nonempty"),
        (torch.zeros((2, 2)), torch.tensor([0]), {}, ValueError, "batch"),
        (torch.zeros((1, 0)), torch.tensor([0]), {}, ValueError, "one output"),
        (torch.zeros((1, 2), dtype=torch.long), torch.tensor([0]), {}, TypeError, "floating"),
        (torch.zeros((1, 2)), torch.tensor([0], dtype=torch.int32), {}, TypeError, "torch.long"),
        (torch.tensor([[float("nan"), 0.0]]), torch.tensor([0]), {}, ValueError, "finite"),
        (torch.zeros((1, 2)), torch.tensor([2]), {}, ValueError, "outside"),
        (torch.zeros((1, 1)), torch.tensor([0]), {"ordinal": True}, ValueError, "nominal"),
        (torch.zeros((1, 2)), torch.tensor([0]), {"loss": "unknown"}, ValueError, "loss"),
        (
            torch.zeros((1, 2)),
            torch.tensor([0]),
            {"reduction": "median"},
            ValueError,
            "reduction",
        ),
    ],
)
def test_rejects_malformed_loss_inputs(logits, targets, kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        classification_loss(logits, targets, **kwargs)
