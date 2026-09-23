"""Numerically stable categorical losses for typed SemantScript heads."""

from __future__ import annotations

from typing import Any, Literal

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
except ModuleNotFoundError as error:  # pragma: no cover - branch depends on environment
    torch = None
    F = None
    Tensor = Any
    _TORCH_IMPORT_ERROR: ModuleNotFoundError | None = error
else:
    _TORCH_IMPORT_ERROR = None

type LossName = Literal["proper", "cross_entropy"]
type Reduction = Literal["none", "mean", "sum"]

SPHERICAL_WEIGHT = 0.5
RPS_WEIGHT = 1.0


def classification_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    ordinal: bool = False,
    loss: LossName = "proper",
    reduction: Reduction = "mean",
) -> Tensor:
    """Return the configured per-head classification loss.

    Categorical heads use ``[batch, classes]`` logits. The model ABI's boolean
    ``binary-sigmoid`` head is also accepted as ``[batch, 1]`` and is evaluated
    as the equivalent two-class distribution ``softmax([0, logit])`` over
    ``[false, true]``.

    The default strictly proper objective is log loss plus half the spherical
    loss. Ordinal heads additionally receive normalized ranked probability
    score (RPS). ``loss="cross_entropy"`` selects plain log loss as an explicit
    baseline and does not add either spherical loss or RPS.
    """

    _require_torch()
    _validate_arguments(logits, targets, ordinal=ordinal, loss=loss, reduction=reduction)
    categorical_logits = _categorical_logits(logits)
    log_loss = F.cross_entropy(categorical_logits, targets, reduction="none")

    if loss == "cross_entropy":
        return _reduce(log_loss, reduction)

    probabilities = torch.softmax(categorical_logits, dim=-1)
    spherical_loss = _spherical_loss(categorical_logits, targets)
    per_example = log_loss + SPHERICAL_WEIGHT * spherical_loss
    if ordinal:
        per_example = per_example + RPS_WEIGHT * _rps(probabilities, targets)
    return _reduce(per_example, reduction)


def proper_scoring_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    ordinal: bool = False,
    reduction: Reduction = "mean",
) -> Tensor:
    """Return log + spherical loss, with normalized RPS for ordinal heads."""

    return classification_loss(
        logits,
        targets,
        ordinal=ordinal,
        loss="proper",
        reduction=reduction,
    )


def _require_torch() -> None:
    if torch is None or F is None:
        raise ImportError(
            "SemantScript model losses require the optional PyTorch training dependency; "
            "install the project model/training extra before computing a loss"
        ) from _TORCH_IMPORT_ERROR


def _categorical_logits(logits: Tensor) -> Tensor:
    # Loss math must remain float32 under fp16/bf16 autocast. Preserve float64
    # when explicitly supplied for numerical analysis and gradient checks.
    work = logits if logits.dtype == torch.float64 else logits.float()
    if work.shape[1] == 1:
        return torch.cat((torch.zeros_like(work), work), dim=1)
    return work


def _spherical_loss(logits: Tensor, targets: Tensor) -> Tensor:
    # p_y / ||p||_2 is invariant to softmax normalization, so evaluate it from
    # max-centered exponentials. The norm is at least one because every row has
    # a centered maximum of zero; no epsilon or probability clipping is needed.
    centered = logits - logits.amax(dim=-1, keepdim=True)
    exponentials = torch.exp(centered)
    correct = exponentials.gather(1, targets[:, None]).squeeze(1)
    norm = torch.linalg.vector_norm(exponentials, ord=2, dim=-1)
    return 1.0 - correct / norm


def _rps(probabilities: Tensor, targets: Tensor) -> Tensor:
    class_count = probabilities.shape[1]
    cumulative = torch.cumsum(probabilities, dim=-1)[:, :-1]
    thresholds = torch.arange(
        class_count - 1,
        device=targets.device,
        dtype=targets.dtype,
    )
    observed = (targets[:, None] <= thresholds[None, :]).to(probabilities.dtype)
    return torch.square(cumulative - observed).mean(dim=-1)


def _validate_arguments(
    logits: Tensor,
    targets: Tensor,
    *,
    ordinal: bool,
    loss: object,
    reduction: object,
) -> None:
    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if not isinstance(targets, Tensor):
        raise TypeError("targets must be a torch.Tensor")
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if targets.ndim != 1:
        raise ValueError("targets must have shape [batch]")
    if logits.shape[0] == 0:
        raise ValueError("loss requires a nonempty batch")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets batch dimensions must match")
    if logits.shape[1] < 1:
        raise ValueError("logits must contain at least one output column")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must have a floating-point dtype")
    if targets.dtype != torch.long:
        raise TypeError("targets must have dtype torch.long")
    if logits.device != targets.device:
        raise ValueError("logits and targets must be on the same device")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must contain only finite values")
    if not isinstance(ordinal, bool):
        raise TypeError("ordinal must be a boolean")
    if logits.shape[1] == 1 and ordinal:
        raise ValueError("the one-logit binary ABI is nominal, not ordinal")
    class_count = 2 if logits.shape[1] == 1 else logits.shape[1]
    if bool(((targets < 0) | (targets >= class_count)).any().item()):
        raise ValueError("targets contain a class index outside the head support")
    if loss not in ("proper", "cross_entropy"):
        raise ValueError("loss must be 'proper' or 'cross_entropy'")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _reduce(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    return values.mean()


__all__ = [
    "RPS_WEIGHT",
    "SPHERICAL_WEIGHT",
    "LossName",
    "Reduction",
    "classification_loss",
    "proper_scoring_loss",
]
