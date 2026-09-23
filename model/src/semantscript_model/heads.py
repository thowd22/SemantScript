"""Typed classification heads for the SemantScript model ABI.

PyTorch is an optional training dependency.  Importing this module is therefore
safe in compiler/trainer-only environments; constructing a head reports the
missing extra with an actionable error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:  # pragma: no cover - branch depends on environment
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR: ModuleNotFoundError | None = error
else:
    _TORCH_IMPORT_ERROR = None


HeadKind = Literal["binary-sigmoid", "categorical-softmax"]
HeadArchitecture = Literal["linear", "mlp"]


@dataclass(frozen=True, slots=True)
class HeadConfig:
    """Configuration for one scalar-output classification head."""

    input_size: int
    kind: HeadKind
    cardinality: int
    architecture: HeadArchitecture = "linear"
    mlp_hidden_size: int | None = None

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError("head input_size must be positive")
        if self.kind == "binary-sigmoid":
            if self.cardinality != 2:
                raise ValueError("binary-sigmoid heads require cardinality 2")
        elif self.kind == "categorical-softmax":
            if self.cardinality < 2:
                raise ValueError("categorical-softmax heads require cardinality at least 2")
        else:
            raise ValueError(f"unsupported head kind: {self.kind!r}")

        if self.architecture == "linear":
            if self.mlp_hidden_size is not None:
                raise ValueError("linear heads cannot set mlp_hidden_size")
        elif self.architecture == "mlp":
            if self.mlp_hidden_size is not None and self.mlp_hidden_size <= 0:
                raise ValueError("mlp_hidden_size must be positive")
        else:
            raise ValueError(f"unsupported head architecture: {self.architecture!r}")

    @property
    def output_size(self) -> int:
        """Return the ABI logit width, not the support cardinality."""

        return 1 if self.kind == "binary-sigmoid" else self.cardinality

    @property
    def effective_mlp_hidden_size(self) -> int:
        """Return the one-hidden-layer width used by a small MLP head."""

        return self.mlp_hidden_size or self.input_size


def _missing_torch() -> ImportError:
    return ImportError(
        "SemantScript model execution requires the optional PyTorch training dependency; "
        "install the project model/training extra before constructing model modules"
    )


if nn is not None:

    class ClassificationHead(nn.Module):
        """A linear or one-hidden-layer MLP scalar-output head."""

        def __init__(self, config: HeadConfig) -> None:
            super().__init__()
            self.config = config
            if config.architecture == "linear":
                self.projection = nn.Linear(config.input_size, config.output_size)
            else:
                self.projection = nn.Sequential(
                    nn.Linear(config.input_size, config.effective_mlp_hidden_size),
                    nn.GELU(),
                    nn.Linear(config.effective_mlp_hidden_size, config.output_size),
                )

        def forward(self, function_embedding: Any) -> Any:
            if function_embedding.ndim != 2:
                raise ValueError("function_embedding must have shape [BATCH, HIDDEN]")
            if function_embedding.shape[-1] != self.config.input_size:
                raise ValueError(
                    "function_embedding hidden width does not match the head input_size"
                )
            parameter = next(self.projection.parameters())
            logits = self.projection(function_embedding.to(dtype=parameter.dtype))
            return logits.to(dtype=torch.float32)

else:

    class ClassificationHead:  # pragma: no cover - exercised only without PyTorch
        """Deferred failure placeholder used when PyTorch is unavailable."""

        def __init__(self, config: HeadConfig) -> None:
            del config
            raise _missing_torch() from _TORCH_IMPORT_ERROR


__all__ = [
    "ClassificationHead",
    "HeadArchitecture",
    "HeadConfig",
    "HeadKind",
]
