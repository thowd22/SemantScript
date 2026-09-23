"""Composition of a shared sentence encoder and one typed output head."""

from __future__ import annotations

from typing import Any

from .encoder import SentenceEncoder
from .heads import ClassificationHead

try:
    from torch import nn
except ModuleNotFoundError as error:  # pragma: no cover - branch depends on environment
    nn = None
    _TORCH_IMPORT_ERROR: ModuleNotFoundError | None = error
else:
    _TORCH_IMPORT_ERROR = None


def _missing_torch() -> ImportError:
    return ImportError(
        "SemantScript model execution requires the optional PyTorch training dependency; "
        "install the project model/training extra before constructing model modules"
    )


if nn is not None:

    class SemanticClassifier(nn.Module):
        """Run one non-generative encoder pass followed by one fixed head."""

        def __init__(self, encoder: SentenceEncoder, head: ClassificationHead) -> None:
            super().__init__()
            if encoder.hidden_size != head.config.input_size:
                raise ValueError("encoder hidden_size must match the head input_size")
            self.encoder = encoder
            self.head = head

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            function_embedding = self.encoder(input_ids, attention_mask)
            return self.head(function_embedding)

else:

    class SemanticClassifier:  # pragma: no cover - exercised only without PyTorch
        """Deferred failure placeholder used when PyTorch is unavailable."""

        def __init__(self, encoder: SentenceEncoder, head: ClassificationHead) -> None:
            del encoder, head
            raise _missing_torch() from _TORCH_IMPORT_ERROR


__all__ = ["SemanticClassifier"]
