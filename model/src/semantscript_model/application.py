"""Shared encoder, one per-application adapter and one head per sema function.

An application artifact runs a single encoder pass per call and keeps every
function's weights small: the encoder is shared, the adapter is a residual
bottleneck that adapts the sentence embedding for the application, and each
function owns only a classification head. Per-function views expose the
familiar ``(encoder, adapter, head)`` module shape so verification, lifecycle
binding and state digests keep working one function at a time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .encoder import SentenceEncoder
from .heads import ClassificationHead, FieldHeads

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:  # pragma: no cover - branch depends on environment
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR: ModuleNotFoundError | None = error
else:
    _TORCH_IMPORT_ERROR = None

DEFAULT_ADAPTER_BOTTLENECK_SIZE = 64
MAXIMUM_APPLICATION_FUNCTIONS = 4096


def _missing_torch() -> ImportError:
    return ImportError(
        "SemantScript model execution requires the optional PyTorch training dependency; "
        "install the project model/training extra before constructing model modules"
    )


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    """Shape of the residual bottleneck adapter over the sentence embedding."""

    hidden_size: int
    bottleneck_size: int = DEFAULT_ADAPTER_BOTTLENECK_SIZE

    def __post_init__(self) -> None:
        for name in ("hidden_size", "bottleneck_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"adapter {name} must be a positive integer")

    @property
    def parameter_count(self) -> int:
        return 2 * self.hidden_size * self.bottleneck_size + self.bottleneck_size + self.hidden_size


if nn is not None:

    class ApplicationAdapter(nn.Module):
        """Residual bottleneck ``x + up(gelu(down(x)))`` that starts as the identity."""

        def __init__(self, config: AdapterConfig) -> None:
            super().__init__()
            if not isinstance(config, AdapterConfig):
                raise TypeError("config must be an AdapterConfig")
            self.config = config
            self.down = nn.Linear(config.hidden_size, config.bottleneck_size)
            self.up = nn.Linear(config.bottleneck_size, config.hidden_size)
            # A zero output projection makes the adapter an exact identity at
            # initialization, so a freshly attached adapter cannot disturb a
            # pretrained encoder before training moves it.
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        @property
        def hidden_size(self) -> int:
            return self.config.hidden_size

        def forward(self, sentence_embedding: Any) -> Any:
            if sentence_embedding.ndim != 2:
                raise ValueError("sentence_embedding must have shape [BATCH, HIDDEN]")
            if sentence_embedding.shape[-1] != self.config.hidden_size:
                raise ValueError("sentence_embedding hidden width does not match the adapter")
            parameter = self.down.weight
            embedding = sentence_embedding.to(dtype=parameter.dtype)
            adapted = embedding + self.up(torch.nn.functional.gelu(self.down(embedding)))
            return adapted.to(dtype=torch.float32)

    class FunctionModel(nn.Module):
        """One function's view over the shared encoder and adapter plus its own head."""

        def __init__(
            self,
            encoder: SentenceEncoder,
            adapter: ApplicationAdapter,
            head: ClassificationHead | FieldHeads,
        ) -> None:
            super().__init__()
            _validate_widths(encoder, adapter, head)
            self.encoder = encoder
            self.adapter = adapter
            self.head = head

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            sentence_embedding = self.encoder(input_ids, attention_mask)
            return self.head(self.adapter(sentence_embedding))

    class SharedEncoderApplication(nn.Module):
        """Every function of one application over a single encoder and adapter."""

        def __init__(
            self,
            encoder: SentenceEncoder,
            adapter: ApplicationAdapter,
            heads: Mapping[str, ClassificationHead | FieldHeads] | None = None,
        ) -> None:
            super().__init__()
            if not isinstance(encoder, SentenceEncoder):
                raise TypeError("encoder must be a SentenceEncoder")
            if not isinstance(adapter, ApplicationAdapter):
                raise TypeError("adapter must be an ApplicationAdapter")
            if encoder.hidden_size != adapter.config.hidden_size:
                raise ValueError("adapter hidden_size must match the encoder hidden_size")
            self.encoder = encoder
            self.adapter = adapter
            self.heads = nn.ModuleDict()
            for function_id, head in (heads or {}).items():
                self.add_head(function_id, head)

        @property
        def hidden_size(self) -> int:
            return self.encoder.hidden_size

        @property
        def function_ids(self) -> tuple[str, ...]:
            return tuple(self.heads.keys())

        def add_head(self, function_id: str, head: ClassificationHead | FieldHeads) -> None:
            """Attach a new function head; the shared modules are untouched."""

            if not isinstance(function_id, str) or not function_id or "." in function_id:
                raise ValueError("function_id must be a nonempty string without dots")
            if function_id in self.heads:
                raise ValueError(f"function {function_id!r} already has a head")
            if len(self.heads) >= MAXIMUM_APPLICATION_FUNCTIONS:
                raise ValueError(
                    f"application exceeds {MAXIMUM_APPLICATION_FUNCTIONS} function heads"
                )
            _validate_widths(self.encoder, self.adapter, head)
            self.heads[function_id] = head

        def function_model(self, function_id: str) -> FunctionModel:
            """A per-function module sharing this application's encoder and adapter."""

            if not isinstance(function_id, str) or function_id not in self.heads:
                raise KeyError(f"application has no head for function {function_id!r}")
            return FunctionModel(self.encoder, self.adapter, self.heads[function_id])

        def embed(self, input_ids: Any, attention_mask: Any) -> Any:
            """The adapted embedding every head reads: one encoder pass per call."""

            return self.adapter(self.encoder(input_ids, attention_mask))

        def forward(self, input_ids: Any, attention_mask: Any) -> dict[str, Any]:
            function_embedding = self.embed(input_ids, attention_mask)
            return {
                function_id: head(function_embedding) for function_id, head in self.heads.items()
            }

    def _validate_widths(encoder: Any, adapter: Any, head: Any) -> None:
        if not isinstance(head, (ClassificationHead, FieldHeads)):
            raise TypeError("head must be a ClassificationHead or FieldHeads")
        hidden = getattr(encoder, "hidden_size", None)
        if getattr(getattr(adapter, "config", None), "hidden_size", None) != hidden:
            raise ValueError("adapter hidden_size must match the encoder hidden_size")
        if head.config.input_size != hidden:
            raise ValueError("head input_size must match the encoder hidden_size")

else:

    class ApplicationAdapter:  # pragma: no cover - exercised only without PyTorch
        """Deferred failure placeholder used when PyTorch is unavailable."""

        def __init__(self, config: AdapterConfig) -> None:
            del config
            raise _missing_torch() from _TORCH_IMPORT_ERROR

    class FunctionModel:  # pragma: no cover - exercised only without PyTorch
        """Deferred failure placeholder used when PyTorch is unavailable."""

        def __init__(self, encoder: Any, adapter: Any, head: Any) -> None:
            del encoder, adapter, head
            raise _missing_torch() from _TORCH_IMPORT_ERROR

    class SharedEncoderApplication:  # pragma: no cover - exercised only without PyTorch
        """Deferred failure placeholder used when PyTorch is unavailable."""

        def __init__(self, encoder: Any, adapter: Any, heads: Any = None) -> None:
            del encoder, adapter, heads
            raise _missing_torch() from _TORCH_IMPORT_ERROR


__all__ = [
    "DEFAULT_ADAPTER_BOTTLENECK_SIZE",
    "MAXIMUM_APPLICATION_FUNCTIONS",
    "AdapterConfig",
    "ApplicationAdapter",
    "FunctionModel",
    "SharedEncoderApplication",
]
