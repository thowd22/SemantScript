"""Shared sentence encoder and deterministic attention-mask-aware pooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ModuleNotFoundError as error:  # pragma: no cover - branch depends on environment
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR: ModuleNotFoundError | None = error
else:
    _TORCH_IMPORT_ERROR = None


DEFAULT_ENCODER = "answerdotai/ModernBERT-base"
DEFAULT_ENCODER_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Configuration for a Hugging Face encoder loaded on demand."""

    model_name: str = DEFAULT_ENCODER
    revision: str | None = DEFAULT_ENCODER_REVISION
    local_files_only: bool = False
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("encoder model_name must be nonempty")


def _missing_torch() -> ImportError:
    return ImportError(
        "SemantScript model execution requires the optional PyTorch training dependency; "
        "install the project model/training extra before constructing model modules"
    )


def _load_pretrained_encoder(config: EncoderConfig) -> Any:
    try:
        from transformers import AutoModel
    except ModuleNotFoundError as error:
        raise ImportError(
            "Loading a pretrained SemantScript encoder requires the optional transformers "
            "dependency; install the project model/training extra or inject an encoder"
        ) from error

    return AutoModel.from_pretrained(
        config.model_name,
        revision=config.revision,
        local_files_only=config.local_files_only,
        trust_remote_code=config.trust_remote_code,
    )


if nn is not None:

    class SentenceEncoder(nn.Module):
        """Wrap an injectable token encoder and emit float32 masked-mean embeddings."""

        def __init__(
            self,
            config: EncoderConfig | None = None,
            *,
            encoder: Any | None = None,
        ) -> None:
            super().__init__()
            self.config = config or EncoderConfig()
            self.encoder = encoder if encoder is not None else _load_pretrained_encoder(self.config)

            hidden_size = getattr(getattr(self.encoder, "config", None), "hidden_size", None)
            if not isinstance(hidden_size, int) or hidden_size <= 0:
                raise ValueError("encoder.config.hidden_size must be a positive integer")
            self.hidden_size = hidden_size
            layer_count = getattr(getattr(self.encoder, "config", None), "num_hidden_layers", None)
            self.layer_count: int | None = (
                layer_count
                if isinstance(layer_count, int)
                and not isinstance(layer_count, bool)
                and layer_count > 0
                else None
            )

        def validate_depth(self, depth: int | None) -> int | None:
            """A depth is a layer count between 1 and the encoder's layer count; None is the full stack."""

            if depth is None:
                return None
            if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
                raise ValueError("encoder depth must be a positive integer or None")
            if self.layer_count is None:
                raise ValueError(
                    "this encoder does not expose num_hidden_layers; depth routing is unavailable"
                )
            if depth > self.layer_count:
                raise ValueError(
                    f"encoder depth {depth} exceeds the encoder's {self.layer_count} layers"
                )
            return None if depth == self.layer_count else depth

        def forward(self, input_ids: Any, attention_mask: Any, depth: int | None = None) -> Any:
            """Masked-mean embedding after ``depth`` layers (``None``: the full stack).

            A prefix embedding reads the hidden state after ``depth`` transformer
            layers through the encoder's final normalization when it has one, so
            depth routing changes how much of the shared encoder runs, not how the
            embedding is shaped.
            """

            if input_ids.ndim != 2 or attention_mask.ndim != 2:
                raise ValueError("input_ids and attention_mask must have shape [BATCH, SEQUENCE]")
            if input_ids.shape != attention_mask.shape:
                raise ValueError("input_ids and attention_mask must have identical shapes")
            depth = self.validate_depth(depth)

            encoded = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **({} if depth is None else {"output_hidden_states": True}),
            )
            if depth is None:
                hidden_state = _output_field(encoded, "last_hidden_state")
            else:
                hidden_states = _output_field(encoded, "hidden_states")
                if not isinstance(hidden_states, (tuple, list)) or len(hidden_states) <= depth:
                    raise ValueError(
                        "encoder output must provide one hidden state per layer for depth routing"
                    )
                hidden_state = hidden_states[depth]
                final_norm = getattr(self.encoder, "final_norm", None)
                if callable(final_norm):
                    hidden_state = final_norm(hidden_state)
            if hidden_state.ndim != 3:
                raise ValueError("encoder hidden state must have shape [BATCH, SEQUENCE, HIDDEN]")
            if hidden_state.shape[:2] != attention_mask.shape:
                raise ValueError("encoder output batch and sequence dimensions must match the mask")
            if hidden_state.shape[-1] != self.hidden_size:
                raise ValueError("encoder output hidden width does not match encoder.config")

            hidden = hidden_state.to(dtype=torch.float32)
            mask = attention_mask.to(device=hidden.device, dtype=torch.float32).unsqueeze(-1)
            token_count = mask.sum(dim=1).clamp_min(1.0)
            return (hidden * mask).sum(dim=1) / token_count

    class PrefixSentenceEncoder(nn.Module):
        """A fixed-depth view over a SentenceEncoder: the first ``depth`` layers, pooled.

        Exported on its own it is the ONNX encoder a depth-routed domain runs; in
        training it is the encoder a function model of that domain reads.
        """

        def __init__(self, base: SentenceEncoder, depth: int | None) -> None:
            super().__init__()
            if not isinstance(base, SentenceEncoder):
                raise TypeError("base must be a SentenceEncoder")
            self.base = base
            self.depth = base.validate_depth(depth)

        @property
        def hidden_size(self) -> int:
            return self.base.hidden_size

        @property
        def layer_count(self) -> int | None:
            return self.base.layer_count

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            return self.base(input_ids, attention_mask, depth=self.depth)

    def _output_field(encoded: Any, name: str) -> Any:
        value = getattr(encoded, name, None)
        if value is None and isinstance(encoded, dict):
            value = encoded.get(name)
        if value is None:
            raise ValueError(f"encoder output must provide {name}")
        return value

else:

    class SentenceEncoder:  # pragma: no cover - exercised only without PyTorch
        """Deferred failure placeholder used when PyTorch is unavailable."""

        def __init__(
            self,
            config: EncoderConfig | None = None,
            *,
            encoder: Any | None = None,
        ) -> None:
            del config, encoder
            raise _missing_torch() from _TORCH_IMPORT_ERROR

    class PrefixSentenceEncoder:  # pragma: no cover - exercised only without PyTorch
        """Deferred failure placeholder used when PyTorch is unavailable."""

        def __init__(self, base: Any, depth: int | None) -> None:
            del base, depth
            raise _missing_torch() from _TORCH_IMPORT_ERROR


__all__ = [
    "DEFAULT_ENCODER",
    "DEFAULT_ENCODER_REVISION",
    "EncoderConfig",
    "PrefixSentenceEncoder",
    "SentenceEncoder",
]
