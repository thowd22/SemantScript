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

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            if input_ids.ndim != 2 or attention_mask.ndim != 2:
                raise ValueError("input_ids and attention_mask must have shape [BATCH, SEQUENCE]")
            if input_ids.shape != attention_mask.shape:
                raise ValueError("input_ids and attention_mask must have identical shapes")

            encoded = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            last_hidden_state = getattr(encoded, "last_hidden_state", None)
            if last_hidden_state is None and isinstance(encoded, dict):
                last_hidden_state = encoded.get("last_hidden_state")
            if last_hidden_state is None:
                raise ValueError("encoder output must provide last_hidden_state")
            if last_hidden_state.ndim != 3:
                raise ValueError(
                    "encoder last_hidden_state must have shape [BATCH, SEQUENCE, HIDDEN]"
                )
            if last_hidden_state.shape[:2] != attention_mask.shape:
                raise ValueError("encoder output batch and sequence dimensions must match the mask")
            if last_hidden_state.shape[-1] != self.hidden_size:
                raise ValueError("encoder output hidden width does not match encoder.config")

            hidden = last_hidden_state.to(dtype=torch.float32)
            mask = attention_mask.to(device=hidden.device, dtype=torch.float32).unsqueeze(-1)
            token_count = mask.sum(dim=1).clamp_min(1.0)
            return (hidden * mask).sum(dim=1) / token_count

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


__all__ = [
    "DEFAULT_ENCODER",
    "DEFAULT_ENCODER_REVISION",
    "EncoderConfig",
    "SentenceEncoder",
]
