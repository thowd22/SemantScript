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
from typing import Any, cast

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
        """One function's view over the shared encoder and its adapter plus its own head.

        ``depth`` is the domain's encoder depth (``None`` for the full stack): the
        view embeds through that many shared layers, exactly what the exported
        prefix encoder of the domain computes.
        """

        def __init__(
            self,
            encoder: SentenceEncoder,
            adapter: ApplicationAdapter,
            head: ClassificationHead | FieldHeads,
            depth: int | None = None,
        ) -> None:
            super().__init__()
            _validate_widths(encoder, adapter, head)
            self.encoder = encoder
            self.adapter = adapter
            self.head = head
            self.depth = encoder.validate_depth(depth)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            sentence_embedding = self.encoder(input_ids, attention_mask, depth=self.depth)
            return self.head(self.adapter(sentence_embedding))

    DEFAULT_ADAPTER_REF = "adapter.application"

    def _adapter_key(ref: str) -> str:
        # ModuleDict keys cannot contain dots; adapter refs do (adapter.app.domain).
        return ref.replace(".", "__")

    class SharedEncoderApplication(nn.Module):
        """Every function of one application over a single encoder and its adapters.

        The common case is one adapter shared by every function. With routed
        domains, ``adapters`` maps an adapter ref to its module, ``adapter_depths``
        gives each adapter the number of shared encoder layers its functions run
        (``None`` for the full stack) and every head is attached to one adapter.
        The encoder prefixes stay shared: one encoder serves the whole application.
        """

        def __init__(
            self,
            encoder: SentenceEncoder,
            adapter: ApplicationAdapter | None = None,
            heads: Mapping[str, ClassificationHead | FieldHeads] | None = None,
            *,
            adapters: Mapping[str, ApplicationAdapter] | None = None,
            adapter_depths: Mapping[str, int | None] | None = None,
            function_adapters: Mapping[str, str] | None = None,
        ) -> None:
            super().__init__()
            if not isinstance(encoder, SentenceEncoder):
                raise TypeError("encoder must be a SentenceEncoder")
            if (adapter is None) == (adapters is None):
                raise ValueError("pass exactly one of adapter or adapters")
            resolved = {DEFAULT_ADAPTER_REF: adapter} if adapters is None else dict(adapters)
            if not resolved:
                raise ValueError("an application needs at least one adapter")
            self.encoder = encoder
            self.adapters = nn.ModuleDict()
            self._adapter_refs: dict[str, str] = {}
            self._adapter_depths: dict[str, int | None] = {}
            self._function_adapters: dict[str, str] = {}
            depths = {} if adapter_depths is None else dict(adapter_depths)
            for ref, module in resolved.items():
                self.add_adapter(ref, module, depth=depths.pop(ref, None))
            if depths:
                raise ValueError(f"adapter_depths names unknown adapters: {sorted(depths)}")
            bindings = {} if function_adapters is None else dict(function_adapters)
            for function_id, head in (heads or {}).items():
                self.add_head(function_id, head, adapter_ref=bindings.pop(function_id, None))
            if bindings:
                raise ValueError(f"function_adapters names unknown functions: {sorted(bindings)}")
            self.heads: nn.ModuleDict  # declared in add_head for the type checker

        # ---- adapters ----------------------------------------------------

        def add_adapter(
            self, ref: str, adapter: ApplicationAdapter, *, depth: int | None = None
        ) -> None:
            """Attach a domain adapter under its ref with the encoder depth it reads."""

            if not isinstance(ref, str) or not ref:
                raise ValueError("adapter ref must be a nonempty string")
            if not isinstance(adapter, ApplicationAdapter):
                raise TypeError("adapter must be an ApplicationAdapter")
            if self.encoder.hidden_size != adapter.config.hidden_size:
                raise ValueError("adapter hidden_size must match the encoder hidden_size")
            key = _adapter_key(ref)
            if ref in self._adapter_refs or key in self.adapters:
                raise ValueError(f"adapter {ref!r} is already attached")
            self.adapters[key] = adapter
            self._adapter_refs[ref] = key
            self._adapter_depths[ref] = self.encoder.validate_depth(depth)

        @property
        def adapter_refs(self) -> tuple[str, ...]:
            return tuple(self._adapter_refs)

        def adapter_for(self, ref: str) -> ApplicationAdapter:
            key = self._adapter_refs.get(ref)
            if key is None:
                raise KeyError(f"application has no adapter {ref!r}")
            return cast(ApplicationAdapter, self.adapters[key])

        def adapter_depth(self, ref: str) -> int | None:
            if ref not in self._adapter_depths:
                raise KeyError(f"application has no adapter {ref!r}")
            return self._adapter_depths[ref]

        @property
        def adapter(self) -> ApplicationAdapter:
            """The application's adapter when it has exactly one."""

            if len(self._adapter_refs) != 1:
                raise AttributeError("application has several adapters; use adapter_for(ref)")
            return self.adapter_for(next(iter(self._adapter_refs)))

        # ---- heads -------------------------------------------------------

        @property
        def hidden_size(self) -> int:
            return self.encoder.hidden_size

        @property
        def function_ids(self) -> tuple[str, ...]:
            return tuple(self.heads.keys())

        def add_head(
            self,
            function_id: str,
            head: ClassificationHead | FieldHeads,
            *,
            adapter_ref: str | None = None,
        ) -> None:
            """Attach a new function head to an adapter; the shared modules are untouched."""

            if not hasattr(self, "heads"):
                self.heads = nn.ModuleDict()
            if not isinstance(function_id, str) or not function_id or "." in function_id:
                raise ValueError("function_id must be a nonempty string without dots")
            if function_id in self.heads:
                raise ValueError(f"function {function_id!r} already has a head")
            if len(self.heads) >= MAXIMUM_APPLICATION_FUNCTIONS:
                raise ValueError(
                    f"application exceeds {MAXIMUM_APPLICATION_FUNCTIONS} function heads"
                )
            if adapter_ref is None:
                if len(self._adapter_refs) != 1:
                    raise ValueError(
                        "adapter_ref is required when the application has several adapters"
                    )
                adapter_ref = next(iter(self._adapter_refs))
            adapter = self.adapter_for(adapter_ref)
            _validate_widths(self.encoder, adapter, head)
            self.heads[function_id] = head
            self._function_adapters[function_id] = adapter_ref

        def adapter_ref_of(self, function_id: str) -> str:
            ref = self._function_adapters.get(function_id)
            if ref is None:
                raise KeyError(f"application has no head for function {function_id!r}")
            return ref

        def depth_of(self, function_id: str) -> int | None:
            return self.adapter_depth(self.adapter_ref_of(function_id))

        def function_model(self, function_id: str) -> FunctionModel:
            """A per-function module sharing this application's encoder and adapter."""

            if not isinstance(function_id, str) or function_id not in self.heads:
                raise KeyError(f"application has no head for function {function_id!r}")
            ref = self.adapter_ref_of(function_id)
            return FunctionModel(
                self.encoder,
                self.adapter_for(ref),
                self.heads[function_id],
                self.adapter_depth(ref),
            )

        def embed(self, input_ids: Any, attention_mask: Any, function_id: str | None = None) -> Any:
            """The adapted embedding a head reads: one encoder pass at the adapter's depth."""

            if function_id is None:
                if len(self._adapter_refs) != 1:
                    raise ValueError(
                        "function_id is required when the application has several adapters"
                    )
                ref = next(iter(self._adapter_refs))
            else:
                ref = self.adapter_ref_of(function_id)
            sentence_embedding = self.encoder(
                input_ids, attention_mask, depth=self.adapter_depth(ref)
            )
            return self.adapter_for(ref)(sentence_embedding)

        def forward(self, input_ids: Any, attention_mask: Any) -> dict[str, Any]:
            embeddings: dict[str, Any] = {}
            results: dict[str, Any] = {}
            for function_id, head in self.heads.items():
                ref = self.adapter_ref_of(function_id)
                if ref not in embeddings:
                    embeddings[ref] = self.adapter_for(ref)(
                        self.encoder(input_ids, attention_mask, depth=self.adapter_depth(ref))
                    )
                results[function_id] = head(embeddings[ref])
            return results

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

        def __init__(self, encoder: Any, adapter: Any, head: Any, depth: Any = None) -> None:
            del encoder, adapter, head, depth
            raise _missing_torch() from _TORCH_IMPORT_ERROR

    class SharedEncoderApplication:  # pragma: no cover - exercised only without PyTorch
        """Deferred failure placeholder used when PyTorch is unavailable."""

        def __init__(
            self, encoder: Any, adapter: Any = None, heads: Any = None, **kwargs: Any
        ) -> None:
            del encoder, adapter, heads, kwargs
            raise _missing_torch() from _TORCH_IMPORT_ERROR

    DEFAULT_ADAPTER_REF = "adapter.application"


__all__ = [
    "DEFAULT_ADAPTER_BOTTLENECK_SIZE",
    "DEFAULT_ADAPTER_REF",
    "MAXIMUM_APPLICATION_FUNCTIONS",
    "AdapterConfig",
    "ApplicationAdapter",
    "FunctionModel",
    "SharedEncoderApplication",
]
