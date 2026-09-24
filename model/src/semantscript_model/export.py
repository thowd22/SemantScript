"""Low-level ONNX export and parity validation for model components.

The deployment artifact uses three independently loadable graphs.  This module
deliberately imports the optional export stack only when an export is requested,
so importing :mod:`semantscript_model` remains safe in compiler-only installs.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from numbers import Real
from pathlib import Path
from stat import S_ISREG
from typing import Any, Literal

ONNX_OPSET = 17
MAXIMUM_PARITY_SEQUENCE_LENGTH = 8192
DEFAULT_MAXIMUM_COMPONENT_BYTES = 1024 * 1024 * 1024
DEFAULT_PARITY_RELATIVE_TOLERANCE = 1e-4
DEFAULT_PARITY_ABSOLUTE_TOLERANCE = 1e-5
MAXIMUM_PARITY_RELATIVE_TOLERANCE = 1e-3
MAXIMUM_PARITY_ABSOLUTE_TOLERANCE = 1e-3
_STANDARD_ONNX_DOMAINS = frozenset(("", "ai.onnx"))

TensorDimension = int | str
ComponentRole = Literal["encoder", "adapter", "head"]


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    """One immutable tensor descriptor for an artifact manifest."""

    name: str
    dtype: Literal["int64", "float32"]
    shape: tuple[TensorDimension, ...]


@dataclass(frozen=True, slots=True)
class ExportedOnnxComponent:
    """Validated file and ABI metadata for one exported ONNX graph."""

    role: ComponentRole
    path: Path
    byte_length: int
    sha256: str
    opset: int
    inputs: tuple[TensorMetadata, ...]
    outputs: tuple[TensorMetadata, ...]
    external_data: Literal[False]
    maximum_absolute_difference: float


@dataclass(frozen=True, slots=True)
class ExportedOnnxComponents:
    """The three runtime graphs plus bounded end-to-end parity information."""

    encoder: ExportedOnnxComponent
    adapter: ExportedOnnxComponent
    head: ExportedOnnxComponent
    parity_batch_size: int
    parity_sequence_length: int
    chain_maximum_absolute_difference: float


@dataclass(frozen=True, slots=True)
class QuantizedOnnxComponent:
    """Validated file identity and settings of one dynamically quantized graph."""

    role: ComponentRole
    path: Path
    byte_length: int
    sha256: str
    opset: int
    method: Literal["dynamic"]
    weight_type: Literal["int8", "uint8"]
    per_channel: bool
    reduce_range: bool
    quantized_matmul_count: int
    source_byte_length: int
    source_sha256: str


def quantize_onnx_encoder(
    source: str | Path,
    destination: str | Path,
    *,
    weight_type: str = "int8",
    per_channel: bool = False,
    reduce_range: bool = False,
    maximum_component_bytes: int = DEFAULT_MAXIMUM_COMPONENT_BYTES,
) -> QuantizedOnnxComponent:
    """Derive a dynamically quantized copy of one exported encoder graph.

    Weights of every MatMul with a constant operand become 8-bit integers and
    activations are quantized per call by ``DynamicQuantizeLinear``; both are
    standard ONNX operators, so the result imports the same opset as the
    source and needs no extra runtime capability. The source must already be
    a validated exported component; the destination must not exist. Whether
    the quantized graph still makes the same decisions is not decided here:
    callers verify that against the source graph before publishing.
    """

    if weight_type not in ("int8", "uint8"):
        raise ValueError('weight_type must be "int8" or "uint8"')
    for name, flag in (("per_channel", per_channel), ("reduce_range", reduce_range)):
        if not isinstance(flag, bool):
            raise TypeError(f"{name} must be a boolean")
    maximum_component_bytes = _validate_maximum_component_bytes(maximum_component_bytes)
    onnx, quantization = _load_quantization_dependencies()

    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file() or source_path.is_symlink():
        raise FileNotFoundError(f"quantization source is not a regular file: {source_path}")
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(f"quantization destination already exists: {destination_path}")
    if destination_path.resolve() == source_path.resolve():
        raise ValueError("quantization destination must differ from the source")
    _validate_exported_component_file(source_path, maximum_component_bytes)
    _validate_onnx_container(onnx, source_path)
    source_byte_length, source_sha256 = _file_identity(source_path)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        quantization.quantize_dynamic(
            str(source_path),
            str(destination_path),
            per_channel=per_channel,
            reduce_range=reduce_range,
            weight_type=(
                quantization.QuantType.QInt8
                if weight_type == "int8"
                else quantization.QuantType.QUInt8
            ),
        )
        _validate_exported_component_file(destination_path, maximum_component_bytes)
        _validate_onnx_container(onnx, destination_path)
        model = onnx.load(str(destination_path), load_external_data=False)
        quantized_matmul_count = sum(
            1 for node in model.graph.node if node.op_type == "MatMulInteger"
        )
        if quantized_matmul_count == 0:
            raise ValueError("dynamic quantization produced no MatMulInteger node")
    except BaseException:
        destination_path.unlink(missing_ok=True)
        raise
    byte_length, digest = _file_identity(destination_path)
    return QuantizedOnnxComponent(
        role="encoder",
        path=destination_path,
        byte_length=byte_length,
        sha256=digest,
        opset=ONNX_OPSET,
        method="dynamic",
        weight_type=weight_type,
        per_channel=per_channel,
        reduce_range=reduce_range,
        quantized_matmul_count=quantized_matmul_count,
        source_byte_length=source_byte_length,
        source_sha256=source_sha256,
    )


def _load_quantization_dependencies() -> tuple[Any, Any]:
    try:
        import onnx
        from onnxruntime import quantization
    except ImportError as error:
        raise ImportError(
            "ONNX quantization requires the optional onnx and onnxruntime dependencies"
        ) from error
    return onnx, quantization


def _file_identity(path: Path) -> tuple[int, str]:
    digest = sha256()
    byte_length = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_length += len(chunk)
    return byte_length, digest.hexdigest()


class _IdentityAdapter:
    """Created lazily as an ``nn.Module`` without importing PyTorch globally."""

    @staticmethod
    def create(torch: Any) -> Any:
        class IdentityAdapter(torch.nn.Module):
            def forward(self, sentence_embedding: Any) -> Any:
                return sentence_embedding

        return IdentityAdapter()


def _load_export_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy
        import onnx
        import onnxruntime
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        missing = error.name or "an ONNX export dependency"
        raise ImportError(
            "SemantScript ONNX export requires the optional PyTorch, ONNX, and "
            f"ONNX Runtime dependencies; missing {missing!r}"
        ) from error
    return torch, onnx, onnxruntime, numpy


@dataclass(frozen=True, slots=True)
class ExportedApplicationComponents:
    """One encoder, one adapter and one head per function with chained parity."""

    encoder: ExportedOnnxComponent
    adapter: ExportedOnnxComponent
    heads: Mapping[str, ExportedOnnxComponent]
    parity_batch_size: int
    parity_sequence_length: int
    chain_maximum_absolute_differences: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ExportedRoutedApplicationComponents:
    """One encoder per depth, one adapter per domain and one head per function.

    ``encoders`` is keyed by depth key (``"full"`` or ``"depth-006"``), ``adapters``
    by adapter ref and ``heads`` by function name; every head was parity-checked
    through its own function's encoder-adapter-head chain.
    """

    encoders: Mapping[str, ExportedOnnxComponent]
    adapters: Mapping[str, ExportedOnnxComponent]
    heads: Mapping[str, ExportedOnnxComponent]
    parity_batch_size: int
    parity_sequence_length: int
    chain_maximum_absolute_differences: Mapping[str, float]


def depth_key(depth: int | None) -> str:
    """The resource key of the encoder prefix a depth runs (``None``: the full stack)."""

    if depth is None:
        return "full"
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("encoder depth must be a positive integer or None")
    return f"depth-{depth:03d}"


def export_onnx_components(
    encoder: Any,
    head: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    encoder_path: str | Path,
    adapter_path: str | Path,
    head_path: str | Path,
    adapter: Any | None = None,
    relative_tolerance: float = DEFAULT_PARITY_RELATIVE_TOLERANCE,
    absolute_tolerance: float = DEFAULT_PARITY_ABSOLUTE_TOLERANCE,
    maximum_component_bytes: int = DEFAULT_MAXIMUM_COMPONENT_BYTES,
) -> ExportedOnnxComponents:
    """Export, validate, and parity-check the three model ABI components.

    The parity batch is intentionally restricted to one bounded sequence, which
    matches the v1 runtime.  Destination files must not already exist and must
    resolve to three distinct paths.  Without an explicit ``adapter`` the
    identity adapter is exported.
    """

    destinations = _prepare_destinations(encoder_path, adapter_path, head_path)
    exported = export_application_components(
        encoder,
        adapter,
        {"function": head},
        input_ids,
        attention_mask,
        encoder_path=destinations[0],
        adapter_path=destinations[1],
        head_paths={"function": destinations[2]},
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        maximum_component_bytes=maximum_component_bytes,
        _destinations_prepared=True,
    )
    return ExportedOnnxComponents(
        encoder=exported.encoder,
        adapter=exported.adapter,
        head=exported.heads["function"],
        parity_batch_size=exported.parity_batch_size,
        parity_sequence_length=exported.parity_sequence_length,
        chain_maximum_absolute_difference=exported.chain_maximum_absolute_differences["function"],
    )


def export_application_components(
    encoder: Any,
    adapter: Any | None,
    heads: Mapping[str, Any],
    input_ids: Any,
    attention_mask: Any,
    *,
    encoder_path: str | Path,
    adapter_path: str | Path,
    head_paths: Mapping[str, str | Path],
    relative_tolerance: float = DEFAULT_PARITY_RELATIVE_TOLERANCE,
    absolute_tolerance: float = DEFAULT_PARITY_ABSOLUTE_TOLERANCE,
    maximum_component_bytes: int = DEFAULT_MAXIMUM_COMPONENT_BYTES,
    _destinations_prepared: bool = False,
) -> ExportedApplicationComponents:
    """Export one shared encoder and adapter plus one head per function.

    Every head is parity-checked on its own and through the exported
    encoder-adapter-head chain, exactly as the single-function export does.
    ``heads`` and ``head_paths`` must name the same functions; the destinations
    must be distinct and absent.
    """

    relative_tolerance = _validate_parity_tolerance(
        relative_tolerance,
        "relative_tolerance",
        MAXIMUM_PARITY_RELATIVE_TOLERANCE,
    )
    absolute_tolerance = _validate_parity_tolerance(
        absolute_tolerance,
        "absolute_tolerance",
        MAXIMUM_PARITY_ABSOLUTE_TOLERANCE,
    )
    maximum_component_bytes = _validate_maximum_component_bytes(maximum_component_bytes)
    if not isinstance(heads, Mapping) or not heads:
        raise ValueError("heads must be a nonempty mapping of function ids to head modules")
    if not isinstance(head_paths, Mapping) or set(head_paths) != set(heads):
        raise ValueError("head_paths must name exactly the functions in heads")
    function_ids = tuple(heads)

    torch, onnx, onnxruntime, numpy = _load_export_dependencies()
    if _destinations_prepared:
        encoder_destination = Path(encoder_path)
        adapter_destination = Path(adapter_path)
        head_destinations = {name: Path(head_paths[name]) for name in function_ids}
    else:
        prepared = _prepare_destination_paths(
            [encoder_path, adapter_path, *(head_paths[name] for name in function_ids)]
        )
        encoder_destination, adapter_destination = prepared[0], prepared[1]
        head_destinations = dict(zip(function_ids, prepared[2:], strict=True))
    _validate_parity_inputs(torch, input_ids, attention_mask)
    adapter_module = _IdentityAdapter.create(torch) if adapter is None else adapter

    encoder_inputs = (
        TensorMetadata("input_ids", "int64", ("BATCH", "SEQUENCE")),
        TensorMetadata("attention_mask", "int64", ("BATCH", "SEQUENCE")),
    )
    roots = (encoder, adapter_module, *heads.values())
    training_states = tuple(
        (module, bool(module.training)) for root in roots for module in root.modules()
    )
    exported_paths: list[Path] = []
    try:
        for root in roots:
            root.eval()
        # ``no_grad`` keeps intermediate tensors usable by the legacy tracer;
        # tensors created by ``inference_mode`` cannot subsequently be traced
        # through a parameterized head.
        with torch.no_grad():
            sentence_embedding = encoder(input_ids, attention_mask)
            _validate_float_output(torch, sentence_embedding, "encoder", expected_batch=1)
            hidden_size = int(sentence_embedding.shape[1])
            function_embedding = adapter_module(sentence_embedding)
            _validate_float_output(
                torch,
                function_embedding,
                "adapter",
                expected_batch=1,
                expected_width=hidden_size,
            )
            head_logits: dict[str, Any] = {}
            logit_widths: dict[str, int] = {}
            for name, head in heads.items():
                logits = head(function_embedding)
                _validate_float_output(
                    torch,
                    logits,
                    f"head {name}",
                    expected_batch=1,
                    expected_width=getattr(getattr(head, "config", None), "output_size", None),
                )
                head_logits[name] = logits
                logit_widths[name] = int(logits.shape[1])

        encoder_outputs = (TensorMetadata("sentence_embedding", "float32", ("BATCH", hidden_size)),)
        adapter_inputs = (TensorMetadata("sentence_embedding", "float32", ("BATCH", hidden_size)),)
        adapter_outputs = (TensorMetadata("function_embedding", "float32", ("BATCH", hidden_size)),)
        head_inputs = (TensorMetadata("function_embedding", "float32", ("BATCH", hidden_size)),)

        exported_paths.append(encoder_destination)
        _export_graph(
            torch,
            encoder,
            (input_ids, attention_mask),
            encoder_destination,
            input_names=("input_ids", "attention_mask"),
            output_names=("sentence_embedding",),
            dynamic_axes={
                "input_ids": {0: "BATCH", 1: "SEQUENCE"},
                "attention_mask": {0: "BATCH", 1: "SEQUENCE"},
                "sentence_embedding": {0: "BATCH"},
            },
        )
        _validate_exported_component_file(encoder_destination, maximum_component_bytes)
        encoder_difference, ort_sentence_embedding = _validate_and_run(
            onnx,
            onnxruntime,
            numpy,
            encoder_destination,
            encoder_inputs,
            encoder_outputs,
            {
                "input_ids": input_ids.detach().cpu().numpy(),
                "attention_mask": attention_mask.detach().cpu().numpy(),
            },
            sentence_embedding.detach().cpu().numpy(),
            relative_tolerance,
            absolute_tolerance,
        )
        encoder_metadata = _component_metadata(
            "encoder",
            encoder_destination,
            encoder_inputs,
            encoder_outputs,
            encoder_difference,
        )

        exported_paths.append(adapter_destination)
        _export_graph(
            torch,
            adapter_module,
            (sentence_embedding,),
            adapter_destination,
            input_names=("sentence_embedding",),
            output_names=("function_embedding",),
            dynamic_axes={
                "sentence_embedding": {0: "BATCH"},
                "function_embedding": {0: "BATCH"},
            },
        )
        _validate_exported_component_file(adapter_destination, maximum_component_bytes)
        adapter_difference, _ = _validate_and_run(
            onnx,
            onnxruntime,
            numpy,
            adapter_destination,
            adapter_inputs,
            adapter_outputs,
            {"sentence_embedding": sentence_embedding.detach().cpu().numpy()},
            function_embedding.detach().cpu().numpy(),
            relative_tolerance,
            absolute_tolerance,
        )
        adapter_metadata = _component_metadata(
            "adapter",
            adapter_destination,
            adapter_inputs,
            adapter_outputs,
            adapter_difference,
        )
        # Exercise the actual adjacent edge, rather than only independent graph inputs.
        chained_adapter = _run_onnx(
            onnxruntime,
            adapter_destination,
            {"sentence_embedding": ort_sentence_embedding},
        )

        head_metadata: dict[str, ExportedOnnxComponent] = {}
        chain_differences: dict[str, float] = {}
        for name, head in heads.items():
            destination = head_destinations[name]
            head_outputs = (TensorMetadata("logits", "float32", ("BATCH", logit_widths[name])),)
            exported_paths.append(destination)
            _export_graph(
                torch,
                head,
                (function_embedding,),
                destination,
                input_names=("function_embedding",),
                output_names=("logits",),
                dynamic_axes={"function_embedding": {0: "BATCH"}, "logits": {0: "BATCH"}},
            )
            _validate_exported_component_file(destination, maximum_component_bytes)
            expected_logits = head_logits[name].detach().cpu().numpy()
            head_difference, _ = _validate_and_run(
                onnx,
                onnxruntime,
                numpy,
                destination,
                head_inputs,
                head_outputs,
                {"function_embedding": function_embedding.detach().cpu().numpy()},
                expected_logits,
                relative_tolerance,
                absolute_tolerance,
            )
            head_metadata[name] = _component_metadata(
                "head",
                destination,
                head_inputs,
                head_outputs,
                head_difference,
            )
            chain_logits = _run_onnx(
                onnxruntime,
                destination,
                {"function_embedding": chained_adapter},
            )
            chain_differences[name] = _assert_parity(
                numpy,
                f"encoder-adapter-head chain for {name}",
                expected_logits,
                chain_logits,
                relative_tolerance,
                absolute_tolerance,
            )
    except BaseException:
        for path in exported_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        for module, training in training_states:
            module.training = training

    return ExportedApplicationComponents(
        encoder=encoder_metadata,
        adapter=adapter_metadata,
        heads=head_metadata,
        parity_batch_size=int(input_ids.shape[0]),
        parity_sequence_length=int(input_ids.shape[1]),
        chain_maximum_absolute_differences=chain_differences,
    )


def export_routed_application_components(
    encoder: Any,
    adapters: Mapping[str, Any],
    adapter_depths: Mapping[str, int | None],
    heads: Mapping[str, Any],
    function_adapters: Mapping[str, str],
    input_ids: Any,
    attention_mask: Any,
    *,
    encoder_paths: Mapping[str, str | Path],
    adapter_paths: Mapping[str, str | Path],
    head_paths: Mapping[str, str | Path],
    relative_tolerance: float = DEFAULT_PARITY_RELATIVE_TOLERANCE,
    absolute_tolerance: float = DEFAULT_PARITY_ABSOLUTE_TOLERANCE,
    maximum_component_bytes: int = DEFAULT_MAXIMUM_COMPONENT_BYTES,
) -> ExportedRoutedApplicationComponents:
    """Export a depth-routed application: an encoder prefix per depth, an adapter per domain.

    ``adapter_depths`` gives each adapter ref the encoder depth its functions run
    (``None`` for the full stack); ``encoder_paths`` is keyed by ``depth_key`` and
    must name exactly the depths in use. Every adapter is parity-checked against
    the embedding of its depth and every head through its function's full chain.
    Depth routing measured on this stack (ONNX Runtime executes a graph whole
    regardless of which outputs are fetched) needs one graph per depth; prefixes
    share their weights in training and differ only in how many layers they keep.
    """

    relative_tolerance = _validate_parity_tolerance(
        relative_tolerance, "relative_tolerance", MAXIMUM_PARITY_RELATIVE_TOLERANCE
    )
    absolute_tolerance = _validate_parity_tolerance(
        absolute_tolerance, "absolute_tolerance", MAXIMUM_PARITY_ABSOLUTE_TOLERANCE
    )
    maximum_component_bytes = _validate_maximum_component_bytes(maximum_component_bytes)
    if not isinstance(adapters, Mapping) or not adapters:
        raise ValueError("adapters must be a nonempty mapping of adapter refs to modules")
    torch_module, _, _, _ = _load_export_dependencies()
    # A ref without a trained adapter (single-function classifiers) exports the identity.
    adapters = {
        ref: (_IdentityAdapter.create(torch_module) if module is None else module)
        for ref, module in adapters.items()
    }
    if not isinstance(heads, Mapping) or not heads:
        raise ValueError("heads must be a nonempty mapping of function ids to head modules")
    if set(adapter_depths) != set(adapters):
        raise ValueError("adapter_depths must name exactly the adapters")
    if set(function_adapters) != set(heads):
        raise ValueError("function_adapters must name exactly the functions in heads")
    if any(ref not in adapters for ref in function_adapters.values()):
        raise ValueError("function_adapters must reference declared adapters")
    if set(adapter_paths) != set(adapters):
        raise ValueError("adapter_paths must name exactly the adapters")
    if set(head_paths) != set(heads):
        raise ValueError("head_paths must name exactly the functions in heads")
    depths_in_use = {depth_key(adapter_depths[ref]): adapter_depths[ref] for ref in adapters}
    if set(encoder_paths) != set(depths_in_use):
        raise ValueError(
            f"encoder_paths must name exactly the depths in use: {sorted(depths_in_use)}"
        )

    torch, onnx, onnxruntime, numpy = _load_export_dependencies()
    from .encoder import PrefixSentenceEncoder, SentenceEncoder

    if not isinstance(encoder, SentenceEncoder):
        raise TypeError("encoder must be a SentenceEncoder")
    encoder_keys = sorted(depths_in_use, key=lambda key: (key != "full", key))
    adapter_refs = tuple(adapters)
    function_ids = tuple(heads)
    prepared = _prepare_destination_paths(
        [
            *(encoder_paths[key] for key in encoder_keys),
            *(adapter_paths[ref] for ref in adapter_refs),
            *(head_paths[name] for name in function_ids),
        ]
    )
    encoder_destinations = dict(zip(encoder_keys, prepared[: len(encoder_keys)], strict=True))
    offset = len(encoder_keys)
    adapter_destinations = dict(
        zip(adapter_refs, prepared[offset : offset + len(adapter_refs)], strict=True)
    )
    head_destinations = dict(zip(function_ids, prepared[offset + len(adapter_refs) :], strict=True))
    _validate_parity_inputs(torch, input_ids, attention_mask)

    encoder_inputs = (
        TensorMetadata("input_ids", "int64", ("BATCH", "SEQUENCE")),
        TensorMetadata("attention_mask", "int64", ("BATCH", "SEQUENCE")),
    )
    roots = (encoder, *adapters.values(), *heads.values())
    training_states = tuple(
        (module, bool(module.training)) for root in roots for module in root.modules()
    )
    exported_paths: list[Path] = []
    try:
        for root in roots:
            root.eval()
        prefixes = {
            key: (encoder if depth is None else PrefixSentenceEncoder(encoder, depth))
            for key, depth in depths_in_use.items()
        }
        with torch.no_grad():
            sentence_embeddings: dict[str, Any] = {}
            for key, prefix in prefixes.items():
                embedding = prefix(input_ids, attention_mask)
                _validate_float_output(torch, embedding, f"encoder {key}", expected_batch=1)
                sentence_embeddings[key] = embedding
            hidden_size = int(next(iter(sentence_embeddings.values())).shape[1])
            function_embeddings: dict[str, Any] = {}
            for ref, adapter in adapters.items():
                embedding = adapter(sentence_embeddings[depth_key(adapter_depths[ref])])
                _validate_float_output(
                    torch, embedding, f"adapter {ref}", expected_batch=1, expected_width=hidden_size
                )
                function_embeddings[ref] = embedding
            head_logits: dict[str, Any] = {}
            logit_widths: dict[str, int] = {}
            for name, head in heads.items():
                logits = head(function_embeddings[function_adapters[name]])
                _validate_float_output(
                    torch,
                    logits,
                    f"head {name}",
                    expected_batch=1,
                    expected_width=getattr(getattr(head, "config", None), "output_size", None),
                )
                head_logits[name] = logits
                logit_widths[name] = int(logits.shape[1])

        embedding_metadata = TensorMetadata("sentence_embedding", "float32", ("BATCH", hidden_size))
        function_metadata = TensorMetadata("function_embedding", "float32", ("BATCH", hidden_size))
        feeds = {
            "input_ids": input_ids.detach().cpu().numpy(),
            "attention_mask": attention_mask.detach().cpu().numpy(),
        }
        encoder_metadata: dict[str, ExportedOnnxComponent] = {}
        ort_embeddings: dict[str, Any] = {}
        for key in encoder_keys:
            destination = encoder_destinations[key]
            exported_paths.append(destination)
            _export_graph(
                torch,
                prefixes[key],
                (input_ids, attention_mask),
                destination,
                input_names=("input_ids", "attention_mask"),
                output_names=("sentence_embedding",),
                dynamic_axes={
                    "input_ids": {0: "BATCH", 1: "SEQUENCE"},
                    "attention_mask": {0: "BATCH", 1: "SEQUENCE"},
                    "sentence_embedding": {0: "BATCH"},
                },
            )
            _validate_exported_component_file(destination, maximum_component_bytes)
            difference, ort_embeddings[key] = _validate_and_run(
                onnx,
                onnxruntime,
                numpy,
                destination,
                encoder_inputs,
                (embedding_metadata,),
                feeds,
                sentence_embeddings[key].detach().cpu().numpy(),
                relative_tolerance,
                absolute_tolerance,
            )
            encoder_metadata[key] = _component_metadata(
                "encoder", destination, encoder_inputs, (embedding_metadata,), difference
            )

        adapter_metadata: dict[str, ExportedOnnxComponent] = {}
        chained_adapters: dict[str, Any] = {}
        for ref in adapter_refs:
            destination = adapter_destinations[ref]
            key = depth_key(adapter_depths[ref])
            exported_paths.append(destination)
            _export_graph(
                torch,
                adapters[ref],
                (sentence_embeddings[key],),
                destination,
                input_names=("sentence_embedding",),
                output_names=("function_embedding",),
                dynamic_axes={
                    "sentence_embedding": {0: "BATCH"},
                    "function_embedding": {0: "BATCH"},
                },
            )
            _validate_exported_component_file(destination, maximum_component_bytes)
            difference, _ = _validate_and_run(
                onnx,
                onnxruntime,
                numpy,
                destination,
                (embedding_metadata,),
                (function_metadata,),
                {"sentence_embedding": sentence_embeddings[key].detach().cpu().numpy()},
                function_embeddings[ref].detach().cpu().numpy(),
                relative_tolerance,
                absolute_tolerance,
            )
            adapter_metadata[ref] = _component_metadata(
                "adapter", destination, (embedding_metadata,), (function_metadata,), difference
            )
            chained_adapters[ref] = _run_onnx(
                onnxruntime, destination, {"sentence_embedding": ort_embeddings[key]}
            )

        head_metadata: dict[str, ExportedOnnxComponent] = {}
        chain_differences: dict[str, float] = {}
        for name in function_ids:
            destination = head_destinations[name]
            ref = function_adapters[name]
            head_outputs = (TensorMetadata("logits", "float32", ("BATCH", logit_widths[name])),)
            exported_paths.append(destination)
            _export_graph(
                torch,
                heads[name],
                (function_embeddings[ref],),
                destination,
                input_names=("function_embedding",),
                output_names=("logits",),
                dynamic_axes={"function_embedding": {0: "BATCH"}, "logits": {0: "BATCH"}},
            )
            _validate_exported_component_file(destination, maximum_component_bytes)
            expected_logits = head_logits[name].detach().cpu().numpy()
            difference, _ = _validate_and_run(
                onnx,
                onnxruntime,
                numpy,
                destination,
                (function_metadata,),
                head_outputs,
                {"function_embedding": function_embeddings[ref].detach().cpu().numpy()},
                expected_logits,
                relative_tolerance,
                absolute_tolerance,
            )
            head_metadata[name] = _component_metadata(
                "head", destination, (function_metadata,), head_outputs, difference
            )
            chain_logits = _run_onnx(
                onnxruntime, destination, {"function_embedding": chained_adapters[ref]}
            )
            chain_differences[name] = _assert_parity(
                numpy,
                f"encoder-adapter-head chain for {name}",
                expected_logits,
                chain_logits,
                relative_tolerance,
                absolute_tolerance,
            )
    except BaseException:
        for path in exported_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        for module, training in training_states:
            module.training = training

    return ExportedRoutedApplicationComponents(
        encoders=encoder_metadata,
        adapters=adapter_metadata,
        heads=head_metadata,
        parity_batch_size=int(input_ids.shape[0]),
        parity_sequence_length=int(input_ids.shape[1]),
        chain_maximum_absolute_differences=chain_differences,
    )


def _validate_parity_tolerance(value: float, name: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"ONNX parity {name} must be a real number")
    numeric_value = float(value)
    if not isfinite(numeric_value) or not 0.0 <= numeric_value <= maximum:
        raise ValueError(f"ONNX parity {name} must be finite and between 0 and {maximum}")
    return numeric_value


def _validate_maximum_component_bytes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("maximum_component_bytes must be an integer")
    if value <= 0:
        raise ValueError("maximum_component_bytes must be positive")
    return value


def _prepare_destinations(
    encoder_path: str | Path,
    adapter_path: str | Path,
    head_path: str | Path,
) -> tuple[Path, Path, Path]:
    prepared = _prepare_destination_paths([encoder_path, adapter_path, head_path])
    return prepared[0], prepared[1], prepared[2]


def _prepare_destination_paths(values: Sequence[str | Path]) -> tuple[Path, ...]:
    paths = tuple(Path(os.path.abspath(Path(value).expanduser())) for value in values)
    if len(set(paths)) != len(paths):
        raise ValueError("encoder, adapter, and head ONNX destinations must be distinct")
    destination_identities: set[tuple[int, int, str]] = set()
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite ONNX destination: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        parent_status = path.parent.stat()
        identity = (parent_status.st_dev, parent_status.st_ino, path.name)
        if identity in destination_identities:
            raise ValueError("encoder, adapter, and head ONNX destinations must be distinct")
        destination_identities.add(identity)
    return paths


def _validate_exported_component_file(path: Path, maximum_component_bytes: int) -> None:
    file_status = path.lstat()
    if not S_ISREG(file_status.st_mode):
        path.unlink(missing_ok=True)
        raise ValueError(f"exported ONNX component is not a regular file: {path}")
    if file_status.st_size > maximum_component_bytes:
        byte_length = file_status.st_size
        path.unlink(missing_ok=True)
        raise ValueError(
            f"exported ONNX component {path} is {byte_length} bytes, exceeding "
            f"maximum_component_bytes={maximum_component_bytes}"
        )


def _validate_parity_inputs(torch: Any, input_ids: Any, attention_mask: Any) -> None:
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise TypeError("ONNX parity input_ids and attention_mask must be PyTorch tensors")
    if input_ids.dtype != torch.int64 or attention_mask.dtype != torch.int64:
        raise TypeError("ONNX parity input_ids and attention_mask must have dtype int64")
    if input_ids.ndim != 2 or tuple(input_ids.shape) != tuple(attention_mask.shape):
        raise ValueError("ONNX parity inputs must have identical [BATCH, SEQUENCE] shapes")
    if input_ids.shape[0] != 1:
        raise ValueError("ONNX parity batch size must be exactly one")
    if not 1 <= input_ids.shape[1] <= MAXIMUM_PARITY_SEQUENCE_LENGTH:
        raise ValueError(
            f"ONNX parity sequence length must be between 1 and {MAXIMUM_PARITY_SEQUENCE_LENGTH}"
        )


def _validate_float_output(
    torch: Any,
    value: Any,
    label: str,
    *,
    expected_batch: int,
    expected_width: int | None = None,
) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{label} must emit one rank-two tensor")
    if value.dtype != torch.float32:
        raise ValueError(f"{label} must emit float32")
    if value.shape[0] != expected_batch or value.shape[1] <= 0:
        raise ValueError(f"{label} emitted an invalid ABI shape")
    if expected_width is not None and value.shape[1] != expected_width:
        raise ValueError(f"{label} output width does not match its configuration")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} emitted non-finite values on the parity batch")


def _export_graph(
    torch: Any,
    module: Any,
    arguments: tuple[Any, ...],
    path: Path,
    *,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> None:
    torch.onnx.export(
        module,
        arguments,
        str(path),
        dynamo=False,
        opset_version=ONNX_OPSET,
        external_data=False,
        export_params=True,
        keep_initializers_as_inputs=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
    )


def _validate_and_run(
    onnx: Any,
    onnxruntime: Any,
    numpy: Any,
    path: Path,
    inputs: tuple[TensorMetadata, ...],
    outputs: tuple[TensorMetadata, ...],
    feeds: dict[str, Any],
    expected: Any,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[float, Any]:
    _validate_onnx_container(onnx, path)
    actual = _run_onnx(onnxruntime, path, feeds, inputs=inputs, outputs=outputs)
    difference = _assert_parity(
        numpy,
        path.name,
        expected,
        actual,
        relative_tolerance,
        absolute_tolerance,
    )
    return difference, actual


def _validate_onnx_container(onnx: Any, path: Path) -> None:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    if model.functions:
        raise ValueError(f"{path} contains unsupported ONNX local functions")
    if model.training_info:
        raise ValueError(f"{path} contains unsupported ONNX training graphs")

    standard_versions = [
        int(item.version) for item in model.opset_import if item.domain in _STANDARD_ONNX_DOMAINS
    ]
    if standard_versions != [ONNX_OPSET]:
        raise ValueError(f"{path} must import exactly ONNX opset {ONNX_OPSET}")
    for item in model.opset_import:
        if item.domain not in _STANDARD_ONNX_DOMAINS:
            raise ValueError(f"{path} imports unsupported ONNX domain {item.domain!r}")

    for message in _walk_protobuf_messages(model):
        message_type = message.DESCRIPTOR.full_name
        if message_type == "onnx.NodeProto" and message.domain not in _STANDARD_ONNX_DOMAINS:
            raise ValueError(f"{path} uses unsupported ONNX domain {message.domain!r}")
        if message_type == "onnx.TensorProto":
            if message.external_data or message.data_location == onnx.TensorProto.EXTERNAL:
                raise ValueError(f"{path} contains unsupported external tensor data")


def _walk_protobuf_messages(message: Any) -> Any:
    yield message
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        if field.is_repeated:
            for child in value:
                yield from _walk_protobuf_messages(child)
        else:
            yield from _walk_protobuf_messages(value)


def _run_onnx(
    onnxruntime: Any,
    path: Path,
    feeds: dict[str, Any],
    *,
    inputs: tuple[TensorMetadata, ...] | None = None,
    outputs: tuple[TensorMetadata, ...] | None = None,
) -> Any:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session = onnxruntime.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if inputs is not None and outputs is not None:
        _validate_runtime_abi(session, path, inputs, outputs)
    result = session.run(None, feeds)
    if len(result) != 1:
        raise ValueError(f"{path} must emit exactly one ONNX output")
    return result[0]


def _validate_runtime_abi(
    session: Any,
    path: Path,
    inputs: tuple[TensorMetadata, ...],
    outputs: tuple[TensorMetadata, ...],
) -> None:
    actual_inputs = tuple(session.get_inputs())
    actual_outputs = tuple(session.get_outputs())
    if len(actual_inputs) != len(inputs) or len(actual_outputs) != len(outputs):
        raise ValueError(f"{path} has an unexpected number of ONNX inputs or outputs")
    for actual, expected in zip(
        (*actual_inputs, *actual_outputs), (*inputs, *outputs), strict=True
    ):
        expected_type = "tensor(int64)" if expected.dtype == "int64" else "tensor(float)"
        if actual.name != expected.name or actual.type != expected_type:
            raise ValueError(f"{path} tensor ABI does not match {expected.name!r}")
        if tuple(actual.shape) != expected.shape:
            raise ValueError(
                f"{path} tensor {expected.name!r} shape {tuple(actual.shape)!r} "
                f"does not match {expected.shape!r}"
            )


def _assert_parity(
    numpy: Any,
    label: str,
    expected: Any,
    actual: Any,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> float:
    if actual.dtype != numpy.float32 or actual.shape != expected.shape:
        raise ValueError(f"{label} ONNX output has an unexpected dtype or shape")
    if not bool(numpy.isfinite(actual).all()):
        raise ValueError(f"{label} ONNX output contains non-finite values")
    try:
        numpy.testing.assert_allclose(
            actual,
            expected,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    except AssertionError as error:
        raise ValueError(f"{label} ONNX output failed PyTorch parity") from error
    return float(numpy.max(numpy.abs(actual - expected), initial=0.0))


def _component_metadata(
    role: ComponentRole,
    path: Path,
    inputs: tuple[TensorMetadata, ...],
    outputs: tuple[TensorMetadata, ...],
    maximum_absolute_difference: float,
) -> ExportedOnnxComponent:
    byte_length, digest = _file_identity(path)
    return ExportedOnnxComponent(
        role=role,
        path=path,
        byte_length=byte_length,
        sha256=digest,
        opset=ONNX_OPSET,
        inputs=inputs,
        outputs=outputs,
        external_data=False,
        maximum_absolute_difference=maximum_absolute_difference,
    )


__all__ = [
    "DEFAULT_MAXIMUM_COMPONENT_BYTES",
    "DEFAULT_PARITY_ABSOLUTE_TOLERANCE",
    "DEFAULT_PARITY_RELATIVE_TOLERANCE",
    "MAXIMUM_PARITY_ABSOLUTE_TOLERANCE",
    "MAXIMUM_PARITY_RELATIVE_TOLERANCE",
    "MAXIMUM_PARITY_SEQUENCE_LENGTH",
    "ONNX_OPSET",
    "ExportedApplicationComponents",
    "ExportedOnnxComponent",
    "ExportedOnnxComponents",
    "TensorMetadata",
    "export_application_components",
    "export_onnx_components",
]
