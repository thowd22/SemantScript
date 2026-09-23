"""Low-level ONNX export and parity validation for model components.

The deployment artifact uses three independently loadable graphs.  This module
deliberately imports the optional export stack only when an export is requested,
so importing :mod:`semantscript_model` remains safe in compiler-only installs.
"""

from __future__ import annotations

import os
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


def export_onnx_components(
    encoder: Any,
    head: Any,
    input_ids: Any,
    attention_mask: Any,
    *,
    encoder_path: str | Path,
    adapter_path: str | Path,
    head_path: str | Path,
    relative_tolerance: float = DEFAULT_PARITY_RELATIVE_TOLERANCE,
    absolute_tolerance: float = DEFAULT_PARITY_ABSOLUTE_TOLERANCE,
    maximum_component_bytes: int = DEFAULT_MAXIMUM_COMPONENT_BYTES,
) -> ExportedOnnxComponents:
    """Export, validate, and parity-check the three model ABI components.

    The parity batch is intentionally restricted to one bounded sequence, which
    matches the v1 runtime.  Destination files must not already exist and must
    resolve to three distinct paths.
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

    torch, onnx, onnxruntime, numpy = _load_export_dependencies()
    destinations = _prepare_destinations(encoder_path, adapter_path, head_path)
    _validate_parity_inputs(torch, input_ids, attention_mask)

    encoder_inputs = (
        TensorMetadata("input_ids", "int64", ("BATCH", "SEQUENCE")),
        TensorMetadata("attention_mask", "int64", ("BATCH", "SEQUENCE")),
    )

    training_states = tuple(
        (module, bool(module.training)) for root in (encoder, head) for module in root.modules()
    )
    adapter = _IdentityAdapter.create(torch)
    exported_paths: list[Path] = []
    try:
        encoder.eval()
        head.eval()
        # ``no_grad`` keeps intermediate tensors usable by the legacy tracer;
        # tensors created by ``inference_mode`` cannot subsequently be traced
        # through a parameterized head.
        with torch.no_grad():
            sentence_embedding = encoder(input_ids, attention_mask)
            _validate_float_output(torch, sentence_embedding, "encoder", expected_batch=1)
            hidden_size = int(sentence_embedding.shape[1])
            function_embedding = adapter(sentence_embedding)
            logits = head(function_embedding)
            _validate_float_output(
                torch,
                logits,
                "head",
                expected_batch=1,
                expected_width=getattr(getattr(head, "config", None), "output_size", None),
            )
            logit_width = int(logits.shape[1])

        encoder_outputs = (TensorMetadata("sentence_embedding", "float32", ("BATCH", hidden_size)),)
        adapter_inputs = (TensorMetadata("sentence_embedding", "float32", ("BATCH", hidden_size)),)
        adapter_outputs = (TensorMetadata("function_embedding", "float32", ("BATCH", hidden_size)),)
        head_inputs = (TensorMetadata("function_embedding", "float32", ("BATCH", hidden_size)),)
        head_outputs = (TensorMetadata("logits", "float32", ("BATCH", logit_width)),)

        exported_paths.append(destinations[0])
        _export_graph(
            torch,
            encoder,
            (input_ids, attention_mask),
            destinations[0],
            input_names=("input_ids", "attention_mask"),
            output_names=("sentence_embedding",),
            dynamic_axes={
                "input_ids": {0: "BATCH", 1: "SEQUENCE"},
                "attention_mask": {0: "BATCH", 1: "SEQUENCE"},
                "sentence_embedding": {0: "BATCH"},
            },
        )
        _validate_exported_component_file(destinations[0], maximum_component_bytes)
        encoder_difference, ort_sentence_embedding = _validate_and_run(
            onnx,
            onnxruntime,
            numpy,
            destinations[0],
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
            destinations[0],
            encoder_inputs,
            encoder_outputs,
            encoder_difference,
        )

        exported_paths.append(destinations[1])
        _export_graph(
            torch,
            adapter,
            (sentence_embedding,),
            destinations[1],
            input_names=("sentence_embedding",),
            output_names=("function_embedding",),
            dynamic_axes={
                "sentence_embedding": {0: "BATCH"},
                "function_embedding": {0: "BATCH"},
            },
        )
        _validate_exported_component_file(destinations[1], maximum_component_bytes)
        adapter_difference, _ = _validate_and_run(
            onnx,
            onnxruntime,
            numpy,
            destinations[1],
            adapter_inputs,
            adapter_outputs,
            {"sentence_embedding": sentence_embedding.detach().cpu().numpy()},
            function_embedding.detach().cpu().numpy(),
            relative_tolerance,
            absolute_tolerance,
        )
        adapter_metadata = _component_metadata(
            "adapter",
            destinations[1],
            adapter_inputs,
            adapter_outputs,
            adapter_difference,
        )

        exported_paths.append(destinations[2])
        _export_graph(
            torch,
            head,
            (function_embedding,),
            destinations[2],
            input_names=("function_embedding",),
            output_names=("logits",),
            dynamic_axes={"function_embedding": {0: "BATCH"}, "logits": {0: "BATCH"}},
        )
        _validate_exported_component_file(destinations[2], maximum_component_bytes)
        head_difference, _ = _validate_and_run(
            onnx,
            onnxruntime,
            numpy,
            destinations[2],
            head_inputs,
            head_outputs,
            {"function_embedding": function_embedding.detach().cpu().numpy()},
            logits.detach().cpu().numpy(),
            relative_tolerance,
            absolute_tolerance,
        )
        head_metadata = _component_metadata(
            "head",
            destinations[2],
            head_inputs,
            head_outputs,
            head_difference,
        )

        # Exercise the actual adjacent edge, rather than only independent graph inputs.
        chained_adapter = _run_onnx(
            onnxruntime,
            destinations[1],
            {"sentence_embedding": ort_sentence_embedding},
        )
        chain_logits = _run_onnx(
            onnxruntime,
            destinations[2],
            {"function_embedding": chained_adapter},
        )
        chain_difference = _assert_parity(
            numpy,
            "encoder-adapter-head chain",
            logits.detach().cpu().numpy(),
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

    return ExportedOnnxComponents(
        encoder=encoder_metadata,
        adapter=adapter_metadata,
        head=head_metadata,
        parity_batch_size=1,
        parity_sequence_length=int(input_ids.shape[1]),
        chain_maximum_absolute_difference=chain_difference,
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
    paths = tuple(
        Path(os.path.abspath(Path(value).expanduser()))
        for value in (
            encoder_path,
            adapter_path,
            head_path,
        )
    )
    if len(set(paths)) != 3:
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
    digest = sha256()
    byte_length = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_length += len(chunk)
    return ExportedOnnxComponent(
        role=role,
        path=path,
        byte_length=byte_length,
        sha256=digest.hexdigest(),
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
    "ExportedOnnxComponent",
    "ExportedOnnxComponents",
    "TensorMetadata",
    "export_onnx_components",
]
