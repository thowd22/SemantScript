from __future__ import annotations

import importlib.util
from dataclasses import FrozenInstanceError
from hashlib import sha256
from types import SimpleNamespace

import pytest

import semantscript_model.export as export_module
from semantscript_model.export import (
    MAXIMUM_PARITY_ABSOLUTE_TOLERANCE,
    MAXIMUM_PARITY_RELATIVE_TOLERANCE,
    ONNX_OPSET,
    QuantizedOnnxComponent,
    export_onnx_components,
    quantize_onnx_encoder,
)
from semantscript_model.heads import ClassificationHead, HeadConfig

EXPORT_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "onnx", "onnxruntime", "torch")
)


@pytest.mark.skipif(
    not EXPORT_DEPENDENCIES_AVAILABLE,
    reason="optional ONNX export dependencies are not installed",
)
@pytest.mark.parametrize(
    ("kind", "cardinality", "architecture", "output_width"),
    [
        ("categorical-softmax", 3, "linear", 3),
        ("binary-sigmoid", 2, "mlp", 1),
    ],
)
def test_exports_split_components_with_validated_abi_and_parity(
    tmp_path,
    kind: str,
    cardinality: int,
    architecture: str,
    output_width: int,
) -> None:
    import numpy
    import onnx
    import onnxruntime
    import torch

    from semantscript_model.encoder import SentenceEncoder

    class FakeTokenEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.embedding = torch.nn.Embedding(32, 4)

        def forward(self, *, input_ids, attention_mask, return_dict):
            del attention_mask, return_dict
            return {"last_hidden_state": self.embedding(input_ids)}

    torch.manual_seed(19)
    encoder = SentenceEncoder(encoder=FakeTokenEncoder())
    head = ClassificationHead(
        HeadConfig(
            input_size=4,
            kind=kind,  # type: ignore[arg-type]
            cardinality=cardinality,
            architecture=architecture,  # type: ignore[arg-type]
            mlp_hidden_size=5 if architecture == "mlp" else None,
        )
    )
    encoder.train()
    head.train()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.int64)
    paths = (
        tmp_path / "models" / "encoder" / "model.onnx",
        tmp_path / "models" / "adapters" / "application.onnx",
        tmp_path / "models" / "heads" / "function" / "head-000.onnx",
    )

    exported = export_onnx_components(
        encoder,
        head,
        input_ids,
        attention_mask,
        encoder_path=paths[0],
        adapter_path=paths[1],
        head_path=paths[2],
    )

    assert encoder.training is True
    assert head.training is True
    assert exported.parity_batch_size == 1
    assert exported.parity_sequence_length == 4
    assert exported.chain_maximum_absolute_difference < 1e-4
    assert exported.encoder.inputs[0].shape == ("BATCH", "SEQUENCE")
    assert exported.encoder.outputs[0].name == "sentence_embedding"
    assert exported.encoder.outputs[0].shape == ("BATCH", 4)
    assert exported.adapter.inputs[0].name == "sentence_embedding"
    assert exported.adapter.outputs[0].name == "function_embedding"
    assert exported.head.inputs[0].name == "function_embedding"
    assert exported.head.outputs[0].shape == ("BATCH", output_width)

    for component, path in zip(
        (exported.encoder, exported.adapter, exported.head), paths, strict=True
    ):
        assert component.path == path.resolve()
        assert component.opset == ONNX_OPSET
        assert component.external_data is False
        assert component.byte_length == path.stat().st_size
        assert component.sha256 == sha256(path.read_bytes()).hexdigest()
        assert len(component.sha256) == 64
        model = onnx.load(path, load_external_data=False)
        assert [(item.domain, item.version) for item in model.opset_import] == [("", 17)]
        assert not model.functions
        assert not model.training_info
        assert all(not tensor.external_data for tensor in model.graph.initializer)

    encoder_session = onnxruntime.InferenceSession(
        str(paths[0]), providers=["CPUExecutionProvider"]
    )
    longer_ids = numpy.asarray([[5, 6]], dtype=numpy.int64)
    longer_mask = numpy.asarray([[1, 1]], dtype=numpy.int64)
    longer_embedding = encoder_session.run(
        None, {"input_ids": longer_ids, "attention_mask": longer_mask}
    )[0]
    assert longer_embedding.shape == (1, 4)
    assert longer_embedding.dtype == numpy.float32

    with pytest.raises(FrozenInstanceError):
        exported.encoder.opset = 18  # type: ignore[misc]


@pytest.mark.skipif(
    not EXPORT_DEPENDENCIES_AVAILABLE,
    reason="optional ONNX export dependencies are not installed",
)
def test_export_rejects_destination_collisions_before_writing(tmp_path) -> None:
    import torch

    collision = tmp_path / "same.onnx"
    tensor = torch.ones((1, 1), dtype=torch.int64)

    with pytest.raises(ValueError, match="destinations must be distinct"):
        export_onnx_components(
            object(),
            object(),
            tensor,
            tensor,
            encoder_path=collision,
            adapter_path=collision,
            head_path=tmp_path / "head.onnx",
        )

    assert not collision.exists()


@pytest.mark.skipif(
    not EXPORT_DEPENDENCIES_AVAILABLE,
    reason="optional ONNX export dependencies are not installed",
)
def test_export_rejects_existing_destination_before_writing(tmp_path) -> None:
    import torch

    existing = tmp_path / "encoder.onnx"
    existing.write_bytes(b"owned by caller")
    tensor = torch.ones((1, 1), dtype=torch.int64)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_onnx_components(
            object(),
            object(),
            tensor,
            tensor,
            encoder_path=existing,
            adapter_path=tmp_path / "adapter.onnx",
            head_path=tmp_path / "head.onnx",
        )

    assert existing.read_bytes() == b"owned by caller"


@pytest.mark.skipif(
    not EXPORT_DEPENDENCIES_AVAILABLE,
    reason="optional ONNX export dependencies are not installed",
)
def test_export_rejects_oversized_component_before_parser_or_runtime(tmp_path, monkeypatch) -> None:
    import torch

    class Encoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            del attention_mask
            return input_ids.to(dtype=torch.float32)

    class Head(torch.nn.Module):
        def forward(self, sentence_embedding):
            return sentence_embedding

    def write_oversized_component(
        torch_module,
        module,
        arguments,
        path,
        *,
        input_names,
        output_names,
        dynamic_axes=None,
    ) -> None:
        del (
            torch_module,
            module,
            arguments,
            input_names,
            output_names,
            dynamic_axes,
        )
        path.write_bytes(b"oversized")

    def unexpected_parser_or_runtime_call(*args, **kwargs):
        del args, kwargs
        pytest.fail("oversized component reached the ONNX parser or runtime")

    monkeypatch.setattr(export_module, "_export_graph", write_oversized_component)
    monkeypatch.setattr(
        export_module, "_validate_onnx_container", unexpected_parser_or_runtime_call
    )
    monkeypatch.setattr(export_module, "_run_onnx", unexpected_parser_or_runtime_call)

    tensor = torch.ones((1, 1), dtype=torch.int64)
    paths = (
        tmp_path / "encoder.onnx",
        tmp_path / "adapter.onnx",
        tmp_path / "head.onnx",
    )
    with pytest.raises(ValueError, match="exceeding maximum_component_bytes=4"):
        export_onnx_components(
            Encoder(),
            Head(),
            tensor,
            tensor,
            encoder_path=paths[0],
            adapter_path=paths[1],
            head_path=paths[2],
            maximum_component_bytes=4,
        )

    assert not any(path.exists() for path in paths)


@pytest.mark.parametrize("maximum_component_bytes", [True, 1.5, "1024", None])
def test_export_rejects_non_integer_component_byte_quota(tmp_path, maximum_component_bytes) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        export_onnx_components(
            object(),
            object(),
            object(),
            object(),
            encoder_path=tmp_path / "encoder.onnx",
            adapter_path=tmp_path / "adapter.onnx",
            head_path=tmp_path / "head.onnx",
            maximum_component_bytes=maximum_component_bytes,
        )


@pytest.mark.parametrize("maximum_component_bytes", [0, -1])
def test_export_rejects_nonpositive_component_byte_quota(
    tmp_path, maximum_component_bytes: int
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        export_onnx_components(
            object(),
            object(),
            object(),
            object(),
            encoder_path=tmp_path / "encoder.onnx",
            adapter_path=tmp_path / "adapter.onnx",
            head_path=tmp_path / "head.onnx",
            maximum_component_bytes=maximum_component_bytes,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "maximum"),
    [
        (
            "relative_tolerance",
            MAXIMUM_PARITY_RELATIVE_TOLERANCE * 2,
            MAXIMUM_PARITY_RELATIVE_TOLERANCE,
        ),
        (
            "absolute_tolerance",
            MAXIMUM_PARITY_ABSOLUTE_TOLERANCE * 2,
            MAXIMUM_PARITY_ABSOLUTE_TOLERANCE,
        ),
    ],
)
def test_export_rejects_parity_tolerance_above_hard_limit(
    tmp_path, keyword: str, value: float, maximum: float
) -> None:
    with pytest.raises(ValueError, match=f"between 0 and {maximum}"):
        export_onnx_components(
            object(),
            object(),
            object(),
            object(),
            encoder_path=tmp_path / "encoder.onnx",
            adapter_path=tmp_path / "adapter.onnx",
            head_path=tmp_path / "head.onnx",
            **{keyword: value},
        )


@pytest.mark.skipif(
    not EXPORT_DEPENDENCIES_AVAILABLE,
    reason="optional ONNX export dependencies are not installed",
)
def test_quantizes_exported_encoder_into_standard_int8_graph(tmp_path) -> None:
    import onnx
    import torch

    from semantscript_model.encoder import SentenceEncoder

    class LinearTokenEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.embedding = torch.nn.Embedding(16, 8)
            self.projection = torch.nn.Linear(8, 8)

        def forward(self, *, input_ids, attention_mask, return_dict):
            del attention_mask
            assert return_dict is True
            return SimpleNamespace(last_hidden_state=self.projection(self.embedding(input_ids)))

    torch.manual_seed(3)
    encoder = SentenceEncoder(encoder=LinearTokenEncoder())
    head = ClassificationHead(HeadConfig(input_size=8, kind="categorical-softmax", cardinality=3))
    export_onnx_components(
        encoder,
        head,
        torch.tensor([[1, 2, 3]], dtype=torch.int64),
        torch.tensor([[1, 1, 1]], dtype=torch.int64),
        encoder_path=tmp_path / "encoder.onnx",
        adapter_path=tmp_path / "adapter.onnx",
        head_path=tmp_path / "head.onnx",
    )

    component = quantize_onnx_encoder(tmp_path / "encoder.onnx", tmp_path / "encoder-int8.onnx")

    assert isinstance(component, QuantizedOnnxComponent)
    assert component.role == "encoder"
    assert component.method == "dynamic" and component.weight_type == "int8"
    assert component.quantized_matmul_count >= 1
    assert component.opset == ONNX_OPSET
    assert component.path == tmp_path / "encoder-int8.onnx"
    assert component.byte_length == (tmp_path / "encoder-int8.onnx").stat().st_size
    assert component.sha256 == sha256((tmp_path / "encoder-int8.onnx").read_bytes()).hexdigest()
    assert component.source_sha256 == sha256((tmp_path / "encoder.onnx").read_bytes()).hexdigest()
    model = onnx.load(str(tmp_path / "encoder-int8.onnx"))
    assert [(item.domain, item.version) for item in model.opset_import] == [("", ONNX_OPSET)]
    operators = {node.op_type for node in model.graph.node}
    assert "MatMulInteger" in operators
    assert {node.domain for node in model.graph.node} == {""}

    # The quantized graph keeps the runtime ABI: same inputs, one float32 output.
    import numpy
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(tmp_path / "encoder-int8.onnx"), providers=["CPUExecutionProvider"]
    )
    output = session.run(
        None,
        {
            "input_ids": numpy.array([[1, 2, 3]], dtype=numpy.int64),
            "attention_mask": numpy.array([[1, 1, 1]], dtype=numpy.int64),
        },
    )
    assert len(output) == 1 and output[0].dtype == numpy.float32 and output[0].shape == (1, 8)

    with pytest.raises(FileExistsError):
        quantize_onnx_encoder(tmp_path / "encoder.onnx", tmp_path / "encoder-int8.onnx")
    with pytest.raises(ValueError, match="weight_type"):
        quantize_onnx_encoder(
            tmp_path / "encoder.onnx", tmp_path / "other.onnx", weight_type="int4"
        )
    with pytest.raises(TypeError, match="per_channel"):
        quantize_onnx_encoder(
            tmp_path / "encoder.onnx",
            tmp_path / "other.onnx",
            per_channel="yes",  # type: ignore[arg-type]
        )
    assert not (tmp_path / "other.onnx").exists()
    with pytest.raises(FileNotFoundError):
        quantize_onnx_encoder(tmp_path / "missing.onnx", tmp_path / "other.onnx")


@pytest.mark.skipif(
    not EXPORT_DEPENDENCIES_AVAILABLE,
    reason="optional ONNX export dependencies are not installed",
)
def test_quantizer_removes_destination_when_nothing_is_quantizable(tmp_path) -> None:
    import torch

    from semantscript_model.encoder import SentenceEncoder

    class EmbeddingOnlyEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.embedding = torch.nn.Embedding(8, 4)

        def forward(self, *, input_ids, attention_mask, return_dict):
            del attention_mask
            assert return_dict is True
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    encoder = SentenceEncoder(encoder=EmbeddingOnlyEncoder())
    head = ClassificationHead(HeadConfig(input_size=4, kind="categorical-softmax", cardinality=2))
    export_onnx_components(
        encoder,
        head,
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.tensor([[1, 1]], dtype=torch.int64),
        encoder_path=tmp_path / "encoder.onnx",
        adapter_path=tmp_path / "adapter.onnx",
        head_path=tmp_path / "head.onnx",
    )

    with pytest.raises(ValueError, match="no MatMulInteger"):
        quantize_onnx_encoder(tmp_path / "encoder.onnx", tmp_path / "encoder-int8.onnx")
    assert not (tmp_path / "encoder-int8.onnx").exists()
