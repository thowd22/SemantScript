"""Depth-routed encoders and per-domain adapters (TASK-6.7)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from semantscript_model import (  # noqa: E402
    AdapterConfig,
    ApplicationAdapter,
    ClassificationHead,
    HeadConfig,
    PrefixSentenceEncoder,
    SentenceEncoder,
    SharedEncoderApplication,
)


class LayeredTokenEncoder(torch.nn.Module):
    """A toy encoder whose layer i adds i to every hidden value and exposes hidden states."""

    def __init__(self, hidden: int = 8, layers: int = 4, final_norm: bool = True) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, num_hidden_layers=layers)
        self.embedding = torch.nn.Embedding(16, hidden)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(hidden, hidden) for _ in range(layers))
        if final_norm:
            self.final_norm = torch.nn.LayerNorm(hidden)
        self.calls = 0

    def forward(self, *, input_ids, attention_mask, return_dict, output_hidden_states=False):
        del attention_mask
        assert return_dict is True
        self.calls += 1
        state = self.embedding(input_ids)
        states = [state]
        for layer in self.layers:
            state = torch.tanh(layer(state))
            states.append(state)
        last = self.final_norm(state) if hasattr(self, "final_norm") else state
        return SimpleNamespace(
            last_hidden_state=last,
            hidden_states=tuple(states) if output_hidden_states else None,
        )


def head(cardinality: int = 3, hidden: int = 8) -> ClassificationHead:
    return ClassificationHead(
        HeadConfig(input_size=hidden, kind="categorical-softmax", cardinality=cardinality)
    )


def test_prefix_depth_reads_the_normalized_hidden_state_after_that_many_layers() -> None:
    torch.manual_seed(3)
    token_encoder = LayeredTokenEncoder(layers=4)
    encoder = SentenceEncoder(encoder=token_encoder)
    assert encoder.layer_count == 4
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[1, 1, 0]])
    full = encoder(ids, mask)
    torch.testing.assert_close(encoder(ids, mask, depth=4), full)  # the full stack is depth None
    assert encoder.validate_depth(4) is None and encoder.validate_depth(2) == 2
    shallow = encoder(ids, mask, depth=2)
    assert shallow.shape == full.shape and not torch.allclose(shallow, full)
    # By hand: masked mean of final_norm(hidden_states[2]).
    out = token_encoder(
        input_ids=ids, attention_mask=mask, return_dict=True, output_hidden_states=True
    )
    expected = token_encoder.final_norm(out.hidden_states[2])
    expected = (expected * mask.unsqueeze(-1)).sum(dim=1) / 2
    torch.testing.assert_close(shallow, expected)
    prefix = PrefixSentenceEncoder(encoder, 2)
    torch.testing.assert_close(prefix(ids, mask), shallow)
    assert prefix.depth == 2 and prefix.hidden_size == 8
    for bad in (0, 5, True):
        with pytest.raises(ValueError):
            encoder(ids, mask, depth=bad)
    no_layers = SentenceEncoder(encoder=LayeredTokenEncoder(layers=4))
    no_layers.layer_count = None
    with pytest.raises(ValueError, match="depth routing is unavailable"):
        no_layers(ids, mask, depth=1)
    # Without a final norm the raw hidden state is pooled.
    plain = SentenceEncoder(encoder=LayeredTokenEncoder(layers=3, final_norm=False))
    assert plain(ids, mask, depth=1).shape == (1, 8)


def test_application_routes_functions_to_domain_adapters_at_their_depths() -> None:
    torch.manual_seed(5)
    encoder = SentenceEncoder(encoder=LayeredTokenEncoder(layers=4))
    shallow = ApplicationAdapter(AdapterConfig(hidden_size=8, bottleneck_size=4))
    deep = ApplicationAdapter(AdapterConfig(hidden_size=8, bottleneck_size=4))
    with torch.no_grad():
        shallow.up.weight.normal_()
        deep.up.weight.normal_()
    application = SharedEncoderApplication(
        encoder,
        adapters={"adapter.app.refund": shallow, "adapter.app.fraud": deep},
        adapter_depths={"adapter.app.refund": 2, "adapter.app.fraud": None},
        heads={"nf_a": head(3), "nf_b": head(2)},
        function_adapters={"nf_a": "adapter.app.refund", "nf_b": "adapter.app.fraud"},
    )
    assert application.adapter_refs == ("adapter.app.refund", "adapter.app.fraud")
    assert application.adapter_ref_of("nf_a") == "adapter.app.refund"
    assert application.depth_of("nf_a") == 2 and application.depth_of("nf_b") is None
    assert application.adapter_for("adapter.app.fraud") is deep
    # nn.Module rewraps the property's AttributeError, so getattr's default still applies.
    with pytest.raises(AttributeError, match="adapter"):
        application.adapter  # noqa: B018
    assert getattr(application, "adapter", None) is None

    ids = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[1, 1, 1]])
    view_a = application.function_model("nf_a")
    assert view_a.depth == 2 and view_a.adapter is shallow
    expected_a = head_logits = view_a(ids, mask)
    torch.testing.assert_close(
        expected_a, application.heads["nf_a"](shallow(encoder(ids, mask, depth=2)))
    )
    outputs = application(ids, mask)
    torch.testing.assert_close(outputs["nf_a"], head_logits)
    torch.testing.assert_close(outputs["nf_b"], application.function_model("nf_b")(ids, mask))
    torch.testing.assert_close(application.embed(ids, mask, "nf_b"), deep(encoder(ids, mask)))
    with pytest.raises(ValueError, match="function_id is required"):
        application.embed(ids, mask)

    # A third domain and a head on it can be attached later; the others are untouched.
    late = ApplicationAdapter(AdapterConfig(hidden_size=8, bottleneck_size=4))
    application.add_adapter("adapter.app.late", late, depth=1)
    application.add_head("nf_c", head(4), adapter_ref="adapter.app.late")
    assert application.depth_of("nf_c") == 1
    with pytest.raises(ValueError, match="adapter_ref is required"):
        application.add_head("nf_d", head(2))
    with pytest.raises(ValueError, match="already attached"):
        application.add_adapter("adapter.app.late", late)
    with pytest.raises(ValueError, match="unknown adapters"):
        SharedEncoderApplication(encoder, adapters={"a": late}, adapter_depths={"b": 1})
    with pytest.raises(ValueError, match="exactly one of adapter or adapters"):
        SharedEncoderApplication(encoder)
    # State dict keys survive dotted adapter refs.
    assert any(key.startswith("adapters.adapter__app__refund.") for key in application.state_dict())


def test_single_adapter_application_keeps_its_original_shape() -> None:
    encoder = SentenceEncoder(encoder=LayeredTokenEncoder(layers=2))
    adapter = ApplicationAdapter(AdapterConfig(hidden_size=8, bottleneck_size=4))
    application = SharedEncoderApplication(encoder, adapter, {"nf_a": head(3)})
    assert application.adapter is adapter
    assert application.adapter_refs == ("adapter.application",)
    application.add_head("nf_b", head(2))
    assert application.adapter_ref_of("nf_b") == "adapter.application"
    ids = torch.tensor([[1, 2]])
    mask = torch.tensor([[1, 1]])
    torch.testing.assert_close(application.embed(ids, mask), adapter(encoder(ids, mask)))
    assert set(application(ids, mask)) == {"nf_a", "nf_b"}


def test_routed_export_writes_one_encoder_per_depth_and_one_adapter_per_domain(tmp_path) -> None:
    import numpy
    import onnxruntime

    from semantscript_model import depth_key, export_routed_application_components

    torch.manual_seed(7)
    encoder = SentenceEncoder(encoder=LayeredTokenEncoder(layers=3))
    shallow = ApplicationAdapter(AdapterConfig(hidden_size=8, bottleneck_size=4))
    deep = ApplicationAdapter(AdapterConfig(hidden_size=8, bottleneck_size=4))
    with torch.no_grad():
        shallow.up.weight.normal_()
        deep.up.weight.normal_()
    application = SharedEncoderApplication(
        encoder,
        adapters={"adapter.app.refund": shallow, "adapter.app.fraud": deep},
        adapter_depths={"adapter.app.refund": 1, "adapter.app.fraud": None},
        heads={"nf_a": head(3), "nf_b": head(2), "nf_c": head(4)},
        function_adapters={
            "nf_a": "adapter.app.refund",
            "nf_b": "adapter.app.fraud",
            "nf_c": "adapter.app.refund",
        },
    )
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.int64)
    assert depth_key(None) == "full" and depth_key(6) == "depth-006"

    exported = export_routed_application_components(
        encoder,
        {ref: application.adapter_for(ref) for ref in application.adapter_refs},
        {ref: application.adapter_depth(ref) for ref in application.adapter_refs},
        dict(application.heads.items()),
        {name: application.adapter_ref_of(name) for name in application.function_ids},
        input_ids,
        attention_mask,
        encoder_paths={"full": tmp_path / "encoder.onnx", "depth-001": tmp_path / "encoder-1.onnx"},
        adapter_paths={
            "adapter.app.refund": tmp_path / "refund.onnx",
            "adapter.app.fraud": tmp_path / "fraud.onnx",
        },
        head_paths={
            "nf_a": tmp_path / "a.onnx",
            "nf_b": tmp_path / "b.onnx",
            "nf_c": tmp_path / "c.onnx",
        },
    )
    assert set(exported.encoders) == {"full", "depth-001"}
    assert set(exported.adapters) == {"adapter.app.refund", "adapter.app.fraud"}
    assert (
        set(exported.heads)
        == set(exported.chain_maximum_absolute_differences)
        == {"nf_a", "nf_b", "nf_c"}
    )
    assert (tmp_path / "encoder-1.onnx").stat().st_size < (tmp_path / "encoder.onnx").stat().st_size

    def session(name):
        return onnxruntime.InferenceSession(
            str(tmp_path / name), providers=["CPUExecutionProvider"]
        )

    feeds = {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()}
    prefix_embedding = session("encoder-1.onnx").run(None, feeds)[0]
    full_embedding = session("encoder.onnx").run(None, feeds)[0]
    assert not numpy.allclose(prefix_embedding, full_embedding)
    with torch.no_grad():
        numpy.testing.assert_allclose(
            prefix_embedding,
            encoder(input_ids, attention_mask, depth=1).numpy(),
            rtol=1e-4,
            atol=1e-5,
        )
    for name, head_file, adapter_file, embedding in (
        ("nf_a", "a.onnx", "refund.onnx", prefix_embedding),
        ("nf_b", "b.onnx", "fraud.onnx", full_embedding),
        ("nf_c", "c.onnx", "refund.onnx", prefix_embedding),
    ):
        adapted = session(adapter_file).run(None, {"sentence_embedding": embedding})[0]
        logits = session(head_file).run(None, {"function_embedding": adapted})[0]
        with torch.no_grad():
            expected = application.function_model(name)(input_ids, attention_mask).numpy()
        numpy.testing.assert_allclose(logits, expected, rtol=1e-4, atol=1e-5)

    with pytest.raises(ValueError, match="encoder_paths must name exactly the depths in use"):
        export_routed_application_components(
            encoder,
            {"adapter.app.refund": shallow},
            {"adapter.app.refund": 1},
            {"nf_a": application.heads["nf_a"]},
            {"nf_a": "adapter.app.refund"},
            input_ids,
            attention_mask,
            encoder_paths={"full": tmp_path / "x.onnx"},
            adapter_paths={"adapter.app.refund": tmp_path / "y.onnx"},
            head_paths={"nf_a": tmp_path / "z.onnx"},
        )
