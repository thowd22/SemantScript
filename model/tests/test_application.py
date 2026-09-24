from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from semantscript_model import (  # noqa: E402
    AdapterConfig,
    ApplicationAdapter,
    ClassificationHead,
    FunctionModel,
    HeadConfig,
    SentenceEncoder,
    SharedEncoderApplication,
)


class TokenEncoder(torch.nn.Module):
    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embedding = torch.nn.Embedding(16, hidden)

    def forward(self, *, input_ids, attention_mask, return_dict):
        del attention_mask
        assert return_dict is True
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def head(cardinality: int = 3, hidden: int = 8) -> ClassificationHead:
    return ClassificationHead(
        HeadConfig(input_size=hidden, kind="categorical-softmax", cardinality=cardinality)
    )


def test_adapter_starts_as_the_identity_and_preserves_shape() -> None:
    adapter = ApplicationAdapter(AdapterConfig(hidden_size=8, bottleneck_size=4))
    embedding = torch.randn(3, 8)
    torch.testing.assert_close(adapter(embedding), embedding)
    assert AdapterConfig(hidden_size=8, bottleneck_size=4).parameter_count == 2 * 8 * 4 + 4 + 8
    with torch.no_grad():
        adapter.up.weight.fill_(0.5)
    adapted = adapter(embedding)
    assert adapted.shape == (3, 8) and adapted.dtype == torch.float32
    assert not torch.allclose(adapted, embedding)
    with pytest.raises(ValueError, match="hidden width"):
        adapter(torch.randn(3, 7))
    with pytest.raises(ValueError, match="positive integer"):
        AdapterConfig(hidden_size=0)
    with pytest.raises(TypeError, match="AdapterConfig"):
        ApplicationAdapter({"hidden_size": 8})  # type: ignore[arg-type]


def test_application_shares_encoder_and_adapter_across_function_views() -> None:
    encoder = SentenceEncoder(encoder=TokenEncoder())
    adapter = ApplicationAdapter(AdapterConfig(hidden_size=8, bottleneck_size=4))
    application = SharedEncoderApplication(encoder, adapter, {"nf_a": head(3)})
    application.add_head("nf_b", head(2))
    assert application.function_ids == ("nf_a", "nf_b")
    assert application.hidden_size == 8

    view_a = application.function_model("nf_a")
    view_b = application.function_model("nf_b")
    assert isinstance(view_a, FunctionModel)
    assert view_a.encoder is application.encoder and view_b.encoder is application.encoder
    assert view_a.adapter is application.adapter
    assert view_a.head is application.heads["nf_a"] and view_b.head is application.heads["nf_b"]
    application_state = application.state_dict()
    for key, value in view_a.state_dict().items():
        shared_key = (
            key.replace("head.", "heads.nf_a.", 1)
            if key.startswith("head.")
            else key.replace("adapter.", "adapters.adapter__application.", 1)
            if key.startswith("adapter.")
            else key
        )
        assert torch.equal(application_state[shared_key], value), key
    assert not any(key.startswith("heads.nf_b.") for key in view_a.state_dict())

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    outputs = application(input_ids, attention_mask)
    assert set(outputs) == {"nf_a", "nf_b"}
    assert outputs["nf_a"].shape == (2, 3) and outputs["nf_b"].shape == (2, 2)
    torch.testing.assert_close(view_a(input_ids, attention_mask), outputs["nf_a"])
    torch.testing.assert_close(
        application.heads["nf_b"](application.embed(input_ids, attention_mask)),
        outputs["nf_b"],
    )

    with pytest.raises(ValueError, match="already has a head"):
        application.add_head("nf_a", head(3))
    with pytest.raises(ValueError, match="without dots"):
        application.add_head("nf.a", head(3))
    with pytest.raises(ValueError, match="input_size must match"):
        application.add_head("nf_c", head(3, hidden=4))
    with pytest.raises(KeyError):
        application.function_model("nf_missing")
    with pytest.raises(ValueError, match="hidden_size must match"):
        SharedEncoderApplication(encoder, ApplicationAdapter(AdapterConfig(hidden_size=4)))
    with pytest.raises(TypeError, match="SentenceEncoder"):
        SharedEncoderApplication(TokenEncoder(), adapter)  # type: ignore[arg-type]


def test_freezing_shared_modules_leaves_them_untouched_by_head_training() -> None:
    encoder = SentenceEncoder(encoder=TokenEncoder())
    adapter = ApplicationAdapter(AdapterConfig(hidden_size=8, bottleneck_size=4))
    application = SharedEncoderApplication(encoder, adapter, {"nf_a": head(3)})
    application.add_head("nf_b", head(2))
    for parameter in (*application.encoder.parameters(), *application.adapter.parameters()):
        parameter.requires_grad_(False)
    for parameter in application.heads["nf_a"].parameters():
        parameter.requires_grad_(False)
    before = {key: value.clone() for key, value in application.state_dict().items()}
    view = application.function_model("nf_b")
    optimizer = torch.optim.SGD([p for p in view.parameters() if p.requires_grad], lr=0.5)
    logits = view(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]))
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    optimizer.step()
    after = application.state_dict()
    for key, value in before.items():
        if key.startswith("heads.nf_b."):
            assert not torch.equal(after[key], value), key
        else:
            assert torch.equal(after[key], value), key
