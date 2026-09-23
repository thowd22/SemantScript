from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from semantscript_model.classifier import SemanticClassifier
from semantscript_model.encoder import (
    DEFAULT_ENCODER_REVISION,
    EncoderConfig,
    SentenceEncoder,
)
from semantscript_model.heads import ClassificationHead, HeadConfig
from semantscript_model.losses import proper_scoring_loss

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


def test_head_config_maps_support_to_model_abi_width() -> None:
    binary = HeadConfig(input_size=8, kind="binary-sigmoid", cardinality=2)
    categorical = HeadConfig(input_size=8, kind="categorical-softmax", cardinality=5)

    assert binary.output_size == 1
    assert categorical.output_size == 5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"input_size": 8, "kind": "binary-sigmoid", "cardinality": 3},
            "cardinality 2",
        ),
        (
            {"input_size": 8, "kind": "categorical-softmax", "cardinality": 1},
            "at least 2",
        ),
        (
            {
                "input_size": 8,
                "kind": "categorical-softmax",
                "cardinality": 3,
                "architecture": "linear",
                "mlp_hidden_size": 4,
            },
            "cannot set mlp_hidden_size",
        ),
    ],
)
def test_head_config_rejects_invalid_abi_shapes(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HeadConfig(**kwargs)  # type: ignore[arg-type]


def test_encoder_config_uses_recorded_base_model() -> None:
    assert EncoderConfig().model_name == "answerdotai/ModernBERT-base"
    assert EncoderConfig().revision == DEFAULT_ENCODER_REVISION
    assert EncoderConfig().revision == "8949b909ec900327062f0ebf497f51aef5e6f0c8"


@pytest.mark.skipif(TORCH_AVAILABLE, reason="exercises the dependency-free environment")
def test_model_construction_reports_missing_pytorch() -> None:
    with pytest.raises(ImportError, match="optional PyTorch training dependency"):
        ClassificationHead(HeadConfig(input_size=4, kind="binary-sigmoid", cardinality=2))

    with pytest.raises(ImportError, match="optional PyTorch training dependency"):
        SentenceEncoder(encoder=object())


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch optional dependency is not installed")
def test_sentence_encoder_uses_masked_float32_mean_pooling() -> None:
    import torch

    class FakeTokenEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=3)

        def forward(self, *, input_ids, attention_mask, return_dict):
            del attention_mask
            assert return_dict is True
            values = input_ids.to(dtype=torch.float16).unsqueeze(-1)
            offsets = torch.tensor([0, 1, 2], dtype=torch.float16)
            return SimpleNamespace(last_hidden_state=values + offsets)

    encoder = SentenceEncoder(encoder=FakeTokenEncoder())
    input_ids = torch.tensor([[1, 3, 99], [2, 20, 30]], dtype=torch.int64)
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.int64)

    result = encoder(input_ids, attention_mask)

    assert result.dtype == torch.float32
    torch.testing.assert_close(
        result,
        torch.tensor([[2, 3, 4], [2, 3, 4]], dtype=torch.float32),
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch optional dependency is not installed")
def test_heads_emit_binary_and_categorical_abi_shapes() -> None:
    import torch

    embedding = torch.ones((2, 4), dtype=torch.float16)
    binary = ClassificationHead(HeadConfig(input_size=4, kind="binary-sigmoid", cardinality=2))
    categorical = ClassificationHead(
        HeadConfig(
            input_size=4,
            kind="categorical-softmax",
            cardinality=3,
            architecture="mlp",
            mlp_hidden_size=2,
        )
    )

    binary_logits = binary(embedding)
    categorical_logits = categorical(embedding)

    assert binary_logits.shape == (2, 1)
    assert categorical_logits.shape == (2, 3)
    assert binary_logits.dtype == torch.float32
    assert categorical_logits.dtype == torch.float32


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch optional dependency is not installed")
def test_half_precision_head_parameters_still_emit_float32_logits() -> None:
    import torch

    head = ClassificationHead(
        HeadConfig(input_size=4, kind="categorical-softmax", cardinality=3)
    ).half()

    logits = head(torch.ones((2, 4), dtype=torch.float16))

    assert logits.shape == (2, 3)
    assert logits.dtype == torch.float32


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch optional dependency is not installed")
def test_classifier_composes_injectable_encoder_and_head() -> None:
    import torch

    class FakeTokenEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=2)

        def forward(self, *, input_ids, attention_mask, return_dict):
            del attention_mask, return_dict
            hidden = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 1, 2)
            return SimpleNamespace(last_hidden_state=hidden)

    encoder = SentenceEncoder(encoder=FakeTokenEncoder())
    head = ClassificationHead(HeadConfig(input_size=2, kind="categorical-softmax", cardinality=4))
    classifier = SemanticClassifier(encoder, head)

    logits = classifier(
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.tensor([[1, 1]], dtype=torch.int64),
    )

    assert logits.shape == (1, 4)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch optional dependency is not installed")
def test_optimizer_step_updates_injected_encoder_and_head() -> None:
    import torch

    class TrainableTokenEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.embedding = torch.nn.Embedding(16, 4)

        def forward(self, *, input_ids, attention_mask, return_dict):
            del attention_mask, return_dict
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    torch.manual_seed(7)
    token_encoder = TrainableTokenEncoder()
    encoder = SentenceEncoder(encoder=token_encoder)
    head = ClassificationHead(HeadConfig(input_size=4, kind="categorical-softmax", cardinality=3))
    classifier = SemanticClassifier(encoder, head)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.1)
    encoder_before = token_encoder.embedding.weight.detach().clone()
    head_parameter = next(head.parameters())
    head_before = head_parameter.detach().clone()

    logits = classifier(
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        torch.ones((2, 2), dtype=torch.int64),
    )
    loss = proper_scoring_loss(logits, torch.tensor([0, 2], dtype=torch.long))
    loss.backward()
    optimizer.step()

    assert token_encoder.embedding.weight.grad is not None
    assert head_parameter.grad is not None
    assert not torch.equal(token_encoder.embedding.weight, encoder_before)
    assert not torch.equal(head_parameter, head_before)


@pytest.mark.skipif(
    not TORCH_AVAILABLE or TRANSFORMERS_AVAILABLE,
    reason="requires PyTorch without transformers",
)
def test_pretrained_encoder_reports_missing_transformers() -> None:
    with pytest.raises(ImportError, match="optional transformers dependency"):
        SentenceEncoder()
