from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import semantscript_trainer.training as training_module
from semantscript_trainer.canonical_input import serialize_canonical_inputs_string
from semantscript_trainer.training import (
    DEFAULT_ENCODER_REVISION,
    TrainingConfig,
    TrainingConfigurationError,
    TrainingExecutionError,
    train_corpus,
)
from semantscript_trainer.training_contract import (
    TrainingCorpus,
    TrainingHeadContract,
    TrainingRow,
)

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - exercised in the minimal dependency job
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch training extra is unavailable")


class TinyTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, Any]:
        assert add_special_tokens is True
        assert padding is True
        assert truncation is True
        assert max_length >= 1
        assert return_tensors == "pt"
        self.texts.extend(texts)
        token_ids = []
        for text in texts:
            if '"left-' in text:
                token_ids.append(1)
            elif '"center-' in text:
                token_ids.append(2)
            elif '"right-' in text:
                token_ids.append(3)
            else:  # pragma: no cover - test fixture guard
                raise AssertionError(f"unexpected canonical text: {text}")
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long).unsqueeze(1),
            "attention_mask": torch.ones((len(texts), 1), dtype=torch.long),
        }

    def __bool__(self) -> bool:
        return False


if torch is not None:

    class TinyTokenEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.embedding = torch.nn.Embedding(4, 4)

        def forward(self, *, input_ids, attention_mask, return_dict):
            del attention_mask
            assert return_dict is True
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

else:
    TinyTokenEncoder = None


def test_training_config_is_reproducible_and_resource_bounded() -> None:
    config = TrainingConfig(loss="cross_entropy", head_architecture="mlp", mlp_hidden_size=8)

    assert len(DEFAULT_ENCODER_REVISION) == 40
    assert config.loss == "cross_entropy"
    assert config.head_architecture == "mlp"
    with pytest.raises(TrainingConfigurationError, match=r"batch_size \*"):
        TrainingConfig(batch_size=256, maximum_sequence_length=257)
    with pytest.raises(TrainingConfigurationError, match="immutable revision"):
        TrainingConfig(encoder_revision="")
    with pytest.raises(TrainingConfigurationError, match="linear heads"):
        TrainingConfig(mlp_hidden_size=4)
    with pytest.raises(TrainingConfigurationError, match="positive finite"):
        TrainingConfig(learning_rate=10**1000)


def test_import_and_missing_torch_failure_are_optional_dependency_safe() -> None:
    trainer_source = Path(__file__).parents[1] / "src"
    model_source = Path(__file__).parents[2] / "model" / "src"
    script = f"""
import builtins
import sys
sys.path[:0] = [{str(trainer_source)!r}, {str(model_source)!r}]
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in {{'torch', 'transformers'}}:
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from semantscript_trainer.training import TrainingExecutionError, _require_torch
try:
    _require_torch()
except TrainingExecutionError:
    pass
else:
    raise AssertionError('missing PyTorch did not fail lazily')
"""

    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr


@requires_torch
def test_fine_tunes_encoder_and_ir_derived_categorical_head_offline() -> None:
    contract = categorical_ir()
    corpus = categorical_corpus()
    tokenizer = TinyTokenizer()
    encoder = TinyTokenEncoder()
    original_encoder_weights = encoder.embedding.weight.detach().clone()

    result = train_corpus(
        contract,
        corpus,
        config=TrainingConfig(
            epochs=30,
            batch_size=6,
            learning_rate=0.08,
            weight_decay=0,
            maximum_sequence_length=8,
            evaluation_ratio=0.25,
            seed=19,
            device="cpu",
        ),
        tokenizer=tokenizer,
        encoder=encoder,
    )

    assert result.head.parameterization == "categorical-softmax"
    assert result.head.logit_count == 3
    assert result.model.head.config.output_size == 3
    assert result.training_row_count == 9
    assert result.held_out_row_count == 3
    assert result.held_out_accuracy == 1.0
    assert len(result.metrics) == 30
    assert all(0 <= epoch.held_out_accuracy <= 1 for epoch in result.metrics)
    assert not torch.equal(encoder.embedding.weight.detach(), original_encoder_weights)

    expected_texts = {
        serialize_canonical_inputs_string(contract["inputs"], {"position": value})
        for group_index in range(4)
        for value in (
            f"left-{group_index}",
            f"center-{group_index}",
            f"right-{group_index}",
        )
    }
    assert set(tokenizer.texts) == expected_texts


def test_training_requires_an_independent_held_out_group() -> None:
    contract = categorical_ir()
    one_group = TrainingCorpus(
        function_id=contract["id"],
        semantic_sha256=contract["semanticSha256"],
        base_dataset_sha256="3" * 64,
        adversarial_dataset_sha256=None,
        head=TrainingHeadContract(
            kind="nominal",
            source_kind="string-union",
            support=("low", "middle", "high"),
        ),
        rows=(
            TrainingRow(
                row_id="only",
                group_id="same",
                origin="gold",
                inputs={"position": "left"},
                label_index=0,
            ),
        ),
    )

    with pytest.raises(TrainingConfigurationError, match="held-out accuracy"):
        train_corpus(
            contract,
            one_group,
            config=TrainingConfig(device="cpu"),
        )


def test_rejects_falsey_config_canonical_mismatch_and_excess_total_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = categorical_ir()
    corpus = categorical_corpus()

    with pytest.raises(TrainingConfigurationError, match="config must"):
        train_corpus(contract, corpus, config=0)  # type: ignore[arg-type]

    bad_row = TrainingRow(
        row_id=corpus.rows[0].row_id,
        group_id=corpus.rows[0].group_id,
        origin=corpus.rows[0].origin,
        inputs={"position": 7},
        label_index=corpus.rows[0].label_index,
    )
    bad_corpus = TrainingCorpus(
        function_id=corpus.function_id,
        semantic_sha256=corpus.semantic_sha256,
        base_dataset_sha256=corpus.base_dataset_sha256,
        adversarial_dataset_sha256=corpus.adversarial_dataset_sha256,
        head=corpus.head,
        rows=(bad_row, *corpus.rows[1:]),
    )
    with pytest.raises(TrainingConfigurationError, match="canonically encoded"):
        train_corpus(contract, bad_corpus)

    with monkeypatch.context() as bounded:
        bounded.setattr(training_module, "_MAXIMUM_BATCH_CLASS_VALUES", 20)
        with pytest.raises(TrainingConfigurationError, match="head cardinality"):
            train_corpus(contract, corpus)

    monkeypatch.setattr(training_module, "_MAXIMUM_BATCH_PASSES", 3)
    with pytest.raises(TrainingConfigurationError, match="batch passes"):
        train_corpus(
            contract,
            corpus,
            config=TrainingConfig(epochs=2, batch_size=9, evaluation_ratio=0.25),
        )


def test_rejects_exact_canonical_input_leakage_across_held_out_split() -> None:
    contract = categorical_ir()
    head = TrainingHeadContract(
        kind="nominal",
        source_kind="string-union",
        support=("low", "middle", "high"),
    )
    rows = tuple(
        TrainingRow(
            row_id=f"duplicate:{index}",
            group_id=f"group:{index}",
            origin="synthetic",
            inputs={"position": "left-duplicate"},
            label_index=0,
        )
        for index in range(2)
    )
    corpus = TrainingCorpus(
        function_id=contract["id"],
        semantic_sha256=contract["semanticSha256"],
        base_dataset_sha256="3" * 64,
        adversarial_dataset_sha256=None,
        head=head,
        rows=rows,
    )

    with pytest.raises(TrainingConfigurationError, match="both training and held-out"):
        train_corpus(contract, corpus)


@requires_torch
def test_rejects_oversized_head_before_allocating_projection() -> None:
    class HugeWidthEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=6_000_000)

        def forward(self, **kwargs):  # pragma: no cover - construction must reject first
            raise AssertionError(kwargs)

    with pytest.raises(TrainingConfigurationError, match="head exceeds"):
        train_corpus(
            categorical_ir(),
            categorical_corpus(),
            config=TrainingConfig(device="cpu"),
            tokenizer=TinyTokenizer(),
            encoder=HugeWidthEncoder(),
        )


@requires_torch
def test_logit_validation_rejects_wrong_shape_and_nonfinite_values() -> None:
    with pytest.raises(TrainingExecutionError, match="shape"):
        training_module._validate_logits(torch.zeros((2, 2)), torch, batch_size=2, logit_count=3)
    with pytest.raises(TrainingExecutionError, match="non-finite"):
        training_module._validate_logits(
            torch.tensor([[0.0, float("nan"), 1.0]]),
            torch,
            batch_size=1,
            logit_count=3,
        )


@requires_torch
@pytest.mark.parametrize(
    ("input_dtype", "mask", "message"),
    [
        (torch.float32, torch.tensor([[1]], dtype=torch.long), "must use int64"),
        (torch.long, torch.tensor([[2]], dtype=torch.long), "only zero and one"),
        (torch.long, torch.tensor([[0]], dtype=torch.long), "at least one attended token"),
    ],
)
def test_tokenizer_tensors_enforce_the_int64_binary_mask_abi(
    input_dtype: Any,
    mask: Any,
    message: str,
) -> None:
    class InvalidTokenizer:
        def __call__(self, texts, **kwargs):
            del texts, kwargs
            return {
                "input_ids": torch.tensor([[1]], dtype=input_dtype),
                "attention_mask": mask,
            }

    row = TrainingRow(
        row_id="tokenizer-abi",
        group_id="tokenizer-abi",
        origin="gold",
        inputs={"position": "left-0"},
        label_index=0,
    )
    prepared = training_module._PreparedRow(row=row, text="canonical")

    with pytest.raises(TrainingExecutionError, match=message):
        training_module._tensorize_batch(
            [prepared],
            InvalidTokenizer(),
            torch,
            torch.device("cpu"),
            8,
        )


def categorical_ir() -> dict[str, Any]:
    return {
        "kind": "semantscript.neural-function",
        "irVersion": 1,
        "stage": "lowered",
        "id": "nf_" + "1" * 64,
        "semanticSha256": "2" * 64,
        "source": {
            "path": "training.sem.ts",
            "line": 1,
            "column": 1,
            "sourceSha256": "4" * 64,
        },
        "definition": {
            "template": [{"kind": "text", "text": "Classify position."}],
            "examples": [],
            "constraints": [],
        },
        "inputs": [
            {
                "name": "position",
                "index": 0,
                "tsType": "string",
                "type": {"kind": "string"},
            }
        ],
        "output": {
            "kind": "scalar",
            "tsType": '"low" | "middle" | "high"',
            "head": {
                "kind": "nominal",
                "sourceKind": "string-union",
                "support": ["low", "middle", "high"],
            },
        },
    }


def categorical_corpus() -> TrainingCorpus:
    contract = categorical_ir()
    head = TrainingHeadContract(
        kind="nominal",
        source_kind="string-union",
        support=("low", "middle", "high"),
    )
    rows: list[TrainingRow] = []
    for group_index in range(4):
        for label_index, value in enumerate(("left", "center", "right")):
            rows.append(
                TrainingRow(
                    row_id=f"row:{group_index}:{label_index}",
                    group_id=f"group:{group_index}",
                    origin="gold" if group_index == 0 else "synthetic",
                    inputs={"position": f"{value}-{group_index}"},
                    label_index=label_index,
                )
            )
    return TrainingCorpus(
        function_id=contract["id"],
        semantic_sha256=contract["semanticSha256"],
        base_dataset_sha256="3" * 64,
        adversarial_dataset_sha256=None,
        head=head,
        rows=tuple(rows),
    )
