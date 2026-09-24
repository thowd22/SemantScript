from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent))  # the offline fixtures live in test_training

from test_training import (  # noqa: E402
    TinyTokenEncoder,
    TinyTokenizer,
    categorical_corpus,
    categorical_ir,
)

from semantscript_trainer.application import (  # noqa: E402
    ApplicationTrainingResult,
    FunctionCorpus,
    add_function_head,
    train_application,
)
from semantscript_trainer.training import (  # noqa: E402
    TrainingConfig,
    TrainingConfigurationError,
    TrainingResult,
)
from semantscript_trainer.training_contract import (  # noqa: E402
    TrainingCorpus,
    TrainingHeadContract,
    TrainingRow,
)
from semantscript_trainer.verification import model_state_sha256  # noqa: E402

BINARY_ID = "nf_" + "5" * 64
BINARY_SEMANTIC = "6" * 64


def binary_ir() -> dict[str, Any]:
    ir = deepcopy(categorical_ir())
    ir["id"] = BINARY_ID
    ir["semanticSha256"] = BINARY_SEMANTIC
    ir["output"] = {
        "kind": "scalar",
        "tsType": '"no" | "yes"',
        "head": {"kind": "nominal", "sourceKind": "string-union", "support": ["no", "yes"]},
    }
    return ir


def binary_corpus() -> TrainingCorpus:
    rows = []
    for group_index in range(4):
        for value in ("left", "center", "right"):
            rows.append(
                TrainingRow(
                    row_id=f"binary:{group_index}:{value}",
                    group_id=f"group:{group_index}",
                    origin="synthetic",
                    inputs={"position": f"{value}-{group_index}"},
                    label_index=0 if value == "left" else 1,
                )
            )
    return TrainingCorpus(
        function_id=BINARY_ID,
        semantic_sha256=BINARY_SEMANTIC,
        base_dataset_sha256="7" * 64,
        adversarial_dataset_sha256=None,
        head=TrainingHeadContract(
            kind="nominal", source_kind="string-union", support=("no", "yes")
        ),
        rows=tuple(rows),
    )


def config(**overrides: Any) -> TrainingConfig:
    settings: dict[str, Any] = {
        "epochs": 30,
        "batch_size": 6,
        "learning_rate": 0.08,
        "weight_decay": 0,
        "maximum_sequence_length": 8,
        "evaluation_ratio": 0.25,
        "seed": 19,
        "device": "cpu",
    }
    settings.update(overrides)
    return TrainingConfig(**settings)


def test_trains_two_functions_jointly_over_one_shared_encoder() -> None:
    functions = [
        FunctionCorpus(categorical_ir(), categorical_corpus()),
        FunctionCorpus(binary_ir(), binary_corpus()),
    ]
    tokenizer = TinyTokenizer()
    encoder = TinyTokenEncoder()
    before = encoder.embedding.weight.detach().clone()

    result = train_application(
        functions, config=config(), tokenizer=tokenizer, encoder=encoder, adapter_bottleneck_size=4
    )

    assert isinstance(result, ApplicationTrainingResult)
    assert result.function_ids == (categorical_ir()["id"], BINARY_ID)
    assert result.model.function_ids == result.function_ids
    assert result.adapter_bottleneck_size == 4
    assert len(result.metrics) == 30
    assert set(result.metrics[-1].held_out_accuracy) == set(result.function_ids)
    assert not torch.equal(encoder.embedding.weight.detach(), before)
    for function_id, function in result.functions.items():
        assert isinstance(function, TrainingResult)
        assert function.function_id == function_id
        assert function.model.encoder is result.model.encoder
        assert function.model.adapter is result.model.adapter
        assert function.model.head is result.model.heads[function_id]
        assert function.held_out_accuracy == 1.0
        assert len(function.metrics) == 30
        assert function.held_out_row_count == 3
        assert function.config is result.config
    assert result.functions[BINARY_ID].head.logit_count == 2
    # The adapter was actually trained away from the identity.
    assert not torch.allclose(
        result.model.adapter.up.weight, torch.zeros_like(result.model.adapter.up.weight)
    )


def test_adding_a_head_leaves_shared_modules_and_existing_heads_byte_identical() -> None:
    first = train_application(
        [FunctionCorpus(categorical_ir(), categorical_corpus())],
        config=config(),
        tokenizer=TinyTokenizer(),
        encoder=TinyTokenEncoder(),
        adapter_bottleneck_size=4,
    )
    categorical_id = categorical_ir()["id"]
    before = {key: value.detach().clone() for key, value in first.model.state_dict().items()}
    categorical_digest = model_state_sha256(first.functions[categorical_id].model)
    flags = {name: parameter.requires_grad for name, parameter in first.model.named_parameters()}

    extended = add_function_head(
        first,
        FunctionCorpus(binary_ir(), binary_corpus()),
        config=config(epochs=40),
        tokenizer=TinyTokenizer(),
    )

    assert extended.model is first.model
    assert extended.function_ids == (categorical_id, BINARY_ID)
    assert extended.functions[categorical_id] is first.functions[categorical_id]
    assert model_state_sha256(extended.functions[categorical_id].model) == categorical_digest
    after = extended.model.state_dict()
    for key, value in before.items():
        assert torch.equal(after[key], value), f"{key} changed while adding a head"
    assert {key for key in after if key not in before} == {
        key for key in after if key.startswith(f"heads.{BINARY_ID}.")
    }
    assert extended.functions[BINARY_ID].held_out_accuracy == 1.0
    assert len(extended.functions[BINARY_ID].metrics) == 40
    assert {
        name: parameter.requires_grad for name, parameter in extended.model.named_parameters()
    } == {
        **flags,
        **{
            name: True
            for name, _ in extended.model.named_parameters()
            if name.startswith(f"heads.{BINARY_ID}.")
        },
    }

    with pytest.raises(TrainingConfigurationError, match="already trains"):
        add_function_head(
            extended, FunctionCorpus(binary_ir(), binary_corpus()), tokenizer=TinyTokenizer()
        )
    with pytest.raises(TrainingConfigurationError, match="canonical input version"):
        add_function_head(
            extended,
            FunctionCorpus(binary_ir(), binary_corpus()),
            config=config(canonical_input_version=1),
            tokenizer=TinyTokenizer(),
        )


def test_rejects_inconsistent_function_sets() -> None:
    corpus = categorical_corpus()
    with pytest.raises(TrainingConfigurationError, match="identity does not match"):
        FunctionCorpus(binary_ir(), corpus)
    with pytest.raises(TrainingConfigurationError, match="more than once"):
        train_application(
            [FunctionCorpus(categorical_ir(), corpus), FunctionCorpus(categorical_ir(), corpus)],
            config=config(),
            tokenizer=TinyTokenizer(),
            encoder=TinyTokenEncoder(),
        )
    with pytest.raises(TrainingConfigurationError, match="between 1 and"):
        train_application(
            [], config=config(), tokenizer=TinyTokenizer(), encoder=TinyTokenEncoder()
        )
    with pytest.raises(TrainingConfigurationError, match="FunctionCorpus"):
        train_application(
            [object()], config=config(), tokenizer=TinyTokenizer(), encoder=TinyTokenEncoder()
        )  # type: ignore[list-item]
