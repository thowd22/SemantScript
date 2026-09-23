from __future__ import annotations

import math
import os
from typing import Any

import pytest

from semantscript_trainer.training import (
    DEFAULT_ENCODER_NAME,
    DEFAULT_ENCODER_REVISION,
    TrainingConfig,
    train_corpus,
)
from semantscript_trainer.training_contract import (
    TrainingCorpus,
    TrainingHeadContract,
    TrainingRow,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SEMANTSCRIPT_MODERNBERT_LIVE") != "1",
    reason="set SEMANTSCRIPT_MODERNBERT_LIVE=1 to run the pinned ModernBERT CUDA smoke test",
)


def test_pinned_modernbert_fine_tunes_and_reports_held_out_accuracy() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    assert torch.cuda.is_available(), "live ModernBERT smoke requires torch CUDA/ROCm access"

    contract = _ir()
    corpus = _corpus()
    result = train_corpus(
        contract,
        corpus,
        config=TrainingConfig(
            encoder_name=DEFAULT_ENCODER_NAME,
            encoder_revision=DEFAULT_ENCODER_REVISION,
            local_files_only=False,
            epochs=1,
            batch_size=1,
            maximum_sequence_length=32,
            evaluation_ratio=0.5,
            seed=7,
            device="cuda",
        ),
    )

    base_encoder = result.model.encoder.encoder
    head_parameter = next(result.model.head.parameters())
    encoder_parameter = next(base_encoder.parameters())
    assert base_encoder.__class__.__module__.startswith("transformers.models.modernbert")
    assert head_parameter.device.type == "cuda"
    assert encoder_parameter.device.type == "cuda"
    assert head_parameter.grad is not None
    assert encoder_parameter.grad is not None
    assert result.device.startswith("cuda")
    assert result.training_row_count == 1
    assert result.held_out_row_count == 1
    assert len(result.metrics) == 1
    assert math.isfinite(result.metrics[0].mean_training_loss)
    assert 0.0 <= result.held_out_accuracy <= 1.0


def _ir() -> dict[str, Any]:
    return {
        "kind": "semantscript.neural-function",
        "irVersion": 1,
        "stage": "lowered",
        "id": "nf_" + "1" * 64,
        "semanticSha256": "2" * 64,
        "source": {
            "path": "training-live.sem.ts",
            "line": 1,
            "column": 1,
            "sourceSha256": "3" * 64,
        },
        "definition": {
            "template": [{"kind": "text", "text": "Classify the score."}],
            "examples": [],
            "constraints": [],
        },
        "inputs": [
            {
                "name": "score",
                "index": 0,
                "tsType": "number",
                "type": {"kind": "number"},
            }
        ],
        "output": {
            "kind": "scalar",
            "tsType": "boolean",
            "head": {
                "kind": "nominal",
                "sourceKind": "boolean",
                "support": [False, True],
            },
        },
    }


def _corpus() -> TrainingCorpus:
    head = TrainingHeadContract(
        kind="nominal",
        source_kind="boolean",
        support=(False, True),
    )
    return TrainingCorpus(
        function_id="nf_" + "1" * 64,
        semantic_sha256="2" * 64,
        base_dataset_sha256="4" * 64,
        adversarial_dataset_sha256=None,
        head=head,
        rows=(
            TrainingRow(
                row_id="base:0",
                group_id="base:0",
                origin="gold",
                inputs={"score": 0},
                label_index=0,
            ),
            TrainingRow(
                row_id="base:1",
                group_id="base:1",
                # Gold/source examples are mandatory training rows. Use a
                # generated row so this smoke test exercises the held-out
                # calibration path instead of producing an empty split.
                origin="synthetic",
                inputs={"score": 1},
                label_index=1,
            ),
        ),
    )
