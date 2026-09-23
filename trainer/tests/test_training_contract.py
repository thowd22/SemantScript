from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from semantscript_trainer.adversarial import (
    AdversarialCase,
    AdversarialDataset,
    AdversarialGenerationConfig,
    CounterfactualPair,
)
from semantscript_trainer.dataset import DatasetCase, TrainingDataset
from semantscript_trainer.teacher import TeacherDescriptor
from semantscript_trainer.training_contract import (
    HeldOutSplitConfig,
    TrainingContractError,
    assemble_training_corpus,
    derive_training_head,
    split_training_corpus,
)


def test_derives_boolean_binary_abi_and_stable_label_indices() -> None:
    head = derive_training_head(ir(boolean_head()))

    assert head.support == (False, True)
    assert head.parameterization == "binary-sigmoid"
    assert head.logit_count == 1
    assert head.ordinal is False
    assert head.label_index(False) == 0
    assert head.label_index(True) == 1
    with pytest.raises(TrainingContractError, match="outside"):
        head.label_index(0)


def test_numeric_mapping_uses_binary64_and_preserves_signed_zero() -> None:
    contract = ir(
        {
            "kind": "nominal",
            "sourceKind": "number-enum",
            "support": [-0.0, 0.0, 2**53],
        }
    )
    head = derive_training_head(contract)

    assert head.label_index(-0.0) == 0
    assert head.label_index(0.0) == 1
    assert head.label_index(2**53 + 1) == 2
    with pytest.raises(TrainingContractError, match="outside"):
        head.label_index(True)


def test_derives_ordered_decimal_support_and_categorical_width() -> None:
    head = derive_training_head(ir(bounded_head()))

    assert head.support == (-0.5, 0.0, 0.5)
    assert head.ordinal is True
    assert head.parameterization == "categorical-softmax"
    assert head.logit_count == 3
    assert head.label_index(0.5) == 2

    bounded_int = bounded_head()
    bounded_int.update(
        sourceKind="bounded-int",
        minimum="-1",
        maximum="1",
        step="1",
        supportDecimal=["-1", "0", "1"],
    )
    integer_head = derive_training_head(ir(bounded_int))
    assert integer_head.support == (-1, 0, 1)
    assert all(type(value) is int for value in integer_head.support)


def test_rejects_object_duplicate_and_malformed_supports() -> None:
    structured = ir(boolean_head())
    structured["output"] = {
        "kind": "object",
        "tsType": "Result",
        "fields": [{"name": "ok", "head": boolean_head()}],
    }
    duplicate = ir(
        {
            "kind": "nominal",
            "sourceKind": "number-enum",
            "support": [2**53, 2**53 + 1],
        }
    )
    malformed = ir(bounded_head())
    malformed["output"]["head"]["supportDecimal"] = ["0", "1e0"]
    false_boolean = ir({"kind": "nominal", "sourceKind": "boolean", "support": [0, 1]})

    with pytest.raises(TrainingContractError, match="scalar outputs"):
        derive_training_head(structured)
    with pytest.raises(TrainingContractError, match="duplicate semantic"):
        derive_training_head(duplicate)
    with pytest.raises(TrainingContractError, match="canonical decimal"):
        derive_training_head(malformed)
    with pytest.raises(TrainingContractError, match="boolean head"):
        derive_training_head(false_boolean)


def test_assembles_identity_bound_rows_and_groups_counterfactual_source() -> None:
    contract = ir(boolean_head())
    base = base_dataset()
    sidecar = counterfactual_dataset(base)

    corpus = assemble_training_corpus(contract, base, sidecar)

    assert len(corpus.rows) == 5
    assert [row.label_index for row in corpus.rows] == [0, 0, 1, 0, 1]
    pair_group = sidecar.pairs[0].pair_id
    assert [row.group_id for row in corpus.rows] == [
        "base:0",
        pair_group,
        "base:2",
        pair_group,
        pair_group,
    ]
    exposed = corpus.rows[0].inputs
    exposed["score"] = 999
    assert corpus.rows[0].inputs == {"score": 0}


def test_rejects_dataset_identity_label_and_pair_source_mismatches() -> None:
    contract = ir(boolean_head())
    base = base_dataset()
    sidecar = counterfactual_dataset(base)

    with pytest.raises(TrainingContractError, match="identity"):
        assemble_training_corpus(contract, replace(base, semantic_sha256="f" * 64))
    with pytest.raises(TrainingContractError, match="outside"):
        assemble_training_corpus(
            contract,
            replace(
                base,
                cases=(DatasetCase({"score": 0}, "x", "gold"),),
                requested_case_count=1,
            ),
        )
    bad_pair = replace(sidecar.pairs[0], source_case_index=0)
    with pytest.raises(TrainingContractError, match="source must be synthetic"):
        assemble_training_corpus(contract, base, replace(sidecar, pairs=(bad_pair,)))


def test_constraint_boundary_sides_share_one_split_group() -> None:
    contract = ir(boolean_head(), constraints=[constraint()])
    base = base_dataset()
    sidecar = boundary_dataset(base)

    corpus = assemble_training_corpus(contract, base, sidecar)
    boundary_rows = [row for row in corpus.rows if row.origin == "constraint-boundary"]

    assert len(boundary_rows) == 2
    assert {row.group_id for row in boundary_rows} == {"boundary:0"}


def test_split_is_deterministic_group_aware_and_nonempty() -> None:
    base = base_dataset()
    corpus = assemble_training_corpus(ir(boolean_head()), base, counterfactual_dataset(base))
    config = HeldOutSplitConfig(evaluation_ratio=0.4, seed=42)

    first = split_training_corpus(corpus, config)
    second = split_training_corpus(corpus, config)

    assert first == second
    assert first.training
    assert first.evaluation
    assert {row.group_id for row in first.training}.isdisjoint(
        row.group_id for row in first.evaluation
    )
    pair_id = corpus.rows[1].group_id
    pair_partitions = [
        any(row.group_id == pair_id for row in partition)
        for partition in (first.training, first.evaluation)
    ]
    assert pair_partitions in ([True, False], [False, True])


def test_split_keeps_every_gold_example_in_training_for_post_training_verification() -> None:
    corpus = assemble_training_corpus(ir(boolean_head()), base_dataset())

    for seed in range(20):
        split = split_training_corpus(
            corpus,
            HeldOutSplitConfig(evaluation_ratio=0.8, seed=seed),
        )
        assert all(row.origin != "gold" for row in split.evaluation)
        assert [row.row_id for row in split.training if row.origin == "gold"] == ["base:0"]


def test_single_group_stays_in_training_and_split_config_is_bounded() -> None:
    one = replace(
        base_dataset(), cases=(DatasetCase({"score": 0}, False, "gold"),), requested_case_count=1
    )
    corpus = assemble_training_corpus(ir(boolean_head()), one)
    split = split_training_corpus(corpus)

    assert len(split.training) == 1
    assert split.evaluation == ()
    for ratio in (0, 1, float("nan"), True):
        with pytest.raises(TrainingContractError, match="evaluation_ratio"):
            HeldOutSplitConfig(evaluation_ratio=ratio)


def boolean_head() -> dict[str, Any]:
    return {
        "kind": "nominal",
        "sourceKind": "boolean",
        "support": [False, True],
    }


def bounded_head() -> dict[str, Any]:
    return {
        "kind": "ordinal",
        "sourceKind": "bounded-number",
        "minimum": "-0.5",
        "maximum": "0.5",
        "step": "0.5",
        "supportDecimal": ["-0.5", "0", "0.5"],
        "expectedValue": "numeric",
    }


def constraint() -> dict[str, Any]:
    return {
        "kind": "always",
        "source": "score >= 2",
        "predicate": {
            "node": "binary",
            "operator": ">=",
            "left": {"node": "input", "name": "score"},
            "right": {"node": "literal", "value": 2},
        },
        "output": True,
    }


def ir(head: dict[str, Any], *, constraints: list[dict[str, Any]] | None = None) -> dict[str, Any]:
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
            "sourceSha256": "3" * 64,
        },
        "definition": {
            "template": [{"kind": "text", "text": "Classify."}],
            "examples": [],
            "constraints": constraints or [],
        },
        "inputs": [{"name": "score", "index": 0, "tsType": "number", "type": {"kind": "number"}}],
        "output": {"kind": "scalar", "tsType": "boolean", "head": head},
    }


def base_dataset() -> TrainingDataset:
    return TrainingDataset(
        function_id="nf_" + "1" * 64,
        semantic_sha256="2" * 64,
        requested_case_count=3,
        teacher=TeacherDescriptor("fake", "base", "4" * 64),
        cases=(
            DatasetCase({"score": 0}, False, "gold"),
            DatasetCase({"score": 1}, False, "synthetic"),
            DatasetCase({"score": 2}, True, "synthetic"),
        ),
        cache_key_sha256="5" * 64,
        payload_sha256="6" * 64,
        dataset_sha256="7" * 64,
    )


def counterfactual_dataset(base: TrainingDataset) -> AdversarialDataset:
    pair_id = "cf_" + "8" * 64
    anchor_id = "ac_" + "9" * 64
    twin_id = "ac_" + "a" * 64
    return AdversarialDataset(
        function_id=base.function_id,
        base_dataset_sha256=base.dataset_sha256,
        teacher=TeacherDescriptor("fake", "adversarial", "b" * 64),
        config=AdversarialGenerationConfig(),
        cases=(
            AdversarialCase(
                case_id=anchor_id,
                inputs={"score": 1},
                output=False,
                tag="counterfactual",
                pair_id=pair_id,
                pair_role="anchor",
            ),
            AdversarialCase(
                case_id=twin_id,
                inputs={"score": 2},
                output=True,
                tag="counterfactual",
                pair_id=pair_id,
                pair_role="twin",
            ),
        ),
        pairs=(
            CounterfactualPair(
                pair_id=pair_id,
                anchor_case_id=anchor_id,
                twin_case_id=twin_id,
                source_case_index=1,
                changed_path="/score",
                reason="Crossing the boundary changes the label.",
            ),
        ),
        cache_key_sha256="c" * 64,
        payload_sha256="d" * 64,
        dataset_sha256="e" * 64,
    )


def boundary_dataset(base: TrainingDataset) -> AdversarialDataset:
    return AdversarialDataset(
        function_id=base.function_id,
        base_dataset_sha256=base.dataset_sha256,
        teacher=TeacherDescriptor("fake", "adversarial", "b" * 64),
        config=AdversarialGenerationConfig(counterfactual_ratio=0),
        cases=(
            AdversarialCase(
                case_id="ac_" + "8" * 64,
                inputs={"score": 0},
                output=False,
                tag="constraint-boundary",
                constraint_index=0,
                predicate_result=False,
            ),
            AdversarialCase(
                case_id="ac_" + "9" * 64,
                inputs={"score": 2},
                output=True,
                tag="constraint-boundary",
                constraint_index=0,
                predicate_result=True,
            ),
        ),
        pairs=(),
        cache_key_sha256="c" * 64,
        payload_sha256="d" * 64,
        dataset_sha256="e" * 64,
    )
