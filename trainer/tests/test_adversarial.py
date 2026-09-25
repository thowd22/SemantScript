from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import semantscript_trainer.adversarial as adversarial_module
from semantscript_trainer import (
    ADVERSARIAL_DATASET_KIND,
    AdversarialCacheError,
    AdversarialConfigurationError,
    AdversarialDatasetGenerator,
    AdversarialGenerationConfig,
    AdversarialGenerationError,
    BoundaryPairProposal,
    CounterfactualProposal,
    GeneratedCase,
    SyntheticDatasetGenerator,
    TeacherDescriptor,
    UnsynthesizableConstraintError,
)


def test_generates_tagged_boundary_cases_and_linked_counterfactual_pair(
    tmp_path: Path,
) -> None:
    contract = ir()
    base = base_dataset(contract, tmp_path / "base")
    teacher = FakeAdversarialTeacher()
    generator = AdversarialDatasetGenerator(teacher, tmp_path / "cache")

    dataset = generator.generate(contract, base)

    assert dataset.boundary_count == 2
    assert dataset.counterfactual_case_count == 2
    assert [case.tag for case in dataset.cases] == [
        "constraint-boundary",
        "constraint-boundary",
        "counterfactual",
        "counterfactual",
    ]
    assert [case.predicate_result for case in dataset.cases[:2]] == [False, True]
    assert len(dataset.pairs) == 1
    pair = dataset.pairs[0]
    assert pair.changed_path == "/score"
    assert pair.reason == "Crossing the minimum score changes the required label."
    assert dataset.cases[2].case_id == pair.anchor_case_id
    assert dataset.cases[3].case_id == pair.twin_case_id
    assert dataset.cases[2].pair_id == dataset.cases[3].pair_id == pair.pair_id
    assert teacher.boundary_calls == [0]
    assert teacher.counterfactual_calls == [
        GeneratedCase(inputs={"score": 0, "note": "base"}, output=False)
    ]


def test_valid_sidecar_cache_hit_avoids_all_teacher_calls(tmp_path: Path) -> None:
    contract = ir()
    base = base_dataset(contract, tmp_path / "base")
    first = AdversarialDatasetGenerator(FakeAdversarialTeacher(), tmp_path / "cache")
    expected = first.generate(contract, base)
    fresh_teacher = FakeAdversarialTeacher()

    actual = AdversarialDatasetGenerator(fresh_teacher, tmp_path / "cache").generate(
        deepcopy(contract),
        base,
    )

    assert actual == expected
    assert fresh_teacher.boundary_calls == []
    assert fresh_teacher.counterfactual_calls == []
    document = json.loads(first.cache_path(contract, base).read_text(encoding="utf-8"))
    assert document["kind"] == ADVERSARIAL_DATASET_KIND


def test_ratio_zero_disables_twins_but_keeps_constraint_boundaries(tmp_path: Path) -> None:
    contract = ir()
    base = base_dataset(contract, tmp_path / "base")
    teacher = FakeAdversarialTeacher()
    config = AdversarialGenerationConfig(counterfactual_ratio=0)

    dataset = AdversarialDatasetGenerator(
        teacher,
        tmp_path / "cache",
        config=config,
    ).generate(contract, base)

    assert dataset.boundary_count == 2
    assert dataset.counterfactual_case_count == 0
    assert dataset.pairs == ()
    assert teacher.boundary_calls == [0]
    assert teacher.counterfactual_calls == []


def test_ratio_uses_deterministic_nearest_count_and_changes_cache_key(tmp_path: Path) -> None:
    contract = ir(constraints=[])
    generated = tuple(
        GeneratedCase(inputs={"score": index, "note": f"case-{index}"}, output=False)
        for index in range(3)
    )
    base = base_dataset(contract, tmp_path / "base", cases=generated, total=3)
    half = AdversarialDatasetGenerator(
        RatioTeacher(),
        tmp_path / "cache",
        config=AdversarialGenerationConfig(counterfactual_ratio=0.5),
    )
    full = AdversarialDatasetGenerator(
        RatioTeacher(),
        tmp_path / "cache",
        config=AdversarialGenerationConfig(counterfactual_ratio=1),
    )

    half_dataset = half.generate(contract, base)

    assert len(half_dataset.pairs) == 2
    assert half.cache_path(contract, base) != full.cache_path(contract, base)


@pytest.mark.parametrize("ratio", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_ratio_rejects_invalid_values(ratio: Any) -> None:
    with pytest.raises(AdversarialConfigurationError, match="counterfactual_ratio"):
        AdversarialGenerationConfig(counterfactual_ratio=ratio)


def test_signed_zero_ratio_is_normalized_for_stable_cache_identity(tmp_path: Path) -> None:
    contract = ir(constraints=[])
    base = base_dataset(contract, tmp_path / "base")
    positive = AdversarialDatasetGenerator(
        FakeAdversarialTeacher(),
        tmp_path / "cache",
        config=AdversarialGenerationConfig(counterfactual_ratio=0.0),
    )
    negative = AdversarialDatasetGenerator(
        FakeAdversarialTeacher(),
        tmp_path / "cache",
        config=AdversarialGenerationConfig(counterfactual_ratio=-0.0),
    )

    assert negative.config.counterfactual_ratio == 0.0
    assert negative.config.document() == positive.config.document()
    assert negative.cache_path(contract, base) == positive.cache_path(contract, base)


def test_rejects_twins_that_relabel_an_existing_lifecycle_input(tmp_path: Path) -> None:
    contract = ir()
    base = base_dataset(
        contract,
        tmp_path / "base",
        cases=(
            GeneratedCase(inputs={"score": 0, "note": "x"}, output=False),
            GeneratedCase(inputs={"score": 0, "note": "y"}, output=False),
        ),
        total=2,
    )

    class CollidingTeacher(FakeAdversarialTeacher):
        def generate_counterfactual(
            self,
            ir: dict[str, Any],
            anchor: GeneratedCase,
            /,
        ) -> CounterfactualProposal:
            self.counterfactual_calls.append(anchor)
            if anchor.inputs["note"] == "x" and len(self.counterfactual_calls) == 1:
                # Same inputs as the other base case, but relabeled: must be rejected.
                return CounterfactualProposal(
                    twin=GeneratedCase(inputs={"score": 0, "note": "y"}, output=True),
                    reason="Changing the note flips the label.",
                )
            return CounterfactualProposal(
                twin=GeneratedCase(inputs={**anchor.inputs, "score": 10}, output=True),
                reason="Crossing the minimum score changes the required label.",
            )

    teacher = CollidingTeacher()
    dataset = AdversarialDatasetGenerator(teacher, tmp_path / "cache").generate(contract, base)

    assert len(teacher.counterfactual_calls) == 3
    labels: dict[str, set[Any]] = {}
    for case in list(base.cases) + list(dataset.cases):
        labels.setdefault(json.dumps(case.inputs, sort_keys=True), set()).add(case.output)
    assert all(len(values) == 1 for values in labels.values())

    class AlwaysColliding(FakeAdversarialTeacher):
        def generate_counterfactual(
            self,
            ir: dict[str, Any],
            anchor: GeneratedCase,
            /,
        ) -> CounterfactualProposal:
            self.counterfactual_calls.append(anchor)
            other = "y" if anchor.inputs["note"] == "x" else "x"
            return CounterfactualProposal(
                twin=GeneratedCase(inputs={"score": 0, "note": other}, output=True),
                reason="Changing the note flips the label.",
            )

    with pytest.raises(AdversarialGenerationError, match="different label"):
        AdversarialDatasetGenerator(
            AlwaysColliding(),
            tmp_path / "cache-exhausted",
            config=AdversarialGenerationConfig(maximum_attempts=2),
        ).generate(contract, base)


def test_rejects_boundary_claim_on_wrong_predicate_side(tmp_path: Path) -> None:
    contract = ir()
    base = base_dataset(contract, tmp_path / "base")
    teacher = FakeAdversarialTeacher(
        boundary=BoundaryPairProposal(
            predicate_false=GeneratedCase(inputs={"score": 9, "note": "edge"}, output=False),
            predicate_true=GeneratedCase(inputs={"score": 9, "note": "edge"}, output=False),
        )
    )

    with pytest.raises(UnsynthesizableConstraintError, match="two-sided"):
        AdversarialDatasetGenerator(
            teacher,
            tmp_path / "cache",
            config=AdversarialGenerationConfig(maximum_attempts=1),
        ).generate(contract, base)
    assert teacher.boundary_calls == [0]


def test_rejects_constraint_violating_boundary_label(tmp_path: Path) -> None:
    contract = ir()
    base = base_dataset(contract, tmp_path / "base")
    teacher = FakeAdversarialTeacher(
        boundary=BoundaryPairProposal(
            predicate_false=GeneratedCase(inputs={"score": 9, "note": "edge"}, output=False),
            predicate_true=GeneratedCase(inputs={"score": 10, "note": "edge"}, output=False),
        )
    )

    with pytest.raises(UnsynthesizableConstraintError, match="always constraint"):
        AdversarialDatasetGenerator(
            teacher,
            tmp_path / "cache",
            config=AdversarialGenerationConfig(maximum_attempts=1),
        ).generate(contract, base)


def test_rejects_counterfactual_with_more_than_one_changed_path(tmp_path: Path) -> None:
    contract = ir()
    base = base_dataset(contract, tmp_path / "base")
    teacher = FakeAdversarialTeacher(
        counterfactual=CounterfactualProposal(
            twin=GeneratedCase(inputs={"score": 10, "note": "also changed"}, output=True),
            reason="Two edits are not minimal.",
        )
    )

    with pytest.raises(AdversarialGenerationError, match="exactly one JSON path"):
        AdversarialDatasetGenerator(
            teacher,
            tmp_path / "cache",
            config=AdversarialGenerationConfig(maximum_attempts=1),
        ).generate(contract, base)


def test_rejects_counterfactual_numeric_edit_that_is_equal_in_binary64(
    tmp_path: Path,
) -> None:
    contract = ir(constraints=[])
    base = base_dataset(
        contract,
        tmp_path / "base",
        cases=(
            GeneratedCase(
                inputs={"score": 2**53, "note": "base"},
                output=False,
            ),
        ),
    )
    teacher = FakeAdversarialTeacher(
        counterfactual=CounterfactualProposal(
            twin=GeneratedCase(
                inputs={"score": 2**53 + 1, "note": "base"},
                output=True,
            ),
            reason="This Python integer edit vanishes after binary64 rounding.",
        )
    )

    with pytest.raises(AdversarialGenerationError, match="found 0"):
        AdversarialDatasetGenerator(
            teacher,
            tmp_path / "cache",
            config=AdversarialGenerationConfig(maximum_attempts=1),
        ).generate(contract, base)


def test_constant_predicate_fails_before_spending_teacher_work(tmp_path: Path) -> None:
    contract = ir(
        constraints=[
            {
                "kind": "always",
                "source": "false",
                "predicate": {"node": "literal", "value": False},
                "output": True,
            }
        ]
    )
    base = base_dataset(contract, tmp_path / "base")
    teacher = FakeAdversarialTeacher()

    with pytest.raises(UnsynthesizableConstraintError, match="constant false"):
        AdversarialDatasetGenerator(teacher, tmp_path / "cache").generate(contract, base)
    assert teacher.boundary_calls == []
    assert teacher.counterfactual_calls == []


def test_corrupt_sidecar_fails_closed_without_recalling_teacher(tmp_path: Path) -> None:
    contract = ir()
    base = base_dataset(contract, tmp_path / "base")
    first = AdversarialDatasetGenerator(FakeAdversarialTeacher(), tmp_path / "cache")
    first.generate(contract, base)
    path = first.cache_path(contract, base)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["cases"][0]["predicateResult"] = True
    path.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fresh_teacher = FakeAdversarialTeacher()

    with pytest.raises(AdversarialCacheError, match="digest"):
        AdversarialDatasetGenerator(fresh_teacher, tmp_path / "cache").generate(contract, base)
    assert fresh_teacher.boundary_calls == []
    assert fresh_teacher.counterfactual_calls == []


def test_semantic_config_type_tamper_fails_after_recomputed_digest(tmp_path: Path) -> None:
    contract = ir()
    base = base_dataset(contract, tmp_path / "base")
    first = AdversarialDatasetGenerator(FakeAdversarialTeacher(), tmp_path / "cache")
    first.generate(contract, base)
    path = first.cache_path(contract, base)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["config"]["counterfactualRatio"] = True
    document["payloadSha256"] = hashlib.sha256(
        adversarial_module._PAYLOAD_DIGEST_DOMAIN
        + adversarial_module._canonical_json_bytes(document["payload"])
    ).hexdigest()
    path.write_bytes(adversarial_module._canonical_document_bytes(document))

    with pytest.raises(AdversarialCacheError, match="config"):
        AdversarialDatasetGenerator(FakeAdversarialTeacher(), tmp_path / "cache").generate(
            contract,
            base,
        )


def test_public_dataset_rejects_orphan_counterfactual_cases(tmp_path: Path) -> None:
    contract = ir()
    base = base_dataset(contract, tmp_path / "base")
    dataset = AdversarialDatasetGenerator(FakeAdversarialTeacher(), tmp_path / "cache").generate(
        contract, base
    )

    with pytest.raises(AdversarialConfigurationError, match="exact bijection"):
        replace(dataset, pairs=())


def test_incremental_byte_limit_stops_sidecar_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = ir()
    base = base_dataset(contract, tmp_path / "base")
    monkeypatch.setattr(adversarial_module, "MAXIMUM_DATASET_CACHE_BYTES", 100)
    generator = AdversarialDatasetGenerator(FakeAdversarialTeacher(), tmp_path / "cache")

    with pytest.raises(AdversarialGenerationError, match="maximum byte length"):
        generator.generate(contract, base)
    assert not generator.cache_path(contract, base).exists()


def test_unpaired_unicode_in_request_is_a_typed_configuration_error(tmp_path: Path) -> None:
    contract = ir()
    base = base_dataset(contract, tmp_path / "base")
    contract["definition"]["template"][0]["text"] = "\ud800"

    with pytest.raises(AdversarialConfigurationError, match="canonicalized"):
        AdversarialDatasetGenerator(FakeAdversarialTeacher(), tmp_path / "cache").cache_path(
            contract, base
        )


def test_unsafe_sidecar_lock_is_translated_to_adversarial_cache_error(tmp_path: Path) -> None:
    contract = ir()
    base = base_dataset(contract, tmp_path / "base")
    generator = AdversarialDatasetGenerator(FakeAdversarialTeacher(), tmp_path / "cache")
    path = generator.cache_path(contract, base)
    path.parent.mkdir(parents=True)
    path.with_suffix(".lock").symlink_to(tmp_path / "elsewhere")

    with pytest.raises(AdversarialCacheError, match="unsafe"):
        generator.generate(contract, base)


class BaseTeacher:
    def __init__(self, cases: tuple[GeneratedCase, ...]) -> None:
        self.cases = cases
        self.descriptor = TeacherDescriptor("base", "base-v1", "1" * 64)

    def generate(self, ir: dict[str, Any], n: int, /) -> tuple[GeneratedCase, ...]:
        assert n == len(self.cases)
        return self.cases


class FakeAdversarialTeacher:
    def __init__(
        self,
        *,
        boundary: BoundaryPairProposal | None = None,
        counterfactual: CounterfactualProposal | None = None,
    ) -> None:
        self.descriptor = TeacherDescriptor("fake", "adversarial-v1", "2" * 64)
        self.boundary = boundary or BoundaryPairProposal(
            predicate_false=GeneratedCase(inputs={"score": 9, "note": "edge"}, output=False),
            predicate_true=GeneratedCase(inputs={"score": 10, "note": "edge"}, output=True),
        )
        self.counterfactual = counterfactual or CounterfactualProposal(
            twin=GeneratedCase(inputs={"score": 10, "note": "base"}, output=True),
            reason="Crossing the minimum score changes the required label.",
        )
        self.boundary_calls: list[int] = []
        self.counterfactual_calls: list[GeneratedCase] = []

    def generate_boundary_pair(
        self,
        ir: dict[str, Any],
        constraint_index: int,
        /,
    ) -> BoundaryPairProposal:
        self.boundary_calls.append(constraint_index)
        return self.boundary

    def generate_counterfactual(
        self,
        ir: dict[str, Any],
        anchor: GeneratedCase,
        /,
    ) -> CounterfactualProposal:
        self.counterfactual_calls.append(anchor)
        return self.counterfactual


class RatioTeacher(FakeAdversarialTeacher):
    def generate_counterfactual(
        self,
        ir: dict[str, Any],
        anchor: GeneratedCase,
        /,
    ) -> CounterfactualProposal:
        self.counterfactual_calls.append(anchor)
        return CounterfactualProposal(
            twin=GeneratedCase(
                inputs={**anchor.inputs, "note": anchor.inputs["note"] + "-changed"},
                output=not anchor.output,
            ),
            reason="Changing the note flips this synthetic label.",
        )


def base_dataset(
    contract: dict[str, Any],
    cache: Path,
    *,
    cases: tuple[GeneratedCase, ...] | None = None,
    total: int = 1,
):
    generated = cases or (GeneratedCase(inputs={"score": 0, "note": "base"}, output=False),)
    return SyntheticDatasetGenerator(BaseTeacher(generated), cache).generate(contract, total)


def ir(*, constraints: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    resolved_constraints = constraints
    if resolved_constraints is None:
        resolved_constraints = [
            {
                "kind": "always",
                "source": "score >= 10",
                "predicate": {
                    "node": "binary",
                    "operator": ">=",
                    "left": {"node": "input", "name": "score"},
                    "right": {"node": "literal", "value": 10},
                },
                "output": True,
            }
        ]
    return {
        "kind": "semantscript.neural-function",
        "irVersion": 1,
        "stage": "lowered",
        "id": "nf_" + "7" * 64,
        "semanticSha256": "8" * 64,
        "source": {
            "path": "adversarial.sem.ts",
            "line": 1,
            "column": 1,
            "sourceSha256": "9" * 64,
        },
        "definition": {
            "template": [{"kind": "text", "text": "Classify score."}],
            "examples": [],
            "constraints": resolved_constraints,
        },
        "inputs": [
            {"name": "score", "index": 0, "tsType": "number", "type": {"kind": "number"}},
            {"name": "note", "index": 1, "tsType": "string", "type": {"kind": "string"}},
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


class SkippingTeacher(RatioTeacher):
    """Cannot produce a single-field twin for one anchor; every other anchor works."""

    def generate_counterfactual(
        self,
        ir: dict[str, Any],
        anchor: GeneratedCase,
        /,
    ) -> CounterfactualProposal:
        if anchor.inputs["score"] == 1:
            self.counterfactual_calls.append(anchor)
            return CounterfactualProposal(
                twin=GeneratedCase(inputs={"score": 2, "note": "two-fields"}, output=True),
                reason="Two fields changed.",
            )
        return super().generate_counterfactual(ir, anchor)


def test_anchor_without_a_valid_twin_is_skipped_for_the_next_ranked_spare(tmp_path: Path) -> None:
    contract = ir()
    cases = tuple(
        GeneratedCase(inputs={"score": index, "note": f"case-{index}"}, output=False)
        for index in range(4)
    )
    base = base_dataset(contract, tmp_path / "base", cases=cases, total=4)
    teacher = SkippingTeacher()
    generator = AdversarialDatasetGenerator(
        teacher,
        tmp_path / "cache",
        config=AdversarialGenerationConfig(counterfactual_ratio=0.5, maximum_attempts=1),
    )

    dataset = generator.generate(contract, base)

    assert len(dataset.pairs) == 2
    sources = [pair.source_case_index for pair in dataset.pairs]
    assert len(set(sources)) == 2
    bad_index = next(index for index, case in enumerate(base.cases) if case.inputs["score"] == 1)
    assert bad_index not in sources
    # The cache entry with the substituted anchor reloads without the teacher.
    again = AdversarialDatasetGenerator(
        FakeAdversarialTeacher(),
        tmp_path / "cache",
        config=AdversarialGenerationConfig(counterfactual_ratio=0.5, maximum_attempts=1),
    ).generate(contract, base)
    assert again.dataset_sha256 == dataset.dataset_sha256


def test_skips_are_bounded_by_the_pairs_wanted(tmp_path: Path) -> None:
    contract = ir()
    cases = tuple(
        GeneratedCase(inputs={"score": 1, "note": f"case-{index}"}, output=False)
        for index in range(3)
    )
    base = base_dataset(contract, tmp_path / "base", cases=cases, total=3)
    generator = AdversarialDatasetGenerator(
        SkippingTeacher(),
        tmp_path / "cache",
        config=AdversarialGenerationConfig(counterfactual_ratio=1.0, maximum_attempts=1),
    )
    with pytest.raises(AdversarialGenerationError, match="exactly one JSON path"):
        generator.generate(contract, base)
