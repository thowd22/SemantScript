from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

import semantscript_trainer.dataset as dataset_module
from semantscript_trainer import (
    DatasetCacheError,
    DatasetCase,
    DatasetConfigurationError,
    GeneratedCase,
    SyntheticDatasetGenerator,
    TeacherDescriptor,
    TeacherResponseError,
    TeacherTransportError,
    TrainingDataset,
    loads_strict_json,
)


class FakeTeacher:
    def __init__(
        self,
        cases: tuple[GeneratedCase, ...] = (),
        *,
        model: str = "fake-v1",
        failure: Exception | None = None,
    ) -> None:
        self._cases = cases
        self._failure = failure
        self.calls: list[tuple[dict[str, Any], int]] = []
        self._descriptor = TeacherDescriptor(
            provider="fake",
            model=model,
            configuration_sha256="a" * 64,
        )

    @property
    def descriptor(self) -> TeacherDescriptor:
        return self._descriptor

    def generate(self, ir: dict[str, Any], n: int, /) -> tuple[GeneratedCase, ...]:
        self.calls.append((ir, n))
        if self._failure is not None:
            raise self._failure
        return self._cases


def test_exact_total_keeps_gold_first_and_requests_only_remainder(tmp_path: Path) -> None:
    contract = ir(examples=[example("known", True)])
    generated = (
        GeneratedCase(inputs={"message": "first"}, output=False),
        GeneratedCase(inputs={"message": "second"}, output=True),
    )
    teacher = FakeTeacher(generated)

    dataset = SyntheticDatasetGenerator(teacher, tmp_path).generate(contract, 3)

    assert isinstance(dataset, TrainingDataset)
    assert dataset.requested_case_count == 3
    assert dataset.gold_count == 1
    assert dataset.synthetic_count == 2
    assert dataset.teacher == teacher.descriptor
    assert dataset.cases == (
        DatasetCase(inputs={"message": "known"}, output=True, origin="gold"),
        DatasetCase(inputs={"message": "first"}, output=False, origin="synthetic"),
        DatasetCase(inputs={"message": "second"}, output=True, origin="synthetic"),
    )
    assert teacher.calls == [(contract, 2)]


def test_returned_case_values_are_defensive_snapshots(tmp_path: Path) -> None:
    contract = ir(examples=[example("known", True)])
    dataset = SyntheticDatasetGenerator(FakeTeacher(), tmp_path).generate(contract, 1)

    exposed = dataset.cases[0].inputs
    exposed["message"] = "tampered"

    assert dataset.cases[0].inputs == {"message": "known"}
    assert (
        dataset.payload_sha256
        == SyntheticDatasetGenerator(FakeTeacher(), tmp_path).generate(contract, 1).payload_sha256
    )


def test_gold_can_satisfy_total_without_calling_teacher(tmp_path: Path) -> None:
    contract = ir(examples=[example("one", True), example("two", False)])
    teacher = FakeTeacher()

    dataset = SyntheticDatasetGenerator(teacher, tmp_path).generate(contract, 2)

    assert [case.origin for case in dataset.cases] == ["gold", "gold"]
    assert [case.inputs for case in dataset.cases] == [
        {"message": "one"},
        {"message": "two"},
    ]
    assert teacher.calls == []


def test_empty_dataset_is_cached_without_calling_teacher(tmp_path: Path) -> None:
    contract = ir()
    first_teacher = FakeTeacher()
    first = SyntheticDatasetGenerator(first_teacher, tmp_path)

    expected = first.generate(contract, 0)
    fresh_teacher = FakeTeacher()
    actual = SyntheticDatasetGenerator(fresh_teacher, tmp_path).generate(contract, 0)

    assert expected == actual
    assert expected.cases == ()
    assert first_teacher.calls == []
    assert fresh_teacher.calls == []


def test_empty_dataset_still_validates_the_ir_contract(tmp_path: Path) -> None:
    contract = ir()
    contract["inputs"] = "not-an-array"
    teacher = FakeTeacher()

    with pytest.raises(DatasetConfigurationError, match="case contract"):
        SyntheticDatasetGenerator(teacher, tmp_path).generate(contract, 0)
    assert teacher.calls == []


def test_strict_ir_loading_preserves_signed_zero_and_unicode_gold(tmp_path: Path) -> None:
    contract = ir(
        examples=[{"inputs": {"zero": -0.0, "label": "café 🚀"}, "output": True}],
    )
    contract["inputs"] = [
        {
            "name": "zero",
            "index": 0,
            "tsType": "-0",
            "type": {"kind": "literal", "value": -0.0},
        },
        {
            "name": "label",
            "index": 1,
            "tsType": "string",
            "type": {"kind": "string"},
        },
    ]
    contract["definition"]["template"][0]["text"] = "Classify café 🚀."
    loaded = loads_strict_json(
        json.dumps(contract, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    )
    assert isinstance(loaded, dict)

    dataset = SyntheticDatasetGenerator(FakeTeacher(), tmp_path).generate(loaded, 1)
    signed_zero = dataset.cases[0].inputs["zero"]

    assert isinstance(signed_zero, float)
    assert math.copysign(1.0, signed_zero) == -1.0
    assert "café 🚀" in SyntheticDatasetGenerator(FakeTeacher(), tmp_path).cache_path(
        loaded, 1
    ).read_text(encoding="utf-8")


def test_valid_cache_hit_with_fresh_teacher_avoids_provider_call(tmp_path: Path) -> None:
    contract = ir()
    case = GeneratedCase(inputs={"message": "generated"}, output=True)
    first_teacher = FakeTeacher((case,))
    first_generator = SyntheticDatasetGenerator(first_teacher, tmp_path)
    expected = first_generator.generate(contract, 1)
    cache_path = first_generator.cache_path(contract, 1)
    first_bytes = cache_path.read_bytes()

    fresh_teacher = FakeTeacher((case,))
    actual = SyntheticDatasetGenerator(fresh_teacher, tmp_path).generate(contract, 1)

    assert actual == expected
    assert fresh_teacher.calls == []
    assert cache_path.read_bytes() == first_bytes


def test_concurrent_same_request_calls_teacher_once(tmp_path: Path) -> None:
    contract = ir()
    teacher = FakeTeacher((GeneratedCase(inputs={"message": "generated"}, output=True),))
    generator = SyntheticDatasetGenerator(teacher, tmp_path)
    ready = Barrier(2)

    def generate() -> TrainingDataset:
        ready.wait()
        return generator.generate(contract, 1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(generate)
        second = executor.submit(generate)

    assert first.result() == second.result()
    assert teacher.calls == [(contract, 1)]


def test_teacher_cannot_mutate_the_snapshotted_ir(tmp_path: Path) -> None:
    contract = ir()

    class MutatingTeacher(FakeTeacher):
        def generate(self, value: dict[str, Any], n: int, /) -> tuple[GeneratedCase, ...]:
            value["definition"]["template"][0]["text"] = "mutated"
            return super().generate(value, n)

    teacher = MutatingTeacher((GeneratedCase(inputs={"message": "generated"}, output=True),))
    generator = SyntheticDatasetGenerator(teacher, tmp_path)
    target = generator.cache_path(contract, 1)

    with pytest.raises(TeacherResponseError, match="mutated"):
        generator.generate(contract, 1)
    assert contract["definition"]["template"][0]["text"] == "Classify the message."
    assert not target.exists()


@pytest.mark.parametrize("total", [-1, True, 1.5, 100_001])
def test_invalid_total_is_rejected_before_teacher_or_cache(tmp_path: Path, total: object) -> None:
    teacher = FakeTeacher()
    generator = SyntheticDatasetGenerator(teacher, tmp_path)

    with pytest.raises(DatasetConfigurationError, match="case count"):
        generator.generate(ir(), total)  # type: ignore[arg-type]
    assert teacher.calls == []


def test_dataset_specific_count_cap_is_enforced_before_teacher(tmp_path: Path) -> None:
    teacher = FakeTeacher()

    with pytest.raises(DatasetConfigurationError, match="dataset case count exceeds"):
        SyntheticDatasetGenerator(teacher, tmp_path).generate(ir(), 20_001)
    assert teacher.calls == []


def test_total_below_gold_count_is_rejected_before_teacher(tmp_path: Path) -> None:
    teacher = FakeTeacher()
    contract = ir(examples=[example("one", True), example("two", False)])

    with pytest.raises(DatasetConfigurationError, match="gold"):
        SyntheticDatasetGenerator(teacher, tmp_path).generate(contract, 1)
    assert teacher.calls == []


@pytest.mark.parametrize(
    "bad_example",
    [
        {"inputs": {"wrong": "known"}, "output": True},
        {"inputs": {"message": "known"}, "output": "not-boolean"},
        {"inputs": {"message": "known"}},
    ],
)
def test_invalid_gold_is_rejected_before_teacher(
    tmp_path: Path, bad_example: dict[str, Any]
) -> None:
    teacher = FakeTeacher()

    with pytest.raises(DatasetConfigurationError, match="gold"):
        SyntheticDatasetGenerator(teacher, tmp_path).generate(ir(examples=[bad_example]), 1)
    assert teacher.calls == []


def test_prompt_or_teacher_change_invalidates_cache(tmp_path: Path) -> None:
    first_ir = ir()
    second_ir = deepcopy(first_ir)
    second_ir["definition"]["template"][0]["text"] = "Use the revised policy."
    case = GeneratedCase(inputs={"message": "generated"}, output=True)

    first = SyntheticDatasetGenerator(FakeTeacher((case,)), tmp_path)
    first.generate(first_ir, 1)
    revised_teacher = FakeTeacher((case,), model="fake-v2")
    revised = SyntheticDatasetGenerator(revised_teacher, tmp_path)

    assert first.cache_path(first_ir, 1) != first.cache_path(second_ir, 1)
    assert first.cache_path(first_ir, 1) != revised.cache_path(first_ir, 1)
    revised.generate(first_ir, 1)
    assert revised_teacher.calls == [(first_ir, 1)]


def test_source_and_training_metadata_do_not_change_cache_identity(tmp_path: Path) -> None:
    first_ir = ir()
    relocated = deepcopy(first_ir)
    relocated["source"] = {
        "path": "elsewhere/review.sem.ts",
        "line": 900,
        "column": 12,
        "sourceSha256": "f" * 64,
    }
    relocated["trainingProvenance"] = {"status": "complete", "seed": 99}
    generator = SyntheticDatasetGenerator(FakeTeacher(), tmp_path)

    assert generator.cache_path(first_ir, 0) == generator.cache_path(relocated, 0)


def test_corrupt_cache_fails_closed_without_recalling_teacher(tmp_path: Path) -> None:
    contract = ir()
    case = GeneratedCase(inputs={"message": "generated"}, output=True)
    first = SyntheticDatasetGenerator(FakeTeacher((case,)), tmp_path)
    cache_path = first.cache_path(contract, 1)
    first.generate(contract, 1)
    cache_path.write_bytes(b'{"kind":"semantscript.training-dataset"')
    fresh_teacher = FakeTeacher((case,))

    with pytest.raises(DatasetCacheError, match="cache"):
        SyntheticDatasetGenerator(fresh_teacher, tmp_path).generate(contract, 1)
    assert fresh_teacher.calls == []


def test_dangling_cache_symlink_fails_closed_without_teacher_call(tmp_path: Path) -> None:
    contract = ir()
    teacher = FakeTeacher((GeneratedCase(inputs={"message": "generated"}, output=True),))
    generator = SyntheticDatasetGenerator(teacher, tmp_path)
    cache_path = generator.cache_path(contract, 1)
    cache_path.parent.mkdir(parents=True)
    cache_path.symlink_to(tmp_path / "missing-cache-target")

    with pytest.raises(DatasetCacheError, match="read"):
        generator.generate(contract, 1)
    assert teacher.calls == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX FIFO regression")
def test_fifo_cache_entry_is_rejected_without_blocking_or_teacher_call(tmp_path: Path) -> None:
    contract = ir()
    teacher = FakeTeacher((GeneratedCase(inputs={"message": "generated"}, output=True),))
    generator = SyntheticDatasetGenerator(teacher, tmp_path)
    cache_path = generator.cache_path(contract, 1)
    cache_path.parent.mkdir(parents=True)
    os.mkfifo(cache_path)

    with pytest.raises(DatasetCacheError, match="regular file"):
        generator.generate(contract, 1)
    assert teacher.calls == []


def test_aggregate_case_size_is_bounded_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = ir()
    teacher = FakeTeacher((GeneratedCase(inputs={"message": "x" * 1_000}, output=True),))
    generator = SyntheticDatasetGenerator(teacher, tmp_path)
    cache_path = generator.cache_path(contract, 1)
    monkeypatch.setattr(dataset_module, "MAXIMUM_DATASET_CACHE_BYTES", 128)

    with pytest.raises(DatasetCacheError, match="cases exceed"):
        generator.generate(contract, 1)
    assert not cache_path.exists()


def test_boolean_dataset_version_is_rejected_without_recalling_teacher(tmp_path: Path) -> None:
    contract = ir()
    case = GeneratedCase(inputs={"message": "generated"}, output=True)
    first = SyntheticDatasetGenerator(FakeTeacher((case,)), tmp_path)
    cache_path = first.cache_path(contract, 1)
    first.generate(contract, 1)
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    document["datasetVersion"] = True
    cache_path.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fresh_teacher = FakeTeacher((case,))

    with pytest.raises(DatasetCacheError, match="version"):
        SyntheticDatasetGenerator(fresh_teacher, tmp_path).generate(contract, 1)
    assert fresh_teacher.calls == []


def test_same_request_produces_same_digest_and_canonical_bytes(tmp_path: Path) -> None:
    contract = ir(examples=[example("known", True)])
    case = GeneratedCase(inputs={"message": "generated"}, output=False)
    first = SyntheticDatasetGenerator(FakeTeacher((case,)), tmp_path / "first")
    second = SyntheticDatasetGenerator(FakeTeacher((case,)), tmp_path / "second")

    first_dataset = first.generate(contract, 2)
    second_dataset = second.generate(deepcopy(contract), 2)

    assert first_dataset == second_dataset
    assert first_dataset.payload_sha256 == second_dataset.payload_sha256
    assert first_dataset.dataset_sha256 == second_dataset.dataset_sha256
    assert len(first_dataset.payload_sha256) == 64
    assert len(first_dataset.dataset_sha256) == 64
    assert first.cache_path(contract, 2).read_bytes() == second.cache_path(contract, 2).read_bytes()


def test_v1_empty_dataset_matches_pinned_golden_bytes_and_digests(tmp_path: Path) -> None:
    contract = ir()
    generator = SyntheticDatasetGenerator(FakeTeacher(), tmp_path)

    dataset = generator.generate(contract, 0)
    expected = Path(__file__).with_name("training_dataset_v1_empty.golden.json").read_bytes()

    assert generator.cache_path(contract, 0).read_bytes() == expected
    assert dataset.cache_key_sha256 == (
        "e9e5480283a3c92999c93511b2499ef0ce63c87a27c7fb9f66c6d8caa2fbf1e2"
    )
    assert dataset.payload_sha256 == (
        "b7f30f2e6cc2256ef2be14d68f9a8d72c0adb6f16d86377a92a89dc15d631197"
    )
    assert dataset.dataset_sha256 == (
        "1ac2c055c250b4b31c6ed8ce1b5fa3ae97686cf38740dff324419869f4014179"
    )


def test_cache_path_uses_versioned_sharded_request_digest(tmp_path: Path) -> None:
    generator = SyntheticDatasetGenerator(FakeTeacher(), tmp_path)

    path = generator.cache_path(ir(), 0)

    assert path.parent.parent == tmp_path / "datasets" / "v1"
    assert path.parent.name == path.stem[:2]
    assert len(path.stem) == 64
    assert path.suffix == ".json"


def test_provider_failure_does_not_publish_a_cache_entry(tmp_path: Path) -> None:
    failure = TeacherTransportError("provider unavailable")
    teacher = FakeTeacher(failure=failure)
    generator = SyntheticDatasetGenerator(teacher, tmp_path)
    cache_path = generator.cache_path(ir(), 1)

    with pytest.raises(TeacherTransportError, match="provider unavailable"):
        generator.generate(ir(), 1)
    assert teacher.calls == [(ir(), 1)]
    assert not cache_path.exists()


def test_publication_failure_is_typed_and_never_exposes_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = ir()
    teacher = FakeTeacher((GeneratedCase(inputs={"message": "generated"}, output=True),))
    generator = SyntheticDatasetGenerator(teacher, tmp_path)
    cache_path = generator.cache_path(contract, 1)

    def fail_replace(source: str | Path, target: str | Path) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(dataset_module.os, "replace", fail_replace)

    with pytest.raises(DatasetCacheError, match="publish"):
        generator.generate(contract, 1)
    assert teacher.calls == [(contract, 1)]
    assert not cache_path.exists()


def example(message: str, output: bool) -> dict[str, Any]:
    return {"inputs": {"message": message}, "output": output}


def ir(*, examples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "kind": "semantscript.neural-function",
        "irVersion": 1,
        "stage": "lowered",
        "id": "nf_" + "1" * 64,
        "semanticSha256": "2" * 64,
        "source": {
            "path": "example.sem.ts",
            "line": 1,
            "column": 1,
            "sourceSha256": "3" * 64,
        },
        "definition": {
            "template": [{"kind": "text", "text": "Classify the message."}],
            "examples": examples or [],
            "constraints": [],
        },
        "inputs": [
            {
                "name": "message",
                "index": 0,
                "tsType": "string",
                "type": {"kind": "string"},
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
