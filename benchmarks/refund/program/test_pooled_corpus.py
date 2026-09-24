from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.refund.program.claude_cli_teacher import (
    ClaudeCliTeacherConfig,
    ClaudeCliTrainingTeacher,
)
from benchmarks.refund.program.pooled_corpus import (
    POOLED_CONFIG_KIND,
    PooledCorpusTeacher,
    pooled_teacher_from_projection,
)

from semantscript_trainer import TeacherConfigurationError
from semantscript_trainer.semantic_json import semantic_json_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
POOL = REPOSITORY_ROOT / "benchmarks/refund/data/uci-pool/candidates.json"
HELDOUT = REPOSITORY_ROOT / "benchmarks/refund/data/heldout-uci-2026-09-23"


def _teacher(replay: Path | None = None) -> PooledCorpusTeacher:
    return PooledCorpusTeacher(
        pool_path=POOL,
        replay_path=replay,
        heldout_directory=HELDOUT,
        fraud_percent=15,
        adversarial=ClaudeCliTrainingTeacher(ClaudeCliTeacherConfig(model="claude-opus-5-5")),
    )


def test_pool_labels_are_unique_admissible_outputs_and_exclude_held_out(
    compiled_program,
) -> None:
    teacher = _teacher()
    ir = compiled_program.source_ir
    cases = teacher.labeled_pool(ir)
    report = teacher.last_pool_report
    assert report is not None
    assert report.excluded_held_out == 240
    assert report.ambiguous == 0
    assert report.labeled == len(cases) > 7000
    held_out = {
        case["inputSha256"]
        for name in ("release-verification.json", "final-benchmark-dataset.json")
        for case in json.loads((HELDOUT / name).read_text())["cases"]
    }
    digests = [semantic_json_sha256(case.inputs) for case in cases]
    assert not held_out.intersection(digests)
    assert len(set(digests)) == len(digests)
    assert set(report.label_counts) == {'"approve"', '"deny"', '"review"'}

    assert teacher.generate(ir, 3) == cases[:3]
    with pytest.raises(TeacherConfigurationError, match="can supply"):
        teacher.generate(ir, len(cases) + 1)


def test_projection_round_trips_and_binds_inputs(compiled_program) -> None:
    teacher = _teacher()
    projection = teacher.configuration_projection
    assert projection["kind"] == POOLED_CONFIG_KIND
    assert projection["pool"]["path"] == "benchmarks/refund/data/uci-pool/candidates.json"
    assert projection["exclusions"]["count"] == 240
    rebuilt = pooled_teacher_from_projection(projection)
    assert rebuilt.descriptor == teacher.descriptor
    assert rebuilt.descriptor.provider.startswith("real-input-pool-rule-labels")
    assert rebuilt.descriptor.model == "claude-opus-5-5"

    with pytest.raises(TeacherConfigurationError, match="0 through 100"):
        PooledCorpusTeacher(
            pool_path=POOL,
            replay_path=None,
            heldout_directory=HELDOUT,
            fraud_percent=101,
            adversarial=ClaudeCliTrainingTeacher(ClaudeCliTeacherConfig()),
        )


@pytest.fixture(scope="module")
def compiled_program(tmp_path_factory: pytest.TempPathFactory):
    from benchmarks.refund.program.pipeline import compile_refund_program

    return compile_refund_program(tmp_path_factory.mktemp("pooled-compile") / "compiler")


def test_stale_status_twins_add_rule_labeled_denials_for_both_statuses(compiled_program) -> None:
    ir = compiled_program.source_ir
    plain = _teacher().labeled_pool(ir)
    twinned_teacher = PooledCorpusTeacher(
        pool_path=POOL,
        replay_path=None,
        heldout_directory=HELDOUT,
        fraud_percent=15,
        adversarial=ClaudeCliTrainingTeacher(ClaudeCliTeacherConfig(model="claude-opus-5-5")),
        stale_status_twins=True,
    )
    twinned = twinned_teacher.labeled_pool(ir)
    stale = [c for c in twinned if c.inputs["order"]["ageDays"] > 90]
    assert len(twinned) > len(plain)
    assert all(c.output == "deny" for c in stale)
    statuses = {c.inputs["order"]["status"] for c in stale}
    assert statuses == {"paid", "fraudulent"}
    assert twinned_teacher.configuration_projection["staleStatusTwins"] is True
    assert twinned_teacher.descriptor != _teacher().descriptor
    rebuilt = pooled_teacher_from_projection(twinned_teacher.configuration_projection)
    assert rebuilt.descriptor == twinned_teacher.descriptor
