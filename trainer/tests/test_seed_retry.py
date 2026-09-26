"""The seed retry's settings and its narrow-failure classification."""

from __future__ import annotations

from dataclasses import replace

import pytest

from semantscript_trainer.verification import (
    CalibrationRecordV1,
    HeadVerificationV1,
    HeldOutConstraintEvidence,
    SeedRetryConfig,
    VerificationConfig,
    VerificationConfigurationError,
    VerificationMetricsV1,
    VerificationResult,
    seed_retry_decision,
)

FUNCTION_ID = "nf_" + "1" * 64
TOLERANCE = VerificationConfig(maximum_constraint_violation_rate=0.01, ece_threshold=0.1)


def result(
    *,
    violations: int = 0,
    records: int | None = 394,
    ece: float = 0.05,
    example_failures: int = 0,
    type_errors: int = 0,
    failed: bool = True,
    held_out: HeldOutConstraintEvidence | None = None,
) -> VerificationResult:
    calibration = CalibrationRecordV1(
        temperature=1.0, ece=ece, brier=0.1, sample_count=10, split_sha256="2" * 64, ece_bins=15
    )
    head = HeadVerificationV1(
        output_path="", accuracy=0.95, pair_consistency=1.0, calibration=calibration
    )
    metrics = VerificationMetricsV1(
        accuracy=0.95,
        ece=ece,
        brier=0.1,
        pair_consistency=1.0,
        heads=(head,),
        example_failures=example_failures,
        constraint_violations=violations,
        type_errors=type_errors,
    )
    return VerificationResult(
        function_id=FUNCTION_ID,
        semantic_sha256="3" * 64,
        model_state_sha256="4" * 64,
        tokenizer_sha256="5" * 64,
        status="failed" if failed else "passed",
        verified_at="2026-09-25T00:00:00Z",
        metrics=metrics,
        attested_cases=1,
        pair_count=0,
        failures=("a gate failed",) if failed else (),
        held_out=held_out,
        record_count=records,
    )


def test_defaults_and_bounds() -> None:
    assert SeedRetryConfig() == SeedRetryConfig(attempts=3, margin=2.0)
    assert SeedRetryConfig(attempts=1, margin=1).attempts == 1
    for attempts in (0, 21, True, 2.0):
        with pytest.raises(VerificationConfigurationError, match="seed attempts"):
            SeedRetryConfig(attempts=attempts)  # type: ignore[arg-type]
    for margin in (0.5, 10.5, float("nan"), float("inf"), True, "2"):
        with pytest.raises(VerificationConfigurationError, match="seed retry margin"):
            SeedRetryConfig(margin=margin)  # type: ignore[arg-type]


def test_record_count_is_validated_and_not_part_of_equality() -> None:
    measured = result(violations=4)
    assert measured == replace(measured, record_count=None)
    with pytest.raises(VerificationConfigurationError, match="record_count"):
        result(records=0)


def test_suggestions_are_one_per_failure_and_not_part_of_equality() -> None:
    failed = result(violations=4)
    assert failed.suggestions == ()
    assert replace(failed, suggestions=("rerun with --no-cache",)) == failed
    for suggestions in (("a", "b"), ("",)):
        with pytest.raises(VerificationConfigurationError, match="one non-empty string"):
            replace(failed, suggestions=suggestions)


def test_rate_within_the_margin_retries() -> None:
    # The Express example's seeds 1 to 3: 4, 6 and 5 of 394 against 1%.
    for violations in (4, 6, 5):
        decision = seed_retry_decision(
            [result(violations=violations)], TOLERANCE, SeedRetryConfig()
        )
        assert decision.retry, decision.reason
        assert f"({violations} of 394) within 2 x tolerance 0.01" in decision.reason


def test_rate_outside_the_margin_stops_and_says_why() -> None:
    # Seed 4: 8 of 394 is 2.03%, past 2 x 1%; a margin of 3 admits it.
    decision = seed_retry_decision([result(violations=8)], TOLERANCE, SeedRetryConfig())
    assert not decision.retry
    assert decision.reason == (
        f"{FUNCTION_ID}: violation rate 2.0305% (8 of 394) is outside the retry margin "
        "2 x 0.01 = 2.0000%"
    )
    assert seed_retry_decision([result(violations=8)], TOLERANCE, SeedRetryConfig(margin=3)).retry


def test_ece_within_and_outside_the_margin() -> None:
    assert seed_retry_decision([result(ece=0.19)], TOLERANCE, SeedRetryConfig()).retry
    decision = seed_retry_decision([result(ece=0.25)], TOLERANCE, SeedRetryConfig())
    assert not decision.retry and "ECE 0.2500 is outside the retry margin 2 x 0.1" in (
        decision.reason
    )
    both = seed_retry_decision([result(violations=5, ece=0.15)], TOLERANCE, SeedRetryConfig())
    assert both.retry and "violation rate" in both.reason and "ECE 0.1500" in both.reason


def test_gold_misses_type_errors_and_zero_tolerance_never_retry() -> None:
    gold = seed_retry_decision(
        [result(violations=4, example_failures=1)], TOLERANCE, SeedRetryConfig(margin=10)
    )
    assert not gold.retry and "1 gold/human example prediction(s) failed" in gold.reason
    assert "not a seed effect" in gold.reason
    typed = seed_retry_decision([result(type_errors=2)], TOLERANCE, SeedRetryConfig())
    assert not typed.retry and "2 output type check(s) failed" in typed.reason
    zero = seed_retry_decision(
        [result(violations=1)], VerificationConfig(), SeedRetryConfig(margin=10)
    )
    assert not zero.retry and "under a zero tolerance" in zero.reason


def test_one_non_narrow_function_stops_the_whole_attempt() -> None:
    narrow = result(violations=4)
    other = replace(result(example_failures=1), function_id="nf_" + "9" * 64)
    decision = seed_retry_decision([narrow, other], TOLERANCE, SeedRetryConfig())
    assert not decision.retry and decision.reason.startswith("nf_" + "9" * 64)


def test_passing_or_unclassifiable_results_do_not_retry() -> None:
    assert seed_retry_decision(
        [result(failed=False)], TOLERANCE, SeedRetryConfig()
    ) == seed_retry_decision([], TOLERANCE, SeedRetryConfig())
    unknown = seed_retry_decision(
        [result(violations=4, records=None)], TOLERANCE, SeedRetryConfig()
    )
    assert not unknown.retry and "record count is unknown" in unknown.reason
    unclassified = seed_retry_decision([result()], TOLERANCE, SeedRetryConfig())
    assert not unclassified.retry and "does not classify" in unclassified.reason
    with pytest.raises(VerificationConfigurationError):
        seed_retry_decision([result()], TOLERANCE, object())  # type: ignore[arg-type]


def held_out(violating: int, size: int = 512) -> HeldOutConstraintEvidence:
    return HeldOutConstraintEvidence(
        sample_size=size, violating_inputs=violating, violations=violating, seed=3
    )


def test_held_out_rate_is_classified_like_the_corpus_rate() -> None:
    # 8 of 512 is 1.56%: within 2 x 1% but not within 1.2 x 1%.
    narrow = seed_retry_decision([result(held_out=held_out(8))], TOLERANCE, SeedRetryConfig())
    assert narrow.retry
    assert "held-out violation rate 1.5625% (8 of 512 held-out inputs) within 2 x" in (
        narrow.reason
    )
    outside = seed_retry_decision(
        [result(held_out=held_out(8))], TOLERANCE, SeedRetryConfig(margin=1.2)
    )
    assert not outside.retry
    assert "held-out violation rate 1.5625%" in outside.reason
    assert "outside the retry margin 1.2 x 0.01" in outside.reason
    zero = seed_retry_decision(
        [result(held_out=held_out(1))], VerificationConfig(), SeedRetryConfig()
    )
    assert not zero.retry and "1 of 512 held-out inputs broke a constraint" in zero.reason
    assert "zero tolerance" in zero.reason
    # Within the tolerance the held-out figure is not a failure the retry classifies.
    within = seed_retry_decision([result(held_out=held_out(2))], TOLERANCE, SeedRetryConfig())
    assert not within.retry and "does not classify" in within.reason


def test_held_out_evidence_is_validated() -> None:
    assert held_out(0).rate == 0.0
    assert HeldOutConstraintEvidence(0, 0, 0, 1).rate == 0.0
    assert held_out(4).to_document() == {
        "sampleSize": 512,
        "violations": 4,
        "violationRate": 4 / 512,
        "seed": 3,
    }
    assert HeldOutConstraintEvidence.from_record(held_out(4).to_record()) == held_out(4)
    with pytest.raises(VerificationConfigurationError, match="violating_inputs"):
        HeldOutConstraintEvidence(10, 11, 11, 1)
    with pytest.raises(VerificationConfigurationError, match="held-out violations"):
        HeldOutConstraintEvidence(10, 2, 1, 1)
    with pytest.raises(VerificationConfigurationError, match="held-out seed"):
        HeldOutConstraintEvidence(10, 0, 0, -1)
    with pytest.raises(VerificationConfigurationError, match="held_out_samples"):
        VerificationConfig(held_out_samples=0)
    assert VerificationConfig().held_out_samples == 512
