from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from semantscript_trainer.adversarial import (  # noqa: E402
    AdversarialCase,
    AdversarialDataset,
    AdversarialGenerationConfig,
    CounterfactualPair,
)
from semantscript_trainer.canonical_input import (  # noqa: E402
    serialize_canonical_inputs_string,
)
from semantscript_trainer.dataset import DatasetCase, TrainingDataset  # noqa: E402
from semantscript_trainer.remedies import remedy  # noqa: E402
from semantscript_trainer.teacher import TeacherDescriptor  # noqa: E402
from semantscript_trainer.training import (  # noqa: E402
    EpochMetrics,
    TrainingConfig,
    TrainingResult,
)
from semantscript_trainer.training_contract import (  # noqa: E402
    MAXIMUM_TRAINING_ROW_COUNT,
    HeldOutSplitConfig,
    TrainingCorpus,
    TrainingRow,
    TrainingSplit,
    assemble_training_corpus,
    split_training_corpus,
)
from semantscript_trainer.verification import (  # noqa: E402
    SeedRetryConfig,
    VerificationConfig,
    VerificationConfigurationError,
    VerificationExecutionError,
    VerificationGateError,
    calibration_split_sha256,
    evaluate_training_result,
    model_state_sha256,
    seed_retry_decision,
    tokenizer_json_bytes,
    verify_training_result,
)

VERIFIED_AT = "2026-09-22T12:34:56Z"


class FixedTokenizer:
    """One token per known text; ``default`` (when set) for every other text.

    The held-out constraint sample reaches inputs no fixture lists, so a fixture
    whose function has constraints gives them a token (and so one logit).
    """

    def __init__(self, token_by_text: dict[str, int], default: int | None = None) -> None:
        self._token_by_text = token_by_text
        self._default = default
        self.semantscript_tokenizer_json = json.dumps(
            {"kind": "semantscript-test-tokenizer", "tokens": token_by_text},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

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
        return {
            "input_ids": torch.tensor(
                [
                    [
                        self._token_by_text[text]
                        if self._default is None
                        else self._token_by_text.get(text, self._default)
                    ]
                    for text in texts
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones((len(texts), 1), dtype=torch.long),
        }


class FixedBinaryModel(torch.nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32)[:, None])

    def forward(self, *, input_ids, attention_mask):
        del attention_mask
        return self.logits[input_ids[:, 0]]


class EvalFailureModel(torch.nn.Module):
    def eval(self) -> None:
        raise RuntimeError("injected eval failure")


class MutatingBinaryModel(FixedBinaryModel):
    def forward(self, *, input_ids, attention_mask):
        result = super().forward(input_ids=input_ids, attention_mask=attention_mask)
        self.logits.add_(0.25)
        return result


class MutatingTokenizer(FixedTokenizer):
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        self.semantscript_tokenizer_json = b'{"kind":"mutated-during-verification"}'
        return result


def test_passing_binary_flow_emits_exact_ir_and_manifest_projections() -> None:
    contract, base, _, training, tokenizer = binary_fixture({0: -12.0, 1: 12.0, 2: 12.0})

    result = verify_training_result(
        contract,
        training,
        base,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
    )

    calibration = result.metrics.heads[0].calibration
    expected_calibration = {
        "method": "temperature-scaling",
        "temperature": calibration.temperature,
        "ece": calibration.ece,
        "brier": calibration.brier,
        "sampleCount": 1,
        "splitSha256": calibration_split_sha256(training, training.split.evaluation),
        "eceBins": 15,
    }
    expected_head = {
        "outputPath": "",
        "accuracy": 1.0,
        "pairConsistency": 1.0,
        "calibration": expected_calibration,
    }

    assert result.status == "passed"
    assert result.attested_cases == 1
    assert result.pair_count == 0
    assert result.metrics.accuracy == 1.0
    assert result.metrics.example_failures == 0
    assert result.metrics.constraint_violations == 0
    assert len(result.model_state_sha256) == 64
    assert (
        result.tokenizer_sha256 == hashlib.sha256(tokenizer.semantscript_tokenizer_json).hexdigest()
    )
    assert result.to_ir_document() == {
        "status": "passed",
        "verifiedAt": VERIFIED_AT,
        "metrics": {
            "accuracy": 1.0,
            "ece": result.metrics.ece,
            "brier": result.metrics.brier,
            "pairConsistency": 1.0,
            "heads": [expected_head],
            "exampleFailures": 0,
            "constraintViolations": 0,
            "typeErrors": 0,
        },
    }
    assert result.to_manifest_head_metadata() == {
        "calibration": expected_calibration,
        "verification": {"accuracy": 1.0, "pairConsistency": 1.0},
    }
    assert result.to_manifest_function_verification() == {
        "status": "passed",
        "accuracy": 1.0,
        "ece": result.metrics.ece,
        "brier": result.metrics.brier,
        "pairConsistency": 1.0,
        "attestedCases": 1,
        "exampleFailures": 0,
        "constraintViolations": 0,
        "typeErrors": 0,
        # No constraints: an empty held-out sample, recorded with the seed it used.
        "heldOutConstraints": {"sampleSize": 0, "violations": 0, "violationRate": 0.0, "seed": 1},
    }
    assert len(training.split.evaluation) == 1
    assert training.split.evaluation[0].origin != "gold"


def test_gold_miss_fails_build_and_retains_typed_report() -> None:
    contract, base, _, training, tokenizer = binary_fixture({0: 12.0, 1: 12.0, 2: 12.0})

    result = evaluate_training_result(
        contract,
        training,
        base,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
    )

    assert result.status == "failed"
    assert result.metrics.example_failures == 1
    assert result.failures == (
        "1 gold/human example prediction(s) failed:\n"
        '  - gold example base:0: inputs {"score":0} expected false, predicted true',
    )
    # The next step comes from the evidence: no teacher-labelled row is
    # mispredicted like the gold miss, so the example itself is to be checked.
    assert result.suggestions == (
        remedy("gold-check-example", case="base:0", expected="false", predicted="true"),
    )
    with pytest.raises(VerificationGateError) as caught:
        verify_training_result(
            contract,
            training,
            base,
            tokenizer=tokenizer,
            verified_at=VERIFIED_AT,
        )
    assert caught.value.result.status == "failed"
    assert caught.value.result.metrics.example_failures == 1
    with pytest.raises(VerificationGateError):
        result.to_manifest_function_verification()
    with pytest.raises(VerificationConfigurationError, match="passing verification"):
        replace(result, status="passed", failures=())


def test_verification_result_rejects_noncanonical_function_identity() -> None:
    contract, base, _, training, tokenizer = binary_fixture({0: -12.0, 1: 12.0, 2: 12.0})
    result = verify_training_result(
        contract,
        training,
        base,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
    )

    with pytest.raises(VerificationConfigurationError, match="function_id"):
        replace(result, function_id="nf_bad")
    with pytest.raises(VerificationConfigurationError, match="model_state_sha256"):
        replace(result, model_state_sha256="bad")
    with pytest.raises(VerificationConfigurationError, match="tokenizer_sha256"):
        replace(result, tokenizer_sha256="bad")


def test_evidence_fingerprints_are_deterministic_and_bind_exact_state() -> None:
    contract, base, _, training, tokenizer = binary_fixture({0: -12.0, 1: 12.0, 2: 12.0})

    first = verify_training_result(
        contract,
        training,
        base,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
    )
    second = verify_training_result(
        contract,
        training,
        base,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
    )

    assert first.model_state_sha256 == second.model_state_sha256
    assert first.model_state_sha256 == model_state_sha256(training.model)
    assert first.tokenizer_sha256 == second.tokenizer_sha256
    assert tokenizer_json_bytes(tokenizer) == tokenizer.semantscript_tokenizer_json
    assert "modelStateSha256" not in first.to_ir_document()
    assert "tokenizerSha256" not in first.to_ir_document()
    assert "modelStateSha256" not in first.to_manifest_function_verification()
    assert "tokenizerSha256" not in first.to_manifest_function_verification()

    with torch.no_grad():
        training.model.logits.add_(0.25)
    mutated_model = verify_training_result(
        contract,
        training,
        base,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
    )
    assert mutated_model.model_state_sha256 != first.model_state_sha256
    assert mutated_model.tokenizer_sha256 == first.tokenizer_sha256

    tokenizer.semantscript_tokenizer_json = b'{"kind":"different-test-tokenizer"}'
    mutated_tokenizer = verify_training_result(
        contract,
        training,
        base,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
    )
    assert mutated_tokenizer.model_state_sha256 == mutated_model.model_state_sha256
    assert mutated_tokenizer.tokenizer_sha256 != first.tokenizer_sha256


def test_hugging_face_tokenizer_json_protocol_returns_normalized_bytes() -> None:
    class Backend:
        def to_str(self, *, pretty: bool) -> str:
            assert pretty is False
            return '{"model":{"type":"WordLevel"},"version":"1.0"}'

    class Tokenizer:
        backend_tokenizer = Backend()

    # Runtime padding/truncation state is cleared so the digest is stable across encodes.
    expected = b'{"model":{"type":"WordLevel"},"version":"1.0","padding":null,"truncation":null}'
    assert tokenizer_json_bytes(Tokenizer()) == expected


@pytest.mark.parametrize(
    "tokenizer",
    (object(), type("InvalidJsonTokenizer", (), {"semantscript_tokenizer_json": b"{"})()),
)
def test_verification_fails_closed_without_valid_tokenizer_json(tokenizer: object) -> None:
    contract, base, _, training, _ = binary_fixture({0: -12.0, 1: 12.0, 2: 12.0})

    with pytest.raises(VerificationExecutionError, match="tokenizer JSON"):
        evaluate_training_result(
            contract,
            training,
            base,
            tokenizer=tokenizer,
            verified_at=VERIFIED_AT,
        )


def test_state_or_tokenizer_mutation_during_verification_is_rejected() -> None:
    contract, base, _, training, tokenizer = binary_fixture({0: -12.0, 1: 12.0, 2: 12.0})
    mutating_model = MutatingBinaryModel([-12.0, 12.0, 12.0])

    with pytest.raises(VerificationExecutionError, match="state changed"):
        evaluate_training_result(
            contract,
            replace(training, model=mutating_model),
            base,
            tokenizer=tokenizer,
            verified_at=VERIFIED_AT,
        )

    with pytest.raises(VerificationExecutionError, match="tokenizer JSON changed"):
        evaluate_training_result(
            contract,
            training,
            base,
            tokenizer=MutatingTokenizer(tokenizer._token_by_text),
            verified_at=VERIFIED_AT,
        )


def test_typed_verification_records_reject_contradictory_scalar_evidence() -> None:
    contract, base, _, training, tokenizer = binary_fixture({0: -12.0, 1: 12.0, 2: 12.0})
    result = verify_training_result(
        contract,
        training,
        base,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
    )

    with pytest.raises(VerificationConfigurationError, match="sole head metrics"):
        replace(result.metrics, accuracy=0.5)

    inconsistent_head = replace(result.metrics.heads[0], pair_consistency=0.5)
    inconsistent_metrics = replace(
        result.metrics,
        pair_consistency=0.5,
        heads=(inconsistent_head,),
    )
    with pytest.raises(VerificationConfigurationError, match="without counterfactual pairs"):
        replace(result, metrics=inconsistent_metrics)


def test_constraint_violation_and_counterfactual_pair_failure_are_measured() -> None:
    contract, base, adversarial, training, tokenizer = constrained_fixture()

    result = evaluate_training_result(
        contract,
        training,
        base,
        adversarial,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
    )

    assert result.status == "failed"
    assert result.metrics.constraint_violations == 3
    assert result.metrics.pair_consistency == 0.0
    assert result.metrics.heads[0].pair_consistency == 0.0
    assert result.pair_count == 1
    constraint_failure = next(f for f in result.failures if "adversarial constraint" in f)
    assert constraint_failure.startswith(
        "3 adversarial constraint check(s) failed (0.428571 of 7 records exceeds the "
        "configured tolerance 0):\n"
        '  - constraint 0 (score >= 10) violated by case base:2: inputs {"score":20} '
        "predicted false\n"
    )
    assert constraint_failure.count("\n  - ") == 3
    # The seed retry reads the rate from the measured record count (3 of 7).
    assert result.record_count == 7
    zero = seed_retry_decision([result], VerificationConfig(), SeedRetryConfig())
    assert not zero.retry and "zero tolerance" in zero.reason
    # ECE also failed on this fixture, so give it room and look at the rate alone.
    loose_ece = {"ece_threshold": 1.0}
    within = seed_retry_decision(
        [result],
        VerificationConfig(maximum_constraint_violation_rate=0.25, **loose_ece),
        SeedRetryConfig(margin=2),
    )
    assert within.retry and "violation rate 42.8571% (3 of 7) within 2 x tolerance 0.25" in (
        within.reason
    )
    outside = seed_retry_decision(
        [result],
        VerificationConfig(maximum_constraint_violation_rate=0.2, **loose_ece),
        SeedRetryConfig(margin=2),
    )
    assert not outside.retry and "outside the retry margin 2 x 0.2 = 40.0000%" in outside.reason

    tolerated = evaluate_training_result(
        contract,
        training,
        base,
        adversarial,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
        config=VerificationConfig(maximum_constraint_violation_rate=1.0),
    )
    assert tolerated.metrics.constraint_violations == 3
    assert not any("adversarial constraint" in failure for failure in tolerated.failures)
    # The pair-consistency gate is separate; with tolerated violations the only
    # remaining failures must not mention constraints, and a fully tolerated
    # result may carry status "passed" with a non-zero violation count.
    if tolerated.status == "passed":
        assert tolerated.to_manifest_function_verification()["constraintViolations"] == 3

    with pytest.raises(VerificationConfigurationError, match="maximum_constraint_violation_rate"):
        VerificationConfig(maximum_constraint_violation_rate=1.5)


MEMORISED = {0: -12.0, 1: -12.0, 20: 12.0, 9: -12.0, 10: 12.0, 11: 12.0}


def test_a_model_that_keeps_the_rule_only_on_its_corpus_fails_the_held_out_gate() -> None:
    # Right on every corpus record (the boundary pair 9/10, the twin 11 and the
    # synthetic 20) and "false" everywhere else: the corpus check sees nothing.
    contract, base, adversarial, training, tokenizer = constrained_fixture(
        MEMORISED, unseen_logit=-12.0
    )
    loose_ece = VerificationConfig(ece_threshold=1.0)
    result = evaluate_training_result(
        contract,
        training,
        base,
        adversarial,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
        config=loose_ece,
    )

    assert result.metrics.constraint_violations == 0
    assert result.status == "failed"
    held_out = result.held_out
    assert held_out is not None and held_out.seed == training.config.seed
    assert held_out.sample_size > 10 and 0 < held_out.violating_inputs < held_out.sample_size
    assert held_out.violations == held_out.violating_inputs
    (failure,) = result.failures
    assert failure.startswith(
        f"held-out constraint check failed on {held_out.violating_inputs} of "
        f"{held_out.sample_size} sampled inputs ("
    )
    assert f"seed {training.config.seed}; broken: constraint 0 on" in failure
    assert "none of them a training input" in failure
    offending = [line for line in failure.splitlines() if line.startswith("  - constraint 0")]
    assert len(offending) == min(held_out.violating_inputs, 10)
    for line in offending:
        assert line.startswith(
            '  - constraint 0 (score >= 10) violated by held-out input {"score":'
        )
        assert line.endswith(": predicted false")
        score = float(line.split('{"score":', 1)[1].split("}", 1)[0])
        assert score >= 10 and score not in (10, 11, 20)
    # The next step pastes one offending input with the output the rule requires.
    (suggestion,) = result.suggestions
    assert suggestion.startswith('add the examples entry { inputs: {"score": ')
    assert "output: true }: constraint 0 (score >= 10) is broken on" in suggestion
    assert f"of {held_out.sample_size} held-out inputs (seed {training.config.seed})" in (
        suggestion
    )
    assert "--held-out-samples" not in suggestion
    # The same seed draws the same sample; a zero tolerance never retries it.
    again = evaluate_training_result(
        contract,
        training,
        base,
        adversarial,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
        config=loose_ece,
    )
    assert again.held_out == held_out and again.failures == result.failures
    decision = seed_retry_decision([result], loose_ece, SeedRetryConfig())
    assert not decision.retry and "held-out inputs broke a constraint" in decision.reason

    # A tolerance at or above the held-out rate admits the model and the manifest
    # records the figure beside the corpus count.
    tolerated = evaluate_training_result(
        contract,
        training,
        base,
        adversarial,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
        config=replace(loose_ece, maximum_constraint_violation_rate=held_out.rate),
    )
    assert tolerated.status == "passed", tolerated.failures
    projection = tolerated.to_manifest_function_verification()
    assert projection["constraintViolations"] == 0
    assert projection["heldOutConstraints"] == {
        "sampleSize": held_out.sample_size,
        "violations": held_out.violating_inputs,
        "violationRate": held_out.rate,
        "seed": training.config.seed,
    }


def test_held_out_sample_follows_its_seed_and_size() -> None:
    contract, base, adversarial, training, tokenizer = constrained_fixture(
        MEMORISED, unseen_logit=-12.0
    )

    def held_out(**kwargs: Any) -> Any:
        return evaluate_training_result(
            contract,
            training,
            base,
            adversarial,
            tokenizer=tokenizer,
            verified_at=VERIFIED_AT,
            **kwargs,
        ).held_out

    first = held_out(held_out_seed=7)
    assert first.seed == 7 and held_out(held_out_seed=7) == first
    assert held_out(held_out_seed=8).seed == 8
    small = held_out(held_out_seed=7, config=VerificationConfig(held_out_samples=8))
    assert small.sample_size == 8


def test_a_model_that_keeps_the_rule_everywhere_passes_the_held_out_gate() -> None:
    contract, base, adversarial, training, tokenizer = constrained_fixture(
        MEMORISED, unseen_logit=12.0
    )
    result = evaluate_training_result(
        contract,
        training,
        base,
        adversarial,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
        config=VerificationConfig(ece_threshold=1.0),
    )
    assert result.status == "passed", result.failures
    assert result.held_out is not None and result.held_out.violating_inputs == 0
    assert result.to_manifest_function_verification()["heldOutConstraints"]["violations"] == 0


def test_ece_above_configured_gate_fails_even_without_gold_miss() -> None:
    contract, base, _, training, tokenizer = binary_fixture({0: -12.0, 1: 0.0, 2: 12.0})

    result = evaluate_training_result(
        contract,
        training,
        base,
        tokenizer=tokenizer,
        config=VerificationConfig(ece_threshold=0.1, ece_bins=2),
        verified_at=VERIFIED_AT,
    )

    assert result.status == "failed"
    assert result.metrics.example_failures == 0
    assert result.metrics.ece == pytest.approx(0.5)
    assert result.metrics.brier == pytest.approx(0.25)
    assert result.failures == ("ECE 0.5 exceeds configured threshold 0.1",)
    outside = seed_retry_decision(
        [result], VerificationConfig(ece_threshold=0.1, ece_bins=2), SeedRetryConfig()
    )
    assert not outside.retry
    assert outside.reason.endswith("ECE 0.5000 is outside the retry margin 2 x 0.1 = 0.2000")
    within = seed_retry_decision(
        [result], VerificationConfig(ece_threshold=0.3, ece_bins=2), SeedRetryConfig()
    )
    assert within.retry and within.reason.endswith("ECE 0.5000 within 2 x threshold 0.3")


def test_calibration_split_digest_changes_on_bound_row_or_provenance_mutation() -> None:
    _, _, _, training, _ = binary_fixture({0: -12.0, 1: 12.0, 2: 12.0})
    row = training.split.evaluation[0]
    mutated_row = TrainingRow(
        row_id=row.row_id,
        group_id=row.group_id,
        origin=row.origin,
        inputs=row.inputs,
        label_index=0,
    )

    original = calibration_split_sha256(training, training.split.evaluation)

    assert calibration_split_sha256(training, (mutated_row,)) != original
    assert (
        calibration_split_sha256(
            replace(training, base_dataset_sha256="8" * 64),
            training.split.evaluation,
        )
        != original
    )
    with pytest.raises(VerificationConfigurationError, match="maximum row count"):
        calibration_split_sha256(
            training,
            (row,) * (MAXIMUM_TRAINING_ROW_COUNT + 1),
        )


def test_verifier_rejects_a_tampered_held_out_partition() -> None:
    contract, base, _, training, tokenizer = binary_fixture({0: -12.0, 1: 12.0, 2: 12.0})
    held_out = training.split.evaluation[0]
    replacement = next(
        row
        for row in training.split.training
        if row.origin != "gold" and row.group_id != held_out.group_id
    )
    tampered_training_rows = tuple(
        held_out if row.row_id == replacement.row_id else row for row in training.split.training
    )
    tampered = replace(
        training,
        split=TrainingSplit(training=tampered_training_rows, evaluation=(replacement,)),
    )

    with pytest.raises(VerificationConfigurationError, match="deterministic corpus split"):
        evaluate_training_result(
            contract,
            tampered,
            base,
            tokenizer=tokenizer,
            verified_at=VERIFIED_AT,
        )


def test_verification_inference_respects_the_training_token_cap() -> None:
    contract, base, _, training, tokenizer = binary_fixture({0: -12.0, 1: 12.0, 2: 12.0})
    oversized = replace(
        training,
        config=replace(training.config, maximum_sequence_length=8192),
    )

    with pytest.raises(VerificationConfigurationError, match="maximum token count"):
        evaluate_training_result(
            contract,
            oversized,
            base,
            tokenizer=tokenizer,
            config=VerificationConfig(batch_size=9),
            verified_at=VERIFIED_AT,
        )


def test_verification_config_rejects_temperature_bounds_outside_model_contract() -> None:
    with pytest.raises(VerificationConfigurationError, match="temperature bounds"):
        VerificationConfig(minimum_temperature=0.0001)
    with pytest.raises(VerificationConfigurationError, match="temperature bounds"):
        VerificationConfig(maximum_temperature=1001.0)


def test_model_eval_failure_is_normalized() -> None:
    contract, base, _, training, tokenizer = binary_fixture({0: -12.0, 1: 12.0, 2: 12.0})

    with pytest.raises(VerificationExecutionError, match="evaluation mode"):
        evaluate_training_result(
            contract,
            replace(training, model=EvalFailureModel()),
            base,
            tokenizer=tokenizer,
            verified_at=VERIFIED_AT,
        )


def test_zero_counterfactual_pairs_are_vacuously_consistent() -> None:
    contract, base, _, training, tokenizer = binary_fixture({0: -12.0, 1: 12.0, 2: 12.0})

    result = evaluate_training_result(
        contract,
        training,
        base,
        tokenizer=tokenizer,
        verified_at=VERIFIED_AT,
    )

    assert result.pair_count == 0
    assert result.metrics.pair_consistency == 1.0
    assert result.metrics.heads[0].pair_consistency == 1.0


def binary_fixture(
    logits_by_score: dict[int, float],
) -> tuple[dict[str, Any], TrainingDataset, TrainingCorpus, TrainingResult, FixedTokenizer]:
    contract = boolean_ir()
    base = training_dataset(
        (
            DatasetCase({"score": 0}, False, "gold"),
            DatasetCase({"score": 1}, True, "synthetic"),
            DatasetCase({"score": 2}, True, "synthetic"),
        )
    )
    corpus = assemble_training_corpus(contract, base)
    training, tokenizer = training_result(contract, base, corpus, None, logits_by_score)
    return contract, base, corpus, training, tokenizer


def constrained_fixture(
    logits_by_score: dict[int, float] | None = None,
    *,
    unseen_logit: float = 12.0,
) -> tuple[
    dict[str, Any],
    TrainingDataset,
    AdversarialDataset,
    TrainingResult,
    FixedTokenizer,
]:
    """A ``score >= 10 -> true`` function; ``unseen_logit`` answers every held-out input.

    The default answers ``true`` off the corpus, which keeps the rule everywhere
    (``false`` is only ever forbidden when the score is at least 10).
    """

    contract = boolean_ir(constraints=[minimum_constraint()])
    base = training_dataset(
        (
            DatasetCase({"score": 0}, False, "gold"),
            DatasetCase({"score": 1}, False, "synthetic"),
            DatasetCase({"score": 20}, True, "synthetic"),
        )
    )
    pair_id = "cf_" + "8" * 64
    anchor_id = "ac_" + "9" * 64
    twin_id = "ac_" + "a" * 64
    adversarial = AdversarialDataset(
        function_id=base.function_id,
        base_dataset_sha256=base.dataset_sha256,
        teacher=TeacherDescriptor("fake", "adversarial", "b" * 64),
        config=AdversarialGenerationConfig(),
        cases=(
            AdversarialCase(
                case_id="ac_" + "c" * 64,
                inputs={"score": 9},
                output=False,
                tag="constraint-boundary",
                constraint_index=0,
                predicate_result=False,
            ),
            AdversarialCase(
                case_id="ac_" + "d" * 64,
                inputs={"score": 10},
                output=True,
                tag="constraint-boundary",
                constraint_index=0,
                predicate_result=True,
            ),
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
                inputs={"score": 11},
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
                reason="Crossing the minimum changes the required label.",
            ),
        ),
        cache_key_sha256="c" * 64,
        payload_sha256="d" * 64,
        dataset_sha256="e" * 64,
    )
    corpus = assemble_training_corpus(contract, base, adversarial)
    training, tokenizer = training_result(
        contract,
        base,
        corpus,
        adversarial,
        logits_by_score
        if logits_by_score is not None
        else {0: -12.0, 1: -12.0, 20: -12.0, 9: -12.0, 10: -12.0, 11: -12.0},
        unseen_logit=unseen_logit,
    )
    return contract, base, adversarial, training, tokenizer


def training_result(
    contract: dict[str, Any],
    base: TrainingDataset,
    corpus: TrainingCorpus,
    adversarial: AdversarialDataset | None,
    logits_by_score: dict[int, float],
    *,
    unseen_logit: float | None = None,
) -> tuple[TrainingResult, FixedTokenizer]:
    schema = contract["inputs"]
    texts = {
        serialize_canonical_inputs_string(schema, {"score": score}, version=2): logit
        for score, logit in logits_by_score.items()
    }
    token_by_text = {text: index for index, text in enumerate(texts)}
    logits = [texts[text] for text in token_by_text]
    default = None
    if unseen_logit is not None:
        default = len(logits)
        logits.append(unseen_logit)
    model = FixedBinaryModel(logits)
    config = TrainingConfig(
        epochs=1,
        batch_size=4,
        maximum_sequence_length=8,
        device="cpu",
    )
    split = split_training_corpus(
        corpus,
        HeldOutSplitConfig(
            evaluation_ratio=config.evaluation_ratio,
            seed=config.seed,
        ),
    )
    result = TrainingResult(
        model=model,
        head=corpus.head,
        split=split,
        config=config,
        device="cpu",
        metrics=(EpochMetrics(epoch=1, mean_training_loss=0.0, held_out_accuracy=1.0),),
        function_id=corpus.function_id,
        semantic_sha256=corpus.semantic_sha256,
        base_dataset_sha256=base.dataset_sha256,
        adversarial_dataset_sha256=(None if adversarial is None else adversarial.dataset_sha256),
    )
    return result, FixedTokenizer(token_by_text, default)


def training_dataset(cases: tuple[DatasetCase, ...]) -> TrainingDataset:
    return TrainingDataset(
        function_id="nf_" + "1" * 64,
        semantic_sha256="2" * 64,
        requested_case_count=len(cases),
        teacher=TeacherDescriptor("fake", "base", "4" * 64),
        cases=cases,
        cache_key_sha256="5" * 64,
        payload_sha256="6" * 64,
        dataset_sha256="7" * 64,
    )


def boolean_ir(*, constraints: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "kind": "semantscript.neural-function",
        "irVersion": 1,
        "stage": "lowered",
        "id": "nf_" + "1" * 64,
        "semanticSha256": "2" * 64,
        "source": {
            "path": "verification.sem.ts",
            "line": 1,
            "column": 1,
            "sourceSha256": "3" * 64,
        },
        "definition": {
            "template": [{"kind": "text", "text": "Classify score."}],
            "examples": [{"inputs": {"score": 0}, "output": False}],
            "constraints": constraints or [],
        },
        "inputs": [{"name": "score", "index": 0, "tsType": "number", "type": {"kind": "number"}}],
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


def minimum_constraint() -> dict[str, Any]:
    return {
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


def test_fast_tokenizer_digest_ignores_runtime_padding_and_truncation_state() -> None:
    from semantscript_trainer.verification import tokenizer_json_bytes

    class Backend:
        def __init__(self) -> None:
            self.padding: dict[str, object] | None = None
            self.truncation: dict[str, object] | None = None

        def to_str(self, pretty: bool = False) -> str:
            assert pretty is False
            return json.dumps(
                {
                    "version": "1.0",
                    "truncation": self.truncation,
                    "padding": self.padding,
                    "model": {"type": "WordPiece", "vocab": {"[PAD]": 0, "a": 1}},
                }
            )

    class Tokenizer:
        def __init__(self) -> None:
            self.backend_tokenizer = Backend()

    tokenizer = Tokenizer()
    before = tokenizer_json_bytes(tokenizer)
    tokenizer.backend_tokenizer.truncation = {"max_length": 128, "strategy": "LongestFirst"}
    tokenizer.backend_tokenizer.padding = {"strategy": "BatchLongest", "pad_id": 0}
    after = tokenizer_json_bytes(tokenizer)

    assert before == after
    assert b'"vocab":{"[PAD]":0,"a":1}' in before
    assert b'"padding":null' in before and b'"truncation":null' in before
