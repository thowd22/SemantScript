"""Shared-encoder experiment (TASK-6.1): two refund functions in one artifact.

Function A is the canonical refund decision with its frozen corpus and the
judge-attested release record. Function B is the companion risk function
(refund-risk.sem.ts) over the same inputs, labeled from its own compiled
constraints on the real UCI pool; its attested set is the release inputs
relabeled by those constraints, so it is rule-labeled, not human or judge
adjudicated, and the report says so.

Regimes:
  b-alone   train B alone with the single-function trainer
  joint     train A and B jointly over one encoder and adapter
  a-then-b  train A alone on the shared architecture, then add B's head with
            the encoder, adapter and A's head frozen

For every regime the report carries each function's held-out (calibration
split) accuracy and its attested-set accuracy; the joint application is then
verified per function, bound into verified IRs, exported as one artifact with
two heads and exercised through the Node runtime for both functions.

Usage:
  python -m benchmarks.refund.program.run_application_experiment \\
    --corpus-manifest benchmarks/refund/data/pooled-v4-2026-09-24/manifest.json \\
    --heldout-dir benchmarks/refund/data/heldout-uci-2026-09-23 \\
    --pool benchmarks/refund/data/uci-pool/candidates.json \\
    --output-dir benchmarks/refund/data/release-application-2026-09-24
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from benchmarks.refund.program.pipeline import (
    compile_refund_program,
    derive_refund_artifact_training_key_sha256,
    derive_training_input_ledger,
    parse_release_verification_record,
    run_refund_runtime,
)
from benchmarks.refund.program.run_release_pipeline import (
    ReleasePipelineError,
    _dump,
    _git_commit,
    _load_tokenizer,
    _Logger,
    _package_version,
    _pinned_encoder_files,
    _teacher_from_manifest,
    _utc_now,
)

from semantscript_trainer import (
    AdversarialDatasetGenerator,
    AdversarialGenerationConfig,
    ArtifactFunction,
    ArtifactProvenance,
    GeneratedCase,
    TeacherDescriptor,
    TrainingConfig,
    TrainingProvenanceCounts,
    VerificationConfig,
    VerifiedIrProvenance,
    add_function_head,
    application_function,
    build_verified_ir,
    evaluate_training_result,
    export_multi_function_artifact,
    require_passing_verification,
    train_application,
    train_classifier,
)
from semantscript_trainer.canonical_input import serialize_canonical_inputs
from semantscript_trainer.constraints import ConstraintEvaluationBudget, compile_constraints
from semantscript_trainer.dataset import SyntheticDatasetGenerator
from semantscript_trainer.semantic_json import semantic_json_sha256
from semantscript_trainer.teacher import (
    BoundaryPairProposal,
    CounterfactualProposal,
    NeuralFunctionIr,
    TeacherResponseError,
)
from semantscript_trainer.training import DEFAULT_ENCODER_NAME, DEFAULT_ENCODER_REVISION
from semantscript_trainer.verification import tokenizer_json_bytes

RISK_SOURCE = "refund-risk.sem.ts"
RISK_SUPPORT = ("high", "low", "medium")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


# Adversarial protocol without a language model: single-field edits of real
# inputs across the refund schema's thresholds, labeled by the constraints.
_REFUND_SCHEMA_EDITS: Mapping[tuple[str, str], tuple[Any, ...]] = MappingProxyType(
    {
        ("order", "status"): ("paid", "fraudulent"),
        ("customer", "tier"): ("standard", "enterprise"),
        ("customer", "priorRefunds"): (0, 1, 2, 4, 5, 6),
        ("order", "ageDays"): (1, 30, 59, 60, 61, 90, 91, 120),
        ("order", "total"): (250, 499, 500, 750, 1499, 1500, 2000, 4999, 5000),
    }
)


class ConstraintLabelTeacher:
    """Label real pool inputs with the unique output a function's constraints admit.

    No language model is involved: the provider string says so, and the
    configuration digest binds the pool, the exclusions and the fraud flag.
    """

    def __init__(self, pool_path: Path, heldout_directory: Path, fraud_percent: int) -> None:
        self._pool_path = pool_path.resolve()
        self._pool_bytes = self._pool_path.read_bytes()
        self._pool = json.loads(self._pool_bytes)
        self._fraud_percent = fraud_percent
        digests: set[str] = set()
        for name in ("release-verification.json", "final-benchmark-dataset.json"):
            document = json.loads((heldout_directory / name).read_text(encoding="utf-8"))
            for case in document["cases"]:
                digests.add(case["inputSha256"])
        self._excluded = frozenset(digests)
        self._heldout = heldout_directory.resolve()
        self.report: dict[str, Any] = {}

    @property
    def configuration_projection(self) -> dict[str, Any]:
        return {
            "kind": "semantscript.refund-constraint-label-teacher-config",
            "configVersion": 1,
            "labeling": "compiled-constraints-unique-admissible-output",
            "pool": {
                "path": str(self._pool_path.relative_to(_REPOSITORY_ROOT)),
                "sha256": hashlib.sha256(self._pool_bytes).hexdigest(),
            },
            "exclusions": {
                "heldoutDirectory": str(self._heldout.relative_to(_REPOSITORY_ROOT)),
                "sha256": hashlib.sha256("\n".join(sorted(self._excluded)).encode()).hexdigest(),
                "count": len(self._excluded),
            },
            "fraudPercent": self._fraud_percent,
        }

    @property
    def descriptor(self) -> TeacherDescriptor:
        encoded = json.dumps(
            self.configuration_projection, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return TeacherDescriptor(
            provider="real-input-pool-rule-labels",
            model="compiled-constraints",
            configuration_sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def labeled_pool(self, ir: NeuralFunctionIr) -> tuple[GeneratedCase, ...]:
        constraints = compile_constraints(ir)
        support = list(ir["output"]["head"]["support"])
        cases: list[GeneratedCase] = []
        seen: set[str] = set()
        excluded = duplicates = ambiguous = 0
        counts: dict[str, int] = {}
        for row in sorted(self._pool["candidates"], key=lambda entry: entry["key"]):
            inputs = copy.deepcopy(row["inputs"])
            if int(row["key"][:8], 16) % 100 < self._fraud_percent:
                inputs["order"]["status"] = "fraudulent"
            digest = semantic_json_sha256(inputs)
            if digest in self._excluded:
                excluded += 1
                continue
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            admissible = [
                value
                for value in support
                if not constraints.evaluate_output_contract(inputs, value)[1]
            ]
            if len(admissible) != 1:
                ambiguous += 1
                continue
            counts[admissible[0]] = counts.get(admissible[0], 0) + 1
            cases.append(GeneratedCase(inputs=inputs, output=admissible[0]))
        self.report = {
            "poolRows": len(self._pool["candidates"]),
            "labeled": len(cases),
            "excludedHeldOut": excluded,
            "duplicateInputs": duplicates,
            "ambiguous": ambiguous,
            "labelCounts": counts,
        }
        return tuple(cases)

    def generate(self, ir: NeuralFunctionIr, n: int, /) -> tuple[GeneratedCase, ...]:
        cases = self.labeled_pool(ir)
        if n > len(cases):
            raise ReleasePipelineError(f"constraint teacher can supply {len(cases)} cases, not {n}")
        return cases[:n]

    def _edits(self, inputs: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        edits = []
        for (record, field), values in _REFUND_SCHEMA_EDITS.items():
            for value in values:
                if inputs[record][field] == value:
                    continue
                edited = copy.deepcopy(inputs)
                edited[record][field] = value
                edits.append((f"/{record}/{field}", edited))
        return edits

    def _unique_label(
        self, constraints: Any, support: Sequence[str], inputs: dict[str, Any]
    ) -> str | None:
        if semantic_json_sha256(inputs) in self._excluded:
            return None
        admissible = [v for v in support if not constraints.evaluate_output_contract(inputs, v)[1]]
        return admissible[0] if len(admissible) == 1 else None

    def generate_boundary_pair(
        self, ir: NeuralFunctionIr, constraint_index: int, /
    ) -> BoundaryPairProposal:
        constraints = compile_constraints(ir)
        support = list(ir["output"]["head"]["support"])
        budget = ConstraintEvaluationBudget()
        for row in sorted(self._pool["candidates"], key=lambda entry: entry["key"]):
            inputs = copy.deepcopy(row["inputs"])
            if int(row["key"][:8], 16) % 100 < self._fraud_percent:
                inputs["order"]["status"] = "fraudulent"
            base_label = self._unique_label(constraints, support, inputs)
            if base_label is None:
                continue
            base_result = constraints.evaluate(constraint_index, inputs, budget=budget)
            for _, edited in self._edits(inputs):
                if constraints.evaluate(constraint_index, edited, budget=budget) is base_result:
                    continue
                edited_label = self._unique_label(constraints, support, edited)
                if edited_label is None:
                    continue
                false_case, true_case = (
                    (GeneratedCase(inputs, base_label), GeneratedCase(edited, edited_label))
                    if base_result is False
                    else (GeneratedCase(edited, edited_label), GeneratedCase(inputs, base_label))
                )
                return BoundaryPairProposal(predicate_false=false_case, predicate_true=true_case)
        raise TeacherResponseError(
            f"no single-field edit of a real input flips constraint {constraint_index}"
        )

    def generate_counterfactual(
        self, ir: NeuralFunctionIr, anchor: GeneratedCase, /
    ) -> CounterfactualProposal:
        constraints = compile_constraints(ir)
        support = list(ir["output"]["head"]["support"])
        for path, edited in self._edits(copy.deepcopy(anchor.inputs)):
            label = self._unique_label(constraints, support, edited)
            if label is None or label == anchor.output:
                continue
            record, field = path.strip("/").split("/")
            return CounterfactualProposal(
                twin=GeneratedCase(edited, label),
                reason=(
                    f"changing {record}.{field} from {json.dumps(anchor.inputs[record][field])} to "
                    f"{json.dumps(edited[record][field])} crosses a rule threshold, so the risk moves from "
                    f"{anchor.output} to {label}"
                ),
            )
        raise TeacherResponseError(
            "no single-field edit of the anchor changes its constraint label"
        )


def rule_labeled_cases(
    ir: NeuralFunctionIr, inputs: Sequence[dict[str, Any]]
) -> tuple[GeneratedCase, ...]:
    constraints = compile_constraints(ir)
    support = list(ir["output"]["head"]["support"])
    cases = []
    for entry in inputs:
        admissible = [v for v in support if not constraints.evaluate_output_contract(entry, v)[1]]
        if len(admissible) != 1:
            raise ReleasePipelineError(
                "held-out input is ambiguous under the companion constraints"
            )
        cases.append(GeneratedCase(inputs=copy.deepcopy(entry), output=admissible[0]))
    return tuple(cases)


def case_accuracy(
    model: Any,
    tokenizer: Any,
    ir: NeuralFunctionIr,
    cases: Sequence[GeneratedCase],
    config: TrainingConfig,
) -> float:
    import torch

    support = list(ir["output"]["head"]["support"])
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    with torch.no_grad():
        for case in cases:
            text = serialize_canonical_inputs(
                ir["inputs"], case.inputs, version=config.canonical_input_version
            ).decode()
            encoded = tokenizer(
                [text],
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=config.maximum_sequence_length,
                return_tensors="pt",
            )
            logits = model(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
            )
            if support[int(logits[0].argmax())] == case.output:
                correct += 1
    return correct / len(cases)


def function_summary(
    result: Any,
    tokenizer: Any,
    ir: NeuralFunctionIr,
    attested: Sequence[GeneratedCase],
    config: TrainingConfig,
) -> dict[str, Any]:
    return {
        "heldOutAccuracy": result.held_out_accuracy,
        "heldOutRows": result.held_out_row_count,
        "trainingRows": result.training_row_count,
        "selectedEpoch": result.selected_epoch,
        "attestedAccuracy": case_accuracy(result.model, tokenizer, ir, attested, config),
        "attestedCases": len(attested),
        "epochs": [
            {
                "epoch": m.epoch,
                "meanTrainingLoss": m.mean_training_loss,
                "heldOutAccuracy": m.held_out_accuracy,
            }
            for m in result.metrics
        ],
    }


def provenance_for(
    base: Any,
    training: Any,
    verification: Any,
    ir: NeuralFunctionIr,
    config: TrainingConfig,
    trained_at: str,
) -> VerifiedIrProvenance:
    rows = training.split.training + training.split.evaluation
    origins = {
        name: sum(row.origin == name for row in rows)
        for name in ("gold", "synthetic", "constraint-boundary", "counterfactual")
    }
    weights_path, _ = _pinned_encoder_files(config)
    return VerifiedIrProvenance(
        teacher=base.teacher,
        base_model_name=config.encoder_name,
        base_model_revision=config.encoder_revision,
        base_model_weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        dataset_sha256=base.dataset_sha256,
        counts=TrainingProvenanceCounts(
            examples=len(ir["definition"]["examples"]),
            synthetic=origins["synthetic"],
            adversarial=origins["constraint-boundary"] + origins["counterfactual"],
            calibration=len(training.split.evaluation),
            verification=len(rows) + verification.attested_cases - origins["gold"],
            attested_verification=verification.attested_cases,
        ),
        seed=config.seed,
        trainer_version=_package_version(),
        trainer_commit=_git_commit(),
        trained_at=trained_at,
    )


def run_experiment(arguments: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    output = Path(arguments.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ReleasePipelineError("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    log = _Logger(output / "pipeline.log")
    heldout = Path(arguments.heldout_dir)
    config = TrainingConfig(
        encoder_name=DEFAULT_ENCODER_NAME,
        encoder_revision=DEFAULT_ENCODER_REVISION,
        local_files_only=True,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        maximum_sequence_length=arguments.max_sequence_length,
        evaluation_ratio=arguments.evaluation_ratio,
        seed=arguments.seed,
        device=arguments.device,
        select_best_epoch=True,
        learning_rate_schedule="linear",
        warmup_ratio=arguments.warmup_ratio,
        canonical_input_version=2,
    )
    verification_config = VerificationConfig(
        maximum_constraint_violation_rate=arguments.maximum_constraint_violation_rate
    )

    log("compiling the refund decision (A) and the companion risk function (B)")
    compiled_a = compile_refund_program(output / "compiler-a")
    compiled_b = compile_refund_program(
        output / "compiler-b", source_file=RISK_SOURCE, support=RISK_SUPPORT
    )
    ir_a, ir_b = compiled_a.source_ir, compiled_b.source_ir
    if (
        ir_a["model"]["encoder"] != ir_b["model"]["encoder"]
        or ir_a["model"]["adapter"] != ir_b["model"]["adapter"]
    ):
        raise ReleasePipelineError("both functions must compile under the same application refs")

    log("replaying the frozen refund corpus for A")
    manifest = json.loads(Path(arguments.corpus_manifest).read_text(encoding="utf-8"))
    corpus_root = Path(arguments.corpus_manifest).parent
    teacher_a = _teacher_from_manifest(manifest)
    base_a = SyntheticDatasetGenerator(teacher_a, corpus_root / "synthetic").generate(
        ir_a, manifest["synthetic"]["requestedCaseCount"]
    )
    if base_a.dataset_sha256 != manifest["synthetic"]["datasetSha256"]:
        raise ReleasePipelineError("replayed synthetic dataset digest differs from the manifest")
    adversarial_a = AdversarialDatasetGenerator(
        teacher_a,
        corpus_root / "adversarial",
        config=AdversarialGenerationConfig(
            counterfactual_ratio=manifest["adversarial"]["config"]["counterfactualRatio"],
            maximum_attempts=manifest["adversarial"]["config"]["maximumAttempts"],
        ),
    ).generate(ir_a, base_a)
    if adversarial_a.dataset_sha256 != manifest["adversarial"]["datasetSha256"]:
        raise ReleasePipelineError("replayed adversarial dataset digest differs from the manifest")
    release_a = parse_release_verification_record(
        (heldout / "release-verification.json").read_bytes(), ir_a
    )
    attested_a = release_a.generated_cases

    log("labeling the real pool for B from its compiled constraints")
    teacher_b = ConstraintLabelTeacher(Path(arguments.pool), heldout, arguments.fraud_percent)
    available = len(teacher_b.labeled_pool(ir_b))
    base_b = SyntheticDatasetGenerator(teacher_b, output / "b-synthetic").generate(ir_b, available)
    log(f"B corpus: {json.dumps(teacher_b.report)}")
    adversarial_b = AdversarialDatasetGenerator(
        teacher_b,
        output / "b-adversarial",
        config=AdversarialGenerationConfig(
            counterfactual_ratio=arguments.counterfactual_ratio, maximum_attempts=3
        ),
    ).generate(ir_b, base_b)
    log(
        f"B adversarial sidecar: {len(adversarial_b.cases)} cases, {len(adversarial_b.pairs)} pairs, "
        "single-field edits of real inputs labeled by the constraints (no language model)"
    )
    attested_b = rule_labeled_cases(ir_b, [case.inputs for case in attested_a])
    log(
        f"B attested set: {len(attested_b)} release inputs relabeled by B's constraints (rule-labeled, not judge-adjudicated)"
    )

    tokenizer = _load_tokenizer(config)
    function_a = application_function(ir_a, base_a, adversarial_a)
    function_b = application_function(ir_b, base_b, adversarial_b)
    report: dict[str, Any] = {
        "kind": "semantscript.refund-shared-encoder-experiment",
        "startedAt": _utc_now(),
        "functions": {
            "A": {
                "id": compiled_a.function_id,
                "semanticSha256": compiled_a.semantic_sha256,
                "support": ["approve", "deny", "review"],
                "attested": "judge-adjudicated release record",
            },
            "B": {
                "id": compiled_b.function_id,
                "semanticSha256": compiled_b.semantic_sha256,
                "support": list(RISK_SUPPORT),
                "attested": "release inputs relabeled by B's constraints (rule-labeled)",
                "corpus": teacher_b.report,
                "adversarial": {
                    "cases": len(adversarial_b.cases),
                    "pairs": len(adversarial_b.pairs),
                    "datasetSha256": adversarial_b.dataset_sha256,
                },
                "teacher": teacher_b.descriptor.configuration_sha256,
            },
        },
        "config": {
            "epochs": config.epochs,
            "batchSize": config.batch_size,
            "learningRate": config.learning_rate,
            "evaluationRatio": config.evaluation_ratio,
            "seed": config.seed,
            "warmupRatio": config.warmup_ratio,
            "canonicalInputVersion": 2,
            "adapterBottleneckSize": arguments.adapter_bottleneck_size,
        },
        "regimes": {},
    }

    log("regime b-alone: single-function trainer on B")
    t0 = time.monotonic()
    alone_b = train_classifier(ir_b, base_b, adversarial_b, config=config, tokenizer=tokenizer)
    report["regimes"]["b-alone"] = {
        "seconds": round(time.monotonic() - t0, 1),
        "B": function_summary(alone_b, tokenizer, ir_b, attested_b, config),
    }
    log(
        f"b-alone: held-out {alone_b.held_out_accuracy:.4f}, attested {report['regimes']['b-alone']['B']['attestedAccuracy']:.4f}"
    )
    del alone_b

    (output / "experiment-report.json").write_text(_dump(report), encoding="utf-8")
    log("regime joint: shared encoder and adapter, both heads")
    t0 = time.monotonic()
    joint = train_application(
        [function_a, function_b],
        config=config,
        tokenizer=tokenizer,
        adapter_bottleneck_size=arguments.adapter_bottleneck_size,
    )
    trained_at = _utc_now()
    report["regimes"]["joint"] = {
        "seconds": round(time.monotonic() - t0, 1),
        "selectedEpoch": joint.selected_epoch,
        "A": function_summary(
            joint.functions[compiled_a.function_id], tokenizer, ir_a, attested_a, config
        ),
        "B": function_summary(
            joint.functions[compiled_b.function_id], tokenizer, ir_b, attested_b, config
        ),
    }
    log(
        f"joint: A held-out {joint.functions[compiled_a.function_id].held_out_accuracy:.4f} attested {report['regimes']['joint']['A']['attestedAccuracy']:.4f}; B held-out {joint.functions[compiled_b.function_id].held_out_accuracy:.4f} attested {report['regimes']['joint']['B']['attestedAccuracy']:.4f}"
    )

    (output / "experiment-report.json").write_text(_dump(report), encoding="utf-8")
    log("regime a-then-b: A alone on the shared architecture, then B's head on the frozen encoder")
    t0 = time.monotonic()
    first = train_application(
        [function_a],
        config=config,
        tokenizer=tokenizer,
        adapter_bottleneck_size=arguments.adapter_bottleneck_size,
    )
    from semantscript_trainer.verification import model_state_sha256

    digest_a_before = model_state_sha256(first.functions[compiled_a.function_id].model)
    a_summary = function_summary(
        first.functions[compiled_a.function_id], tokenizer, ir_a, attested_a, config
    )
    extended = add_function_head(first, function_b, tokenizer=tokenizer)
    digest_a_after = model_state_sha256(extended.functions[compiled_a.function_id].model)
    report["regimes"]["a-then-b"] = {
        "seconds": round(time.monotonic() - t0, 1),
        "A": a_summary,
        "B": function_summary(
            extended.functions[compiled_b.function_id], tokenizer, ir_b, attested_b, config
        ),
        "aStateUnchangedAfterAddingB": digest_a_before == digest_a_after,
    }
    log(
        f"a-then-b: A held-out {a_summary['heldOutAccuracy']:.4f} attested {a_summary['attestedAccuracy']:.4f} (state unchanged: {digest_a_before == digest_a_after}); B held-out {extended.functions[compiled_b.function_id].held_out_accuracy:.4f} attested {report['regimes']['a-then-b']['B']['attestedAccuracy']:.4f}"
    )
    del first, extended

    (output / "experiment-report.json").write_text(_dump(report), encoding="utf-8")
    log("verifying both functions of the joint application")
    verifications = {}
    for label, ir, base, adversarial, attested in (
        ("A", ir_a, base_a, adversarial_a, attested_a),
        ("B", ir_b, base_b, None, attested_b),
    ):
        function_id = ir["id"]
        verification = evaluate_training_result(
            ir,
            joint.functions[function_id],
            base,
            adversarial,
            tokenizer=tokenizer,
            attested_verification=attested,
            config=verification_config,
        )
        report["regimes"]["joint"][label]["verification"] = {
            "status": verification.status,
            "failures": list(verification.failures),
            "metrics": verification.to_ir_document()["metrics"],
            "attestedCases": verification.attested_cases,
        }
        log(
            f"{label} verification {verification.status}: accuracy {verification.metrics.accuracy:.4f}, ece {verification.metrics.ece:.4f}, violations {verification.metrics.constraint_violations}"
        )
        verifications[label] = verification
    for verification in verifications.values():
        require_passing_verification(verification)

    log("binding verified IRs and exporting one artifact with two heads")
    built = {}
    for label, ir, base in (("A", ir_a, base_a), ("B", ir_b, base_b)):
        training = joint.functions[ir["id"]]
        built[label] = build_verified_ir(
            ir,
            training,
            verifications[label],
            provenance_for(base, training, verifications[label], ir, config, trained_at),
        )
        (output / f"verified-ir-{label.lower()}.json").write_bytes(built[label].source_ir_bytes)
    ledger = derive_training_input_ledger(
        ir_a,
        base_a,
        joint.functions[compiled_a.function_id],
        release_a,
        created_at=_utc_now(),
        adversarial_dataset=adversarial_a,
    )
    (output / "training-input-ledger-a.json").write_text(_dump(ledger), encoding="utf-8")
    training_key = derive_refund_artifact_training_key_sha256(ledger["sources"])
    joint.model.to("cpu")
    sample_inputs = dict(base_a.cases[0].inputs)
    parity_text = serialize_canonical_inputs(ir_a["inputs"], sample_inputs, version=2).decode(
        "utf-8"
    )
    encoded = tokenizer(
        [parity_text],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=config.maximum_sequence_length,
        return_tensors="pt",
    )
    exported = export_multi_function_artifact(
        output / "artifact",
        [
            ArtifactFunction(
                built["A"].document,
                joint.functions[compiled_a.function_id],
                verifications["A"],
                built["A"].source_ir_bytes,
            ),
            ArtifactFunction(
                built["B"].document,
                joint.functions[compiled_b.function_id],
                verifications["B"],
                built["B"].source_ir_bytes,
            ),
        ],
        tokenizer_json=tokenizer_json_bytes(tokenizer),
        provenance=ArtifactProvenance(
            application_id="refund-benchmark",
            application_version="1.0.0",
            compiler_version=_package_version(),
            trainer_version=_package_version(),
            created_at=_utc_now(),
            training_key_sha256=training_key,
        ),
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )
    log(
        f"exported artifact {exported.manifest_sha256} with {len(exported.manifest['functions'])} functions and {len(exported.manifest['resources'])} resources"
    )

    log("running one diagnostic call per function through the Node runtime")
    diagnostics = {
        "A": run_refund_runtime(exported.artifact_root, compiled_a.function_id, sample_inputs),
        "B": run_refund_runtime(
            exported.artifact_root, compiled_b.function_id, sample_inputs, support=RISK_SUPPORT
        ),
    }
    for label, diagnostic in diagnostics.items():
        log(f"runtime {label}: {diagnostic['value']} (confidence {diagnostic['confidence']:.4f})")
    report["artifact"] = {
        "root": str(exported.artifact_root),
        "releaseDirectory": str(exported.release_directory),
        "manifestSha256": exported.manifest_sha256,
        "functions": [fn["id"] for fn in exported.manifest["functions"]],
        "resources": [r["role"] for r in exported.manifest["resources"]],
        "trainingKeySha256": training_key,
    }
    report["runtimeDiagnostics"] = diagnostics
    report["completedAt"] = _utc_now()
    report["elapsedSeconds"] = round(time.monotonic() - started, 1)
    (output / "experiment-report.json").write_text(_dump(report), encoding="utf-8")
    log(f"done in {report['elapsedSeconds']}s")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument("--heldout-dir", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fraud-percent", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--evaluation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--maximum-constraint-violation-rate", type=float, default=0.01)
    parser.add_argument("--adapter-bottleneck-size", type=int, default=64)
    parser.add_argument("--counterfactual-ratio", type=float, default=0.02)
    arguments = parser.parse_args(argv)
    try:
        run_experiment(arguments)
    except Exception as error:
        print(f"experiment failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
