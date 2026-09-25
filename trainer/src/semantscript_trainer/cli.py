"""Bundle-to-artifact driver behind ``semantscript train``.

``train_bundle`` takes one compiler IR bundle and turns every source-stage
function in it into a verified head of one content-addressed artifact: it
generates the synthetic dataset (and the adversarial sidecar when the function
has constraints), trains one classifier for a single function or the
shared-encoder application for several, verifies every function, binds verified
IR with derived provenance counts and exports the artifact. ``python -m
semantscript_trainer.cli train`` wires the driver to a teacher TOML, a cache
directory and the training, verification and adversarial settings, and writes a
per-function JSON report that the Node CLI renders.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from semantscript_trainer.adversarial import (
    AdversarialDataset,
    AdversarialDatasetGenerator,
    AdversarialGenerationConfig,
    AdversarialGenerationError,
)
from semantscript_trainer.application import (
    DEFAULT_ADAPTER_BOTTLENECK_SIZE,
    ApplicationTrainingResult,
    FunctionCorpus,
    add_function_head,
    application_function,
    domain_depths,
    train_application,
)
from semantscript_trainer.artifact import (
    ArtifactFunction,
    ArtifactProvenance,
    ExportedArtifact,
    export_multi_function_artifact,
)
from semantscript_trainer.build_cache import (
    ApplicationCache,
    BuildCacheError,
    CachedFunction,
    cache_recipe,
    combined_shared_sha256,
)
from semantscript_trainer.canonical_input import serialize_canonical_inputs
from semantscript_trainer.constraints import ConstraintError
from semantscript_trainer.dataset import (
    DatasetCacheError,
    DatasetError,
    SyntheticDatasetGenerator,
    TrainingDataset,
)
from semantscript_trainer.doctor import add_arguments as add_doctor_arguments
from semantscript_trainer.doctor import run_from_arguments as run_doctor_from_arguments
from semantscript_trainer.lifecycle import (
    TrainingProvenanceCounts,
    VerifiedIrProvenance,
    build_verified_ir,
)
from semantscript_trainer.semantic_json import semantic_json_sha256
from semantscript_trainer.teacher import (
    AdversarialTeacher,
    JsonValue,
    NeuralFunctionIr,
    Teacher,
    TeacherBudgetExceeded,
    TeacherDescriptor,
    TeacherResponseError,
    TeacherTransportError,
)
from semantscript_trainer.teacher_config import create_teacher, load_teacher_config
from semantscript_trainer.teacher_spend import (
    ResponseJournal,
    SpendMeter,
    TeacherPriceUnknown,
    language_model_config,
    resolve_price,
)
from semantscript_trainer.training import (
    TrainingConfig,
    TrainingResult,
    _load_tokenizer,
)
from semantscript_trainer.verification import (
    VerificationConfig,
    VerificationResult,
    evaluate_training_result,
    model_state_sha256,
    tokenizer_json_bytes,
)

BUNDLE_KIND = "semantscript.ir-bundle"
REPORT_KIND = "semantscript.train-report"
REPORT_VERSION = 1
DEFAULT_CASES = 64
TRAINER_VERSION = "0.0.0"
_TRAINING_KEY_KIND = "semantscript.training-key"


class TrainBundleError(RuntimeError):
    """The bundle, the teacher or the configuration cannot produce an artifact."""


class TrainBundleFailure(TrainBundleError):
    """Verification refused at least one function; the report records why."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True, slots=True)
class TrainedFunction:
    """One bundle function after generation, training and verification."""

    ir: NeuralFunctionIr
    base: TrainingDataset
    adversarial: AdversarialDataset | None
    training: TrainingResult
    verification: VerificationResult
    reused: bool = False


@dataclass(frozen=True, slots=True)
class TrainBundleResult:
    """The per-function report and the published artifact."""

    report: dict[str, Any]
    exported: ExportedArtifact
    functions: tuple[TrainedFunction, ...]


def train_bundle(
    bundle: Mapping[str, Any],
    artifact_root: str | Path,
    *,
    teacher: Teacher,
    cache_directory: str | Path,
    cases: int = DEFAULT_CASES,
    training_config: TrainingConfig | None = None,
    verification_config: VerificationConfig | None = None,
    adversarial_config: AdversarialGenerationConfig | None = None,
    application_id: str | None = None,
    application_version: str = "0.0.0",
    compiler_version: str = "0.0.0",
    trainer_version: str = TRAINER_VERSION,
    trainer_commit: str | None = None,
    tokenizer: Any | None = None,
    encoder: Any | None = None,
    base_model_weights_sha256: str | None = None,
    adapter_bottleneck_size: int = DEFAULT_ADAPTER_BOTTLENECK_SIZE,
    use_cache: bool = True,
    full: bool = False,
    log: Callable[[str], None] | None = None,
    meter: SpendMeter | None = None,
) -> TrainBundleResult:
    """Train, verify and export every function of ``bundle`` into ``artifact_root``.

    Every function trains over one shared encoder and adapter. With the build
    cache on, functions whose id, semantic digest and recipe match a cached
    record are reused: a bundle with no changes performs no training (and no
    export when its release is still published), and a changed function trains
    only its own head on the frozen shared modules. ``full`` retrains everything
    jointly; ``use_cache=False`` ignores and does not write the cache.
    ``tokenizer``, ``encoder`` and ``base_model_weights_sha256`` default to the
    pinned Hugging Face encoder of ``training_config``; tests inject fakes.
    ``meter`` (the spend meter the teacher charges) adds a running cost line after
    each expression's datasets and the ``teacher.spend`` object to the report.
    """

    say = log if log is not None else (lambda _message: None)
    functions = _bundle_functions(bundle)
    _require_gold_examples(functions)
    if isinstance(cases, bool) or not isinstance(cases, int) or cases < 0:
        raise TrainBundleError("cases must be a non-negative integer")
    resolved_training = training_config if training_config is not None else TrainingConfig()
    resolved_verification = (
        verification_config if verification_config is not None else VerificationConfig()
    )
    resolved_adversarial = (
        adversarial_config if adversarial_config is not None else AdversarialGenerationConfig()
    )
    descriptor = getattr(teacher, "descriptor", None)
    if not isinstance(descriptor, TeacherDescriptor):
        raise TrainBundleError("teacher must expose a TeacherDescriptor")
    resolved_application_id = (
        application_id if application_id is not None else _application_id(functions[0])
    )
    cache_root = Path(cache_directory)
    recipe = cache_recipe(
        resolved_training,
        resolved_verification,
        resolved_adversarial,
        adapter_bottleneck_size,
        descriptor,
    )
    cache = ApplicationCache(cache_root, resolved_application_id) if use_cache else None

    reused: dict[str, CachedFunction] = {}
    if cache is not None and not full:
        intact = cache.shared_state_intact()
        for ir in functions:
            function_id = cast(str, ir["id"])
            model = cast(dict[str, Any], ir["model"])
            record = cache.cached_function(function_id)
            if (
                record is not None
                and intact
                and record.semantic_sha256 == ir["semanticSha256"]
                and record.recipe_sha256 == recipe.sha256
                and record.model_sha256 == semantic_json_sha256(cast(Any, dict(model)))
                and record.shared_state_sha256
                == cache.function_shared_sha256(cast(str, model["adapter"]))
            ):
                reused[function_id] = record
    to_train = [ir for ir in functions if cast(str, ir["id"]) not in reused]
    if cache is not None:
        say(f"build cache: {len(reused)} function(s) reused, {len(to_train)} to train")

    def report_skeleton(status: str, trained_at: str) -> dict[str, Any]:
        return {
            "kind": REPORT_KIND,
            "reportVersion": REPORT_VERSION,
            "status": status,
            "trainedAt": trained_at,
            "application": {"id": resolved_application_id, "version": application_version},
            "teacher": {
                "provider": descriptor.provider,
                "model": descriptor.model,
                "configurationSha256": descriptor.configuration_sha256,
                **({} if meter is None else {"spend": meter.summary()}),
            },
            "cache": {
                "reused": len(reused),
                "trained": len(to_train),
                "directory": None if cache is None else str(cache.root),
            },
            "trainingKeySha256": None,
            "artifact": None,
            "functions": [],
        }

    if cache is not None and not to_train:
        existing = _published_release(Path(artifact_root), cache.release_sha256())
        if existing is not None:
            say(f"build cache: nothing changed; release {existing.manifest_sha256} stands")
            report = report_skeleton("reused", _utc_now())
            report["trainingKeySha256"] = _training_key_from_records(
                [reused[cast(str, ir["id"])] for ir in functions]
            )
            report["functions"] = [
                _cached_function_report(ir, reused[cast(str, ir["id"])]) for ir in functions
            ]
            report["artifact"] = {
                "root": str(existing.artifact_root),
                "releaseDirectory": str(existing.release_directory),
                "manifestSha256": existing.manifest_sha256,
            }
            return TrainBundleResult(report=report, exported=existing, functions=())

    resolved_tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(resolved_training)

    datasets: list[tuple[NeuralFunctionIr, TrainingDataset, AdversarialDataset | None]] = []
    for ir in functions:
        function_id = cast(str, ir["id"])
        definition = cast(dict[str, Any], ir["definition"])
        gold = len(cast(list[Any], definition.get("examples", [])))
        total = max(cases, gold)
        if function_id not in reused:
            say(f"{function_id}: generating {total} cases ({gold} gold)")
        base = SyntheticDatasetGenerator(teacher, cache_root).generate(ir, total)
        distinct = len({json.dumps(case.inputs, sort_keys=True) for case in base.cases})
        if function_id not in reused and distinct * 2 < len(base.cases):
            from semantscript_trainer.teachers.constraints import describe_expression

            say(
                f"warning: {describe_expression(ir)}: only {distinct} distinct inputs among "
                f"{len(base.cases)} cases; the teacher repeated inputs because the input space "
                "is small, so held-out accuracy says less than the count suggests (a "
                "[teacher.ranges] table widens a constraints teacher's number ranges; "
                "docs/teachers.md)"
            )
        adversarial: AdversarialDataset | None = None
        if cast(list[Any], definition.get("constraints", [])):
            if not isinstance(teacher, AdversarialTeacher):
                raise TrainBundleError(
                    f"{function_id} declares constraints; the teacher must support "
                    "boundary and counterfactual generation"
                )
            if function_id not in reused:
                say(f"{function_id}: generating adversarial cases around its constraints")
            adversarial = AdversarialDatasetGenerator(
                teacher, cache_root, config=resolved_adversarial
            ).generate(ir, base)
        if meter is not None and function_id not in reused:
            say(f"{function_id}: {meter.line()}")
        record = reused.get(function_id)
        if record is not None and (
            record.dataset_sha256 != base.dataset_sha256
            or record.adversarial_dataset_sha256
            != (None if adversarial is None else adversarial.dataset_sha256)
        ):
            say(f"{function_id}: cached datasets changed; training it again")
            del reused[function_id]
            to_train.append(ir)
        datasets.append((ir, base, adversarial))
    by_id = {cast(str, ir["id"]): (ir, base, adversarial) for ir, base, adversarial in datasets}

    trained_at = _utc_now()
    shared_changed = not reused
    restored_adapter_refs: tuple[str, ...] = ()
    if reused:
        say(f"build cache: restoring the shared encoder and {len(reused)} head(s)")
        application = _rehydrate_application(
            cast(ApplicationCache, cache),
            [
                (reused[function_id], application_function(*by_id[function_id]))
                for function_id in reused
            ],
            resolved_training,
            encoder,
            adapter_bottleneck_size,
        )
        restored_adapter_refs = application.model.adapter_refs
        for ir in to_train:
            function_id = cast(str, ir["id"])
            adapter_ref = cast(str, cast(dict[str, Any], ir["model"])["adapter"])
            say(
                f"{function_id}: training its head on the frozen shared encoder"
                + (
                    ""
                    if adapter_ref in application.model.adapter_refs
                    else f" with a new adapter for domain {adapter_ref}"
                )
            )
            application = add_function_head(
                application,
                application_function(*by_id[function_id]),
                config=resolved_training,
                tokenizer=resolved_tokenizer,
            )
    else:
        domains = domain_depths(
            [application_function(ir, base, adversarial) for ir, base, adversarial in datasets]
        )
        say(
            f"training {len(datasets)} function(s) over one shared encoder and "
            f"{len(domains)} adapter(s)"
            + (
                ""
                if all(depth is None for depth in domains.values())
                else " with depth routing "
                + ", ".join(
                    f"{ref}@{'full' if depth is None else depth}" for ref, depth in domains.items()
                )
            )
        )
        application = train_application(
            [application_function(ir, base, adversarial) for ir, base, adversarial in datasets],
            config=resolved_training,
            tokenizer=resolved_tokenizer,
            encoder=encoder,
            adapter_bottleneck_size=adapter_bottleneck_size,
        )
    trainings = dict(application.functions)
    shared_model = application.model

    trained: list[TrainedFunction] = []
    for ir, base, adversarial in datasets:
        function_id = cast(str, ir["id"])
        training = trainings[function_id]
        record = reused.get(function_id)
        if record is not None:
            trained.append(
                TrainedFunction(ir, base, adversarial, training, record.verification, reused=True)
            )
            continue
        verification = evaluate_training_result(
            ir,
            training,
            base,
            adversarial,
            tokenizer=resolved_tokenizer,
            config=resolved_verification,
            verified_at=trained_at,
        )
        say(
            f"{function_id}: verification {verification.status}, accuracy "
            f"{verification.metrics.accuracy:.4f}, ece {verification.metrics.ece:.4f}, "
            f"held-out accuracy {training.held_out_accuracy:.4f}"
        )
        trained.append(TrainedFunction(ir, base, adversarial, training, verification))

    training_key = _training_key_sha256(trained)
    report = report_skeleton("passed", trained_at)
    report["trainingKeySha256"] = training_key
    report["functions"] = [_function_report(entry) for entry in trained]
    failed = [entry for entry in trained if entry.verification.status != "passed"]
    if failed:
        report["status"] = "failed"
        names = ", ".join(cast(str, entry.ir["id"]) for entry in failed)
        raise TrainBundleFailure(f"verification failed for {names}", report)

    weights_sha256 = (
        base_model_weights_sha256
        if base_model_weights_sha256 is not None
        else _pinned_weights_sha256(resolved_training)
    )
    commit = trainer_commit if trainer_commit is not None else _git_commit()
    artifact_functions: list[ArtifactFunction] = []
    verified_bytes: dict[str, bytes] = {}
    for entry in trained:
        function_id = cast(str, entry.ir["id"])
        record = reused.get(function_id)
        if record is not None:
            source_ir_bytes = record.verified_ir_bytes
            document = cast(NeuralFunctionIr, json.loads(source_ir_bytes))
        else:
            provenance = VerifiedIrProvenance(
                teacher=entry.base.teacher,
                base_model_name=resolved_training.encoder_name,
                base_model_revision=resolved_training.encoder_revision,
                base_model_weights_sha256=weights_sha256,
                dataset_sha256=entry.base.dataset_sha256,
                counts=_provenance_counts(entry),
                seed=resolved_training.seed,
                trainer_version=trainer_version,
                trainer_commit=commit,
                trained_at=trained_at,
            )
            built = build_verified_ir(entry.ir, entry.training, entry.verification, provenance)
            source_ir_bytes = built.source_ir_bytes
            document = built.document
        verified_bytes[function_id] = source_ir_bytes
        artifact_functions.append(
            ArtifactFunction(document, entry.training, entry.verification, source_ir_bytes)
        )

    first = trained[0]
    sample_inputs = dict(first.base.cases[0].inputs)
    parity_text = serialize_canonical_inputs(
        cast(list[Any], first.ir["inputs"]),
        sample_inputs,
        version=resolved_training.canonical_input_version,
    ).decode("utf-8")
    encoded = resolved_tokenizer(
        [parity_text],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=resolved_training.maximum_sequence_length,
        return_tensors="pt",
    )
    to_device = getattr(shared_model, "to", None)
    if callable(to_device):
        to_device("cpu")
    say(f"exporting {len(artifact_functions)} function(s) to {artifact_root}")
    exported = export_multi_function_artifact(
        artifact_root,
        artifact_functions,
        tokenizer_json=tokenizer_json_bytes(resolved_tokenizer),
        provenance=ArtifactProvenance(
            application_id=resolved_application_id,
            application_version=application_version,
            compiler_version=compiler_version,
            trainer_version=trainer_version,
            created_at=_utc_now(),
            training_key_sha256=training_key,
        ),
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )
    report["artifact"] = {
        "root": str(exported.artifact_root),
        "releaseDirectory": str(exported.release_directory),
        "manifestSha256": exported.manifest_sha256,
    }
    say(f"published release {exported.manifest_sha256}")

    if cache is not None:
        model_adapters = {ref: shared_model.adapter_for(ref) for ref in shared_model.adapter_refs}
        if shared_changed:
            # Records built on the previous shared weights can never be reused again.
            shutil.rmtree(cache.root / "functions", ignore_errors=True)
            digests = cache.store_shared_state(shared_model.encoder, model_adapters)
            encoder_sha256 = digests.encoder_sha256
            adapter_sha256 = dict(digests.adapter_sha256)
        else:
            encoder_sha256 = cast(str, cache.shared_state_sha256())
            index = cache.index() or {}
            adapter_sha256 = dict(cast(dict[str, str], index.get("adapterStateSha256", {})))
            for ref, module in model_adapters.items():
                if ref not in restored_adapter_refs:
                    # A domain whose adapter was (re)trained in this build on the frozen
                    # encoder: store its new weights; the restored adapters are untouched.
                    adapter_sha256[ref] = cache.store_adapter_state(ref, module)
        for entry in trained:
            if entry.reused:
                continue
            cache.store_function(
                ir=entry.ir,
                recipe_sha256=recipe.sha256,
                shared_state_sha256=combined_shared_sha256(
                    encoder_sha256,
                    adapter_sha256[cast(str, cast(dict[str, Any], entry.ir["model"])["adapter"])],
                ),
                training=entry.training,
                verification=entry.verification,
                verified_ir_bytes=verified_bytes[cast(str, entry.ir["id"])],
                dataset_cases=len(entry.base.cases),
                dataset_gold=entry.base.gold_count,
                adversarial_cases=0 if entry.adversarial is None else len(entry.adversarial.cases),
            )
        cache.store_index(
            recipe=recipe,
            function_ids=[cast(str, entry.ir["id"]) for entry in trained],
            release_sha256=exported.manifest_sha256,
            shared_state_sha256=encoder_sha256,
            tokenizer_sha256=trained[0].verification.tokenizer_sha256,
            adapter_state_sha256=adapter_sha256,
        )
        say(f"build cache: stored under {cache.root}")
    return TrainBundleResult(report=report, exported=exported, functions=tuple(trained))


def _published_release(artifact_root: Path, manifest_sha256: str | None) -> ExportedArtifact | None:
    """The artifact root's current release when it is the one the cache last published."""

    if manifest_sha256 is None:
        return None
    pointer_path = artifact_root / "current.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if not isinstance(pointer, dict) or pointer.get("manifestSha256") != manifest_sha256:
            return None
        release = artifact_root / str(pointer["release"])
        manifest_path = release / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
            return None
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return ExportedArtifact(
        artifact_root=artifact_root,
        release_directory=release,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        pointer_path=pointer_path,
        manifest=manifest,
    )


def _rehydrate_application(
    cache: ApplicationCache,
    entries: Sequence[tuple[CachedFunction, FunctionCorpus]],
    config: TrainingConfig,
    encoder: Any | None,
    adapter_bottleneck_size: int,
) -> ApplicationTrainingResult:
    """Rebuild the shared application and every reused function from exact cached state."""

    from semantscript_trainer.application import (
        _build_adapter,
        _function_states,
        _load_application_modules,
        domain_depths,
    )
    from semantscript_trainer.training import _build_heads, _build_sentence_encoder

    application_module, _ = _load_application_modules()
    sentence_encoder = _build_sentence_encoder(config, encoder)
    encoder_state, adapter_states = cache.load_shared_state()
    # Only adapters some reused function still reads are restored; a domain whose
    # every function changed trains a fresh adapter on the frozen encoder.
    depths = domain_depths([function for _, function in entries])
    heads: dict[str, Any] = {}
    adapters: dict[str, Any] = {}
    try:
        sentence_encoder.load_state_dict(encoder_state)
        for ref in depths:
            if ref not in adapter_states:
                raise KeyError(f"cached adapter {ref!r} is missing")
            adapter = _build_adapter(
                application_module, sentence_encoder.hidden_size, adapter_bottleneck_size
            )
            adapter.load_state_dict(adapter_states[ref])
            adapters[ref] = adapter
        for record, function in entries:
            head = _build_heads(function.corpus.output_heads, sentence_encoder.hidden_size, config)
            head.load_state_dict(cache.load_head_state(record))
            heads[function.function_id] = head
        model = application_module.SharedEncoderApplication(
            sentence_encoder,
            adapters=adapters,
            adapter_depths=depths,
            heads=heads,
            function_adapters={
                function.function_id: function.adapter_ref for _, function in entries
            },
        )
    except (RuntimeError, TypeError, ValueError, KeyError) as error:
        raise BuildCacheError(
            f"cached weights do not fit the configured modules: {error}"
        ) from error
    states = _function_states([function for _, function in entries], config)
    results: dict[str, TrainingResult] = {}
    for (record, function), state in zip(entries, states, strict=True):
        training = TrainingResult(
            model=model.function_model(function.function_id),
            head=function.corpus.head,
            split=state.split,
            config=config,
            device=record.device,
            metrics=record.metrics,
            function_id=function.function_id,
            semantic_sha256=function.corpus.semantic_sha256,
            base_dataset_sha256=function.corpus.base_dataset_sha256,
            adversarial_dataset_sha256=function.corpus.adversarial_dataset_sha256,
            selected_epoch=record.selected_epoch,
            output_heads=function.corpus.output_heads,
        )
        if model_state_sha256(training.model) != record.model_state_sha256:
            raise BuildCacheError(
                f"restored weights for {function.function_id} do not match their verified state; "
                "rebuild with the cache off"
            )
        results[function.function_id] = training
    return ApplicationTrainingResult(
        model=model,
        config=config,
        device=entries[0][0].device,
        adapter_bottleneck_size=adapter_bottleneck_size,
        functions=results,
        metrics=(),
        selected_epoch=None,
    )


def _cached_function_report(ir: NeuralFunctionIr, record: CachedFunction) -> dict[str, Any]:
    epoch_index = (
        record.selected_epoch - 1 if record.selected_epoch is not None else len(record.metrics) - 1
    )
    last = record.metrics[epoch_index] if record.metrics else None
    verification = record.verification
    return {
        "id": ir["id"],
        "semanticSha256": ir["semanticSha256"],
        "sourcePath": record.source_path,
        "cache": "reused",
        "dataset": {
            "sha256": record.dataset_sha256,
            "cases": record.dataset_cases,
            "gold": record.dataset_gold,
        },
        "adversarial": (
            None
            if record.adversarial_dataset_sha256 is None
            else {"sha256": record.adversarial_dataset_sha256, "cases": record.adversarial_cases}
        ),
        "training": {
            "trainingRows": None,
            "heldOutRows": None,
            "epochs": len(record.metrics),
            "selectedEpoch": record.selected_epoch,
            "heldOutAccuracy": None if last is None else last.held_out_accuracy,
            "heldOutFieldAccuracy": (
                None
                if last is None or last.held_out_field_accuracy is None
                else list(last.held_out_field_accuracy)
            ),
        },
        "verification": {
            "status": verification.status,
            "failures": list(verification.failures),
            "attestedCases": verification.attested_cases,
            "pairCount": verification.pair_count,
            "metrics": verification.to_ir_document()["metrics"],
        },
    }


def _training_key_from_records(records: Sequence[CachedFunction]) -> str:
    projection: JsonValue = {
        "kind": _TRAINING_KEY_KIND,
        "keyVersion": 1,
        "functions": [
            {
                "id": record.function_id,
                "semanticSha256": record.semantic_sha256,
                "datasetSha256": record.dataset_sha256,
                "adversarialDatasetSha256": record.adversarial_dataset_sha256,
            }
            for record in sorted(records, key=lambda record: record.function_id)
        ],
    }
    return semantic_json_sha256(projection)


def _bundle_functions(bundle: Mapping[str, Any]) -> list[NeuralFunctionIr]:
    if not isinstance(bundle, Mapping):
        raise TrainBundleError("bundle must be a JSON object")
    if bundle.get("kind") != BUNDLE_KIND or bundle.get("bundleVersion") != 1:
        raise TrainBundleError(f"bundle must be a {BUNDLE_KIND} version 1 document")
    functions = bundle.get("functions")
    if not isinstance(functions, list) or not functions:
        raise TrainBundleError("bundle must contain at least one neural function")
    resolved: list[NeuralFunctionIr] = []
    seen: set[str] = set()
    for raw in functions:
        if not isinstance(raw, Mapping):
            raise TrainBundleError("bundle functions must be objects")
        function_id = raw.get("id")
        if not isinstance(function_id, str) or function_id in seen:
            raise TrainBundleError("bundle functions must have unique string ids")
        seen.add(function_id)
        if raw.get("stage") != "source":
            raise TrainBundleError(f"{function_id} is not a source-stage record")
        model = raw.get("model")
        if not isinstance(model, Mapping):
            raise TrainBundleError(f"{function_id} has no model binding")
        encoder_ref = model.get("encoder")
        adapter_ref = model.get("adapter")
        if not isinstance(encoder_ref, str) or not isinstance(adapter_ref, str):
            raise TrainBundleError(f"{function_id} must bind encoder and adapter refs")
        definition = raw.get("definition")
        if not isinstance(definition, Mapping):
            raise TrainBundleError(f"{function_id} has no definition")
        resolved.append(cast(NeuralFunctionIr, copy.deepcopy(dict(raw))))
    return resolved


def _require_gold_examples(functions: Sequence[NeuralFunctionIr]) -> None:
    """Fail before any generation when an expression has nothing to verify against.

    Verification reproduces every gold example exactly and refuses a function with
    none, whatever the teacher, so a bundle with such an expression cannot publish.
    """

    missing = [
        ir
        for ir in functions
        if not cast(list[Any], cast(dict[str, Any], ir["definition"]).get("examples") or [])
    ]
    if not missing:
        return
    from semantscript_trainer.teachers.constraints import describe_expression

    names = ", ".join(describe_expression(ir) for ir in missing)
    raise TrainBundleError(
        f"{names} {'has' if len(missing) == 1 else 'have'} no gold examples: verification "
        "needs at least one attested example per expression, whatever the teacher. Add an "
        "examples: [{ inputs, output }] entry to the sema call"
    )


def _application_id(ir: NeuralFunctionIr) -> str:
    encoder_ref = cast(str, cast(dict[str, Any], ir["model"])["encoder"])
    prefix = "encoder."
    if not encoder_ref.startswith(prefix):
        return "application"
    # A depth-routed function names its encoder prefix (`encoder.<app>.depth-NNN`).
    return re.sub(r"\.depth-\d{3}$", "", encoder_ref[len(prefix) :])


def _provenance_counts(entry: TrainedFunction) -> TrainingProvenanceCounts:
    rows = entry.training.split.training + entry.training.split.evaluation
    origins = {
        name: sum(row.origin == name for row in rows)
        for name in ("gold", "synthetic", "constraint-boundary", "counterfactual")
    }
    examples = cast(dict[str, Any], entry.ir["definition"]).get("examples", [])
    return TrainingProvenanceCounts(
        examples=len(cast(list[Any], examples)),
        synthetic=origins["synthetic"],
        adversarial=origins["constraint-boundary"] + origins["counterfactual"],
        calibration=len(entry.training.split.evaluation),
        verification=len(rows) + entry.verification.attested_cases - origins["gold"],
        attested_verification=entry.verification.attested_cases,
    )


def _function_report(entry: TrainedFunction) -> dict[str, Any]:
    training = entry.training
    epoch_index = (
        training.selected_epoch - 1
        if training.selected_epoch is not None
        else len(training.metrics) - 1
    )
    field_accuracy = training.metrics[epoch_index].held_out_field_accuracy
    verification = entry.verification
    source = cast(dict[str, Any], entry.ir.get("source", {}))
    adversarial: dict[str, Any] | None = None
    if entry.adversarial is not None:
        adversarial = {
            "sha256": entry.adversarial.dataset_sha256,
            "cases": len(entry.adversarial.cases),
        }
    return {
        "id": entry.ir["id"],
        "semanticSha256": entry.ir["semanticSha256"],
        "sourcePath": source.get("path"),
        "cache": "reused" if entry.reused else "trained",
        "dataset": {
            "sha256": entry.base.dataset_sha256,
            "cases": len(entry.base.cases),
            "gold": entry.base.gold_count,
        },
        "adversarial": adversarial,
        "training": {
            "trainingRows": training.training_row_count,
            "heldOutRows": training.held_out_row_count,
            "epochs": len(training.metrics),
            "selectedEpoch": training.selected_epoch,
            "heldOutAccuracy": training.held_out_accuracy,
            "heldOutFieldAccuracy": None if field_accuracy is None else list(field_accuracy),
        },
        "verification": {
            "status": verification.status,
            "failures": list(verification.failures),
            "attestedCases": verification.attested_cases,
            "pairCount": verification.pair_count,
            "metrics": verification.to_ir_document()["metrics"],
        },
    }


def _training_key_sha256(trained: list[TrainedFunction]) -> str:
    projection: JsonValue = {
        "kind": _TRAINING_KEY_KIND,
        "keyVersion": 1,
        "functions": [
            {
                "id": cast(str, entry.ir["id"]),
                "semanticSha256": cast(str, entry.ir["semanticSha256"]),
                "datasetSha256": entry.base.dataset_sha256,
                "adversarialDatasetSha256": (
                    None if entry.adversarial is None else entry.adversarial.dataset_sha256
                ),
            }
            for entry in sorted(trained, key=lambda entry: cast(str, entry.ir["id"]))
        ],
    }
    return semantic_json_sha256(projection)


def _pinned_weights_sha256(config: TrainingConfig) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:  # pragma: no cover - training extra absent
        raise TrainBundleError(
            "provenance needs huggingface_hub; install the training extra"
        ) from error
    import hashlib

    weights = hf_hub_download(
        config.encoder_name,
        "model.safetensors",
        revision=config.encoder_revision,
        local_files_only=config.local_files_only,
    )
    return hashlib.sha256(Path(weights).read_bytes()).hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else "0000000"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _dump(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m semantscript_trainer.cli",
        description="Train, verify and export every function of a SemantScript IR bundle.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="produce an artifact from an IR bundle")
    train.add_argument("--bundle", required=True, type=Path, help="compiler IR bundle")
    train.add_argument("--artifact", required=True, type=Path, help="artifact root to publish")
    train.add_argument(
        "--teacher",
        required=True,
        type=Path,
        help="TOML with a [teacher] table, or the keyword constraints for the built-in "
        "constraints teacher",
    )
    train.add_argument("--cache-dir", type=Path, default=Path(".semantscript/cache"))
    train.add_argument("--report", type=Path, help="where to write the JSON report")
    train.add_argument("--cases", type=int, default=DEFAULT_CASES)
    train.add_argument("--application-id")
    train.add_argument("--application-version", default="0.0.0")
    train.add_argument("--compiler-version", default="0.0.0")
    train.add_argument("--encoder-name")
    train.add_argument("--encoder-revision")
    train.add_argument("--local-files-only", action="store_true")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--max-sequence-length", type=int)
    train.add_argument("--evaluation-ratio", type=float)
    train.add_argument("--seed", type=int)
    train.add_argument("--device")
    train.add_argument("--head-architecture")
    train.add_argument("--select-best-epoch", action="store_true")
    train.add_argument("--ece-threshold", type=float)
    train.add_argument("--max-constraint-violation-rate", type=float)
    train.add_argument("--counterfactual-ratio", type=float)
    train.add_argument("--adapter-bottleneck-size", type=int)
    train.add_argument(
        "--no-cache", action="store_true", help="ignore and do not write the build cache"
    )
    train.add_argument("--full", action="store_true", help="retrain every function jointly")
    train.add_argument(
        "--estimate",
        action="store_true",
        help="print what the teacher would cost (requests, tokens, USD, time) as JSON and "
        "exit without calling it",
    )
    train.add_argument(
        "--max-cost-usd",
        type=float,
        help="stop before the teacher request that would take the run past this many USD",
    )
    teacher = commands.add_parser("teacher", help="teacher utilities")
    teacher_commands = teacher.add_subparsers(dest="teacher_command", required=True)
    probe = teacher_commands.add_parser(
        "probe", help="send one small request through the configured teacher"
    )
    probe.add_argument(
        "--teacher", required=True, type=Path, help="teacher TOML, or the keyword constraints"
    )
    probe.add_argument("--cache-dir", type=Path, default=Path(".semantscript/cache"))
    probe.add_argument("--json", action="store_true", help="print the JSON result")
    doctor = commands.add_parser(
        "doctor", help="check the interpreter, packages, device and teacher before a run"
    )
    add_doctor_arguments(doctor)
    return parser


def _training_config(arguments: argparse.Namespace) -> TrainingConfig:
    values = {
        "encoder_name": arguments.encoder_name,
        "encoder_revision": arguments.encoder_revision,
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "maximum_sequence_length": arguments.max_sequence_length,
        "evaluation_ratio": arguments.evaluation_ratio,
        "seed": arguments.seed,
        "device": arguments.device,
        "head_architecture": arguments.head_architecture,
    }
    kwargs: dict[str, Any] = {key: value for key, value in values.items() if value is not None}
    if arguments.local_files_only:
        kwargs["local_files_only"] = True
    if arguments.select_best_epoch:
        kwargs["select_best_epoch"] = True
    return TrainingConfig(**kwargs)


def _verification_config(arguments: argparse.Namespace) -> VerificationConfig:
    kwargs: dict[str, Any] = {}
    if arguments.ece_threshold is not None:
        kwargs["ece_threshold"] = arguments.ece_threshold
    if arguments.max_constraint_violation_rate is not None:
        kwargs["maximum_constraint_violation_rate"] = arguments.max_constraint_violation_rate
    return VerificationConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    """Run ``semantscript_trainer.cli train`` (or ``doctor``) from ``argv`` and return the
    process exit status."""
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "doctor":
        return run_doctor_from_arguments(arguments)
    if arguments.command == "teacher":
        return run_teacher_probe(arguments)
    if arguments.command != "train":  # pragma: no cover - argparse enforces the choice
        return 2

    def log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    if arguments.estimate:
        return _run_estimate(arguments, log)

    report: dict[str, Any] | None = None
    code = 0
    meter: SpendMeter | None = None
    journal: ResponseJournal | None = None
    model_config = None
    try:
        if arguments.max_cost_usd is not None and not (
            arguments.max_cost_usd > 0 and arguments.max_cost_usd < float("inf")
        ):
            raise ValueError("--max-cost-usd must be a positive number")
        bundle = json.loads(Path(arguments.bundle).read_text(encoding="utf-8"))
        config = load_teacher_config(arguments.teacher)
        model_config = language_model_config(config)
        price = None
        try:
            price = resolve_price(config, cache_directory=arguments.cache_dir)
        except TeacherPriceUnknown as error:
            if arguments.max_cost_usd is not None:
                raise
            log(f"warning: {error}; the run counts requests and tokens but not USD")
        meter = SpendMeter(price, max_cost_usd=arguments.max_cost_usd, log=log)
        if model_config is None:
            teacher = create_teacher(config)
        else:
            # Only the Anthropic backend journals responses (Ollama answers are
            # deterministic, temperature 0 with a seed, and resending them is free).
            if not arguments.no_cache and model_config.backend == "anthropic":
                journal = ResponseJournal(arguments.cache_dir, model_config.configuration_sha256)
            teacher = create_teacher(config, meter=meter, journal=journal)
        adversarial_config = (
            AdversarialGenerationConfig(counterfactual_ratio=arguments.counterfactual_ratio)
            if arguments.counterfactual_ratio is not None
            else None
        )
        result = train_bundle(
            bundle,
            arguments.artifact,
            teacher=teacher,
            cache_directory=arguments.cache_dir,
            cases=arguments.cases,
            training_config=_training_config(arguments),
            verification_config=_verification_config(arguments),
            adversarial_config=adversarial_config,
            application_id=arguments.application_id,
            application_version=arguments.application_version,
            compiler_version=arguments.compiler_version,
            use_cache=not arguments.no_cache,
            full=arguments.full,
            log=log,
            meter=meter,
            **(
                {"adapter_bottleneck_size": arguments.adapter_bottleneck_size}
                if arguments.adapter_bottleneck_size is not None
                else {}
            ),
        )
        report = result.report
    except TeacherBudgetExceeded as error:
        journaled = "no" if journal is None else str(journal.count())
        log(
            f"error: {error}. Every dataset finished before the stop stays cached in "
            f"{arguments.cache_dir}"
            + (
                "; rerun with a higher --max-cost-usd (or without it) to resume"
                if journal is None
                else f", and {journaled} paid teacher response(s) are kept in "
                f"{journal.directory}; rerun with a higher --max-cost-usd (or without it) "
                "to resume: journaled responses replay at no cost"
            )
        )
        code = 1
    except TrainBundleFailure as error:
        log(f"error: {error}")
        report = error.report
        code = 1
    except (OSError, ValueError, RuntimeError, TypeError, BuildCacheError) as error:
        log(f"error: {error}")
        code = 1
        if journal is not None and _rejected_teacher_answer(error):
            dropped = journal.discard_touched()
            if dropped:
                log(
                    f"note: the teacher's answers were rejected, so the {dropped} journaled "
                    "response(s) this run used were discarded: a rerun asks the teacher "
                    "again instead of replaying them"
                )
    if meter is not None and (meter.requests or meter.replayed):
        log(meter.line())
        if model_config is not None:
            meter.record_stats(arguments.cache_dir, model_config.configuration_sha256)
    if report is not None:
        text = _dump(report)
        if arguments.report is not None:
            Path(arguments.report).parent.mkdir(parents=True, exist_ok=True)
            Path(arguments.report).write_text(text, encoding="utf-8")
        sys.stdout.write(text)
    return code


def _rejected_teacher_answer(error: BaseException) -> bool:
    """Whether a failed run stopped because the teacher's answers were rejected (as
    opposed to a transport error, a spend cap or a local problem)."""

    rejected = False
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TeacherTransportError | TeacherBudgetExceeded | DatasetCacheError):
            return False
        if isinstance(
            current,
            TeacherResponseError | AdversarialGenerationError | ConstraintError | DatasetError,
        ):
            rejected = True
        current = current.__cause__ or current.__context__
    return rejected


def _run_estimate(arguments: argparse.Namespace, log: Callable[[str], None]) -> int:
    """``train --estimate``: the teacher's cost as JSON on stdout; no request is sent."""

    from semantscript_trainer.teacher_estimate import estimate_bundle

    try:
        bundle = json.loads(Path(arguments.bundle).read_text(encoding="utf-8"))
        functions = _bundle_functions(bundle)
        adversarial_config = (
            AdversarialGenerationConfig(counterfactual_ratio=arguments.counterfactual_ratio)
            if arguments.counterfactual_ratio is not None
            else None
        )
        estimate = estimate_bundle(
            functions,
            load_teacher_config(arguments.teacher),
            cache_directory=arguments.cache_dir,
            cases=arguments.cases,
            adversarial_config=adversarial_config,
            use_cache=not arguments.no_cache,
        )
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        log(f"error: {error}")
        return 1
    if arguments.max_cost_usd is not None:
        estimate["maxCostUsd"] = arguments.max_cost_usd
    sys.stdout.write(_dump(estimate))
    return 0


PROBE_KIND = "semantscript.teacher-probe"


def run_teacher_probe(arguments: argparse.Namespace) -> int:
    """``teacher probe``: one small request through the configured teacher (its fallback
    for a mixed constraints teacher; none for a pure one), with model, latency, tokens
    and USD cost. Exits 1 when the request fails."""

    from semantscript_trainer.doctor import ProbeResult, _key_check, _with_key, probe_teacher

    try:
        config = load_teacher_config(arguments.teacher)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    model_config = language_model_config(config)
    result: dict[str, Any] = {"kind": PROBE_KIND, "probeVersion": 1}
    if model_config is None:
        result.update(
            ok=True,
            backend="constraints",
            model=None,
            requestSent=False,
            latencySeconds=None,
            inputTokens=0,
            outputTokens=0,
            costUsd=0.0,
            priceSource="the constraints teacher sends no request",
            summary="the constraints teacher sends no request; nothing to probe, USD 0",
            fix=None,
        )
    else:
        key = _key_check(model_config, os.environ)
        probe = (
            probe_teacher(_with_key(model_config, os.environ), "request")
            if key.status == "pass"
            else ProbeResult(False, f"no request sent: {key.summary}", key.fix)
        )
        price = None
        price_note = None
        try:
            price = resolve_price(model_config, cache_directory=arguments.cache_dir)
        except TeacherPriceUnknown as error:
            price_note = str(error)
        cost = (
            0.0
            if not probe.request_sent
            else None
            if price is None or probe.input_tokens is None or probe.output_tokens is None
            else price.cost(probe.input_tokens, probe.output_tokens)
        )
        summary = probe.summary
        if cost is not None and probe.request_sent:
            summary += f", USD {cost:.6f}"
        if model_config is not config:
            summary = f"fallback: {summary}"
        result.update(
            ok=probe.ok,
            backend=model_config.backend,
            model=model_config.model,
            baseUrl=model_config.base_url,
            requestSent=probe.request_sent,
            latencySeconds=probe.latency_seconds,
            inputTokens=probe.input_tokens,
            outputTokens=probe.output_tokens,
            costUsd=None if cost is None else round(cost, 8),
            priceSource=(
                None
                if not probe.request_sent
                else price.source
                if price is not None
                else price_note
            ),
            summary=summary,
            fix=probe.fix,
        )
    if arguments.json:
        sys.stdout.write(json.dumps(result, ensure_ascii=True) + "\n")
    else:
        sys.stdout.write(f"teacher probe: {result['summary']}\n")
        if result.get("fix"):
            sys.stdout.write(f"  fix: {result['fix']}\n")
    return 0 if result["ok"] else 1


__all__ = [
    "BUNDLE_KIND",
    "DEFAULT_CASES",
    "REPORT_KIND",
    "REPORT_VERSION",
    "TrainBundleError",
    "TrainBundleFailure",
    "TrainBundleResult",
    "TrainedFunction",
    "main",
    "run_teacher_probe",
    "train_bundle",
]


if __name__ == "__main__":
    raise SystemExit(main())
