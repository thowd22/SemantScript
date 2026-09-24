"""Single adapter versus compile-time routed domain adapters on typed-decisions (TASK-6.7).

The four workflows of the suite are one application with four domains (one
per file). Two artifacts are trained with the same recipe, data and seed: a
*single* configuration where all twenty functions share the application's one
adapter, and a *routed* configuration where each domain has its own adapter
over the shared encoder (full depth). Both are compared per domain on
test-split accuracy, the gate's calibrated accuracy, pair consistency and ECE,
per-case CPU latency of the domain's fused stage through ONNX Runtime, and
artifact bytes. Interference is then measured on the routed layout: three
domains train jointly, the fourth is added on the frozen encoder with a fresh
adapter, and the first three are re-scored; their heads and adapters are
untouched by construction, so the tolerance the experiment can state is the
observed change (expected zero), and the fourth domain's incremental accuracy
is reported next to its jointly trained accuracy.

Usage::

    PYTHONNOUSERSITE=1 HSA_ENABLE_DXG_DETECTION=1 \\
    PYTHONPATH=trainer/src:model/src:.python-packages python3 \\
        benchmarks/typed-decisions/program/run_routed_domains.py \\
        --single-bundle benchmarks/typed-decisions/program/dist/semantscript.ir.v1.json \\
        --routed-bundle benchmarks/typed-decisions/program/dist-routed/semantscript.ir.v1.json \\
        --output-dir benchmarks/typed-decisions/data/results-routed-2026-09-24
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_typed_decisions import (
    ATTESTED_PER_QUESTION,
    LATENCY_WARMUP,
    TRAINING_CASES,
    WORKFLOWS,
    DatasetTeacher,
    attested_cases,
    dataset_root,
    gold_label,
    load_split,
    log,
    question_spec,
    state_text,
    summarize,
    utc_now,
)

from semantscript_trainer.application import (
    add_function_head,
    application_function,
    train_application,
)
from semantscript_trainer.dataset import SyntheticDatasetGenerator
from semantscript_trainer.training import TrainingConfig, _load_tokenizer
from semantscript_trainer.verification import (
    VerificationConfig,
    evaluate_training_result,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RESULT_KIND = "semantscript.typed-decisions-routed-domains"
RESULT_VERSION = 1


def workflow_of(ir: dict[str, Any]) -> str:
    return ir["source"]["path"].removesuffix(".sem.ts").split("/")[-1]


def training_config(arguments: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        local_files_only=True,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=0.01,
        maximum_sequence_length=arguments.max_sequence_length,
        evaluation_ratio=0.1,
        seed=1,
        device=arguments.device,
        head_architecture="linear",
        select_best_epoch=True,
    )


def build_corpora(
    functions: list[dict[str, Any]], root: Path, cache: Path
) -> tuple[
    list[Any], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]
]:
    """Datasets for every function from the frozen train split; the held rows attest."""

    corpora = []
    bases: dict[str, Any] = {}
    held: dict[str, list[dict[str, Any]]] = {}
    tests: dict[str, list[dict[str, Any]]] = {}
    teachers: dict[str, DatasetTeacher] = {}
    for ir in functions:
        workflow = workflow_of(ir)
        if workflow not in teachers:
            train_rows = load_split(root, workflow, "train")
            teachers[workflow] = DatasetTeacher(workflow, train_rows[:TRAINING_CASES])
            held[workflow] = train_rows[TRAINING_CASES:]
            tests[workflow] = load_split(root, workflow, "test")
        base = SyntheticDatasetGenerator(teachers[workflow], cache).generate(ir, TRAINING_CASES)
        bases[ir["id"]] = base
        corpora.append(application_function(ir, base))
    return corpora, bases, held, tests


def verify_all(
    functions: list[dict[str, Any]],
    application: Any,
    bases: dict[str, Any],
    held: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
) -> dict[str, Any]:
    verifications = {}
    for ir in functions:
        question = question_spec(ir)
        verifications[ir["id"]] = evaluate_training_result(
            ir,
            application.functions[ir["id"]],
            bases[ir["id"]],
            None,
            tokenizer=tokenizer,
            attested_verification=attested_cases(held[workflow_of(ir)], question),
            config=VerificationConfig(),
            verified_at=utc_now(),
        )
    return verifications


def score_domains(
    functions: list[dict[str, Any]],
    application: Any,
    verifications: dict[str, Any],
    tests: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Per-domain test accuracy through each function's own view (adapter and depth)."""

    import torch

    model = application.model
    model.eval()
    device = torch.device("cuda" if config.device == "cuda" else "cpu")
    model.to(device)
    by_workflow: dict[str, list[dict[str, Any]]] = {}
    for ir in functions:
        by_workflow.setdefault(workflow_of(ir), []).append(ir)
    domains: dict[str, Any] = {}
    with torch.no_grad():
        for workflow, members in by_workflow.items():
            rows = tests[workflow]
            texts = [
                state_text(members[0], row["state"], config.canonical_input_version) for row in rows
            ]
            embeddings_by_adapter: dict[str, Any] = {}
            questions: dict[str, Any] = {}
            correct_total = 0
            for ir in members:
                ref = model.adapter_ref_of(ir["id"])
                if ref not in embeddings_by_adapter:
                    parts = []
                    for offset in range(0, len(texts), 8):
                        encoded = tokenizer(
                            texts[offset : offset + 8],
                            add_special_tokens=True,
                            padding=True,
                            truncation=True,
                            max_length=config.maximum_sequence_length,
                            return_tensors="pt",
                        )
                        parts.append(
                            model.embed(
                                encoded["input_ids"].to(device),
                                encoded["attention_mask"].to(device),
                                ir["id"],
                            )
                        )
                    embeddings_by_adapter[ref] = torch.cat(parts, dim=0)
                question = question_spec(ir)
                logits = model.heads[ir["id"]](embeddings_by_adapter[ref]).float()
                temperature = verifications[ir["id"]].metrics.heads[0].calibration.temperature
                if question.kind == "noul":
                    p_true = torch.sigmoid(logits[:, 0] / temperature)
                    probabilities = torch.stack([1 - p_true, p_true], dim=1)
                else:
                    probabilities = torch.softmax(logits / temperature, dim=1)
                predicted = probabilities.argmax(dim=1).tolist()
                correct = sum(
                    int(predicted[index] == question.support.index(gold_label(row, question)))
                    for index, row in enumerate(rows)
                )
                correct_total += correct
                metrics = verifications[ir["id"]].metrics
                questions[question.name] = {
                    "functionId": ir["id"],
                    "accuracy": round(correct / len(rows), 4),
                    "gate": verifications[ir["id"]].status,
                    "gateAccuracy": round(metrics.accuracy, 4),
                    "gateEce": round(metrics.ece, 4),
                    "pairConsistency": round(metrics.pair_consistency, 4),
                }
            domains[workflow] = {
                "adapterRef": model.adapter_ref_of(members[0]["id"]),
                "questions": questions,
                "accuracy": round(correct_total / (len(rows) * len(members)), 4),
                "decisions": len(rows) * len(members),
                "gateAccuracy": round(
                    statistics.fmean(q["gateAccuracy"] for q in questions.values()), 4
                ),
                "gateEce": round(statistics.fmean(q["gateEce"] for q in questions.values()), 4),
                "pairConsistency": round(
                    statistics.fmean(q["pairConsistency"] for q in questions.values()), 4
                ),
                "gatePassed": sum(q["gate"] == "passed" for q in questions.values()),
            }
    model.to("cpu")
    return domains


def measure_cpu(
    functions: list[dict[str, Any]],
    application: Any,
    tests: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Export the application (routed or not) and time each domain's stage on CPU."""

    import numpy
    import onnxruntime

    from semantscript_model.export import depth_key, export_routed_application_components

    model = application.model
    model.to("cpu")
    model.eval()
    ids = [ir["id"] for ir in functions]
    refs = list(model.adapter_refs)
    depths = {ref: model.adapter_depth(ref) for ref in refs}
    sample = tokenizer(
        [
            state_text(
                functions[0],
                tests[workflow_of(functions[0])][0]["state"],
                config.canonical_input_version,
            )
        ],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=config.maximum_sequence_length,
        return_tensors="pt",
    )
    with tempfile.TemporaryDirectory(prefix="semantscript-routed-") as temporary:
        directory = Path(temporary)
        encoder_paths = {
            depth_key(depth): directory / f"encoder-{depth_key(depth)}.onnx"
            for depth in set(depths.values())
        }
        adapter_paths = {ref: directory / f"adapter-{index}.onnx" for index, ref in enumerate(refs)}
        export_routed_application_components(
            model.encoder,
            {ref: model.adapter_for(ref) for ref in refs},
            depths,
            {function_id: model.heads[function_id] for function_id in ids},
            {function_id: model.adapter_ref_of(function_id) for function_id in ids},
            sample["input_ids"],
            sample["attention_mask"],
            encoder_paths=encoder_paths,
            adapter_paths=adapter_paths,
            head_paths={function_id: directory / f"{function_id}.onnx" for function_id in ids},
        )
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3

        def session(path: Path) -> Any:
            return onnxruntime.InferenceSession(
                str(path), options, providers=["CPUExecutionProvider"]
            )

        encoders = {key: session(path) for key, path in encoder_paths.items()}
        adapters = {ref: session(path) for ref, path in adapter_paths.items()}
        heads = {function_id: session(directory / f"{function_id}.onnx") for function_id in ids}
        bytes_by_role = {
            "encoders": sum(path.stat().st_size for path in encoder_paths.values()),
            "adapters": sum(path.stat().st_size for path in adapter_paths.values()),
            "heads": sum((directory / f"{function_id}.onnx").stat().st_size for function_id in ids),
        }
        bytes_by_role["total"] = sum(bytes_by_role.values())
        by_workflow: dict[str, list[dict[str, Any]]] = {}
        for ir in functions:
            by_workflow.setdefault(workflow_of(ir), []).append(ir)
        latency: dict[str, Any] = {}
        for workflow, members in by_workflow.items():
            ref = model.adapter_ref_of(members[0]["id"])
            encoder = encoders[depth_key(depths[ref])]
            adapter = adapters[ref]
            member_heads = [heads[ir["id"]] for ir in members]
            samples = []
            for index, row in enumerate(tests[workflow]):
                encoded = tokenizer(
                    [state_text(members[0], row["state"], config.canonical_input_version)],
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=config.maximum_sequence_length,
                    return_tensors="pt",
                )
                feed = {
                    "input_ids": encoded["input_ids"].numpy().astype(numpy.int64),
                    "attention_mask": encoded["attention_mask"].numpy().astype(numpy.int64),
                }
                started = time.perf_counter()
                sentence = encoder.run(None, feed)[0]
                function_embedding = adapter.run(None, {"sentence_embedding": sentence})[0]
                for head in member_heads:
                    head.run(None, {"function_embedding": function_embedding})
                if index >= LATENCY_WARMUP:
                    samples.append((time.perf_counter() - started) * 1000)
            latency[workflow] = summarize(samples)
    return {
        "provider": "CPUExecutionProvider",
        "artifactBytes": bytes_by_role,
        "stageLatency": latency,
    }


def train_configuration(
    name: str,
    functions: list[dict[str, Any]],
    root: Path,
    cache: Path,
    config: TrainingConfig,
    tokenizer: Any,
) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch

    corpora, bases, held, tests = build_corpora(functions, root, cache)
    refs = sorted({corpus.adapter_ref for corpus in corpora})
    log(f"{name}: training {len(functions)} functions jointly over {len(refs)} adapter(s): {refs}")
    started = time.monotonic()
    application = train_application(corpora, config=config, tokenizer=tokenizer)
    train_seconds = time.monotonic() - started
    verifications = verify_all(functions, application, bases, held, tokenizer)
    record: dict[str, Any] = {
        "adapters": refs,
        "trainSeconds": round(train_seconds, 1),
        "selectedEpoch": application.selected_epoch,
        "domains": score_domains(functions, application, verifications, tests, tokenizer, config),
    }
    record["accuracy"] = round(
        sum(d["accuracy"] * d["decisions"] for d in record["domains"].values())
        / sum(d["decisions"] for d in record["domains"].values()),
        4,
    )
    record["cpuOnnx"] = measure_cpu(functions, application, tests, tokenizer, config)
    log(
        f"{name}: accuracy {record['accuracy']} per domain {json.dumps({k: v['accuracy'] for k, v in record['domains'].items()})}"
    )
    torch.cuda.empty_cache()
    return record, application, bases, held, tests


def interference(
    functions: list[dict[str, Any]],
    root: Path,
    cache: Path,
    config: TrainingConfig,
    tokenizer: Any,
    joint: dict[str, Any],
    added_workflow: str,
) -> dict[str, Any]:
    """Train three domains jointly, add the fourth on the frozen encoder, re-score the three."""

    import torch

    from semantscript_trainer.verification import model_state_sha256

    kept = [ir for ir in functions if workflow_of(ir) != added_workflow]
    added = [ir for ir in functions if workflow_of(ir) == added_workflow]
    corpora, bases, held, tests = build_corpora(functions, root, cache)
    kept_corpora = [corpus for corpus in corpora if workflow_of(corpus.ir) != added_workflow]
    log(
        f"interference: training {len(kept)} functions of {len({workflow_of(ir) for ir in kept})} domains jointly"
    )
    application = train_application(kept_corpora, config=config, tokenizer=tokenizer)
    verifications = verify_all(kept, application, bases, held, tokenizer)
    before = score_domains(kept, application, verifications, tests, tokenizer, config)
    digests_before = {
        ir["id"]: model_state_sha256(application.functions[ir["id"]].model) for ir in kept
    }
    log(f"interference: adding {len(added)} functions of {added_workflow} on the frozen encoder")
    started = time.monotonic()
    for corpus in corpora:
        if workflow_of(corpus.ir) == added_workflow:
            application = add_function_head(application, corpus, config=config, tokenizer=tokenizer)
    add_seconds = time.monotonic() - started
    verifications.update(verify_all(added, application, bases, held, tokenizer))
    after_kept = score_domains(kept, application, verifications, tests, tokenizer, config)
    after_added = score_domains(added, application, verifications, tests, tokenizer, config)
    digests_after = {
        ir["id"]: model_state_sha256(application.functions[ir["id"]].model) for ir in kept
    }
    changed = sorted(fid for fid in digests_before if digests_before[fid] != digests_after[fid])
    deltas = {
        workflow: round(after_kept[workflow]["accuracy"] - before[workflow]["accuracy"], 4)
        for workflow in before
    }
    torch.cuda.empty_cache()
    return {
        "addedDomain": added_workflow,
        "keptDomains": sorted(before),
        "before": {workflow: before[workflow]["accuracy"] for workflow in before},
        "after": {workflow: after_kept[workflow]["accuracy"] for workflow in after_kept},
        "accuracyDeltas": deltas,
        "maximumAbsoluteDelta": max(abs(delta) for delta in deltas.values()),
        "functionsWhoseWeightsChanged": changed,
        "addedDomainIncrementalAccuracy": after_added[added_workflow]["accuracy"],
        "addedDomainJointAccuracy": joint["domains"][added_workflow]["accuracy"],
        "addSeconds": round(add_seconds, 1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--single-bundle", required=True, type=Path)
    parser.add_argument("--routed-bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--cache-dir", type=Path, default=_REPOSITORY_ROOT / "benchmarks/typed-decisions/data/cache"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-sequence-length", type=int, default=768)
    parser.add_argument("--interference-domain", default="agent_trace_observability")
    parser.add_argument(
        "--skip", default="", help="comma-separated stages to skip: single,routed,interference"
    )
    arguments = parser.parse_args(argv)
    skip = {name for name in arguments.skip.split(",") if name}
    root = dataset_root()
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.json"
    results: dict[str, Any] = (
        json.loads(results_path.read_text(encoding="utf-8")) if results_path.is_file() else {}
    )
    config = training_config(arguments)
    tokenizer = _load_tokenizer(config)
    single = json.loads(arguments.single_bundle.read_text(encoding="utf-8"))["functions"]
    routed = json.loads(arguments.routed_bundle.read_text(encoding="utf-8"))["functions"]
    assert [ir["id"] for ir in single] == [ir["id"] for ir in routed], (
        "bundles must hold the same functions"
    )
    results.update(
        {
            "kind": RESULT_KIND,
            "resultVersion": RESULT_VERSION,
            "startedAt": results.get("startedAt", utc_now()),
            "workflows": list(WORKFLOWS),
            "functions": len(single),
            "trainingCases": TRAINING_CASES,
            "attestedPerQuestion": ATTESTED_PER_QUESTION,
            "recipe": {
                "epochs": config.epochs,
                "batchSize": config.batch_size,
                "learningRate": config.learning_rate,
                "maximumSequenceLength": config.maximum_sequence_length,
                "seed": config.seed,
            },
        }
    )

    def save() -> None:
        results["completedAt"] = utc_now()
        results_path.write_text(
            json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        log(f"wrote {results_path}")

    if "single" not in skip:
        record, application, *_ = train_configuration(
            "single", single, root, arguments.cache_dir, config, tokenizer
        )
        results["single"] = record
        del application
        save()
    if "routed" not in skip:
        record, application, *_ = train_configuration(
            "routed", routed, root, arguments.cache_dir, config, tokenizer
        )
        results["routed"] = record
        del application
        save()
    if "interference" not in skip:
        results["interference"] = interference(
            routed,
            root,
            arguments.cache_dir,
            config,
            tokenizer,
            results["routed"],
            arguments.interference_domain,
        )
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
