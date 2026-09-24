"""Universal encoder plus tiny heads: does head-only training compile in seconds? (TASK-7.3)

Replays the frozen refund corpus and measures, on the GPU, wall-clock and
release-attested accuracy for four regimes:

* ``full``: full fine-tuning with the committed release recipe at epoch budgets;
* ``head-only``: the same trainer with ``freeze_encoder`` (encoder in eval mode,
  only the head trains) at the same budgets;
* ``cached``: encode the corpus once with the frozen encoder, then fit the head
  on the cached embeddings with the trainer's proper loss, evaluating the release
  cases after every epoch (the encode time is reported apart from the fit time);
* ``laya``: score every corpus row and release case with the pinned Laya
  typed-decisions checkpoint through the benchmark's own choice question
  (zero-training accuracy), distill those probabilities into a fixed head on the
  cached embeddings without using any generated label, then refine on the
  generated labels from that start.

Time-to-target is the first budget (or cached-embedding epoch) whose release
misses are zero. Every regime ends with the trainer's own verification so the
numbers are comparable with the release gate.

Usage::

    PYTHONNOUSERSITE=1 HSA_ENABLE_DXG_DETECTION=1 \\
    PYTHONPATH=.:trainer/src:model/src:.python-packages python3 -m \\
        benchmarks.refund.program.run_warm_start_experiment \\
        --corpus-manifest benchmarks/refund/data/pooled-v4-2026-09-24/manifest.json \\
        --heldout-dir benchmarks/refund/data/heldout-uci-2026-09-23 \\
        --laya-source /tmp/semantscript-laya-inspect-20260923 \\
        --output-dir benchmarks/refund/data/results-warm-start-2026-09-24
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.refund.live.laya_worker import (
    LAYA_CHECKPOINT,
    LAYA_CHECKPOINT_REVISION,
    LAYA_CODE_REVISION,
    LAYA_VERSION,
    SUPPORT,
)
from benchmarks.refund.program.pipeline import (
    compile_refund_program,
    parse_release_verification_record,
)
from benchmarks.refund.program.run_release_pipeline import _teacher_from_manifest, _utc_now

from semantscript_trainer.adversarial import (
    AdversarialDatasetGenerator,
    AdversarialGenerationConfig,
)
from semantscript_trainer.canonical_input import serialize_canonical_inputs
from semantscript_trainer.dataset import SyntheticDatasetGenerator
from semantscript_trainer.training import (
    DEFAULT_ENCODER_NAME,
    DEFAULT_ENCODER_REVISION,
    EpochMetrics,
    TrainingConfig,
    TrainingResult,
    _batch_loss,
    _build_head,
    _build_sentence_encoder,
    _load_tokenizer,
    _prepare_rows,
    train_classifier,
)
from semantscript_trainer.training_contract import (
    HeldOutSplitConfig,
    assemble_training_corpus,
    split_training_corpus,
)
from semantscript_trainer.verification import VerificationConfig, evaluate_training_result

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RESULT_KIND = "semantscript.warm-start-experiment"
RESULT_VERSION = 1
DEFAULT_BUDGETS = (1, 2, 4, 8)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def release_recipe(device: str, epochs: int, *, freeze: bool = False) -> TrainingConfig:
    """The committed compact-release recipe (release-compact-2026-09-24), by budget."""

    return TrainingConfig(
        encoder_name=DEFAULT_ENCODER_NAME,
        encoder_revision=DEFAULT_ENCODER_REVISION,
        local_files_only=True,
        epochs=epochs,
        batch_size=16,
        learning_rate=5e-3 if freeze else 3e-5,
        weight_decay=0.01,
        maximum_sequence_length=128,
        evaluation_ratio=0.1,
        seed=1,
        device=device,
        head_architecture="linear",
        select_best_epoch=False,
        freeze_encoder=freeze,
    )


def gate_metrics(verification: Any, release_cases: int, seconds: float) -> dict[str, Any]:
    metrics = verification.metrics
    return {
        "seconds": round(seconds, 2),
        "status": verification.status,
        "releaseMisses": metrics.example_failures,
        "releaseAccuracy": round(1 - metrics.example_failures / release_cases, 4),
        "calibratedAccuracy": round(metrics.accuracy, 4),
        "ece": round(metrics.ece, 4),
        "brier": round(metrics.brier, 4),
        "constraintViolations": metrics.constraint_violations,
        "pairConsistency": round(metrics.pair_consistency, 4),
        "temperature": round(metrics.heads[0].calibration.temperature, 4),
    }


def verify(
    ir: Any, training: Any, base: Any, adversarial: Any, release: Any, tokenizer: Any
) -> Any:
    return evaluate_training_result(
        ir,
        training,
        base,
        adversarial,
        tokenizer=tokenizer,
        attested_verification=release.generated_cases,
        config=VerificationConfig(maximum_constraint_violation_rate=0.01),
        verified_at=_utc_now(),
    )


# ---- regimes through the trainer -------------------------------------------


def run_trainer_regime(
    name: str,
    ir: Any,
    base: Any,
    adversarial: Any,
    release: Any,
    tokenizer: Any,
    device: str,
    budgets: Sequence[int],
    *,
    freeze: bool,
) -> dict[str, Any]:
    import torch

    runs = []
    for epochs in budgets:
        config = release_recipe(device, epochs, freeze=freeze)
        log(f"{name}: training {epochs} epoch(s)")
        started = time.monotonic()
        training = train_classifier(ir, base, adversarial, config=config, tokenizer=tokenizer)
        train_seconds = time.monotonic() - started
        verification = verify(ir, training, base, adversarial, release, tokenizer)
        entry = {
            "epochs": epochs,
            "heldOutCurve": [round(m.held_out_accuracy, 4) for m in training.metrics],
            **gate_metrics(verification, len(release.case_ids), train_seconds),
        }
        log(f"{name}: {json.dumps(entry)}")
        runs.append(entry)
        del training
        torch.cuda.empty_cache()
    return {"regime": name, "freezeEncoder": freeze, "runs": runs, **time_to_target(runs)}


def time_to_target(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    for run in runs:
        if run["releaseMisses"] == 0:
            return {"timeToTargetSeconds": run["seconds"], "targetBudget": run["epochs"]}
    return {"timeToTargetSeconds": None, "targetBudget": None}


# ---- cached embeddings ------------------------------------------------------


class EmbeddingCache:
    """Frozen-encoder embeddings for the corpus split and the release cases."""

    def __init__(
        self, ir: Any, corpus: Any, release: Any, config: TrainingConfig, tokenizer: Any
    ) -> None:
        import torch

        self.torch = torch
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if config.device == "cuda" else "cpu")
        self.split = split_training_corpus(
            corpus, HeldOutSplitConfig(evaluation_ratio=config.evaluation_ratio, seed=config.seed)
        )
        self.corpus = corpus
        prepared = _prepare_rows(ir["inputs"], self.split, config.canonical_input_version)
        self.training_rows = prepared["training"]
        self.evaluation_rows = prepared["evaluation"]
        self.encoder = _build_sentence_encoder(config, None).to(self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        started = time.monotonic()
        self.training_embeddings = self.encode([row.text for row in self.training_rows])
        self.evaluation_embeddings = self.encode([row.text for row in self.evaluation_rows])
        release_texts = [
            serialize_canonical_inputs(
                ir["inputs"], case.inputs, version=config.canonical_input_version
            ).decode("utf-8")
            for case in release.generated_cases
        ]
        self.release_embeddings = self.encode(release_texts)
        self.encode_seconds = time.monotonic() - started
        label = corpus.head.label_index
        self.training_targets = torch.tensor(
            [row.row.label_index for row in self.training_rows],
            dtype=torch.long,
            device=self.device,
        )
        self.evaluation_targets = torch.tensor(
            [row.row.label_index for row in self.evaluation_rows],
            dtype=torch.long,
            device=self.device,
        )
        self.release_targets = torch.tensor(
            [label(case.output) for case in release.generated_cases],
            dtype=torch.long,
            device=self.device,
        )
        self.row_count = len(self.training_rows) + len(self.evaluation_rows) + len(release_texts)

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> Any:
        torch = self.torch
        chunks = []
        with torch.no_grad():
            for offset in range(0, len(texts), batch_size):
                encoded = self.tokenizer(
                    list(texts[offset : offset + batch_size]),
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=self.config.maximum_sequence_length,
                    return_tensors="pt",
                )
                chunks.append(
                    self.encoder(
                        encoded["input_ids"].to(self.device),
                        encoded["attention_mask"].to(self.device),
                    ).float()
                )
        return torch.cat(chunks, dim=0)

    def new_head(self) -> Any:
        return _build_head(self.corpus.head, self.encoder.hidden_size, self.config).to(self.device)

    def accuracy(self, head: Any, embeddings: Any, targets: Any) -> tuple[float, int]:
        torch = self.torch
        head.eval()
        with torch.no_grad():
            predictions = head(embeddings).argmax(dim=-1)
        misses = int((predictions != targets).sum().item())
        return 1 - misses / len(targets), misses

    def fit(
        self,
        head: Any,
        epochs: int,
        *,
        soft_targets: Any | None = None,
        learning_rate: float = 5e-3,
        batch_size: int = 64,
    ) -> list[dict[str, Any]]:
        """Fit the head on cached embeddings; hard labels use the trainer's proper loss."""

        torch = self.torch
        optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.01)
        heads = self.corpus.output_heads
        count = self.training_embeddings.shape[0]
        history = []
        elapsed = 0.0
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        for epoch in range(1, epochs + 1):
            started = time.monotonic()
            head.train()
            order = torch.randperm(count, generator=generator).to(self.device)
            loss_total = 0.0
            for offset in range(0, count, batch_size):
                index = order[offset : offset + batch_size]
                logits = head(self.training_embeddings[index])
                if soft_targets is None:
                    loss = _batch_loss(
                        logits, self.training_targets[index][:, None], heads, self.config.loss
                    )
                else:
                    loss = torch.nn.functional.kl_div(
                        torch.log_softmax(logits, dim=-1),
                        soft_targets[index],
                        reduction="batchmean",
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_total += float(loss.detach().item()) * len(index)
            elapsed += time.monotonic() - started
            held_out, _ = self.accuracy(head, self.evaluation_embeddings, self.evaluation_targets)
            release_accuracy, misses = self.accuracy(
                head, self.release_embeddings, self.release_targets
            )
            history.append(
                {
                    "epoch": epoch,
                    "fitSeconds": round(elapsed, 3),
                    "meanLoss": round(loss_total / count, 5),
                    "heldOutAccuracy": round(held_out, 4),
                    "releaseAccuracy": round(release_accuracy, 4),
                    "releaseMisses": misses,
                }
            )
        return history

    def training_result(self, head: Any, history: Sequence[dict[str, Any]]) -> TrainingResult:
        from semantscript_model.classifier import SemanticClassifier

        model = SemanticClassifier(self.encoder, head)
        return TrainingResult(
            model=model,
            head=self.corpus.head,
            split=self.split,
            config=self.config,
            device=str(self.device),
            metrics=tuple(
                EpochMetrics(
                    epoch=entry["epoch"],
                    mean_training_loss=entry["meanLoss"],
                    held_out_accuracy=entry["heldOutAccuracy"],
                )
                for entry in history
            ),
            function_id=self.corpus.function_id,
            semantic_sha256=self.corpus.semantic_sha256,
            base_dataset_sha256=self.corpus.base_dataset_sha256,
            adversarial_dataset_sha256=self.corpus.adversarial_dataset_sha256,
            output_heads=self.corpus.output_heads,
        )


def first_zero_miss(history: Sequence[dict[str, Any]], offset_seconds: float) -> dict[str, Any]:
    for entry in history:
        if entry["releaseMisses"] == 0:
            return {
                "timeToTargetSeconds": round(offset_seconds + entry["fitSeconds"], 3),
                "targetEpoch": entry["epoch"],
            }
    return {"timeToTargetSeconds": None, "targetEpoch": None}


def run_cached_regime(
    cache: EmbeddingCache,
    ir: Any,
    base: Any,
    adversarial: Any,
    release: Any,
    tokenizer: Any,
    epochs: int,
    *,
    name: str = "cached",
    initial_head: Any | None = None,
    extra_seconds: float = 0.0,
) -> dict[str, Any]:
    head = cache.new_head() if initial_head is None else initial_head
    log(f"{name}: fitting the head on cached embeddings for {epochs} epoch(s)")
    history = cache.fit(head, epochs)
    training = cache.training_result(head, history)
    started = time.monotonic()
    verification = verify(ir, training, base, adversarial, release, tokenizer)
    verify_seconds = time.monotonic() - started
    target = first_zero_miss(history, cache.encode_seconds + extra_seconds)
    entry = {
        "regime": name,
        "encodeSeconds": round(cache.encode_seconds, 2),
        "encodedRows": cache.row_count,
        "epochs": history,
        "fitSeconds": history[-1]["fitSeconds"],
        "verifySeconds": round(verify_seconds, 2),
        "final": gate_metrics(
            verification,
            len(release.case_ids),
            cache.encode_seconds + extra_seconds + history[-1]["fitSeconds"],
        ),
        **target,
    }
    log(f"{name}: {json.dumps({k: v for k, v in entry.items() if k != 'epochs'})}")
    return entry


# ---- Laya warm start ----------------------------------------------------------


def laya_questions(inputs: Sequence[dict[str, Any]]) -> list[str]:
    """The exact choice question the benchmark's Laya adapter sends, via its compiled builder."""

    script = """
import { buildRefundPrompt } from "./benchmarks/refund/dist/adapters/common.js";
let text = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { text += chunk; });
process.stdin.on("end", () => {
  const out = JSON.parse(text).map((inputs) => { const p = buildRefundPrompt(inputs); return `${p.system}\\n${p.user}`; });
  process.stdout.write(JSON.stringify(out));
});
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=_REPOSITORY_ROOT,
        input=json.dumps(list(inputs)),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def load_laya(source: Path, checkpoint: Path, device: str) -> Any:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True, check=True
    ).stdout.strip()
    if revision != LAYA_CODE_REVISION:
        raise RuntimeError(f"Laya source at {source} is {revision}, not {LAYA_CODE_REVISION}")
    if checkpoint.name != LAYA_CHECKPOINT_REVISION:
        raise RuntimeError("Laya checkpoint path must be the pinned snapshot")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source))
    with contextlib.redirect_stdout(sys.stderr):
        import laya

        if laya.__version__ != LAYA_VERSION:
            raise RuntimeError("Laya package version mismatch")
        return laya.load(str(checkpoint), device=device, fast=False)


def laya_probabilities(agent: Any, questions: Sequence[str], batch_size: int) -> list[list[float]]:
    definition = {
        "refund": {
            "type": "choice",
            "instructions": "Choose the refund decision that best satisfies the supplied policy.",
            "criteria": {value: None for value in SUPPORT},
        }
    }
    rows: list[list[float]] = []
    with contextlib.redirect_stdout(sys.stderr):
        for offset in range(0, len(questions), batch_size):
            results = agent.predict_batch(list(questions[offset : offset + batch_size]), definition)
            for result in results:
                answer = result["answers"]["refund"]
                values = [float(answer["probabilities"][label]) for label in SUPPORT]
                total = sum(values)
                rows.append([value / total for value in values])
    return rows


def run_laya_regime(
    cache: EmbeddingCache,
    ir: Any,
    base: Any,
    adversarial: Any,
    release: Any,
    tokenizer: Any,
    source: Path,
    checkpoint: Path,
    device: str,
    distill_epochs: int,
    refine_epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    torch = cache.torch
    support = list(cache.corpus.head.support)
    if support != list(SUPPORT):
        raise RuntimeError("refund support order changed")
    log("laya: loading the pinned checkpoint")
    started = time.monotonic()
    agent = load_laya(source, checkpoint, device)
    load_seconds = time.monotonic() - started
    training_inputs = [row.row.inputs for row in cache.training_rows]
    release_inputs = [case.inputs for case in release.generated_cases]
    log(f"laya: scoring {len(training_inputs)} corpus rows and {len(release_inputs)} release cases")
    started = time.monotonic()
    questions = laya_questions([*training_inputs, *release_inputs])
    scored = laya_probabilities(agent, questions, batch_size)
    score_seconds = time.monotonic() - started
    del agent
    torch.cuda.empty_cache()
    corpus_scores = torch.tensor(
        scored[: len(training_inputs)], dtype=torch.float32, device=cache.device
    )
    release_scores = torch.tensor(
        scored[len(training_inputs) :], dtype=torch.float32, device=cache.device
    )
    zero_training_release = float(
        (release_scores.argmax(dim=-1) == cache.release_targets).float().mean().item()
    )
    zero_training_corpus = float(
        (corpus_scores.argmax(dim=-1) == cache.training_targets).float().mean().item()
    )
    log(
        f"laya: zero-training accuracy release {zero_training_release:.4f}, "
        f"corpus labels {zero_training_corpus:.4f} ({score_seconds:.1f}s)"
    )

    head = cache.new_head()
    log(f"laya: distilling into a fixed head for {distill_epochs} epoch(s), no generated labels")
    distill_history = cache.fit(head, distill_epochs, soft_targets=corpus_scores)
    distilled_release, distilled_misses = cache.accuracy(
        head, cache.release_embeddings, cache.release_targets
    )
    distilled_held_out, _ = cache.accuracy(
        head, cache.evaluation_embeddings, cache.evaluation_targets
    )
    log(
        f"laya: distilled head release accuracy {distilled_release:.4f} ({distilled_misses} misses)"
    )
    refine = run_cached_regime(
        cache,
        ir,
        base,
        adversarial,
        release,
        tokenizer,
        refine_epochs,
        name="laya-warm-start",
        initial_head=head,
        extra_seconds=load_seconds + score_seconds + distill_history[-1]["fitSeconds"],
    )
    return {
        "regime": "laya",
        "checkpoint": LAYA_CHECKPOINT,
        "checkpointRevision": LAYA_CHECKPOINT_REVISION,
        "codeRevision": LAYA_CODE_REVISION,
        "loadSeconds": round(load_seconds, 2),
        "scoreSeconds": round(score_seconds, 2),
        "scoredRows": len(scored),
        "zeroTraining": {
            "releaseAccuracy": round(zero_training_release, 4),
            "corpusLabelAgreement": round(zero_training_corpus, 4),
        },
        "distilled": {
            "epochs": distill_history,
            "releaseAccuracy": round(distilled_release, 4),
            "releaseMisses": distilled_misses,
            "heldOutAccuracy": round(distilled_held_out, 4),
        },
        "refined": refine,
    }


# ---- driver ------------------------------------------------------------------


def environment(device: str) -> dict[str, Any]:
    import torch

    versions = {"python": platform.python_version(), "torch": torch.__version__}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover
        pass
    return {
        "capturedAt": _utc_now(),
        "operatingSystem": f"{platform.system()} {platform.release()}",
        "device": device,
        "accelerator": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": versions,
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--corpus-manifest", required=True)
    parser.add_argument("--heldout-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--laya-source", required=True)
    parser.add_argument(
        "--laya-checkpoint",
        default=str(
            Path.home()
            / ".cache/huggingface/hub/models--convaiinnovations--laya-typed-decisions/snapshots"
            / LAYA_CHECKPOINT_REVISION
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--cached-epochs", type=int, default=40)
    parser.add_argument("--distill-epochs", type=int, default=20)
    parser.add_argument("--laya-batch-size", type=int, default=16)
    parser.add_argument("--skip", default="", help="comma-separated regimes to skip")
    arguments = parser.parse_args(argv)
    budgets = tuple(int(item) for item in arguments.budgets.split(",") if item)
    skipped = {item.strip() for item in arguments.skip.split(",") if item.strip()}
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(arguments.corpus_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_root = manifest_path.parent
    heldout = Path(arguments.heldout_dir)
    with tempfile.TemporaryDirectory(prefix="semantscript-warm-start-") as temporary:
        compiled = compile_refund_program(Path(temporary) / "compiler")
    ir = compiled.source_ir
    teacher = _teacher_from_manifest(manifest)
    base = SyntheticDatasetGenerator(teacher, corpus_root / "synthetic").generate(
        ir, manifest["synthetic"]["requestedCaseCount"]
    )
    adversarial = AdversarialDatasetGenerator(
        teacher,
        corpus_root / "adversarial",
        config=AdversarialGenerationConfig(
            counterfactual_ratio=manifest["adversarial"]["config"]["counterfactualRatio"],
            maximum_attempts=manifest["adversarial"]["config"]["maximumAttempts"],
        ),
    ).generate(ir, base)
    release = parse_release_verification_record(
        (heldout / "release-verification.json").read_bytes(), ir
    )
    corpus = assemble_training_corpus(ir, base, adversarial)
    config = release_recipe(arguments.device, 1, freeze=True)
    tokenizer = _load_tokenizer(config)
    log(
        f"corpus: {base.synthetic_count} synthetic + {len(adversarial.cases)} adversarial rows; "
        f"release cases: {len(release.case_ids)}"
    )

    results: dict[str, Any] = {
        "kind": RESULT_KIND,
        "resultVersion": RESULT_VERSION,
        "startedAt": _utc_now(),
        "function": {"id": compiled.function_id, "semanticSha256": compiled.semantic_sha256},
        "corpus": {
            "manifest": str(manifest_path.resolve().relative_to(_REPOSITORY_ROOT)),
            "manifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "baseDatasetSha256": base.dataset_sha256,
            "adversarialDatasetSha256": adversarial.dataset_sha256,
            "rows": len(corpus.rows),
        },
        "releaseCases": len(release.case_ids),
        "releasePayloadSha256": release.payload_sha256,
        "recipe": {
            "full": asdict(release_recipe("cuda", 8)),
            "headOnly": asdict(release_recipe("cuda", 8, freeze=True)),
        },
        "budgets": list(budgets),
        "target": "zero release-attested misses (the release gate's attested criterion)",
        "environment": environment(arguments.device),
        "regimes": {},
    }

    def save() -> None:
        (output / "results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )

    if "full" not in skipped:
        results["regimes"]["full"] = run_trainer_regime(
            "full",
            ir,
            base,
            adversarial,
            release,
            tokenizer,
            arguments.device,
            budgets,
            freeze=False,
        )
        save()
    if "head-only" not in skipped:
        results["regimes"]["head-only"] = run_trainer_regime(
            "head-only",
            ir,
            base,
            adversarial,
            release,
            tokenizer,
            arguments.device,
            budgets,
            freeze=True,
        )
        save()
    cache = EmbeddingCache(ir, corpus, release, config, tokenizer)
    log(f"cached: encoded {cache.row_count} rows in {cache.encode_seconds:.1f}s")
    if "cached" not in skipped:
        results["regimes"]["cached"] = run_cached_regime(
            cache, ir, base, adversarial, release, tokenizer, arguments.cached_epochs
        )
        save()
    if "laya" not in skipped:
        results["regimes"]["laya"] = run_laya_regime(
            cache,
            ir,
            base,
            adversarial,
            release,
            tokenizer,
            Path(arguments.laya_source),
            Path(arguments.laya_checkpoint),
            arguments.device,
            arguments.distill_epochs,
            arguments.cached_epochs,
            arguments.laya_batch_size,
        )
        save()
    results["completedAt"] = _utc_now()
    save()
    log(f"wrote {output / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
