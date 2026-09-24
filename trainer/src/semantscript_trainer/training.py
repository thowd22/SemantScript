"""Resource-bounded encoder and typed-head fine-tuning."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from semantscript_trainer.adversarial import AdversarialDataset
from semantscript_trainer.canonical_input import (
    CANONICAL_INPUT_ENCODINGS,
    CanonicalInputError,
    serialize_canonical_inputs,
)
from semantscript_trainer.dataset import TrainingDataset
from semantscript_trainer.teacher import NeuralFunctionIr
from semantscript_trainer.training_contract import (
    MAXIMUM_SPLIT_SEED,
    HeldOutSplitConfig,
    TrainingCorpus,
    TrainingHeadContract,
    TrainingRow,
    TrainingSplit,
    assemble_training_corpus,
    derive_training_head,
    split_training_corpus,
)

DEFAULT_ENCODER_NAME = "answerdotai/ModernBERT-base"
DEFAULT_ENCODER_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
MAXIMUM_EPOCH_COUNT = 100
MAXIMUM_BATCH_SIZE = 256
MAXIMUM_SEQUENCE_LENGTH = 8_192
MAXIMUM_BATCH_TOKENS = 65_536
MAXIMUM_OPTIMIZATION_STEPS = 100_000
MAXIMUM_CANONICAL_ROW_BYTES = 1 * 1024 * 1024
MAXIMUM_CORPUS_TEXT_BYTES = 64 * 1024 * 1024

# Bound evaluation as well as optimizer work. The head cap is checked before
# allocating its projection, and the trainable cap is checked before AdamW can
# allocate its two optimizer-state tensors per parameter.
_MAXIMUM_BATCH_PASSES = 200_000
_MAXIMUM_BATCH_CLASS_VALUES = 4_000_000
_MAXIMUM_HEAD_PARAMETER_COUNT = 16_000_000
_MAXIMUM_TRAINABLE_PARAMETER_COUNT = 350_000_000

type DeviceName = Literal["auto", "cpu", "cuda"]
type HeadArchitecture = Literal["linear", "mlp"]
type LossName = Literal["proper", "cross_entropy"]
type ScheduleName = Literal["constant", "linear"]

_IMMUTABLE_REVISION = re.compile(r"^[a-f0-9]{40}$")


class TrainingConfigurationError(ValueError):
    """A fine-tuning request is unsafe, inconsistent, or unsupported."""


class TrainingExecutionError(RuntimeError):
    """A configured fine-tuning run cannot execute safely."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Bounded fine-tuning and held-out evaluation controls."""

    encoder_name: str = DEFAULT_ENCODER_NAME
    encoder_revision: str = DEFAULT_ENCODER_REVISION
    local_files_only: bool = False
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    maximum_sequence_length: int = 512
    evaluation_ratio: float = 0.2
    seed: int = 1
    device: DeviceName = "auto"
    head_architecture: HeadArchitecture = "linear"
    mlp_hidden_size: int | None = None
    loss: LossName = "proper"
    # Keep the epoch with the best held-out accuracy instead of the last one. The
    # selection uses only the calibration split, never attested or benchmark data.
    select_best_epoch: bool = False
    # "linear" warms the learning rate up over warmup_ratio of the steps and then
    # decays it linearly to zero, which damps last-epoch noise; "constant" keeps
    # the historical behavior.
    learning_rate_schedule: ScheduleName = "constant"
    warmup_ratio: float = 0.0
    # Which canonical-input encoding the rows are serialized with (1: exact-JSON
    # envelope, 2: compact text). Recorded in the verified IR and the artifact so
    # the runtime serializes calls the same way.
    canonical_input_version: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.canonical_input_version, bool)
            or self.canonical_input_version not in CANONICAL_INPUT_ENCODINGS
        ):
            raise TrainingConfigurationError("canonical_input_version must be 1 or 2")
        if not isinstance(self.select_best_epoch, bool):
            raise TrainingConfigurationError("select_best_epoch must be a boolean")
        if self.learning_rate_schedule not in ("constant", "linear"):
            raise TrainingConfigurationError("learning_rate_schedule must be constant or linear")
        if (
            isinstance(self.warmup_ratio, bool)
            or not isinstance(self.warmup_ratio, (int, float))
            or not math.isfinite(float(self.warmup_ratio))
            or not 0 <= float(self.warmup_ratio) <= 0.5
        ):
            raise TrainingConfigurationError("warmup_ratio must be a finite number from 0 to 0.5")
        if not isinstance(self.encoder_name, str) or not self.encoder_name:
            raise TrainingConfigurationError("encoder_name must be nonempty")
        if (
            not isinstance(self.encoder_revision, str)
            or _IMMUTABLE_REVISION.fullmatch(self.encoder_revision) is None
        ):
            raise TrainingConfigurationError(
                "encoder_revision must be an immutable revision: a 40-character lowercase commit SHA"
            )
        _bounded_integer("epochs", self.epochs, minimum=1, maximum=MAXIMUM_EPOCH_COUNT)
        _bounded_integer("batch_size", self.batch_size, minimum=1, maximum=MAXIMUM_BATCH_SIZE)
        _bounded_integer(
            "maximum_sequence_length",
            self.maximum_sequence_length,
            minimum=1,
            maximum=MAXIMUM_SEQUENCE_LENGTH,
        )
        if self.batch_size * self.maximum_sequence_length > MAXIMUM_BATCH_TOKENS:
            raise TrainingConfigurationError(
                f"batch_size * maximum_sequence_length must not exceed {MAXIMUM_BATCH_TOKENS}"
            )
        _positive_finite("learning_rate", self.learning_rate)
        _nonnegative_finite("weight_decay", self.weight_decay)
        _positive_finite("gradient_clip_norm", self.gradient_clip_norm)
        try:
            HeldOutSplitConfig(evaluation_ratio=self.evaluation_ratio, seed=self.seed)
        except OverflowError as error:
            raise TrainingConfigurationError(
                "evaluation_ratio must be a finite number between 0 and 1"
            ) from error
        except ValueError as error:
            raise TrainingConfigurationError(str(error)) from error
        if self.seed > MAXIMUM_SPLIT_SEED:
            raise TrainingConfigurationError(
                f"seed must be an integer from 0 through {MAXIMUM_SPLIT_SEED}"
            )
        if self.device not in ("auto", "cpu", "cuda"):
            raise TrainingConfigurationError("device must be 'auto', 'cpu', or 'cuda'")
        if self.head_architecture not in ("linear", "mlp"):
            raise TrainingConfigurationError("head_architecture must be 'linear' or 'mlp'")
        if self.head_architecture == "linear" and self.mlp_hidden_size is not None:
            raise TrainingConfigurationError("linear heads cannot set mlp_hidden_size")
        if self.head_architecture == "mlp" and self.mlp_hidden_size is not None:
            _bounded_integer(
                "mlp_hidden_size",
                self.mlp_hidden_size,
                minimum=1,
                maximum=8_192,
            )
        if self.loss not in ("proper", "cross_entropy"):
            raise TrainingConfigurationError("loss must be 'proper' or 'cross_entropy'")
        if not isinstance(self.local_files_only, bool):
            raise TrainingConfigurationError("local_files_only must be a boolean")


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Scalar metrics retained without keeping computation graphs alive."""

    epoch: int
    mean_training_loss: float
    held_out_accuracy: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """A fitted classifier and its reproducible training/evaluation record."""

    model: Any
    head: TrainingHeadContract
    split: TrainingSplit
    config: TrainingConfig
    device: str
    metrics: tuple[EpochMetrics, ...]
    function_id: str
    semantic_sha256: str
    base_dataset_sha256: str
    adversarial_dataset_sha256: str | None
    selected_epoch: int | None = None

    @property
    def held_out_accuracy(self) -> float:
        if self.selected_epoch is not None:
            return self.metrics[self.selected_epoch - 1].held_out_accuracy
        return self.metrics[-1].held_out_accuracy

    @property
    def training_row_count(self) -> int:
        return len(self.split.training)

    @property
    def held_out_row_count(self) -> int:
        return len(self.split.evaluation)


@dataclass(frozen=True, slots=True)
class _PreparedRow:
    row: TrainingRow
    text: str


def train_classifier(
    ir: NeuralFunctionIr,
    base: TrainingDataset,
    adversarial: AdversarialDataset | None = None,
    /,
    *,
    config: TrainingConfig | None = None,
    tokenizer: Any | None = None,
    encoder: Any | None = None,
) -> TrainingResult:
    """Assemble generated rows and fine-tune one encoder plus IR-derived head."""

    corpus = assemble_training_corpus(ir, base, adversarial)
    return train_corpus(ir, corpus, config=config, tokenizer=tokenizer, encoder=encoder)


def train_corpus(
    ir: NeuralFunctionIr,
    corpus: TrainingCorpus,
    /,
    *,
    config: TrainingConfig | None = None,
    tokenizer: Any | None = None,
    encoder: Any | None = None,
) -> TrainingResult:
    """Fine-tune one classifier from an already validated training corpus."""

    if not isinstance(ir, dict):
        raise TrainingConfigurationError("IR must be an object")
    if not isinstance(corpus, TrainingCorpus):
        raise TrainingConfigurationError("corpus must be a TrainingCorpus")
    resolved = TrainingConfig() if config is None else config
    if not isinstance(resolved, TrainingConfig):
        raise TrainingConfigurationError("config must be a TrainingConfig")
    if ir.get("id") != corpus.function_id or ir.get("semanticSha256") != corpus.semantic_sha256:
        raise TrainingConfigurationError("training corpus identity does not match the IR")
    try:
        ir_head = derive_training_head(ir)
    except ValueError as error:
        raise TrainingConfigurationError(str(error)) from error
    if ir_head != corpus.head:
        raise TrainingConfigurationError("training corpus head does not match the IR output spec")
    raw_schema = ir.get("inputs")
    if not isinstance(raw_schema, list):
        raise TrainingConfigurationError("IR inputs must be an array")

    split = split_training_corpus(
        corpus,
        HeldOutSplitConfig(
            evaluation_ratio=resolved.evaluation_ratio,
            seed=resolved.seed,
        ),
    )
    if not split.evaluation:
        raise TrainingConfigurationError(
            "held-out accuracy requires at least two independent row groups"
        )
    if resolved.batch_size * len(corpus.head.support) > _MAXIMUM_BATCH_CLASS_VALUES:
        raise TrainingConfigurationError(
            "batch_size * head cardinality exceeds maximum class values "
            f"{_MAXIMUM_BATCH_CLASS_VALUES}"
        )
    steps_per_epoch = math.ceil(len(split.training) / resolved.batch_size)
    if steps_per_epoch * resolved.epochs > MAXIMUM_OPTIMIZATION_STEPS:
        raise TrainingConfigurationError(
            f"training run exceeds maximum optimization steps {MAXIMUM_OPTIMIZATION_STEPS}"
        )
    evaluation_steps = math.ceil(len(split.evaluation) / resolved.batch_size)
    total_batch_passes = (steps_per_epoch + evaluation_steps) * resolved.epochs
    if total_batch_passes > _MAXIMUM_BATCH_PASSES:
        raise TrainingConfigurationError(
            f"training and evaluation exceed maximum batch passes {_MAXIMUM_BATCH_PASSES}"
        )

    prepared = _prepare_rows(raw_schema, split, resolved.canonical_input_version)
    _validate_no_canonical_leakage(prepared)
    torch = _require_torch()
    _seed_torch(torch, resolved.seed)
    device = _resolve_device(torch, resolved.device)
    tokenizer = _load_tokenizer(resolved) if tokenizer is None else tokenizer
    model = _build_model(corpus.head, resolved, encoder)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise TrainingConfigurationError("classifier has no trainable parameters")
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count > _MAXIMUM_TRAINABLE_PARAMETER_COUNT:
        raise TrainingConfigurationError(
            "classifier exceeds maximum trainable parameter count "
            f"{_MAXIMUM_TRAINABLE_PARAMETER_COUNT}"
        )
    try:
        model.to(device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise TrainingExecutionError(f"could not move classifier to {device}: {error}") from error
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise TrainingConfigurationError("classifier has no trainable parameters after placement")

    try:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(resolved.learning_rate),
            weight_decay=float(resolved.weight_decay),
            foreach=False,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise TrainingExecutionError(f"could not construct AdamW optimizer: {error}") from error
    scheduler = None
    if resolved.learning_rate_schedule == "linear":
        total_steps = max(steps_per_epoch * resolved.epochs, 1)
        warmup_steps = int(total_steps * float(resolved.warmup_ratio))

        def linear_factor(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            remaining = total_steps - step
            return max(remaining / max(total_steps - warmup_steps, 1), 0.0)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, linear_factor)
    metrics: list[EpochMetrics] = []
    training_rows = prepared["training"]
    evaluation_rows = prepared["evaluation"]
    best_epoch: int | None = None
    best_state: dict[str, Any] | None = None
    for epoch in range(1, resolved.epochs + 1):
        order = list(range(len(training_rows)))
        random.Random(resolved.seed + epoch - 1).shuffle(order)
        model.train()
        loss_total = 0.0
        example_count = 0
        for offset in range(0, len(order), resolved.batch_size):
            batch = [training_rows[index] for index in order[offset : offset + resolved.batch_size]]
            input_ids, attention_mask, targets = _tensorize_batch(
                batch,
                tokenizer,
                torch,
                device,
                resolved.maximum_sequence_length,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = _forward_model(model, input_ids, attention_mask)
            _validate_logits(
                logits,
                torch,
                batch_size=len(batch),
                logit_count=corpus.head.logit_count,
            )
            loss = _classification_loss(
                logits,
                targets,
                ordinal=corpus.head.ordinal,
                loss_name=resolved.loss,
            )
            if not bool(torch.isfinite(loss).item()):
                raise TrainingExecutionError("training loss became non-finite")
            try:
                loss.backward()
            except RuntimeError as error:
                raise TrainingExecutionError(f"backpropagation failed: {error}") from error
            try:
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    float(resolved.gradient_clip_norm),
                    error_if_nonfinite=True,
                )
            except RuntimeError as error:
                raise TrainingExecutionError(
                    f"training gradients became non-finite: {error}"
                ) from error
            try:
                optimizer.step()
            except RuntimeError as error:
                raise TrainingExecutionError(f"optimizer step failed: {error}") from error
            if scheduler is not None:
                scheduler.step()
            size = len(batch)
            loss_total += float(loss.detach().cpu().item()) * size
            example_count += size

        held_out_accuracy = _accuracy(
            model,
            evaluation_rows,
            tokenizer,
            torch,
            device,
            resolved.maximum_sequence_length,
            resolved.batch_size,
            corpus.head.logit_count,
        )
        metrics.append(
            EpochMetrics(
                epoch=epoch,
                mean_training_loss=loss_total / example_count,
                held_out_accuracy=held_out_accuracy,
            )
        )
        if resolved.select_best_epoch and (
            best_epoch is None or held_out_accuracy > metrics[best_epoch - 1].held_out_accuracy
        ):
            best_epoch = epoch
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }

    if resolved.select_best_epoch and best_state is not None and best_epoch != resolved.epochs:
        try:
            model.load_state_dict(best_state)
        except RuntimeError as error:
            raise TrainingExecutionError(f"could not restore the best epoch: {error}") from error

    return TrainingResult(
        model=model,
        head=corpus.head,
        split=split,
        config=resolved,
        device=str(device),
        metrics=tuple(metrics),
        function_id=corpus.function_id,
        semantic_sha256=corpus.semantic_sha256,
        base_dataset_sha256=corpus.base_dataset_sha256,
        adversarial_dataset_sha256=corpus.adversarial_dataset_sha256,
        selected_epoch=best_epoch if resolved.select_best_epoch else None,
    )


def _prepare_rows(
    schema: Sequence[Mapping[str, Any]],
    split: TrainingSplit,
    version: int = 1,
) -> dict[str, tuple[_PreparedRow, ...]]:
    byte_count = 0

    def prepare(rows: tuple[TrainingRow, ...]) -> tuple[_PreparedRow, ...]:
        nonlocal byte_count
        result: list[_PreparedRow] = []
        for row in rows:
            try:
                encoded = serialize_canonical_inputs(
                    schema,
                    row.inputs,
                    version=version,
                    maximum_bytes=MAXIMUM_CANONICAL_ROW_BYTES,
                )
            except (CanonicalInputError, TypeError, ValueError) as error:
                raise TrainingConfigurationError(
                    f"training row {row.row_id!r} cannot be canonically encoded: {error}"
                ) from error
            byte_count += len(encoded)
            if byte_count > MAXIMUM_CORPUS_TEXT_BYTES:
                raise TrainingConfigurationError(
                    f"canonical training text exceeds maximum {MAXIMUM_CORPUS_TEXT_BYTES} bytes"
                )
            result.append(_PreparedRow(row=row, text=encoded.decode("utf-8", errors="strict")))
        return tuple(result)

    return {"training": prepare(split.training), "evaluation": prepare(split.evaluation)}


def _validate_no_canonical_leakage(
    prepared: Mapping[str, tuple[_PreparedRow, ...]],
) -> None:
    seen: dict[str, tuple[int, str, str]] = {}
    for partition in ("training", "evaluation"):
        for entry in prepared[partition]:
            previous = seen.get(entry.text)
            if previous is None:
                seen[entry.text] = (entry.row.label_index, partition, entry.row.row_id)
                continue
            previous_label, previous_partition, previous_row_id = previous
            if previous_label != entry.row.label_index:
                raise TrainingConfigurationError(
                    "canonical input has conflicting labels in rows "
                    f"{previous_row_id!r} and {entry.row.row_id!r}"
                )
            if previous_partition != partition:
                raise TrainingConfigurationError(
                    "canonical input appears in both training and held-out rows: "
                    f"{previous_row_id!r} and {entry.row.row_id!r}"
                )


def _build_model(head: TrainingHeadContract, config: TrainingConfig, encoder: Any | None) -> Any:
    try:
        from semantscript_model.classifier import SemanticClassifier
    except (ImportError, OSError) as error:
        raise TrainingExecutionError(
            "fine-tuning requires the optional training dependencies; install the training extra"
        ) from error
    sentence_encoder = _build_sentence_encoder(config, encoder)
    model_head = _build_head(head, sentence_encoder.hidden_size, config)
    try:
        return SemanticClassifier(sentence_encoder, model_head)
    except (ImportError, RuntimeError, TypeError, ValueError) as error:
        raise TrainingExecutionError(f"could not construct classifier: {error}") from error


def _build_sentence_encoder(config: TrainingConfig, encoder: Any | None) -> Any:
    try:
        from semantscript_model.encoder import EncoderConfig, SentenceEncoder
    except (ImportError, OSError) as error:
        raise TrainingExecutionError(
            "fine-tuning requires the optional training dependencies; install the training extra"
        ) from error
    try:
        return SentenceEncoder(
            EncoderConfig(
                model_name=config.encoder_name,
                revision=config.encoder_revision,
                local_files_only=config.local_files_only,
                trust_remote_code=False,
            ),
            encoder=encoder,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise TrainingExecutionError(f"could not construct sentence encoder: {error}") from error


def _build_head(head: TrainingHeadContract, input_size: int, config: TrainingConfig) -> Any:
    try:
        from semantscript_model.heads import ClassificationHead, HeadConfig
    except (ImportError, OSError) as error:
        raise TrainingExecutionError(
            "fine-tuning requires the optional training dependencies; install the training extra"
        ) from error
    head_parameter_count = _head_parameter_count(input_size, head, config)
    if head_parameter_count > _MAXIMUM_HEAD_PARAMETER_COUNT:
        raise TrainingConfigurationError(
            f"classification head exceeds maximum parameter count {_MAXIMUM_HEAD_PARAMETER_COUNT}"
        )
    try:
        return ClassificationHead(
            HeadConfig(
                input_size=input_size,
                kind=head.parameterization,
                cardinality=len(head.support),
                architecture=config.head_architecture,
                mlp_hidden_size=config.mlp_hidden_size,
            )
        )
    except (ImportError, RuntimeError, TypeError, ValueError) as error:
        raise TrainingExecutionError(f"could not construct classification head: {error}") from error


def _head_parameter_count(
    input_size: int,
    head: TrainingHeadContract,
    config: TrainingConfig,
) -> int:
    output_size = head.logit_count
    if config.head_architecture == "linear":
        return input_size * output_size + output_size
    hidden_size = config.mlp_hidden_size or input_size
    return input_size * hidden_size + hidden_size + hidden_size * output_size + output_size


def _load_tokenizer(config: TrainingConfig) -> Any:
    try:
        from transformers import AutoTokenizer
    except (ImportError, OSError) as error:
        raise TrainingExecutionError(
            "fine-tuning requires Transformers; install the project training extra"
        ) from error
    try:
        return AutoTokenizer.from_pretrained(
            config.encoder_name,
            revision=config.encoder_revision,
            local_files_only=config.local_files_only,
            trust_remote_code=False,
            use_fast=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingExecutionError(f"could not load the configured tokenizer: {error}") from error


def _tensorize_batch(
    batch: Sequence[_PreparedRow],
    tokenizer: Any,
    torch: Any,
    device: Any,
    maximum_sequence_length: int,
) -> tuple[Any, Any, Any]:
    texts = [entry.text for entry in batch]
    try:
        tokenized = tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=maximum_sequence_length,
            return_tensors="pt",
        )
    except Exception as error:
        raise TrainingExecutionError(f"tokenization failed: {error}") from error
    if not isinstance(tokenized, Mapping):
        raise TrainingExecutionError("tokenizer must return a mapping of tensors")
    input_ids = tokenized.get("input_ids")
    attention_mask = tokenized.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise TrainingExecutionError("tokenizer must return input_ids and attention_mask tensors")
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise TrainingExecutionError(
            "tokenizer tensors must have matching [batch, sequence] shapes"
        )
    if input_ids.shape[0] != len(batch) or input_ids.shape[1] > maximum_sequence_length:
        raise TrainingExecutionError("tokenizer returned tensors outside the configured bounds")
    if input_ids.shape[1] == 0:
        raise TrainingExecutionError("tokenizer returned an empty sequence")
    if input_ids.dtype != torch.long or attention_mask.dtype != torch.long:
        raise TrainingExecutionError("tokenizer input_ids and attention_mask must use int64")
    if bool(((attention_mask != 0) & (attention_mask != 1)).any().item()):
        raise TrainingExecutionError("tokenizer attention_mask must contain only zero and one")
    if bool((attention_mask.sum(dim=1) == 0).any().item()):
        raise TrainingExecutionError("every tokenized row must contain at least one attended token")
    targets = torch.tensor(
        [entry.row.label_index for entry in batch],
        dtype=torch.long,
    )
    return input_ids.to(device), attention_mask.to(device), targets.to(device)


def _classification_loss(
    logits: Any,
    targets: Any,
    *,
    ordinal: bool,
    loss_name: LossName,
) -> Any:
    try:
        from semantscript_model.losses import classification_loss
    except ImportError as error:
        raise TrainingExecutionError(
            "fine-tuning requires the optional PyTorch training dependency"
        ) from error
    return classification_loss(logits, targets, ordinal=ordinal, loss=loss_name)


def _accuracy(
    model: Any,
    rows: tuple[_PreparedRow, ...],
    tokenizer: Any,
    torch: Any,
    device: Any,
    maximum_sequence_length: int,
    batch_size: int,
    logit_count: int,
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            input_ids, attention_mask, targets = _tensorize_batch(
                batch,
                tokenizer,
                torch,
                device,
                maximum_sequence_length,
            )
            logits = _forward_model(model, input_ids, attention_mask)
            _validate_logits(
                logits,
                torch,
                batch_size=len(batch),
                logit_count=logit_count,
            )
            if logit_count == 1:
                # Runtime ABI ties select the earlier support member (false).
                predictions = (logits[:, 0] > 0).to(dtype=torch.long)
            else:
                predictions = torch.argmax(logits, dim=-1)
            correct += int((predictions == targets).sum().detach().cpu().item())
    return correct / len(rows)


def _forward_model(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask)
    except (RuntimeError, TypeError, ValueError) as error:
        raise TrainingExecutionError(f"classifier forward pass failed: {error}") from error


def _validate_logits(
    logits: Any,
    torch: Any,
    *,
    batch_size: int,
    logit_count: int,
) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TrainingExecutionError("classifier must return a tensor of logits")
    if logits.ndim != 2 or logits.shape != (batch_size, logit_count):
        raise TrainingExecutionError(
            f"classifier logits must have shape [{batch_size}, {logit_count}]"
        )
    if not torch.is_floating_point(logits):
        raise TrainingExecutionError("classifier logits must use a floating-point dtype")
    if not bool(torch.isfinite(logits).all().item()):
        raise TrainingExecutionError("classifier logits became non-finite")


def _require_torch() -> Any:
    try:
        import torch
    except (ImportError, OSError) as error:
        raise TrainingExecutionError(
            "fine-tuning requires PyTorch; install the project training extra"
        ) from error
    return torch


def _resolve_device(torch: Any, requested: DeviceName) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise TrainingExecutionError("CUDA/ROCm device requested but torch.cuda is unavailable")
    return torch.device(requested)


def _seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bounded_integer(name: str, value: object, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise TrainingConfigurationError(
            f"{name} must be an integer from {minimum} through {maximum}"
        )


def _positive_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigurationError(f"{name} must be a positive finite number")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, ValueError):
        finite = False
    if not finite or value <= 0:
        raise TrainingConfigurationError(f"{name} must be a positive finite number")


def _nonnegative_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigurationError(f"{name} must be a nonnegative finite number")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, ValueError):
        finite = False
    if not finite or value < 0:
        raise TrainingConfigurationError(f"{name} must be a nonnegative finite number")


__all__ = [
    "DEFAULT_ENCODER_NAME",
    "DEFAULT_ENCODER_REVISION",
    "MAXIMUM_BATCH_SIZE",
    "MAXIMUM_BATCH_TOKENS",
    "MAXIMUM_CANONICAL_ROW_BYTES",
    "MAXIMUM_CORPUS_TEXT_BYTES",
    "MAXIMUM_EPOCH_COUNT",
    "MAXIMUM_OPTIMIZATION_STEPS",
    "MAXIMUM_SEQUENCE_LENGTH",
    "DeviceName",
    "EpochMetrics",
    "HeadArchitecture",
    "LossName",
    "TrainingConfig",
    "TrainingConfigurationError",
    "TrainingExecutionError",
    "TrainingResult",
    "train_classifier",
    "train_corpus",
]
