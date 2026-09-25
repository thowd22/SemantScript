"""Train one application: a shared encoder, one adapter and one head per function.

Every function of an application is a ``(ir, corpus)`` pair. Joint training
fine-tunes the encoder, the adapter and every head over interleaved
per-function batches; adding a function later trains only its head while the
shared modules and the existing heads stay frozen and byte-identical. Each
function gets its own ``TrainingResult`` whose model is a per-function view of
the shared modules, so verification, lifecycle binding and export run one
function at a time exactly as for a single-function build.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from semantscript_trainer.adversarial import AdversarialDataset
from semantscript_trainer.dataset import TrainingDataset
from semantscript_trainer.teacher import NeuralFunctionIr
from semantscript_trainer.training import (
    _MAXIMUM_BATCH_PASSES,
    MAXIMUM_OPTIMIZATION_STEPS,
    EpochMetrics,
    TrainingConfig,
    TrainingConfigurationError,
    TrainingExecutionError,
    TrainingResult,
    _accuracy,
    _batch_loss,
    _build_heads,
    _build_sentence_encoder,
    _load_tokenizer,
    _prepare_rows,
    _PreparedRow,
    _require_torch,
    _resolve_device,
    _seed_torch,
    _tensorize_batch,
    _validate_logits,
    _validate_no_canonical_leakage,
    freeze_encoder_parameters,
    set_training_mode,
)
from semantscript_trainer.training_contract import (
    HeldOutSplitConfig,
    TrainingCorpus,
    TrainingSplit,
    assemble_training_corpus,
    derive_output_heads,
    split_training_corpus,
)

DEFAULT_ADAPTER_BOTTLENECK_SIZE = 64
MAXIMUM_APPLICATION_FUNCTIONS = 4096


@dataclass(frozen=True, slots=True)
class FunctionCorpus:
    """One function's IR and its identity-matched training corpus."""

    ir: NeuralFunctionIr
    corpus: TrainingCorpus

    def __post_init__(self) -> None:
        if not isinstance(self.ir, dict):
            raise TrainingConfigurationError("function IR must be an object")
        if not isinstance(self.corpus, TrainingCorpus):
            raise TrainingConfigurationError("function corpus must be a TrainingCorpus")
        if (
            self.ir.get("id") != self.corpus.function_id
            or self.ir.get("semanticSha256") != self.corpus.semantic_sha256
        ):
            raise TrainingConfigurationError("function corpus identity does not match its IR")
        try:
            heads = derive_output_heads(self.ir)
        except ValueError as error:
            raise TrainingConfigurationError(str(error)) from error
        if heads != self.corpus.output_heads:
            raise TrainingConfigurationError("function corpus heads do not match its IR output")
        if not isinstance(self.ir.get("inputs"), list):
            raise TrainingConfigurationError("function IR inputs must be an array")

    @property
    def adapter_ref(self) -> str:
        """The domain adapter this function's head reads (``model.adapter`` in the IR)."""

        return model_refs(self.ir)[1]

    @property
    def encoder_ref(self) -> str:
        return model_refs(self.ir)[0]

    @property
    def encoder_depth(self) -> int | None:
        """Shared encoder layers this function's domain runs; None is the full stack."""

        return model_refs(self.ir)[2]

    @property
    def function_id(self) -> str:
        return self.corpus.function_id


DEFAULT_ENCODER_REF = "encoder.main"
DEFAULT_ADAPTER_REF = "adapter.application"


def model_refs(ir: Mapping[str, Any]) -> tuple[str, str, int | None]:
    """``(encoder ref, adapter ref, encoder depth)`` of an IR record.

    A record without a ``model`` block (partial fixtures) binds the application
    defaults at the full depth.
    """

    model = ir.get("model")
    if model is None:
        return DEFAULT_ENCODER_REF, DEFAULT_ADAPTER_REF, None
    if not isinstance(model, Mapping):
        raise TrainingConfigurationError("function IR model must be an object")
    encoder = model.get("encoder", DEFAULT_ENCODER_REF)
    adapter = model.get("adapter", DEFAULT_ADAPTER_REF)
    if not isinstance(encoder, str) or not encoder or not isinstance(adapter, str) or not adapter:
        raise TrainingConfigurationError("function IR model refs must be nonempty strings")
    depth = model.get("encoderDepth")
    if depth is not None and (isinstance(depth, bool) or not isinstance(depth, int) or depth < 1):
        raise TrainingConfigurationError(
            f"function {ir.get('id')!r} declares an invalid encoderDepth {depth!r}"
        )
    return encoder, adapter, depth


def application_function(
    ir: NeuralFunctionIr,
    base: TrainingDataset,
    adversarial: AdversarialDataset | None = None,
    /,
) -> FunctionCorpus:
    """Assemble one function's datasets into a FunctionCorpus."""

    return FunctionCorpus(ir, assemble_training_corpus(ir, base, adversarial))


@dataclass(frozen=True, slots=True)
class ApplicationEpochMetrics:
    """One joint epoch: mean loss over every batch and held-out accuracy per function."""

    epoch: int
    mean_training_loss: float
    held_out_accuracy: Mapping[str, float]

    @property
    def mean_held_out_accuracy(self) -> float:
        values = list(self.held_out_accuracy.values())
        return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True, slots=True)
class ApplicationTrainingResult:
    """The shared application model plus one per-function training record."""

    model: Any
    config: TrainingConfig
    device: str
    adapter_bottleneck_size: int
    functions: Mapping[str, TrainingResult]
    metrics: tuple[ApplicationEpochMetrics, ...]
    selected_epoch: int | None = None

    @property
    def function_ids(self) -> tuple[str, ...]:
        return tuple(self.functions)


@dataclass(slots=True)
class _FunctionState:
    function: FunctionCorpus
    split: TrainingSplit
    prepared: dict[str, tuple[_PreparedRow, ...]]
    logit_count: int


def train_application(
    functions: Sequence[FunctionCorpus],
    /,
    *,
    config: TrainingConfig | None = None,
    tokenizer: Any | None = None,
    encoder: Any | None = None,
    adapter_bottleneck_size: int = DEFAULT_ADAPTER_BOTTLENECK_SIZE,
) -> ApplicationTrainingResult:
    """Jointly fine-tune the shared encoder, the adapter and one head per function."""

    resolved = _resolve_config(config)
    states = _function_states(functions, resolved)
    torch = _require_torch()
    application_module, _ = _load_application_modules()
    _seed_torch(torch, resolved.seed)
    device = _resolve_device(torch, resolved.device)
    tokenizer = _load_tokenizer(resolved) if tokenizer is None else tokenizer
    sentence_encoder = _build_sentence_encoder(resolved, encoder)
    domains = domain_depths(functions)
    adapters = {
        ref: _build_adapter(
            application_module, sentence_encoder.hidden_size, adapter_bottleneck_size
        )
        for ref in domains
    }
    heads = {
        state.function.function_id: _build_heads(
            state.function.corpus.output_heads, sentence_encoder.hidden_size, resolved
        )
        for state in states
    }
    try:
        model = application_module.SharedEncoderApplication(
            sentence_encoder,
            adapters=adapters,
            adapter_depths=domains,
            heads=heads,
            function_adapters={
                function.function_id: function.adapter_ref for function in functions
            },
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise TrainingExecutionError(f"could not construct application model: {error}") from error
    if resolved.freeze_encoder:
        freeze_encoder_parameters(model)
    metrics, selected_epoch = _fit(model, states, resolved, tokenizer, torch, device)
    return ApplicationTrainingResult(
        model=model,
        config=resolved,
        device=str(device),
        adapter_bottleneck_size=adapter_bottleneck_size,
        functions=_function_results(model, states, resolved, str(device), metrics, selected_epoch),
        metrics=metrics,
        selected_epoch=selected_epoch,
    )


def add_function_head(
    application: ApplicationTrainingResult,
    function: FunctionCorpus,
    /,
    *,
    config: TrainingConfig | None = None,
    tokenizer: Any | None = None,
) -> ApplicationTrainingResult:
    """Train one new head on the frozen shared encoder and its domain adapter.

    The shared modules and every existing head are byte-identical afterwards,
    so existing verification results and artifacts stay valid; only the new
    function's head is fitted. When the function belongs to a domain the
    application has no adapter for yet, a fresh adapter for that domain is
    attached and trained together with the head, still on the frozen encoder,
    so a new domain never moves another domain's weights.
    """

    if not isinstance(application, ApplicationTrainingResult):
        raise TrainingConfigurationError("application must be an ApplicationTrainingResult")
    resolved = application.config if config is None else _resolve_config(config)
    if resolved.canonical_input_version != application.config.canonical_input_version:
        raise TrainingConfigurationError(
            "a new head must use the application's canonical input version"
        )
    if function.function_id in application.functions:
        raise TrainingConfigurationError(
            f"application already trains function {function.function_id!r}"
        )
    states = _function_states([function], resolved)
    torch = _require_torch()
    device = _resolve_device(torch, resolved.device)
    _seed_torch(torch, resolved.seed)
    tokenizer = _load_tokenizer(resolved) if tokenizer is None else tokenizer
    model = application.model
    head = _build_heads(states[0].function.corpus.output_heads, model.hidden_size, resolved)
    trainable: list[Any] = list(head.parameters())
    try:
        if function.adapter_ref not in model.adapter_refs:
            application_module, _ = _load_application_modules()
            adapter = _build_adapter(
                application_module, model.hidden_size, application.adapter_bottleneck_size
            )
            model.add_adapter(function.adapter_ref, adapter, depth=function.encoder_depth)
            trainable.extend(adapter.parameters())
        elif model.adapter_depth(function.adapter_ref) != _normalized_depth(
            model, function.encoder_depth
        ):
            raise TrainingConfigurationError(
                f"function {function.function_id!r} declares encoderDepth "
                f"{function.encoder_depth!r} but its domain {function.adapter_ref!r} "
                f"trained at {model.adapter_depth(function.adapter_ref)!r}"
            )
        model.add_head(function.function_id, head, adapter_ref=function.adapter_ref)
    except (RuntimeError, TypeError, ValueError) as error:
        raise TrainingExecutionError(f"could not attach the new head: {error}") from error
    frozen = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and not any(parameter is candidate for candidate in trainable)
    ]
    for parameter in frozen:
        parameter.requires_grad_(False)
    try:
        metrics, selected_epoch = _fit(model, states, resolved, tokenizer, torch, device)
    finally:
        for parameter in frozen:
            parameter.requires_grad_(True)
    functions = dict(application.functions)
    functions.update(
        _function_results(model, states, resolved, str(device), metrics, selected_epoch)
    )
    return ApplicationTrainingResult(
        model=model,
        config=application.config,
        device=str(device),
        adapter_bottleneck_size=application.adapter_bottleneck_size,
        functions=functions,
        metrics=metrics,
        selected_epoch=selected_epoch,
    )


def domain_depths(functions: Sequence[FunctionCorpus]) -> dict[str, int | None]:
    """Adapter ref to encoder depth over ``functions``; a domain has one depth."""

    depths: dict[str, int | None] = {}
    for function in functions:
        depth = function.encoder_depth
        if function.adapter_ref in depths and depths[function.adapter_ref] != depth:
            raise TrainingConfigurationError(
                f"domain {function.adapter_ref!r} declares several encoder depths"
            )
        depths[function.adapter_ref] = depth
    return depths


def _normalized_depth(model: Any, depth: int | None) -> int | None:
    return cast(int | None, model.encoder.validate_depth(depth))


def _resolve_config(config: TrainingConfig | None) -> TrainingConfig:
    resolved = TrainingConfig() if config is None else config
    if not isinstance(resolved, TrainingConfig):
        raise TrainingConfigurationError("config must be a TrainingConfig")
    return resolved


def _load_application_modules() -> tuple[Any, Any]:
    try:
        from semantscript_model import application as application_module
        from semantscript_model.application import AdapterConfig
    except (ImportError, OSError) as error:
        raise TrainingExecutionError(
            "fine-tuning requires the optional training dependencies; install the training extra"
        ) from error
    return application_module, AdapterConfig


def _build_adapter(application_module: Any, hidden_size: int, bottleneck_size: int) -> Any:
    _, adapter_config = _load_application_modules()
    try:
        return application_module.ApplicationAdapter(
            adapter_config(hidden_size=hidden_size, bottleneck_size=bottleneck_size)
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise TrainingConfigurationError(f"could not construct the adapter: {error}") from error


def _function_states(
    functions: Sequence[FunctionCorpus], config: TrainingConfig
) -> list[_FunctionState]:
    if isinstance(functions, (str, bytes)) or not isinstance(functions, Sequence):
        raise TrainingConfigurationError("functions must be a sequence of FunctionCorpus")
    if not 1 <= len(functions) <= MAXIMUM_APPLICATION_FUNCTIONS:
        raise TrainingConfigurationError(
            f"an application trains between 1 and {MAXIMUM_APPLICATION_FUNCTIONS} functions"
        )
    states: list[_FunctionState] = []
    seen: set[str] = set()
    total_steps = 0
    total_passes = 0
    for function in functions:
        if not isinstance(function, FunctionCorpus):
            raise TrainingConfigurationError("functions must be FunctionCorpus instances")
        if function.function_id in seen:
            raise TrainingConfigurationError(
                f"function {function.function_id!r} appears more than once"
            )
        seen.add(function.function_id)
        split = split_training_corpus(
            function.corpus,
            HeldOutSplitConfig(evaluation_ratio=config.evaluation_ratio, seed=config.seed),
        )
        if not split.evaluation:
            raise TrainingConfigurationError(_single_group_message(function))
        schema = function.ir["inputs"]
        if not isinstance(schema, list):
            raise TrainingConfigurationError("function IR inputs must be an array")
        prepared = _prepare_rows(schema, split, config.canonical_input_version)
        _validate_no_canonical_leakage(prepared)
        steps = math.ceil(len(split.training) / config.batch_size)
        total_steps += steps * config.epochs
        total_passes += (
            steps + math.ceil(len(split.evaluation) / config.batch_size)
        ) * config.epochs
        states.append(
            _FunctionState(
                function=function,
                split=split,
                prepared=prepared,
                logit_count=function.corpus.logit_count,
            )
        )
    if total_steps > MAXIMUM_OPTIMIZATION_STEPS:
        raise TrainingConfigurationError(
            f"application training exceeds maximum optimization steps {MAXIMUM_OPTIMIZATION_STEPS}"
        )
    if total_passes > _MAXIMUM_BATCH_PASSES:
        raise TrainingConfigurationError(
            f"application training exceeds maximum batch passes {_MAXIMUM_BATCH_PASSES}"
        )
    return states


def _fit(
    model: Any,
    states: Sequence[_FunctionState],
    config: TrainingConfig,
    tokenizer: Any,
    torch: Any,
    device: Any,
) -> tuple[tuple[ApplicationEpochMetrics, ...], int | None]:
    try:
        model.to(device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise TrainingExecutionError(f"could not move application to {device}: {error}") from error
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise TrainingConfigurationError("application has no trainable parameters")
    if sum(parameter.numel() for parameter in parameters) > config.maximum_trainable_parameters:
        raise TrainingConfigurationError(
            "application exceeds maximum trainable parameter count "
            f"{config.maximum_trainable_parameters}"
        )
    try:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
            foreach=False,
        )
    except (RuntimeError, TypeError, ValueError) as error:
        raise TrainingExecutionError(f"could not construct AdamW optimizer: {error}") from error
    steps_per_epoch = sum(
        math.ceil(len(state.prepared["training"]) / config.batch_size) for state in states
    )
    scheduler = None
    if config.learning_rate_schedule == "linear":
        total_steps = max(steps_per_epoch * config.epochs, 1)
        warmup_steps = int(total_steps * float(config.warmup_ratio))

        def linear_factor(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            remaining = total_steps - step
            return max(remaining / max(total_steps - warmup_steps, 1), 0.0)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, linear_factor)

    metrics: list[ApplicationEpochMetrics] = []
    best_epoch: int | None = None
    best_state: dict[str, Any] | None = None
    for epoch in range(1, config.epochs + 1):
        batches: list[tuple[_FunctionState, list[_PreparedRow]]] = []
        for position, state in enumerate(states):
            rows = state.prepared["training"]
            order = list(range(len(rows)))
            random.Random(config.seed + epoch - 1 + 7919 * position).shuffle(order)
            for offset in range(0, len(order), config.batch_size):
                batches.append(
                    (state, [rows[index] for index in order[offset : offset + config.batch_size]])
                )
        # Interleave functions so the shared modules never see one function's
        # rows in a long run.
        random.Random(config.seed + 104729 * epoch).shuffle(batches)
        set_training_mode(model, config.freeze_encoder)
        loss_total = 0.0
        example_count = 0
        for state, batch in batches:
            input_ids, attention_mask, targets = _tensorize_batch(
                batch, tokenizer, torch, device, config.maximum_sequence_length
            )
            optimizer.zero_grad(set_to_none=True)
            try:
                embedding = model.embed(input_ids, attention_mask, state.function.function_id)
                logits = model.heads[state.function.function_id](embedding)
            except (RuntimeError, TypeError, ValueError) as error:
                raise TrainingExecutionError(f"application forward pass failed: {error}") from error
            _validate_logits(logits, torch, batch_size=len(batch), logit_count=state.logit_count)
            loss = _batch_loss(logits, targets, state.function.corpus.output_heads, config.loss)
            if not bool(torch.isfinite(loss).item()):
                raise TrainingExecutionError("training loss became non-finite")
            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(config.gradient_clip_norm), error_if_nonfinite=True
                )
                optimizer.step()
            except RuntimeError as error:
                raise TrainingExecutionError(f"optimization step failed: {error}") from error
            if scheduler is not None:
                scheduler.step()
            loss_total += float(loss.detach().cpu().item()) * len(batch)
            example_count += len(batch)
        held_out = {
            state.function.function_id: _accuracy(
                model.function_model(state.function.function_id),
                state.prepared["evaluation"],
                tokenizer,
                torch,
                device,
                config.maximum_sequence_length,
                config.batch_size,
                state.function.corpus.output_heads,
            )[0]
            for state in states
        }
        metrics.append(
            ApplicationEpochMetrics(
                epoch=epoch,
                mean_training_loss=loss_total / max(example_count, 1),
                held_out_accuracy=held_out,
            )
        )
        if config.select_best_epoch and (
            best_epoch is None
            or metrics[-1].mean_held_out_accuracy > metrics[best_epoch - 1].mean_held_out_accuracy
        ):
            best_epoch = epoch
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
    if config.select_best_epoch and best_state is not None and best_epoch != config.epochs:
        model.load_state_dict(best_state)
    model.eval()
    return tuple(metrics), best_epoch if config.select_best_epoch else None


def _function_results(
    model: Any,
    states: Sequence[_FunctionState],
    config: TrainingConfig,
    device: str,
    metrics: tuple[ApplicationEpochMetrics, ...],
    selected_epoch: int | None,
) -> dict[str, TrainingResult]:
    results: dict[str, TrainingResult] = {}
    for state in states:
        function_id = state.function.function_id
        results[function_id] = TrainingResult(
            model=model.function_model(function_id),
            head=state.function.corpus.head,
            split=state.split,
            config=config,
            device=device,
            metrics=tuple(
                EpochMetrics(
                    epoch=entry.epoch,
                    mean_training_loss=entry.mean_training_loss,
                    held_out_accuracy=entry.held_out_accuracy[function_id],
                )
                for entry in metrics
            ),
            function_id=function_id,
            semantic_sha256=state.function.corpus.semantic_sha256,
            base_dataset_sha256=state.function.corpus.base_dataset_sha256,
            adversarial_dataset_sha256=state.function.corpus.adversarial_dataset_sha256,
            selected_epoch=selected_epoch,
            output_heads=state.function.corpus.output_heads,
        )
    return results


__all__ = [
    "DEFAULT_ADAPTER_BOTTLENECK_SIZE",
    "MAXIMUM_APPLICATION_FUNCTIONS",
    "ApplicationEpochMetrics",
    "ApplicationTrainingResult",
    "FunctionCorpus",
    "add_function_head",
    "application_function",
    "domain_depths",
    "model_refs",
    "train_application",
]


def _single_group_message(function: FunctionCorpus) -> str:
    """Why nothing could be held out, and what to change (docs/diagnostics.md)."""

    from semantscript_trainer.teachers.constraints import describe_expression

    rows = function.corpus.rows
    distinct = len({json.dumps(row.inputs, sort_keys=True) for row in rows})
    return (
        f"{describe_expression(function.ir)} ({function.function_id}) needs at least two "
        f"independent row groups, but its {len(rows)} training rows ({distinct} distinct "
        "inputs) form one: identical inputs, and each case with its counterfactual twin, "
        "must stay on the same side of the held-out split, and in a small input space they "
        "link every row, the gold examples included. Lower --counterfactual-ratio (0.5, or "
        "0 when the inputs are a few booleans or enum values), request fewer --cases, or, "
        "with the constraints teacher, widen the number ranges in [teacher.ranges] "
        "(docs/diagnostics.md)"
    )
