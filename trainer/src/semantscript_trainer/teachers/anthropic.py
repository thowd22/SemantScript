"""Anthropic Messages and Message Batches teacher backend."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import math
import re
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from semantscript_trainer.adversarial_contract import (
    build_boundary_pair_schema,
    build_counterfactual_schema,
    parse_boundary_pair_response,
    parse_counterfactual_response,
)
from semantscript_trainer.adversarial_prompt import (
    build_boundary_messages,
    build_counterfactual_messages,
)
from semantscript_trainer.case_contract import (
    build_case_schema,
    parse_case_response,
    validate_case_count,
)
from semantscript_trainer.teacher import (
    MAXIMUM_TEACHER_RESPONSE_BYTES,
    BoundaryPairProposal,
    CounterfactualProposal,
    GeneratedCase,
    TeacherBatchError,
    TeacherBatchTimeout,
    TeacherBudgetExceeded,
    TeacherConfigurationError,
    TeacherDescriptor,
    TeacherResponseError,
    TeacherTransportError,
)
from semantscript_trainer.teacher_config import TeacherConfig
from semantscript_trainer.teacher_prompt import build_case_messages
from semantscript_trainer.teacher_spend import ResponseJournal, SpendMeter

_FUNCTION_ID = re.compile(r"^nf_[a-f0-9]{64}$")
_CUSTOM_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_REQUEST_DIGEST_DOMAIN = b"semantscript-anthropic-batch-request-v1\x00"


@dataclass(frozen=True, slots=True)
class BatchHandle:
    """Submitted batch plus the identities needed for safe later collection."""

    batch_id: str
    custom_ids: tuple[str, ...]
    function_id: str
    configuration_sha256: str
    request_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.batch_id, str) or not self.batch_id.strip():
            raise TeacherBatchError("batch handle requires a nonempty batch ID")
        if not isinstance(self.custom_ids, tuple) or not self.custom_ids:
            raise TeacherBatchError("batch handle requires at least one custom ID")
        if len(self.custom_ids) > 100_000:
            raise TeacherBatchError("batch handle exceeds the maximum of 100000 custom IDs")
        if any(
            not isinstance(value, str) or _CUSTOM_ID.fullmatch(value) is None
            for value in self.custom_ids
        ):
            raise TeacherBatchError("batch handle contains an invalid custom ID")
        if len(set(self.custom_ids)) != len(self.custom_ids):
            raise TeacherBatchError("batch handle contains duplicate custom IDs")
        if (
            not isinstance(self.function_id, str)
            or _FUNCTION_ID.fullmatch(self.function_id) is None
        ):
            raise TeacherBatchError("batch handle contains an invalid function ID")
        if (
            not isinstance(self.configuration_sha256, str)
            or _SHA256.fullmatch(self.configuration_sha256) is None
        ):
            raise TeacherBatchError("batch handle contains an invalid configuration digest")
        if (
            not isinstance(self.request_sha256, str)
            or _SHA256.fullmatch(self.request_sha256) is None
        ):
            raise TeacherBatchError("batch handle contains an invalid request digest")


class AnthropicTeacher:
    """Generate exactly one schema-constrained case per Anthropic request."""

    def __init__(
        self,
        config: TeacherConfig,
        *,
        client: Any | None = None,
        schema_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        meter: SpendMeter | None = None,
        journal: ResponseJournal | None = None,
    ) -> None:
        """``meter`` charges every request and enforces the run's spend cap before it is
        sent; ``journal`` keeps every paid direct response and replays it for free when
        the identical request comes again (a rerun after a stop)."""
        if config.backend != "anthropic":
            raise TeacherConfigurationError(
                f"AnthropicTeacher requires backend='anthropic', got {config.backend!r}"
            )
        self._config = config
        self._client = client
        self._schema_transform = schema_transform
        self._clock = clock
        self._sleeper = sleeper
        self._meter = meter
        self._journal = journal
        self._last_journal_key: str | None = None

    @property
    def descriptor(self) -> TeacherDescriptor:
        return TeacherDescriptor(
            provider="anthropic",
            model=self._config.model,
            configuration_sha256=self._config.configuration_sha256,
        )

    def generate(self, ir: Mapping[str, Any], n: int) -> tuple[GeneratedCase, ...]:
        """Generate ``n`` validated cases using the configured submission mode."""

        _validate_count(n)
        if n == 0:
            return ()

        mode = self._config.mode
        if mode == "auto":
            mode = "batch" if n >= self._config.batch_threshold else "direct"
        if mode == "direct":
            return self._generate_direct(ir, n)
        if mode == "batch":
            return self._generate_batch(ir, n)
        raise TeacherConfigurationError(f"unsupported Anthropic teacher mode {mode!r}")

    def generate_boundary_pair(
        self,
        ir: Mapping[str, Any],
        constraint_index: int,
        /,
    ) -> BoundaryPairProposal:
        try:
            system, user = build_boundary_messages(dict(ir), constraint_index)
            schema = build_boundary_pair_schema(dict(ir))
        except TeacherConfigurationError:
            raise
        except Exception as error:
            raise TeacherConfigurationError(
                f"could not build Anthropic boundary request: {error}"
            ) from error
        text = self._request_adversarial(schema, system, user, "boundary")
        with self._discard_rejected():
            return parse_boundary_pair_response(dict(ir), text)

    def generate_counterfactual(
        self,
        ir: Mapping[str, Any],
        anchor: GeneratedCase,
        /,
    ) -> CounterfactualProposal:
        try:
            system, user = build_counterfactual_messages(dict(ir), anchor)
            schema = build_counterfactual_schema(dict(ir))
        except TeacherConfigurationError:
            raise
        except Exception as error:
            raise TeacherConfigurationError(
                f"could not build Anthropic counterfactual request: {error}"
            ) from error
        text = self._request_adversarial(schema, system, user, "counterfactual")
        with self._discard_rejected():
            return parse_counterfactual_response(dict(ir), text)

    def submit_batch(self, ir: Mapping[str, Any], n: int) -> BatchHandle:
        """Submit one non-streaming Messages request for each desired case."""

        _validate_positive_count(n)
        requests, custom_ids, request_sha256 = self._batch_requests(ir, n)
        return self._submit_requests(ir, requests, custom_ids, request_sha256)

    def _batch_requests(
        self, ir: Mapping[str, Any], n: int
    ) -> tuple[list[dict[str, Any]], tuple[str, ...], str]:
        wire_schema = self._wire_schema(ir)
        custom_ids = self._custom_ids(ir, n)
        request_digest = hashlib.sha256(_REQUEST_DIGEST_DOMAIN)
        requests: list[dict[str, Any]] = []
        for index, custom_id in enumerate(custom_ids):
            request = {
                "custom_id": custom_id,
                "params": self._request_params(ir, index, n, wire_schema),
            }
            _update_request_digest(request_digest, request)
            requests.append(request)
        return requests, custom_ids, request_digest.hexdigest()

    def _submit_requests(
        self,
        ir: Mapping[str, Any],
        requests: list[dict[str, Any]],
        custom_ids: tuple[str, ...],
        request_sha256: str,
    ) -> BatchHandle:
        if self._meter is not None:
            self._meter.reserve(
                sum(
                    self._meter.estimate_request_usd(_prompt_characters(item["params"]), batch=True)
                    for item in requests
                ),
                requests=len(requests),
            )
        try:
            batch = self._get_client().messages.batches.create(requests=requests)
        except TeacherConfigurationError:
            raise
        except Exception as error:
            raise _transport_error("failed to submit Anthropic message batch", error) from error

        batch_id = _field(batch, "id")
        if not isinstance(batch_id, str) or not batch_id:
            raise TeacherBatchError("Anthropic batch submission returned an invalid batch ID")
        return BatchHandle(
            batch_id=batch_id,
            custom_ids=custom_ids,
            function_id=_function_id(ir),
            configuration_sha256=self._config.configuration_sha256,
            request_sha256=request_sha256,
        )

    def _generate_batch(self, ir: Mapping[str, Any], n: int) -> tuple[GeneratedCase, ...]:
        """Submit (or, after a stop, resume) one batch and collect it.

        With a journal the batch handle is recorded as soon as the batch is submitted,
        so a rerun of a run stopped while the batch was processing collects the batch
        it already paid for instead of submitting a new one. A batch whose results the
        run rejects is forgotten, so the rerun submits it again.
        """

        _validate_positive_count(n)
        requests, custom_ids, request_sha256 = self._batch_requests(ir, n)
        journal = self._journal
        key = journal.next_batch_key(request_sha256) if journal is not None else None
        stored = journal.load_batch(key) if journal is not None and key is not None else None
        handle: BatchHandle | None = None
        replay = False
        if stored is not None:
            try:
                handle = BatchHandle(
                    batch_id=stored["batch_id"],
                    custom_ids=tuple(stored["custom_ids"]),
                    function_id=_function_id(ir),
                    configuration_sha256=self._config.configuration_sha256,
                    request_sha256=request_sha256,
                )
            except TeacherBatchError:
                handle = None
            if handle is not None and handle.custom_ids != custom_ids:
                handle = None
            replay = handle is not None and stored.get("collected") is True
        for attempt in (1, 2):
            resumed = handle is not None
            if handle is None:
                handle = self._submit_requests(ir, requests, custom_ids, request_sha256)
                replay = False
                if journal is not None and key is not None:
                    journal.store_batch(key, handle.batch_id, handle.custom_ids, collected=False)
            try:
                cases = self._collect(ir, handle, replay=replay)
            except TeacherBatchTimeout:
                raise
            except TeacherTransportError as error:
                # A recorded batch the provider no longer has (results are kept for a
                # limited time): forget it and submit the request once more.
                if resumed and attempt == 1 and _status_code(error) == 404:
                    if journal is not None and key is not None:
                        journal.discard_batch(key)
                    handle = None
                    continue
                raise
            except (TeacherBatchError, TeacherResponseError):
                if journal is not None and key is not None:
                    journal.discard_batch(key)
                raise
            if journal is not None and key is not None and not replay:
                journal.store_batch(key, handle.batch_id, handle.custom_ids, collected=True)
            return cases
        raise TeacherBatchError("Anthropic batch could not be resumed or resubmitted")

    def collect_batch(
        self,
        ir: Mapping[str, Any],
        handle: BatchHandle,
    ) -> tuple[GeneratedCase, ...]:
        """Wait for and atomically reconcile all results in a submitted batch."""

        return self._collect(ir, handle, replay=False)

    def _collect(
        self,
        ir: Mapping[str, Any],
        handle: BatchHandle,
        *,
        replay: bool,
    ) -> tuple[GeneratedCase, ...]:
        """``replay`` marks a batch an earlier run already collected and charged."""

        if not isinstance(handle, BatchHandle):
            raise TeacherBatchError("collect_batch requires a BatchHandle")
        self._validate_batch_handle(ir, handle)
        expected = set(handle.custom_ids)

        self._wait_for_batch(handle.batch_id)
        try:
            result_stream = self._get_client().messages.batches.results(handle.batch_id)
        except TeacherConfigurationError:
            raise
        except Exception as error:
            raise _transport_error("failed to retrieve Anthropic batch results", error) from error

        cases: dict[str, GeneratedCase] = {}
        aggregate_response_bytes = [0]
        try:
            for entry in result_stream:
                custom_id = _field(entry, "custom_id")
                if not isinstance(custom_id, str) or custom_id not in expected:
                    raise TeacherBatchError(
                        f"Anthropic batch returned unexpected custom_id {custom_id!r}"
                    )
                if custom_id in cases:
                    raise TeacherBatchError(
                        f"Anthropic batch returned duplicate custom_id {custom_id!r}"
                    )

                outcome = _field(entry, "result")
                outcome_type = _field(outcome, "type")
                if outcome_type == "succeeded":
                    if self._meter is not None and replay:
                        self._meter.replay()
                    elif self._meter is not None:
                        self._meter.charge(_field(_field(outcome, "message"), "usage"), batch=True)
                    cases[custom_id] = self._decode_message(
                        ir,
                        _field(outcome, "message"),
                        custom_id=custom_id,
                        aggregate_response_bytes=aggregate_response_bytes,
                    )
                    continue
                if outcome_type == "errored":
                    raise TeacherBatchError(_batch_item_error(custom_id, outcome))
                if outcome_type in ("canceled", "expired"):
                    raise TeacherBatchError(
                        f"Anthropic batch item {custom_id!r} was {outcome_type}"
                    )
                raise TeacherBatchError(
                    f"Anthropic batch item {custom_id!r} has unknown result type {outcome_type!r}"
                )
        except (TeacherBatchError, TeacherResponseError):
            raise
        except Exception as error:
            raise _transport_error(
                "failed while streaming Anthropic batch results", error
            ) from error

        missing = [custom_id for custom_id in handle.custom_ids if custom_id not in cases]
        if missing:
            raise TeacherBatchError(
                "Anthropic batch results omitted custom IDs: " + ", ".join(missing)
            )
        return tuple(cases[custom_id] for custom_id in handle.custom_ids)

    def _validate_batch_handle(self, ir: Mapping[str, Any], handle: BatchHandle) -> None:
        function_id = _function_id(ir)
        if not hmac.compare_digest(handle.function_id, function_id):
            raise TeacherBatchError("batch handle does not belong to this neural function")
        if not hmac.compare_digest(
            handle.configuration_sha256,
            self._config.configuration_sha256,
        ):
            raise TeacherBatchError("batch handle does not belong to this teacher configuration")

        expected_ids = self._custom_ids(ir, len(handle.custom_ids))
        if handle.custom_ids != expected_ids:
            raise TeacherBatchError("batch handle custom IDs do not match its bound request")

        wire_schema = self._wire_schema(ir)
        request_digest = hashlib.sha256(_REQUEST_DIGEST_DOMAIN)
        for index, custom_id in enumerate(handle.custom_ids):
            _update_request_digest(
                request_digest,
                {
                    "custom_id": custom_id,
                    "params": self._request_params(
                        ir,
                        index,
                        len(handle.custom_ids),
                        wire_schema,
                    ),
                },
            )
        if not hmac.compare_digest(handle.request_sha256, request_digest.hexdigest()):
            raise TeacherBatchError("batch handle request digest does not match this request")

    def _generate_direct(
        self,
        ir: Mapping[str, Any],
        n: int,
    ) -> tuple[GeneratedCase, ...]:
        wire_schema = self._wire_schema(ir)
        result: list[GeneratedCase] = []
        aggregate_response_bytes = [0]
        for index in range(n):
            try:
                message = self._send(self._request_params(ir, index, n, wire_schema))
            except (TeacherConfigurationError, TeacherBudgetExceeded):
                raise
            except Exception as error:
                raise _transport_error(
                    f"Anthropic Messages request {index + 1} of {n} failed",
                    error,
                ) from error
            with self._discard_rejected():
                result.append(
                    self._decode_message(
                        ir,
                        message,
                        custom_id=f"case {index + 1}",
                        aggregate_response_bytes=aggregate_response_bytes,
                    )
                )
        return tuple(result)

    def _wait_for_batch(self, batch_id: str) -> None:
        deadline = self._clock() + self._config.poll_timeout_seconds
        while True:
            try:
                batch = self._get_client().messages.batches.retrieve(batch_id)
            except TeacherConfigurationError:
                raise
            except Exception as error:
                raise _transport_error("failed to poll Anthropic message batch", error) from error

            status = _field(batch, "processing_status")
            if status == "ended":
                return
            if status not in ("in_progress", "canceling"):
                raise TeacherBatchError(
                    f"Anthropic batch {batch_id!r} has unknown processing status {status!r}"
                )

            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TeacherBatchTimeout(
                    f"Anthropic batch {batch_id!r} did not finish within "
                    f"{self._config.poll_timeout_seconds:g} seconds"
                )
            self._sleeper(min(self._config.poll_interval_seconds, remaining))

    def _decode_message(
        self,
        ir: Mapping[str, Any],
        message: Any,
        *,
        custom_id: str,
        aggregate_response_bytes: list[int],
    ) -> GeneratedCase:
        stop_reason = _field(message, "stop_reason")
        if stop_reason != "end_turn":
            raise TeacherResponseError(
                f"Anthropic response for {custom_id!r} stopped with {stop_reason!r}"
            )

        text = _sole_text_block(
            _field(message, "content"),
            f"Anthropic response for {custom_id!r} must contain exactly one text block",
        )
        encoded_length = len(text.encode("utf-8", errors="surrogatepass"))
        aggregate_response_bytes[0] += encoded_length
        if aggregate_response_bytes[0] > MAXIMUM_TEACHER_RESPONSE_BYTES:
            raise TeacherResponseError(
                f"Anthropic responses exceed aggregate byte limit {MAXIMUM_TEACHER_RESPONSE_BYTES}"
            )
        try:
            return parse_case_response(ir, text)
        except TeacherResponseError:
            raise
        except Exception as error:
            raise TeacherResponseError(
                f"Anthropic response for {custom_id!r} violates the generated-case contract: {error}"
            ) from error

    def _wire_schema(self, ir: Mapping[str, Any]) -> dict[str, Any]:
        return self._transform_wire_schema(build_case_schema(dict(ir)))

    def _transform_wire_schema(self, schema: Mapping[str, Any]) -> dict[str, Any]:
        try:
            lowered = _lower_anthropic_schema(schema)
            transformed = self._get_schema_transform()(lowered)
        except TeacherConfigurationError:
            raise
        except Exception as error:
            raise TeacherConfigurationError(
                f"could not build Anthropic structured-output schema: {error}"
            ) from error
        if not isinstance(transformed, dict):
            raise TeacherConfigurationError("Anthropic schema transform must return a dictionary")
        return transformed

    def _request_adversarial(
        self,
        schema: Mapping[str, Any],
        system: str,
        user: str,
        context: str,
    ) -> str:
        wire_schema = self._transform_wire_schema(schema)
        try:
            message = self._send(
                {
                    "model": self._config.model,
                    "max_tokens": self._config.max_tokens,
                    "system": _cached_system(system),
                    "messages": [{"role": "user", "content": user}],
                    "output_config": {
                        "format": {
                            "type": "json_schema",
                            "schema": wire_schema,
                        }
                    },
                    "thinking": {"type": "disabled"},
                }
            )
        except (TeacherConfigurationError, TeacherBudgetExceeded):
            raise
        except Exception as error:
            raise _transport_error(
                f"Anthropic {context} request failed",
                error,
            ) from error
        with self._discard_rejected():
            stop_reason = _field(message, "stop_reason")
            if stop_reason != "end_turn":
                raise TeacherResponseError(
                    f"Anthropic {context} response stopped with {stop_reason!r}"
                )
            text = _sole_text_block(
                _field(message, "content"),
                f"Anthropic {context} response must contain exactly one text block",
            )
            if not text.strip():
                raise TeacherResponseError(
                    f"Anthropic {context} response must contain exactly one nonempty text block"
                )
            if len(text.encode("utf-8", errors="surrogatepass")) > MAXIMUM_TEACHER_RESPONSE_BYTES:
                raise TeacherResponseError(
                    f"Anthropic {context} response exceeds byte limit "
                    f"{MAXIMUM_TEACHER_RESPONSE_BYTES}"
                )
        return text

    def _request_params(
        self,
        ir: Mapping[str, Any],
        index: int,
        total: int,
        wire_schema: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            system, user = build_case_messages(ir, index, total)
        except Exception as error:
            raise TeacherConfigurationError(
                f"could not build Anthropic teacher prompt: {error}"
            ) from error
        return {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "system": _cached_system(system),
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": wire_schema,
                }
            },
            # Explicit so every route answers with the text block alone: extended
            # thinking is off by default on the Anthropic API, but OpenRouter's
            # Anthropic-format route turns it on and prepends a thinking block.
            "thinking": {"type": "disabled"},
        }

    def _send(self, params: dict[str, Any]) -> Any:
        """One Messages request through the journal and the spend meter."""

        key = None
        self._last_journal_key = None
        if self._journal is not None:
            key = self._journal.next_key(params)
            self._last_journal_key = key
            replayed = self._journal.load(key)
            if replayed is not None:
                if self._meter is not None:
                    self._meter.replay()
                return replayed
        if self._meter is not None:
            self._meter.reserve(self._meter.estimate_request_usd(_prompt_characters(params)))
        started = time.monotonic()
        message = self._get_client().messages.create(**params)
        if self._meter is not None:
            self._meter.charge(_field(message, "usage"), seconds=time.monotonic() - started)
        if self._journal is not None and key is not None:
            self._journal.store(key, message)
        return message

    @contextlib.contextmanager
    def _discard_rejected(self) -> Iterator[None]:
        """Drop the last journaled response when decoding it fails, so a rerun sends
        the request again instead of replaying an answer that breaks the contract."""

        try:
            yield
        except Exception:
            if self._journal is not None and self._last_journal_key is not None:
                self._journal.discard(self._last_journal_key)
                self._last_journal_key = None
            raise

    def _custom_ids(self, ir: Mapping[str, Any], n: int) -> tuple[str, ...]:
        function_id = _function_id(ir)
        digest = hashlib.sha256(
            (function_id + "\x00" + self._config.configuration_sha256 + "\x00" + str(n)).encode()
        ).hexdigest()[:16]
        return tuple(f"case_{digest}_{index:08d}" for index in range(n))

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as error:
                raise TeacherConfigurationError(
                    "Anthropic backend requires the 'anthropic' Python package"
                ) from error

            options: dict[str, Any] = {
                "timeout": self._config.timeout_seconds,
                "max_retries": self._config.max_retries,
            }
            if self._config.api_key is not None:
                options["api_key"] = self._config.api_key
            if self._config.base_url is not None:
                options["base_url"] = self._config.base_url
            try:
                self._client = Anthropic(**options)
            except Exception as error:
                raise TeacherConfigurationError(
                    f"could not configure the Anthropic client: {error}"
                ) from error
        return self._client

    def _get_schema_transform(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        if self._schema_transform is None:
            try:
                from anthropic import transform_schema
            except ImportError as error:
                raise TeacherConfigurationError(
                    "Anthropic backend requires the 'anthropic' Python package"
                ) from error
            self._schema_transform = transform_schema
        return self._schema_transform


def _cached_system(system: str) -> list[dict[str, Any]]:
    """The system prompt as one text block marked for prompt caching.

    Every request for one function and one request kind starts with the same system
    prompt (instructions plus the compact contract), so from the second request on the
    provider reads it from its cache at a tenth of the input price. A prompt shorter
    than the model's minimum cacheable length is simply not cached.
    """

    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def _prompt_characters(params: Mapping[str, Any]) -> int:
    """Characters of the prompt a request sends: system, messages and the output schema."""

    system = params.get("system")
    text = (
        system
        if isinstance(system, str)
        else "".join(str(_field(block, "text") or "") for block in system or [])
    )
    messages = params.get("messages") or []
    user = "".join(str(_field(item, "content") or "") for item in messages)
    schema = json.dumps(
        _field(_field(params.get("output_config"), "format"), "schema"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return len(text) + len(user) + len(schema)


def _validate_count(n: int) -> None:
    validate_case_count(n)


def _validate_positive_count(n: int) -> None:
    _validate_count(n)
    if n == 0:
        raise TeacherConfigurationError("batch case count must be positive")


def _function_id(ir: Mapping[str, Any]) -> str:
    function_id = ir.get("id")
    if not isinstance(function_id, str) or _FUNCTION_ID.fullmatch(function_id) is None:
        raise TeacherConfigurationError("IR id must match nf_<64 lowercase hexadecimal characters>")
    return function_id


def _update_request_digest(digest: Any, request: Mapping[str, Any]) -> None:
    try:
        encoded = json.dumps(
            request,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TeacherConfigurationError(
            f"Anthropic batch request is not canonical JSON: {error}"
        ) from error
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _lower_anthropic_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Lower the exact local schema to the subset accepted by SDK 1.7."""

    if "const" in schema:
        value = schema["const"]
        return {"type": _json_schema_type(value), "enum": [value]}

    enum = schema.get("enum")
    if isinstance(enum, list):
        return _typed_enum_schema(enum)

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        if not any_of:
            raise TeacherConfigurationError("Anthropic schema anyOf must not be empty")
        return {
            "anyOf": [
                _lower_anthropic_schema(_schema_mapping(item, "anyOf variant")) for item in any_of
            ]
        }

    kind = schema.get("type")
    if kind == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            raise TeacherConfigurationError(
                "Anthropic object schema requires properties and required"
            )
        if any(not isinstance(name, str) for name in required):
            raise TeacherConfigurationError("Anthropic object required names must be strings")
        return {
            "type": "object",
            "additionalProperties": False,
            "required": list(required),
            "properties": {
                name: _lower_anthropic_schema(_schema_mapping(value, f"property {name!r}"))
                for name, value in properties.items()
            },
        }

    if kind == "array":
        prefix_items = schema.get("prefixItems")
        if isinstance(prefix_items, list):
            lowered = [
                _lower_anthropic_schema(_schema_mapping(item, "tuple item"))
                for item in prefix_items
            ]
            result: dict[str, Any] = {
                "type": "array",
                "minItems": 0 if not lowered else 1,
                "description": (
                    f"Exact tuple length is {len(lowered)}; position schemas in order are "
                    + json.dumps(lowered, ensure_ascii=False, separators=(",", ":"))
                    + ". The exact tuple contract is validated locally."
                ),
            }
            item_schema = _tuple_item_schema(lowered)
            if item_schema is not None:
                result["items"] = item_schema
            return result

        items = schema.get("items")
        if not isinstance(items, Mapping):
            raise TeacherConfigurationError("Anthropic array schema requires object items")
        return {"type": "array", "items": _lower_anthropic_schema(items)}

    if kind in ("string", "boolean", "number", "integer", "null"):
        return {"type": kind}
    raise TeacherConfigurationError(f"unsupported Anthropic schema node {schema!r}")


def _typed_enum_schema(values: list[Any]) -> dict[str, Any]:
    if not values:
        raise TeacherConfigurationError("Anthropic enum schema must not be empty")
    groups: dict[str, list[Any]] = {}
    for value in values:
        kind = _json_schema_type(value)
        groups.setdefault(kind, []).append(value)
    if len(groups) == 1:
        kind, members = next(iter(groups.items()))
        return {"type": kind, "enum": members}
    return {"anyOf": [{"type": kind, "enum": members} for kind, members in sorted(groups.items())]}


def _json_schema_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float) and math.isfinite(value):
        return "number"
    raise TeacherConfigurationError(
        f"Anthropic enum and literal values must be finite JSON scalars, got {value!r}"
    )


def _tuple_item_schema(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        key = json.dumps(
            item, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        unique.setdefault(key, item)
    variants = list(unique.values())
    return variants[0] if len(variants) == 1 else {"anyOf": variants}


def _schema_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TeacherConfigurationError(f"Anthropic schema {context} must be an object")
    return value


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _status_code(error: BaseException) -> int | None:
    cause = error.__cause__
    status = getattr(cause, "status_code", None)
    return status if isinstance(status, int) else None


def _transport_error(message: str, error: Exception) -> TeacherTransportError:
    status = getattr(error, "status_code", None)
    request_id = getattr(error, "request_id", None)
    details = message
    if status is not None:
        details += f" (status {status})"
    if request_id:
        details += f" (request_id {request_id})"
    return TeacherTransportError(f"{details}: {error}")


def _batch_item_error(custom_id: str, outcome: Any) -> str:
    response = _field(outcome, "error")
    error = _field(response, "error")
    error_type = _field(error, "type")
    message = _field(error, "message")
    request_id = _field(response, "request_id")
    details = f"Anthropic batch item {custom_id!r} errored"
    if isinstance(error_type, str):
        details += f" with {error_type}"
    if isinstance(request_id, str) and request_id:
        details += f" (request_id {request_id})"
    if isinstance(message, str) and message:
        details += f": {message}"
    return details


__all__ = ["AnthropicTeacher", "BatchHandle"]


def _sole_text_block(content: Any, message: str) -> str:
    """The text of the response's only text block; other block kinds (thinking) are ignored."""

    if not isinstance(content, (list, tuple)):
        raise TeacherResponseError(message)
    texts = [
        _field(block, "text")
        for block in content
        if _field(block, "type") == "text" and isinstance(_field(block, "text"), str)
    ]
    if len(texts) != 1:
        raise TeacherResponseError(message)
    return texts[0]
