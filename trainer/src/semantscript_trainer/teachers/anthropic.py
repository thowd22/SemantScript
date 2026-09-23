"""Anthropic Messages and Message Batches teacher backend."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import time
from collections.abc import Callable, Mapping
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
    TeacherConfigurationError,
    TeacherDescriptor,
    TeacherResponseError,
    TeacherTransportError,
)
from semantscript_trainer.teacher_config import TeacherConfig
from semantscript_trainer.teacher_prompt import build_case_messages

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
    ) -> None:
        if config.backend != "anthropic":
            raise TeacherConfigurationError(
                f"AnthropicTeacher requires backend='anthropic', got {config.backend!r}"
            )
        self._config = config
        self._client = client
        self._schema_transform = schema_transform
        self._clock = clock
        self._sleeper = sleeper

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
            handle = self.submit_batch(ir, n)
            return self.collect_batch(ir, handle)
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
        return parse_counterfactual_response(dict(ir), text)

    def submit_batch(self, ir: Mapping[str, Any], n: int) -> BatchHandle:
        """Submit one non-streaming Messages request for each desired case."""

        _validate_positive_count(n)
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
            request_sha256=request_digest.hexdigest(),
        )

    def collect_batch(
        self,
        ir: Mapping[str, Any],
        handle: BatchHandle,
    ) -> tuple[GeneratedCase, ...]:
        """Wait for and atomically reconcile all results in a submitted batch."""

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
                message = self._get_client().messages.create(
                    **self._request_params(ir, index, n, wire_schema)
                )
            except TeacherConfigurationError:
                raise
            except Exception as error:
                raise _transport_error(
                    f"Anthropic Messages request {index + 1} of {n} failed",
                    error,
                ) from error
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

        content = _field(message, "content")
        if not isinstance(content, (list, tuple)) or len(content) != 1:
            raise TeacherResponseError(
                f"Anthropic response for {custom_id!r} must contain exactly one text block"
            )
        block = content[0]
        if _field(block, "type") != "text" or not isinstance(_field(block, "text"), str):
            raise TeacherResponseError(
                f"Anthropic response for {custom_id!r} must contain exactly one text block"
            )
        text = _field(block, "text")
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
            message = self._get_client().messages.create(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": wire_schema,
                    }
                },
            )
        except TeacherConfigurationError:
            raise
        except Exception as error:
            raise _transport_error(
                f"Anthropic {context} request failed",
                error,
            ) from error
        stop_reason = _field(message, "stop_reason")
        if stop_reason != "end_turn":
            raise TeacherResponseError(f"Anthropic {context} response stopped with {stop_reason!r}")
        content = _field(message, "content")
        if not isinstance(content, (list, tuple)) or len(content) != 1:
            raise TeacherResponseError(
                f"Anthropic {context} response must contain exactly one text block"
            )
        block = content[0]
        text = _field(block, "text")
        if _field(block, "type") != "text" or not isinstance(text, str) or not text.strip():
            raise TeacherResponseError(
                f"Anthropic {context} response must contain exactly one nonempty text block"
            )
        if len(text.encode("utf-8", errors="surrogatepass")) > MAXIMUM_TEACHER_RESPONSE_BYTES:
            raise TeacherResponseError(
                f"Anthropic {context} response exceeds byte limit {MAXIMUM_TEACHER_RESPONSE_BYTES}"
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
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": wire_schema,
                }
            },
        }

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
