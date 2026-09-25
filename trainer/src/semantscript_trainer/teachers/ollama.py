"""Ollama teacher backed by its OpenAI-compatible chat endpoint."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

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
    NeuralFunctionIr,
    TeacherConfigurationError,
    TeacherDescriptor,
    TeacherResponseError,
    TeacherTransportError,
)
from semantscript_trainer.teacher_config import TeacherConfig
from semantscript_trainer.teacher_prompt import build_case_messages
from semantscript_trainer.teacher_spend import SpendMeter

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_LOCAL_API_KEY = "ollama"
_RESPONSE_SCHEMA_NAME = "semantscript_case_v1"


class _ChatCompletions(Protocol):
    def create(self, **kwargs: Any) -> object: ...


class _Chat(Protocol):
    completions: _ChatCompletions


class _OpenAIClient(Protocol):
    chat: _Chat


class OllamaTeacher:
    """Generate independently validated cases through local Ollama."""

    __slots__ = ("_client", "_config", "_descriptor", "_meter")

    def __init__(
        self,
        config: TeacherConfig,
        *,
        client: _OpenAIClient | None = None,
        meter: SpendMeter | None = None,
        journal: object | None = None,
    ) -> None:
        """``meter`` counts every request, its tokens and latency (local requests cost
        nothing). ``journal`` is accepted for a uniform provider interface and unused:
        replaying a free local response saves nothing."""
        del journal
        if config.backend != "ollama":
            raise TeacherConfigurationError("OllamaTeacher requires the 'ollama' backend")
        self._config = config
        self._client = client
        self._meter = meter
        self._descriptor = TeacherDescriptor(
            provider="ollama",
            model=config.model,
            configuration_sha256=config.configuration_sha256,
        )

    @property
    def descriptor(self) -> TeacherDescriptor:
        return self._descriptor

    def generate(self, ir: NeuralFunctionIr, n: int, /) -> tuple[GeneratedCase, ...]:
        validate_case_count(n)
        if n == 0:
            return ()

        try:
            schema = build_case_schema(ir)
        except TeacherConfigurationError:
            raise
        except Exception as error:
            raise TeacherConfigurationError(
                f"could not build Ollama structured-output schema: {error}"
            ) from error

        cases: list[GeneratedCase] = []
        aggregate_response_bytes = 0
        for index in range(n):
            try:
                system_message, user_message = build_case_messages(ir, index, n)
            except TeacherConfigurationError:
                raise
            except Exception as error:
                raise TeacherConfigurationError(
                    f"could not build Ollama teacher prompt: {error}"
                ) from error
            response = self._request_case(schema, system_message, user_message, index, n)
            content = _extract_content(response, index, n)
            aggregate_response_bytes += len(content.encode("utf-8", errors="surrogatepass"))
            if aggregate_response_bytes > MAXIMUM_TEACHER_RESPONSE_BYTES:
                raise TeacherResponseError(
                    f"Ollama responses exceed aggregate byte limit {MAXIMUM_TEACHER_RESPONSE_BYTES}"
                )
            try:
                cases.append(parse_case_response(ir, content))
            except TeacherResponseError:
                raise
            except Exception as error:
                raise TeacherResponseError(
                    f"Ollama response {index + 1} of {n} violates the generated-case "
                    f"contract: {error}"
                ) from error
        return tuple(cases)

    def generate_boundary_pair(
        self,
        ir: NeuralFunctionIr,
        constraint_index: int,
        /,
    ) -> BoundaryPairProposal:
        try:
            schema = build_boundary_pair_schema(ir)
            system, user = build_boundary_messages(ir, constraint_index)
        except TeacherConfigurationError:
            raise
        except Exception as error:
            raise TeacherConfigurationError(
                f"could not build Ollama boundary request: {error}"
            ) from error
        content = _extract_content(
            self._request_case(schema, system, user, constraint_index, constraint_index + 1),
            constraint_index,
            constraint_index + 1,
        )
        _check_response_size(content, "Ollama boundary response")
        return parse_boundary_pair_response(ir, content)

    def generate_counterfactual(
        self,
        ir: NeuralFunctionIr,
        anchor: GeneratedCase,
        /,
    ) -> CounterfactualProposal:
        try:
            schema = build_counterfactual_schema(ir)
            system, user = build_counterfactual_messages(ir, anchor)
        except TeacherConfigurationError:
            raise
        except Exception as error:
            raise TeacherConfigurationError(
                f"could not build Ollama counterfactual request: {error}"
            ) from error
        content = _extract_content(
            self._request_case(schema, system, user, 0, 1),
            0,
            1,
        )
        _check_response_size(content, "Ollama counterfactual response")
        return parse_counterfactual_response(ir, content)

    def _request_case(
        self,
        schema: Mapping[str, object],
        system_message: str,
        user_message: str,
        index: int,
        total: int,
    ) -> object:
        started = time.monotonic()
        try:
            response = self._get_client().chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                stream=False,
                temperature=0,
                reasoning_effort="none",
                seed=self._config.seed,
                max_tokens=self._config.max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": _RESPONSE_SCHEMA_NAME,
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
        except (TeacherConfigurationError, TeacherTransportError):
            raise
        except Exception as error:
            raise TeacherTransportError(
                f"Ollama request {index + 1} of {total} failed: {error}"
            ) from error
        if self._meter is not None:
            self._meter.charge(getattr(response, "usage", None), seconds=time.monotonic() - started)
        return response

    def _get_client(self) -> _OpenAIClient:
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as error:
            raise TeacherConfigurationError(
                "the Ollama teacher requires the official 'openai' Python package"
            ) from error

        try:
            self._client = OpenAI(
                base_url=self._config.base_url or DEFAULT_OLLAMA_BASE_URL,
                api_key=self._config.api_key or _LOCAL_API_KEY,
                timeout=self._config.timeout_seconds,
                max_retries=self._config.max_retries,
            )
        except Exception as error:
            raise TeacherConfigurationError(
                f"could not configure the Ollama client: {error}"
            ) from error
        return self._client


def _extract_content(response: object, index: int, total: int) -> str:
    choices = getattr(response, "choices", None)
    if (
        not isinstance(choices, Sequence)
        or isinstance(choices, (str, bytes, bytearray))
        or len(choices) != 1
    ):
        raise TeacherResponseError(
            f"Ollama response {index + 1} of {total} must contain exactly one choice"
        )

    choice = choices[0]
    if getattr(choice, "finish_reason", None) != "stop":
        raise TeacherResponseError(
            f"Ollama response {index + 1} of {total} did not finish with reason 'stop'"
        )

    message = getattr(choice, "message", None)
    if message is None or getattr(message, "refusal", None) is not None:
        raise TeacherResponseError(
            f"Ollama response {index + 1} of {total} did not contain an accepted message"
        )
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise TeacherResponseError(
            f"Ollama response {index + 1} of {total} content must be a nonempty string"
        )
    return content


def _check_response_size(content: str, context: str) -> None:
    if len(content.encode("utf-8", errors="surrogatepass")) > MAXIMUM_TEACHER_RESPONSE_BYTES:
        raise TeacherResponseError(f"{context} exceeds byte limit {MAXIMUM_TEACHER_RESPONSE_BYTES}")


__all__ = ["DEFAULT_OLLAMA_BASE_URL", "OllamaTeacher"]
