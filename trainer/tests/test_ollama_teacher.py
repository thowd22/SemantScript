from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import semantscript_trainer.teachers.ollama as ollama_module
from semantscript_trainer.adversarial_contract import (
    build_boundary_pair_schema,
    build_counterfactual_schema,
)
from semantscript_trainer.case_contract import build_case_schema
from semantscript_trainer.teacher import (
    BoundaryPairProposal,
    CounterfactualProposal,
    GeneratedCase,
    TeacherConfigurationError,
    TeacherResponseError,
    TeacherTransportError,
)
from semantscript_trainer.teacher_config import TeacherConfig
from semantscript_trainer.teachers.ollama import DEFAULT_OLLAMA_BASE_URL, OllamaTeacher

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LIVE_URL = os.environ.get("SEMANTSCRIPT_OLLAMA_LIVE_URL")
LIVE_MODEL = os.environ.get("SEMANTSCRIPT_OLLAMA_LIVE_MODEL")


def _refund_ir() -> dict[str, Any]:
    return json.loads(
        (REPOSITORY_ROOT / "examples" / "ir" / "refund-decision.v1.json").read_text(
            encoding="utf-8"
        )
    )


def _case_document(*, output: str = "approve") -> str:
    return json.dumps(
        {
            "inputs": {
                "customer": {"priorRefunds": 0, "tier": "enterprise"},
                "order": {"ageDays": 45, "status": "paid", "total": 129},
            },
            "output": output,
        }
    )


def _response(
    content: object = None,
    *,
    finish_reason: object = "stop",
    refusal: object = None,
) -> SimpleNamespace:
    if content is None:
        content = _case_document()
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, refusal=refusal),
            )
        ]
    )


class _FakeCompletions:
    def __init__(self, *outcomes: object) -> None:
        self.calls: list[dict[str, Any]] = []
        self._outcomes = list(outcomes)

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, *outcomes: object) -> None:
        self.completions = _FakeCompletions(*outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


def _config(**overrides: object) -> TeacherConfig:
    values: dict[str, object] = {
        "backend": "ollama",
        "model": "qwen3:14b-q4_K_M",
        "max_tokens": 777,
        "seed": 19,
    }
    values.update(overrides)
    return TeacherConfig.from_mapping(values)


def test_generates_one_locally_validated_case_per_openai_chat_request() -> None:
    client = _FakeClient(_response(), _response(_case_document(output="deny")))
    config = _config()
    teacher = OllamaTeacher(config, client=client)
    ir = _refund_ir()

    result = teacher.generate(ir, 2)

    assert result == (
        GeneratedCase(
            inputs={
                "customer": {"priorRefunds": 0, "tier": "enterprise"},
                "order": {"ageDays": 45, "status": "paid", "total": 129},
            },
            output="approve",
        ),
        GeneratedCase(
            inputs={
                "customer": {"priorRefunds": 0, "tier": "enterprise"},
                "order": {"ageDays": 45, "status": "paid", "total": 129},
            },
            output="deny",
        ),
    )
    assert teacher.descriptor.provider == "ollama"
    assert teacher.descriptor.model == "qwen3:14b-q4_K_M"
    assert teacher.descriptor.configuration_sha256 == config.configuration_sha256

    assert len(client.completions.calls) == 2
    for index, call in enumerate(client.completions.calls):
        assert call["model"] == "qwen3:14b-q4_K_M"
        assert call["stream"] is False
        assert call["temperature"] == 0
        assert call["reasoning_effort"] == "none"
        assert call["seed"] == 19
        assert call["max_tokens"] == 777
        assert call["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "semantscript_case_v1",
                "strict": True,
                "schema": build_case_schema(ir),
            },
        }
        assert [message["role"] for message in call["messages"]] == ["system", "user"]
        assert f"Generate case {index + 1} of 2" in call["messages"][1]["content"]


def test_meter_counts_local_requests_tokens_and_latency_at_no_cost() -> None:
    from semantscript_trainer.teacher_spend import SpendMeter, resolve_price

    response = _response()
    response.usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=60)
    config = _config()
    meter = SpendMeter(resolve_price(config), max_cost_usd=0.01)
    teacher = OllamaTeacher(config, client=_FakeClient(response), meter=meter)

    teacher.generate(_refund_ir(), 1)

    assert (meter.requests, meter.input_tokens, meter.output_tokens) == (1, 1200, 60)
    assert meter.cost_usd == 0.0
    assert meter.timed_requests == 1
    assert "USD 0.0000 of the USD 0.01 cap" in meter.line()


def test_generates_structured_boundary_and_counterfactual_responses() -> None:
    contract = _refund_ir()
    false_case = json.loads(_case_document(output="approve"))
    true_case = json.loads(_case_document(output="deny"))
    true_case["inputs"]["order"]["status"] = "fraudulent"
    boundary_text = json.dumps({"predicateFalse": false_case, "predicateTrue": true_case})
    counterfactual_text = json.dumps(
        {
            "twin": true_case,
            "reason": "Fraudulent status forbids approval.",
        }
    )
    client = _FakeClient(_response(boundary_text), _response(counterfactual_text))
    teacher = OllamaTeacher(_config(), client=client)
    anchor = GeneratedCase(inputs=false_case["inputs"], output="approve")

    assert teacher.generate_boundary_pair(contract, 0) == BoundaryPairProposal(
        predicate_false=anchor,
        predicate_true=GeneratedCase(inputs=true_case["inputs"], output="deny"),
    )
    assert teacher.generate_counterfactual(contract, anchor) == CounterfactualProposal(
        twin=GeneratedCase(inputs=true_case["inputs"], output="deny"),
        reason="Fraudulent status forbids approval.",
    )
    assert client.completions.calls[0]["response_format"]["json_schema"]["schema"] == (
        build_boundary_pair_schema(contract)
    )
    assert client.completions.calls[1]["response_format"]["json_schema"]["schema"] == (
        build_counterfactual_schema(contract)
    )
    assert all(call["reasoning_effort"] == "none" for call in client.completions.calls)


def test_rejects_case_that_violates_ir_after_structured_response() -> None:
    teacher = OllamaTeacher(
        _config(), client=_FakeClient(_response(_case_document(output="not-supported")))
    )

    with pytest.raises(TeacherResponseError):
        teacher.generate(_refund_ir(), 1)


def test_bounds_aggregate_response_bytes_while_collecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _case_document()
    monkeypatch.setattr(
        ollama_module,
        "MAXIMUM_TEACHER_RESPONSE_BYTES",
        len(content.encode("utf-8")) * 2 - 1,
    )
    teacher = OllamaTeacher(
        _config(),
        client=_FakeClient(_response(content), _response(content)),
    )

    with pytest.raises(TeacherResponseError, match="aggregate byte limit"):
        teacher.generate(_refund_ir(), 2)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (SimpleNamespace(choices=[]), "exactly one choice"),
        (
            SimpleNamespace(choices=[_response().choices[0], _response().choices[0]]),
            "exactly one choice",
        ),
        (_response(finish_reason="length"), "finish with reason 'stop'"),
        (_response(refusal="no"), "accepted message"),
        (_response(content=17), "nonempty string"),
        (_response(content=" "), "nonempty string"),
    ],
)
def test_rejects_malformed_or_incomplete_chat_response(
    response: object,
    message: str,
) -> None:
    teacher = OllamaTeacher(_config(), client=_FakeClient(response))

    with pytest.raises(TeacherResponseError, match=message):
        teacher.generate(_refund_ir(), 1)


def test_wraps_openai_transport_failure_without_returning_partial_cases() -> None:
    failure = TimeoutError("request timed out")
    teacher = OllamaTeacher(_config(), client=_FakeClient(_response(), failure))

    with pytest.raises(TeacherTransportError, match="request 2 of 2") as raised:
        teacher.generate(_refund_ir(), 2)

    assert raised.value.__cause__ is failure


def test_lazily_builds_official_client_with_local_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient(_response())
    constructor_calls: list[dict[str, object]] = []
    openai = ModuleType("openai")

    def create_client(**kwargs: object) -> _FakeClient:
        constructor_calls.append(kwargs)
        return client

    openai.OpenAI = create_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", openai)
    teacher = OllamaTeacher(_config())
    assert constructor_calls == []

    teacher.generate(_refund_ir(), 1)

    assert constructor_calls == [
        {
            "base_url": DEFAULT_OLLAMA_BASE_URL,
            "api_key": "ollama",
            "timeout": 600.0,
            "max_retries": 2,
        }
    ]


def test_forwards_configured_client_connection_options(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient(_response())
    constructor_calls: list[dict[str, object]] = []
    openai = ModuleType("openai")

    def create_client(**kwargs: object) -> _FakeClient:
        constructor_calls.append(kwargs)
        return client

    openai.OpenAI = create_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", openai)
    teacher = OllamaTeacher(
        _config(
            api_key="configured-key",
            base_url="http://172.20.0.1:11434/v1",
            timeout_seconds=12.5,
            max_retries=0,
        )
    )

    teacher.generate(_refund_ir(), 1)

    assert constructor_calls == [
        {
            "base_url": "http://172.20.0.1:11434/v1",
            "api_key": "configured-key",
            "timeout": 12.5,
            "max_retries": 0,
        }
    ]


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_rejects_invalid_case_count(count: object) -> None:
    teacher = OllamaTeacher(_config(), client=_FakeClient())
    with pytest.raises(TeacherConfigurationError, match="case count"):
        teacher.generate(_refund_ir(), count)  # type: ignore[arg-type]


def test_zero_cases_does_not_connect() -> None:
    teacher = OllamaTeacher(_config(), client=_FakeClient())
    assert teacher.generate(_refund_ir(), 0) == ()


def test_rejects_non_ollama_configuration() -> None:
    config = TeacherConfig(backend="anthropic", model="claude-sonnet-5")
    with pytest.raises(TeacherConfigurationError, match="requires the 'ollama' backend"):
        OllamaTeacher(config)


@pytest.mark.skipif(
    not LIVE_URL or not LIVE_MODEL,
    reason="set SEMANTSCRIPT_OLLAMA_LIVE_URL and SEMANTSCRIPT_OLLAMA_LIVE_MODEL",
)
def test_live_ollama_openai_compatible_endpoint() -> None:
    teacher = OllamaTeacher(
        TeacherConfig(
            backend="ollama",
            model=LIVE_MODEL or "",
            base_url=LIVE_URL,
            max_tokens=1_024,
            # A cold 30B load can take several minutes before generation starts.
            timeout_seconds=300,
            max_retries=0,
            seed=1,
        )
    )

    generated = teacher.generate(_refund_ir(), 1)

    assert len(generated) == 1
