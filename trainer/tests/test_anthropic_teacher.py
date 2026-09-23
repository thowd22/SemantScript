from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import semantscript_trainer.teachers.anthropic as anthropic_module
from semantscript_trainer.teacher import (
    BoundaryPairProposal,
    CounterfactualProposal,
    GeneratedCase,
    TeacherBatchError,
    TeacherBatchTimeout,
    TeacherConfigurationError,
    TeacherResponseError,
    TeacherTransportError,
)
from semantscript_trainer.teacher_config import TeacherConfig
from semantscript_trainer.teachers.anthropic import AnthropicTeacher, BatchHandle


def message(text: str, stop_reason: str | None = "end_turn") -> Any:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
    )


def test_direct_requests_use_structured_output_and_validate_each_case() -> None:
    client = FakeClient(
        direct_messages=[
            message(case_text("first", True)),
            message(case_text("second", False)),
        ]
    )
    transformed: list[dict[str, Any]] = []

    def transform(schema: dict[str, Any]) -> dict[str, Any]:
        transformed.append(schema)
        return {**schema, "description": "wire schema"}

    teacher = AnthropicTeacher(config(mode="direct"), client=client, schema_transform=transform)

    assert teacher.descriptor.provider == "anthropic"
    assert teacher.descriptor.model == "claude-sonnet-5"
    assert teacher.generate(ir(), 2) == (
        GeneratedCase(inputs={"message": "first"}, output=True),
        GeneratedCase(inputs={"message": "second"}, output=False),
    )
    assert len(transformed) == 1
    assert len(client.messages.create_calls) == 2
    for params in client.messages.create_calls:
        assert params["model"] == "claude-sonnet-5"
        assert params["max_tokens"] == 321
        assert params["messages"][0]["role"] == "user"
        assert params["system"]
        assert params["output_config"] == {
            "format": {
                "type": "json_schema",
                "schema": {**transformed[0], "description": "wire schema"},
            }
        }
        assert "output_format" not in params
        assert "temperature" not in params


def test_direct_adversarial_requests_return_boundary_pair_and_reasoned_twin() -> None:
    contract = ir()
    contract["definition"]["constraints"] = [
        {
            "kind": "always",
            "source": 'message === "block"',
            "predicate": {
                "node": "binary",
                "operator": "===",
                "left": {"node": "input", "name": "message"},
                "right": {"node": "literal", "value": "block"},
            },
            "output": True,
        }
    ]
    boundary = json.dumps(
        {
            "predicateFalse": {"inputs": {"message": "allow"}, "output": False},
            "predicateTrue": {"inputs": {"message": "block"}, "output": True},
        }
    )
    counterfactual = json.dumps(
        {
            "twin": {"inputs": {"message": "block"}, "output": True},
            "reason": "The blocking token changes the label.",
        }
    )
    client = FakeClient(direct_messages=[message(boundary), message(counterfactual)])
    transformed: list[dict[str, Any]] = []
    teacher = AnthropicTeacher(
        config(mode="direct"),
        client=client,
        schema_transform=lambda schema: transformed.append(schema) or schema,
    )
    anchor = GeneratedCase(inputs={"message": "allow"}, output=False)

    assert teacher.generate_boundary_pair(contract, 0) == BoundaryPairProposal(
        predicate_false=anchor,
        predicate_true=GeneratedCase(inputs={"message": "block"}, output=True),
    )
    assert teacher.generate_counterfactual(contract, anchor) == CounterfactualProposal(
        twin=GeneratedCase(inputs={"message": "block"}, output=True),
        reason="The blocking token changes the label.",
    )
    assert len(transformed) == 2
    assert len(client.messages.create_calls) == 2
    assert "predicateFalse" in transformed[0]["properties"]
    assert "reason" in transformed[1]["properties"]
    assert all(
        call["output_config"]["format"]["type"] == "json_schema"
        for call in client.messages.create_calls
    )


def test_direct_mode_bounds_aggregate_response_bytes_while_collecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = case_text("bounded", True)
    monkeypatch.setattr(
        anthropic_module,
        "MAXIMUM_TEACHER_RESPONSE_BYTES",
        len(text.encode("utf-8")) * 2 - 1,
    )
    teacher = AnthropicTeacher(
        config(mode="direct"),
        client=FakeClient(direct_messages=[message(text), message(text)]),
        schema_transform=lambda schema: schema,
    )

    with pytest.raises(TeacherResponseError, match="aggregate byte limit"):
        teacher.generate(ir(), 2)


def test_schema_lowering_accepts_every_input_kind_under_sdk_17_transform() -> None:
    contract = all_input_types_ir()
    document = {
        "inputs": {
            "text": "hello",
            "flag": True,
            "score": 1.5,
            "nothing": None,
            "fixed": "fixed",
            "tone": "warm",
            "numbers": [1, 2.5],
            "pair": ["left", False],
            "empty": [],
            "record": {"required": 3},
            "choice": "chosen",
        },
        "output": "accept",
    }
    client = FakeClient(direct_messages=[message(json.dumps(document))])
    lowered: list[dict[str, Any]] = []

    def transform(schema: dict[str, Any]) -> dict[str, Any]:
        lowered.append(schema)
        return sdk_17_transform(schema)

    teacher = AnthropicTeacher(config(mode="direct"), client=client, schema_transform=transform)

    assert teacher.generate(contract, 1) == (
        GeneratedCase(inputs=document["inputs"], output="accept"),
    )
    assert len(lowered) == 1
    assert_sdk_17_transformable(lowered[0])
    properties = lowered[0]["properties"]["inputs"]["properties"]
    assert properties["fixed"] == {"type": "string", "enum": ["fixed"]}
    assert properties["tone"] == {"type": "string", "enum": ["warm", "cool"]}
    assert "prefixItems" not in properties["pair"]
    assert properties["pair"]["minItems"] == 1
    assert isinstance(properties["pair"]["items"], dict)
    assert "items" not in properties["empty"]
    assert lowered[0]["properties"]["output"] == {
        "type": "string",
        "enum": ["accept", "reject"],
    }


def test_zero_cases_do_not_construct_or_call_the_sdk() -> None:
    teacher = AnthropicTeacher(config(mode="direct"), client=FakeClient())

    assert teacher.generate(ir(), 0) == ()
    assert teacher._client.messages.create_calls == []


def test_auto_mode_batches_and_reconciles_unordered_results() -> None:
    client = FakeClient()
    clock = FakeClock()
    teacher = AnthropicTeacher(
        config(mode="auto", batch_threshold=2, poll_interval_seconds=3.0),
        client=client,
        schema_transform=lambda schema: schema,
        clock=clock,
        sleeper=clock.sleep,
    )
    client.messages.batches.retrieve_values = [
        SimpleNamespace(processing_status="in_progress"),
        SimpleNamespace(processing_status="ended"),
    ]

    original_results = client.messages.batches.results

    def results(batch_id: str) -> list[Any]:
        requests = client.messages.batches.create_calls[0]["requests"]
        client.messages.batches.result_values = [
            batch_success(requests[1]["custom_id"], case_text("second", False)),
            batch_success(requests[0]["custom_id"], case_text("first", True)),
        ]
        return original_results(batch_id)

    client.messages.batches.results = results
    assert teacher.generate(ir(), 2) == (
        GeneratedCase(inputs={"message": "first"}, output=True),
        GeneratedCase(inputs={"message": "second"}, output=False),
    )
    requests = client.messages.batches.create_calls[0]["requests"]
    custom_ids = tuple(request["custom_id"] for request in requests)
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) for value in custom_ids)
    assert len(set(custom_ids)) == 2
    assert all(request["params"].get("stream") is None for request in requests)
    assert clock.value == 3.0
    assert client.messages.batches.results_calls == ["msgbatch_test"]


def test_auto_mode_below_threshold_uses_direct_requests() -> None:
    client = FakeClient(direct_messages=[message(case_text("direct", True))])
    teacher = AnthropicTeacher(
        config(mode="auto", batch_threshold=2),
        client=client,
        schema_transform=lambda schema: schema,
    )

    assert teacher.generate(ir(), 1) == (GeneratedCase(inputs={"message": "direct"}, output=True),)
    assert len(client.messages.create_calls) == 1
    assert client.messages.batches.create_calls == []


def test_generate_batch_submits_one_request_per_case() -> None:
    client = FakeClient()
    client.messages.batches.retrieve_values = [SimpleNamespace(processing_status="ended")]
    teacher = AnthropicTeacher(
        config(mode="batch"),
        client=client,
        schema_transform=lambda schema: schema,
    )

    original_results = client.messages.batches.results

    def results(batch_id: str) -> list[Any]:
        requests = client.messages.batches.create_calls[0]["requests"]
        client.messages.batches.result_values = [
            batch_success(request["custom_id"], case_text(str(index), index % 2 == 0))
            for index, request in enumerate(requests)
        ]
        return original_results(batch_id)

    client.messages.batches.results = results
    generated = teacher.generate(ir(), 3)

    assert [case.inputs["message"] for case in generated] == ["0", "1", "2"]
    assert len(client.messages.batches.create_calls[0]["requests"]) == 3


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens", "tool_use", None])
def test_rejects_nonterminal_structured_responses(stop_reason: str | None) -> None:
    client = FakeClient(direct_messages=[message(case_text("value", True), stop_reason)])
    teacher = AnthropicTeacher(
        config(mode="direct"),
        client=client,
        schema_transform=lambda schema: schema,
    )

    with pytest.raises(TeacherResponseError, match="stopped with"):
        teacher.generate(ir(), 1)


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(stop_reason="end_turn", content=[]),
        SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="text", text="{}"),
                SimpleNamespace(type="text", text="{}"),
            ],
        ),
        SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="thinking", text="not JSON")],
        ),
        message("not JSON"),
        message('{"inputs":{"message":3},"output":true}'),
    ],
)
def test_rejects_invalid_content_and_case_contracts(response: Any) -> None:
    teacher = AnthropicTeacher(
        config(mode="direct"),
        client=FakeClient(direct_messages=[response]),
        schema_transform=lambda schema: schema,
    )

    with pytest.raises(TeacherResponseError):
        teacher.generate(ir(), 1)


def test_batch_rejects_item_errors_and_incomplete_id_sets() -> None:
    client = FakeClient()
    client.messages.batches.retrieve_values = [SimpleNamespace(processing_status="ended")]
    teacher = AnthropicTeacher(
        config(mode="batch"),
        client=client,
        schema_transform=lambda schema: schema,
    )
    handle = teacher.submit_batch(ir(), 2)
    client.messages.batches.result_values = [
        SimpleNamespace(
            custom_id=handle.custom_ids[0],
            result=SimpleNamespace(
                type="errored",
                error=SimpleNamespace(
                    request_id="req_test",
                    error=SimpleNamespace(type="invalid_request_error", message="bad schema"),
                ),
            ),
        )
    ]

    with pytest.raises(
        TeacherBatchError, match=r"invalid_request_error.*req_test|req_test.*bad schema"
    ):
        teacher.collect_batch(ir(), handle)

    client.messages.batches.retrieve_values = [SimpleNamespace(processing_status="ended")]
    client.messages.batches.result_values = [
        batch_success(handle.custom_ids[0], case_text("first", True))
    ]
    with pytest.raises(TeacherBatchError, match="omitted custom IDs"):
        teacher.collect_batch(ir(), handle)


def test_batch_timeout_uses_injected_monotonic_clock_and_sleeper() -> None:
    client = FakeClient()
    client.messages.batches.retrieve_values = [SimpleNamespace(processing_status="in_progress")]
    clock = FakeClock()
    teacher = AnthropicTeacher(
        config(
            mode="batch",
            poll_interval_seconds=2.0,
            poll_timeout_seconds=5.0,
        ),
        client=client,
        schema_transform=lambda schema: schema,
        clock=clock,
        sleeper=clock.sleep,
    )
    handle = teacher.submit_batch(ir(), 1)

    with pytest.raises(TeacherBatchTimeout, match="did not finish"):
        teacher.collect_batch(ir(), handle)
    assert clock.value == 5.0
    assert client.messages.batches.results_calls == []


def test_wraps_sdk_failures_as_typed_transport_errors() -> None:
    error = FakeSdkError("overloaded", status_code=529, request_id="req_123")
    client = FakeClient(create_error=error)
    teacher = AnthropicTeacher(
        config(mode="direct"),
        client=client,
        schema_transform=lambda schema: schema,
    )

    with pytest.raises(TeacherTransportError, match=r"status 529.*req_123.*overloaded"):
        teacher.generate(ir(), 1)


def test_lazily_constructs_official_client_and_schema_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient(direct_messages=[message(case_text("value", True))])
    constructor_calls: list[dict[str, Any]] = []
    transformed: list[dict[str, Any]] = []
    anthropic = ModuleType("anthropic")

    def create_client(**options: Any) -> FakeClient:
        constructor_calls.append(options)
        return client

    def transform_schema(schema: dict[str, Any]) -> dict[str, Any]:
        transformed.append(schema)
        return schema

    anthropic.Anthropic = create_client  # type: ignore[attr-defined]
    anthropic.transform_schema = transform_schema  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", anthropic)
    teacher = AnthropicTeacher(config(mode="direct"))
    assert constructor_calls == []

    assert teacher.generate(ir(), 1) == (GeneratedCase(inputs={"message": "value"}, output=True),)
    assert constructor_calls == [
        {
            "api_key": "test-key",
            "timeout": 12.0,
            "max_retries": 0,
        }
    ]
    assert len(transformed) == 1


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_rejects_invalid_case_count(count: Any) -> None:
    teacher = AnthropicTeacher(config(), client=FakeClient())

    with pytest.raises(TeacherConfigurationError, match="case count"):
        teacher.generate(ir(), count)


def test_configuration_and_batch_handle_validation() -> None:
    with pytest.raises(TeacherConfigurationError, match="requires backend"):
        AnthropicTeacher(replace(config(), backend="ollama"), client=FakeClient())

    teacher = AnthropicTeacher(
        config(mode="batch"),
        client=FakeClient(),
        schema_transform=lambda schema: schema,
    )
    with pytest.raises(TeacherConfigurationError, match="positive"):
        teacher.submit_batch(ir(), 0)
    with pytest.raises(TeacherBatchError, match="duplicate custom IDs"):
        BatchHandle(
            "batch-test",
            ("same", "same"),
            ir()["id"],
            teacher.descriptor.configuration_sha256,
            "0" * 64,
        )
    assert teacher._client.messages.batches.retrieve_calls == []


def test_batch_handle_rejects_tampering_and_wrong_resume_context_before_polling() -> None:
    client = FakeClient()
    teacher = AnthropicTeacher(
        config(mode="batch"),
        client=client,
        schema_transform=lambda schema: schema,
    )
    handle = teacher.submit_batch(ir(), 1)
    assert handle.function_id == ir()["id"]
    assert handle.configuration_sha256 == teacher.descriptor.configuration_sha256
    assert re.fullmatch(r"[a-f0-9]{64}", handle.request_sha256)

    wrong_ir = ir()
    wrong_ir["id"] = "nf_" + "2" * 64
    with pytest.raises(TeacherBatchError, match="neural function"):
        teacher.collect_batch(wrong_ir, handle)

    other_teacher = AnthropicTeacher(
        replace(config(mode="batch"), model="claude-opus-5"),
        client=client,
        schema_transform=lambda schema: schema,
    )
    with pytest.raises(TeacherBatchError, match="teacher configuration"):
        other_teacher.collect_batch(ir(), handle)

    with pytest.raises(TeacherBatchError, match="custom IDs"):
        teacher.collect_batch(
            ir(),
            replace(handle, custom_ids=("case_tampered_00000000",)),
        )
    with pytest.raises(TeacherBatchError, match="request digest"):
        teacher.collect_batch(ir(), replace(handle, request_sha256="0" * 64))
    assert client.messages.batches.retrieve_calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_id": ""},
        {"custom_ids": ()},
        {"custom_ids": ("contains a space",)},
        {"function_id": "not-a-function"},
        {"configuration_sha256": "short"},
        {"request_sha256": "short"},
    ],
)
def test_batch_handle_validates_persisted_fields(changes: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "batch_id": "msgbatch_test",
        "custom_ids": ("case_test_00000000",),
        "function_id": ir()["id"],
        "configuration_sha256": "a" * 64,
        "request_sha256": "b" * 64,
    }
    values.update(changes)

    with pytest.raises(TeacherBatchError):
        BatchHandle(**values)


def config(**changes: Any) -> TeacherConfig:
    values: dict[str, Any] = {
        "backend": "anthropic",
        "model": "claude-sonnet-5",
        "api_key": "test-key",
        "max_tokens": 321,
        "timeout_seconds": 12.0,
        "max_retries": 0,
        "mode": "direct",
        "batch_threshold": 3,
        "poll_interval_seconds": 1.0,
        "poll_timeout_seconds": 10.0,
    }
    values.update(changes)
    return TeacherConfig(**values)


def ir() -> dict[str, Any]:
    return {
        "id": "nf_" + "1" * 64,
        "definition": {
            "template": [{"kind": "text", "text": "Classify the message."}],
            "examples": [],
            "constraints": [],
        },
        "inputs": [
            {
                "name": "message",
                "index": 0,
                "type": {"kind": "string"},
            }
        ],
        "output": {
            "kind": "scalar",
            "head": {"kind": "boolean", "support": [False, True]},
        },
    }


def all_input_types_ir() -> dict[str, Any]:
    return {
        "id": "nf_" + "3" * 64,
        "definition": {
            "template": [{"kind": "text", "text": "Exercise every input kind."}],
            "examples": [],
            "constraints": [],
        },
        "inputs": [
            {"name": "text", "index": 0, "type": {"kind": "string"}},
            {"name": "flag", "index": 1, "type": {"kind": "boolean"}},
            {"name": "score", "index": 2, "type": {"kind": "number"}},
            {"name": "nothing", "index": 3, "type": {"kind": "null"}},
            {
                "name": "fixed",
                "index": 4,
                "type": {"kind": "literal", "value": "fixed"},
            },
            {
                "name": "tone",
                "index": 5,
                "type": {
                    "kind": "enum",
                    "name": "Tone",
                    "base": "string",
                    "values": ["warm", "cool"],
                },
            },
            {
                "name": "numbers",
                "index": 6,
                "type": {"kind": "array", "items": {"kind": "number"}},
            },
            {
                "name": "pair",
                "index": 7,
                "type": {
                    "kind": "tuple",
                    "items": [{"kind": "string"}, {"kind": "boolean"}],
                },
            },
            {
                "name": "empty",
                "index": 8,
                "type": {"kind": "tuple", "items": []},
            },
            {
                "name": "record",
                "index": 9,
                "type": {
                    "kind": "object",
                    "fields": [
                        {
                            "name": "required",
                            "optional": False,
                            "type": {"kind": "number"},
                        },
                        {
                            "name": "optional",
                            "optional": True,
                            "type": {"kind": "string"},
                        },
                    ],
                },
            },
            {
                "name": "choice",
                "index": 10,
                "type": {
                    "kind": "union",
                    "variants": [
                        {"kind": "literal", "value": "chosen"},
                        {"kind": "number"},
                    ],
                },
            },
        ],
        "output": {
            "kind": "scalar",
            "head": {"kind": "nominal", "support": ["accept", "reject"]},
        },
    }


def case_text(value: str, output: bool) -> str:
    return json.dumps({"inputs": {"message": value}, "output": output})


def batch_success(custom_id: str, text: str) -> Any:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="succeeded", message=message(text)),
    )


class FakeSdkError(RuntimeError):
    def __init__(self, message: str, *, status_code: int, request_id: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class FakeBatches:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[str] = []
        self.results_calls: list[str] = []
        self.retrieve_values: list[Any] = []
        self.result_values: list[Any] = []

    def create(self, **params: Any) -> Any:
        self.create_calls.append(params)
        return SimpleNamespace(id="msgbatch_test")

    def retrieve(self, batch_id: str) -> Any:
        self.retrieve_calls.append(batch_id)
        if len(self.retrieve_values) > 1:
            return self.retrieve_values.pop(0)
        if self.retrieve_values:
            return self.retrieve_values[0]
        return SimpleNamespace(processing_status="ended")

    def results(self, batch_id: str) -> list[Any]:
        self.results_calls.append(batch_id)
        return self.result_values


class FakeMessages:
    def __init__(
        self,
        direct_messages: list[Any] | None = None,
        create_error: Exception | None = None,
    ) -> None:
        self.batches = FakeBatches()
        self.create_calls: list[dict[str, Any]] = []
        self.direct_messages = list(direct_messages or [])
        self.create_error = create_error

    def create(self, **params: Any) -> Any:
        self.create_calls.append(params)
        if self.create_error is not None:
            raise self.create_error
        return self.direct_messages.pop(0)


class FakeClient:
    def __init__(
        self,
        direct_messages: list[Any] | None = None,
        create_error: Exception | None = None,
    ) -> None:
        self.messages = FakeMessages(direct_messages, create_error)


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def sdk_17_transform(schema: dict[str, Any]) -> dict[str, Any]:
    """Use the pinned transformer when installed; otherwise enforce its input contract."""

    try:
        from anthropic import transform_schema
    except ImportError:
        assert_sdk_17_transformable(schema)
        return schema
    return transform_schema(schema)


def assert_sdk_17_transformable(schema: Any) -> None:
    assert isinstance(schema, dict)
    assert "const" not in schema
    assert "prefixItems" not in schema
    if "anyOf" in schema:
        assert isinstance(schema["anyOf"], list) and schema["anyOf"]
        for variant in schema["anyOf"]:
            assert_sdk_17_transformable(variant)
        return

    assert schema.get("type") in {
        "object",
        "array",
        "string",
        "integer",
        "number",
        "boolean",
        "null",
    }
    if schema["type"] == "object":
        assert isinstance(schema.get("properties"), dict)
        for value in schema["properties"].values():
            assert_sdk_17_transformable(value)
    if schema["type"] == "array" and "items" in schema:
        assert isinstance(schema["items"], dict)
        assert_sdk_17_transformable(schema["items"])
    if "minItems" in schema:
        assert schema["minItems"] in (0, 1)
