from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from benchmarks.refund.program import claude_cli_teacher
from benchmarks.refund.program.claude_cli_teacher import (
    CLAUDE_CLI_DATA_CLASSIFICATION,
    CLAUDE_CLI_MODEL,
    CLAUDE_CLI_PROTOCOL,
    CLAUDE_CLI_VERSION,
    BoundedProcessResult,
    ClaudeCliTeacherConfig,
    ClaudeCliTrainingTeacher,
    run_bounded_process,
)

from semantscript_trainer import (
    AdversarialTeacher,
    BoundaryPairProposal,
    CounterfactualProposal,
    GeneratedCase,
    SyntheticDatasetGenerator,
    Teacher,
    TeacherConfigurationError,
    TeacherResponseError,
    TeacherTransportError,
)


@pytest.fixture(autouse=True)
def _no_transport_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_cli_teacher, "_TRANSPORT_BACKOFF_SECONDS", 0.0)


class FakeRunner:
    def __init__(self, structured_outputs: list[dict[str, Any]] | None = None) -> None:
        self.structured_outputs = list(structured_outputs or [])
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def __call__(
        self,
        command: Sequence[str],
        *,
        stdin: bytes,
        timeout_seconds: float,
        stdout_limit: int,
        stderr_limit: int,
    ) -> BoundedProcessResult:
        captured = {
            "stdin": stdin,
            "timeout_seconds": timeout_seconds,
            "stdout_limit": stdout_limit,
            "stderr_limit": stderr_limit,
        }
        self.calls.append((tuple(command), captured))
        if tuple(command)[-1] == "--version":
            return BoundedProcessResult(0, f"{CLAUDE_CLI_VERSION}\n".encode(), b"")
        output = self.structured_outputs.pop(0)
        return BoundedProcessResult(0, envelope(output), b"")


def envelope(output: Any) -> bytes:
    """Serialize one success envelope in the pinned CLI's shape."""

    wrapper = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 3,
        "stop_reason": "tool_use",
        "permission_denials": [],
        "modelUsage": {CLAUDE_CLI_MODEL: {"costUSD": 0.01}},
        "structured_output": output,
        "session_id": "not-persisted-and-not-exposed",
    }
    return json.dumps(wrapper, separators=(",", ":")).encode()


def test_exact_training_only_identity_and_configuration_digest() -> None:
    teacher = ClaudeCliTrainingTeacher(process_runner=FakeRunner())

    assert isinstance(teacher, Teacher)
    assert isinstance(teacher, AdversarialTeacher)
    assert teacher.descriptor.provider == "anthropic-claude-cli-training-only"
    assert teacher.descriptor.model == "claude-sonnet-5" == CLAUDE_CLI_MODEL
    assert teacher.provenance.cli_version == "unverified"
    teacher.verify_installation()
    assert teacher.provenance.cli_version == "2.1.281 (Claude Code)" == CLAUDE_CLI_VERSION
    assert teacher.configuration_projection["cli"]["requiredVersion"] is None
    assert teacher.provenance.protocol == CLAUDE_CLI_PROTOCOL
    assert teacher.provenance.data_classification == "synthetic-training-only"
    assert teacher.provenance.data_classification == CLAUDE_CLI_DATA_CLASSIFICATION
    assert (
        teacher.descriptor.configuration_sha256
        == "f0e5a7236c7607ebbba046bef591ba94765f380670051e976fc920e2e455127a"
    )
    assert teacher.provenance.configuration_sha256 == teacher.descriptor.configuration_sha256
    assert teacher.configuration_projection["request"]["tools"] == []
    assert teacher.configuration_projection["request"]["sessionPersistence"] is False
    assert teacher.configuration_projection["promptProtocol"].endswith("/v2")
    assert teacher.configuration_projection["generation"] == {
        "concurrency": 1,
        "maximumCaseAttempts": 3,
        "duplicateRounds": 2,
    }
    assert teacher.last_run_report is None


def test_generate_uses_exact_safe_one_turn_structured_protocol() -> None:
    runner = FakeRunner(
        [
            {"inputs": {"score": 1, "note": "first"}, "output": False},
            {"inputs": {"score": 12, "note": "second"}, "output": True},
        ]
    )
    teacher = ClaudeCliTrainingTeacher(process_runner=runner)

    assert teacher.generate(refund_ir(), 2) == (
        GeneratedCase(inputs={"score": 1, "note": "first"}, output=False),
        GeneratedCase(inputs={"score": 12, "note": "second"}, output=True),
    )

    assert len(runner.calls) == 3
    assert runner.calls[0][0] == ("claude", "--version")
    for index, (command, call) in enumerate(runner.calls[1:]):
        assert command[0:4] == (
            "claude",
            "--print",
            "--model",
            "claude-sonnet-5",
        )
        assert option(command, "--effort") == "high"
        assert option(command, "--input-format") == "text"
        assert option(command, "--output-format") == "json"
        assert option(command, "--tools") == ""
        assert option(command, "--setting-sources") == ""
        assert option(command, "--mcp-config") == '{"mcpServers":{}}'
        assert option(command, "--permission-mode") == "dontAsk"
        assert option(command, "--permission-prompts") == "none"
        assert option(command, "--prompt-suggestions") == "false"
        for flag in (
            "--no-session-persistence",
            "--disable-slash-commands",
            "--safe-mode",
            "--restricted",
            "--strict-mcp-config",
            "--no-chrome",
        ):
            assert flag in command
        for forbidden in (
            "--continue",
            "--resume",
            "--fallback-model",
            "--allowed-tools",
            "--dangerously-skip-permissions",
        ):
            assert forbidden not in command
        schema = json.loads(option(command, "--json-schema"))
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["inputs", "output"]
        assert b"Generate case " + str(index + 1).encode() in call["stdin"]
        assert call["timeout_seconds"] == 600.0


def test_synthetic_dataset_generator_preserves_nonhuman_origin(tmp_path: Path) -> None:
    runner = FakeRunner([{"inputs": {"score": 1, "note": "synthetic"}, "output": False}])
    dataset = SyntheticDatasetGenerator(
        ClaudeCliTrainingTeacher(process_runner=runner),
        tmp_path,
    ).generate(refund_ir(), 1)

    assert dataset.teacher.provider == "anthropic-claude-cli-training-only"
    assert dataset.synthetic_count == 1
    assert dataset.gold_count == 0
    assert dataset.cases[0].origin == "synthetic"
    assert "human" not in dataset.teacher.provider


def test_adversarial_methods_reuse_local_prompts_schemas_and_parsers() -> None:
    false_case = {"inputs": {"score": 9, "note": "edge"}, "output": False}
    true_case = {"inputs": {"score": 10, "note": "edge"}, "output": True}
    runner = FakeRunner(
        [
            {"predicateFalse": false_case, "predicateTrue": true_case},
            {"twin": true_case, "reason": "The score crosses the mandatory boundary."},
        ]
    )
    teacher = ClaudeCliTrainingTeacher(process_runner=runner)
    anchor = GeneratedCase(inputs=false_case["inputs"], output=False)

    assert teacher.generate_boundary_pair(refund_ir(), 0) == BoundaryPairProposal(
        predicate_false=anchor,
        predicate_true=GeneratedCase(inputs=true_case["inputs"], output=True),
    )
    assert teacher.generate_counterfactual(refund_ir(), anchor) == CounterfactualProposal(
        twin=GeneratedCase(inputs=true_case["inputs"], output=True),
        reason="The score crosses the mandatory boundary.",
    )

    boundary_schema = json.loads(option(runner.calls[1][0], "--json-schema"))
    counterfactual_schema = json.loads(option(runner.calls[2][0], "--json-schema"))
    assert set(boundary_schema["properties"]) == {"predicateFalse", "predicateTrue"}
    assert set(counterfactual_schema["properties"]) == {"reason", "twin"}


def test_adversarial_requests_retry_transient_transport_failures() -> None:
    false_case = {"inputs": {"score": 9, "note": "edge"}, "output": False}
    true_case = {"inputs": {"score": 10, "note": "edge"}, "output": True}

    class FlakyRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__([{"predicateFalse": false_case, "predicateTrue": true_case}])
            self.failed_once = False

        def __call__(self, command: Sequence[str], **kwargs: Any) -> BoundedProcessResult:
            if tuple(command)[-1] != "--version" and not self.failed_once:
                self.failed_once = True
                self.calls.append((tuple(command), kwargs))
                return BoundedProcessResult(1, b"", b"rate limited")
            return super().__call__(command, **kwargs)

    runner = FlakyRunner()
    teacher = ClaudeCliTrainingTeacher(process_runner=runner)

    assert teacher.generate_boundary_pair(refund_ir(), 0) == BoundaryPairProposal(
        predicate_false=GeneratedCase(inputs=false_case["inputs"], output=False),
        predicate_true=GeneratedCase(inputs=true_case["inputs"], output=True),
    )
    assert len(runner.calls) == 3

    class AlwaysFailing(FakeRunner):
        def __call__(self, command: Sequence[str], **kwargs: Any) -> BoundedProcessResult:
            if tuple(command)[-1] == "--version":
                return BoundedProcessResult(0, f"{CLAUDE_CLI_VERSION}\n".encode(), b"")
            self.calls.append((tuple(command), kwargs))
            return BoundedProcessResult(1, b"", b"")

    failing = AlwaysFailing()
    with pytest.raises(TeacherTransportError, match="status 1"):
        ClaudeCliTrainingTeacher(process_runner=failing).generate_counterfactual(
            refund_ir(), GeneratedCase(inputs=false_case["inputs"], output=False)
        )
    assert len(failing.calls) == 3


def test_zero_generation_makes_no_process_call() -> None:
    runner = FakeRunner()

    assert ClaudeCliTrainingTeacher(process_runner=runner).generate(refund_ir(), 0) == ()
    assert runner.calls == []


def test_version_pin_is_checked_before_generation_and_cached() -> None:
    class WrongVersionRunner(FakeRunner):
        def __call__(self, command: Sequence[str], **kwargs: Any) -> BoundedProcessResult:
            if tuple(command)[-1] == "--version":
                return BoundedProcessResult(0, b"2.1.279 (Claude Code)\n", b"")
            raise AssertionError("model request must not run after a version mismatch")

    pinned = ClaudeCliTeacherConfig(cli_version=CLAUDE_CLI_VERSION)
    with pytest.raises(TeacherConfigurationError, match="does not match"):
        ClaudeCliTrainingTeacher(pinned, process_runner=WrongVersionRunner()).generate(
            refund_ir(), 1
        )

    recording = ClaudeCliTrainingTeacher(process_runner=WrongVersionRunner())
    recording.verify_installation()
    assert recording.provenance.cli_version == "2.1.279 (Claude Code)"

    runner = FakeRunner(
        [
            {"inputs": {"score": 1, "note": "one"}, "output": False},
            {"inputs": {"score": 2, "note": "two"}, "output": False},
        ]
    )
    teacher = ClaudeCliTrainingTeacher(process_runner=runner)
    teacher.generate(refund_ir(), 1)
    teacher.generate(refund_ir(), 1)
    assert sum(call[0][-1] == "--version" for call in runner.calls) == 1


@pytest.mark.parametrize(
    "wrapper, message",
    [
        ({}, "successful result"),
        (
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "num_turns": 9,
                "structured_output": {},
            },
            "within 8 turns",
        ),
        (
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "num_turns": 3,
                "modelUsage": {"claude-sonnet-5": {}, "claude-haiku-4-5-20251001": {}},
                "structured_output": {},
            },
            "pinned model",
        ),
        (
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "num_turns": 1,
                "permission_denials": [{"tool": "Read"}],
                "structured_output": {},
            },
            "denied capability",
        ),
        (
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "num_turns": 1,
                "modelUsage": {"claude-sonnet-5": {}},
                "structured_output": "{}",
            },
            "omitted structured output",
        ),
    ],
)
def test_rejects_unsafe_or_nonstructured_cli_result(
    wrapper: dict[str, Any],
    message: str,
) -> None:
    class ResultRunner(FakeRunner):
        def __call__(
            self,
            command: Sequence[str],
            **kwargs: Any,
        ) -> BoundedProcessResult:
            if tuple(command)[-1] == "--version":
                return BoundedProcessResult(0, f"{CLAUDE_CLI_VERSION}\n".encode(), b"")
            return BoundedProcessResult(0, json.dumps(wrapper).encode(), b"")

    with pytest.raises(TeacherResponseError, match=message):
        ClaudeCliTrainingTeacher(process_runner=ResultRunner()).generate(refund_ir(), 1)


def test_local_contract_rejects_structured_output_outside_ir_after_all_attempts() -> None:
    bad = {"inputs": {"score": "not-a-number", "note": "bad"}, "output": False}
    runner = FakeRunner([bad, bad, bad])
    teacher = ClaudeCliTrainingTeacher(process_runner=runner)

    with pytest.raises(TeacherResponseError, match=r"inputs\.score"):
        teacher.generate(refund_ir(), 1)
    assert len(runner.calls) == 4
    assert b"rejectedAttempts" not in runner.calls[1][1]["stdin"]
    assert b"rejectedAttempts" in runner.calls[3][1]["stdin"]
    assert b"not-a-number" in runner.calls[3][1]["stdin"]


def test_schema_rejection_is_retried_with_a_rejection_note() -> None:
    runner = FakeRunner(
        [
            {"inputs": {"score": "bad", "note": "x"}, "output": False},
            {"inputs": {"score": 4, "note": "fixed"}, "output": False},
        ]
    )
    teacher = ClaudeCliTrainingTeacher(process_runner=runner)

    assert teacher.generate(refund_ir(), 1) == (
        GeneratedCase(inputs={"score": 4, "note": "fixed"}, output=False),
    )
    assert len(runner.calls) == 3
    retry_stdin = runner.calls[2][1]["stdin"]
    assert b"rejectedAttempts" in retry_stdin
    assert b'"bad"' in retry_stdin
    report = teacher.last_run_report
    assert report is not None
    assert (report.positions, report.requests, report.schema_rejections) == (1, 2, 1)
    assert (report.constraint_rejections, report.residual_duplicates) == (0, 0)
    assert report.document()["schemaRejections"] == 1


def test_constraint_violation_is_retried_with_the_violation_reason() -> None:
    runner = FakeRunner(
        [
            {"inputs": {"score": 12, "note": "x"}, "output": False},
            {"inputs": {"score": 12, "note": "x"}, "output": True},
        ]
    )
    teacher = ClaudeCliTrainingTeacher(process_runner=runner)

    assert teacher.generate(refund_ir(), 1) == (
        GeneratedCase(inputs={"score": 12, "note": "x"}, output=True),
    )
    assert b"always constraint 0" in runner.calls[2][1]["stdin"]
    assert teacher.last_run_report is not None
    assert teacher.last_run_report.constraint_rejections == 1


def test_duplicate_inputs_are_re_requested_and_residuals_tolerated() -> None:
    same = {"inputs": {"score": 1, "note": "same"}, "output": False}
    other = {"inputs": {"score": 2, "note": "other"}, "output": False}
    runner = FakeRunner([same, same, other])
    teacher = ClaudeCliTrainingTeacher(process_runner=runner)

    assert teacher.generate(refund_ir(), 2) == (
        GeneratedCase(inputs={"score": 1, "note": "same"}, output=False),
        GeneratedCase(inputs={"score": 2, "note": "other"}, output=False),
    )
    assert b"duplicate another case" in runner.calls[3][1]["stdin"]
    report = teacher.last_run_report
    assert report is not None
    assert (
        report.duplicate_retries,
        report.duplicate_rounds,
        report.residual_duplicates,
        report.requests,
    ) == (1, 1, 0, 3)

    stubborn = FakeRunner([same] * 4)
    teacher = ClaudeCliTrainingTeacher(process_runner=stubborn)
    assert (
        teacher.generate(refund_ir(), 2)
        == (GeneratedCase(inputs={"score": 1, "note": "same"}, output=False),) * 2
    )
    report = teacher.last_run_report
    assert report is not None
    assert (
        report.duplicate_retries,
        report.duplicate_rounds,
        report.residual_duplicates,
        report.requests,
    ) == (2, 2, 1, 4)


class PositionRunner:
    """Thread-safe runner that answers by case position with shuffled latency."""

    def __init__(self, failing_position: int | None = None) -> None:
        self.lock = threading.Lock()
        self.calls = 0
        self.active = 0
        self.peak = 0
        self.failing_position = failing_position

    def __call__(
        self, command: Sequence[str], *, stdin: bytes, **kwargs: Any
    ) -> BoundedProcessResult:
        if tuple(command)[-1] == "--version":
            return BoundedProcessResult(0, f"{CLAUDE_CLI_VERSION}\n".encode(), b"")
        match = re.search(rb"Generate case (\d+) of (\d+)", stdin)
        assert match is not None
        position = int(match.group(1))
        with self.lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.02 * ((position * 3) % 4))
        with self.lock:
            self.active -= 1
        if position == self.failing_position:
            return BoundedProcessResult(9, b"", b"")
        output = {"inputs": {"score": position, "note": f"case {position}"}, "output": False}
        return BoundedProcessResult(0, envelope(output), b"")


def test_concurrent_positions_return_in_order() -> None:
    runner = PositionRunner()
    teacher = ClaudeCliTrainingTeacher(
        ClaudeCliTeacherConfig(concurrency=4),
        process_runner=runner,
    )

    cases = teacher.generate(refund_ir(), 8)

    assert [case.inputs["score"] for case in cases] == list(range(1, 9))
    assert runner.calls == 8
    assert runner.peak > 1
    assert teacher.last_run_report is not None
    assert teacher.last_run_report.requests == 8


def test_first_failure_stops_concurrent_generation() -> None:
    runner = PositionRunner(failing_position=3)
    teacher = ClaudeCliTrainingTeacher(
        ClaudeCliTeacherConfig(concurrency=3, maximum_case_attempts=2),
        process_runner=runner,
    )

    with pytest.raises(TeacherTransportError, match="status 9"):
        teacher.generate(refund_ir(), 6)
    assert runner.calls <= 8


@pytest.mark.parametrize(
    "field, value",
    [
        ("concurrency", 0),
        ("concurrency", 9),
        ("maximum_case_attempts", 0),
        ("maximum_case_attempts", 6),
        ("cli_version", ""),
        ("cli_version", "two\nlines"),
    ],
)
def test_rejects_out_of_range_generation_settings(field: str, value: int) -> None:
    with pytest.raises(TeacherConfigurationError):
        ClaudeCliTeacherConfig(**{field: value})


def test_transport_errors_do_not_echo_stderr_prompt_or_structured_output() -> None:
    secret = "do-not-echo-this-secret"

    class FailedRunner(FakeRunner):
        def __call__(self, command: Sequence[str], **kwargs: Any) -> BoundedProcessResult:
            if tuple(command)[-1] == "--version":
                return BoundedProcessResult(0, f"{CLAUDE_CLI_VERSION}\n".encode(), b"")
            return BoundedProcessResult(17, secret.encode(), secret.encode())

    with pytest.raises(TeacherTransportError) as captured:
        ClaudeCliTrainingTeacher(process_runner=FailedRunner()).generate(refund_ir(), 1)
    assert "status 17" in str(captured.value)
    assert secret not in str(captured.value)

    class ExplodingRunner(FakeRunner):
        def __call__(self, command: Sequence[str], **kwargs: Any) -> BoundedProcessResult:
            raise RuntimeError(secret)

    with pytest.raises(TeacherTransportError) as captured:
        ClaudeCliTrainingTeacher(process_runner=ExplodingRunner()).verify_installation()
    assert "RuntimeError" in str(captured.value)
    assert secret not in str(captured.value)


@pytest.mark.parametrize(
    "config",
    [
        ClaudeCliTeacherConfig(timeout_seconds=1),
        ClaudeCliTeacherConfig(stdout_limit_bytes=1024),
        ClaudeCliTeacherConfig(stderr_limit_bytes=1024),
        ClaudeCliTeacherConfig(max_budget_usd=0.5),
        ClaudeCliTeacherConfig(concurrency=2),
        ClaudeCliTeacherConfig(maximum_case_attempts=1),
        ClaudeCliTeacherConfig(cli_version="9.9.9 (Claude Code)"),
    ],
)
def test_behavior_affecting_config_changes_digest(config: ClaudeCliTeacherConfig) -> None:
    assert config.configuration_sha256 != ClaudeCliTeacherConfig().configuration_sha256


def test_bounded_process_caps_both_streams_and_times_out() -> None:
    ok = run_bounded_process(
        (sys.executable, "-c", "import sys; sys.stdout.write('ok'); sys.stderr.write('warn')"),
        stdin=b"",
        timeout_seconds=2,
        stdout_limit=16,
        stderr_limit=16,
    )
    assert ok == BoundedProcessResult(0, b"ok", b"warn")

    with pytest.raises(RuntimeError, match="stdout exceeded"):
        run_bounded_process(
            (sys.executable, "-c", "print('x' * 10000)"),
            stdin=b"",
            timeout_seconds=2,
            stdout_limit=32,
            stderr_limit=32,
        )
    with pytest.raises(RuntimeError, match="stderr exceeded"):
        run_bounded_process(
            (sys.executable, "-c", "import sys; sys.stderr.write('x' * 10000)"),
            stdin=b"",
            timeout_seconds=2,
            stdout_limit=32,
            stderr_limit=32,
        )
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timeout"):
        run_bounded_process(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            stdin=b"",
            timeout_seconds=0.1,
            stdout_limit=32,
            stderr_limit=32,
        )
    assert time.monotonic() - started < 3


def test_bounded_process_terminates_descendant_process(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    script = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
    )
    with pytest.raises(RuntimeError, match="timeout"):
        run_bounded_process(
            (sys.executable, "-c", script, str(pid_path)),
            stdin=b"",
            timeout_seconds=0.3,
            stdout_limit=64,
            stderr_limit=64,
        )
    child_pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and process_exists(child_pid):
        time.sleep(0.02)
    assert not process_exists(child_pid)


def test_bounded_process_passes_empty_arguments_through() -> None:
    result = run_bounded_process(
        (sys.executable, "-c", "import sys; sys.stdout.write(repr(sys.argv[1:]))", "", "x"),
        stdin=b"",
        timeout_seconds=5,
        stdout_limit=64,
        stderr_limit=64,
    )
    assert result == BoundedProcessResult(0, b"['', 'x']", b"")

    with pytest.raises(RuntimeError, match="command is invalid"):
        run_bounded_process(
            ("", "-c", "pass"),
            stdin=b"",
            timeout_seconds=5,
            stdout_limit=64,
            stderr_limit=64,
        )


def test_real_runner_accepts_the_production_command(tmp_path: Path) -> None:
    """Drive the real bounded runner with a stub CLI so ``--tools ""`` is exercised."""

    stub = tmp_path / "claude-stub"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "argv = sys.argv[1:]\n"
        "if argv == ['--version']:\n"
        f"    print({CLAUDE_CLI_VERSION!r}); raise SystemExit(0)\n"
        "assert argv[argv.index('--tools') + 1] == ''\n"
        "assert argv[argv.index('--setting-sources') + 1] == ''\n"
        "assert sys.stdin.read().startswith('Generate case 1 of 1')\n"
        "print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False,\n"
        "    'num_turns': 3, 'permission_denials': [], 'modelUsage': {'claude-sonnet-5': {}},\n"
        "    'structured_output': {'inputs': {'score': 3, 'note': 'stub'}, 'output': False}}))\n"
    )
    stub.chmod(0o700)
    teacher = ClaudeCliTrainingTeacher(
        ClaudeCliTeacherConfig(executable=str(stub), timeout_seconds=20)
    )

    assert teacher.generate(refund_ir(), 1) == (
        GeneratedCase(inputs={"score": 3, "note": "stub"}, output=False),
    )


def option(command: Sequence[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status_path = Path(f"/proc/{pid}/status")
    if status_path.exists() and "State:\tZ" in status_path.read_text():
        return False
    return True


def refund_ir() -> dict[str, Any]:
    return {
        "kind": "semantscript.neural-function",
        "irVersion": 1,
        "stage": "lowered",
        "id": "nf_" + "7" * 64,
        "semanticSha256": "8" * 64,
        "source": {
            "path": "refund-with-confidence.sem.ts",
            "line": 1,
            "column": 1,
            "sourceSha256": "9" * 64,
        },
        "definition": {
            "template": [{"kind": "text", "text": "Classify refund."}],
            "examples": [],
            "constraints": [
                {
                    "kind": "always",
                    "source": "score >= 10",
                    "predicate": {
                        "node": "binary",
                        "operator": ">=",
                        "left": {"node": "input", "name": "score"},
                        "right": {"node": "literal", "value": 10},
                    },
                    "output": True,
                }
            ],
        },
        "inputs": [
            {"name": "score", "index": 0, "tsType": "number", "type": {"kind": "number"}},
            {"name": "note", "index": 1, "tsType": "string", "type": {"kind": "string"}},
        ],
        "output": {
            "kind": "scalar",
            "tsType": "boolean",
            "head": {
                "kind": "nominal",
                "sourceKind": "boolean",
                "support": [False, True],
            },
        },
    }
