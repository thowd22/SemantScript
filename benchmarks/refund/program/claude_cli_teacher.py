"""Training-only Claude Code CLI teacher for the refund benchmark.

This module deliberately does not produce benchmark or human-authored records.  Its
``GeneratedCase`` values are accepted only by the trainer's synthetic/adversarial
dataset builders, whose persisted records retain those origins.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import selectors
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

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
from semantscript_trainer.strict_json import StrictJsonError, StrictJsonLimits, loads_strict_json
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
from semantscript_trainer.teacher_prompt import build_case_messages

CLAUDE_CLI_MODEL = "claude-sonnet-5"
CLAUDE_CLI_VERSION = "2.1.280 (Claude Code)"
CLAUDE_CLI_PROTOCOL = "semantscript.refund-training.claude-cli/v1"
CLAUDE_CLI_DATA_CLASSIFICATION = "synthetic-training-only"

_PROVIDER = "anthropic-claude-cli-training-only"
_PROMPT_PROTOCOL = "semantscript-trainer-ir-prompts/v1"
_MAXIMUM_STDIN_BYTES = 8 * 1024 * 1024
_MAXIMUM_STDOUT_BYTES = 8 * 1024 * 1024
_MAXIMUM_STDERR_BYTES = 64 * 1024
_MAXIMUM_VERSION_OUTPUT_BYTES = 4 * 1024
_MAXIMUM_TIMEOUT_SECONDS = 3_600.0
_PROCESS_DRAIN_SECONDS = 1.0
_MCP_CONFIG = '{"mcpServers":{}}'
# One structured-output exchange is a thinking/text block, a StructuredOutput tool
# call, and its tool result. The pinned CLI's turn accounting varies with the
# response shape (2 and 3 observed for a single exchange), so this bound is only a
# guard against runaway re-prompting. The no-agent guarantees are the disabled
# tools, empty permission denials, and the pinned ``modelUsage`` identity.
_MAXIMUM_RESULT_TURNS = 8


@dataclass(frozen=True, slots=True)
class ClaudeCliTeacherConfig:
    """Secret-free, behavior-affecting configuration for training generation."""

    executable: str = "claude"
    timeout_seconds: float = 600.0
    stdout_limit_bytes: int = _MAXIMUM_STDOUT_BYTES
    stderr_limit_bytes: int = _MAXIMUM_STDERR_BYTES
    max_budget_usd: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.executable, str) or not self.executable.strip():
            raise TeacherConfigurationError("Claude CLI executable must be a nonempty string")
        if "\x00" in self.executable:
            raise TeacherConfigurationError("Claude CLI executable must not contain NUL")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or not 0 < float(self.timeout_seconds) <= _MAXIMUM_TIMEOUT_SECONDS
        ):
            raise TeacherConfigurationError(
                f"Claude CLI timeout must be finite and at most {_MAXIMUM_TIMEOUT_SECONDS:g} seconds"
            )
        _bounded_positive_integer(
            self.stdout_limit_bytes,
            "Claude CLI stdout limit",
            _MAXIMUM_STDOUT_BYTES,
        )
        _bounded_positive_integer(
            self.stderr_limit_bytes,
            "Claude CLI stderr limit",
            _MAXIMUM_STDERR_BYTES,
        )
        if self.stdout_limit_bytes > MAXIMUM_TEACHER_RESPONSE_BYTES:
            raise TeacherConfigurationError(
                "Claude CLI stdout limit exceeds the trainer teacher-response limit"
            )
        if (
            isinstance(self.max_budget_usd, bool)
            or not isinstance(self.max_budget_usd, (int, float))
            or not math.isfinite(float(self.max_budget_usd))
            or not 0 < float(self.max_budget_usd) <= 100
        ):
            raise TeacherConfigurationError(
                "Claude CLI maximum budget must be finite and between 0 and 100 USD"
            )

    def public_projection(self) -> dict[str, Any]:
        """Return the complete, secret-free protocol record hashed into provenance."""

        return {
            "kind": "semantscript.refund-claude-cli-teacher-config",
            "configVersion": 1,
            "dataClassification": CLAUDE_CLI_DATA_CLASSIFICATION,
            "protocol": CLAUDE_CLI_PROTOCOL,
            "promptProtocol": _PROMPT_PROTOCOL,
            "cli": {
                "executable": self.executable,
                "exactVersion": CLAUDE_CLI_VERSION,
            },
            "request": {
                "model": CLAUDE_CLI_MODEL,
                "effort": "high",
                "print": True,
                "inputFormat": "text",
                "outputFormat": "json",
                "structuredOutput": "ir-derived-json-schema",
                "tools": [],
                "skills": False,
                "sessionPersistence": False,
                "safeMode": True,
                "restrictedMode": True,
                "settingSources": [],
                "mcpServers": {},
                "permissionMode": "dontAsk",
                "permissionPrompts": "none",
                "promptSuggestions": False,
                "chrome": False,
                "fallbackModel": None,
                "maxBudgetUsd": float(self.max_budget_usd),
            },
            "limits": {
                "timeoutSeconds": float(self.timeout_seconds),
                "stdinBytes": _MAXIMUM_STDIN_BYTES,
                "stdoutBytes": self.stdout_limit_bytes,
                "stderrBytes": self.stderr_limit_bytes,
            },
        }

    @property
    def configuration_sha256(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.public_projection())).hexdigest()


@dataclass(frozen=True, slots=True)
class ClaudeCliTeacherProvenance:
    """Exact training-only identity suitable for run manifests and review."""

    provider: str
    model: str
    cli_version: str
    protocol: str
    data_classification: str
    configuration_sha256: str


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    """A completed process with byte-bounded captured streams."""

    returncode: int
    stdout: bytes
    stderr: bytes


class ClaudeCliProcessRunner(Protocol):
    """Injectable process boundary; test runners must not contact Claude."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        stdin: bytes,
        timeout_seconds: float,
        stdout_limit: int,
        stderr_limit: int,
    ) -> BoundedProcessResult: ...


class ClaudeCliTrainingTeacher:
    """Claude Sonnet 5 teacher restricted to synthetic/adversarial training data."""

    def __init__(
        self,
        config: ClaudeCliTeacherConfig | None = None,
        *,
        process_runner: ClaudeCliProcessRunner | None = None,
    ) -> None:
        config = config or ClaudeCliTeacherConfig()
        if not isinstance(config, ClaudeCliTeacherConfig):
            raise TeacherConfigurationError("Claude CLI teacher config is invalid")
        self._config = config
        self._process_runner = process_runner or run_bounded_process
        self._version_verified = False

    @property
    def descriptor(self) -> TeacherDescriptor:
        return TeacherDescriptor(
            provider=_PROVIDER,
            model=CLAUDE_CLI_MODEL,
            configuration_sha256=self._config.configuration_sha256,
        )

    @property
    def provenance(self) -> ClaudeCliTeacherProvenance:
        return ClaudeCliTeacherProvenance(
            provider=_PROVIDER,
            model=CLAUDE_CLI_MODEL,
            cli_version=CLAUDE_CLI_VERSION,
            protocol=CLAUDE_CLI_PROTOCOL,
            data_classification=CLAUDE_CLI_DATA_CLASSIFICATION,
            configuration_sha256=self._config.configuration_sha256,
        )

    @property
    def configuration_projection(self) -> dict[str, Any]:
        """Return the public configuration whose digest appears in provenance."""

        return self._config.public_projection()

    def verify_installation(self) -> None:
        """Verify the exact CLI build without making a model request."""

        if self._version_verified:
            return
        result = self._run(
            (self._config.executable, "--version"),
            stdin=b"",
            timeout_seconds=min(float(self._config.timeout_seconds), 15.0),
            stdout_limit=_MAXIMUM_VERSION_OUTPUT_BYTES,
            stderr_limit=_MAXIMUM_VERSION_OUTPUT_BYTES,
            context="version check",
        )
        if result.returncode != 0:
            raise TeacherConfigurationError(
                f"Claude CLI version check exited with status {result.returncode}"
            )
        try:
            observed = result.stdout.decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError as error:
            raise TeacherConfigurationError("Claude CLI returned a non-UTF-8 version") from error
        if observed != CLAUDE_CLI_VERSION:
            raise TeacherConfigurationError(
                "Claude CLI version does not match the benchmark pin "
                f"{CLAUDE_CLI_VERSION!r}; observed {observed!r}"
            )
        self._version_verified = True

    def generate(self, ir: NeuralFunctionIr, n: int, /) -> tuple[GeneratedCase, ...]:
        expected = validate_case_count(n)
        if expected == 0:
            return ()
        result: list[GeneratedCase] = []
        aggregate_bytes = 0
        for index in range(expected):
            try:
                system, user = build_case_messages(ir, index, expected)
                schema = build_case_schema(ir)
            except TeacherConfigurationError:
                raise
            except Exception as error:
                raise TeacherConfigurationError(
                    f"could not build Claude CLI training request: {error}"
                ) from error
            structured, response_bytes = self._request(system, user, schema, "training case")
            aggregate_bytes += response_bytes
            if aggregate_bytes > MAXIMUM_TEACHER_RESPONSE_BYTES:
                raise TeacherResponseError(
                    "Claude CLI responses exceed the aggregate trainer byte limit"
                )
            result.append(parse_case_response(ir, _canonical_json_bytes(structured)))
        return tuple(result)

    def generate_boundary_pair(
        self,
        ir: NeuralFunctionIr,
        constraint_index: int,
        /,
    ) -> BoundaryPairProposal:
        try:
            system, user = build_boundary_messages(ir, constraint_index)
            schema = build_boundary_pair_schema(ir)
        except TeacherConfigurationError:
            raise
        except Exception as error:
            raise TeacherConfigurationError(
                f"could not build Claude CLI boundary request: {error}"
            ) from error
        structured, _ = self._request(system, user, schema, "boundary pair")
        return parse_boundary_pair_response(ir, _canonical_json_bytes(structured))

    def generate_counterfactual(
        self,
        ir: NeuralFunctionIr,
        anchor: GeneratedCase,
        /,
    ) -> CounterfactualProposal:
        try:
            system, user = build_counterfactual_messages(ir, anchor)
            schema = build_counterfactual_schema(ir)
        except TeacherConfigurationError:
            raise
        except Exception as error:
            raise TeacherConfigurationError(
                f"could not build Claude CLI counterfactual request: {error}"
            ) from error
        structured, _ = self._request(system, user, schema, "counterfactual")
        return parse_counterfactual_response(ir, _canonical_json_bytes(structured))

    def _request(
        self,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        context: str,
    ) -> tuple[dict[str, Any], int]:
        self.verify_installation()
        schema_json = _canonical_json_string(schema)
        command = self._command(system, schema_json)
        try:
            stdin = user.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise TeacherConfigurationError("Claude CLI prompt contains invalid Unicode") from error
        if len(stdin) > _MAXIMUM_STDIN_BYTES:
            raise TeacherConfigurationError("Claude CLI prompt exceeds its byte limit")
        completed = self._run(
            command,
            stdin=stdin,
            timeout_seconds=float(self._config.timeout_seconds),
            stdout_limit=self._config.stdout_limit_bytes,
            stderr_limit=self._config.stderr_limit_bytes,
            context=context,
        )
        if completed.returncode != 0:
            raise TeacherTransportError(
                f"Claude CLI {context} exited with status {completed.returncode}"
            )
        return _parse_cli_result(completed.stdout, context), len(completed.stdout)

    def _command(self, system: str, schema_json: str) -> tuple[str, ...]:
        return (
            self._config.executable,
            "--print",
            "--model",
            CLAUDE_CLI_MODEL,
            "--effort",
            "high",
            "--input-format",
            "text",
            "--output-format",
            "json",
            "--json-schema",
            schema_json,
            "--system-prompt",
            system,
            "--no-session-persistence",
            "--tools",
            "",
            "--disable-slash-commands",
            "--safe-mode",
            "--restricted",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            _MCP_CONFIG,
            "--permission-mode",
            "dontAsk",
            "--permission-prompts",
            "none",
            "--prompt-suggestions",
            "false",
            "--no-chrome",
            "--max-budget-usd",
            _canonical_budget(self._config.max_budget_usd),
        )

    def _run(
        self,
        command: Sequence[str],
        *,
        stdin: bytes,
        timeout_seconds: float,
        stdout_limit: int,
        stderr_limit: int,
        context: str,
    ) -> BoundedProcessResult:
        try:
            result = self._process_runner(
                command,
                stdin=stdin,
                timeout_seconds=timeout_seconds,
                stdout_limit=stdout_limit,
                stderr_limit=stderr_limit,
            )
        except _BoundedProcessError as error:
            raise TeacherTransportError(f"Claude CLI {context} failed: {error.reason}") from error
        except TeacherTransportError:
            raise
        except Exception as error:
            raise TeacherTransportError(
                f"Claude CLI {context} process runner failed ({type(error).__name__})"
            ) from error
        if not isinstance(result, BoundedProcessResult):
            raise TeacherTransportError("Claude CLI process runner returned an invalid result")
        return result


class _BoundedProcessError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def run_bounded_process(
    command: Sequence[str],
    *,
    stdin: bytes,
    timeout_seconds: float,
    stdout_limit: int,
    stderr_limit: int,
) -> BoundedProcessResult:
    """Run in an empty cwd, cap both streams, and terminate the whole process group."""

    if (
        isinstance(command, (str, bytes))
        or not isinstance(command, Sequence)
        or not command
        or not isinstance(command[0], str)
        or not command[0]
        or any(not isinstance(part, str) or "\x00" in part for part in command)
    ):
        # Empty arguments after the executable are legal and required: the CLI
        # documents ``--tools ""`` as the way to disable every built-in tool.
        raise _BoundedProcessError("command is invalid")
    if not isinstance(stdin, bytes) or len(stdin) > _MAXIMUM_STDIN_BYTES:
        raise _BoundedProcessError("stdin exceeds its byte limit")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise _BoundedProcessError("timeout is invalid")
    _bounded_positive_integer(stdout_limit, "stdout limit", _MAXIMUM_STDOUT_BYTES)
    _bounded_positive_integer(stderr_limit, "stderr limit", _MAXIMUM_STDERR_BYTES)

    environment = dict(os.environ)
    environment.update(
        {
            "CLAUDE_CODE_SAFE_MODE": "1",
            "NO_COLOR": "1",
        }
    )
    process: subprocess.Popen[bytes] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="semantscript-claude-cli-") as cwd:
            try:
                process = subprocess.Popen(
                    tuple(command),
                    cwd=Path(cwd),
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                    close_fds=True,
                )
            except OSError as error:
                raise _BoundedProcessError(
                    f"process could not start ({type(error).__name__})"
                ) from error
            return _exchange_bounded(
                process,
                stdin,
                timeout_seconds=float(timeout_seconds),
                stdout_limit=stdout_limit,
                stderr_limit=stderr_limit,
            )
    finally:
        if process is not None:
            _terminate_process_group(process)


def _exchange_bounded(
    process: subprocess.Popen[bytes],
    stdin: bytes,
    *,
    timeout_seconds: float,
    stdout_limit: int,
    stderr_limit: int,
) -> BoundedProcessResult:
    if process.stdin is None or process.stdout is None or process.stderr is None:
        _terminate_process_group(process)
        raise _BoundedProcessError("process pipes were not created")
    streams = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    try:
        for stream in (process.stdin, process.stdout, process.stderr):
            os.set_blocking(stream.fileno(), False)
        streams.register(process.stdout, selectors.EVENT_READ, "stdout")
        streams.register(process.stderr, selectors.EVENT_READ, "stderr")
        input_offset = 0
        if stdin:
            streams.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        else:
            process.stdin.close()

        deadline = time.monotonic() + timeout_seconds
        while streams.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process_group(process)
                raise _BoundedProcessError("request exceeded its timeout")
            for key, _ in streams.select(min(remaining, 0.25)):
                stream = cast(Any, key.fileobj)
                label = cast(str, key.data)
                if label == "stdin":
                    try:
                        count = os.write(
                            stream.fileno(), stdin[input_offset : input_offset + 65536]
                        )
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        count = len(stdin) - input_offset
                    input_offset += count
                    if input_offset >= len(stdin):
                        streams.unregister(stream)
                        stream.close()
                    continue

                try:
                    chunk = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    streams.unregister(stream)
                    stream.close()
                    continue
                limit = stdout_limit if label == "stdout" else stderr_limit
                if len(buffers[label]) + len(chunk) > limit:
                    _terminate_process_group(process)
                    raise _BoundedProcessError(f"{label} exceeded its byte limit")
                buffers[label].extend(chunk)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process_group(process)
            raise _BoundedProcessError("request exceeded its timeout")
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            _terminate_process_group(process)
            raise _BoundedProcessError("request exceeded its timeout") from error
        return BoundedProcessResult(
            returncode=returncode,
            stdout=bytes(buffers["stdout"]),
            stderr=bytes(buffers["stderr"]),
        )
    finally:
        streams.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            if process.poll() is None:
                process.wait(timeout=_PROCESS_DRAIN_SECONDS)
            return
        deadline = time.monotonic() + _PROCESS_DRAIN_SECONDS
        while time.monotonic() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.wait(timeout=_PROCESS_DRAIN_SECONDS)
        return

    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=_PROCESS_DRAIN_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_PROCESS_DRAIN_SECONDS)


def _parse_cli_result(response: bytes, context: str) -> dict[str, Any]:
    limits = StrictJsonLimits(
        maximum_bytes=_MAXIMUM_STDOUT_BYTES,
        maximum_depth=64,
        maximum_nodes=500_000,
    )
    try:
        value = loads_strict_json(response, limits=limits)
    except StrictJsonError as error:
        raise TeacherResponseError(f"Claude CLI {context} returned invalid JSON") from error
    if not isinstance(value, dict):
        raise TeacherResponseError(f"Claude CLI {context} result must be an object")
    if value.get("type") != "result" or value.get("subtype") != "success":
        raise TeacherResponseError(f"Claude CLI {context} did not return a successful result")
    if value.get("is_error") is not False:
        raise TeacherResponseError(f"Claude CLI {context} marked the result as an error")
    turns = value.get("num_turns")  # strict JSON yields binary64 floats for integers
    if (
        isinstance(turns, bool)
        or not isinstance(turns, (int, float))
        or not float(turns).is_integer()
        or not 1 <= turns <= _MAXIMUM_RESULT_TURNS
    ):
        raise TeacherResponseError(
            f"Claude CLI {context} must complete within {_MAXIMUM_RESULT_TURNS} turns"
        )
    permission_denials = value.get("permission_denials")
    if permission_denials not in (None, []):
        raise TeacherResponseError(f"Claude CLI {context} attempted a denied capability")
    model_usage = value.get("modelUsage")
    if not isinstance(model_usage, dict) or set(model_usage) != {CLAUDE_CLI_MODEL}:
        raise TeacherResponseError(f"Claude CLI {context} was not served by the pinned model")
    structured = value.get("structured_output")
    if not isinstance(structured, dict):
        raise TeacherResponseError(f"Claude CLI {context} omitted structured output")
    return structured


def _canonical_json_string(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError, UnicodeEncodeError, RecursionError) as error:
        raise TeacherConfigurationError(
            "Claude CLI structured-output schema is not canonical JSON"
        ) from error


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return _canonical_json_string(value).encode("utf-8", errors="strict")


def _canonical_budget(value: float) -> str:
    return format(float(value), ".15g")


def _bounded_positive_integer(value: int, name: str, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= maximum:
        raise TeacherConfigurationError(f"{name} must be an integer from 1 through {maximum}")


__all__ = [
    "CLAUDE_CLI_DATA_CLASSIFICATION",
    "CLAUDE_CLI_MODEL",
    "CLAUDE_CLI_PROTOCOL",
    "CLAUDE_CLI_VERSION",
    "BoundedProcessResult",
    "ClaudeCliProcessRunner",
    "ClaudeCliTeacherConfig",
    "ClaudeCliTeacherProvenance",
    "ClaudeCliTrainingTeacher",
    "run_bounded_process",
]
