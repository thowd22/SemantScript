"""The environment checks behind ``semantscript doctor`` and the train preflight."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from semantscript_trainer import cli as cli_module
from semantscript_trainer.doctor import (
    CHECK_IDS,
    REPORT_KIND,
    Check,
    ProbeResult,
    probe_teacher,
    render_report,
    run_doctor,
    run_import_probe,
)
from semantscript_trainer.teacher_config import TeacherConfig

GPU_OK: dict[str, Any] = {
    "torch": {
        "ok": True,
        "version": "2.9.1+rocm7.2",
        "hip": "7.2.1",
        "cuda": None,
        "cudaAvailable": True,
        "device": {"name": "Radeon", "count": 1, "free": 12 * 1024**3, "total": 16 * 1024**3},
        "mps": False,
    },
    "transformers": {"ok": True, "version": "5.17.0"},
    "onnxruntime": {
        "ok": True,
        "version": "1.30.0",
        "providers": ["CPUExecutionProvider"],
        "onnx": "1.23.0",
    },
}
BROKEN_NUMPY = {
    **GPU_OK,
    "transformers": {
        "ok": False,
        "error": "AttributeError: module 'numpy' has no attribute 'long'",
        "missing": False,
    },
}
CPU_ONLY = {
    **GPU_OK,
    "torch": {
        "ok": True,
        "version": "2.9.1+cpu",
        "hip": None,
        "cuda": None,
        "cudaAvailable": False,
        "mps": False,
    },
}
NO_TORCH = {
    "torch": {
        "ok": False,
        "error": "ModuleNotFoundError: No module named 'torch'",
        "missing": True,
    },
    "transformers": {
        "ok": False,
        "error": "ModuleNotFoundError: No module named 'transformers'",
        "missing": True,
    },
    "onnxruntime": {
        "ok": False,
        "error": "ModuleNotFoundError: No module named 'onnxruntime'",
        "missing": True,
    },
}


class FakeProbe:
    """Answers each import probe from the environment it was given."""

    def __init__(self, choose: Any) -> None:
        self.choose = choose
        self.calls: list[tuple[dict[str, str], tuple[str, ...]]] = []

    def __call__(self, env: Mapping[str, str], sections: Sequence[str]) -> dict[str, Any]:
        self.calls.append((dict(env), tuple(sections)))
        result = self.choose(env)
        return {section: result[section] for section in sections}


def by_id(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {check["id"]: check for check in report["checks"]}


def doctor(**kwargs: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "env": {},
        "check_teacher": False,
        "is_wsl": False,
        "user_site_exists": False,
    }
    return run_doctor(**{**defaults, **kwargs})


def test_report_is_closed_and_ordered() -> None:
    report = doctor(import_probe=FakeProbe(lambda env: GPU_OK))
    assert report["kind"] == REPORT_KIND
    assert report["reportVersion"] == 1
    assert [check["id"] for check in report["checks"]] == list(CHECK_IDS)
    for check in report["checks"]:
        assert set(check) == {"id", "status", "summary", "fix"}
    checks = by_id(report)
    assert checks["python"]["status"] == "pass"
    assert checks["trainer"]["status"] == "pass"
    assert checks["model"]["status"] == "pass"
    assert "ROCm 7.2.1" in checks["torch"]["summary"]
    assert "transformers 5.17.0" in checks["torch"]["summary"]
    assert checks["device"]["summary"] == (
        "trains on ROCm device Radeon: 16.0 GiB total, 12.0 GiB free"
    )
    assert checks["onnxruntime"]["status"] == "pass"
    assert checks["platform-env"] == {
        "id": "platform-env",
        "status": "pass",
        "summary": "no extra environment variables needed",
        "fix": None,
    }
    assert checks["teacher-probe"]["status"] == "skip"
    with pytest.raises(ValueError, match="unknown doctor check"):
        Check("gpu", "pass", "x")


def test_missing_training_extra_fails_with_the_install_fix() -> None:
    checks = by_id(doctor(import_probe=FakeProbe(lambda env: NO_TORCH)))
    assert checks["torch"]["status"] == "fail"
    assert "pip install -e '.[training]'" in checks["torch"]["fix"]
    assert checks["device"]["status"] == "skip"
    assert checks["onnxruntime"]["status"] == "fail"
    assert checks["platform-env"]["status"] == "pass"


def test_cpu_and_requested_devices() -> None:
    cpu = by_id(doctor(import_probe=FakeProbe(lambda env: CPU_ONLY)))
    assert cpu["device"]["status"] == "warn"
    assert "trains on the CPU" in cpu["device"]["summary"]
    assert "pytorch.org" in cpu["device"]["fix"]
    requested_cpu = by_id(doctor(import_probe=FakeProbe(lambda env: CPU_ONLY), device="cpu"))
    assert requested_cpu["device"]["status"] == "pass"
    cuda = by_id(doctor(import_probe=FakeProbe(lambda env: CPU_ONLY), device="cuda"))
    assert cuda["device"]["status"] == "fail"
    unknown = by_id(doctor(import_probe=FakeProbe(lambda env: GPU_OK), device="mps"))
    assert unknown["device"]["status"] == "fail"
    assert "auto, cpu or cuda" in unknown["device"]["fix"]
    broken_query = {**GPU_OK, "torch": {**GPU_OK["torch"], "deviceError": "RuntimeError: x"}}
    del broken_query["torch"]["device"]
    queried = by_id(doctor(import_probe=FakeProbe(lambda env: broken_query)))
    assert queried["torch"]["status"] == "pass"
    assert queried["device"]["status"] == "fail"
    assert "RuntimeError: x" in queried["device"]["summary"]
    mps = {**CPU_ONLY, "torch": {**CPU_ONLY["torch"], "mps": True}}
    apple = by_id(doctor(import_probe=FakeProbe(lambda env: mps)))
    assert apple["device"]["status"] == "warn"
    assert "MPS" in apple["device"]["summary"]


def test_user_site_that_breaks_imports_names_pythonnousersite() -> None:
    probe = FakeProbe(lambda env: GPU_OK if env.get("PYTHONNOUSERSITE") else BROKEN_NUMPY)
    checks = by_id(doctor(import_probe=probe, user_site_exists=True))
    assert checks["platform-env"]["status"] == "fail"
    assert "PYTHONNOUSERSITE=1 needed" in checks["platform-env"]["summary"]
    assert checks["platform-env"]["fix"].startswith("export PYTHONNOUSERSITE=1")
    assert checks["torch"]["status"] == "fail"
    assert "numpy" in checks["torch"]["summary"]
    assert checks["torch"]["fix"] == "export PYTHONNOUSERSITE=1 (see platform-env)"
    assert len(probe.calls) == 2

    # Already set: the full doctor confirms it is still needed; the quick preflight does not.
    confirmed = by_id(
        doctor(import_probe=probe, user_site_exists=True, env={"PYTHONNOUSERSITE": "1"})
    )
    assert confirmed["platform-env"]["status"] == "pass"
    assert "PYTHONNOUSERSITE=1 is set and needed" in confirmed["platform-env"]["summary"]
    quick_probe = FakeProbe(probe.choose)
    doctor(
        import_probe=quick_probe, user_site_exists=True, env={"PYTHONNOUSERSITE": "1"}, quick=True
    )
    assert len(quick_probe.calls) == 1
    quick = by_id(
        doctor(import_probe=probe, user_site_exists=True, env={"PYTHONNOUSERSITE": "1"}, quick=True)
    )
    assert quick["platform-env"]["summary"] == (
        "PYTHONNOUSERSITE=1 set (semantscript doctor checks whether it is needed)"
    )


def test_wsl_rocm_names_hsa_enable_dxg_detection_only_when_it_changes_the_outcome() -> None:
    hidden = {**GPU_OK, "torch": {**GPU_OK["torch"], "cudaAvailable": False}}
    hidden["torch"].pop("device")
    needs = FakeProbe(lambda env: GPU_OK if env.get("HSA_ENABLE_DXG_DETECTION") == "1" else hidden)
    checks = by_id(doctor(import_probe=needs, is_wsl=True))
    assert checks["platform-env"]["status"] == "fail"
    assert checks["platform-env"]["fix"].startswith("export HSA_ENABLE_DXG_DETECTION=1")
    assert checks["device"]["status"] == "warn"
    assert checks["device"]["fix"] == "export HSA_ENABLE_DXG_DETECTION=1 (see platform-env)"

    either = FakeProbe(lambda env: GPU_OK)
    set_anyway = by_id(
        doctor(import_probe=either, is_wsl=True, env={"HSA_ENABLE_DXG_DETECTION": "1"})
    )
    assert set_anyway["platform-env"]["status"] == "pass"
    assert "with or without" in set_anyway["platform-env"]["summary"]

    not_wsl = FakeProbe(lambda env: hidden)
    doctor(import_probe=not_wsl, is_wsl=False)
    assert len(not_wsl.calls) == 1


def write_teacher(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "teacher.toml"
    path.write_text(f"[teacher]\n{body}", encoding="utf-8")
    return path


def teacher_checks(tmp_path: Path, body: str | None, **kwargs: Any) -> dict[str, dict[str, Any]]:
    path = write_teacher(tmp_path, body) if body is not None else kwargs.pop("path", None)
    return by_id(
        doctor(
            import_probe=FakeProbe(lambda env: GPU_OK),
            check_teacher=True,
            teacher_path=path,
            **kwargs,
        )
    )


def test_teacher_config_and_key(tmp_path: Path) -> None:
    missing = teacher_checks(tmp_path, None)
    assert missing["teacher-config"]["status"] == "fail"
    assert "ANTHROPIC_API_KEY" in missing["teacher-config"]["fix"]
    assert missing["teacher-key"]["status"] == "skip"

    invalid = teacher_checks(tmp_path, 'backend = "gpt"\nmodel = "m"\n')
    assert invalid["teacher-config"]["status"] == "fail"
    assert "unsupported teacher backend" in invalid["teacher-config"]["summary"]

    absent = teacher_checks(tmp_path, None, path=tmp_path / "nowhere.toml")
    assert absent["teacher-config"]["summary"].endswith("does not exist")

    openrouter = 'backend = "anthropic"\nmodel = "anthropic/claude-sonnet-5"\n'
    openrouter += 'base_url = "https://openrouter.ai/api"\n'
    wrong_key = teacher_checks(tmp_path, openrouter, env={"OPENROUTER_API_KEY": "sk-or-x"})
    assert wrong_key["teacher-config"]["status"] == "pass"
    assert "via openrouter.ai" in wrong_key["teacher-config"]["summary"]
    assert wrong_key["teacher-key"]["status"] == "fail"
    assert wrong_key["teacher-key"]["fix"].startswith(
        'export ANTHROPIC_API_KEY="$OPENROUTER_API_KEY"'
    )
    assert wrong_key["teacher-probe"]["status"] == "skip"

    token = teacher_checks(
        tmp_path,
        'backend = "anthropic"\nmodel = "claude-sonnet-5"\n',
        env={"ANTHROPIC_AUTH_TOKEN": "t"},
        probe="none",
    )
    assert token["teacher-key"]["status"] == "pass"
    assert "ANTHROPIC_AUTH_TOKEN" in token["teacher-key"]["summary"]

    no_key = teacher_checks(tmp_path, 'backend = "anthropic"\nmodel = "claude-sonnet-5"\n')
    assert no_key["teacher-key"]["summary"] == "ANTHROPIC_API_KEY is not set"

    default = teacher_checks(
        tmp_path,
        None,
        default_teacher_model="claude-sonnet-5",
        env={"ANTHROPIC_API_KEY": "k"},
        probe="free",
    )
    assert default["teacher-config"]["status"] == "pass"
    assert "claude-sonnet-5" in default["teacher-config"]["summary"]
    assert default["teacher-probe"]["status"] == "skip"
    assert "billed" in default["teacher-probe"]["summary"]


def test_probe_modes_and_the_key_handoff(tmp_path: Path) -> None:
    seen: list[tuple[TeacherConfig, str]] = []

    def prober(config: TeacherConfig, mode: str) -> ProbeResult:
        seen.append((config, mode))
        return ProbeResult(True, "answered")

    anthropic = 'backend = "anthropic"\nmodel = "claude-sonnet-5"\n'
    checks = teacher_checks(
        tmp_path, anthropic, env={"ANTHROPIC_API_KEY": "secret"}, teacher_prober=prober
    )
    assert checks["teacher-probe"] == {
        "id": "teacher-probe",
        "status": "pass",
        "summary": "answered",
        "fix": None,
    }
    config, mode = seen[-1]
    assert mode == "request"
    assert config.api_key == "secret"
    assert config.max_retries == 0
    assert "secret" not in json.dumps(checks)

    none = teacher_checks(tmp_path, anthropic, env={"ANTHROPIC_API_KEY": "k"}, probe="none")
    assert none["teacher-probe"]["status"] == "skip"
    ollama = teacher_checks(
        tmp_path, 'backend = "ollama"\nmodel = "qwen"\n', probe="free", teacher_prober=prober
    )
    assert ollama["teacher-key"]["status"] == "pass"
    assert seen[-1][1] == "free"


class FakeAnthropic:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=16, output_tokens=4))


class FakeOpenAI:
    def __init__(self, models: list[str]) -> None:
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id=m) for m in models])
        )
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=30, completion_tokens=2))


def ticking() -> Any:
    times = iter([10.0, 11.25, 12.0, 13.0])
    return lambda: next(times)


def test_probe_teacher_sends_one_minimal_request() -> None:
    config = TeacherConfig(backend="anthropic", model="claude-sonnet-5", api_key="sk-secret")
    client = FakeAnthropic()
    result = probe_teacher(config, client=client, clock=ticking())
    assert result.ok
    assert result.summary == (
        "one request to claude-sonnet-5 answered in 1.2 s, 16 in / 4 out tokens"
    )
    assert (result.input_tokens, result.output_tokens) == (16, 4)
    (request,) = client.requests
    assert request["max_tokens"] == 8
    assert request["thinking"] == {"type": "disabled"}
    assert len(request["messages"]) == 1


def test_probe_teacher_failures_are_scrubbed_and_explained() -> None:
    config = TeacherConfig(backend="anthropic", model="claude-sonnet-5", api_key="sk-secret")

    class AuthenticationError(Exception):
        status_code = 401

    refused = probe_teacher(
        config,
        client=FakeAnthropic(AuthenticationError("invalid x-api-key sk-secret")),
        clock=ticking(),
    )
    assert not refused.ok
    assert "sk-secret" not in refused.summary
    assert "***" in refused.summary
    assert refused.fix is not None and "the key was refused" in refused.fix

    class APIConnectionError(Exception):
        pass

    ollama = TeacherConfig(backend="ollama", model="qwen")
    down = probe_teacher(
        ollama,
        client=SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(APIConnectionError("refused"))
                )
            )
        ),
        clock=ticking(),
    )
    assert not down.ok
    assert down.fix is not None and "ollama serve" in down.fix


def test_probe_teacher_ollama_free_mode_only_lists_models() -> None:
    config = TeacherConfig(backend="ollama", model="qwen2.5:1.5b")
    present = FakeOpenAI(["qwen2.5:1.5b"])
    ok = probe_teacher(config, "free", client=present, clock=ticking())
    assert ok.ok and "no request sent" in ok.summary
    assert present.requests == []
    missing = probe_teacher(config, "free", client=FakeOpenAI(["other"]), clock=ticking())
    assert not missing.ok
    assert missing.fix == "ollama pull qwen2.5:1.5b"
    # A name without a tag is the :latest tag to Ollama, and the reverse.
    tagless = TeacherConfig(backend="ollama", model="glm-4.7-flash")
    latest = probe_teacher(tagless, "free", client=FakeOpenAI(["glm-4.7-flash:latest"]))
    assert latest.ok, latest.summary
    explicit = TeacherConfig(backend="ollama", model="glm-4.7-flash:latest")
    assert probe_teacher(explicit, "free", client=FakeOpenAI(["glm-4.7-flash"])).ok
    other_tag = probe_teacher(tagless, "free", client=FakeOpenAI(["glm-4.7-flash:q4"]))
    assert not other_tag.ok
    asked = FakeOpenAI([])
    answered = probe_teacher(config, "request", client=asked, clock=ticking())
    assert answered.ok and "30 in / 2 out tokens" in answered.summary
    assert asked.requests[0]["max_tokens"] == 8


def test_render_report_prints_fix_lines_under_non_passing_checks() -> None:
    text = render_report(
        {
            "checks": [
                {"id": "python", "status": "pass", "summary": "Python 3.12", "fix": None},
                {"id": "device", "status": "warn", "summary": "CPU", "fix": "get a GPU"},
            ]
        }
    )
    assert text == (
        "  pass  python            Python 3.12\n"
        "  warn  device            CPU\n"
        "        fix: get a GPU\n"
    )


def test_import_probe_reports_a_missing_package_per_section() -> None:
    result = run_import_probe({"PATH": "", "PYTHONPATH": ""}, ("torch",))
    entry = result["torch"]
    assert set(entry) >= {"ok"}
    if not entry["ok"]:
        assert entry["error"]


def test_cli_doctor_subcommand_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "semantscript_trainer.doctor.run_import_probe", lambda env, sections: NO_TORCH
    )
    code = cli_module.main(["doctor", "--json", "--no-teacher", "--probe", "none"])
    report = json.loads(capsys.readouterr().out)
    assert report["kind"] == REPORT_KIND
    assert code == 1  # torch is reported missing by the fake probe
    assert by_id(report)["torch"]["status"] == "fail"


def test_module_entry_point_runs_without_torch_imported() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from semantscript_trainer import doctor; print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "False"
