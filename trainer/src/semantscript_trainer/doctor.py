"""Environment checks behind ``semantscript doctor`` and the ``train`` preflight.

``run_doctor`` checks the Python side of a build: the interpreter, the trainer
and model packages, PyTorch and the device it will train on, ONNX Runtime, the
platform environment variables the run needs, and the teacher (its file, its
key and one probe request). Every check is one line with a status and, unless it
passed, the fix. PyTorch, Transformers and ONNX Runtime are imported in child
interpreters so a missing or broken package becomes a failed check instead of a
traceback, and so the same imports can be retried with ``PYTHONNOUSERSITE=1``
or ``HSA_ENABLE_DXG_DETECTION=1`` to name the variable that makes them work.

``python -m semantscript_trainer.cli doctor --json`` prints the closed report
(kind ``semantscript.doctor-report``, ``reportVersion`` 1) that the Node CLI
validates and renders.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import json
import os
import platform
import site
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from semantscript_trainer.teacher import TeacherConfigurationError
from semantscript_trainer.teacher_config import (
    ConstraintsTeacherConfig,
    TeacherConfig,
    create_teacher,
    load_teacher_config,
)

REPORT_KIND = "semantscript.doctor-report"
REPORT_VERSION = 1
CHECK_IDS = (
    "python",
    "trainer",
    "model",
    "torch",
    "device",
    "onnxruntime",
    "platform-env",
    "teacher-config",
    "teacher-key",
    "teacher-probe",
)
MINIMUM_PYTHON = (3, 12)
DISTRIBUTION = "semantscript-python"
TRAINING_EXTRA_FIX = (
    'install the training extra into this interpreter: pip install -e ".[training]" from the '
    "SemantScript checkout (for a GPU, install the CUDA or ROCm torch build from "
    "https://pytorch.org/get-started/locally/ first)"
)
PROBE_TIMEOUT_SECONDS = 60.0
PROBE_PROMPT = "Reply with the single word ok."
PROBE_MAX_TOKENS = 8
_GIB = 1024**3

Status = Literal["pass", "fail", "warn", "skip"]
ProbeMode = Literal["request", "free", "none"]
DeviceRequest = str  # "auto", "cpu" or "cuda"; anything else fails the device check


@dataclass(frozen=True, slots=True)
class Check:
    """One line of the report: a status, what was found and, unless it passed, the fix."""

    id: str
    status: Status
    summary: str
    fix: str | None = None

    def __post_init__(self) -> None:
        if self.id not in CHECK_IDS:
            raise ValueError(f"unknown doctor check {self.id!r}")
        if self.status not in ("pass", "fail", "warn", "skip"):
            raise ValueError(f"unknown doctor status {self.status!r}")

    def to_json(self) -> dict[str, str | None]:
        return {"id": self.id, "status": self.status, "summary": self.summary, "fix": self.fix}


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The outcome of one minimal teacher request (or a free reachability check)."""

    ok: bool
    summary: str
    fix: str | None = None
    latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    # Whether the request reached the provider: true once it answered, even with an
    # error status; false when nothing was sent (no key, no connection).
    request_sent: bool = False


# One child interpreter per probe. It prints a JSON object with one entry per
# requested section; each entry says whether the import worked and what it found.
_IMPORT_PROBE = r"""
import json, sys
sections = sys.argv[1].split(",")
out = {}
def fail(error):
    return {"ok": False, "error": f"{type(error).__name__}: {error}",
            "missing": isinstance(error, ModuleNotFoundError)}
if "torch" in sections:
    try:
        import torch
        entry = {"ok": True, "version": torch.__version__,
                 "hip": getattr(torch.version, "hip", None),
                 "cuda": getattr(torch.version, "cuda", None)}
        entry["cudaAvailable"] = bool(torch.cuda.is_available())
        if entry["cudaAvailable"]:
            try:
                free, total = torch.cuda.mem_get_info(0)
                entry["device"] = {"name": torch.cuda.get_device_name(0),
                                   "count": torch.cuda.device_count(),
                                   "free": free, "total": total}
            except Exception as error:
                entry["deviceError"] = f"{type(error).__name__}: {error}"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        entry["mps"] = bool(mps is not None and mps.is_available())
        out["torch"] = entry
    except Exception as error:
        out["torch"] = fail(error)
if "transformers" in sections:
    try:
        import transformers
        import transformers.modeling_utils  # noqa: F401  (where broken NumPy stacks fail)
        out["transformers"] = {"ok": True, "version": transformers.__version__}
    except Exception as error:
        out["transformers"] = fail(error)
if "onnxruntime" in sections:
    try:
        import onnxruntime
        entry = {"ok": True, "version": onnxruntime.__version__,
                 "providers": list(onnxruntime.get_available_providers())}
        try:
            import onnx
            entry["onnx"] = onnx.__version__
        except Exception as error:
            entry["onnxError"] = f"{type(error).__name__}: {error}"
        out["onnxruntime"] = entry
    except Exception as error:
        out["onnxruntime"] = fail(error)
print(json.dumps(out))
"""

ALL_SECTIONS = ("torch", "transformers", "onnxruntime")

ImportProbe = Callable[[Mapping[str, str], Sequence[str]], dict[str, Any]]
TeacherProber = Callable[[TeacherConfig, ProbeMode], ProbeResult]


def run_import_probe(env: Mapping[str, str], sections: Sequence[str]) -> dict[str, Any]:
    """Import the requested heavy packages in a child interpreter with ``env``."""

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _IMPORT_PROBE, ",".join(sections)],
            # UTF-8 both ways, so a traceback naming a non-ASCII path decodes
            # on Windows, where a pipe otherwise uses the locale code page.
            env={**env, "PYTHONIOENCODING": "utf-8"},
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {section: {"ok": False, "error": str(error)} for section in sections}
    lines = completed.stdout.strip().splitlines()
    try:
        parsed = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        tail = (completed.stderr.strip().splitlines() or ["no output"])[-1]
        reason = f"the import probe exited with status {completed.returncode}: {tail}"
        return {section: {"ok": False, "error": reason} for section in sections}
    return parsed


def run_doctor(
    *,
    teacher_path: str | Path | None = None,
    default_teacher_model: str | None = None,
    check_teacher: bool = True,
    probe: ProbeMode = "request",
    device: DeviceRequest = "auto",
    env: Mapping[str, str] | None = None,
    import_probe: ImportProbe | None = None,
    teacher_prober: TeacherProber | None = None,
    is_wsl: bool | None = None,
    user_site_exists: bool | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run every Python-side check and return the closed report.

    ``quick`` (the ``train`` preflight) retries the imports under other environment
    variables only when something failed, so it does not confirm that a variable
    already set is still needed.
    """

    environment = dict(os.environ if env is None else env)
    checks: list[Check] = [_python_check(), *_package_checks()]
    checks.extend(
        _heavy_checks(
            environment,
            device,
            import_probe or run_import_probe,
            _detect_wsl() if is_wsl is None else is_wsl,
            _user_site_exists() if user_site_exists is None else user_site_exists,
            quick,
        )
    )
    if check_teacher:
        checks.extend(
            _teacher_checks(
                teacher_path,
                default_teacher_model,
                probe,
                environment,
                teacher_prober or probe_teacher,
            )
        )
    else:
        checks.extend(
            Check(identifier, "skip", "teacher checks not requested (--no-teacher)")
            for identifier in ("teacher-config", "teacher-key", "teacher-probe")
        )
    return {
        "kind": REPORT_KIND,
        "reportVersion": REPORT_VERSION,
        "checks": [check.to_json() for check in checks],
    }


def render_report(report: Mapping[str, Any]) -> str:
    """The one-line-per-check text form (the Node CLI renders the same lines)."""

    lines = []
    for check in report["checks"]:
        lines.append(f"  {check['status']:<5} {check['id']:<17} {check['summary']}")
        if check["status"] != "pass" and check.get("fix"):
            lines.append(f"        fix: {check['fix']}")
    return "\n".join(lines) + "\n"


# --- interpreter and packages -------------------------------------------------


def _python_check() -> Check:
    version = ".".join(str(part) for part in sys.version_info[:3])
    summary = f"Python {version} at {sys.executable}"
    if sys.version_info[:2] < MINIMUM_PYTHON:
        return Check(
            "python",
            "fail",
            summary,
            "SemantScript's trainer needs Python 3.12 or later: pass --python <exe> or set "
            "SEMANTSCRIPT_PYTHON to a 3.12+ interpreter",
        )
    return Check("python", "pass", summary)


def _distribution_version() -> str:
    try:
        return importlib.metadata.version(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return "source checkout"


def _package_checks() -> list[Check]:
    version = _distribution_version()
    checks: list[Check] = []
    trainer = sys.modules["semantscript_trainer"]
    missing = [name for name in ("anthropic", "openai") if not _importable(name)]
    trainer_summary = f"semantscript_trainer {version} from {Path(trainer.__file__).parent}"
    if missing:
        checks.append(
            Check(
                "trainer",
                "fail",
                f"{trainer_summary}; missing {', '.join(missing)}",
                "pip install -e . from the SemantScript checkout (installs the teacher clients "
                "anthropic and openai)",
            )
        )
    else:
        checks.append(Check("trainer", "pass", trainer_summary))
    try:
        model = importlib.import_module("semantscript_model")
    except Exception as error:
        checks.append(
            Check(
                "model",
                "fail",
                f"semantscript_model does not import: {type(error).__name__}: {error}",
                "pip install -e . from the SemantScript checkout, or run the CLI from the "
                "checkout so model/src is on PYTHONPATH",
            )
        )
    else:
        checks.append(
            Check(
                "model",
                "pass",
                f"semantscript_model {version} from {Path(str(model.__file__)).parent}",
            )
        )
    return checks


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


# --- torch, device, ONNX Runtime and the platform environment ----------------


def _imports_ok(result: Mapping[str, Any], sections: Sequence[str]) -> bool:
    return all(result.get(section, {}).get("ok") is True for section in sections)


def _gpu_visible(result: Mapping[str, Any]) -> bool:
    return result.get("torch", {}).get("cudaAvailable") is True


def _heavy_checks(
    env: dict[str, str],
    device: DeviceRequest,
    import_probe: ImportProbe,
    is_wsl: bool,
    user_site_exists: bool,
    quick: bool = False,
) -> list[Check]:
    base = import_probe(env, ALL_SECTIONS)
    needed: dict[str, str] = {}  # variable -> why, for variables not set but required
    notes: list[str] = []

    # PYTHONNOUSERSITE: packages under ~/.local can shadow the ones this
    # interpreter should use (a NumPy 2 in the user site breaks Transformers).
    user_site_flag = bool(env.get("PYTHONNOUSERSITE"))
    if user_site_exists and not user_site_flag and not _imports_ok(base, ALL_SECTIONS):
        isolated = import_probe({**env, "PYTHONNOUSERSITE": "1"}, ALL_SECTIONS)
        if _imports_ok(isolated, ALL_SECTIONS):
            needed["PYTHONNOUSERSITE"] = (
                "packages in the user site "
                f"({site.getusersitepackages()}) break the imports: {_first_error(base)}"
            )
    elif not quick and user_site_exists and user_site_flag and _imports_ok(base, ALL_SECTIONS):
        without = {key: value for key, value in env.items() if key != "PYTHONNOUSERSITE"}
        shared = import_probe(without, ALL_SECTIONS)
        if not _imports_ok(shared, ALL_SECTIONS):
            notes.append(
                "PYTHONNOUSERSITE=1 is set and needed (without it: "
                f"{_first_error(shared)}); {_persist_advice({'PYTHONNOUSERSITE': ''}, 'keep')}"
            )

    # HSA_ENABLE_DXG_DETECTION: ROCm on WSL2 reaches the GPU through /dev/dxg;
    # some ROCm releases only look there when this variable is set.
    torch_entry = base.get("torch", {})
    if is_wsl and torch_entry.get("ok") and torch_entry.get("hip"):
        dxg = env.get("HSA_ENABLE_DXG_DETECTION")
        if not _gpu_visible(base) and dxg != "1":
            retried = import_probe({**env, "HSA_ENABLE_DXG_DETECTION": "1"}, ("torch",))
            if _gpu_visible(retried):
                needed["HSA_ENABLE_DXG_DETECTION"] = "ROCm on WSL2 finds the GPU only with it set"
        elif not quick and _gpu_visible(base) and dxg == "1":
            without = {k: v for k, v in env.items() if k != "HSA_ENABLE_DXG_DETECTION"}
            retried = import_probe(without, ("torch",))
            notes.append(
                "HSA_ENABLE_DXG_DETECTION=1 is set and needed for torch to see the GPU on WSL2"
                if not _gpu_visible(retried)
                else "WSL2 with ROCm: torch sees the GPU with or without "
                "HSA_ENABLE_DXG_DETECTION=1 (a WSL Ollama service may still need it)"
            )
        elif _gpu_visible(base) and dxg != "1":
            notes.append("WSL2 with ROCm: torch sees the GPU without extra variables")

    if quick:
        unverified = [
            f"{name}={env[name]}"
            for name in ("PYTHONNOUSERSITE", "HSA_ENABLE_DXG_DETECTION")
            if env.get(name) and name not in needed
        ]
        if unverified:
            notes.append(
                f"{' and '.join(unverified)} set (semantscript doctor checks whether "
                f"{'they are' if len(unverified) > 1 else 'it is'} needed)"
            )
    env_fix = _export_fix(needed)
    checks = [
        _torch_check(base, env_fix),
        _device_check(base, device, env_fix),
        _onnxruntime_check(base, env_fix),
    ]
    if needed:
        checks.append(
            Check(
                "platform-env",
                "fail",
                "; ".join(f"{name}=1 needed: {why}" for name, why in needed.items()),
                f"{env_fix}; {_persist_advice(needed, 'add')}",
            )
        )
    else:
        checks.append(
            Check(
                "platform-env",
                "pass",
                "; ".join(notes) if notes else "no extra environment variables needed",
            )
        )
    return checks


def _on_windows() -> bool:
    """Whether fix lines use Windows shell syntax (read per call so tests can patch it)."""
    return sys.platform == "win32"


def _export_fix(needed: Mapping[str, str]) -> str | None:
    """The command that sets each needed variable to 1 in the current shell.

    POSIX shells get one ``export``. On Windows the default shell is
    PowerShell, where ``set NAME=1`` only creates a PowerShell variable named
    ``NAME=1``; so the fix gives the PowerShell form first and the cmd form
    (quoted, so ``&&`` adds no trailing space to the value) second.
    """
    if not needed:
        return None
    if _on_windows():
        powershell = "; ".join(f'$env:{name} = "1"' for name in needed)
        cmd = " && ".join(f'set "{name}=1"' for name in needed)
        return f"in PowerShell: {powershell} (in cmd: {cmd})"
    return "export " + " ".join(f"{name}=1" for name in needed)


def _persist_advice(names: Mapping[str, str], verb: str) -> str:
    """How to keep the variables set for later terminals, train and dev."""
    if _on_windows():
        setx = " and ".join(f"setx {name} 1" for name in names)
        return f"{verb} it for new terminals with {setx} (a user environment variable)"
    where = "to the shell profile" if verb == "add" else "in the shell profile"
    return f"{verb} it {where} so train and dev inherit it"


def _first_error(result: Mapping[str, Any]) -> str:
    for section in ALL_SECTIONS:
        entry = result.get(section)
        if isinstance(entry, dict) and entry.get("ok") is False:
            return f"{section}: {entry.get('error', 'failed')}"
    return "unknown import failure"


def _failed_import(
    identifier: str, name: str, entry: Mapping[str, Any], env_fix: str | None
) -> Check:
    error = entry.get("error", "failed")
    if env_fix is not None:
        fix = f"{env_fix} (see platform-env)"  # the import works once the variable is set
    elif entry.get("missing"):
        fix = TRAINING_EXTRA_FIX
    else:
        fix = f"reinstall {name} into this interpreter (pip install --force-reinstall {name})"
    return Check(identifier, "fail", f"{name} does not import: {error}", fix)


def _torch_check(base: Mapping[str, Any], env_fix: str | None) -> Check:
    torch_entry = base.get("torch", {})
    if not torch_entry.get("ok"):
        return _failed_import("torch", "torch", torch_entry, env_fix)
    backend = (
        f"ROCm {torch_entry['hip']}"
        if torch_entry.get("hip")
        else f"CUDA {torch_entry['cuda']}"
        if torch_entry.get("cuda")
        else "CPU-only build"
    )
    transformers_entry = base.get("transformers", {})
    if not transformers_entry.get("ok"):
        failed = _failed_import("torch", "transformers", transformers_entry, env_fix)
        return Check(
            "torch",
            "fail",
            f"torch {torch_entry['version']} ({backend}); {failed.summary}",
            failed.fix,
        )
    return Check(
        "torch",
        "pass",
        f"torch {torch_entry['version']} ({backend}), transformers {transformers_entry['version']}",
    )


def _device_check(base: Mapping[str, Any], device: DeviceRequest, env_fix: str | None) -> Check:
    torch_entry = base.get("torch", {})
    if device not in ("auto", "cpu", "cuda"):
        return Check(
            "device",
            "fail",
            f"--device {device} is not a trainer device",
            "pass --device auto, cpu or cuda (ROCm GPUs are cuda to PyTorch)",
        )
    if not torch_entry.get("ok"):
        return Check("device", "skip", "torch does not import, so no device was checked")
    info = torch_entry.get("device")
    if torch_entry.get("cudaAvailable") and isinstance(info, dict):
        kind = "ROCm" if torch_entry.get("hip") else "CUDA"
        summary = (
            f"{kind} device {info['name']}: {_gib(info['total'])} total, {_gib(info['free'])} free"
        )
        if device == "cpu":
            return Check("device", "pass", f"cpu requested (--device cpu); {summary} unused")
        return Check("device", "pass", f"trains on {summary}")
    if torch_entry.get("cudaAvailable") and torch_entry.get("deviceError"):
        return Check(
            "device",
            "fail",
            f"torch reports a CUDA or ROCm device but querying it failed: "
            f"{torch_entry['deviceError']}",
            (env_fix and f"{env_fix} (see platform-env)")
            or "check the GPU driver (nvidia-smi or rocminfo), or pass --device cpu",
        )
    memory = _system_memory()
    ram = f"{_gib(memory)} RAM" if memory is not None else "RAM unknown"
    if device == "cuda":
        return Check(
            "device",
            "fail",
            f"--device cuda requested but torch sees no CUDA or ROCm device ({ram})",
            (env_fix and f"{env_fix} (see platform-env)")
            or "install the torch build for this GPU (https://pytorch.org/get-started/locally/), "
            "or drop --device cuda to train on the CPU",
        )
    if device == "cpu":
        return Check("device", "pass", f"trains on the CPU (--device cpu, {ram})")
    if torch_entry.get("mps"):
        return Check(
            "device",
            "warn",
            f"Apple MPS is available but the trainer does not use it yet: trains on the CPU ({ram})",
            "expect slower runs; lower --cases or --epochs for a first try",
        )
    return Check(
        "device",
        "warn",
        f"no CUDA or ROCm device: trains on the CPU ({ram}), expect a slow run",
        (env_fix and f"{env_fix} (see platform-env)")
        or "if this machine has an NVIDIA or AMD GPU, install the matching torch build "
        "(https://pytorch.org/get-started/locally/); otherwise lower --cases or --epochs",
    )


def _onnxruntime_check(base: Mapping[str, Any], env_fix: str | None) -> Check:
    entry = base.get("onnxruntime", {})
    if not entry.get("ok"):
        return _failed_import("onnxruntime", "onnxruntime", entry, env_fix)
    providers = ", ".join(entry.get("providers", [])) or "no providers"
    if "onnx" not in entry:
        return Check(
            "onnxruntime",
            "fail",
            f"onnxruntime {entry['version']} imports but onnx does not: {entry.get('onnxError')}",
            (env_fix and f"{env_fix} (see platform-env)") or TRAINING_EXTRA_FIX,
        )
    return Check(
        "onnxruntime",
        "pass",
        f"onnxruntime {entry['version']} and onnx {entry['onnx']} (export checks: {providers})",
    )


def _gib(value: int) -> str:
    return f"{value / _GIB:.1f} GiB"


def _system_memory() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullTotalPhys)
    return None


def _detect_wsl() -> bool:
    return Path("/dev/dxg").exists() or "microsoft" in platform.release().lower()


def _user_site_exists() -> bool:
    try:
        return Path(site.getusersitepackages()).is_dir()
    except Exception:
        return False


# --- teacher ------------------------------------------------------------------


def _teacher_checks(
    teacher_path: str | Path | None,
    default_teacher_model: str | None,
    probe: ProbeMode,
    env: Mapping[str, str],
    prober: TeacherProber,
) -> list[Check]:
    skipped = [
        Check("teacher-key", "skip", "no teacher configuration to check"),
        Check("teacher-probe", "skip", "no teacher configuration to probe"),
    ]
    if teacher_path is not None:
        try:
            config = load_teacher_config(teacher_path)
        except FileNotFoundError:
            return [
                Check(
                    "teacher-config",
                    "fail",
                    f"{teacher_path} does not exist",
                    "write a [teacher] TOML there (docs/teachers.md), pass --teacher <file>, "
                    "or pass --teacher constraints (lower case) to label with the "
                    "expressions' own constraints",
                ),
                *skipped,
            ]
        except (OSError, ValueError, TeacherConfigurationError) as error:
            return [
                Check(
                    "teacher-config",
                    "fail",
                    f"{teacher_path} is not a valid teacher file: {error}",
                    "fix the [teacher] table (docs/teachers.md lists every key)",
                ),
                *skipped,
            ]
        where = (
            "built-in"
            if isinstance(config, ConstraintsTeacherConfig) and not Path(teacher_path).exists()
            else str(teacher_path)
        )
        if isinstance(config, ConstraintsTeacherConfig):
            return _constraints_checks(config, where, probe, env, prober)
    elif default_teacher_model is not None:
        config = TeacherConfig(backend="anthropic", model=default_teacher_model)
        where = "no teacher file; train writes .semantscript/teacher.toml for"
    else:
        return [
            Check(
                "teacher-config",
                "fail",
                "no semantscript.teacher.toml, teacher.toml or .semantscript/teacher.toml, "
                "and ANTHROPIC_API_KEY is not set",
                "write a [teacher] TOML (docs/teachers.md; Ollama needs no key), set "
                "ANTHROPIC_API_KEY to use the default Anthropic teacher, or pass --teacher "
                "constraints when the constraints decide every input",
            ),
            *skipped,
        ]
    checks = [Check("teacher-config", "pass", f"{where} {_describe(config)}")]
    key_check = _key_check(config, env)
    checks.append(key_check)
    if key_check.status == "fail":
        checks.append(Check("teacher-probe", "skip", "not probed: the key is missing"))
        return checks
    if probe == "none":
        checks.append(Check("teacher-probe", "skip", "not probed (--probe none)"))
        return checks
    if probe == "free" and config.backend != "ollama":
        checks.append(
            Check(
                "teacher-probe",
                "skip",
                "not probed: a request to this backend is billed; run semantscript doctor "
                "to send one",
            )
        )
        return checks
    result = prober(_with_key(config, env), probe)
    checks.append(
        Check("teacher-probe", "pass" if result.ok else "fail", result.summary, result.fix)
    )
    return checks


def _constraints_checks(
    config: ConstraintsTeacherConfig,
    where: str,
    probe: ProbeMode,
    env: Mapping[str, str],
    prober: TeacherProber,
) -> list[Check]:
    """The constraints teacher needs no key and sends no request; its fallback may."""

    ranges = len(config.ranges)
    described = (
        f"{where} constraints teacher (seed {config.seed}, "
        f"{ranges} range override{'' if ranges == 1 else 's'}"
    )
    if config.fallback is None:
        return [
            Check(
                "teacher-config",
                "pass",
                described + "; an expression whose constraints do not decide an input fails)",
            ),
            Check("teacher-key", "pass", "the constraints teacher needs no key"),
            Check("teacher-probe", "skip", "the constraints teacher sends no request"),
        ]
    fallback = config.fallback
    checks = [
        Check(
            "teacher-config",
            "pass",
            described + f"; fallback {_describe(fallback)[1:-1]} for the inputs they leave open)",
        )
    ]
    key_check = _key_check(fallback, env)
    checks.append(
        Check("teacher-key", key_check.status, f"fallback: {key_check.summary}", key_check.fix)
    )
    if key_check.status == "fail":
        checks.append(Check("teacher-probe", "skip", "not probed: the fallback key is missing"))
    elif probe == "none":
        checks.append(Check("teacher-probe", "skip", "not probed (--probe none)"))
    elif probe == "free" and fallback.backend != "ollama":
        checks.append(
            Check(
                "teacher-probe",
                "skip",
                "not probed: a request to the fallback backend is billed; run semantscript "
                "doctor to send one",
            )
        )
    else:
        result = prober(_with_key(fallback, env), probe)
        checks.append(
            Check(
                "teacher-probe",
                "pass" if result.ok else "fail",
                f"fallback: {result.summary}",
                result.fix,
            )
        )
    return checks


def _describe(config: TeacherConfig) -> str:
    host = urlsplit(config.base_url).netloc if config.base_url else None
    via = f" via {host}" if host else ""
    # The mode chooses between direct and batch requests, which only Anthropic has.
    mode = f", mode {config.mode}" if config.backend == "anthropic" else ""
    return f"({config.backend} {config.model}{via}{mode})"


def _is_openrouter(config: TeacherConfig) -> bool:
    return config.base_url is not None and "openrouter.ai" in urlsplit(config.base_url).netloc


def _key_check(config: TeacherConfig, env: Mapping[str, str]) -> Check:
    if config.backend == "ollama":
        return Check("teacher-key", "pass", "the Ollama backend needs no key")
    if config.api_key:
        return Check("teacher-key", "pass", "api_key is set in the teacher file")
    if env.get("ANTHROPIC_API_KEY"):
        target = "OpenRouter" if _is_openrouter(config) else "Anthropic"
        return Check("teacher-key", "pass", f"ANTHROPIC_API_KEY is set (sent to {target})")
    if env.get("ANTHROPIC_AUTH_TOKEN"):
        return Check(
            "teacher-key", "pass", "ANTHROPIC_AUTH_TOKEN is set (the Anthropic SDK sends it)"
        )
    if _is_openrouter(config) and env.get("OPENROUTER_API_KEY"):
        return Check(
            "teacher-key",
            "fail",
            "ANTHROPIC_API_KEY is not set; OPENROUTER_API_KEY is, but the anthropic backend "
            "reads only ANTHROPIC_API_KEY",
            f"{_copy_openrouter_key()} for this shell (base_url points at OpenRouter, which "
            "accepts its own key on the Anthropic route)",
        )
    return Check(
        "teacher-key",
        "fail",
        "ANTHROPIC_API_KEY is not set",
        f"{_set_key('<key>')} (an OpenRouter key when base_url is OpenRouter), or "
        "switch to the ollama backend (docs/teachers.md)",
    )


def _set_key(value: str) -> str:
    if _on_windows():
        return (
            f'in PowerShell: $env:ANTHROPIC_API_KEY = "{value}" '
            f'(in cmd: set "ANTHROPIC_API_KEY={value}")'
        )
    return f"export ANTHROPIC_API_KEY={value}"


def _copy_openrouter_key() -> str:
    if _on_windows():
        return (
            "in PowerShell: $env:ANTHROPIC_API_KEY = $env:OPENROUTER_API_KEY "
            '(in cmd: set "ANTHROPIC_API_KEY=%OPENROUTER_API_KEY%")'
        )
    return 'export ANTHROPIC_API_KEY="$OPENROUTER_API_KEY"'


def _with_key(config: TeacherConfig, env: Mapping[str, str]) -> TeacherConfig:
    changes: dict[str, Any] = {"timeout_seconds": PROBE_TIMEOUT_SECONDS, "max_retries": 0}
    if config.backend == "anthropic" and config.api_key is None and env.get("ANTHROPIC_API_KEY"):
        changes["api_key"] = env["ANTHROPIC_API_KEY"]
    return dataclasses.replace(config, **changes)


def _ollama_has_model(model: str, listed: set[Any]) -> bool:
    """Whether the Ollama server lists ``model``; a name without a tag means ``:latest``,
    the same way Ollama resolves it."""

    names = {name for name in listed if isinstance(name, str)}
    if model in names:
        return True
    if ":" not in model:
        return f"{model}:latest" in names
    if model.endswith(":latest"):
        return model.removesuffix(":latest") in names
    return False


def probe_teacher(
    config: TeacherConfig,
    mode: ProbeMode = "request",
    *,
    client: Any | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ProbeResult:
    """Send one minimal request to the configured teacher (or, in ``free`` mode for Ollama,
    only list its models) and report the latency and token usage.

    The Anthropic request is one short user message with ``max_tokens`` 8 and thinking
    disabled, a fraction of a cent on any listed model. The API key never appears in the
    result: error text is scrubbed of it.
    """

    teacher = create_teacher(config, **({"client": client} if client is not None else {}))
    started = clock()
    try:
        api = teacher._get_client()
        if config.backend == "ollama" and mode == "free":
            listed = api.models.list()
            names = {getattr(item, "id", None) for item in getattr(listed, "data", listed)}
            elapsed = clock() - started
            if not _ollama_has_model(config.model, names):
                return ProbeResult(
                    False,
                    f"the Ollama server answered in {elapsed:.1f} s but has no model {config.model}",
                    f"ollama pull {config.model}",
                    latency_seconds=elapsed,
                )
            return ProbeResult(
                True,
                f"the Ollama server answered in {elapsed:.1f} s and has {config.model} "
                "(no request sent)",
                latency_seconds=elapsed,
            )
        if config.backend == "ollama":
            response = api.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": PROBE_PROMPT}],
                max_tokens=PROBE_MAX_TOKENS,
                temperature=0,
            )
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
        else:
            response = api.messages.create(
                model=config.model,
                max_tokens=PROBE_MAX_TOKENS,
                messages=[{"role": "user", "content": PROBE_PROMPT}],
                thinking={"type": "disabled"},
            )
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
    except TeacherConfigurationError as error:
        return ProbeResult(False, _scrub(str(error), config), "see docs/teachers.md")
    except Exception as error:
        elapsed = clock() - started
        what = (
            "listing the Ollama models"
            if config.backend == "ollama" and mode == "free"
            else "one-request probe"
        )
        answered = isinstance(getattr(error, "status_code", None), int)
        return ProbeResult(
            False,
            f"{what} failed after {elapsed:.1f} s: {_scrub(_error_text(error), config)}",
            _probe_fix(config, error),
            latency_seconds=elapsed if answered else None,
            request_sent=answered,
        )
    elapsed = clock() - started
    tokens = (
        f", {input_tokens} in / {output_tokens} out tokens"
        if isinstance(input_tokens, int) and isinstance(output_tokens, int)
        else ""
    )
    return ProbeResult(
        True,
        f"one request to {config.model} answered in {elapsed:.1f} s{tokens}",
        latency_seconds=elapsed,
        input_tokens=input_tokens if isinstance(input_tokens, int) else None,
        output_tokens=output_tokens if isinstance(output_tokens, int) else None,
        request_sent=not (config.backend == "ollama" and mode == "free"),
    )


def _error_text(error: Exception) -> str:
    text = str(error).splitlines()[0] if str(error) else ""
    return f"{type(error).__name__}: {text}"[:300]


def _scrub(text: str, config: TeacherConfig) -> str:
    if config.api_key:
        text = text.replace(config.api_key, "***")
    return text


def _probe_fix(config: TeacherConfig, error: Exception) -> str:
    status = getattr(error, "status_code", None)
    name = type(error).__name__
    if status in (401, 403) or "Authentication" in name or "PermissionDenied" in name:
        return "the key was refused: check ANTHROPIC_API_KEY (an OpenRouter key when base_url is OpenRouter)"
    if status == 404 or "NotFound" in name:
        if config.backend == "ollama":
            return f"ollama pull {config.model}"
        return f"check the model name {config.model!r} in the teacher file"
    if "Connection" in name or "Timeout" in name:
        if config.backend == "ollama":
            return "start the server (ollama serve) or fix base_url in the teacher file"
        return "check the network and base_url in the teacher file"
    return "see docs/teachers.md"


# --- command line -------------------------------------------------------------


def add_arguments(parser: Any) -> None:
    """The ``doctor`` subcommand's options (shared by ``semantscript_trainer.cli``)."""

    parser.add_argument(
        "--teacher", type=Path, help="teacher TOML to check, or the keyword constraints"
    )
    parser.add_argument(
        "--default-teacher-model",
        help="check the default Anthropic teacher with this model when no file exists",
    )
    parser.add_argument("--no-teacher", action="store_true", help="skip the teacher checks")
    parser.add_argument(
        "--probe",
        choices=("request", "free", "none"),
        default="request",
        help="request: send one teacher request; free: only unbilled checks; none: no probe",
    )
    parser.add_argument(
        "--device", default="auto", help="the device train will request (auto, cpu or cuda)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="retry under other environment variables only when an import fails",
    )
    parser.add_argument("--json", action="store_true", help="print the JSON report")


def _tolerate_unencodable_output() -> None:
    """Escape, rather than crash on, characters stdout's encoding lacks."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(errors="backslashreplace")
        except (OSError, ValueError):
            pass


def run_from_arguments(arguments: Any) -> int:
    report = run_doctor(
        teacher_path=arguments.teacher,
        default_teacher_model=arguments.default_teacher_model,
        check_teacher=not arguments.no_teacher,
        probe=arguments.probe,
        device=arguments.device,
        quick=arguments.quick,
    )
    if arguments.json:
        # ASCII-only JSON: a piped stdout on Windows uses the locale code page
        # (cp1252 and the like), which cannot encode a non-ASCII user name or
        # checkout path, and the reader decodes the \u escapes exactly.
        sys.stdout.write(json.dumps(report, ensure_ascii=True) + "\n")
    else:
        _tolerate_unencodable_output()
        sys.stdout.write(render_report(report))
    return 1 if any(check["status"] == "fail" for check in report["checks"]) else 0


__all__ = [
    "CHECK_IDS",
    "REPORT_KIND",
    "REPORT_VERSION",
    "Check",
    "ProbeResult",
    "add_arguments",
    "probe_teacher",
    "render_report",
    "run_doctor",
    "run_from_arguments",
    "run_import_probe",
]
