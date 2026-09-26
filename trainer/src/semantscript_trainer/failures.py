"""The next command for a failure the trainer catches itself.

``semantscript_trainer.cli`` prints ``error: <message>`` for every exception it
handles; :func:`failure_remedy` adds ``; next: <fix>`` for the failures a
developer fixes outside the trainer (a missing training package, a bundle
that is not the build's, a broken teacher file, an unreachable teacher). The
fix texts come from ``diagnostics/remedies.json`` like every printed remedy.
"""

from __future__ import annotations

import os
import sys
from typing import Literal

from semantscript_trainer.remedies import remedy
from semantscript_trainer.teacher import TeacherConfigurationError, TeacherTransportError

type FailureStage = Literal["bundle", "bundle-shape", "teacher-config", "run"]
"""Where the run failed: reading the bundle, checking its shape, loading the teacher
file, or anything after."""

DOCTOR_COMMAND_VARIABLE = "SEMANTSCRIPT_DOCTOR_COMMAND"
"""Set by the CLI to the ``semantscript doctor`` command with the interpreter and
trainer module it started, so the printed command checks this environment."""

MODULE_CHECKS: dict[str, str] = {
    "semantscript_model": "model",
    "torch": "torch",
    "transformers": "torch",
    "tokenizers": "torch",
    "safetensors": "torch",
    "numpy": "torch",
    "huggingface_hub": "torch",
    "onnx": "onnxruntime",
    "onnxruntime": "onnxruntime",
    "onnxscript": "onnxruntime",
}
"""The ``semantscript doctor`` check that installs each module the trainer imports
(the same table as the CLI's ``MODULE_CHECKS``)."""

_EXTRA_MARKERS = ("training extra", "training dependenc", "ONNX training dependencies")


def doctor_command() -> str:
    """``semantscript doctor``, with the CLI's ``--python``/``--trainer-module`` when it set them."""

    command = os.environ.get(DOCTOR_COMMAND_VARIABLE, "").strip()
    return command or "semantscript doctor"


def _chain(error: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in seen and len(seen) < 16:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def _missing_package(error: BaseException) -> tuple[str, str] | None:
    """``(module, doctor check)`` when the failure is a training package that does not import."""

    chain = _chain(error)
    for link in chain:
        name = getattr(link, "name", None)
        if isinstance(link, ModuleNotFoundError) and isinstance(name, str) and name:
            root = name.split(".")[0]
            return name, MODULE_CHECKS.get(root, "trainer")
    text = " ".join(str(link) for link in chain)
    if any(isinstance(link, ImportError) for link in chain) or any(
        marker in text for marker in _EXTRA_MARKERS
    ):
        if "ONNX" in text or "onnx" in text:
            return "onnx", "onnxruntime"
        if "tokenizers" in text:
            return "tokenizers", "torch"
        if "Transformers" in text or "transformers" in text:
            return "transformers", "torch"
        return "torch", "torch"
    return None


def failure_remedy(
    error: BaseException,
    *,
    stage: FailureStage,
    bundle: str,
    teacher: str,
) -> str | None:
    """The fix for ``error``, or ``None`` when its message already says what to change."""

    doctor = doctor_command()
    missing = _missing_package(error)
    if missing is not None:
        module, check = missing
        return remedy(
            "training-extra-missing",
            doctor=doctor,
            check=check,
            python=sys.executable,
            module=module,
        )
    if stage == "bundle":
        return remedy("bundle-unreadable", bundle=bundle)
    if stage == "teacher-config":
        return remedy("teacher-config-invalid", doctor=doctor, teacher=teacher)
    if stage == "bundle-shape":
        return remedy("bundle-invalid", bundle=bundle)
    if isinstance(error, TeacherTransportError) and not isinstance(
        error, TeacherConfigurationError
    ):
        return remedy("teacher-transport", doctor=doctor)
    return None


def with_remedy(message: str, fix: str | None) -> str:
    """``message; next: fix`` unless there is no fix or the message already names one."""

    if fix is None or "next:" in message:
        return message
    return f"{message}; next: {fix}"
