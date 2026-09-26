"""The next command for a failure the trainer catches itself.

``semantscript_trainer.cli`` prints ``error: <message>`` for every exception it
handles; :func:`failure_remedy` adds ``; next: <fix>`` for the failures a
developer fixes outside the trainer (a missing training package, a bundle
that is not the build's, a broken teacher file, an unreachable teacher), for
the ones a rerun with other options fixes (out of memory, an unwritable path,
an option out of range, answers the teacher got wrong), and a last resort for
the rest, so every failure names the next command. The fix texts come from
``diagnostics/remedies.json`` like every printed remedy.
"""

from __future__ import annotations

import os
import sys
from typing import Literal

from semantscript_trainer.remedies import remedy
from semantscript_trainer.teacher import TeacherConfigurationError, TeacherTransportError

type FailureStage = Literal["bundle", "bundle-shape", "teacher-config", "options", "run"]
"""Where the run failed: reading the bundle, checking its shape, loading the teacher
file, checking the training options, or anything after."""

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


_REJECTED = frozenset(
    {"TeacherResponseError", "AdversarialGenerationError", "ConstraintError", "DatasetError"}
)
"""Failures of the teacher's answers: a rerun asks the teacher again."""
_NOT_REJECTED = frozenset(
    {
        "TeacherTransportError",
        "TeacherBudgetExceeded",
        "DatasetCacheError",
        "DatasetConfigurationError",
        "TeacherConfigurationError",
        "ConstraintConfigurationError",
    }
)
"""Failures that are not the teacher's answers, though their classes may share a base."""


def _names(error: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(error).__mro__}


def _out_of_memory(chain: list[BaseException]) -> bool:
    for link in chain:
        if isinstance(link, MemoryError) or "OutOfMemoryError" in _names(link):
            return True
        if "out of memory" in str(link).lower():
            return True
    return False


def _rejected_answer(chain: list[BaseException]) -> bool:
    rejected = False
    for link in chain:
        names = _names(link)
        if names & _NOT_REJECTED:
            return False
        if names & _REJECTED:
            rejected = True
    return rejected


def failure_remedy(
    error: BaseException,
    *,
    stage: FailureStage,
    bundle: str,
    teacher: str,
    cache: str = ".semantscript/cache",
    artifact: str = ".semantscript/artifact",
) -> str:
    """The fix for ``error``: every failure the trainer catches names one.

    A message that already ends with its own ``next:`` clause keeps it
    (:func:`with_remedy` leaves it alone).
    """

    doctor = doctor_command()
    chain = _chain(error)
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
    if stage == "options":
        return remedy("train-option-invalid")
    if isinstance(error, TeacherTransportError) and not isinstance(
        error, TeacherConfigurationError
    ):
        return remedy("teacher-transport", doctor=doctor)
    if _out_of_memory(chain):
        return remedy("train-out-of-memory")
    if any(isinstance(link, OSError) or "DatasetCacheError" in _names(link) for link in chain):
        filenames = [
            str(link.filename)
            for link in chain
            if isinstance(link, OSError) and isinstance(link.filename, str | os.PathLike)
        ]
        filename = filenames[0] if filenames else f"{cache} and {artifact}"
        if filename.startswith(str(cache).rstrip("/\\") + os.sep):
            # A file under the cache: the cache directory is what to fix.
            filename = str(cache)
        return remedy("train-path-unwritable", path=filename)
    if any("ArtifactExportError" in _names(link) for link in chain):
        return remedy("artifact-export-failed", doctor=doctor)
    if any(
        "UnsynthesizableConstraintError" in _names(link) and " is constant " in str(link)
        for link in chain
    ):
        return remedy("constraint-constant")
    if _rejected_answer(chain):
        return remedy("teacher-rejected")
    return remedy("train-failed", doctor=doctor)


def with_remedy(message: str, fix: str | None) -> str:
    """``message; next: fix`` unless there is no fix or the message already names one."""

    if fix is None or "next:" in message:
        return message
    return f"{message}; next: {fix}"
