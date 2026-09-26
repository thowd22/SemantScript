"""The next command the trainer prints for failures it catches itself."""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import semantscript_trainer.cli as cli_module
from semantscript_trainer.failures import (
    DOCTOR_COMMAND_VARIABLE,
    failure_remedy,
    with_remedy,
)
from semantscript_trainer.teacher import TeacherConfigurationError, TeacherTransportError
from semantscript_trainer.training import TrainingExecutionError


def _chained(message: str, cause: BaseException) -> TrainingExecutionError:
    try:
        raise TrainingExecutionError(message) from cause
    except TrainingExecutionError as error:
        return error


def test_a_missing_training_package_names_its_doctor_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DOCTOR_COMMAND_VARIABLE, raising=False)
    error = _chained(
        "fine-tuning requires Transformers; install the project training extra",
        ModuleNotFoundError("No module named 'transformers'", name="transformers"),
    )
    fix = failure_remedy(error, stage="run", bundle="b.json", teacher="t.toml")
    assert fix == (
        f"run semantscript doctor and fix its torch check: {sys.executable} cannot import "
        "transformers"
    )
    # A torch whose shared library fails to load raises OSError under the ImportError message.
    error = _chained(
        "fine-tuning requires PyTorch; install the project training extra", OSError("libc10.so")
    )
    assert "fix its torch check" in str(failure_remedy(error, stage="run", bundle="b", teacher="t"))
    onnx = _chained(
        "artifact export requires the optional ONNX training dependencies",
        ModuleNotFoundError("No module named 'onnx'", name="onnx"),
    )
    assert "fix its onnxruntime check" in str(
        failure_remedy(onnx, stage="run", bundle="b", teacher="t")
    )
    # The model package's own guard is a plain ImportError with the extra's name.
    model = ImportError("install the project model/training extra before constructing modules")
    assert "fix its torch check" in str(failure_remedy(model, stage="run", bundle="b", teacher="t"))
    # The CLI passes the doctor command with the interpreter and module it started.
    monkeypatch.setenv(DOCTOR_COMMAND_VARIABLE, "semantscript doctor --python /opt/py")
    assert str(failure_remedy(model, stage="run", bundle="b", teacher="t")).startswith(
        "run semantscript doctor --python /opt/py and fix its torch check"
    )


def test_bundle_teacher_and_transport_failures_name_their_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DOCTOR_COMMAND_VARIABLE, raising=False)
    assert failure_remedy(
        FileNotFoundError("absent.json"), stage="bundle", bundle="absent.json", teacher="t"
    ) == (
        "run semantscript build to write the IR bundle, or pass --bundle the build's "
        "semantscript.ir.v1.json (now absent.json)"
    )
    assert str(
        failure_remedy(ValueError("x"), stage="bundle-shape", bundle="b.json", teacher="t")
    ).startswith("run semantscript build and pass its semantscript.ir.v1.json as --bundle")
    assert str(
        failure_remedy(ValueError("x"), stage="teacher-config", bundle="b", teacher="t.toml")
    ) == (
        "run semantscript doctor and fix its teacher-config check: t.toml is not a valid "
        "teacher file (docs/teachers.md lists every key)"
    )
    assert str(
        failure_remedy(TeacherTransportError("timed out"), stage="run", bundle="b", teacher="t")
    ).startswith("run semantscript doctor --probe request to test the teacher")
    # A message that already names its fix keeps it: with_remedy adds no second one.
    constraints = TeacherConfigurationError(
        "the constraints do not decide every input; next: add always/never constraints"
    )
    fix = failure_remedy(constraints, stage="run", bundle="b", teacher="t")
    assert with_remedy(f"error: {constraints}", fix) == f"error: {constraints}"


def test_every_run_failure_names_a_next_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DOCTOR_COMMAND_VARIABLE, raising=False)

    def fix(error: BaseException, stage: str = "run") -> str:
        return failure_remedy(
            error,
            stage=stage,  # type: ignore[arg-type]
            bundle="b",
            teacher="t",
            cache="/c",
            artifact="/a",
        )

    class OutOfMemoryError(RuntimeError):
        pass

    assert fix(OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")).startswith(
        "rerun with a smaller --batch-size or with --device cpu"
    )
    try:
        raise RuntimeError("could not create dataset cache directory") from NotADirectoryError(
            20, "Not a directory", "/dev/null/cache"
        )
    except RuntimeError as error:
        assert fix(error).startswith("check that /dev/null/cache can be written")
    try:
        raise RuntimeError("could not write") from PermissionError(13, "denied", "/c/datasets/x")
    except RuntimeError as error:
        assert fix(error).startswith("check that /c can be written")
    assert fix(OSError("disk full")).startswith("check that /c and /a can be written")
    # A network error is an OSError too, but no path to fix.
    assert fix(ConnectionResetError(104, "Connection reset by peer")).startswith(
        "if the message names a setting"
    )

    class ArtifactExportError(RuntimeError):
        pass

    class ArtifactPublicationError(ArtifactExportError):
        pass

    try:
        raise ArtifactPublicationError("artifact export failed") from ValueError(
            "model.onnx ONNX output failed PyTorch parity"
        )
    except ArtifactPublicationError as error:
        assert fix(error).startswith("run semantscript doctor and fix its onnxruntime check")
    assert fix(ValueError("counterfactual_ratio must be ..."), "options").startswith(
        "correct the option the message names"
    )

    class TeacherResponseError(RuntimeError):
        pass

    assert fix(TeacherResponseError("case outside the contract")).startswith(
        "rerun semantscript train: rejected answers are not replayed"
    )

    class AdversarialGenerationError(RuntimeError):
        pass

    class UnsynthesizableConstraintError(AdversarialGenerationError):
        pass

    assert fix(
        UnsynthesizableConstraintError("constraint 0 is constant true and has no opposite side")
    ).startswith("rewrite the predicate of that constraint")
    # Anything else still names a command.
    assert fix(RuntimeError("something new")).startswith(
        "if the message names a setting or a file, correct it"
    )
    assert "run semantscript doctor" in fix(RuntimeError("something new"))

    assert with_remedy("error: x", None) == "error: x"
    assert with_remedy("error: x; next: y", "z") == "error: x; next: y"
    assert with_remedy("error: x", "z") == "error: x; next: z"


def _train_arguments(tmp_path: Path, bundle: Path, teacher: str) -> list[str]:
    return [
        "train",
        "--bundle",
        str(bundle),
        "--artifact",
        str(tmp_path / "artifact"),
        "--teacher",
        teacher,
        "--cache-dir",
        str(tmp_path / "cache"),
        "--report",
        str(tmp_path / "report.json"),
    ]


def test_main_prints_the_next_command_for_handled_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(DOCTOR_COMMAND_VARIABLE, raising=False)
    monkeypatch.chdir(tmp_path)
    absent = tmp_path / "absent.json"
    assert cli_module.main(_train_arguments(tmp_path, absent, "constraints")) == 1
    err = capsys.readouterr().err
    assert "; next: run semantscript build to write the IR bundle" in err

    malformed = tmp_path / "b.json"
    malformed.write_text(json.dumps({"x": 1}))
    assert cli_module.main(_train_arguments(tmp_path, malformed, "constraints")) == 1
    err = capsys.readouterr().err
    assert (
        "error: bundle must be a semantscript.ir-bundle version 1 document; next: run "
        "semantscript build and pass its semantscript.ir.v1.json as --bundle"
    ) in err
    # --estimate says the same.
    assert cli_module.main([*_train_arguments(tmp_path, malformed, "constraints"), "--estimate"])
    assert "next: run semantscript build and pass" in capsys.readouterr().err

    teacher = tmp_path / "t.toml"
    teacher.write_text('backend = "nope"\n')
    assert cli_module.main(_train_arguments(tmp_path, malformed, str(teacher))) == 1
    err = capsys.readouterr().err
    assert f"next: run semantscript doctor and fix its teacher-config check: {teacher}" in err
    assert not (tmp_path / "report.json").exists()


def test_main_names_the_torch_check_when_training_cannot_import_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(DOCTOR_COMMAND_VARIABLE, "semantscript doctor --python /opt/py")
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"kind": "semantscript.ir-bundle", "bundleVersion": 1}))
    real_import = builtins.__import__

    def no_torch(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return real_import(name, *args, **kwargs)

    def train_bundle(bundle: Any, artifact_root: Any, **kwargs: Any) -> Any:
        # The trainer's real guard: a missing torch becomes a handled error.
        from semantscript_trainer.training import _require_torch

        _require_torch()
        return SimpleNamespace(report={})

    monkeypatch.setattr(builtins, "__import__", no_torch)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(cli_module, "train_bundle", train_bundle)
    assert cli_module.main(_train_arguments(tmp_path, bundle, "constraints")) == 1
    err = capsys.readouterr().err
    assert (
        "error: fine-tuning requires PyTorch; install the project training extra; next: run "
        "semantscript doctor --python /opt/py and fix its torch check: "
        f"{sys.executable} cannot import torch"
    ) in err


def test_the_trainer_s_own_error_classes_map_to_their_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from semantscript_trainer.adversarial import UnsynthesizableConstraintError
    from semantscript_trainer.artifact import ArtifactPublicationError
    from semantscript_trainer.dataset import DatasetCacheError
    from semantscript_trainer.teacher import TeacherResponseError

    monkeypatch.delenv(DOCTOR_COMMAND_VARIABLE, raising=False)

    def fix(error: BaseException) -> str:
        return failure_remedy(
            error, stage="run", bundle="b", teacher="t", cache="/c", artifact="/a"
        )

    try:
        raise ArtifactPublicationError("artifact export failed") from ValueError(
            "model.onnx ONNX output failed PyTorch parity"
        )
    except ArtifactPublicationError as error:
        assert fix(error).startswith("run semantscript doctor and fix its onnxruntime check")
    assert fix(DatasetCacheError("could not write")).startswith("check that /c and /a")
    assert fix(TeacherResponseError("bad case")).startswith("rerun semantscript train")
    assert fix(
        UnsynthesizableConstraintError(
            "constraint 1 is constant true and has no opposite predicate side"
        )
    ).startswith("rewrite the predicate")
    assert fix(
        UnsynthesizableConstraintError("constraint 1 did not yield a valid two-sided pair")
    ).startswith("rerun semantscript train")


def test_a_failed_encoder_download_names_network_access_not_a_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huggingface_hub.errors import LocalEntryNotFoundError

    monkeypatch.delenv(DOCTOR_COMMAND_VARIABLE, raising=False)

    def fix(error: BaseException) -> str:
        return failure_remedy(
            error, stage="run", bundle="b", teacher="t", cache="/c", artifact="/a"
        )

    # transformers wraps the hub's error in a bare OSError without errno or filename.
    try:
        try:
            raise LocalEntryNotFoundError("cannot find the requested files in the disk cache")
        except LocalEntryNotFoundError as hub:
            raise OSError(
                "We couldn't connect to 'https://huggingface.co' to load the files"
            ) from hub
    except OSError as error:
        text = fix(error)
    assert text.startswith("check that this machine can reach https://huggingface.co")
    assert "can be written" not in text
    # Without the hub's class in the chain, the text alone is enough.
    assert fix(OSError("x is not a local folder and is not a valid model identifier")).startswith(
        "check that this machine can reach"
    )
