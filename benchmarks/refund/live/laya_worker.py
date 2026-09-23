#!/usr/bin/env python3
"""Persistent, offline-only bridge to the pinned Laya 0.3.9 public API.

Laya's public ``Agent.predict`` result contains calibrated probabilities rounded to
four decimal places, not raw logits. The TypeScript adapter currently accepts a
logit-shaped runner response, so this bridge returns
``log(max(public_probability, 1e-12))``. Applying softmax to those values reconstructs
the normalized public probabilities; they must not be described as raw model logits.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

LAYA_CODE_REVISION = "d120d4ba220711b93c171973118753460310e16b"
LAYA_CHECKPOINT = "convaiinnovations/laya-typed-decisions"
LAYA_CHECKPOINT_REVISION = "dd079950600224fb459af2a0cb1d74e1e57ee9cf"
LAYA_VERSION = "0.3.9"
PROBABILITY_TRANSFORM = "log(max(public_calibrated_probability_rounded_4dp, 1e-12))"
SUPPORT = ("approve", "deny", "review")
MAXIMUM_LINE_BYTES = 1_048_576
MAXIMUM_FILES = 256


def _emit(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if len(encoded) + 1 > MAXIMUM_LINE_BYTES:
        raise RuntimeError("protocol response exceeds byte limit")
    sys.stdout.buffer.write(encoded + b"\n")
    sys.stdout.buffer.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(source_path: Path, expected_revision: str) -> None:
    if expected_revision != LAYA_CODE_REVISION or not source_path.is_dir():
        raise RuntimeError("source identity mismatch")
    revision = subprocess.run(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=10,
    ).stdout.strip()
    if revision != expected_revision:
        raise RuntimeError("source revision mismatch")
    if _command_emits_output(
        ["git", "-C", str(source_path), "status", "--porcelain", "--untracked-files=all"]
    ):
        raise RuntimeError("source worktree is not clean")


def _command_emits_output(command: list[str]) -> bool:
    """Observe at most one byte, terminating as soon as dirty output exists."""

    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    observed: list[bytes] = []
    done = threading.Event()

    def read_one() -> None:
        try:
            observed.append(process.stdout.read(1))
        finally:
            done.set()

    reader = threading.Thread(target=read_one, daemon=True, name="laya-git-clean-check")
    reader.start()
    if not done.wait(timeout=10):
        process.kill()
        process.wait(timeout=1)
        raise RuntimeError("source cleanliness check timed out")
    if observed and observed[0]:
        process.kill()
    returncode = process.wait(timeout=1)
    process.stdout.close()
    reader.join(timeout=1)
    if not observed or (not observed[0] and returncode != 0):
        raise RuntimeError("source cleanliness check failed")
    return bool(observed[0])


def _verify_checkpoint(
    checkpoint_path: Path,
    expected_revision: str,
    expected_files: dict[str, str],
) -> Path:
    resolved = checkpoint_path.resolve(strict=True)
    if expected_revision != LAYA_CHECKPOINT_REVISION or resolved.name != expected_revision:
        raise RuntimeError("checkpoint revision path mismatch")
    if not isinstance(expected_files, dict) or not 0 < len(expected_files) <= MAXIMUM_FILES:
        raise RuntimeError("checkpoint manifest is invalid")

    actual_paths: dict[str, Path] = {}
    for path in resolved.rglob("*"):
        if path.is_file():
            relative = path.relative_to(resolved).as_posix()
            actual_paths[relative] = path
    if set(actual_paths) != set(expected_files):
        raise RuntimeError("checkpoint file set mismatch")

    for relative, expected_digest in expected_files.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise RuntimeError("checkpoint manifest entry is invalid")
        parts = relative.split("/")
        if any(part in ("", ".", "..") for part in parts):
            raise RuntimeError("checkpoint manifest path is invalid")
        if _sha256(actual_paths[relative]) != expected_digest:
            raise RuntimeError("checkpoint file digest mismatch")
    return resolved


def _initialize(message: dict[str, Any]) -> tuple[Any, str, str]:
    if message.get("sourceRevision") != LAYA_CODE_REVISION:
        raise RuntimeError("source revision request mismatch")
    if message.get("checkpointRevision") != LAYA_CHECKPOINT_REVISION:
        raise RuntimeError("checkpoint revision request mismatch")
    if message.get("probabilityTransform") != PROBABILITY_TRANSFORM:
        raise RuntimeError("probability transform mismatch")
    device = message.get("device")
    if device not in ("cpu", "cuda"):
        raise RuntimeError("device is invalid")
    source_path = Path(message["sourcePath"]).resolve(strict=True)
    checkpoint_path = Path(message["checkpointPath"])
    expected_files = message.get("checkpointFilesSha256")
    _verify_source(source_path, LAYA_CODE_REVISION)
    checkpoint_path = _verify_checkpoint(checkpoint_path, LAYA_CHECKPOINT_REVISION, expected_files)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source_path))
    with contextlib.redirect_stdout(sys.stderr):
        import laya

        if laya.__version__ != LAYA_VERSION:
            raise RuntimeError("Laya package version mismatch")
        loaded_from = Path(laya.__file__).resolve()
        if not loaded_from.is_relative_to(source_path.resolve()):
            raise RuntimeError("Laya import escaped the pinned source tree")
        agent = laya.load(str(checkpoint_path), device=device, fast=False)
    actual_device = str(agent.device).split(":", maxsplit=1)[0]
    if actual_device != device:
        raise RuntimeError("Laya did not load on the requested device")
    return agent, device, str(checkpoint_path.resolve())


def _choose(agent: Any, message: dict[str, Any]) -> dict[str, Any]:
    request = message.get("request")
    if not isinstance(request, dict):
        raise RuntimeError("choice request is invalid")
    if (
        request.get("checkpointRevision") != LAYA_CHECKPOINT_REVISION
        or request.get("checkpoint") != LAYA_CHECKPOINT
        or request.get("codeRevision") != LAYA_CODE_REVISION
        or request.get("questionType") != "choice"
        or request.get("applyCheckpointCalibration") is not True
        or request.get("options") != list(SUPPORT)
    ):
        raise RuntimeError("choice request identity is invalid")
    question = request.get("question")
    if not isinstance(question, str) or not question or len(question) > 32_768:
        raise RuntimeError("choice question is invalid")

    questions = {
        "refund": {
            "type": "choice",
            "instructions": "Choose the refund decision that best satisfies the supplied policy.",
            "criteria": {value: None for value in SUPPORT},
        }
    }
    with contextlib.redirect_stdout(sys.stderr):
        result = agent.predict(question, questions)
    answer = result["answers"]["refund"]
    if answer.get("type") != "choice" or answer.get("choice") not in SUPPORT:
        raise RuntimeError("Laya public choice response is invalid")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or tuple(probabilities) != SUPPORT:
        raise RuntimeError("Laya public probability response is invalid")
    values = []
    for label in SUPPORT:
        value = probabilities[label]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RuntimeError("Laya public probability is invalid")
        value = float(value)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise RuntimeError("Laya public probability is invalid")
        values.append(value)
    total = sum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("Laya public probability mass is invalid")
    normalized = [value / total for value in values]
    selected_index = SUPPORT.index(answer["choice"])
    expected_index = max(range(len(normalized)), key=normalized.__getitem__)
    if selected_index != expected_index:
        raise RuntimeError("Laya public choice disagrees with stable probability argmax")
    return {
        "checkpointRevision": LAYA_CHECKPOINT_REVISION,
        "selectedIndex": selected_index,
        "logits": [math.log(max(value, 1e-12)) for value in normalized],
    }


def main() -> int:
    agent = None
    for raw_line in sys.stdin.buffer:
        request_id: int | None = None
        try:
            if len(raw_line) > MAXIMUM_LINE_BYTES or not raw_line.endswith(b"\n"):
                raise RuntimeError("protocol request exceeds byte limit")
            message = json.loads(raw_line)
            if not isinstance(message, dict):
                raise RuntimeError("protocol request is invalid")
            request_id = message.get("id")
            if not isinstance(request_id, int) or isinstance(request_id, bool) or request_id < 1:
                raise RuntimeError("protocol request id is invalid")
            action = message.get("action")
            if action == "initialize" and agent is None:
                agent, device, _checkpoint_path = _initialize(message)
                payload = {
                    "sourceRevision": LAYA_CODE_REVISION,
                    "checkpointRevision": LAYA_CHECKPOINT_REVISION,
                    "device": device,
                    "probabilityTransform": PROBABILITY_TRANSFORM,
                }
            elif action == "choose" and agent is not None:
                payload = _choose(agent, message)
            else:
                raise RuntimeError("protocol action is invalid for current state")
            _emit({"id": request_id, "ok": True, **payload})
        except Exception:
            if request_id is not None:
                _emit({"id": request_id, "ok": False, "error": "request-failed"})
            else:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
