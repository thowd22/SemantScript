"""Configuration and backend construction for build-time teacher models."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

from semantscript_trainer.teacher import TeacherConfigurationError

if TYPE_CHECKING:
    from semantscript_trainer.teacher import Teacher

TeacherBackend = Literal["anthropic", "ollama"]
TeacherMode = Literal["direct", "batch", "auto"]


@dataclass(frozen=True, slots=True)
class TeacherConfig:
    """Secret-safe, provider-neutral teacher configuration."""

    backend: TeacherBackend
    model: str
    api_key: str | None = field(default=None, repr=False, compare=False)
    base_url: str | None = None
    max_tokens: int = 2_048
    timeout_seconds: float = 600.0
    max_retries: int = 2
    mode: TeacherMode = "auto"
    batch_threshold: int = 32
    poll_interval_seconds: float = 60.0
    poll_timeout_seconds: float = 86_400.0
    seed: int = 1

    def __post_init__(self) -> None:
        if self.backend not in ("anthropic", "ollama"):
            raise TeacherConfigurationError(f"unsupported teacher backend {self.backend!r}")
        if not isinstance(self.model, str) or not self.model.strip():
            raise TeacherConfigurationError("teacher model must be a nonempty string")
        if self.api_key is not None and not self.api_key:
            raise TeacherConfigurationError("teacher api_key must be nonempty when provided")
        if self.base_url is not None:
            _validate_base_url(self.base_url)
        _positive_integer(self.max_tokens, "max_tokens")
        _nonnegative_integer(self.max_retries, "max_retries")
        if self.mode not in ("direct", "batch", "auto"):
            raise TeacherConfigurationError(f"unsupported teacher mode {self.mode!r}")
        _positive_integer(self.batch_threshold, "batch_threshold")
        _positive_finite(self.timeout_seconds, "timeout_seconds")
        _positive_finite(self.poll_interval_seconds, "poll_interval_seconds")
        _positive_finite(self.poll_timeout_seconds, "poll_timeout_seconds")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TeacherConfigurationError("seed must be an integer")

    @property
    def configuration_sha256(self) -> str:
        """Digest behavior-affecting values without retaining an API secret."""

        encoded = json.dumps(
            self.public_projection(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def public_projection(self) -> dict[str, str | int | float | None]:
        return {
            "backend": self.backend,
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "mode": self.mode,
            "batch_threshold": self.batch_threshold,
            "poll_interval_seconds": self.poll_interval_seconds,
            "poll_timeout_seconds": self.poll_timeout_seconds,
            "seed": self.seed,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TeacherConfig:
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise TeacherConfigurationError(
                f"unknown teacher configuration keys: {', '.join(unknown)}"
            )
        try:
            return cls(**dict(value))
        except TypeError as error:
            raise TeacherConfigurationError(f"invalid teacher configuration: {error}") from error


def load_teacher_config(path: str | Path) -> TeacherConfig:
    """Load the closed ``[teacher]`` table from a TOML file."""

    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    if set(document) != {"teacher"} or not isinstance(document["teacher"], dict):
        raise TeacherConfigurationError("teacher config file must contain only a [teacher] table")
    return TeacherConfig.from_mapping(document["teacher"])


def create_teacher(
    config: TeacherConfig | Mapping[str, Any],
    **provider_options: Any,
) -> Teacher:
    """Construct the configured backend; imports provider SDKs lazily."""

    resolved = config if isinstance(config, TeacherConfig) else TeacherConfig.from_mapping(config)
    if resolved.backend == "anthropic":
        from semantscript_trainer.teachers.anthropic import AnthropicTeacher

        return AnthropicTeacher(resolved, **provider_options)

    from semantscript_trainer.teachers.ollama import OllamaTeacher

    return OllamaTeacher(resolved, **provider_options)


def _validate_base_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise TeacherConfigurationError("teacher base_url must be an absolute http or https URL")
    if parsed.username is not None or parsed.password is not None:
        raise TeacherConfigurationError("teacher base_url must not contain user information")
    if parsed.query or parsed.fragment:
        raise TeacherConfigurationError("teacher base_url must not contain a query or fragment")


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TeacherConfigurationError(f"{name} must be a positive integer")


def _nonnegative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TeacherConfigurationError(f"{name} must be a non-negative integer")


def _positive_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TeacherConfigurationError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise TeacherConfigurationError(f"{name} must be a positive finite number")
