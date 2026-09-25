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
from semantscript_trainer.teacher_prompt import TEACHER_PROMPT_VERSION

if TYPE_CHECKING:
    from semantscript_trainer.teacher import Teacher

TeacherBackend = Literal["anthropic", "ollama"]
TeacherMode = Literal["direct", "batch", "auto"]
NumberDistribution = Literal["uniform", "log", "count"]

CONSTRAINTS_BACKEND = "constraints"
# Keywords ``--teacher`` accepts in place of a TOML path (when no file of that name exists).
BUILT_IN_TEACHERS = (CONSTRAINTS_BACKEND,)
CONSTRAINTS_TEACHER_KIND = "semantscript.constraints-teacher"
# Bumped whenever sampling, labelling or pair generation changes what a seed produces,
# so cached datasets from an older algorithm are never reused.
CONSTRAINTS_ALGORITHM_VERSION = 3


PRICING_KEYS = (
    "input_usd_per_million",
    "output_usd_per_million",
    "cache_read_usd_per_million",
    "cache_write_usd_per_million",
    "seconds_per_request",
)


@dataclass(frozen=True, slots=True)
class TeacherPricing:
    """A ``[teacher.pricing]`` table: what a request costs and takes, when the pinned or
    fetched figures do not fit. Every key is optional; each one given overrides the
    resolved figure. It never enters a digest, so changing it regenerates nothing."""

    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    cache_read_usd_per_million: float | None = None
    cache_write_usd_per_million: float | None = None
    seconds_per_request: float | None = None

    def __post_init__(self) -> None:
        for name in PRICING_KEYS:
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise TeacherConfigurationError(
                    f"[teacher.pricing] {name} must be a non-negative finite number"
                )

    @classmethod
    def from_mapping(cls, value: Any) -> TeacherPricing:
        if not isinstance(value, Mapping):
            raise TeacherConfigurationError("pricing must be a [teacher.pricing] table")
        unknown = sorted(set(value) - set(PRICING_KEYS))
        if unknown:
            raise TeacherConfigurationError(
                f"unknown [teacher.pricing] keys: {', '.join(unknown)} "
                f"(valid keys: {', '.join(PRICING_KEYS)})"
            )
        return cls(**dict(value))


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
    pricing: TeacherPricing | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.pricing, Mapping):
            object.__setattr__(self, "pricing", TeacherPricing.from_mapping(self.pricing))
        if self.pricing is not None and not isinstance(self.pricing, TeacherPricing):
            raise TeacherConfigurationError("pricing must be a [teacher.pricing] table")
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
        # ``pricing`` is left out on purpose: it changes what a run reports, not what the
        # teacher is asked, so it must never invalidate a dataset.
        return {
            "backend": self.backend,
            "promptVersion": TEACHER_PROMPT_VERSION,
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


@dataclass(frozen=True, slots=True)
class NumberRange:
    """How the constraints teacher samples one numeric input path.

    ``distribution`` is ``uniform`` over ``[low, high]``, ``log`` (half
    log-uniform so small values are as common as large ones, half uniform) or
    ``count`` (zero half the time, else uniform). With ``decimals`` above zero,
    three draws in ten keep that many decimal places; the rest are integers.
    """

    low: float
    high: float
    distribution: NumberDistribution = "uniform"
    decimals: int = 0

    def __post_init__(self) -> None:
        for name in ("low", "high"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise TeacherConfigurationError(f"range {name} must be a finite number")
        if not self.low < self.high:
            raise TeacherConfigurationError("range low must be below high")
        if self.distribution not in ("uniform", "log", "count"):
            raise TeacherConfigurationError(
                f"range distribution must be uniform, log or count, not {self.distribution!r}"
            )
        if self.distribution in ("log", "count") and self.low < 0:
            raise TeacherConfigurationError(
                f"a {self.distribution} range must not start below zero"
            )
        if (
            isinstance(self.decimals, bool)
            or not isinstance(self.decimals, int)
            or not 0 <= self.decimals <= 6
        ):
            raise TeacherConfigurationError("range decimals must be an integer from 0 through 6")

    def projection(self) -> dict[str, str | int | float]:
        return {
            "low": self.low,
            "high": self.high,
            "distribution": self.distribution,
            "decimals": self.decimals,
        }

    @classmethod
    def from_mapping(cls, name: str, value: Any) -> NumberRange:
        if not isinstance(value, Mapping):
            raise TeacherConfigurationError(
                f"range {name!r} must be a table with low, high, distribution and decimals"
            )
        unknown = sorted(set(value) - {"low", "high", "distribution", "decimals"})
        if unknown:
            raise TeacherConfigurationError(f"unknown keys in range {name!r}: {', '.join(unknown)}")
        if "low" not in value or "high" not in value:
            raise TeacherConfigurationError(f"range {name!r} needs low and high")
        try:
            return cls(**dict(value))
        except TeacherConfigurationError as error:
            raise TeacherConfigurationError(f"range {name!r}: {error}") from error


@dataclass(frozen=True, slots=True)
class ConstraintsTeacherConfig:
    """The built-in teacher that labels inputs with an expression's own constraints.

    It samples inputs from the IR types, labels each with the one output its
    constraints admit and builds boundary pairs and counterfactual twins by
    single-field edits. ``ranges`` overrides the inferred sampling range of a
    numeric input, keyed by its dotted path (``order.total``) or its field name
    (``total``). ``fallback`` is a language-model teacher for the inputs the
    constraints leave open (mixed mode); without one, an expression whose
    constraints do not decide an input fails with that input named.
    """

    backend: Literal["constraints"] = CONSTRAINTS_BACKEND
    seed: int = 1
    twin_filter: bool = True
    near_threshold_share: float = 0.3
    maximum_sampling_attempts: int = 200_000
    ranges: tuple[tuple[str, NumberRange], ...] = ()
    fallback: TeacherConfig | None = None

    def __post_init__(self) -> None:
        if self.backend != CONSTRAINTS_BACKEND:
            raise TeacherConfigurationError(f"unsupported teacher backend {self.backend!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TeacherConfigurationError("seed must be an integer")
        if not isinstance(self.twin_filter, bool):
            raise TeacherConfigurationError("twin_filter must be a boolean")
        share = self.near_threshold_share
        if (
            isinstance(share, bool)
            or not isinstance(share, (int, float))
            or not math.isfinite(share)
            or not 0 <= share <= 1
        ):
            raise TeacherConfigurationError("near_threshold_share must be a number from 0 to 1")
        _positive_integer(self.maximum_sampling_attempts, "maximum_sampling_attempts")
        if isinstance(self.ranges, Mapping):
            object.__setattr__(
                self,
                "ranges",
                tuple(
                    (str(name), NumberRange.from_mapping(str(name), value))
                    if not isinstance(value, NumberRange)
                    else (str(name), value)
                    for name, value in sorted(self.ranges.items())
                ),
            )
        if not isinstance(self.ranges, tuple) or not all(
            isinstance(entry, tuple)
            and len(entry) == 2
            and isinstance(entry[0], str)
            and entry[0]
            and isinstance(entry[1], NumberRange)
            for entry in self.ranges
        ):
            raise TeacherConfigurationError("ranges must map input paths to range tables")
        if len({name for name, _ in self.ranges}) != len(self.ranges):
            raise TeacherConfigurationError("ranges must not repeat an input path")
        if isinstance(self.fallback, Mapping):
            if self.fallback.get("backend") == CONSTRAINTS_BACKEND:
                raise TeacherConfigurationError(
                    "the fallback teacher must be a language-model backend (anthropic or ollama)"
                )
            object.__setattr__(self, "fallback", TeacherConfig.from_mapping(self.fallback))
        if self.fallback is not None and not isinstance(self.fallback, TeacherConfig):
            raise TeacherConfigurationError("fallback must be a [teacher.fallback] table")

    def range_for(self, path: str) -> NumberRange | None:
        """The configured range for a dotted input path, else for its last field name."""

        by_name = dict(self.ranges)
        if path in by_name:
            return by_name[path]
        leaf = path.rsplit(".", 1)[-1].removesuffix("[]")
        return by_name.get(leaf)

    def sampling_projection(self) -> dict[str, Any]:
        """Every value that changes which inputs a seed samples and how they are labelled."""

        return {
            "kind": CONSTRAINTS_TEACHER_KIND,
            "algorithmVersion": CONSTRAINTS_ALGORITHM_VERSION,
            "seed": self.seed,
            "twinFilter": self.twin_filter,
            "nearThresholdShare": self.near_threshold_share,
            "maximumSamplingAttempts": self.maximum_sampling_attempts,
            "ranges": {name: value.projection() for name, value in self.ranges},
        }

    def public_projection(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "sampling": self.sampling_projection(),
            "fallback": None if self.fallback is None else self.fallback.public_projection(),
        }

    @property
    def sampling_sha256(self) -> str:
        """Digest of the sampling configuration alone (no fallback)."""

        return _sha256_json(self.sampling_projection())

    @property
    def configuration_sha256(self) -> str:
        return _sha256_json(self.public_projection())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ConstraintsTeacherConfig:
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise TeacherConfigurationError(
                f"unknown constraints teacher configuration keys: {', '.join(unknown)} "
                f"(valid keys: {', '.join(sorted(allowed))})"
            )
        values = dict(value)
        ranges = values.get("ranges", {})
        if not isinstance(ranges, Mapping):
            raise TeacherConfigurationError("ranges must be a [teacher.ranges] table")
        fallback = values.get("fallback")
        if fallback is not None and not isinstance(fallback, Mapping):
            raise TeacherConfigurationError("fallback must be a [teacher.fallback] table")
        try:
            return cls(**values)
        except TypeError as error:
            raise TeacherConfigurationError(f"invalid teacher configuration: {error}") from error


type AnyTeacherConfig = TeacherConfig | ConstraintsTeacherConfig


def teacher_config_from_mapping(value: Mapping[str, Any]) -> AnyTeacherConfig:
    """The closed configuration for a ``[teacher]`` table, chosen by its ``backend``."""

    if not isinstance(value, Mapping):
        raise TeacherConfigurationError("teacher configuration must be a table")
    backend = value.get("backend", "anthropic")
    if backend == CONSTRAINTS_BACKEND:
        return ConstraintsTeacherConfig.from_mapping(value)
    if backend not in ("anthropic", "ollama"):
        raise TeacherConfigurationError(
            f"unsupported teacher backend {backend!r}; expected anthropic, ollama or constraints"
        )
    return TeacherConfig.from_mapping(value)


def load_teacher_config(path: str | Path) -> AnyTeacherConfig:
    """Load the closed ``[teacher]`` table from a TOML file.

    The keyword ``constraints`` (when no regular file of that name exists) selects the
    built-in constraints teacher with its defaults.
    """

    if str(path) in BUILT_IN_TEACHERS and not Path(path).is_file():
        return ConstraintsTeacherConfig()
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    if set(document) != {"teacher"} or not isinstance(document["teacher"], dict):
        raise TeacherConfigurationError("teacher config file must contain only a [teacher] table")
    return teacher_config_from_mapping(document["teacher"])


def create_teacher(
    config: AnyTeacherConfig | Mapping[str, Any],
    **provider_options: Any,
) -> Teacher:
    """Construct the configured backend; imports provider SDKs lazily.

    A constraints configuration builds the built-in constraints teacher, with its
    ``fallback`` language-model teacher (built with ``provider_options``) when set.
    """

    resolved = (
        config
        if isinstance(config, (TeacherConfig, ConstraintsTeacherConfig))
        else teacher_config_from_mapping(config)
    )
    if isinstance(resolved, ConstraintsTeacherConfig):
        from semantscript_trainer.teachers.constraints import ConstraintsTeacher

        fallback = (
            None
            if resolved.fallback is None
            else create_teacher(resolved.fallback, **provider_options)
        )
        return ConstraintsTeacher(resolved, fallback=fallback)
    if resolved.backend == "anthropic":
        from semantscript_trainer.teachers.anthropic import AnthropicTeacher

        return AnthropicTeacher(resolved, **provider_options)

    from semantscript_trainer.teachers.ollama import OllamaTeacher

    return OllamaTeacher(resolved, **provider_options)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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
