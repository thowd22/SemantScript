"""What teacher requests cost: prices, the per-run spend meter and the response journal.

``resolve_price`` gives the USD price of the configured teacher. The constraints
teacher and Ollama cost nothing. An Anthropic-backend model is priced from a
small pinned table of Anthropic's list prices (``PINNED_ANTHROPIC_PRICES``);
when ``base_url`` is OpenRouter the price comes from OpenRouter's public model
list (``GET https://openrouter.ai/api/v1/models``, no key needed), cached for a
day in ``<cache-dir>/teacher-prices.json`` with the pinned table as the offline
fallback. A ``[teacher.pricing]`` table overrides any figure, and a model with
no known price fails with a typed error that names that table.

``SpendMeter`` charges every request from the usage the provider reports
(uncached input, cache writes, cache reads and output tokens), prints a
running line every ``progress_every`` requests and, with a cap, raises
``TeacherBudgetExceeded`` *before* a request whose expected cost would take the
run past it. ``ResponseJournal`` keeps every paid response under
``<cache-dir>/teacher-responses/`` so a run stopped by the cap (or by anything
else) replays them for free on the next run.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import tempfile
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from semantscript_trainer.teacher import TeacherBudgetExceeded, TeacherConfigurationError
from semantscript_trainer.teacher_config import (
    AnyTeacherConfig,
    ConstraintsTeacherConfig,
    TeacherConfig,
)

# Anthropic list prices in USD per million tokens (input, output), pinned 2026-09-25.
# OpenRouter's Anthropic models are priced from its live model list instead; a
# [teacher.pricing] table overrides either.
PINNED_ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
}
PINNED_PRICES_DATE = "2026-09-25"
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25  # five-minute cache entries
BATCH_MULTIPLIER = 0.5
# The shortest prefix each model caches; a shorter system prompt is simply not cached.
MINIMUM_CACHEABLE_TOKENS: dict[str, int] = {
    "claude-sonnet-5": 1024,
    "claude-opus-5": 512,
    "claude-opus-5-5": 512,
    "claude-haiku-4-5": 4096,
    "claude-sonnet-4-6": 1024,
    "claude-opus-4-8": 1024,
}
DEFAULT_MINIMUM_CACHEABLE_TOKENS = 1024
# Characters per token of the compact JSON teacher prompts (system, user message and
# response schema together), calibrated 2026-09-25 against the usage Sonnet 5 reported
# through OpenRouter for the Express example's decideRefund case prompt: 5,468
# characters were 2,611 input tokens (2.09), and a 134-character case answer was 68
# output tokens. No Claude tokenizer runs offline, so every token count the estimate
# prints is characters divided by this ratio. (The old indented prompt ran at about 3.2.)
CHARACTERS_PER_TOKEN = 2.1
# Output tokens a teacher response takes when no response has been seen yet: the
# upper end of the 60 to 110 measured on the Express example with thinking disabled.
DEFAULT_OUTPUT_TOKENS = 110
# The spend cap reserves each request at its prompt's estimated tokens plus this share,
# so a prompt that tokenizes denser than CHARACTERS_PER_TOKEN still stays under the cap.
RESERVE_TOKEN_MARGIN = 1.2
# Wall time per request when neither [teacher.pricing] seconds_per_request nor a
# previous metered run of the same teacher gives one.
PINNED_SECONDS_PER_REQUEST: dict[str, float] = {
    "anthropic": 4.0,  # Sonnet 5 through OpenRouter, Express example, 2026-09-25
    "ollama": 1.0,  # qwen3:14b on one RX 9070 XT, the same prompts, 2026-09-25
    "constraints": 0.0,
}
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
PRICE_CACHE_FILE = "teacher-prices.json"
PRICE_CACHE_SECONDS = 86_400.0
STATS_FILE = "teacher-stats.json"
JOURNAL_DIRECTORY = "teacher-responses"
_JOURNAL_DOMAIN = b"semantscript-teacher-response-v1\x00"

type Fetcher = Callable[[str], bytes]


class TeacherPriceUnknown(TeacherConfigurationError):
    """No price is known for the configured model and no [teacher.pricing] table gives one."""


@dataclass(frozen=True, slots=True)
class TeacherPrice:
    """USD per million tokens for one teacher, and where the figures came from."""

    model: str
    input_usd_per_million: float
    output_usd_per_million: float
    cache_read_usd_per_million: float
    cache_write_usd_per_million: float
    source: str
    batch_multiplier: float = BATCH_MULTIPLIER

    @property
    def free(self) -> bool:
        return self.input_usd_per_million == 0 and self.output_usd_per_million == 0

    def cost(
        self,
        input_tokens: float,
        output_tokens: float,
        *,
        cache_read_tokens: float = 0,
        cache_write_tokens: float = 0,
        batch: bool = False,
    ) -> float:
        total = (
            input_tokens * self.input_usd_per_million
            + output_tokens * self.output_usd_per_million
            + cache_read_tokens * self.cache_read_usd_per_million
            + cache_write_tokens * self.cache_write_usd_per_million
        ) / 1_000_000
        return total * (self.batch_multiplier if batch else 1.0)

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "inputUsdPerMillion": self.input_usd_per_million,
            "outputUsdPerMillion": self.output_usd_per_million,
            "cacheReadUsdPerMillion": self.cache_read_usd_per_million,
            "cacheWriteUsdPerMillion": self.cache_write_usd_per_million,
            "source": self.source,
        }


def free_price(model: str, source: str = "free") -> TeacherPrice:
    return TeacherPrice(model, 0.0, 0.0, 0.0, 0.0, source)


def language_model_config(config: AnyTeacherConfig) -> TeacherConfig | None:
    """The language-model teacher that sends requests: the config itself, the fallback
    of a mixed constraints teacher, or none for a pure constraints teacher."""

    if isinstance(config, ConstraintsTeacherConfig):
        return config.fallback
    return config


def is_openrouter(config: TeacherConfig) -> bool:
    return config.base_url is not None and "openrouter.ai" in urlsplit(config.base_url).netloc


def resolve_price(
    config: AnyTeacherConfig,
    *,
    cache_directory: str | Path | None = None,
    fetch: Fetcher | None = None,
    now: Callable[[], float] = time.time,
) -> TeacherPrice:
    """The USD price of the requests ``config`` sends (see the module docstring)."""

    model_config = language_model_config(config)
    if model_config is None:
        return free_price("constraints", "free (the constraints teacher sends no request)")
    pricing = model_config.pricing
    if model_config.backend == "ollama":
        base = free_price(model_config.model, "free (local Ollama)")
    elif (
        pricing is not None
        and pricing.input_usd_per_million is not None
        and pricing.output_usd_per_million is not None
    ):
        base = free_price(model_config.model, "[teacher.pricing]")
    elif is_openrouter(model_config):
        base = _openrouter_price(model_config.model, cache_directory, fetch, now)
    else:
        base = _pinned_price(
            model_config.model, f"pinned Anthropic list price {PINNED_PRICES_DATE}"
        )
        if base is None:
            raise TeacherPriceUnknown(
                f"no price is known for teacher model {model_config.model!r}; add a "
                "[teacher.pricing] table with input_usd_per_million and "
                "output_usd_per_million (docs/teachers.md)"
            )
    if pricing is None:
        return base
    changes: dict[str, Any] = {}
    for name in (
        "input_usd_per_million",
        "output_usd_per_million",
        "cache_read_usd_per_million",
        "cache_write_usd_per_million",
    ):
        value = getattr(pricing, name)
        if value is not None:
            changes[name] = float(value)
    if not changes:
        return base
    if "input_usd_per_million" in changes:
        changes.setdefault(
            "cache_read_usd_per_million", changes["input_usd_per_million"] * CACHE_READ_MULTIPLIER
        )
        changes.setdefault(
            "cache_write_usd_per_million",
            changes["input_usd_per_million"] * CACHE_WRITE_MULTIPLIER,
        )
    source = (
        "[teacher.pricing]"
        if base.source == "[teacher.pricing]"
        else f"{base.source}, with [teacher.pricing] overrides"
    )
    return replace(base, source=source, **changes)


def _pinned_price(model: str, source: str) -> TeacherPrice | None:
    key = model.removeprefix("anthropic/").replace(".", "-")
    prices = PINNED_ANTHROPIC_PRICES.get(key)
    if prices is None:
        return None
    input_price, output_price = prices
    return TeacherPrice(
        model,
        input_price,
        output_price,
        input_price * CACHE_READ_MULTIPLIER,
        input_price * CACHE_WRITE_MULTIPLIER,
        source,
    )


def _openrouter_price(
    model: str,
    cache_directory: str | Path | None,
    fetch: Fetcher | None,
    now: Callable[[], float],
) -> TeacherPrice:
    try:
        models = _openrouter_models(cache_directory, fetch, now)
        entry = models.get(model)
    except (OSError, ValueError, TypeError, KeyError) as error:
        pinned = _pinned_price(
            model,
            f"pinned Anthropic list price {PINNED_PRICES_DATE} (OpenRouter's price list "
            f"was unreachable: {type(error).__name__})",
        )
        if pinned is None:
            raise TeacherPriceUnknown(
                f"OpenRouter's price list was unreachable ({error}) and no pinned price is "
                f"known for {model!r}; add a [teacher.pricing] table (docs/teachers.md)"
            ) from error
        return pinned
    if entry is None:
        pinned = _pinned_price(
            model, f"pinned Anthropic list price {PINNED_PRICES_DATE} (not on OpenRouter's list)"
        )
        if pinned is None:
            raise TeacherPriceUnknown(
                f"OpenRouter lists no model {model!r} and no pinned price is known; check the "
                "model name or add a [teacher.pricing] table (docs/teachers.md)"
            )
        return pinned
    input_price = entry["prompt"]
    return TeacherPrice(
        model,
        input_price,
        entry["completion"],
        entry.get("input_cache_read", input_price * CACHE_READ_MULTIPLIER),
        entry.get("input_cache_write", input_price * CACHE_WRITE_MULTIPLIER),
        f"OpenRouter price list ({entry.get('fetchedAt', 'fetched')})",
    )


def _openrouter_models(
    cache_directory: str | Path | None,
    fetch: Fetcher | None,
    now: Callable[[], float],
) -> dict[str, dict[str, Any]]:
    cache_path = None if cache_directory is None else Path(cache_directory) / PRICE_CACHE_FILE
    if cache_path is not None:
        with contextlib.suppress(OSError, ValueError, TypeError, KeyError):
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if now() - float(cached["fetchedAtEpoch"]) < PRICE_CACHE_SECONDS:
                return _with_date(cached["models"], cached["fetchedAt"])
    body = (fetch or _http_get)(OPENROUTER_MODELS_URL)
    document = json.loads(body)
    models: dict[str, dict[str, Any]] = {}
    for item in document["data"]:
        pricing = item.get("pricing") or {}
        try:
            entry = {
                "prompt": float(pricing["prompt"]) * 1_000_000,
                "completion": float(pricing["completion"]) * 1_000_000,
            }
        except (KeyError, TypeError, ValueError):
            continue
        for name in ("input_cache_read", "input_cache_write"):
            with contextlib.suppress(KeyError, TypeError, ValueError):
                entry[name] = float(pricing[name]) * 1_000_000
        if all(math.isfinite(value) and value >= 0 for value in entry.values()):
            models[str(item["id"])] = entry
    fetched_at = datetime.fromtimestamp(now(), UTC).strftime("%Y-%m-%d")
    if cache_path is not None:
        with contextlib.suppress(OSError):
            _write_json(
                cache_path,
                {"fetchedAt": fetched_at, "fetchedAtEpoch": now(), "models": models},
            )
    return _with_date(models, fetched_at)


def _with_date(models: Mapping[str, Any], fetched_at: str) -> dict[str, dict[str, Any]]:
    return {name: {**entry, "fetchedAt": fetched_at} for name, entry in models.items()}


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "semantscript-trainer"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return bytes(response.read())


def seconds_per_request(
    config: AnyTeacherConfig, *, cache_directory: str | Path | None = None
) -> tuple[float, str]:
    """Expected wall time of one request and where the figure came from: the
    ``[teacher.pricing]`` override, else the mean a previous metered run of the same
    teacher recorded, else a pinned figure."""

    model_config = language_model_config(config)
    if model_config is None:
        return 0.0, "the constraints teacher sends no request"
    if model_config.pricing is not None and model_config.pricing.seconds_per_request is not None:
        return float(model_config.pricing.seconds_per_request), "[teacher.pricing]"
    if cache_directory is not None:
        with contextlib.suppress(OSError, ValueError, TypeError, KeyError):
            stats = json.loads((Path(cache_directory) / STATS_FILE).read_text(encoding="utf-8"))
            recorded = stats[model_config.configuration_sha256]
            if recorded["requests"] > 0:
                return (
                    float(recorded["seconds"]) / recorded["requests"],
                    f"mean of {recorded['requests']} requests in the last run",
                )
    return PINNED_SECONDS_PER_REQUEST[model_config.backend], "pinned default"


def minimum_cacheable_tokens(model: str) -> int:
    key = model.removeprefix("anthropic/").replace(".", "-")
    return MINIMUM_CACHEABLE_TOKENS.get(key, DEFAULT_MINIMUM_CACHEABLE_TOKENS)


def tokens_for_characters(characters: int) -> int:
    return math.ceil(characters / CHARACTERS_PER_TOKEN)


class SpendMeter:
    """Counts one run's teacher requests, tokens, cost and time, and enforces its cap."""

    def __init__(
        self,
        price: TeacherPrice | None,
        *,
        max_cost_usd: float | None = None,
        log: Callable[[str], None] | None = None,
        progress_every: int = 25,
    ) -> None:
        if max_cost_usd is not None and (
            isinstance(max_cost_usd, bool)
            or not isinstance(max_cost_usd, (int, float))
            or not math.isfinite(max_cost_usd)
            or max_cost_usd <= 0
        ):
            raise TeacherConfigurationError("max_cost_usd must be a positive finite number")
        if max_cost_usd is not None and price is None:
            raise TeacherConfigurationError("a spend cap needs a known teacher price")
        self.price = price
        self.max_cost_usd = None if max_cost_usd is None else float(max_cost_usd)
        self._log = log
        self._progress_every = progress_every
        self.requests = 0
        self.replayed = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.cost_usd = 0.0
        self.seconds = 0.0
        self.timed_requests = 0
        self.largest_output_tokens = 0

    def expected_output_tokens(self) -> int:
        """The output a request is reserved at: the largest answer seen so far (at least
        ``DEFAULT_OUTPUT_TOKENS``) with a quarter on top."""

        return math.ceil(max(DEFAULT_OUTPUT_TOKENS, self.largest_output_tokens) * 1.25)

    def estimate_request_usd(self, prompt_characters: int, *, batch: bool = False) -> float:
        """What a request is reserved at before it is sent: its own prompt, with
        ``RESERVE_TOKEN_MARGIN`` on the token count, at the higher of the input and the
        cache-write price (the first request of a kind writes its prompt to the cache),
        plus ``expected_output_tokens``. Deliberately above what a request costs, so a
        run stays under its cap."""

        if self.price is None:
            return 0.0
        tokens = math.ceil(tokens_for_characters(prompt_characters) * RESERVE_TOKEN_MARGIN)
        input_price = max(self.price.input_usd_per_million, self.price.cache_write_usd_per_million)
        cost = (
            tokens * input_price + self.expected_output_tokens() * self.price.output_usd_per_million
        ) / 1_000_000
        return cost * (self.price.batch_multiplier if batch else 1.0)

    def reserve(self, usd: float, *, requests: int = 1) -> None:
        """Raise ``TeacherBudgetExceeded`` when ``usd`` more would pass the cap."""

        if self.max_cost_usd is None:
            return
        if self.cost_usd + usd > self.max_cost_usd:
            raise TeacherBudgetExceeded(
                f"spend cap USD {format_usd(self.max_cost_usd)} reached: the next "
                f"{'request' if requests == 1 else f'{requests} requests'} (about USD "
                f"{usd:.4f}) would take the run from USD {self.cost_usd:.4f} past it after "
                f"{self.requests} paid request(s)"
            )

    def charge(self, usage: Any, *, seconds: float | None = None, batch: bool = False) -> float:
        """Record one answered request from the provider's usage object; returns its cost."""

        read = _usage_int(usage, "cache_read_input_tokens")
        written = _usage_int(usage, "cache_creation_input_tokens")
        uncached = _usage_int(usage, "input_tokens") or _usage_int(usage, "prompt_tokens")
        output = _usage_int(usage, "output_tokens") or _usage_int(usage, "completion_tokens")
        cost = (
            0.0
            if self.price is None
            else self.price.cost(
                uncached,
                output,
                cache_read_tokens=read,
                cache_write_tokens=written,
                batch=batch,
            )
        )
        self.requests += 1
        self.input_tokens += uncached
        self.output_tokens += output
        self.largest_output_tokens = max(self.largest_output_tokens, output)
        self.cache_read_tokens += read
        self.cache_write_tokens += written
        self.cost_usd += cost
        if seconds is not None:
            self.seconds += seconds
            self.timed_requests += 1
        self._progress()
        return cost

    def replay(self) -> None:
        """Record one response served from the journal at no cost."""

        self.replayed += 1
        self._progress()

    def _progress(self) -> None:
        if self._log is not None and (self.requests + self.replayed) % self._progress_every == 0:
            self._log(self.line())

    def line(self) -> str:
        """The running total, as the trainer prints it."""

        replayed = f" ({self.replayed} replayed from the journal)" if self.replayed else ""
        tokens = f"{self.input_tokens + self.cache_read_tokens + self.cache_write_tokens:,} in"
        if self.cache_read_tokens:
            tokens += f" ({self.cache_read_tokens:,} cached)"
        cost = "USD unknown" if self.price is None else f"USD {self.cost_usd:.4f}"
        cap = (
            "" if self.max_cost_usd is None else f" of the USD {format_usd(self.max_cost_usd)} cap"
        )
        return (
            f"teacher: {self.requests} request(s){replayed}, {tokens} / "
            f"{self.output_tokens:,} out tokens, {cost}{cap}"
        )

    def summary(self) -> dict[str, Any]:
        """The ``teacher.spend`` object of the train report."""

        return {
            "requests": self.requests,
            "replayed": self.replayed,
            "inputTokens": self.input_tokens,
            "cacheReadTokens": self.cache_read_tokens,
            "cacheWriteTokens": self.cache_write_tokens,
            "outputTokens": self.output_tokens,
            "costUsd": None if self.price is None else round(self.cost_usd, 6),
            "maxCostUsd": self.max_cost_usd,
            "priceSource": None if self.price is None else self.price.source,
            "seconds": round(self.seconds, 3),
        }

    def record_stats(self, cache_directory: str | Path, configuration_sha256: str) -> None:
        """Keep this run's mean request latency for the next ``--estimate``."""

        if self.timed_requests == 0:
            return
        path = Path(cache_directory) / STATS_FILE
        stats: dict[str, Any] = {}
        with contextlib.suppress(OSError, ValueError):
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                stats = loaded
        stats[configuration_sha256] = {
            "requests": self.timed_requests,
            "seconds": round(self.seconds, 3),
            "outputTokens": self.output_tokens,
        }
        with contextlib.suppress(OSError):
            _write_json(path, stats)


def format_usd(value: float) -> str:
    """A cap as the user wrote it: ``2``, ``0.5``, ``0.004``."""

    return f"{value:.6f}".rstrip("0").rstrip(".")


class ResponseJournal:
    """Paid teacher responses on disk, keyed by the exact request and its occurrence.

    The occurrence number keeps a retry of an identical prompt distinct from the
    first try, so a rerun replays responses in the order the run received them.
    """

    def __init__(self, cache_directory: str | Path, configuration_sha256: str) -> None:
        self.directory = Path(cache_directory) / JOURNAL_DIRECTORY / configuration_sha256[:32]
        self._seen: Counter[str] = Counter()

    def next_key(self, params: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            params, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        digest = hashlib.sha256(_JOURNAL_DOMAIN + encoded).hexdigest()
        self._seen[digest] += 1
        return f"{digest}-{self._seen[digest]}"

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        try:
            document = json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(document, dict) or not isinstance(document.get("content"), list):
            return None
        return document

    def store(self, key: str, message: Any) -> None:
        content = [
            {"type": "text", "text": _field(block, "text")}
            for block in (_field(message, "content") or [])
            if _field(block, "type") == "text" and isinstance(_field(block, "text"), str)
        ]
        usage = _field(message, "usage")
        document = {
            "stop_reason": _field(message, "stop_reason"),
            "content": content,
            "usage": {
                name: _usage_int(usage, name)
                for name in (
                    "input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                )
            },
        }
        with contextlib.suppress(OSError):
            _write_json(self._path(key), document)

    def count(self) -> int:
        try:
            return sum(1 for _ in self.directory.rglob("*.json"))
        except OSError:
            return 0


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _usage_int(usage: Any, name: str) -> int:
    value = _field(usage, name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, sort_keys=True)
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


__all__ = [
    "BATCH_MULTIPLIER",
    "CHARACTERS_PER_TOKEN",
    "DEFAULT_OUTPUT_TOKENS",
    "PINNED_ANTHROPIC_PRICES",
    "PINNED_SECONDS_PER_REQUEST",
    "ResponseJournal",
    "SpendMeter",
    "TeacherPrice",
    "TeacherPriceUnknown",
    "free_price",
    "language_model_config",
    "minimum_cacheable_tokens",
    "resolve_price",
    "seconds_per_request",
    "tokens_for_characters",
]
