from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from semantscript_trainer.teacher import TeacherBudgetExceeded, TeacherConfigurationError
from semantscript_trainer.teacher_config import (
    ConstraintsTeacherConfig,
    TeacherConfig,
    load_teacher_config,
)
from semantscript_trainer.teacher_prompt import TEACHER_PROMPT_VERSION
from semantscript_trainer.teacher_spend import (
    CHARACTERS_PER_TOKEN,
    ResponseJournal,
    SpendMeter,
    TeacherPriceUnknown,
    resolve_price,
    seconds_per_request,
)

OPENROUTER = {
    "backend": "anthropic",
    "model": "anthropic/claude-sonnet-5",
    "base_url": "https://openrouter.ai/api",
    "mode": "direct",
}
MODELS = json.dumps(
    {
        "data": [
            {
                "id": "anthropic/claude-sonnet-5",
                "pricing": {
                    "prompt": "0.000002",
                    "completion": "0.00001",
                    "input_cache_read": "0.0000002",
                    "input_cache_write": "0.0000025",
                },
            },
            {"id": "broken/model", "pricing": {"prompt": "n/a"}},
        ]
    }
).encode()


class Fetcher:
    def __init__(self, body: bytes | Exception = MODELS) -> None:
        self.body = body
        self.calls = 0

    def __call__(self, url: str) -> bytes:
        self.calls += 1
        assert url == "https://openrouter.ai/api/v1/models"
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def test_openrouter_prices_come_from_its_model_list_and_are_cached_for_a_day(
    tmp_path: Path,
) -> None:
    fetch = Fetcher()
    config = TeacherConfig.from_mapping(OPENROUTER)
    price = resolve_price(config, cache_directory=tmp_path, fetch=fetch, now=lambda: 1_000.0)

    assert (price.input_usd_per_million, price.output_usd_per_million) == pytest.approx((2, 10))
    assert price.cache_read_usd_per_million == pytest.approx(0.2)
    assert price.cache_write_usd_per_million == pytest.approx(2.5)
    assert price.source.startswith("OpenRouter price list")
    # The measured 2026-09-25 request: 264 uncached, 2,347 written, 68 output tokens.
    assert price.cost(264, 68, cache_write_tokens=2347) == pytest.approx(0.0070755)

    resolve_price(config, cache_directory=tmp_path, fetch=fetch, now=lambda: 2_000.0)
    assert fetch.calls == 1
    resolve_price(config, cache_directory=tmp_path, fetch=fetch, now=lambda: 100_000.0)
    assert fetch.calls == 2


def test_offline_openrouter_falls_back_to_the_pinned_table_and_says_so(tmp_path: Path) -> None:
    price = resolve_price(
        TeacherConfig.from_mapping(OPENROUTER),
        cache_directory=tmp_path,
        fetch=Fetcher(OSError("offline")),
    )
    assert (price.input_usd_per_million, price.output_usd_per_million) == (2.0, 10.0)
    assert "unreachable" in price.source

    with pytest.raises(TeacherPriceUnknown, match=r"\[teacher.pricing\]"):
        resolve_price(
            TeacherConfig.from_mapping({**OPENROUTER, "model": "someone/else"}),
            cache_directory=tmp_path,
            fetch=Fetcher(OSError("offline")),
        )


def test_anthropic_models_use_the_pinned_table_and_unknown_models_need_a_pricing_table() -> None:
    price = resolve_price(TeacherConfig(backend="anthropic", model="claude-opus-5"))
    assert (price.input_usd_per_million, price.output_usd_per_million) == (5.0, 25.0)
    assert price.cache_read_usd_per_million == pytest.approx(0.5)
    assert price.cost(1_000_000, 0, batch=True) == pytest.approx(2.5)

    with pytest.raises(TeacherPriceUnknown, match=r"add a \[teacher.pricing\] table"):
        resolve_price(TeacherConfig(backend="anthropic", model="claude-unknown"))
    assert issubclass(TeacherPriceUnknown, TeacherConfigurationError)


def test_pricing_table_overrides_and_never_changes_a_digest(tmp_path: Path) -> None:
    path = tmp_path / "teacher.toml"
    path.write_text(
        '[teacher]\nbackend = "anthropic"\nmodel = "claude-unknown"\n'
        "[teacher.pricing]\ninput_usd_per_million = 3\noutput_usd_per_million = 15\n"
        "seconds_per_request = 2.5\n"
    )
    config = load_teacher_config(path)
    assert isinstance(config, TeacherConfig)
    price = resolve_price(config)
    assert (price.input_usd_per_million, price.cache_read_usd_per_million) == pytest.approx(
        (3.0, 0.3)
    )
    assert price.source == "[teacher.pricing]"
    assert seconds_per_request(config) == (2.5, "[teacher.pricing]")

    plain = TeacherConfig(backend="anthropic", model="claude-unknown")
    assert config.configuration_sha256 == plain.configuration_sha256
    assert "pricing" not in config.public_projection()
    assert config.public_projection()["promptVersion"] == TEACHER_PROMPT_VERSION

    partial = resolve_price(
        TeacherConfig.from_mapping(
            {
                "backend": "anthropic",
                "model": "claude-sonnet-5",
                "pricing": {"output_usd_per_million": 12},
            }
        )
    )
    assert (partial.input_usd_per_million, partial.output_usd_per_million) == (2.0, 12.0)
    assert "overrides" in partial.source

    with pytest.raises(TeacherConfigurationError, match=r"unknown \[teacher.pricing\] keys"):
        TeacherConfig.from_mapping({**OPENROUTER, "pricing": {"input_price": 1}})
    with pytest.raises(TeacherConfigurationError, match="non-negative finite"):
        TeacherConfig.from_mapping({**OPENROUTER, "pricing": {"input_usd_per_million": -1}})


def test_local_and_constraints_teachers_are_free_and_a_mixed_teacher_prices_its_fallback() -> None:
    assert resolve_price(TeacherConfig(backend="ollama", model="qwen3:14b")).free
    assert resolve_price(ConstraintsTeacherConfig()).free
    assert seconds_per_request(ConstraintsTeacherConfig())[0] == 0.0
    mixed = ConstraintsTeacherConfig.from_mapping(
        {"fallback": {"backend": "anthropic", "model": "claude-haiku-4-5"}}
    )
    assert resolve_price(mixed).input_usd_per_million == 1.0
    # The constraints teacher's own digest does not carry the language-model prompt version.
    assert "promptVersion" not in json.dumps(ConstraintsTeacherConfig().public_projection())


def test_seconds_per_request_prefers_the_last_metered_run(tmp_path: Path) -> None:
    config = TeacherConfig(backend="anthropic", model="claude-sonnet-5")
    assert seconds_per_request(config, cache_directory=tmp_path) == (4.0, "pinned default")
    meter = SpendMeter(resolve_price(config))
    meter.charge({"input_tokens": 10, "output_tokens": 5}, seconds=3.0)
    meter.charge({"input_tokens": 10, "output_tokens": 5}, seconds=5.0)
    meter.record_stats(tmp_path, config.configuration_sha256)
    seconds, source = seconds_per_request(config, cache_directory=tmp_path)
    assert seconds == pytest.approx(4.0)
    assert source == "mean of 2 requests in the last run"


def test_meter_reserves_against_the_cap_and_prints_running_lines() -> None:
    lines: list[str] = []
    meter = SpendMeter(
        resolve_price(TeacherConfig(backend="anthropic", model="claude-sonnet-5")),
        max_cost_usd=0.01,
        log=lines.append,
        progress_every=2,
    )
    usage = SimpleNamespace(input_tokens=1000, output_tokens=100)
    assert meter.charge(usage, seconds=1.0) == pytest.approx(0.003)
    meter.reserve(0.006)
    meter.charge(usage, seconds=1.0)
    assert lines == [
        "teacher: 2 request(s), 2,000 in / 200 out tokens, USD 0.0060 of the USD 0.01 cap"
    ]
    with pytest.raises(TeacherBudgetExceeded, match="after 2 paid request"):
        meter.reserve(0.0041)
    meter.replay()
    assert meter.summary() == {
        "requests": 2,
        "replayed": 1,
        "inputTokens": 2000,
        "cacheReadTokens": 0,
        "cacheWriteTokens": 0,
        "outputTokens": 200,
        "costUsd": 0.006,
        "maxCostUsd": 0.01,
        "priceSource": "pinned Anthropic list price 2026-09-25",
        "seconds": 2.0,
    }
    # A request is reserved above what it can be expected to cost: its prompt with a
    # 20% token margin at the cache-write price, plus the largest output seen and a quarter.
    characters = int(CHARACTERS_PER_TOKEN * 1000)
    assert meter.expected_output_tokens() == 138
    assert meter.estimate_request_usd(characters) == pytest.approx(
        (1200 * 2.5 + 138 * 10) / 1_000_000
    )


@pytest.mark.parametrize("cap", [0, -1, float("inf"), float("nan"), True])
def test_meter_rejects_an_invalid_cap(cap: Any) -> None:
    with pytest.raises(TeacherConfigurationError):
        SpendMeter(resolve_price(ConstraintsTeacherConfig()), max_cost_usd=cap)
    with pytest.raises(TeacherConfigurationError, match="known teacher price"):
        SpendMeter(None, max_cost_usd=1.0)


def test_journal_ignores_unreadable_entries(tmp_path: Path) -> None:
    journal = ResponseJournal(tmp_path, "a" * 64)
    key = journal.next_key({"model": "m"})
    assert key.endswith("-1") and journal.next_key({"model": "m"}).endswith("-2")
    path = journal.directory / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text("not json")
    assert journal.load(key) is None
    journal.store(key, {"stop_reason": "end_turn", "content": [{"type": "text", "text": "{}"}]})
    assert journal.load(key)["content"] == [{"type": "text", "text": "{}"}]
