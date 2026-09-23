from __future__ import annotations

from pathlib import Path

import pytest

from semantscript_trainer.teacher_config import TeacherConfig, load_teacher_config


def test_selects_backend_and_model_from_mapping_without_secret_in_digest() -> None:
    first = TeacherConfig.from_mapping(
        {
            "backend": "anthropic",
            "model": "claude-sonnet-5",
            "api_key": "first-secret",
            "mode": "batch",
        }
    )
    second = TeacherConfig.from_mapping(
        {
            "backend": "anthropic",
            "model": "claude-sonnet-5",
            "api_key": "second-secret",
            "mode": "batch",
        }
    )
    changed = TeacherConfig.from_mapping(
        {"backend": "anthropic", "model": "claude-sonnet-5", "mode": "direct"}
    )

    assert first.backend == "anthropic"
    assert first.model == "claude-sonnet-5"
    assert first.configuration_sha256 == second.configuration_sha256
    assert first.configuration_sha256 != changed.configuration_sha256
    assert "secret" not in repr(first)


def test_loads_closed_teacher_toml_table(tmp_path: Path) -> None:
    path = tmp_path / "teacher.toml"
    path.write_text(
        """[teacher]
backend = "ollama"
model = "qwen3:14b-q4_K_M"
base_url = "http://localhost:11434/v1"
seed = 7
""",
        encoding="utf-8",
    )

    config = load_teacher_config(path)
    assert config.backend == "ollama"
    assert config.model == "qwen3:14b-q4_K_M"
    assert config.base_url == "http://localhost:11434/v1"
    assert config.seed == 7


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"backend": "other", "model": "x"}, "unsupported teacher backend"),
        ({"backend": "ollama", "model": ""}, "model must be a nonempty"),
        ({"backend": "ollama", "model": "x", "unknown": 1}, "unknown teacher"),
        (
            {"backend": "ollama", "model": "x", "base_url": "localhost:11434"},
            "absolute http or https",
        ),
        (
            {
                "backend": "ollama",
                "model": "x",
                "base_url": "http://user:secret@localhost:11434/v1",
            },
            "must not contain user information",
        ),
        ({"backend": "anthropic", "model": "x", "max_tokens": True}, "positive integer"),
    ],
)
def test_rejects_invalid_or_unknown_configuration(value: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TeacherConfig.from_mapping(value)


def test_rejects_extra_toml_sections(tmp_path: Path) -> None:
    path = tmp_path / "teacher.toml"
    path.write_text(
        '[teacher]\nbackend="ollama"\nmodel="qwen3"\n[other]\nvalue=1\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"only a \[teacher\] table"):
        load_teacher_config(path)


def test_userinfo_secret_is_not_reflected_by_base_url_validation() -> None:
    secret = "do-not-reflect-this-secret"

    with pytest.raises(ValueError) as raised:
        TeacherConfig(
            backend="ollama",
            model="qwen3:14b-q4_K_M",
            base_url=f"http://user:{secret}@localhost:11434/v1",
        )

    assert secret not in str(raised.value)
