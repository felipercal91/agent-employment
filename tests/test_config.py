from pathlib import Path

import pytest

from job_assistant.config import DEFAULT_MODEL, PROJECT_ROOT, ConfigError, load_settings

ENV_VARS = ("ANTHROPIC_API_KEY", "CLAUDE_MODEL", "DATA_DIR", "OUTPUT_DIR")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def write_env(tmp_path: Path, body: str) -> Path:
    env_file = tmp_path / ".env"
    env_file.write_text(body, encoding="utf-8")
    return env_file


def test_missing_key_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(write_env(tmp_path, "ANTHROPIC_API_KEY=\n"))


def test_defaults(tmp_path: Path) -> None:
    settings = load_settings(write_env(tmp_path, "ANTHROPIC_API_KEY=sk-test\n"))
    assert settings.model == DEFAULT_MODEL
    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.output_dir == PROJECT_ROOT / "output"


def test_overrides(tmp_path: Path) -> None:
    custom = tmp_path / "mydata"
    settings = load_settings(
        write_env(
            tmp_path,
            f"ANTHROPIC_API_KEY=sk-test\nCLAUDE_MODEL=claude-sonnet-5-5\nDATA_DIR={custom}\n",
        )
    )
    assert settings.model == "claude-sonnet-5-5"
    assert settings.data_dir == custom


def test_key_not_in_repr(tmp_path: Path) -> None:
    settings = load_settings(write_env(tmp_path, "ANTHROPIC_API_KEY=sk-secret-value\n"))
    assert "sk-secret-value" not in repr(settings)
