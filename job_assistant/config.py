"""Environment-driven settings. The API key is only ever read from the environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "claude-opus-5-5"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    # repr=False so the key never ends up in logs or tracebacks.
    anthropic_api_key: str = field(repr=False)
    model: str
    data_dir: Path
    output_dir: Path
    vectorstore_dir: Path
    tracker_path: Path
    # Sheet that receives new rows; None = first sheet with recognizable headers.
    tracker_sheet: str | None = None
    tracker_initial_status: str = "Pending review"
    # Contacts/applications silent for longer than this are flagged for follow-up.
    followup_stale_days: int = 7
    # No new follow-up to the same recipient/company within this many days of the last one.
    followup_cooldown_days: int = 7
    # Reasoning effort for classification/review calls (requirements check, HR reviewer); the
    # writer keeps the model's default. low | medium | high; empty = default for everything.
    review_effort: str | None = "low"
    # Required only for API keys not scoped to a workspace.
    workspace_id: str | None = None
    # Optional second provider. Which provider each role uses is chosen per role.
    deepseek_api_key: str | None = field(default=None, repr=False)
    deepseek_model: str = "deepseek-flash"
    deepseek_thinking: bool = False  # default reasoning mode (slower and pricier)
    writer_thinking: bool | None = None  # per role; None = deepseek_thinking
    review_thinking: bool | None = None
    deepseek_writer_model: str | None = None  # None = deepseek_model
    writer_provider: str = "anthropic"  # the CV writer
    review_provider: str = "anthropic"  # requirements check + HR reviewer


def _resolve_dir(value: str | None, default: str) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _effort(value: str | None) -> str | None:
    if value is None:
        return "low"  # not configured: the cheap default for review calls
    value = value.strip().lower()
    if value in {"low", "medium", "high"}:
        return value
    return None  # empty or "default": leave the model's default


def _flag(value: str | None) -> bool | None:
    """true/false from an env var; None when it is unset or empty."""
    text = (value or "").strip().lower()
    if not text:
        return None
    return text in ("1", "true", "yes", "on")


def _provider(name: str, value: str | None, has_deepseek_key: bool) -> str:
    provider = (value or "").strip().lower() or "anthropic"
    if provider not in ("anthropic", "deepseek"):
        raise ConfigError(f"{name} must be 'anthropic' or 'deepseek' (got '{provider}').")
    if provider == "deepseek" and not has_deepseek_key:
        raise ConfigError(f"{name}=deepseek needs DEEPSEEK_API_KEY in .env.")
    return provider


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings from `.env` (if present) and the process environment.

    Variables already set in the environment take precedence over `.env`.
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    data_dir = _resolve_dir(os.getenv("DATA_DIR"), "data")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip() or None
    tracker_env = os.getenv("TRACKER_PATH", "").strip()
    return Settings(
        anthropic_api_key=api_key,
        model=os.getenv("CLAUDE_MODEL", "").strip() or DEFAULT_MODEL,
        data_dir=data_dir,
        output_dir=_resolve_dir(os.getenv("OUTPUT_DIR"), "output"),
        vectorstore_dir=_resolve_dir(os.getenv("VECTORSTORE_DIR"), "vectorstore"),
        tracker_path=_resolve_dir(tracker_env, "") if tracker_env else data_dir / "tracker.xlsx",
        tracker_sheet=os.getenv("TRACKER_SHEET", "").strip() or None,
        tracker_initial_status=os.getenv("TRACKER_INITIAL_STATUS", "").strip() or "Pending review",
        followup_stale_days=int(os.getenv("FOLLOWUP_STALE_DAYS", "").strip() or 7),
        followup_cooldown_days=int(os.getenv("FOLLOWUP_COOLDOWN_DAYS", "").strip() or 7),
        review_effort=_effort(os.getenv("REVIEW_EFFORT")),
        workspace_id=os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip() or None,
        deepseek_api_key=deepseek_key,
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-flash",
        deepseek_thinking=_flag(os.getenv("DEEPSEEK_THINKING")) is True,
        writer_thinking=_flag(os.getenv("WRITER_THINKING")),
        review_thinking=_flag(os.getenv("REVIEW_THINKING")),
        deepseek_writer_model=os.getenv("DEEPSEEK_WRITER_MODEL", "").strip() or None,
        writer_provider=_provider("WRITER_PROVIDER", os.getenv("WRITER_PROVIDER"), bool(deepseek_key)),
        review_provider=_provider("REVIEW_PROVIDER", os.getenv("REVIEW_PROVIDER"), bool(deepseek_key)),
    )
