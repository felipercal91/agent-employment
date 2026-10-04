"""Environment-driven settings. The API key is only ever read from the environment / .env."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

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
    # Safety switch: when False, no code path can call the Claude API (it raises instead).
    anthropic_enabled: bool = True
    # Optional second provider. Which provider each role uses is chosen per role.
    deepseek_api_key: str | None = field(default=None, repr=False)
    deepseek_model: str = "deepseek-flash"
    deepseek_thinking: bool = False  # default reasoning mode (slower and pricier)
    writer_thinking: bool | None = None  # per role; None = deepseek_thinking
    review_thinking: bool | None = None
    deepseek_writer_model: str | None = None  # None = deepseek_model
    writer_provider: str = "anthropic"  # the CV writer
    review_provider: str = "anthropic"  # requirements check + HR reviewer
    profile: str | None = None  # None = the default (first) person; else profiles/<name>/


PROFILE_ENV = "JOB_ASSISTANT_PROFILE"
PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,30}$")
# Where a person's files live. In profile mode these may only come from that profile's own
# folder / .env, never from the shared .env, which holds the default person's paths: otherwise a
# second person would silently read and write the first person's CV, tracker and index.
PATH_KEYS = frozenset({"DATA_DIR", "OUTPUT_DIR", "VECTORSTORE_DIR", "TRACKER_PATH", "TRACKER_SHEET"})


def profile_name(value: str | None = None) -> str | None:
    name = (os.getenv(PROFILE_ENV, "") if value is None else value).strip()
    if not name:
        return None
    if not PROFILE_NAME.match(name):
        raise ConfigError(f"Invalid profile name '{name}': use letters, digits, - or _ (max 31).")
    return name


def profile_dir(name: str) -> Path:
    return PROJECT_ROOT / "profiles" / name


def _load_shared_env(path: Path) -> None:
    """Load the shared .env without letting it set per-person paths (see PATH_KEYS)."""
    for key, value in dotenv_values(path).items():
        if key not in PATH_KEYS and value is not None and key not in os.environ:
            os.environ[key] = value


def _resolve_dir(value: str | None, default: str, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else base / path


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


def _provider(name: str, value: str | None, has_deepseek_key: bool, default: str = "anthropic") -> str:
    provider = (value or "").strip().lower() or default
    if provider not in ("anthropic", "deepseek"):
        raise ConfigError(f"{name} must be 'anthropic' or 'deepseek' (got '{provider}').")
    if provider == "deepseek" and not has_deepseek_key:
        raise ConfigError(f"{name}=deepseek needs DEEPSEEK_API_KEY in .env.")
    return provider


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings from `.env` (if present) and the process environment.

    Variables already set in the environment take precedence over `.env`.
    """
    profile = profile_name()
    shared = env_file or PROJECT_ROOT / ".env"
    if profile:
        load_dotenv(profile_dir(profile) / ".env", override=False)  # the person's own settings first
        _load_shared_env(shared)  # then shared ones (keys, models, providers), minus the paths
    else:
        load_dotenv(shared, override=False)

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip() or None
    if not api_key and not deepseek_key:
        raise ConfigError(
            "No API key found: set ANTHROPIC_API_KEY or DEEPSEEK_API_KEY in .env (copy .env.example)."
        )
    anthropic_on = _flag(os.getenv("ANTHROPIC_ENABLED")) is not False
    default_provider = "anthropic" if api_key and anthropic_on else "deepseek"

    base = profile_dir(profile) if profile else PROJECT_ROOT
    data_dir = _resolve_dir(os.getenv("DATA_DIR"), "data", base)
    tracker_env = os.getenv("TRACKER_PATH", "").strip()
    return Settings(
        anthropic_api_key=api_key,
        profile=profile,
        model=os.getenv("CLAUDE_MODEL", "").strip() or DEFAULT_MODEL,
        data_dir=data_dir,
        output_dir=_resolve_dir(os.getenv("OUTPUT_DIR"), "output", base),
        vectorstore_dir=_resolve_dir(os.getenv("VECTORSTORE_DIR"), "vectorstore", base),
        tracker_path=_resolve_dir(tracker_env, "", base) if tracker_env else data_dir / "tracker.xlsx",
        tracker_sheet=os.getenv("TRACKER_SHEET", "").strip() or None,
        tracker_initial_status=os.getenv("TRACKER_INITIAL_STATUS", "").strip() or "Pending review",
        followup_stale_days=int(os.getenv("FOLLOWUP_STALE_DAYS", "").strip() or 7),
        followup_cooldown_days=int(os.getenv("FOLLOWUP_COOLDOWN_DAYS", "").strip() or 7),
        review_effort=_effort(os.getenv("REVIEW_EFFORT")),
        workspace_id=os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip() or None,
        anthropic_enabled=_flag(os.getenv("ANTHROPIC_ENABLED")) is not False,
        deepseek_api_key=deepseek_key,
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "").strip() or "deepseek-flash",
        deepseek_thinking=_flag(os.getenv("DEEPSEEK_THINKING")) is True,
        writer_thinking=_flag(os.getenv("WRITER_THINKING")),
        review_thinking=_flag(os.getenv("REVIEW_THINKING")),
        deepseek_writer_model=os.getenv("DEEPSEEK_WRITER_MODEL", "").strip() or None,
        writer_provider=_provider("WRITER_PROVIDER", os.getenv("WRITER_PROVIDER"), bool(deepseek_key),
                                  default_provider),
        review_provider=_provider("REVIEW_PROVIDER", os.getenv("REVIEW_PROVIDER"), bool(deepseek_key),
                                  default_provider),
    )
