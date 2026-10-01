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
    # Required only for API keys not scoped to a workspace.
    workspace_id: str | None = None


def _resolve_dir(value: str | None, default: str) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


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

    return Settings(
        anthropic_api_key=api_key,
        model=os.getenv("CLAUDE_MODEL", "").strip() or DEFAULT_MODEL,
        data_dir=_resolve_dir(os.getenv("DATA_DIR"), "data"),
        output_dir=_resolve_dir(os.getenv("OUTPUT_DIR"), "output"),
        vectorstore_dir=_resolve_dir(os.getenv("VECTORSTORE_DIR"), "vectorstore"),
        workspace_id=os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip() or None,
    )
