"""Scaffold the folders and starter files for one person (the default profile or a named one)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from job_assistant.tracker.template import create_tracker_template

SUBFOLDERS = ("cv", "projects", "feedback", "inbox", "postings", "qa_history")
# example file -> name it gets inside the person's data folder
STARTER_FILES = {
    "profile.example.toml": "profile.toml",
    "cv_master.example.toml": "cv_master.toml",
    "discovery.example.toml": "discovery.toml",
}
PROFILE_ENV_TEMPLATE = """\
# Settings for this person only. Shared ones (API keys, models, providers) live in the main .env.
# Where her files live is decided by the profile folder itself: do not set DATA_DIR / OUTPUT_DIR /
# VECTORSTORE_DIR / TRACKER_PATH here.
#
# Uncomment only what should differ from the main .env:
# DEEPSEEK_API_KEY=
# WRITER_PROVIDER=deepseek
# REVIEW_PROVIDER=deepseek
"""


@dataclass
class ScaffoldResult:
    created: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)


def scaffold(
    data_dir: Path, tracker_path: Path, examples_dir: Path, profile_env: Path | None = None
) -> ScaffoldResult:
    """Create what is missing; never overwrite anything the person has already edited."""
    result = ScaffoldResult()
    for sub in SUBFOLDERS:
        folder = data_dir / sub
        (result.kept if folder.is_dir() else result.created).append(f"{sub}/")
        folder.mkdir(parents=True, exist_ok=True)
    for example, name in STARTER_FILES.items():
        target = data_dir / name
        if target.exists():
            result.kept.append(name)
        else:
            shutil.copyfile(examples_dir / example, target)
            result.created.append(name)
    if tracker_path.exists():
        result.kept.append(tracker_path.name)
    else:
        create_tracker_template(tracker_path)
        result.created.append(tracker_path.name)
    if profile_env is not None:
        if profile_env.exists():
            result.kept.append(".env (profile)")
        else:
            profile_env.parent.mkdir(parents=True, exist_ok=True)
            profile_env.write_text(PROFILE_ENV_TEMPLATE, encoding="utf-8")
            result.created.append(".env (profile)")
    return result
