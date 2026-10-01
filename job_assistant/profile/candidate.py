"""Structured candidate profile (skills, years, certifications) loaded from `data/profile.toml`.

This is the explicit, hand-maintained source of truth used for scoring and hard-requirement
checks. Free-text documents (CVs, projects, feedback) live in the RAG index instead.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROFILE_FILENAME = "profile.toml"


@dataclass(frozen=True)
class CandidateProfile:
    name: str
    title: str
    years_experience: int
    skills: list[str]
    certifications: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    spoken_languages: list[str] = field(default_factory=list)
    work_authorization: str = ""
    locations: list[str] = field(default_factory=list)

    def as_prompt_text(self) -> str:
        def fmt(items: list[str]) -> str:
            return ", ".join(items) if items else "(none listed)"

        return "\n".join(
            [
                f"Title: {self.title}",
                f"Years of professional experience: {self.years_experience}",
                f"Skills: {fmt(self.skills)}",
                f"Certifications: {fmt(self.certifications)}",
                f"Education: {fmt(self.education)}",
                f"Spoken languages: {fmt(self.spoken_languages)}",
                f"Work authorization: {self.work_authorization or '(not stated)'}",
                f"Target locations: {fmt(self.locations)}",
            ]
        )


class ProfileNotFoundError(FileNotFoundError):
    pass


def load_candidate_profile(data_dir: Path) -> CandidateProfile:
    path = data_dir / PROFILE_FILENAME
    if not path.is_file():
        raise ProfileNotFoundError(
            f"{path} not found. Copy examples/profile.example.toml to data/profile.toml and fill it in."
        )
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return CandidateProfile(
        name=raw.get("name", ""),
        title=raw["title"],
        years_experience=int(raw["years_experience"]),
        skills=list(raw.get("skills", [])),
        certifications=list(raw.get("certifications", [])),
        education=list(raw.get("education", [])),
        spoken_languages=list(raw.get("spoken_languages", [])),
        work_authorization=raw.get("work_authorization", ""),
        locations=list(raw.get("locations", [])),
    )
