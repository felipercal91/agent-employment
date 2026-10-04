"""The candidate's fixed CV facts (data/cv_master.toml): employers, titles, dates, education.

The writer agent may select and rephrase bullets but can never change these facts: the PDF
header, job headers, education and certifications are rendered straight from this file.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from job_assistant.tailoring.market import Language

MASTER_FILENAME = "cv_master.toml"


class MasterNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class Job:
    id: str
    company: str
    location: str
    title: str
    dates: str
    bullets: list[str]


@dataclass(frozen=True)
class CVMaster:
    contact: dict[str, str]
    raw: dict[str, Any] = field(repr=False)

    def has_language(self, lang: Language) -> bool:
        """True when every part of the CV facts exists in `lang` (English is always required)."""
        if lang == "en":
            return True
        try:
            jobs = self.jobs(lang)
            return bool(jobs) and all(j.bullets and j.title and j.dates for j in jobs) and bool(
                self.education(lang) and self.certifications(lang) and self.languages(lang)
                and self.tools(lang) and self.methodologies(lang) and self.summary_facts(lang)
                and self.default_title(lang)
            )
        except KeyError:
            return False

    def protect_engineer_title(self) -> bool:
        """Quebec protects 'ingénieur' (OIQ). On by default: only a person who is not an engineer or a
        developer (e.g. compliance) should turn it off with `[meta] protect_engineer_title = false`."""
        return bool(self.raw.get("meta", {}).get("protect_engineer_title", True))

    def label(self, key: str, lang: Language, default: str) -> str:
        """Section label override from `[labels]` (e.g. tools_en = "Systems & Tools:")."""
        return str(self.raw.get("labels", {}).get(f"{key}_{lang}", default))

    def field(self) -> str:
        """The candidate's professional field, used in the prompts (cv_master.toml [meta] field)."""
        return str(self.raw.get("meta", {}).get("field", "senior software developers"))

    def default_title(self, lang: Language) -> str:
        return str(self.raw["default_title"][lang])

    def jobs(self, lang: Language) -> list[Job]:
        jobs = []
        for e in self.raw.get("experience", []):
            jobs.append(Job(
                id=e["id"],
                company=e["company"],
                location=e.get(f"location_{lang}", e["location"]),
                title=e[f"title_{lang}"],
                dates=e[f"dates_{lang}"],
                bullets=list(e.get(f"bullets_{lang}", [])),
            ))
        return jobs

    def job_ids(self) -> list[str]:
        return [e["id"] for e in self.raw.get("experience", [])]

    def education(self, lang: Language) -> list[str]:
        return list(self.raw["education"][lang])

    def certifications(self, lang: Language) -> str:
        return str(self.raw["certifications"][lang])

    def languages(self, lang: Language) -> str:
        return str(self.raw["languages"][lang])

    def tools(self, lang: Language) -> list[str]:
        return list(self.raw["skills"][f"tools_{lang}"])

    def methodologies(self, lang: Language) -> list[str]:
        return list(self.raw["skills"][f"methodologies_{lang}"])

    def summary_facts(self, lang: Language) -> str:
        return str(self.raw.get("summary_facts", {}).get(lang, ""))

    def source_text(self, lang: Language) -> str:
        """Everything the writer may draw on, as plain text (also used for grounding checks)."""
        parts = [self.summary_facts(lang)]
        for job in self.jobs(lang):
            parts.append(f"{job.title} | {job.company} | {job.location} | {job.dates}")
            parts.extend(job.bullets)
        parts += self.education(lang)
        parts += [self.certifications(lang), self.languages(lang)]
        parts += self.tools(lang) + self.methodologies(lang)
        return "\n".join(parts)


PLACEHOLDER_NAME = "ALEX SAMPLE"  # the name used by examples/cv_master.example.toml


class MasterPlaceholderError(MasterNotFoundError):
    """cv_master.toml still holds the example person's data."""


def load_master(data_dir: Path, allow_example: bool = False) -> CVMaster:
    path = data_dir / MASTER_FILENAME
    if not path.is_file():
        raise MasterNotFoundError(f"{path} not found: it holds the fixed facts of your CV.")
    with path.open("rb") as f:
        raw = tomllib.load(f)
    if raw["contact"].get("name") == PLACEHOLDER_NAME and not allow_example:
        raise MasterPlaceholderError(
            f"{path} still has the example data ({PLACEHOLDER_NAME}). Replace it with the real CV facts "
            "before generating CVs: otherwise the CV would carry someone else's employers and dates."
        )
    return CVMaster(contact=dict(raw["contact"]), raw=raw)
