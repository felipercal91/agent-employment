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


def load_master(data_dir: Path) -> CVMaster:
    path = data_dir / MASTER_FILENAME
    if not path.is_file():
        raise MasterNotFoundError(f"{path} not found: it holds the fixed facts of your CV.")
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return CVMaster(contact=dict(raw["contact"]), raw=raw)
