"""The candidate's fixed CV facts (data/cv_master.toml): employers, titles, dates, education.

The writer agent may select and rephrase bullets but can never change these facts: the PDF
header, job headers, education and certifications are rendered straight from this file.
"""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from job_assistant.tailoring.market import Language

MASTER_FILENAME = "cv_master.toml"
SPANISH_FILENAME = "cv_master_es.toml"  # optional overlay: the Spanish (Chile) version of the facts


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
    stack: list[str] = field(default_factory=list)  # technologies used in THIS job, printed under its bullets


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

    def keep_all_tools(self) -> bool:
        """`[skills] keep_all_tools = true`: the tools line always prints every tool, in the file's order."""
        return bool(self.raw.get("skills", {}).get("keep_all_tools", False))

    def extra_rules(self, lang: Language = "en") -> list[str]:
        """`[meta] extra_rules`: standing instructions for the writer, specific to this person."""
        meta = self.raw.get("meta", {})
        extra = meta.get("extra_rules_es", []) if lang == "es" else []
        return [str(r) for r in [*meta.get("extra_rules", []), *extra]]

    def banned_terms(self, lang: Language = "en") -> list[str]:
        """`[meta] banned_terms`: words that must never appear in the tailored text (checked in code).
        Spanish adds `banned_terms_es` (its own wording of the same ideas)."""
        meta = self.raw.get("meta", {})
        extra = meta.get("banned_terms_es", []) if lang == "es" else []
        return [str(t) for t in [*meta.get("banned_terms", []), *extra]]

    def contact_for(self, lang: Language) -> dict[str, str]:
        """The header's contact data. Canada (en/fr) prints it as is. Chile (es) replaces the location with
        `[contact_es]` (no mailing address) and drops the Canadian work authorization."""
        if lang != "es":
            return self.contact
        c = {k: v for k, v in self.contact.items() if not k.startswith("work_authorization")}
        c.pop("location", None)
        c.update({k: str(v) for k, v in self.raw.get("contact_es", {}).items()})
        return c

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
                company=e.get(f"company_{lang}", e["company"]),
                location=e.get(f"location_{lang}", e["location"]),
                title=e[f"title_{lang}"],
                dates=e[f"dates_{lang}"],
                bullets=list(e.get(f"bullets_{lang}", [])),
                stack=list(e.get(f"stack_{lang}", [])),
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

    def familiar(self, lang: Language) -> list[str]:
        """Skills the candidate knows only from training or personal projects: printed on their own
        line, never usable as job experience."""
        return list(self.raw["skills"].get(f"familiar_{lang}", []))

    def summary_facts(self, lang: Language) -> str:
        return str(self.raw.get("summary_facts", {}).get(lang, ""))

    def source_text(self, lang: Language) -> str:
        """Everything the writer may draw on, as plain text (also used for grounding checks)."""
        parts = [self.summary_facts(lang)]
        for job in self.jobs(lang):
            parts.append(f"{job.title} | {job.company} | {job.location} | {job.dates}")
            parts.extend(job.bullets)
            if job.stack:
                parts.append("Stack: " + ", ".join(job.stack))
        parts += self.education(lang)
        parts += [self.certifications(lang), self.languages(lang)]
        parts += self.tools(lang) + self.methodologies(lang)
        return "\n".join(parts)


PLACEHOLDER_NAME = "ALEX SAMPLE"  # the name used by examples/cv_master.example.toml


class MasterPlaceholderError(MasterNotFoundError):
    """cv_master.toml still holds the example person's data."""


def merge_spanish(raw: dict[str, Any], es: dict[str, Any]) -> dict[str, Any]:
    """Add the Spanish overlay to the master's data. Only adds keys (`*_es`, `[contact_es]`, ...): every
    English and French value stays as it is, so Canada's resumes cannot change."""
    merged = copy.deepcopy(raw)
    for job in es.get("experience", []):
        target = next((e for e in merged.get("experience", []) if e["id"] == job["id"]), None)
        if target is not None:
            target.update({k: v for k, v in job.items() if k != "id"})
    for section in ("default_title", "education", "certifications", "languages", "skills", "summary_facts", "meta"):
        merged.setdefault(section, {}).update(es.get(section, {}))
    if "contact_es" in es:
        merged["contact_es"] = dict(es["contact_es"])
    return merged


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
    spanish = data_dir / SPANISH_FILENAME
    if spanish.is_file():
        with spanish.open("rb") as f:
            raw = merge_spanish(raw, tomllib.load(f))
    return CVMaster(contact=dict(raw["contact"]), raw=raw)
