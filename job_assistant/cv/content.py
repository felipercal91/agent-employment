"""What the writer agent produces: only the tailorable parts of the CV."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from job_assistant.cv.master import CVMaster
from job_assistant.tailoring.market import Language


class Bullet(BaseModel):
    lead: str = Field(description="Opening phrase shown in bold, 3-8 words, starts with a past-tense verb")
    rest: str = Field(description="The rest of the bullet, continuing the sentence")

    def text(self) -> str:
        return f"{self.lead} {self.rest}".strip()


class JobBullets(BaseModel):
    job_id: str = Field(description="Id of the job from the facts (e.g. 'ust')")
    bullets: list[Bullet]


class PlanItem(BaseModel):
    requirement: str = Field(description="One of the posting's most important requirements")
    evidence: str = Field(description="The fact (short quote) that supports it, or 'none' if the facts do not")
    placement: str = Field(description="Where it will appear: summary, a job's bullet, competencies or nowhere")


class ResumeContent(BaseModel):
    # Written first, so the writer decides what the posting needs before writing anything else.
    tailoring_plan: list[PlanItem] = Field(
        default_factory=list, description="The 5-8 key requirements of the posting mapped to the facts"
    )
    title: str = Field(description="Headline under the name, e.g. 'Senior Java Developer'")
    summary: str = Field(description="Professional summary, 3-4 sentences, at most 90 words")
    competencies: list[str] = Field(description="8-12 core competencies, most relevant first")
    experience: list[JobBullets] = Field(description="One entry per job id, in the facts' order")
    tools: list[str] = Field(description="Tools & Technology, chosen ONLY from the facts' list, most relevant first")
    methodologies: list[str] = Field(description="Methodologies, chosen ONLY from the facts' list")

    def all_text(self) -> str:
        parts = [self.title, self.summary, *self.competencies]
        for job in self.experience:
            parts.extend(b.text() for b in job.bullets)
        return "\n".join(parts)


# An em dash, or a spaced en dash / hyphen unless it sits between two digits (a range like 2021 – 2026).
_DASH_PUNCTUATION = re.compile(r"\s*—\s*|(?<=\D)\s+[–-]\s+|\s+[–-]\s+(?=\D)")


def strip_dashes(text: str) -> str:
    """Dashes used as punctuation become commas. Ranges such as 2021-2026 and words like "cloud-native" stay."""
    return _DASH_PUNCTUATION.sub(", ", text)


def without_dash_punctuation(content: ResumeContent) -> ResumeContent:
    return content.model_copy(update={
        "title": strip_dashes(content.title),
        "summary": strip_dashes(content.summary),
        "competencies": [strip_dashes(c) for c in content.competencies],
        "experience": [
            job.model_copy(update={"bullets": [
                Bullet(lead=strip_dashes(b.lead), rest=strip_dashes(b.rest)) for b in job.bullets
            ]})
            for job in content.experience
        ],
    })


def normalize(content: ResumeContent, master: CVMaster, lang: Language) -> tuple[ResumeContent, list[str]]:
    """Enforce structure the facts dictate; return the content and what had to be fixed."""
    problems: list[str] = []
    by_id = {j.job_id: j for j in content.experience}
    ordered: list[JobBullets] = []
    for job_id in master.job_ids():
        if job_id in by_id and by_id[job_id].bullets:
            ordered.append(by_id[job_id])
        else:
            problems.append(f"Job '{job_id}' had no bullets; the original bullets were used.")
            original = next(j for j in master.jobs(lang) if j.id == job_id)
            ordered.append(JobBullets(job_id=job_id, bullets=[Bullet(lead="", rest=b) for b in original.bullets]))
    unknown_jobs = set(by_id) - set(master.job_ids())
    if unknown_jobs:
        problems.append(f"Unknown job ids dropped: {', '.join(sorted(unknown_jobs))}")

    def subset(items: list[str], allowed: list[str], label: str) -> list[str]:
        allowed_map = {a.casefold(): a for a in allowed}
        kept, dropped = [], []
        for item in items:
            match = allowed_map.get(item.strip().casefold())
            if match and match not in kept:
                kept.append(match)
            elif not match:
                dropped.append(item)
        if dropped:
            problems.append(f"{label}: not in your facts, removed: {', '.join(dropped)}")
        return kept or list(allowed)

    tools = master.tools(lang) if master.keep_all_tools() else subset(content.tools, master.tools(lang), "Tools")
    methods = subset(content.methodologies, master.methodologies(lang), "Methodologies")
    fixed = content.model_copy(update={"experience": ordered, "tools": tools, "methodologies": methods})
    return without_dash_punctuation(fixed), problems
