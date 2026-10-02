"""Tailoring agent: rewrites the CV headline, summary and bullets for one posting.

It may reorder, rephrase and emphasize what the candidate's documents already say; it may not
add experience. Gaps are reported back instead of being papered over.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from job_assistant.llm import structured_llm
from job_assistant.agents.requirements_agent import RequirementsAssessment
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.profile.store import Hit
from job_assistant.scoring.fit import FitResult
from job_assistant.tailoring.market import Market

_LANGUAGE_RULES = {
    "en": "Write the CV content in English (Canadian spelling).",
    "fr": (
        "Write the CV content in French as used in Quebec. The candidate's job title must use "
        "'Développeur' (e.g. 'Développeur Java senior', 'Développeur backend'), never "
        "'Ingénieur': 'ingénieur' is a protected title in Quebec (OIQ) and the candidate is "
        "not a member. Do not use 'Engineer' either."
    ),
}

SYSTEM_PROMPT = """You tailor a senior backend developer's CV to one specific job posting.

Hard rules:
- Use ONLY facts present in the base CV, the candidate profile and the document excerpts. \
You may reorder, merge, shorten and rephrase them, and choose which ones to emphasize.
- Never add a technology, employer, title, metric, certification or responsibility that the \
sources do not state. Never inflate seniority, scope or numbers.
- If the posting asks for something the sources do not support, do not mention it in the CV: \
list it under `gaps` instead.
- Mirror the posting's terminology only where the sources support it (e.g. say "event-driven \
microservices" if the sources show Kafka-based services).
- Each experience bullet must carry an `evidence` field quoting or pinpointing its source.
- Bullets: start with a strong verb, one line or two, concrete, no buzzword padding.

{language_rule}"""


class TailoredBullet(BaseModel):
    text: str = Field(description="CV bullet, in the target language")
    evidence: str = Field(description="Where this comes from: short quote or source name")


class TailoredCV(BaseModel):
    headline: str = Field(description="Job title line for the top of the CV")
    summary: str = Field(description="3-4 sentence professional summary")
    experience_bullets: list[TailoredBullet] = Field(description="6-10 bullets, most relevant first")
    skills: list[str] = Field(description="Skills section, most relevant to the posting first")
    gaps: list[str] = Field(description="Posting requirements NOT supported by the sources")
    reviewer_notes: str = Field(description="Short notes for the candidate, in English")


def build_messages(
    posting_text: str,
    market: Market,
    base_cv: str,
    profile: CandidateProfile,
    fit: FitResult,
    requirements: RequirementsAssessment,
    excerpts: list[Hit],
) -> list[BaseMessage]:
    excerpt_text = "\n\n".join(f"[{h.source}]\n{h.text}" for h in excerpts) or "(none)"
    unclear = [r.requirement for r in requirements.requirements if r.status != "met"]
    return [
        SystemMessage(SYSTEM_PROMPT.format(language_rule=_LANGUAGE_RULES[market.language])),
        HumanMessage(
            f"<job_posting>\n{posting_text}\n</job_posting>\n\n"
            f"<base_cv>\n{base_cv}\n</base_cv>\n\n"
            f"<candidate_profile>\n{profile.as_prompt_text()}\n</candidate_profile>\n\n"
            f"<document_excerpts>\n{excerpt_text}\n</document_excerpts>\n\n"
            f"<screening>\nKeywords covered by the profile: {', '.join(fit.matched) or '-'}\n"
            f"Keywords NOT in the profile: {', '.join(fit.missing) or '-'}\n"
            f"Requirements not confirmed: {'; '.join(unclear) or '-'}\n</screening>"
        ),
    ]


class TailoringAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = structured_llm(llm, TailoredCV)

    def tailor(
        self,
        posting_text: str,
        market: Market,
        base_cv: str,
        profile: CandidateProfile,
        fit: FitResult,
        requirements: RequirementsAssessment,
        excerpts: list[Hit],
    ) -> TailoredCV:
        result = self._llm.invoke(
            build_messages(posting_text, market, base_cv, profile, fit, requirements, excerpts)
        )
        assert isinstance(result, TailoredCV)
        return result
