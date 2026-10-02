"""Requirements Check agent: finds explicit must-have requirements the candidate does not meet.

The LLM only classifies each requirement; the stop/continue decision is made in code:
the pipeline stops when any MUST-HAVE requirement is classified `not_met`.
"""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from job_assistant.llm import structured_llm
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.profile.store import Hit

SYSTEM_PROMPT = """You screen a job posting against ONE candidate's real profile before any \
application material is prepared. Your job is to catch hard blockers early so the candidate \
does not waste time on roles where they will likely be filtered out.

1. Extract the requirements the posting states. Classify each as:
   - must_have: stated as required/mandatory/must/minimum ("X required", "must hold", \
"minimum N years", a certification or degree "required"). Also legal requirements such as \
work authorization, security clearance, or mandatory language fluency.
   - nice_to_have: stated as preferred/asset/plus/bonus, or listed without any wording that \
makes it mandatory.
2. For each requirement decide the status, using ONLY the structured profile and the \
document excerpts you are given:
   - met: the profile or excerpts explicitly show it.
   - not_met: the requirement is concrete and checkable (a named certification, licence, \
degree, clearance, language, years threshold, citizenship) and the profile clearly lacks it. \
A certification not listed under Certifications is not held.
   - unclear: the information is insufficient to decide. Use this for skills the profile \
simply does not mention, rather than guessing either way.
3. Quote the evidence briefly (from the profile or an excerpt), or explain what is missing.

Location, commute, relocation and work mode (on-site, hybrid, remote) are NOT hard blockers: the \
candidate lives in Toronto, is willing to relocate and accepts any work mode, so mark those \
met or unclear, never not_met.

Never assume the candidate has something that is not written. Never mark a nice_to_have as \
must_have to be cautious. Write everything in English."""


class Requirement(BaseModel):
    requirement: str = Field(description="The requirement, paraphrased briefly")
    kind: Literal["must_have", "nice_to_have"]
    status: Literal["met", "not_met", "unclear"]
    evidence: str = Field(description="Short evidence from the profile/excerpts, or what is missing")


class RequirementsAssessment(BaseModel):
    company: str = Field(description="Hiring company as named in the posting; empty string if not stated")
    role_title: str
    location: str = Field(description="Job location as stated (city/province, or 'Remote'); empty if not stated")
    requirements: list[Requirement]

    @property
    def hard_mismatches(self) -> list[Requirement]:
        return [r for r in self.requirements if r.kind == "must_have" and r.status == "not_met"]

    @property
    def unclear_must_haves(self) -> list[Requirement]:
        return [r for r in self.requirements if r.kind == "must_have" and r.status == "unclear"]

    @property
    def should_stop(self) -> bool:
        return bool(self.hard_mismatches)


def build_messages(
    posting_text: str, profile: CandidateProfile, excerpts: list[Hit]
) -> list[BaseMessage]:
    excerpt_text = (
        "\n\n".join(f"[{i}] ({h.source})\n{h.text}" for i, h in enumerate(excerpts, start=1))
        or "(no documents indexed)"
    )
    return [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(
            f"<candidate_profile>\n{profile.as_prompt_text()}\n</candidate_profile>\n\n"
            f"<document_excerpts>\n{excerpt_text}\n</document_excerpts>\n\n"
            f"<job_posting>\n{posting_text}\n</job_posting>"
        ),
    ]


# Location, commute and work-mode requirements are never hard blockers for this candidate: they
# live in Toronto, will relocate and accept any work mode (see the profile's situation facts).
# A model that marks one as not_met is guessing, and a false stop wastes the candidate's time.
LOGISTICS = re.compile(
    r"\b(commut\w*|relocat\w*|on-?site|in-?office|office|hybrid|remote|reside\w*|located|based in|"
    r"work from|days? (a|per) (week|wk)|presenc\w*|sur place|bureau|hybride|télétravail)\b",
    re.IGNORECASE,
)


def soften_logistics(assessment: RequirementsAssessment) -> RequirementsAssessment:
    """Downgrade not_met location/work-mode must-haves to unclear (to be verified, not blocking)."""
    fixed = []
    for r in assessment.requirements:
        if r.kind == "must_have" and r.status == "not_met" and LOGISTICS.search(r.requirement):
            note = "Location/work mode: the candidate lives in Toronto, will relocate and accepts any work mode."
            r = r.model_copy(update={"status": "unclear", "evidence": f"{note} ({r.evidence})"})
        fixed.append(r)
    return assessment.model_copy(update={"requirements": fixed})


class RequirementsAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        # json_schema = native structured outputs (forced tool_choice is rejected by newer models).
        self._llm = structured_llm(llm, RequirementsAssessment)

    def check(
        self, posting_text: str, profile: CandidateProfile, excerpts: list[Hit] | None = None
    ) -> RequirementsAssessment:
        result = self._llm.invoke(build_messages(posting_text, profile, excerpts or []))
        assert isinstance(result, RequirementsAssessment)
        return soften_logistics(result)
