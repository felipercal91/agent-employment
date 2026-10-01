"""Requirements Check agent: finds explicit must-have requirements the candidate does not meet.

The LLM only classifies each requirement; the stop/continue decision is made in code:
the pipeline stops when any MUST-HAVE requirement is classified `not_met`.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

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

Never assume the candidate has something that is not written. Never mark a nice_to_have as \
must_have to be cautious. Write everything in English."""


class Requirement(BaseModel):
    requirement: str = Field(description="The requirement, paraphrased briefly")
    kind: Literal["must_have", "nice_to_have"]
    status: Literal["met", "not_met", "unclear"]
    evidence: str = Field(description="Short evidence from the profile/excerpts, or what is missing")


class RequirementsAssessment(BaseModel):
    role_title: str
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


class RequirementsAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        # json_schema = native structured outputs (forced tool_choice is rejected by newer models).
        self._llm = llm.with_structured_output(RequirementsAssessment, method="json_schema")

    def check(
        self, posting_text: str, profile: CandidateProfile, excerpts: list[Hit] | None = None
    ) -> RequirementsAssessment:
        result = self._llm.invoke(build_messages(posting_text, profile, excerpts or []))
        assert isinstance(result, RequirementsAssessment)
        return result
