"""Follow-up agent: drafts a short follow-up message to a recruiter or about an application."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from job_assistant.llm import structured_llm
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.tailoring.market import Language

SYSTEM_PROMPT = """You draft a follow-up message for a job seeker. The candidate reviews and \
sends it personally; you never send anything.

Rules:
- Short: at most 120 words in the body. One clear, low-pressure ask (an update, next steps, \
or a short call).
- Use only the context given: names, roles, companies, dates and what was discussed. Never \
invent an interview, a promise, a deadline or a detail that is not in the context.
- If no recipient name is given, use a neutral greeting.
- Warm, professional, confident; no apologies for following up, no flattery.
- Sign with the candidate's name if given, otherwise leave [YOUR NAME].

{language_rule}"""

_LANGUAGE_RULES = {
    "en": "Write in English.",
    "fr": (
        "Write in French as used in Quebec (vouvoiement). If the candidate's title appears, "
        "use 'Développeur', never 'Ingénieur'."
    ),
}


@dataclass(frozen=True)
class FollowUpTarget:
    kind: str  # "recruiter" | "application"
    recipient: str  # person name (may be empty for applications)
    company: str
    role: str
    days_since: int | None
    context: str  # notes, status, next action, channel...


class FollowUpMessage(BaseModel):
    subject: str = Field(description="Email subject or LinkedIn message opener")
    body: str


def build_messages(
    target: FollowUpTarget, profile: CandidateProfile, language: Language
) -> list[BaseMessage]:
    since = f"{target.days_since} days" if target.days_since is not None else "unknown"
    return [
        SystemMessage(SYSTEM_PROMPT.format(language_rule=_LANGUAGE_RULES[language])),
        HumanMessage(
            f"<candidate>\nName: {profile.name or '(not given)'}\nTitle: {profile.title}\n</candidate>\n\n"
            f"<follow_up>\nType: {target.kind}\nRecipient: {target.recipient or '(unknown)'}\n"
            f"Company: {target.company or '(unknown)'}\nRole: {target.role or '(unknown)'}\n"
            f"Time since last contact: {since}\nContext:\n{target.context or '(none)'}\n"
            f"</follow_up>"
        ),
    ]


class FollowUpAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = structured_llm(llm, FollowUpMessage)

    def draft(
        self, target: FollowUpTarget, profile: CandidateProfile, language: Language
    ) -> FollowUpMessage:
        result = self._llm.invoke(build_messages(target, profile, language))
        assert isinstance(result, FollowUpMessage)
        return result
