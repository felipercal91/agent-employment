"""Application Q&A agent: drafts answers to application-form questions.

Two safety nets against answering a question "as asked" when it does not fit reality:
1. Deterministic premise rules (e.g. a question assuming a resignation when the profile says
   layoff) run before the LLM and are always reported.
2. The LLM is told to flag any other premise mismatch instead of answering it as-is.
Answers that need a number or decision the profile does not contain are marked
`needs_your_input` rather than invented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from job_assistant.llm import structured_llm
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.profile.loaders import SourceDocument
from job_assistant.tailoring.market import Language

# ---- deterministic premise rules ----------------------------------------------

_RESIGNATION = re.compile(
    r"\b(d[ée]mission|d[ée]missionn[ée]e?|resign(ed|ation)?|quit|voluntarily left|"
    r"renuncia|renunci[óo])\b",
    re.IGNORECASE,
)
_DISMISSAL = re.compile(
    r"\b(fired|terminated for cause|dismissed|cong[ée]di[ée]|renvoy[ée]|despid)\w*",
    re.IGNORECASE,
)
_CURRENT_EMPLOYER = re.compile(
    r"\b(current (employer|manager|company)|employeur actuel|"
    r"emploi actuel|notify your employer)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PremiseIssue:
    question_index: int
    issue: str


def check_premises(questions: list[str], profile: CandidateProfile) -> list[PremiseIssue]:
    reason = profile.situation.get("employment_end_reason", "").strip().lower()
    issues: list[PremiseIssue] = []
    for i, question in enumerate(questions):
        if reason == "layoff" and _RESIGNATION.search(question):
            issues.append(PremiseIssue(i, "Question assumes you resigned; your last job ended in a layoff."))
        if reason in {"layoff", "resignation", "contract_end"} and _DISMISSAL.search(question):
            issues.append(PremiseIssue(i, f"Question assumes a dismissal; your last job ended by {reason}."))
        if reason in {"layoff", "resignation", "contract_end"} and _CURRENT_EMPLOYER.search(question):
            issues.append(PremiseIssue(i, "Question assumes a current employer; you are not currently employed."))
    return issues


# ---- LLM drafting ---------------------------------------------------------------

SYSTEM_PROMPT = """You draft answers to job-application form questions for ONE candidate. \
The candidate reviews every answer before anything is submitted; nothing you write is sent.

Rules:
- Use only facts from the candidate profile (including its situation facts) and the \
candidate's past answers. Use past answers as the model for tone, length and stance.
- PREMISES: if a question assumes something that contradicts the candidate's facts (e.g. \
it assumes a resignation but the facts say layoff; assumes a current employer; assumes a \
degree or licence they lack), set `premise_issue` to explain the mismatch and write an \
answer that states the real situation politely instead of accepting the premise. Known \
mismatches detected beforehand are listed for you; always include them.
- If an answer needs information the facts do not contain (a salary figure, a start date, \
a yes/no the candidate has not stated), do not guess: set `needs_your_input` to what is \
missing and leave a clear placeholder like [YOUR SALARY RANGE] in the answer.
- Consents, attestations, background-check authorizations, disclosure agreements and any \
legal declaration: draft the most likely answer but always set `needs_your_decision` to \
true. The candidate decides those personally.
- Compensation: only use the range in the facts. Never lower it or invent one.
- Keep answers short and professional, as a form field expects.

{language_rule}"""

_LANGUAGE_RULES = {
    "en": "Write the answers in English.",
    "fr": (
        "Write the answers in French as used in Quebec. If a job title is needed, use "
        "'Développeur', never 'Ingénieur'."
    ),
}


class DraftAnswer(BaseModel):
    question: str
    answer: str
    premise_issue: str = Field(description="Empty string if the question's premise is fine")
    needs_your_input: str = Field(description="What information is missing; empty if none")
    needs_your_decision: bool = Field(description="True for consents, attestations, legal declarations")


class FormAnswers(BaseModel):
    answers: list[DraftAnswer]


def build_messages(
    questions: list[str],
    profile: CandidateProfile,
    history: list[SourceDocument],
    known_issues: list[PremiseIssue],
    language: Language,
    posting_text: str = "",
) -> list[BaseMessage]:
    numbered = "\n\n".join(f"Q{i + 1}. {q}" for i, q in enumerate(questions))
    past = "\n\n".join(f"[{d.source}]\n{d.text}" for d in history) or "(no past answers yet)"
    issues = "\n".join(f"- Q{p.question_index + 1}: {p.issue}" for p in known_issues) or "- none"
    posting = f"<job_posting>\n{posting_text}\n</job_posting>\n\n" if posting_text else ""
    return [
        SystemMessage(SYSTEM_PROMPT.format(language_rule=_LANGUAGE_RULES[language])),
        HumanMessage(
            f"<candidate_profile>\n{profile.as_prompt_text()}\n</candidate_profile>\n\n"
            f"<past_answers>\n{past}\n</past_answers>\n\n{posting}"
            f"<known_premise_issues>\n{issues}\n</known_premise_issues>\n\n"
            f"<questions>\n{numbered}\n</questions>\n\n"
            "Answer every question, in order, one entry per question."
        ),
    ]


class QAAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = structured_llm(llm, FormAnswers)

    def answer(
        self,
        questions: list[str],
        profile: CandidateProfile,
        history: list[SourceDocument],
        language: Language,
        posting_text: str = "",
    ) -> tuple[FormAnswers, list[PremiseIssue]]:
        issues = check_premises(questions, profile)
        result = self._llm.invoke(
            build_messages(questions, profile, history, issues, language, posting_text)
        )
        assert isinstance(result, FormAnswers)
        if len(result.answers) == len(questions):  # show the form's exact wording
            result = FormAnswers(
                answers=[a.model_copy(update={"question": q}) for a, q in zip(result.answers, questions)]
            )
        return merge_premise_issues(result, issues), issues


def merge_premise_issues(result: FormAnswers, issues: list[PremiseIssue]) -> FormAnswers:
    """Guarantee deterministic issues show up even if the model omitted them."""
    answers = list(result.answers)
    for issue in issues:
        if issue.question_index < len(answers):
            a = answers[issue.question_index]
            if issue.issue not in a.premise_issue:
                joined = f"{issue.issue} {a.premise_issue}".strip()
                answers[issue.question_index] = a.model_copy(update={"premise_issue": joined})
    return FormAnswers(answers=answers)


def parse_questions(text: str) -> list[str]:
    """Questions separated by blank lines; if there are none, one question per line."""
    text = text.strip()
    if not text:
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) == 1:
        blocks = [line.strip() for line in text.splitlines() if line.strip()]
    return [re.sub(r"^\s*(Q?\d+[.)]|[-*])\s*", "", b) for b in blocks]


def render_answers_markdown(
    answers: FormAnswers, company: str, role: str, language: Language
) -> str:
    lines = [
        f"# Application form answers: {role} @ {company}",
        "",
        "**Status: PENDING YOUR APPROVAL**. Nothing has been submitted. "
        "Copy answers into the form yourself after reviewing them.",
        f"Language: {language}",
        "",
    ]
    for i, a in enumerate(answers.answers, start=1):
        lines += [f"## Q{i}. {a.question}", ""]
        if a.premise_issue:
            lines += [f"> ⚠ **Premise mismatch:** {a.premise_issue}", ""]
        if a.needs_your_input:
            lines += [f"> ✎ **Needs your input:** {a.needs_your_input}", ""]
        if a.needs_your_decision:
            lines += ["> ⚖ **Your decision:** consent/legal declaration. Decide personally.", ""]
        lines += [a.answer, ""]
    return "\n".join(lines)


AnswerFlag = Literal["premise", "input", "decision"]


def flag_counts(answers: FormAnswers) -> dict[AnswerFlag, int]:
    return {
        "premise": sum(1 for a in answers.answers if a.premise_issue),
        "input": sum(1 for a in answers.answers if a.needs_your_input),
        "decision": sum(1 for a in answers.answers if a.needs_your_decision),
    }
