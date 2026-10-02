"""HR reviewer agent ("fiscalizador"): reviews a tailored CV the way a recruiter would.

It scores the CV against the posting, checks every claim against the candidate's facts and
flags generic or machine-sounding wording. The pass/fail decision is made in code from its
scores and issues plus the deterministic checks.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, field_validator

from job_assistant.llm import structured_llm
from job_assistant.cv.checks import Finding
from job_assistant.cv.content import ResumeContent
from job_assistant.cv.master import CVMaster
from job_assistant.tailoring.market import Language

PASS_MIN_SCORE = 7

SYSTEM_PROMPT = """You are a senior technical recruiter in Canada who screens backend developer \
CVs every day, together with the hiring manager's eye for substance. You review ONE tailored CV \
for ONE posting before the candidate sends it. Be demanding and concrete; you are protecting the \
candidate from a rejection.

Today is {today}: dates up to today are in the past.

Score each criterion from 0 to 10. The first five judge the CV as a document, i.e. what the \
writer can still improve using only <facts>:
- requirements_coverage: does the CV surface, where a recruiter sees it in 10 seconds \
(headline, summary, first bullets), the evidence the facts DO contain for the posting's \
requirements? Requirements the facts cannot support (gaps) must NOT lower this score.
- truthfulness: is every claim supported by <facts>? Any inflated scope, invented metric, \
technology or seniority is a blocker.
- human_tone: does it read like an experienced developer wrote it? Flag stock phrases, buzzwords, \
vague claims, repetitive rhythm, the same structure in every bullet, anything that makes a \
recruiter think "generated".
- clarity: easy to scan, concrete, no filler, consistent tense and terminology.
- ats_keywords: does it use the posting's own terms for the skills the candidate really has?
- candidate_fit: separately, how well the candidate's real background matches the posting, \
gaps included. Informational only: it never depends on the writing.

For each problem give an issue with a severity (blocker / major / minor), where it is, what is \
wrong and the exact fix. Fixes may only use facts that exist in <facts>; requirements the \
candidate cannot truthfully meet go in `gaps_cannot_fix`, never in a fix.
Also list `likely_interview_questions`: 3-5 questions the hiring manager will probably ask \
given this CV and this posting (especially about gaps).
Write `notes_for_candidate` in Spanish, 2-4 sentences. Write everything else in English.
{language_rule}"""

_LANGUAGE_RULES = {
    "en": "The CV is in English.",
    "fr": (
        "The CV is in Quebec French: also check the French quality, and that no job title uses "
        "'Ingénieur' or 'Engineer' (protected title in Quebec)."
    ),
}


class Scores(BaseModel):
    # 0-10 each (range enforced in code: numeric bounds are kept out of the output schema)
    requirements_coverage: int
    truthfulness: int
    human_tone: int
    clarity: int
    ats_keywords: int
    candidate_fit: int  # informational: the candidate vs the posting, gaps included

    @field_validator("*")
    @classmethod
    def _clamp(cls, v: int) -> int:
        return max(0, min(10, v))

    def minimum(self) -> int:
        """Lowest CV-quality score (candidate_fit excluded: the writer cannot change it)."""
        return min(self.requirements_coverage, self.truthfulness, self.human_tone,
                   self.clarity, self.ats_keywords)

    def average(self) -> float:
        return round((self.requirements_coverage + self.truthfulness + self.human_tone
                      + self.clarity + self.ats_keywords) / 5, 1)


class ReviewIssue(BaseModel):
    severity: Literal["blocker", "major", "minor"]
    where: str
    problem: str
    fix: str


class HRReview(BaseModel):
    scores: Scores
    issues: list[ReviewIssue]
    gaps_cannot_fix: list[str]
    likely_interview_questions: list[str]
    notes_for_candidate: str


def passes(review: HRReview, findings: list[Finding]) -> bool:
    """Good enough to send after the candidate's own review: no blocker, and the CV-quality
    average and the truthfulness score both reach PASS_MIN_SCORE.

    Remaining "major" observations are still listed in the report for the candidate; they no
    longer force another (paid) writer round, because round 2 rarely improved a CV that was
    already at this level.
    """
    return (
        review.scores.average() >= PASS_MIN_SCORE
        and review.scores.truthfulness >= PASS_MIN_SCORE
        and not any(i.severity == "blocker" for i in review.issues)
        and not any(f.severity == "blocker" for f in findings)
    )


def feedback_lines(review: HRReview, findings: list[Finding]) -> list[str]:
    lines = [f"[{f.severity}] {f.problem} -> {f.fix}" for f in findings]
    lines += [f"[{i.severity}] {i.where}: {i.problem} -> {i.fix}" for i in review.issues]
    return lines


def render_cv_text(content: ResumeContent, master: CVMaster, lang: Language) -> str:
    jobs = {j.id: j for j in master.jobs(lang)}
    lines = [content.title, "", content.summary, "", "Competencies: " + " • ".join(content.competencies), ""]
    for entry in content.experience:
        j = jobs[entry.job_id]
        lines.append(f"{j.title} | {j.company} | {j.location} | {j.dates}")
        lines += [f"- {b.text()}" for b in entry.bullets]
        lines.append("")
    lines += ["Education: " + "; ".join(master.education(lang)),
              "Certifications: " + master.certifications(lang),
              "Tools: " + ", ".join(content.tools),
              "Methodologies: " + ", ".join(content.methodologies),
              "Languages: " + master.languages(lang)]
    return "\n".join(lines)


def build_messages(
    jd_text: str, content: ResumeContent, master: CVMaster, lang: Language,
    findings: list[Finding], today: date | None = None,
) -> list[BaseMessage]:
    auto = "\n".join(f"- [{f.severity}] {f.problem}" for f in findings) or "- none"
    day = (today or date.today()).isoformat()
    return [
        SystemMessage(SYSTEM_PROMPT.format(language_rule=_LANGUAGE_RULES[lang], today=day)),
        HumanMessage(
            f"<job_posting>\n{jd_text}\n</job_posting>\n\n"
            f"<facts>\n{master.source_text(lang)}\n</facts>\n\n"
            f"<cv>\n{render_cv_text(content, master, lang)}\n</cv>\n\n"
            f"<automatic_checks>\n{auto}\n</automatic_checks>"
        ),
    ]


class HRReviewerAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = structured_llm(llm, HRReview)

    def review(
        self, jd_text: str, content: ResumeContent, master: CVMaster, lang: Language,
        findings: list[Finding],
    ) -> HRReview:
        result = self._llm.invoke(build_messages(jd_text, content, master, lang, findings))
        assert isinstance(result, HRReview)
        return result
