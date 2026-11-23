"""HR reviewer agent ("fiscalizador"): reviews a tailored CV the way a recruiter would.

It scores the CV against the posting, checks every claim against the candidate's facts and
flags generic or machine-sounding wording. The pass/fail decision is made in code from its
scores and issues plus the deterministic checks.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, PrivateAttr, field_validator, model_validator

from job_assistant.llm import structured_llm
from job_assistant.cv.checks import Finding
from job_assistant.cv.content import ResumeContent
from job_assistant.cv.master import CVMaster
from job_assistant.tailoring.market import Language

PASS_MIN_SCORE = 7

# Measured by having both reviewers score the same 8 CVs: DeepSeek averaged 6.88 and Claude 6.08
# (+0.8). Subtracting the difference makes a given score mean the same with either provider.
PROVIDER_SCORE_OFFSET = {"anthropic": 0.0, "deepseek": -0.8}

SYSTEM_PROMPT = """You are a senior recruiter in Canada who screens CVs for {field} \
every day, together with the hiring manager's eye for substance. You review ONE tailored CV \
for ONE posting before the candidate sends it. Be demanding and concrete; you are protecting the \
candidate from a rejection.

Today is {today}: dates up to today are in the past.

Scoring anchors: 5 = a recruiter would pass on this CV; 6 = weak spots a recruiter will \
notice; 7 = acceptable once the candidate has read it; 8 = strong, only minor polish; 9-10 = \
exceptional and rare. Start every score at 6 and move it up or down only with evidence from \
the CV text. Most CVs deserve 5-7. Do not give 8 or more unless you can point to the specific \
lines that earn it.

Truthfulness check, bullet by bullet: for every bullet in <cv>, find its source in <facts>. \
Any verb that raises the candidate's role (worked/reviewed/tracked/mentored become \
led/built/owned), any added scope, and any number or technology missing from <facts> is an \
issue (at least major). For every issue, `quote` the exact CV text it refers to.

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
    "es": (
        "The CV is in professional Chilean Spanish, for a posting in Chile: also check the Spanish quality "
        "and that it reads naturally in Chile. Technology names and the English job titles from the facts "
        "stay as they are."
    ),
}
_FRENCH_GENERIC = "The CV is in Quebec French: also check the French quality."


def language_rule(lang: Language, master: CVMaster) -> str:
    if lang == "fr" and not master.protect_engineer_title():
        return _FRENCH_GENERIC
    return _LANGUAGE_RULES[lang]


class Scores(BaseModel):
    # 0-10 each (range enforced in code: numeric bounds are kept out of the output schema)
    requirements_coverage: int
    truthfulness: int
    human_tone: int
    clarity: int
    ats_keywords: int
    candidate_fit: int  # informational: the candidate vs the posting, gaps included

    # Calibration against another provider's scale; not part of the model output or the JSON.
    _offset: float = PrivateAttr(default=0.0)

    @field_validator("*")
    @classmethod
    def _clamp(cls, v: int) -> int:
        return max(0, min(10, v))

    def with_offset(self, offset: float) -> "Scores":
        calibrated = self.model_copy()
        calibrated._offset = offset
        return calibrated

    def adjusted(self, value: float) -> float:
        return max(0.0, value + self._offset)

    def minimum(self) -> int:
        """Lowest CV-quality score (candidate_fit excluded: the writer cannot change it)."""
        return round(self.adjusted(min(self.requirements_coverage, self.truthfulness, self.human_tone,
                                       self.clarity, self.ats_keywords)))

    def raw_average(self) -> float:
        return round((self.requirements_coverage + self.truthfulness + self.human_tone
                      + self.clarity + self.ats_keywords) / 5, 1)

    def average(self) -> float:
        """Average CV-quality score (candidate_fit excluded), calibrated."""
        return round(self.adjusted(self.raw_average()), 1)


class ReviewIssue(BaseModel):
    severity: Literal["blocker", "major", "minor"]
    where: str
    quote: str = ""  # the exact CV text the issue refers to
    problem: str
    fix: str


class HRReview(BaseModel):
    scores: Scores
    issues: list[ReviewIssue]
    gaps_cannot_fix: list[str]
    likely_interview_questions: list[str]
    notes_for_candidate: str

    @model_validator(mode="before")
    @classmethod
    def _fill_secondary_fields(cls, data: Any) -> Any:
        """A reply with scores and issues but missing the closing notes is still a usable review:
        rejecting it lost the whole row. The fields stay required in the schema the model is shown."""
        if isinstance(data, dict):
            return {"gaps_cannot_fix": [], "likely_interview_questions": [], "notes_for_candidate": "", **data}
        return data


def passes(review: HRReview, findings: list[Finding]) -> bool:
    """Good enough to send after the candidate's own review: no blocker, and the CV-quality
    average and the truthfulness score both reach PASS_MIN_SCORE.

    Remaining "major" observations are still listed in the report for the candidate; they no
    longer force another (paid) writer round, because round 2 rarely improved a CV that was
    already at this level.
    """
    return (
        review.scores.average() >= PASS_MIN_SCORE
        and review.scores.adjusted(review.scores.truthfulness) >= PASS_MIN_SCORE
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
        if j.stack:
            lines.append("Stack: " + ", ".join(j.stack))
        lines.append("")
    lines += ["Education: " + "; ".join(master.education(lang)),
              "Certifications: " + master.certifications(lang),
              "Tools: " + ", ".join(content.tools),
              "Methodologies: " + ", ".join(content.methodologies)]
    if master.familiar(lang):
        lines.append("Familiar with / training: " + ", ".join(master.familiar(lang)))
    lines.append("Languages: " + master.languages(lang))
    return "\n".join(lines)


def build_messages(
    jd_text: str, content: ResumeContent, master: CVMaster, lang: Language,
    findings: list[Finding], today: date | None = None,
) -> list[BaseMessage]:
    auto = "\n".join(f"- [{f.severity}] {f.problem}" for f in findings) or "- none"
    day = (today or date.today()).isoformat()
    facts = master.source_text(lang)
    if master.familiar(lang):
        # Kept out of source_text (the keyword and grounding checks must not treat them as skills the
        # candidate has), but the reviewer sees the printed line and must know it is allowed.
        facts += ("\nFamiliar with / training only, NOT professional experience (allowed on the CV only on "
                  "that line): " + ", ".join(master.familiar(lang)))
    return [
        SystemMessage(SYSTEM_PROMPT.format(language_rule=language_rule(lang, master), today=day,
                                           field=master.field())),
        HumanMessage(
            f"<job_posting>\n{jd_text}\n</job_posting>\n\n"
            f"<facts>\n{facts}\n</facts>\n\n"
            f"<cv>\n{render_cv_text(content, master, lang)}\n</cv>\n\n"
            f"<automatic_checks>\n{auto}\n</automatic_checks>"
        ),
    ]


class HRReviewerAgent:
    def __init__(self, llm: BaseChatModel, score_offset: float = 0.0) -> None:
        self._llm = structured_llm(llm, HRReview)
        self._offset = score_offset

    def review(
        self, jd_text: str, content: ResumeContent, master: CVMaster, lang: Language,
        findings: list[Finding],
    ) -> HRReview:
        result = self._llm.invoke(build_messages(jd_text, content, master, lang, findings))
        assert isinstance(result, HRReview)
        if self._offset:
            result = result.model_copy(update={"scores": result.scores.with_offset(self._offset)})
        return result
