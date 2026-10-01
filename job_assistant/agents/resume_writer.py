"""Resume writer agent: tailors the candidate's real CV to one job description.

It writes only the tailorable parts (headline, summary, competencies, bullets, skill order);
employers, titles, dates and education come from the fixed facts at render time.
"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from job_assistant.cv.content import ResumeContent
from job_assistant.cv.master import CVMaster
from job_assistant.tailoring.market import Language

_LANGUAGE_RULES = {
    "en": "Write in Canadian English.",
    "fr": (
        "Write in French as used in Quebec. The headline and any job title must use "
        "'Développeur' (e.g. 'Développeur Java sénior'), never 'Ingénieur' or 'Engineer': "
        "'ingénieur' is a protected title in Quebec (OIQ) and the candidate is not a member."
    ),
}

QUEBEC_EN_RULE = (
    " This position is in Quebec, where 'engineer' is a protected title (OIQ) in English too: "
    "the headline must say 'Developer' (e.g. 'Senior Java Developer'), never 'Engineer'."
)

SYSTEM_PROMPT = """You are an expert resume writer for senior software developers in Canada. You \
tailor ONE candidate's real CV to ONE job posting.

Truth rules (never break these):
- Use only facts present in <facts>: the original bullets, the summary facts, the skill lists. \
You may select, merge, reorder, shorten and rephrase them, and choose what to emphasize.
- Every number in your output must appear in <facts>. Never add a metric, a technology, an \
employer, a responsibility or a level of seniority that the facts do not state.
- `tools` and `methodologies` must be items copied exactly from the facts' lists (reordered, \
most relevant to the posting first; drop the irrelevant ones).
- Requirements of the posting that the facts do not support are left out. Do not hint at them.

Writing rules (the CV must read as written by an experienced developer, not generated):
- Plain, concrete verbs: built, designed, reviewed, led, migrated, integrated, cut, delivered. \
Never: spearheaded, leveraged, utilized, orchestrated, fostered, results-driven, passionate, \
dynamic, cutting-edge, seamless, proven track record, innovative solutions, synergy.
- No em dashes. Vary sentence length and bullet openings; never start 3 bullets with the same verb.
- Each bullet: a bold `lead` of 3-8 words that states what was done, then `rest` with the \
context and the result. One or two lines; no filler adjectives.
- Mirror the posting's vocabulary only where the facts support it (e.g. say "Spring Boot \
microservices" when the posting does and the facts show it).
- Headline: the candidate's real level and stack, aligned with the posting's title when truthful.
- Summary: 3-4 sentences, max 90 words, specific to this posting, first person implied (no "I").
- Competencies: 6-8 capabilities the posting cares about (e.g. "Spring Boot microservices", \
"SQL Server and PostgreSQL data layers"), not a copy of the Tools line; avoid repeating the \
same item across competencies, tools and methodologies.
- Years: "13+ years" describes the whole career. Never attach a long duration to a specific \
technology unless the job dates in the facts support it (check which job mentions it and when).
- Experience: one entry per job id, in the facts' order. Most recent job 4-6 bullets, the \
others 2-4. For an individual-contributor posting, lead each job with hands-on technical work, \
then leadership. Never merge unrelated achievements into one bullet. The whole CV must fit \
on 2 pages.
- The candidate's own notes for this application, if any, are instructions to follow when \
they do not conflict with the truth rules.

{language_rule}"""


def _facts(master: CVMaster, lang: Language) -> str:
    jobs = "\n\n".join(
        f"[job_id: {j.id}] {j.title} | {j.company} | {j.location} | {j.dates}\n"
        + "\n".join(f"- {b}" for b in j.bullets)
        for j in master.jobs(lang)
    )
    return (
        f"Summary facts:\n{master.summary_facts(lang)}\n\nJobs:\n{jobs}\n\n"
        f"Education:\n" + "\n".join(master.education(lang)) + "\n\n"
        f"Certifications: {master.certifications(lang)}\n"
        f"Languages: {master.languages(lang)}\n"
        f"Tools list: {json.dumps(master.tools(lang), ensure_ascii=False)}\n"
        f"Methodologies list: {json.dumps(master.methodologies(lang), ensure_ascii=False)}"
    )


def build_messages(
    jd_text: str,
    master: CVMaster,
    lang: Language,
    candidate_notes: str = "",
    previous: ResumeContent | None = None,
    feedback: list[str] | None = None,
    quebec: bool = False,
) -> list[BaseMessage]:
    parts = [
        f"<facts>\n{_facts(master, lang)}\n</facts>",
        f"<job_posting>\n{jd_text}\n</job_posting>",
    ]
    if candidate_notes:
        parts.append(f"<candidate_notes>\n{candidate_notes}\n</candidate_notes>")
    if previous is not None:
        parts.append(
            "<previous_draft>\n" + previous.model_dump_json(indent=1) + "\n</previous_draft>\n"
            "<reviewer_feedback>\n" + "\n".join(f"- {f}" for f in feedback or []) + "\n</reviewer_feedback>\n"
            "Revise the draft to resolve every point of the feedback, keeping what already works. "
            "If a point asks for something the facts do not support, do not invent it."
        )
    return [
        SystemMessage(SYSTEM_PROMPT.format(
            language_rule=_LANGUAGE_RULES[lang] + (QUEBEC_EN_RULE if quebec and lang == "en" else "")
        )),
        HumanMessage("\n\n".join(parts)),
    ]


class ResumeWriterAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm.with_structured_output(ResumeContent, method="json_schema")

    def write(
        self,
        jd_text: str,
        master: CVMaster,
        lang: Language,
        candidate_notes: str = "",
        previous: ResumeContent | None = None,
        feedback: list[str] | None = None,
        quebec: bool = False,
    ) -> ResumeContent:
        result = self._llm.invoke(
            build_messages(jd_text, master, lang, candidate_notes, previous, feedback, quebec)
        )
        assert isinstance(result, ResumeContent)
        return result
