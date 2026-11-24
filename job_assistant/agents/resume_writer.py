"""Resume writer agent: tailors the candidate's real CV to one job description.

It writes only the tailorable parts (headline, summary, competencies, bullets, skill order);
employers, titles, dates and education come from the fixed facts at render time.
"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from job_assistant.llm import structured_llm
from job_assistant.cv.content import ResumeContent
from job_assistant.cv.master import CVMaster
from job_assistant.cv.roles import RoleBase
from job_assistant.tailoring.market import Language

_LANGUAGE_RULES = {
    "en": "Write in Canadian English.",
    "fr": (
        "Write in French as used in Quebec. The headline and any job title must use "
        "'Développeur' (e.g. 'Développeur Java sénior'), never 'Ingénieur' or 'Engineer': "
        "'ingénieur' is a protected title in Quebec (OIQ) and the candidate is not a member."
    ),
    "es": (
        "Write in professional Chilean Spanish (the resume is for a posting in Chile). Use Chilean job "
        "titles for the headline: 'Analista Funcional', 'Analista de Negocios', 'Líder Técnico', "
        "'Ingeniero de Soporte', 'Desarrollador Java', 'Analista de Datos'. Keep technology names, "
        "certifications and the English job titles in the facts as they are. Bullets in the first person "
        "implied, past tense (Lideré, Diseñé, Implementé); never use 'yo'. Never invent or upgrade a title."
    ),
}
# For a candidate whose field has nothing to do with engineering (`protect_engineer_title = false`).
_FRENCH_GENERIC = (
    "Write in French as used in Quebec. Translate the candidate's real job titles faithfully "
    "and never invent or upgrade a title."
)

QUEBEC_EN_RULE = (
    " This position is in Quebec, where 'engineer' is a protected title (OIQ) in English too: "
    "the headline must say 'Developer' (e.g. 'Senior Java Developer'), never 'Engineer'."
)


def standing_rules(master: CVMaster, lang: Language = "en") -> str:
    """The person's own permanent rules (cv_master.toml [meta] extra_rules), appended to the prompt."""
    rules = master.extra_rules(lang)
    if not rules:
        return ""
    bullets = "\n".join(f"- {r}" for r in rules)
    return f"\n\nStanding rules from the candidate (always follow them):\n{bullets}"


def language_rule(lang: Language, protect_engineer_title: bool, quebec: bool) -> str:
    if lang == "es":
        return _LANGUAGE_RULES["es"]
    if not protect_engineer_title:
        return _LANGUAGE_RULES["en"] if lang == "en" else _FRENCH_GENERIC
    return _LANGUAGE_RULES[lang] + (QUEBEC_EN_RULE if quebec and lang == "en" else "")

SYSTEM_PROMPT = """You are an expert resume writer for {field} in Canada. You \
tailor ONE candidate's real CV to ONE job posting.

Truth rules (never break these):
- Use only facts present in <facts>: the original bullets, the summary facts, the skill lists. \
You may select, merge, reorder, shorten and rephrase them, and choose what to emphasize.
- Every number in your output must appear in <facts>. Never add a metric, a technology, an \
employer, a responsibility or a level of seniority that the facts do not state.
- `tools` and `methodologies` must be items copied exactly from the facts' lists (reordered, \
most relevant to the posting first; drop the irrelevant ones).
- A tool may appear inside a job's bullet only if one of that job's own bullets in <facts> mentions it. A tool that the facts list only in the skills section stays in the Tools line: never turn it into an activity ("performed API checks in Postman") on a job.
- Requirements of the posting that the facts do not support are left out. Do not hint at them.

Writing rules (the CV must read as written by an experienced developer, not generated):
- Plain, concrete verbs: built, designed, reviewed, led, migrated, integrated, cut, delivered. \
Never: spearheaded, leveraged, utilized, orchestrated, fostered, results-driven, passionate, \
dynamic, cutting-edge, seamless, proven track record, innovative solutions, synergy.
- No dashes as punctuation: no em dash, no en dash, no spaced hyphen. Use commas, periods or the \
word "to". Canadian spelling, and CAD for any currency. Vary sentence length and bullet openings; \
never start 3 bullets with the same verb.
- Bullet formula (Google XYZ): accomplished X, measured by Y, by doing Z. Write the bold `lead` \
(3-8 words, past tense) as the accomplishment X, then `rest` with the measure Y and the method Z. \
Use Y only when the facts give a number or a concrete result. When they do not, state X and Z \
plainly: never invent a measure. One or two lines; no filler adjectives, no passive voice.
- Mirror the posting's vocabulary only where the facts support it (e.g. use the posting's own term for a \
tool or practice when the facts show it).
- Headline: the candidate's real level and stack, aligned with the posting's title when truthful.
- Summary: about 3 lines (3 sentences, max 60 words), specific to this posting, first person implied (no "I"). \
It states ONE profile, the one in the headline: never two profiles ("developer and analyst") and never \
"now targeting ... roles".
- Each job's technology stack is printed automatically under its bullets: never write a Stack line.
- Competencies: 6-8 capabilities the posting cares about (e.g. the posting's key skills \
and areas, in its own words), not a copy of the Tools line; avoid repeating the \
same item across competencies, tools and methodologies.
- Years and tenure (CRITICAL): "13+ years" describes overall software engineering / IT / insurance technology career experience (or Java). NEVER attach "13+ years" to Spring Boot (the candidate has 13+ years in software engineering/Java/insurance technology, but only 5 years with Spring Boot, from 2021). Spring Boot must either be stated separately as "5 years" or listed without a year count (e.g. "Software engineer with 13+ years in insurance technology, building production backend services in Java 8/17, including 5 years with Spring Boot microservices").
- Honest titles with 13+ years: Never claim the candidate is a "[Title] with 13+ years" for a role the candidate never held for 13+ years (such as "Business Systems Analyst with 13+ years" or "Senior Data Engineer with 13+ years"). For analyst roles, open with: "Software engineer and technical lead with 13+ years in insurance technology, with strong requirements analysis and stakeholder experience". For data roles, open with: "Software engineer with 13+ years in insurance technology, including SQL, data migration and ETL work".
- Hands-on developer profile (CRITICAL): The summary and the first 3-4 bullets of each job MUST focus on hands-on coding and technical engineering (Java 8/17, Spring Boot, REST APIs, Oracle SQL/PL/SQL, query tuning, microservices, production support/troubleshooting). Put coordination, stakeholder/BA, or PM work into at most ONE bullet at the end of a job.
- Training-only technologies (CRITICAL): Never claim Kafka, React, Docker, or Kubernetes as professional experience in the summary, competencies, or job bullets. They may appear ONLY on the "Familiar with / training" line if present in <facts>.
- Experience: one entry per job id, in the facts' order. Lead each job with hands-on technical work, then leadership/coordination in a single bullet at the end. Never merge unrelated achievements into one bullet. The whole CV must fit on 2 pages.
- The candidate's own notes for this application, if any, are instructions to follow when they do not conflict with the truth rules.

Tailoring (this is what makes the CV worth sending; do all five steps — never output a near-copy of the role master or another CV from the same day):
1. Fill `tailoring_plan` first: the 8-10 most important keywords and requirements of the posting. For each, quote the fact that supports it (or write 'none') and say where it will appear.
2. Select: keep only the bullets that matter for this posting. Most recent job 4-6 bullets, each other job 2-3. Leave out the least relevant ones; do not keep every bullet.
3. Order: inside each job, put the 3-4 most relevant hands-on coding/technical bullets FIRST, and at most one coordination/BA/PM bullet at the end.
4. Rewrite: mirror the posting's top 8-10 supported keywords (verbs, nouns, technology names) across the summary, competencies, and at least half of the kept bullets using ONLY true facts from <facts>, keeping every number and the real scope. Copy a bullet word for word only when it already matches the posting.
5. The headline and summary must answer the posting's top 3 requirements with hands-on evidence from the facts, not generic seniority text.
Example, for a posting that asks for control-testing experience (any field works the same way):
  Fact: "Worked directly with the audit team to define the testing procedures for the new process, \
cutting review time by 30%."
  Good: lead "Defined control-testing procedures", rest "with the audit team for the new process, \
cutting review time by 30%."
  Bad: "Led a company-wide transformation program" (new scope: the fact says worked with).
Keep the verb's real weight: worked/reviewed/tracked/mentored never become led/built/owned.

{language_rule}"""


def _facts(master: CVMaster, lang: Language) -> str:
    jobs = "\n\n".join(
        f"[job_id: {j.id}] {j.title} | {j.company} | {j.location} | {j.dates}\n"
        + "\n".join(f"- {b}" for b in j.bullets)
        + (f"\nStack (printed automatically under the job, never write it yourself): {', '.join(j.stack)}" if j.stack else "")
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
    role_base: RoleBase | None = None,
) -> list[BaseMessage]:
    parts = [
        f"<facts>\n{_facts(master, lang)}\n</facts>",
        f"<job_posting>\n{jd_text}\n</job_posting>",
    ]
    if role_base is not None:
        parts.append(_role_block(role_base, lang))
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
            language_rule=language_rule(lang, master.protect_engineer_title(), quebec),
            field=master.field(),
        ) + standing_rules(master, lang)),
        HumanMessage("\n\n".join(parts)),
    ]


def _role_block(base: RoleBase, lang: Language) -> str:
    role = base.role
    text = (
        f"<target_role>\nRole: {role.name}\nHeadline for this role: {role.headline} (adapt it only to the "
        "posting's own title).\n"
        "Summary rules for this role:\n"
        "- Never attach '13+ years' to a title the candidate did not hold for 13+ years. Specifically, NEVER write "
        "'Business Systems Analyst with 13+ years' or 'Data Engineer with 13+ years'. For BSA/analyst roles, open "
        "with 'Software engineer and technical lead with 13+ years in insurance technology, with strong requirements "
        "analysis and stakeholder experience'. For data roles, open with 'Software engineer with 13+ years in insurance "
        "technology, including SQL, data migration and ETL work'.\n"
        "- Never attach '13+ years' to Spring Boot: candidate has 13+ years in software engineering/Java/insurance "
        "technology, but only 5 years with Spring Boot. Either state Spring Boot separately as '5 years' or list it "
        "without a year count (e.g. 'Software engineer with 13+ years in insurance technology, building production "
        "backend services in Java 8/17, including 5 years with Spring Boot microservices').\n"
        f"ATS keywords of this role: {', '.join(role.ats_keywords)}\n"
        f"Emphasis: {role.focus}\n"
        "Mirror one of these keywords ONLY when the facts support it; the others stay out of the resume. "
        "The summary must be specific to this role and must not read like any other role's summary.\n"
    )
    if role.exclude_jobs:
        text += f"Leave these jobs out of the resume entirely (do not return an entry for them): {', '.join(role.exclude_jobs)}.\n"
    if role.must_include:
        text += ("These master bullets must stay in the resume (keep them, you may reword them to the posting): "
                 + "; ".join(f"job {r.partition(':')[0]} bullet about {r.partition(':')[2]}" for r in role.must_include) + ".\n")
    if base.master_json:
        guide = (f"; write the result in {'French' if lang == 'fr' else 'Spanish'} from the facts, the master is only a guide to emphasis"
                 if lang != "en" else "")
        text += (
            "\nThis candidate's master resume for the role (English), the starting point to adapt to the "
            "posting. Reorder, reword and mirror the posting's keywords; keep what already fits; never "
            f"add anything that is not in <facts>{guide}:\n{base.master_json}\n"
        )
    return text + "</target_role>"


class ResumeWriterAgent:
    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = structured_llm(llm, ResumeContent)

    def write(
        self,
        jd_text: str,
        master: CVMaster,
        lang: Language,
        candidate_notes: str = "",
        previous: ResumeContent | None = None,
        feedback: list[str] | None = None,
        quebec: bool = False,
        role_base: RoleBase | None = None,
    ) -> ResumeContent:
        result = self._llm.invoke(
            build_messages(jd_text, master, lang, candidate_notes, previous, feedback, quebec, role_base)
        )
        assert isinstance(result, ResumeContent)
        return result
