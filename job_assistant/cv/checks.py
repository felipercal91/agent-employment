"""Deterministic checks on a generated CV, run before (and alongside) the HR reviewer agent.

They catch what an LLM reviewer can miss or excuse: invented numbers, technologies absent from
the facts, stock "AI" phrasing, the protected French title, length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from job_assistant.cv.content import ResumeContent
from job_assistant.scoring.fit import extract_keywords
from job_assistant.tailoring.guardrails import protected_title_mentions
from job_assistant.tailoring.market import Language

MAX_PAGES = 2

# Phrases recruiters associate with generated or template CVs.
AI_PHRASES = {
    "en": [
        "spearheaded", "spearheading", "leveraged", "leveraging", "leverage", "results-driven",
        "results-oriented", "passionate", "synergy", "synergies", "dynamic", "cutting-edge",
        "seamless", "seamlessly", "proven track record", "detail-oriented", "self-starter",
        "highly motivated", "go-getter", "thought leader", "best-in-class", "world-class",
        "game-changer", "game-changing", "utilize", "utilized", "utilizing", "orchestrated",
        "fostered", "fostering", "delve", "tapestry", "innovative solutions", "drive innovation",
        "in today's", "ever-evolving", "fast-paced environment", "transformative", "holistic",
        "robust and scalable", "showcasing", "adept at", "instrumental in", "pivotal",
    ],
    "fr": [
        "passionné", "passionnée", "dynamique", "à la pointe", "synergie", "proactif", "proactive",
        "orienté résultats", "orientée résultats", "fort d'une solide", "forte d'une solide",
        "solutions innovantes", "en constante évolution", "levier", "tirer parti", "holistique",
        "incontournable", "véritable", "pilier", "savoir-faire éprouvé",
    ],
}
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass(frozen=True)
class Finding:
    severity: str  # "blocker" | "major" | "minor"
    problem: str
    fix: str


def _norm_number(raw: str) -> str:
    n = raw.replace(",", ".")
    if "." in n:  # "35.0" -> "35", "1.50" -> "1.5"; never touch integer zeros ("10" stays "10")
        n = n.rstrip("0").rstrip(".")
    return n or "0"


def _numbers(text: str) -> set[str]:
    return {_norm_number(n) for n in _NUMBER.findall(text)}


def unsupported_numbers(content: ResumeContent, source: str) -> list[str]:
    """Figures in the CV that appear nowhere in the facts (invented metrics)."""
    allowed = _numbers(source)
    return sorted(_numbers(content.all_text()) - allowed, key=lambda n: float(n.replace(",", ".")))


def ungrounded_technologies(content: ResumeContent, source: str) -> list[str]:
    grounded = set(extract_keywords(source))
    return [k for k in extract_keywords(content.all_text()) if k not in grounded]


def ai_phrases(content: ResumeContent, lang: Language) -> list[str]:
    text = content.all_text().lower()
    return [p for p in AI_PHRASES[lang] if re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text)]


def repeated_openings(content: ResumeContent) -> list[str]:
    """Bullet lead verbs used 3+ times (monotone, template-like rhythm)."""
    counts: dict[str, int] = {}
    for job in content.experience:
        for b in job.bullets:
            word = (b.lead or b.rest).split(" ", 1)[0].lower().strip(",.")
            if word:
                counts[word] = counts.get(word, 0) + 1
    return sorted(w for w, n in counts.items() if n >= 3)


def missing_jd_keywords(content: ResumeContent, jd_text: str, source: str) -> tuple[list[str], list[str]]:
    """JD keywords absent from the CV: (could be added truthfully, true gaps)."""
    jd = extract_keywords(jd_text)
    in_cv = set(extract_keywords(content.all_text() + "\n" + "\n".join(content.tools + content.methodologies)))
    in_facts = set(extract_keywords(source))
    absent = [k for k in jd if k not in in_cv]
    return [k for k in absent if k in in_facts], [k for k in absent if k not in in_facts]


def run_checks(
    content: ResumeContent, lang: Language, source: str, jd_text: str, pages: int | None = None,
    quebec: bool = False,
) -> list[Finding]:
    findings: list[Finding] = []
    numbers = unsupported_numbers(content, source)
    if numbers:
        findings.append(Finding("blocker", f"Figures not found in your facts: {', '.join(numbers)}",
                                "Remove them or use only the figures stated in the facts."))
    techs = ungrounded_technologies(content, source)
    if techs:
        findings.append(Finding("blocker", f"Technologies not in your facts: {', '.join(techs)}",
                                "Remove them; list them as gaps instead."))
    if lang == "fr":
        body = "\n".join([content.title, content.summary, *(b.text() for j in content.experience for b in j.bullets)])
        if protected_title_mentions(body) or re.search(r"\bengineer\b", content.title, re.I):
            findings.append(Finding("blocker", "Uses 'Ingénieur'/'Engineer' as a title in a Quebec CV.",
                                    "Use 'Développeur' (protected title, OIQ)."))
    elif quebec and re.search(r"\bengineer", content.title, re.I):
        # The OIQ protects the title in English too ("engineer") for work in Quebec.
        findings.append(Finding("blocker", "Headline uses 'Engineer' for a Quebec position.",
                                "Use 'Developer' (engineer is a protected title in Quebec, OIQ)."))
    if re.search(r"certified scrum master", content.all_text(), re.I):
        findings.append(Finding("major", "'Certified Scrum Master' reads as the Scrum Alliance CSM.",
                                "Write 'Scrum Master Professional Certificate (SMPC®)'."))
    phrases = ai_phrases(content, lang)
    if phrases:
        findings.append(Finding("major", f"Generic / AI-sounding wording: {', '.join(phrases)}",
                                "Replace with plain, specific verbs and facts."))
    if "—" in content.summary or any("—" in b.text() for j in content.experience for b in j.bullets):
        findings.append(Finding("minor", "Em dashes (—) in summary or bullets read as generated text.",
                                "Use commas or split the sentence."))
    repeats = repeated_openings(content)
    if repeats:
        findings.append(Finding("minor", f"Same opening verb used 3+ times: {', '.join(repeats)}",
                                "Vary the bullet openings."))
    addable, _gaps = missing_jd_keywords(content, jd_text, source)
    if addable:
        findings.append(Finding("major", f"Posting keywords you have but the CV omits: {', '.join(addable)}",
                                "Work them into the summary, competencies or a relevant bullet."))
    if len(content.summary.split()) > 95:
        findings.append(Finding("minor", f"Summary is {len(content.summary.split())} words.",
                                "Keep it under 90 words."))
    if pages is not None and pages > MAX_PAGES:
        findings.append(Finding("major", f"The PDF runs to {pages} pages.",
                                f"Cut weaker bullets to fit {MAX_PAGES} pages."))
    return findings
