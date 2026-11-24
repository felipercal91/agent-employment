"""Deterministic checks on a generated CV, run before (and alongside) the HR reviewer agent.

They catch what an LLM reviewer can miss or excuse: invented numbers, technologies absent from
the facts, stock "AI" phrasing, the protected French title, length.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from job_assistant.cv.content import ResumeContent
from job_assistant.cv.master import CVMaster
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
    "es": [
        "apasionado", "apasionada", "dinámico", "dinámica", "de vanguardia", "sinergia", "proactivo",
        "proactiva", "orientado a resultados", "orientada a resultados", "soluciones innovadoras",
        "en constante evolución", "aprovechar", "holístico", "holística", "trayectoria comprobada",
        "imprescindible", "pilar", "hacer realidad", "habilidades blandas",
    ],
}
# Verbs that claim ownership of a team or outcome. A bullet may only use one when the fact it
# came from already does ("worked/reviewed/mentored" must not become "led").
LEAD_VERBS = {
    "en": {"led", "directed", "managed", "owned", "headed", "oversaw", "architected", "spearheaded"},
    "fr": {"dirigé", "géré", "piloté", "supervisé", "architecturé", "dirige", "gère"},
    "es": {"lideré", "dirigí", "gestioné", "administré", "supervisé", "encabecé", "lideró", "dirigió",
           "gestionó", "supervisó", "lidera", "dirige", "gestiona"},
}
_WORD = re.compile(r"[\wÀ-ÿ'-]+")
# Domains and skills that postings ask for and that a model is tempted to claim. If one appears in
# the CV but nowhere in the candidate's facts, it is invented (e.g. "banking" for an insurance
# developer, "secured APIs" when no security work is documented). Stems, matched as word starts.
CLAIM_TERMS = {
    "en": ["banking", "bank", "fintech", "healthcare", "telecom", "retail", "e-commerce", "ecommerce",
           "lending", "mortgage", "payment", "secure", "secured", "security", "authentication",
           "authorization", "encryption", "compliance", "regulatory", "regulated", "cloud-native",
           "serverless", "unit test", "test automation", "automated test", "on-call", "startup"],
    "fr": ["bancaire", "banque", "fintech", "santé", "télécom", "commerce de détail", "commerce électronique",
           "prêt", "hypothécaire", "paiement", "sécurisé", "sécurité", "authentification",
           "chiffrement", "conformité", "réglementaire", "réglementé", "infonuagique natif",
           "tests unitaires", "tests automatisés", "astreinte"],
    "es": ["bancari", "banco", "fintech", "salud", "telecomunicaciones", "retail", "comercio electrónico",
           "préstamo", "hipotecari", "pago", "seguridad", "autenticación", "autorización",
           "cifrado", "encriptación", "cumplimiento normativo", "regulatori", "regulado", "serverless",
           "pruebas unitarias", "pruebas automatizadas", "automatización de pruebas", "turno de guardia",
           "startup"],
}
MIN_MATCH_RATIO = 0.30  # below this, no source bullet is similar enough to judge


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


def unsupported_claim_terms(content: ResumeContent, source: str, lang: Language) -> list[str]:
    """Domain/skill terms present in the CV but absent from the candidate's facts."""
    text, facts = content.all_text().lower(), source.lower()
    found = []
    for term in CLAIM_TERMS[lang]:
        pattern = rf"(?<![\wÀ-ÿ]){re.escape(term)}"
        if re.search(pattern, text) and not re.search(pattern, facts):
            found.append(term)
    return [t for t in found if not any(o != t and t.startswith(o) for o in found)]  # bank covers banking


# A certification the candidate is still working toward must never be written as obtained.
IN_PROGRESS = re.compile(r"in progress|in process|en cours|candidate|pursuing|studying|expected|currently", re.I)
CERTIFICATIONS = ("PMP", "CAMS", "CFE", "CRCM", "CIPP", "FRM", "CISA", "CISSP", "CPA", "CFA")
_SEGMENT = re.compile(r"[.;•\n]")


def _segments_with(text: str, keyword: str) -> list[str]:
    return [seg for seg in _SEGMENT.split(text) if keyword in extract_keywords(seg)]


def in_progress_certification_claims(content: ResumeContent, source: str) -> list[str]:
    """Certifications the facts say are in progress, but the CV text presents without that caveat."""
    flagged = []
    for cert in CERTIFICATIONS:
        in_facts = _segments_with(source, cert)
        if not in_facts or not all(IN_PROGRESS.search(seg) for seg in in_facts):
            continue  # not in the facts, or held: other checks (or nothing) apply
        if any(not IN_PROGRESS.search(seg) for seg in _segments_with(content.all_text(), cert)):
            flagged.append(cert)
    return flagged


def banned_terms_found(content: ResumeContent, terms: list[str]) -> list[str]:
    """Terms from the person's banned list that appear in the tailorable text (whole words)."""
    text = content.all_text()
    return [t for t in terms if re.search(rf"(?<![\wÀ-ÿ]){re.escape(t)}(?![\wÀ-ÿ])", text, re.IGNORECASE)]


def _skill_names(item: str) -> list[str]:
    """Names inside a skills-list entry: "Git & GitHub" -> Git, GitHub; "Java (J2EE, JSP)" -> Java."""
    base = re.sub(r"\(.*?\)", "", item)
    # "/" is not a separator: "Oracle PL/SQL" and "HTML/CSS" are single names.
    names = [n.strip() for n in re.split(r"\s+(?:&|et|and)\s+", base) if len(n.strip()) >= 3]
    # "Java 8/17" must still guard the bare name "Java".
    return names + [bare for n in names if (bare := re.sub(r"\s+\d[\d/.]*$", "", n)) != n and len(bare) >= 3]


def familiar_claims(content: ResumeContent, master: CVMaster, lang: Language) -> list[str]:
    """Training-only skills (Kafka, Docker, JUnit, Mockito...) used as if they were experience: in the
    headline, summary, competencies or any job bullet. They belong on the 'Familiar with' line only."""
    names: list[str] = []
    for item in master.familiar(lang):
        for name in _skill_names(item):
            names.append(name)
            if name.startswith("Apache "):  # "Apache Kafka" is usually written "Kafka"
                names.append(name.split()[-1])
    text = "\n".join([content.title, content.summary, *content.competencies,
                      *(b.text() for j in content.experience for b in j.bullets)])
    flagged: list[str] = []
    for name in names:
        if _has_name(name, text) and name not in flagged:
            flagged.append(name)
    return flagged


def _has_name(name: str, text: str) -> bool:
    return re.search(rf"(?<![\wÀ-ÿ]){re.escape(name)}(?![\wÀ-ÿ])", text, re.IGNORECASE) is not None


def skills_only_claims(content: ResumeContent, master: CVMaster, lang: Language) -> list[str]:
    """Tools a bullet presents as used on a job, when that job's own facts never mention them.

    A tool listed only in the skills section (e.g. Postman) may stay in the Tools line, but a bullet
    like "Performed API checks in Postman" claims an activity the facts do not state.
    """
    facts = {j.id: " ".join([*j.bullets, *j.stack]) for j in master.jobs(lang)}
    names = [n for tool in master.tools(lang) for n in _skill_names(tool)]
    flagged: list[str] = []
    for job in content.experience:
        job_facts = facts.get(job.job_id, "")
        text = " ".join(b.text() for b in job.bullets)
        for name in names:
            if _has_name(name, text) and not _has_name(name, job_facts) and name not in flagged:
                flagged.append(name)
    return flagged


# One profile per resume: the summary and headline match the title, never two roles or a career pivot.
MIXED_PROFILE = re.compile(
    r"\bnow\s+(?:targeting|seeking|pursuing|moving|transitioning|focus(?:ed|ing))\b"
    r"|\b(?:developer|engineer)\s*(?:,|/|&|and)\s*(?:\w+\s+)?analyst\b"
    r"|\banalyst\s*(?:,|/|&|and)\s*(?:\w+\s+)?(?:developer|engineer)\b",
    re.IGNORECASE,
)


# The role a headline or summary presents the candidate as. "lead" is a level, not a profile, so it mixes with any.
PROFILE_FAMILIES = {
    "developer": re.compile(r"\b(?:developers?|engineers?|programmers?|développeu(?:r|se)s?)\b", re.IGNORECASE),
    "analyst": re.compile(r"\b(?:analysts?|analystes?)\b", re.IGNORECASE),
    "consultant": re.compile(r"\b(?:consultants?|conseill(?:er|ère)s?)\b", re.IGNORECASE),
    "project manager": re.compile(r"\b(?:project|product|program) managers?\b|\bchefs? de projet\b", re.IGNORECASE),
    "tester": re.compile(r"\btesters?\b", re.IGNORECASE),
    "lead": re.compile(r"\b(?:technical|tech|team|java) (?:team )?lead(?:er)?\b|\bchef d'équipe\b", re.IGNORECASE),
}


_HONEST_CROSS_PROFILE = re.compile(
    r"Software engineer (?:and technical lead )?with 13\+ years in insurance technology,\s*(?:with strong requirements analysis|(?:including|building) SQL)",
    re.IGNORECASE,
)


def mixed_profile(content: ResumeContent) -> list[str]:
    """Phrases that present two profiles at once, or a career pivot, in the headline and summary."""
    if _HONEST_CROSS_PROFILE.search(content.summary):
        return []
    text = content.title + " " + content.summary
    found = sorted({m.group(0) for m in MIXED_PROFILE.finditer(text)})
    # Only the headline and the first sentence say who the candidate is; later "teams of developers and
    # analysts" describe other people.
    stated = content.title + " " + re.split(r"(?<=[.!?])\s", content.summary, maxsplit=1)[0]
    in_title = {name for name, rx in PROFILE_FAMILIES.items() if rx.search(content.title)}
    in_stated = {name for name, rx in PROFILE_FAMILIES.items() if rx.search(stated)}
    # The summary may only restate the headline's profile; a headline that already names two is mixed too.
    mixed = len(in_title) > 1 or bool(in_title and in_stated - in_title) or (not in_title and len(in_stated) > 1)
    if mixed and not found:
        found.append("presents " + " and ".join(n for n in PROFILE_FAMILIES if n in in_title | in_stated))
    return found


def tailored_body(content: ResumeContent) -> str:
    """Normalized tailorable text (summary + competencies + bullets) used for similarity checks."""
    parts = [content.summary, *content.competencies, *(b.text() for j in content.experience for b in j.bullets)]
    return re.sub(r"\s+", " ", " ".join(parts)).strip().lower()


def near_duplicate_check(
    content: ResumeContent,
    master_text: str = "",
    sibling_texts: list[tuple[str, str]] | None = None,
    threshold: float = 0.88,
) -> list[str]:
    """Flag CVs that are identical or near-identical to the role master or another CV generated today."""
    current = tailored_body(content)
    if not current:
        return []
    problems: list[str] = []
    if master_text:
        norm_master = re.sub(r"\s+", " ", master_text).strip().lower()
        ratio = difflib.SequenceMatcher(None, current, norm_master).ratio()
        if ratio >= threshold:
            problems.append(f"CV is {ratio:.0%} identical to the role master (threshold {threshold:.0%})")
    for label, sib in (sibling_texts or []):
        norm_sib = re.sub(r"\s+", " ", sib).strip().lower()
        if not norm_sib:
            continue
        ratio = difflib.SequenceMatcher(None, current, norm_sib).ratio()
        if ratio >= threshold:
            problems.append(f"CV is {ratio:.0%} identical to same-day CV '{label}' (threshold {threshold:.0%})")
            break
    return problems


def leadership_upgrades(content: ResumeContent, master: CVMaster, lang: Language) -> list[str]:
    """Bullets that open with a leadership verb the matching fact does not use."""
    lead = LEAD_VERBS[lang]
    sources = {j.id: j.bullets for j in master.jobs(lang)}
    problems: list[str] = []
    for job in content.experience:
        facts = sources.get(job.job_id, [])
        for bullet in job.bullets:
            text = bullet.text()
            if not (set(w.lower() for w in _WORD.findall(text)[:3]) & lead):
                continue
            ratio, match = max(
                ((difflib.SequenceMatcher(None, text.lower(), f.lower()).ratio(), f) for f in facts),
                default=(0.0, ""),
            )
            if ratio >= MIN_MATCH_RATIO and not (set(w.lower() for w in _WORD.findall(match)) & lead):
                problems.append(f"'{bullet.lead}' claims leadership; the fact says: '{match[:80]}...'")
    return problems


def run_checks(
    content: ResumeContent, lang: Language, source: str, jd_text: str, pages: int | None = None,
    quebec: bool = False, master: CVMaster | None = None,
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
    protect_title = True if master is None else master.protect_engineer_title()
    if protect_title and lang == "fr":
        body = "\n".join([content.title, content.summary, *(b.text() for j in content.experience for b in j.bullets)])
        if protected_title_mentions(body) or re.search(r"\bengineer\b", content.title, re.I):
            findings.append(Finding("blocker", "Uses 'Ingénieur'/'Engineer' as a title in a Quebec CV.",
                                    "Use 'Développeur' (protected title, OIQ)."))
    elif protect_title and lang == "en" and quebec and re.search(r"\bengineer", content.title, re.I):
        # The OIQ protects the title in English too ("engineer") for work in Quebec.
        findings.append(Finding("blocker", "Headline uses 'Engineer' for a Quebec position.",
                                "Use 'Developer' (engineer is a protected title in Quebec, OIQ)."))
    unfinished = in_progress_certification_claims(content, source)
    if unfinished:
        findings.append(Finding("blocker", f"Certification still in progress, written as obtained: {', '.join(unfinished)}",
                                "Say 'in progress' or leave it out; the facts say it is not finished."))
    if master is not None:
        banned = banned_terms_found(content, master.banned_terms(lang))
        if banned:
            findings.append(Finding("blocker", f"Words the candidate does not want on her CV: {', '.join(banned)}",
                                    "Remove them (see [meta] banned_terms in cv_master.toml)."))
    claims = unsupported_claim_terms(content, source, lang)
    if claims:
        findings.append(Finding("blocker", f"Claims not supported by your facts: {', '.join(claims)}",
                                "Remove them; if the posting asks for them, they are gaps, not claims."))
    if master is not None:
        skills_only = skills_only_claims(content, master, lang)
        if skills_only:
            # A blocker: it forces the writer to redo the draft instead of shipping a claim the facts lack.
            findings.append(Finding("blocker", f"Tools claimed in a bullet that the facts of that job do not mention: {', '.join(skills_only)}",
                                    "Keep them in the Tools line only, or use the job's own facts."))
        training = familiar_claims(content, master, lang)
        if training:
            findings.append(Finding("blocker", f"Training-only skills presented as experience: {', '.join(training)}",
                                    "Remove them from the summary, competencies and bullets; they stay on the 'Familiar with / training' line."))
        upgrades = leadership_upgrades(content, master, lang)
        if upgrades:
            findings.append(Finding("major", "Leadership claim not in the facts: " + "; ".join(upgrades[:3]),
                                    "Use the verb the fact itself uses (worked, reviewed, defined...)."))
    mixed = mixed_profile(content)
    if mixed:
        findings.append(Finding("blocker", f"The headline or summary mixes profiles: {', '.join(mixed)}",
                                "State ONE profile, the one in the headline; no 'developer and analyst', no 'now targeting'."))
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
