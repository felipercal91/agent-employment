"""Independent pass over generated resumes: do they clear the hard filters?

It re-runs the deterministic checks (no LLM) on the saved result of every resume, plus the rules that
only make sense on the finished PDF. A resume FAILS on any blocker, needs REVIEW on a major issue,
and PASSES otherwise.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from job_assistant.cv.checks import MAX_PAGES, Finding, run_checks, skills_only_claims
from job_assistant.cv.content import ResumeContent
from job_assistant.cv.master import CVMaster
from job_assistant.cv.roles import Role, match_role, ungrounded_terms
from job_assistant.tailoring.market import Language, detect_market

# A language level, not a technology: "React Native" and "cloud-native" are fine, "Spanish (native)" is not.
LEVEL_WORDS = re.compile(
    r"\b(C1|C2|B1|B2|fluent|fluency|proficient|proficiency)\b|\bnative\s+(?:speaker|level|proficiency)\b"
    r"|\((?:native|fluent)\)|\bmother tongue\b|\blangue maternelle\b",
    re.IGNORECASE,
)
DASH_PUNCTUATION = re.compile(r"—|(?<=\D)\s+[–-]\s+|\s+[–-]\s+(?=\D)")
PASS, REVIEW, FAIL = "PASA", "REVISAR", "FALLA"


@dataclass
class Verdict:
    name: str
    status: str
    issues: list[Finding] = field(default_factory=list)
    role: str = ""


def pdf_text(pdf: Path) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)


def page_count(pdf: Path) -> int:
    from pypdf import PdfReader

    return len(PdfReader(str(pdf)).pages)


def verify_resume(
    name: str, content: ResumeContent, lang: Language, jd_text: str, pages: int, text_of_pdf: str,
    master: CVMaster, roles: list[Role] | None = None, title: str = "", location: str = "", company: str = "",
) -> Verdict:
    source = master.source_text(lang)
    quebec = detect_market(jd_text, location, company).is_quebec
    # The skills-only claim is re-reported below as a blocker; drop the softer copy from run_checks.
    issues = [f for f in run_checks(content, lang, source, jd_text, pages, quebec, master)
              if not f.problem.startswith("Tools claimed in a bullet")]
    if pages > MAX_PAGES:
        issues.append(Finding("blocker", f"The PDF has {pages} pages (maximum {MAX_PAGES}).", "Cut weaker bullets."))
    claimed = skills_only_claims(content, master, lang)
    if claimed:
        issues.append(Finding("blocker", f"Tools claimed on a job whose facts do not mention them: {', '.join(claimed)}",
                              "Keep them in the Tools line only."))
    role_name = ""
    if roles:
        picked = match_role(title, jd_text, roles).role
        role_name = picked.name
        for term in ungrounded_terms(picked, content.all_text() + " " + " ".join(content.tools), source):
            issues.append(Finding("blocker", f"Claims '{term}' (role {picked.name}) without support in the facts.",
                                  "Remove it."))
    if DASH_PUNCTUATION.search(content.all_text()):
        issues.append(Finding("major", "Dashes used as punctuation in the summary, competencies or bullets.", "Use commas."))
    if LEVEL_WORDS.search(text_of_pdf):
        issues.append(Finding("blocker", "A language level or proficiency word appears in the PDF.", "List languages by name only."))
    if not re.search(r"permanent resident|r[ée]sident permanent", text_of_pdf, re.IGNORECASE):
        issues.append(Finding("major", "No work authorization line in the PDF.", "Add the work authorization line."))
    for field_name in ("email", "phone"):
        value = master.contact.get(field_name, "")
        if value and value not in text_of_pdf:
            issues.append(Finding("major", f"The contact {field_name} is missing from the PDF.", "Check the header."))
    status = FAIL if any(i.severity == "blocker" for i in issues) else REVIEW if any(
        i.severity == "major" for i in issues) else PASS
    return Verdict(name, status, issues, role_name)


def _field(header: str, key: str) -> str:
    m = re.search(rf"^{key}:\s*(.*)$", header, re.MULTILINE)
    return m.group(1).strip() if m else ""


def verify_folder(folder: Path, master: CVMaster, roles: list[Role] | None = None) -> Verdict | None:
    """Verify one posting folder (PDF + resume_content.json + job_description.txt); None if it has no resume."""
    pdfs = sorted(folder.glob("*.pdf"))
    content_file, jd_file = folder / "resume_content.json", folder / "job_description.txt"
    if not (pdfs and content_file.is_file() and jd_file.is_file()):
        return None
    pdf = pdfs[0]
    content = ResumeContent.model_validate(json.loads(content_file.read_text(encoding="utf-8")))
    jd_text = jd_file.read_text(encoding="utf-8").split("\n\n", 1)[-1]
    lang: Language = "fr" if "_CV_" in pdf.name else "en"
    return verify_resume(
        folder.name, content, lang, jd_text, page_count(pdf), pdf_text(pdf), master, roles,
        title=_field(jd_text, "Title"), location=_field(jd_text, "Location"), company=_field(jd_text, "Company"),
    )


def render_verification(verdicts: list[Verdict]) -> str:
    counts = {s: sum(1 for v in verdicts if v.status == s) for s in (PASS, REVIEW, FAIL)}
    lines = [
        "# Verificación de resumes", "",
        f"**{counts[PASS]} pasan · {counts[REVIEW]} para revisar · {counts[FAIL]} fallan** (de {len(verdicts)})", "",
        "| Resultado | Resume | Master | Problemas |", "| --- | --- | --- | --- |",
    ]
    for v in sorted(verdicts, key=lambda v: (v.status != FAIL, v.status != REVIEW, v.name)):
        problems = "; ".join(f"[{i.severity}] {i.problem}" for i in v.issues if i.severity != "minor") or "-"
        lines.append(f"| {v.status} | {v.name} | {v.role or '-'} | {problems.replace('|', '/')} |")
    return "\n".join(lines) + "\n"
