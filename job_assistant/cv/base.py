"""The candidate's base CV, straight from cv_master.toml: no model, no cost, nothing rewritten.

Useful as the always-ready general version, and to check a layout or a data change (a removed tool,
a language line) before any tailored CV is generated.
"""

from __future__ import annotations

from pathlib import Path

from job_assistant.cv.content import Bullet, JobBullets, ResumeContent
from job_assistant.cv.master import CVMaster
from job_assistant.cv.render import build_html, render_pdf
from job_assistant.tailoring.market import Language

BASE_FOLDER = "Resumen base"
# Words after which a bullet's bold opening phrase can end ("... across", "... with", ...).
_BREAKERS = {"for", "to", "across", "by", "with", "in", "on", "and", "while", "through", "using", "from", "under"}
MIN_LEAD_WORDS, MAX_LEAD_WORDS, FALLBACK_LEAD_WORDS = 3, 9, 5


def split_lead(text: str) -> Bullet:
    """Bold opening phrase + the rest, cut at a natural boundary (a comma or a linking word)."""
    words = text.split()
    cut = min(FALLBACK_LEAD_WORDS, len(words))
    for k in range(MIN_LEAD_WORDS, min(MAX_LEAD_WORDS, len(words) - 1) + 1):
        if words[k - 1].endswith(",") or words[k].lower() in _BREAKERS:
            cut = k
            break
    return Bullet(lead=" ".join(words[:cut]), rest=" ".join(words[cut:]))


def base_content(master: CVMaster, lang: Language) -> ResumeContent:
    jobs = [
        JobBullets(job_id=job.id, bullets=[split_lead(b) for b in job.bullets])
        for job in master.jobs(lang)
    ]
    return ResumeContent(
        title=master.default_title(lang),
        summary=master.summary_facts(lang),
        competencies=master.methodologies(lang),
        experience=jobs,
        tools=master.tools(lang),
        methodologies=[],  # already shown as the competencies: do not print the same list twice
    )


def build_base_cv(master: CVMaster, lang: Language, output_dir: Path) -> tuple[Path, int]:
    """Write the base CV as a PDF in `output_dir/Resumen base/`. Returns (path, pages)."""
    kind = "Resume" if lang == "en" else "CV"
    pdf = output_dir / BASE_FOLDER / f"{master.contact.get('file_prefix', 'CV')}_{kind}_Base.pdf"
    pages = render_pdf(build_html(master, base_content(master, lang), lang), pdf)
    return pdf, pages
