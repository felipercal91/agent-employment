"""Render the CV to PDF with the same layout as the candidate's own CV.

HTML/CSS printed by the locally installed Chrome or Edge (Playwright, no browser download).
Font: Carlito/Calibri, US Letter, single column, ATS-friendly (real text, no tables or images).
"""

from __future__ import annotations

import html
from pathlib import Path

from job_assistant.cv.content import ResumeContent, strip_dashes
from job_assistant.cv.master import CVMaster
from job_assistant.tailoring.market import Language

LABELS = {
    "en": {
        "summary": "PROFESSIONAL SUMMARY", "competencies": "CORE COMPETENCIES",
        "experience": "PROFESSIONAL EXPERIENCE", "education": "EDUCATION",
        "skills": "CERTIFICATIONS & TECHNICAL SKILLS", "certs": "Certifications:",
        "tools": "Tools & Technology:", "methods": "Methodologies:", "languages": "Languages:",
        "familiar": "Familiar with / training:", "stack": "Stack:",
    },
    "fr": {
        "summary": "SOMMAIRE PROFESSIONNEL", "competencies": "COMPÉTENCES CLÉS",
        "experience": "EXPÉRIENCE PROFESSIONNELLE", "education": "FORMATION",
        "skills": "CERTIFICATIONS ET COMPÉTENCES TECHNIQUES", "certs": "Certifications :",
        "tools": "Outils et technologies :", "methods": "Méthodologies :", "languages": "Langues :",
        "familiar": "Notions / formation :", "stack": "Stack technique :",
    },
    "es": {
        "summary": "RESUMEN PROFESIONAL", "competencies": "COMPETENCIAS CLAVE",
        "experience": "EXPERIENCIA PROFESIONAL", "education": "EDUCACIÓN",
        "skills": "CERTIFICACIONES Y HABILIDADES TÉCNICAS", "certs": "Certificaciones:",
        "tools": "Herramientas y tecnologías:", "methods": "Metodologías:", "languages": "Idiomas:",
        "familiar": "Conocimientos / capacitación:", "stack": "Stack técnico:",
    },
}

CSS = """
@page { size: Letter; margin: 0.55in 0.6in 0.5in 0.6in; }
* { box-sizing: border-box; }
body { font-family: Carlito, Calibri, 'Segoe UI', sans-serif; font-size: 10.5pt; line-height: 1.22;
       color: #000; margin: 0; }
.name { text-align: center; font-weight: 700; font-size: 17pt; letter-spacing: 0.2px; margin: 0; }
.title { text-align: center; font-weight: 700; font-style: italic; font-size: 11.5pt; margin: 2px 0 2px; }
.contact { text-align: center; font-size: 9.5pt; margin-bottom: 4px; }
h2 { font-size: 10.8pt; font-weight: 700; margin: 13px 0 5px; padding-bottom: 2px;
     border-bottom: 1.1px solid #000; letter-spacing: 0.1px; }
p { margin: 0; }
.job { margin-top: 7px; break-inside: avoid-page; }
.job-head { display: flex; justify-content: space-between; font-weight: 700; }
.job-head .dates { font-weight: 400; font-style: italic; padding-right: 0.9in; }
.company { font-style: italic; margin-bottom: 2px; }
ul { margin: 2px 0 0 0; padding-left: 16px; }
li { margin: 1.5px 0; padding-left: 2px; }
.line { margin: 2px 0; }
.stack { margin: 2px 0 0 0; }
.block { break-inside: avoid-page; }
h2 { break-after: avoid-page; }
"""


def _e(text: str) -> str:
    return html.escape(text, quote=False)


def build_html(master: CVMaster, content: ResumeContent, lang: Language) -> str:
    L = dict(LABELS[lang])
    for key in ("tools", "methods", "familiar"):
        L[key] = master.label(key, lang, L[key])
    L = {k: _e(v) for k, v in L.items()}  # labels reach the HTML: escape the & in "Tools & Technology"
    def skill_line(label: str, items: list[str]) -> str:  # an empty list prints no empty label
        return f'<p class="line"><b>{label}</b> {_e(", ".join(items))}</p>' if items else ""

    c = master.contact_for(lang)  # Chile (es): no mailing address, no Canadian work authorization
    order = ("email", "phone", "linkedin") if lang == "es" else ("email", "linkedin", "location", "phone")
    contact = " | ".join(_e(c[k]) for k in order if c.get(k))
    jobs = {j.id: j for j in master.jobs(lang)}
    authorization = _e(str(c.get(f"work_authorization_{lang}") or c.get("work_authorization") or ""))
    authorization_html = f'<p class="contact">{authorization}</p>' if authorization else ""

    exp_html = []
    for entry in content.experience:
        job = jobs[entry.job_id]
        bullets = "".join(
            f"<li><b>{_e(b.lead)}</b> {_e(b.rest)}</li>" if b.lead else f"<li>{_e(b.rest)}</li>"
            for b in entry.bullets
        )
        stack = f'<p class="stack"><b>{L["stack"]}</b> {_e(", ".join(job.stack))}</p>' if job.stack else ""
        exp_html.append(
            f'<div class="job"><div class="job-head"><span>{_e(job.title)}</span>'
            f'<span class="dates">{_e(job.dates)}</span></div>'
            f'<div class="company">{_e(job.company)} | {_e(job.location)}</div><ul>{bullets}</ul>{stack}</div>'
        )

    education = "".join(f'<p class="line">{_e(strip_dashes(e))}</p>' for e in master.education(lang))
    return f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<p class="name">{_e(c['name'])}</p>
<p class="title">{_e(content.title)}</p>
<p class="contact">{contact}</p>{authorization_html}
<h2>{L['summary']}</h2><p>{_e(content.summary)}</p>
<h2>{L['competencies']}</h2><p>{' • '.join(_e(x) for x in content.competencies)}</p>
<h2>{L['experience']}</h2>{''.join(exp_html)}
<div class="block"><h2>{L['education']}</h2>{education}</div>
<h2>{L['skills']}</h2>
<p class="line"><b>{L['certs']}</b> {_e(strip_dashes(master.certifications(lang)))}</p>
{skill_line(L['tools'], content.tools)}
{skill_line(L['methods'], content.methodologies)}
{skill_line(L['familiar'], master.familiar(lang))}
<p class="line"><b>{L['languages']}</b> {_e(master.languages(lang))}</p>
</body></html>"""


def render_pdf(html_text: str, out_path: Path) -> int:
    """Print `html_text` to `out_path`. Returns the number of pages."""
    from playwright.sync_api import sync_playwright
    from pypdf import PdfReader

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = None
        errors = []
        for channel in ("chrome", "msedge", None):
            try:
                browser = p.chromium.launch(channel=channel) if channel else p.chromium.launch()
                break
            except Exception as e:  # browser not installed: try the next one
                errors.append(f"{channel}: {e}")
        if browser is None:
            raise RuntimeError("No Chrome/Edge found to print the PDF: " + " | ".join(errors))
        page = browser.new_page()
        page.set_content(html_text, wait_until="load")
        page.pdf(path=str(out_path), format="Letter", print_background=True, prefer_css_page_size=True)
        browser.close()
    return len(PdfReader(str(out_path)).pages)
