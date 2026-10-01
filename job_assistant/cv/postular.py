"""The "Postular" sheet: postings the user chose; the assistant writes a CV for each one.

Rows with Estado empty / "Nuevo" / "Regenerar" / "Forzar" are processed. "Forzar" skips the
requirements-check stop. Results are written back to the same row.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.worksheet.datavalidation import DataValidation

from job_assistant.agents.hr_reviewer import HRReviewerAgent
from job_assistant.agents.requirements_agent import RequirementsAgent
from job_assistant.agents.resume_writer import ResumeWriterAgent
from job_assistant.cv.jd import JobDescription, JobDescriptionError, fetch_job_description
from job_assistant.cv.master import CVMaster
from job_assistant.cv.pipeline import DEFAULT_MAX_ROUNDS, Renderer, generate_resume, render_review_report
from job_assistant.cv.render import render_pdf
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.profile.store import ProfileStore
from job_assistant.tailoring.drafts import slugify
from job_assistant.tailoring.market import Language, detect_market, is_french_text
from job_assistant.tracker.workbook import (
    SheetTable,
    TableColumn,
    backup,
    ensure_writable,
    write_header,
)

logger = logging.getLogger(__name__)

SHEET = "Postular"

STATUS_READY = "CV listo – revisar"
STATUS_MANUAL = "CV con observaciones"
STATUS_NO_JD = "Falta descripción"
STATUS_MISMATCH = "No cumple requisito"
STATUS_ERROR = "Error"
STATUS_TODO = {"", "nuevo", "regenerar", "forzar"}
MIN_FULL_DESCRIPTION = 200  # chars: a real job description
MIN_SUMMARY = 40  # chars: a pasted summary, used only when the link cannot be read
SUMMARY_WARNING = "Basado en un resumen corto (el link no se pudo leer): pega la descripción completa para un CV más preciso."
STATUSES = ["Nuevo", "Regenerar", "Forzar", STATUS_READY, STATUS_MANUAL, STATUS_NO_JD,
            STATUS_MISMATCH, STATUS_ERROR, "Aplicado", "Descartado"]


def _col(field: str, header: str, width: int, *aliases: str) -> TableColumn:
    return TableColumn(field, header, width, frozenset(aliases))


COLUMNS = [
    _col("link", "Link", 45, "url", "enlace"),
    _col("company", "Empresa", 20, "company"),
    _col("role", "Puesto", 28, "role", "position"),
    _col("location", "Ubicación", 16, "location"),
    _col("language", "Idioma (auto/en/fr)", 10, "idioma", "language"),
    _col("notes", "Notas para el CV", 30, "notas", "notes"),
    _col("description", "Descripción (si el link falla)", 30, "descripcion", "description"),
    _col("status", "Estado", 20, "status"),
    _col("cv_file", "Archivo CV", 40, "cv", "archivo"),
    _col("hr_score", "Puntaje HR", 10, "puntaje", "score"),
    _col("hr_notes", "Observaciones HR", 60, "observaciones"),
    _col("processed", "Procesado", 12, "fecha"),
]


def table(path: Path) -> SheetTable:
    return SheetTable(path, SHEET, COLUMNS)


def ensure_sheet(path: Path) -> bool:
    """Create the Postular sheet (with dropdowns) if missing. Returns True if created."""
    ensure_writable(path)
    wb = load_workbook(path)
    if SHEET in wb.sheetnames:
        return False
    ws = wb.create_sheet(SHEET, 0)  # first tab: it is the daily entry point
    write_header(ws, [(c.field, c.header, c.width) for c in COLUMNS])
    letters = {c.field: ws.cell(1, i).column_letter for i, c in enumerate(COLUMNS, start=1)}
    for field, options in (("status", STATUSES), ("language", ["auto", "en", "fr"])):
        dv = DataValidation(type="list", formula1=f"\"{','.join(options)}\"", allow_blank=True)
        dv.add(f"{letters[field]}2:{letters[field]}500")
        ws.add_data_validation(dv)
    backup(path)
    wb.save(path)
    return True


def pending_rows(path: Path, only_row: int | None = None) -> list[dict[str, Any]]:
    rows = [r for r in table(path).rows() if (r.get("link") or r.get("description"))]
    if only_row is not None:
        return [r for r in rows if r["_row"] == only_row]
    return [r for r in rows if str(r.get("status") or "").strip().lower() in STATUS_TODO]


def choose_language(row_value: object, jd_text: str) -> Language:
    """The posting's own language decides, unless the row forces one."""
    forced = str(row_value or "").strip().lower()
    if forced in {"en", "fr"}:
        return forced  # type: ignore[return-value]
    return "fr" if is_french_text(jd_text) else "en"


def cv_filename(prefix: str, lang: Language, company: str, role: str) -> str:
    kind = "Resume" if lang == "en" else "CV"
    def part(text: str, n: int) -> str:
        return "-".join(w.capitalize() for w in slugify(text, n).split("-")) or "Unknown"
    return f"{prefix}_{kind}_{part(company, 30)}_{part(role, 40)}.pdf"


@dataclass
class RowOutcome:
    row: int
    company: str
    role: str
    status: str
    detail: str
    cv_path: Path | None = None
    score: float | None = None


JDFetcher = Callable[[str], JobDescription]


def process_row(
    row: dict[str, Any],
    *,
    path: Path,
    master: CVMaster,
    profile: CandidateProfile,
    requirements: RequirementsAgent,
    writer: ResumeWriterAgent,
    reviewer: HRReviewerAgent,
    output_dir: Path,
    store: ProfileStore | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    fetch: JDFetcher = fetch_job_description,
    renderer: Renderer = render_pdf,
    today: date | None = None,
) -> RowOutcome:
    today = today or date.today()
    sheet = table(path)
    link = str(row.get("link") or "").strip()
    pasted = str(row.get("description") or "").strip()
    force = str(row.get("status") or "").strip().lower() == "forzar"

    # 1. Job description
    summary_only = False
    if len(pasted) >= MIN_FULL_DESCRIPTION:
        jd = JobDescription(text=pasted, source="pasted in the sheet")
    else:
        try:
            jd = fetch(link)
        except JobDescriptionError as e:
            if len(pasted) >= MIN_SUMMARY:
                # A short summary is better than nothing, but the review will be less precise.
                jd = JobDescription(text=pasted, source="short summary pasted in the sheet")
                summary_only = True
            else:
                sheet.update(row["_row"], {"status": STATUS_NO_JD, "hr_notes": str(e), "processed": today})
                return RowOutcome(row["_row"], str(row.get("company") or ""), str(row.get("role") or ""),
                                  STATUS_NO_JD, str(e))

    company = str(row.get("company") or jd.company or "").strip()
    role = str(row.get("role") or jd.title or "").strip()
    location = str(row.get("location") or jd.location or "").strip()
    header = f"Company: {company}\nTitle: {role}\nLocation: {location}\n\n" if (company or role) else ""
    jd_text = header + jd.text

    # 2. Requirements check (hard mismatches stop here unless the row says Forzar)
    excerpts = store.search(jd_text[:2000], k=6) if store is not None else []
    assessment = requirements.check(jd_text, profile, excerpts)
    company = company or assessment.company or "Unknown"
    role = role or assessment.role_title or "Role"
    location = location or assessment.location
    verify = [f"{r.requirement} ({r.evidence})" for r in assessment.unclear_must_haves]
    if assessment.should_stop and not force:
        reasons = "; ".join(r.requirement for r in assessment.hard_mismatches)
        sheet.update(row["_row"], {
            "status": STATUS_MISMATCH, "company": company, "role": role, "location": location,
            "hr_notes": f"No cumples: {reasons}. Pon Estado = Forzar para generar el CV igual.",
            "processed": today,
        })
        return RowOutcome(row["_row"], company, role, STATUS_MISMATCH, reasons)

    # 3. Writer <-> HR reviewer loop, then PDF
    lang = choose_language(row.get("language"), jd.text)
    folder = output_dir / "cv" / f"{today.isoformat()}_{slugify(company, 30)}_{slugify(role, 40)}"
    pdf = folder / cv_filename(str(master.contact.get("file_prefix", "CV")), lang, company, role)
    quebec = detect_market(jd.text, location, company).is_quebec
    result = generate_resume(jd_text, master, lang, writer, reviewer, pdf,
                             str(row.get("notes") or ""), max_rounds, renderer, quebec)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "job_description.txt").write_text(
        f"{link}\nSource: {jd.source}\n\n{jd_text}\n", encoding="utf-8"
    )
    if assessment.should_stop:
        verify = [f"NO CUMPLES: {r.requirement}" for r in assessment.hard_mismatches] + verify
    report = render_review_report(result, company, role, link, verify)
    if summary_only:
        report = report.replace(
            "\n## Notas del fiscalizador",
            f"\n> **{SUMMARY_WARNING}**\n\n## Notas del fiscalizador",
            1,
        )
    (folder / "revision_hr.md").write_text(report, encoding="utf-8")

    status = STATUS_READY if result.passed else STATUS_MANUAL
    gaps = "; ".join(result.review.gaps_cannot_fix[:3])
    notes = result.review.notes_for_candidate + (f" Gaps: {gaps}." if gaps else "")
    if summary_only:
        notes = f"{SUMMARY_WARNING} {notes}"
    sheet.update(
        row["_row"],
        {
            "status": status, "company": company, "role": role, "location": location,
            "language": lang, "cv_file": pdf.name, "hr_score": result.review.scores.average(),
            "hr_notes": notes, "processed": today,
        },
        links={"cv_file": pdf.resolve().as_uri()},
    )
    return RowOutcome(row["_row"], company, role, status, notes, pdf, result.review.scores.average())
