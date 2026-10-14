"""The daily flow: find new postings, put them in the Postular sheet, and report what happened.

Sources are the ones that need no scraping of job portals: company job boards (public ATS APIs) and
the manual inbox. Anything already in the tracker (any sheet, any status) is never added again, so
what was applied to, discarded or rejected does not come back.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from job_assistant.cv import postular
from job_assistant.cv.postular import (
    STATUS_ERROR,
    STATUS_MANUAL,
    STATUS_MISMATCH,
    STATUS_NO_JD,
    STATUS_READY,
)
from job_assistant.discovery.discover import Candidate, DiscoveryConfig, DiscoveryResult, discover
from job_assistant.discovery.sources import Board, Posting, archive_inbox_file, fetch_boards, read_inbox
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.tracker.excel import Tracker
from job_assistant.tracker.workbook import as_date

MAX_DESCRIPTION_CHARS = 30000  # an Excel cell holds 32,767 characters
REPORT_FOLDER = "Reportes"

Fetcher = Callable[[list[Board]], tuple[list[Posting], list[str]]]
InboxReader = Callable[[Path], list[Posting]]


def posted_date(posting: Posting) -> date | None:
    return as_date(posting.posted_at) if posting.posted_at else None


def is_fresh(posting: Posting, max_age_days: int, today: date) -> bool:
    """Unknown publication date counts as fresh; the inbox is always fresh (the user picked it)."""
    if posting.source == "inbox":
        return True
    published = posted_date(posting)
    return published is None or (today - published).days <= max_age_days


@dataclass
class DailySearch:
    postings_read: int = 0
    from_inbox: int = 0
    stale: int = 0
    discovery: DiscoveryResult = field(default_factory=DiscoveryResult)
    to_add: list[Candidate] = field(default_factory=list)
    over_cap: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Known:
    link: str | None
    company: str | None
    role: str | None


class KnownPostings:
    """Everything already in the workbook (Aplicaciones + Postular, any status), as `discover` expects."""

    def __init__(self, tracker: Tracker | None, postular_path: Path | None) -> None:
        self._tracker, self._path = tracker, postular_path

    def rows(self) -> list[Any]:
        known: list[Any] = list(self._tracker.rows()) if self._tracker else []
        if self._path and self._path.is_file():
            known += [_Known(r.get("link"), r.get("company"), r.get("role")) for r in postular.table(self._path).rows()]
        return known


def search_new(
    profile: CandidateProfile,
    config: DiscoveryConfig,
    tracker: Any,
    inbox_dir: Path,
    today: date,
    *,
    fetch: Fetcher = fetch_boards,
    read: InboxReader = read_inbox,
) -> DailySearch:
    inbox = read(inbox_dir)
    fetched, errors = fetch(config.boards) if config.boards else ([], [])
    postings = inbox + fetched
    fresh = [p for p in postings if is_fresh(p, config.max_age_days, today)]
    result = discover(fresh, profile, config, tracker, errors)
    cap = max(0, config.daily_max_new)
    return DailySearch(
        postings_read=len(postings), from_inbox=len(inbox), stale=len(postings) - len(fresh),
        discovery=result, to_add=result.candidates[:cap], over_cap=max(0, len(result.candidates) - cap),
        errors=list(errors),
    )


def add_to_postular(path: Path, candidates: list[Candidate]) -> list[int]:
    """Append the candidates to the Postular sheet in ONE write (one backup). Returns their rows."""
    records = []
    for c in candidates:
        p = c.posting
        records.append({
            "link": p.url or None, "company": p.company or None, "role": p.title or None,
            "location": p.location or None, "description": p.text[:MAX_DESCRIPTION_CHARS] or None,
            "status": "Nuevo", "source": p.source, "posted": posted_date(p), "fit": c.fit.score,
        })
    return postular.table(path).append_many(records) if records else []


def archive_handled_inbox(search: DailySearch) -> int:
    """Move inbox files that were added or were already tracked; files held back by the cap stay."""
    handled = [c.posting for c in search.to_add] + list(search.discovery.already_tracked)
    moved = 0
    for posting in handled:
        if posting.inbox_file is not None and posting.inbox_file.exists():
            archive_inbox_file(posting.inbox_file)
            moved += 1
    return moved


# ---- report ---------------------------------------------------------------------------------------


@dataclass
class DailyReport:
    today: date
    search: DailySearch
    added: int
    results: list[dict[str, Any]]  # Postular rows processed today
    usage: str = ""
    run_dir: Path | None = None
    recruiters_due: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _cell(text: object, limit: int = 60) -> str:
    value = str(text or "").replace("|", "/").replace("\n", " ").strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def render_daily_report(report: DailyReport) -> str:
    s, d = report.search, report.search.discovery
    ready = sorted((r for r in report.results if r.get("status") in {STATUS_READY, STATUS_MANUAL} and r.get("cv_file")),
                   key=lambda r: -float(r.get("hr_score") or 0))
    stopped = [r for r in report.results if r.get("status") == STATUS_MISMATCH]
    problems = [r for r in report.results if r.get("status") in {STATUS_NO_JD, STATUS_ERROR}]
    lines = [
        f"# Reporte diario {report.today.isoformat()}",
        "",
        "## Resumen",
        "",
        f"- Vacantes leídas: **{s.postings_read}** ({s.from_inbox} de tu bandeja). Fuera de fecha: {s.stale}. "
        f"Descartadas por filtros: {d.filtered_out}.",
        f"- Ya estaban en tu hoja (no se repiten): **{len(d.already_tracked)}**. Bajo el fit mínimo: {len(d.below_min_fit)}.",
        f"- **Nuevas agregadas a Postular: {report.added}**"
        + (f" (hay {s.over_cap} más que calzan; entran los próximos días por el tope diario)." if s.over_cap else "."),
        f"- CVs generados hoy: **{len(ready)}**. Detenidas por requisitos: {len(stopped)}. Con problemas: {len(problems)}.",
    ]
    if report.usage:
        lines.append(f"- Uso de IA: {report.usage}")
    if report.run_dir:
        lines.append(f"- Carpeta de los CVs: `{report.run_dir}`")
    lines += ["", "## CVs para revisar hoy (mejor puntaje primero)", ""]
    if ready:
        lines += ["| Puntaje | Empresa | Puesto | Estado | Archivo |", "| --- | --- | --- | --- | --- |"]
        lines += [
            f"| {r.get('hr_score')} | {_cell(r.get('company'))} | {_cell(r.get('role'))} | {_cell(r.get('status'))} | "
            f"{_cell(r.get('cv_file'), 80)} |" for r in ready
        ]
    else:
        lines.append("- Ninguno hoy.")
    if stopped:
        lines += ["", "## Detenidas por requisito obligatorio (Estado = Forzar para generarlas igual)", ""]
        lines += [f"- {_cell(r.get('company'))} · {_cell(r.get('role'))}: {_cell(r.get('hr_notes'), 160)}" for r in stopped]
    if problems:
        lines += ["", "## Con problemas", ""]
        lines += [f"- {_cell(r.get('company') or r.get('link'))}: {r.get('status')}. {_cell(r.get('hr_notes'), 160)}" for r in problems]
    if report.recruiters_due:
        lines += ["", "## Reclutadores sin contacto reciente", ""] + [f"- {x}" for x in report.recruiters_due]
    if s.errors:
        lines += ["", "## Fuentes que fallaron", ""] + [f"- {e}" for e in s.errors]
    if report.notes:
        lines += ["", "## Avisos", ""] + [f"- {n}" for n in report.notes]
    lines += [
        "", "## Qué sigue", "",
        "1. Lee cada CV (empieza por `revision_hr.md`) y corrige lo que haga falta.",
        "2. En la hoja **Postular**, pon **Aprobado = Sí** en los que quieras postular. Ese es el filtro del paso siguiente.",
        "3. Las vacantes de LinkedIn o Indeed que veas por tu cuenta: guárdalas en `data/inbox/` y entran mañana.",
        "",
    ]
    return "\n".join(lines)


def next_report_path(output_dir: Path, today: date) -> Path:
    base = f"Reporte {today.month}-{today.day}"
    folder = output_dir / REPORT_FOLDER
    candidate, n = folder / f"{base}.md", 1
    while candidate.exists():
        n += 1
        candidate = folder / f"{base} {n}.md"
    return candidate


def write_report(output_dir: Path, today: date, text: str) -> Path:
    path = next_report_path(output_dir, today)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def todays_results(path: Path, today: date) -> list[dict[str, Any]]:
    return [r for r in postular.table(path).rows() if as_date(r.get("processed")) == today]
