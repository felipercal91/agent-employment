"""Read/write the user's existing Excel tracker without changing its layout.

- Columns are located by header name (Spanish or English aliases), not by position, and the
  header row may sit below a title row. Every sheet with a recognizable header is read.
- Duplicates: same normalized link, or same company + same role.
- Before any write: refuse if Excel has the file open, and save a timestamped backup.
- New rows copy the style/number format of the last existing row in the target sheet.
"""

from __future__ import annotations

import re
import shutil
import unicodedata
from copy import copy
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlsplit

from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet

FIELDS = ("company", "role", "end_client", "location", "link", "fit", "status", "date", "source", "notes")

HEADER_ALIASES: dict[str, set[str]] = {
    "company": {"empresa", "company", "compania", "employer"},
    "end_client": {"cliente final", "end client", "final client", "cliente"},
    "location": {"ubicacion", "location", "ciudad", "city", "lugar"},
    "source": {"fuente", "source", "portal", "canal"},
    "role": {"puesto", "role", "rol", "position", "posicion", "cargo", "title", "job title", "vacante"},
    "link": {"link", "url", "enlace", "posting", "job link"},
    "fit": {"fit", "fit %", "fit%", "match", "match %", "score", "% fit"},
    "status": {"estado", "status", "stage", "etapa"},
    "date": {"fecha", "date", "applied", "fecha aplicacion", "date applied"},
    "notes": {"notas", "notes", "comentarios", "comments", "observaciones"},
}
REQUIRED_FIELDS = {"company", "role"}
HEADER_SCAN_ROWS = 10


class TrackerError(RuntimeError):
    pass


class TrackerLockedError(TrackerError):
    pass


def _norm(text: object) -> str:
    raw = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", raw).strip().lower()


_COMPANY_SUFFIXES = re.compile(r"\b(inc|ltd|ltee|llc|corp|corporation|co|limited|group)\b\.?")


def normalize_company(name: str) -> str:
    text = _norm(name).replace("&", " and ")
    text = _COMPANY_SUFFIXES.sub(" ", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def normalize_link(url: str) -> str:
    """Host + path, lowercased, without scheme, www, query, fragment or trailing slash.

    LinkedIn/Indeed job ids live in the path or `currentJobId`/`jk` query params; keep those.
    """
    url = (url or "").strip()
    if not url:
        return ""
    parts = urlsplit(url if "://" in url else f"https://{url}")
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/").lower()
    keep = [
        q.lower()
        for q in parts.query.split("&")
        if q.split("=")[0].lower() in {"currentjobid", "jk", "gh_jid"}
    ]
    return f"{host}{path}" + (f"?{'&'.join(sorted(keep))}" if keep else "")


def normalize_role(role: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _norm(role)).strip()


@dataclass(frozen=True)
class TrackerRow:
    sheet: str
    row: int
    company: str
    role: str
    link: str
    fit: object
    status: str
    date: object
    notes: str
    location: str = ""
    end_client: str = ""
    source: str = ""


@dataclass
class SheetLayout:
    name: str
    header_row: int
    columns: dict[str, int] = field(default_factory=dict)  # field -> 1-based column index


@dataclass(frozen=True)
class DuplicateCheck:
    duplicates: list[TrackerRow]  # same link, or same company + role
    same_company: list[TrackerRow]  # same company, different role (informational)

    @property
    def is_duplicate(self) -> bool:
        return bool(self.duplicates)


@dataclass(frozen=True)
class NewEntry:
    company: str
    role: str
    link: str = ""
    fit: int | None = None  # 0-100
    status: str = "Pending review"
    notes: str = ""
    location: str = ""
    end_client: str = ""
    source: str = ""
    date: date | None = field(default_factory=date.today)


def _detect_layout(ws: Worksheet) -> SheetLayout | None:
    for r in range(1, min(ws.max_row, HEADER_SCAN_ROWS) + 1):
        columns: dict[str, int] = {}
        for c in range(1, ws.max_column + 1):
            header = _norm(ws.cell(r, c).value)
            for fld, aliases in HEADER_ALIASES.items():
                if header in aliases and fld not in columns:
                    columns[fld] = c
        if REQUIRED_FIELDS <= columns.keys() and len(columns) >= 3:
            return SheetLayout(name=ws.title, header_row=r, columns=columns)
    return None


class Tracker:
    def __init__(self, path: Path, target_sheet: str | None = None) -> None:
        if not path.is_file():
            raise TrackerError(f"Tracker not found: {path}")
        self.path = path
        self.target_sheet = target_sheet

    # ---- reading -----------------------------------------------------------

    def layouts(self) -> list[SheetLayout]:
        wb = load_workbook(self.path, read_only=False)
        return [layout for ws in wb.worksheets if (layout := _detect_layout(ws))]

    def rows(self) -> list[TrackerRow]:
        wb = load_workbook(self.path, data_only=True)
        result: list[TrackerRow] = []
        for ws in wb.worksheets:
            layout = _detect_layout(ws)
            if layout is None:
                continue
            for r in range(layout.header_row + 1, ws.max_row + 1):
                values = {
                    fld: ws.cell(r, col).value for fld, col in layout.columns.items()
                }
                if not values.get("company") and not values.get("role"):
                    continue
                result.append(
                    TrackerRow(
                        sheet=ws.title,
                        row=r,
                        company=str(values.get("company") or "").strip(),
                        role=str(values.get("role") or "").strip(),
                        link=str(values.get("link") or "").strip(),
                        fit=values.get("fit"),
                        status=str(values.get("status") or "").strip(),
                        date=values.get("date"),
                        notes=str(values.get("notes") or "").strip(),
                        location=str(values.get("location") or "").strip(),
                        end_client=str(values.get("end_client") or "").strip(),
                        source=str(values.get("source") or "").strip(),
                    )
                )
        return result

    def find_duplicates(self, company: str, role: str = "", link: str = "") -> DuplicateCheck:
        company_key = normalize_company(company)
        role_key = normalize_role(role)
        link_key = normalize_link(link)
        duplicates: list[TrackerRow] = []
        same_company: list[TrackerRow] = []
        for row in self.rows():
            if link_key and normalize_link(row.link) == link_key:
                duplicates.append(row)
            elif company_key and normalize_company(row.company) == company_key:
                if role_key and normalize_role(row.role) == role_key:
                    duplicates.append(row)
                else:
                    same_company.append(row)
        return DuplicateCheck(duplicates=duplicates, same_company=same_company)

    # ---- writing -----------------------------------------------------------

    def _lock_file(self) -> Path:
        return self.path.with_name(f"~${self.path.name}")

    def backup(self) -> Path:
        backup_dir = self.path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        target = backup_dir / f"{self.path.stem}-{stamp}{self.path.suffix}"
        shutil.copy2(self.path, target)
        return target

    def add(self, entry: NewEntry) -> TrackerRow:
        """Append `entry` to the target sheet. Refuses duplicates and open files."""
        if self._lock_file().exists():
            raise TrackerLockedError(
                f"{self.path.name} is open in Excel. Close it and try again."
            )
        if self.find_duplicates(entry.company, entry.role, entry.link).is_duplicate:
            raise TrackerError(f"{entry.company} / {entry.role} is already in the tracker.")

        wb = load_workbook(self.path)
        layouts = {l.name: l for ws in wb.worksheets if (l := _detect_layout(ws))}
        if not layouts:
            raise TrackerError("No sheet with recognizable headers (company/role) found.")
        sheet_name = self.target_sheet or next(iter(layouts))
        if sheet_name not in layouts:
            raise TrackerError(f"Sheet '{sheet_name}' not found or has no recognizable headers.")
        layout = layouts[sheet_name]
        ws = wb[sheet_name]

        last = _last_data_row(ws, layout)
        new_row = last + 1
        values: dict[str, object] = {
            "company": entry.company,
            "role": entry.role,
            "link": entry.link or None,
            "fit": _fit_value(ws, layout, entry.fit),
            "status": entry.status,
            "date": entry.date,
            "notes": entry.notes or None,
            "location": entry.location or None,
            "end_client": entry.end_client or None,
            "source": entry.source or None,
        }
        for fld, col in layout.columns.items():
            cell = ws.cell(new_row, col)
            if last > layout.header_row:
                _copy_style(ws.cell(last, col), cell)
            cell.value = values[fld]
            if fld == "link" and entry.link:
                cell.hyperlink = entry.link
            if last == layout.header_row:  # first row of an empty sheet: sensible formats
                if fld == "date":
                    cell.number_format = "yyyy-mm-dd"
                elif fld == "fit" and isinstance(cell.value, float):
                    cell.number_format = "0%"

        _extend_autofilter(ws, new_row)
        self.backup()
        wb.save(self.path)
        return TrackerRow(
            sheet=sheet_name, row=new_row, company=entry.company, role=entry.role,
            link=entry.link, fit=values["fit"], status=entry.status, date=entry.date,
            notes=entry.notes, location=entry.location, end_client=entry.end_client,
            source=entry.source,
        )


def _last_data_row(ws: Worksheet, layout: SheetLayout) -> int:
    last = layout.header_row
    key_cols = [layout.columns[f] for f in ("company", "role")]
    for r in range(layout.header_row + 1, ws.max_row + 1):
        if any(ws.cell(r, c).value not in (None, "") for c in key_cols):
            last = r
    return last


def _fit_value(ws: Worksheet, layout: SheetLayout, fit: int | None) -> float | int | None:
    """Match the sheet's convention: fractions (0.85, usually % formatted) or whole numbers."""
    if fit is None or "fit" not in layout.columns:
        return None
    col = layout.columns["fit"]
    existing = [
        ws.cell(r, col).value
        for r in range(layout.header_row + 1, ws.max_row + 1)
        if isinstance(ws.cell(r, col).value, (int, float))
    ]
    if existing:
        as_fraction = all(0 <= v <= 1 for v in existing)
    else:  # empty sheet: a "%" in the header means a percent-formatted fraction
        as_fraction = "%" in str(ws.cell(layout.header_row, col).value or "")
    return round(fit / 100, 2) if as_fraction else fit


def _copy_style(src: object, dst: object) -> None:
    if getattr(src, "has_style", False):
        dst.font = copy(src.font)  # type: ignore[attr-defined]
        dst.border = copy(src.border)  # type: ignore[attr-defined]
        dst.fill = copy(src.fill)  # type: ignore[attr-defined]
        dst.number_format = src.number_format  # type: ignore[attr-defined]
        dst.alignment = copy(src.alignment)  # type: ignore[attr-defined]


def _extend_autofilter(ws: Worksheet, new_row: int) -> None:
    """Keep an existing autofilter range covering the appended row."""
    ref = ws.auto_filter.ref
    if not ref:
        return
    match = re.fullmatch(r"([A-Z]+\d+):([A-Z]+)(\d+)", ref)
    if match and int(match.group(3)) < new_row:
        ws.auto_filter.ref = f"{match.group(1)}:{match.group(2)}{new_row}"
