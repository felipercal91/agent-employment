"""Shared workbook plumbing: write safety (lock check + rotating backups) and header-mapped tables."""

from __future__ import annotations

import re
import shutil
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.worksheet import Worksheet

MAX_BACKUPS = 30
HEADER_SCAN_ROWS = 10

_HEADER_FONT = Font(bold=True, color="FFFFFF")
_HEADER_FILL = PatternFill("solid", fgColor="1F3864")
_THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


class WorkbookError(RuntimeError):
    pass


class WorkbookLockedError(WorkbookError):
    pass


def norm_header(text: object) -> str:
    raw = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", raw).strip().lower()


def ensure_writable(path: Path) -> None:
    if path.with_name(f"~${path.name}").exists():
        raise WorkbookLockedError(f"{path.name} is open in Excel. Close it and try again.")


def backup(path: Path) -> Path:
    """Copy `path` to <dir>/backups/, keeping the newest MAX_BACKUPS copies."""
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    target = backup_dir / f"{path.stem}-{stamp}{path.suffix}"
    shutil.copy2(path, target)
    old = sorted(backup_dir.glob(f"{path.stem}-*{path.suffix}"))
    for stale in old[:-MAX_BACKUPS]:
        stale.unlink()
    return target


def write_header(ws: Worksheet, columns: list[tuple[str, str, int]]) -> None:
    for i, (_, header, width) in enumerate(columns, start=1):
        cell = ws.cell(1, i, header)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = Alignment(vertical="center")
        cell.border = BORDER
        ws.column_dimensions[cell.column_letter].width = width
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{ws.cell(1, len(columns)).column_letter}1"


def as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue
    return None


@dataclass(frozen=True)
class TableColumn:
    field: str
    header: str  # header written when the sheet is created
    width: int
    aliases: frozenset[str]
    date: bool = False


class SheetTable:
    """A sheet used as a table: one header row, one record per row, columns found by alias."""

    def __init__(self, path: Path, sheet_name: str, columns: list[TableColumn]) -> None:
        self.path = path
        self.sheet_name = sheet_name
        self.columns = columns

    def _map(self, ws: Worksheet) -> tuple[int, dict[str, int]] | None:
        for r in range(1, min(ws.max_row, HEADER_SCAN_ROWS) + 1):
            found: dict[str, int] = {}
            for c in range(1, ws.max_column + 1):
                header = norm_header(ws.cell(r, c).value)
                for col in self.columns:
                    if col.field not in found and (header in col.aliases or header == norm_header(col.header)):
                        found[col.field] = c
            if len(found) >= max(2, len(self.columns) // 2):
                return r, found
        return None

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        wb = load_workbook(self.path, data_only=True)
        if self.sheet_name not in wb.sheetnames:
            return []
        ws = wb[self.sheet_name]
        mapping = self._map(ws)
        if mapping is None:
            return []
        header_row, cols = mapping
        result = []
        for r in range(header_row + 1, ws.max_row + 1):
            record: dict[str, Any] = {f: ws.cell(r, c).value for f, c in cols.items()}
            if all(v in (None, "") for v in record.values()):
                continue
            for col in self.columns:
                record.setdefault(col.field, None)
                if isinstance(record[col.field], str):
                    record[col.field] = record[col.field].strip()
            record["_row"] = r
            result.append(record)
        return result

    def _open_for_write(self) -> tuple[Workbook, Worksheet, int, dict[str, int]]:
        ensure_writable(self.path)
        wb = load_workbook(self.path)
        if self.sheet_name not in wb.sheetnames:
            ws = wb.create_sheet(self.sheet_name)
            write_header(ws, [(c.field, c.header, c.width) for c in self.columns])
        ws = wb[self.sheet_name]
        mapping = self._map(ws)
        if mapping is None:
            raise WorkbookError(f"Sheet '{self.sheet_name}' has no recognizable header row.")
        return wb, ws, *mapping

    def _set(self, ws: Worksheet, row: int, cols: dict[str, int], values: dict[str, Any]) -> None:
        for fld, value in values.items():
            if fld not in cols:
                continue
            cell = ws.cell(row, cols[fld])
            cell.value = value
            cell.border = BORDER
            cell.alignment = Alignment(vertical="top", wrap_text=isinstance(value, str) and len(value) > 40)
            if isinstance(value, date):
                cell.number_format = "yyyy-mm-dd"

    def append(self, values: dict[str, Any]) -> int:
        wb, ws, header_row, cols = self._open_for_write()
        last = header_row
        for r in range(header_row + 1, ws.max_row + 1):
            if any(ws.cell(r, c).value not in (None, "") for c in cols.values()):
                last = r
        row = last + 1
        self._set(ws, row, cols, values)
        ref = ws.auto_filter.ref
        if ref and (m := re.fullmatch(r"([A-Z]+\d+):([A-Z]+)(\d+)", ref)) and int(m.group(3)) < row:
            ws.auto_filter.ref = f"{m.group(1)}:{m.group(2)}{row}"
        backup(self.path)
        wb.save(self.path)
        return row

    def update(self, row: int, values: dict[str, Any]) -> None:
        wb, ws, header_row, cols = self._open_for_write()
        if row <= header_row:
            raise WorkbookError(f"Row {row} is not a data row.")
        self._set(ws, row, cols, values)
        backup(self.path)
        wb.save(self.path)
