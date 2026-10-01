"""Create an empty tracker workbook with the layout every phase of the assistant expects."""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.worksheet import Worksheet

APPLICATION_STATUSES = [
    "Por aplicar",              # passed requirements check, material not prepared yet
    "Pendiente de aprobación",  # material prepared, waiting for your review
    "Aplicado",
    "En proceso",
    "Entrevista",
    "Oferta",
    "Rechazado",
    "Descartado",
]
CONTACT_STATUSES = ["Activo", "Esperando respuesta", "En pausa", "Cerrado"]

APPLICATION_COLUMNS = [
    ("Empresa", 22), ("Puesto", 32), ("Cliente final", 18), ("Ubicación", 18), ("Link", 40),
    ("Fit %", 8), ("Estado", 22), ("Fecha", 12), ("Fuente", 14), ("Notas", 60),
]
CONTACT_COLUMNS = [
    ("Nombre", 22), ("Empresa staffing", 22), ("Cliente final", 18), ("Rol", 30),
    ("Email / LinkedIn", 30), ("Último contacto", 14), ("Estado", 20), ("Próxima acción", 40),
    ("Notas", 50),
]

_HEADER_FONT = Font(bold=True, color="FFFFFF")
_HEADER_FILL = PatternFill("solid", fgColor="1F3864")
_THIN = Side(style="thin", color="BFBFBF")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
FIRST_DATA_ROW = 2


def _setup_sheet(ws: Worksheet, columns: list[tuple[str, int]], statuses: list[str]) -> None:
    for i, (header, width) in enumerate(columns, start=1):
        cell = ws.cell(1, i, header)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = Alignment(vertical="center")
        cell.border = _BORDER
        ws.column_dimensions[cell.column_letter].width = width
    ws.freeze_panes = "A2"
    last_col = ws.cell(1, len(columns)).column_letter
    ws.auto_filter.ref = f"A1:{last_col}1"

    status_col = next(i for i, (h, _) in enumerate(columns, start=1) if h == "Estado")
    letter = ws.cell(1, status_col).column_letter
    validation = DataValidation(
        type="list", formula1=f"\"{','.join(statuses)}\"", allow_blank=True
    )
    validation.add(f"{letter}2:{letter}1000")
    ws.add_data_validation(validation)


def style_data_row(ws: Worksheet, row: int, columns: list[tuple[str, int]]) -> None:
    """Formats for data rows; new rows copy these from the row above."""
    for i, (header, _) in enumerate(columns, start=1):
        cell = ws.cell(row, i)
        cell.border = _BORDER
        cell.alignment = Alignment(vertical="top", wrap_text=header in {"Notas", "Próxima acción"})
        if header == "Fit %":
            cell.number_format = "0%"
        elif header in {"Fecha", "Último contacto"}:
            cell.number_format = "yyyy-mm-dd"


def create_tracker_template(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"{path} already exists; not overwriting it.")
    wb = Workbook()
    apps = wb.active
    apps.title = "Aplicaciones"
    _setup_sheet(apps, APPLICATION_COLUMNS, APPLICATION_STATUSES)
    contacts = wb.create_sheet("Reclutadores")
    _setup_sheet(contacts, CONTACT_COLUMNS, CONTACT_STATUSES)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
