"""Recruiter/contact CRM ("Reclutadores" sheet) and follow-up history ("Seguimientos" sheet).

The history is the assistant's memory: a follow-up is never drafted for a recipient or company
that already has a draft or a sent message within the cooldown window.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from job_assistant.tracker.workbook import SheetTable, TableColumn, WorkbookError, as_date

CONTACTS_SHEET = "Reclutadores"
FOLLOWUPS_SHEET = "Seguimientos"
INACTIVE_STATUSES = {"cerrado", "en pausa", "closed", "paused"}
FOLLOWUP_DRAFT = "Borrador"
FOLLOWUP_SENT = "Enviado"
FOLLOWUP_DISCARDED = "Descartado"


def _col(field: str, header: str, width: int, *aliases: str, is_date: bool = False) -> TableColumn:
    return TableColumn(field, header, width, frozenset(aliases), is_date)


CONTACT_COLUMNS = [
    _col("name", "Nombre", 22, "nombre", "name", "recruiter", "reclutador", "contacto"),
    _col("agency", "Empresa staffing", 22, "empresa staffing", "agency", "staffing", "agencia", "empresa"),
    _col("end_client", "Cliente final", 18, "cliente final", "end client", "client", "cliente"),
    _col("role", "Rol", 30, "rol", "role", "puesto", "position"),
    _col("channel", "Email / LinkedIn", 30, "email / linkedin", "email", "linkedin", "contacto via", "canal"),
    _col("last_contact", "Último contacto", 14, "ultimo contacto", "last contact", "fecha", is_date=True),
    _col("status", "Estado", 20, "estado", "status"),
    _col("next_action", "Próxima acción", 40, "proxima accion", "next action", "next step"),
    _col("notes", "Notas", 50, "notas", "notes"),
]

FOLLOWUP_COLUMNS = [
    _col("date", "Fecha", 12, "fecha", "date", is_date=True),
    _col("recipient", "Destinatario", 24, "destinatario", "recipient", "para"),
    _col("company", "Empresa", 22, "empresa", "company"),
    _col("kind", "Tipo", 14, "tipo", "kind", "type"),
    _col("status", "Estado", 14, "estado", "status"),
    _col("file", "Archivo", 60, "archivo", "file", "draft"),
    _col("notes", "Notas", 40, "notas", "notes"),
]


def norm(text: object) -> str:
    raw = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()


@dataclass(frozen=True)
class Contact:
    row: int
    name: str
    agency: str
    end_client: str
    role: str
    channel: str
    last_contact: date | None
    status: str
    next_action: str
    notes: str

    @property
    def active(self) -> bool:
        return norm(self.status) not in INACTIVE_STATUSES

    def days_since_contact(self, today: date) -> int | None:
        return (today - self.last_contact).days if self.last_contact else None

    @property
    def company(self) -> str:
        return self.end_client or self.agency


def _contact(record: dict[str, Any]) -> Contact:
    return Contact(
        row=record["_row"],
        name=str(record["name"] or ""),
        agency=str(record["agency"] or ""),
        end_client=str(record["end_client"] or ""),
        role=str(record["role"] or ""),
        channel=str(record["channel"] or ""),
        last_contact=as_date(record["last_contact"]),
        status=str(record["status"] or ""),
        next_action=str(record["next_action"] or ""),
        notes=str(record["notes"] or ""),
    )


class ContactBook:
    def __init__(self, path: Path) -> None:
        self.table = SheetTable(path, CONTACTS_SHEET, CONTACT_COLUMNS)

    def all(self) -> list[Contact]:
        return [_contact(r) for r in self.table.rows() if r.get("name")]

    def find(self, name: str) -> Contact:
        """Exact (normalized) name match first, then a unique partial match."""
        key = norm(name)
        contacts = self.all()
        exact = [c for c in contacts if norm(c.name) == key]
        partial = [c for c in contacts if key and key in norm(c.name)]
        matches = exact or partial
        if len(matches) != 1:
            found = ", ".join(c.name for c in matches) or "none"
            raise WorkbookError(f"Contact '{name}' must match exactly one row (found: {found}).")
        return matches[0]

    def add(self, name: str, *, agency: str = "", end_client: str = "", role: str = "",
            channel: str = "", status: str = "Activo", next_action: str = "", notes: str = "",
            last_contact: date | None = None) -> int:
        if any(norm(c.name) == norm(name) for c in self.all()):
            raise WorkbookError(f"Contact '{name}' already exists.")
        return self.table.append({
            "name": name, "agency": agency or None, "end_client": end_client or None,
            "role": role or None, "channel": channel or None,
            "last_contact": last_contact or date.today(), "status": status,
            "next_action": next_action or None, "notes": notes or None,
        })

    def touch(self, name: str, *, when: date | None = None, status: str | None = None,
              next_action: str | None = None, note: str | None = None) -> Contact:
        """Record a contact made today (or `when`), optionally updating status/next action."""
        contact = self.find(name)
        values: dict[str, Any] = {"last_contact": when or date.today()}
        if status:
            values["status"] = status
        if next_action is not None:
            values["next_action"] = next_action
        if note:
            stamp = (when or date.today()).isoformat()
            values["notes"] = f"{contact.notes}\n{stamp}: {note}".strip()
        self.table.update(contact.row, values)
        return self.find(name)

    def stale(self, days: int, today: date | None = None) -> list[tuple[Contact, int | None]]:
        """Active contacts with no contact for more than `days` days (or never recorded)."""
        today = today or date.today()
        result = []
        for c in self.all():
            if not c.active:
                continue
            since = c.days_since_contact(today)
            if since is None or since > days:
                result.append((c, since))
        return sorted(result, key=lambda pair: -(pair[1] if pair[1] is not None else 10**6))


@dataclass(frozen=True)
class FollowUpEntry:
    row: int
    date: date | None
    recipient: str
    company: str
    kind: str
    status: str
    file: str


class FollowUpLog:
    def __init__(self, path: Path) -> None:
        self.table = SheetTable(path, FOLLOWUPS_SHEET, FOLLOWUP_COLUMNS)

    def entries(self) -> list[FollowUpEntry]:
        return [
            FollowUpEntry(
                row=r["_row"], date=as_date(r["date"]), recipient=str(r["recipient"] or ""),
                company=str(r["company"] or ""), kind=str(r["kind"] or ""),
                status=str(r["status"] or ""), file=str(r["file"] or ""),
            )
            for r in self.table.rows()
        ]

    def recent(self, recipient: str = "", company: str = "", cooldown_days: int = 7,
               today: date | None = None) -> list[FollowUpEntry]:
        """Drafted or sent follow-ups to the same recipient or company within the cooldown.

        Pending drafts block regardless of age: send or discard them first.
        """
        today = today or date.today()
        r_key, c_key = norm(recipient), norm(company)
        result = []
        for e in self.entries():
            status = norm(e.status)
            if status not in {norm(FOLLOWUP_DRAFT), norm(FOLLOWUP_SENT)}:
                continue
            same = (r_key and norm(e.recipient) == r_key) or (c_key and norm(e.company) == c_key)
            if not same:
                continue
            within = e.date is not None and (today - e.date).days < cooldown_days
            if status == norm(FOLLOWUP_DRAFT) or within:
                result.append(e)
        return result

    def record(self, recipient: str, company: str, kind: str, file: str,
               when: date | None = None) -> int:
        return self.table.append({
            "date": when or date.today(), "recipient": recipient or None,
            "company": company or None, "kind": kind, "status": FOLLOWUP_DRAFT, "file": file,
        })

    def set_status(self, target: str, status: str, when: date | None = None) -> FollowUpEntry:
        """Mark the latest pending draft for `target` (recipient or company) as sent/discarded."""
        key = norm(target)
        drafts = [
            e for e in self.entries()
            if norm(e.status) == norm(FOLLOWUP_DRAFT) and key in {norm(e.recipient), norm(e.company)}
        ]
        if not drafts:
            raise WorkbookError(f"No pending follow-up draft for '{target}'.")
        entry = drafts[-1]
        values: dict[str, Any] = {"status": status}
        if status == FOLLOWUP_SENT:
            values["date"] = when or date.today()  # cooldown counts from the real send date
        self.table.update(entry.row, values)
        return entry
