from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from job_assistant.agents.followup_agent import FollowUpMessage, FollowUpTarget, build_messages
from job_assistant.pipeline import (
    FollowUpBlocked,
    draft_followup,
    due_applications,
    followup_language,
    target_from_application,
    target_from_contact,
)
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.tracker.contacts import (
    FOLLOWUP_DISCARDED,
    FOLLOWUP_SENT,
    ContactBook,
    FollowUpLog,
)
from job_assistant.tracker.excel import NewEntry, Tracker, TrackerError
from job_assistant.tracker.template import create_tracker_template

TODAY = date(2026, 9, 30)
PROFILE = CandidateProfile(name="Felipe", title="Senior Java Developer", years_experience=13, skills=[])


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return create_tracker_template(tmp_path / "tracker.xlsx")


# ---- contacts ------------------------------------------------------------------


def test_add_find_and_touch(path: Path) -> None:
    book = ContactBook(path)
    book.add("Ana Gómez", agency="TEKsystems", end_client="Citibank", role="Senior Java Dev",
             last_contact=TODAY - timedelta(days=10))
    with pytest.raises(TrackerError):
        book.add("ana gomez")  # duplicate (accent/case-insensitive)
    assert book.find("ana").end_client == "Citibank"

    updated = book.touch("Ana Gómez", when=TODAY, status="Esperando respuesta",
                         next_action="Wait for client feedback", note="Sent updated CV")
    assert updated.last_contact == TODAY
    assert updated.status == "Esperando respuesta"
    assert updated.notes.endswith("2026-09-30: Sent updated CV")


def test_find_ambiguous_raises(path: Path) -> None:
    book = ContactBook(path)
    book.add("Ana Gómez")
    book.add("Ana Silva")
    with pytest.raises(TrackerError):
        book.find("ana")
    assert book.find("ana silva").name == "Ana Silva"


def test_stale_excludes_inactive_and_sorts(path: Path) -> None:
    book = ContactBook(path)
    book.add("Old", last_contact=TODAY - timedelta(days=20))
    book.add("Older", last_contact=TODAY - timedelta(days=40))
    book.add("Fresh", last_contact=TODAY - timedelta(days=3))
    book.add("Closed", status="Cerrado", last_contact=TODAY - timedelta(days=90))
    stale = book.stale(7, TODAY)
    assert [(c.name, d) for c, d in stale] == [("Older", 40), ("Old", 20)]


# ---- follow-up history / cooldown ----------------------------------------------


def test_cooldown_and_pending_drafts_block(path: Path) -> None:
    log = FollowUpLog(path)
    log.record("Ana Gómez", "Citibank", "recruiter", "f.md", when=TODAY - timedelta(days=30))
    # an old but still pending draft blocks
    assert log.recent("Ana Gómez", "", 7, TODAY)
    log.set_status("Ana Gómez", FOLLOWUP_SENT, when=TODAY - timedelta(days=3))
    # sent 3 days ago: blocks by recipient and by company within the 7-day cooldown
    assert log.recent("ana gomez", "", 7, TODAY)
    assert log.recent("", "citibank", 7, TODAY)
    # ...but not after the cooldown
    assert not log.recent("Ana Gómez", "Citibank", 7, TODAY + timedelta(days=5))


def test_discarded_does_not_block(path: Path) -> None:
    log = FollowUpLog(path)
    log.record("Bob", "Acme", "recruiter", "f.md", when=TODAY)
    log.set_status("acme", FOLLOWUP_DISCARDED)
    assert not log.recent("Bob", "Acme", 7, TODAY)
    with pytest.raises(TrackerError):
        log.set_status("Bob", FOLLOWUP_SENT)  # no pending draft left


# ---- due applications ------------------------------------------------------------


def test_due_applications(path: Path) -> None:
    tracker = Tracker(path)
    tracker.add(NewEntry(company="Acme", role="Dev", status="Aplicado", date=TODAY - timedelta(days=12)))
    tracker.add(NewEntry(company="Beta", role="Dev", status="Aplicado", date=TODAY - timedelta(days=2)))
    tracker.add(NewEntry(company="Gamma", role="Dev", status="Rechazado", date=TODAY - timedelta(days=30)))
    tracker.add(NewEntry(company="Delta", role="Dev", status="En proceso", date=TODAY - timedelta(days=20)))
    due = due_applications(tracker, 7, TODAY)
    assert [(r.company, d) for r, d in due] == [("Delta", 20), ("Acme", 12)]


# ---- targets, language, drafting ---------------------------------------------------


def test_targets_and_language(path: Path) -> None:
    book = ContactBook(path)
    book.add("Julie", agency="Randstad", end_client="Desjardins", role="Développeur Java",
             next_action="Envoyer références", last_contact=TODAY - timedelta(days=9))
    target = target_from_contact(book.find("Julie"), TODAY)
    assert target.days_since == 9 and target.company == "Desjardins"
    assert "Pending next action: Envoyer références" in target.context
    assert followup_language(target) == "fr"  # Quebec employer

    tracker = Tracker(path)
    tracker.add(NewEntry(company="Shopify", role="Dev", status="Aplicado", location="Toronto, ON",
                         date=TODAY - timedelta(days=10)))
    app_target = target_from_application(tracker.rows()[0], TODAY)
    assert app_target.recipient == "" and app_target.days_since == 10
    assert followup_language(app_target, "Toronto, ON") == "en"


def test_prompt_has_no_invented_fields() -> None:
    target = FollowUpTarget("application", "", "Acme", "Dev", None, "")
    _, human = build_messages(target, PROFILE, "fr")
    assert "Recipient: (unknown)" in human.content
    assert "Time since last contact: unknown" in human.content


class FakeAgent:
    def __init__(self) -> None:
        self.calls = 0

    def draft(self, *args: Any) -> FollowUpMessage:
        self.calls += 1
        return FollowUpMessage(subject="Following up", body="Hi Ana, ...")


def test_draft_followup_writes_logs_and_blocks_second(path: Path, tmp_path: Path) -> None:
    book, log, agent = ContactBook(path), FollowUpLog(path), FakeAgent()
    book.add("Ana Gómez", end_client="Citibank", last_contact=TODAY - timedelta(days=10))
    target = target_from_contact(book.find("Ana Gómez"), TODAY)

    folder = draft_followup(target, PROFILE, agent, log, tmp_path / "out", 7, "en")  # type: ignore[arg-type]
    text = (folder / "message.md").read_text(encoding="utf-8")
    assert "PENDING YOUR APPROVAL" in text and "**Subject:** Following up" in text
    [entry] = log.entries()
    assert entry.status == "Borrador" and entry.recipient == "Ana Gómez"

    with pytest.raises(FollowUpBlocked):
        draft_followup(target, PROFILE, agent, log, tmp_path / "out", 7, "en")  # type: ignore[arg-type]
    assert agent.calls == 1  # the LLM is not called when blocked


def test_template_has_three_sheets(path: Path) -> None:
    from openpyxl import load_workbook

    assert load_workbook(path).sheetnames == ["Aplicaciones", "Reclutadores", "Seguimientos"]
    assert [l.name for l in Tracker(path).layouts()] == ["Aplicaciones"]


def test_followup_sheet_created_on_old_workbook(tmp_path: Path) -> None:
    from openpyxl import Workbook, load_workbook

    old = tmp_path / "old.xlsx"
    wb = Workbook()
    wb.active.append(["Empresa", "Puesto", "Estado"])
    wb.save(old)
    FollowUpLog(old).record("X", "Y", "recruiter", "f.md", when=TODAY)
    assert "Seguimientos" in load_workbook(old).sheetnames
    assert FollowUpLog(old).entries()[0].recipient == "X"
