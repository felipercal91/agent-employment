from datetime import date
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font

from job_assistant.agents.requirements_agent import Requirement, RequirementsAssessment
from job_assistant.pipeline import PostingEvaluation, add_to_tracker
from job_assistant.scoring.fit import FitResult
from job_assistant.tracker.excel import (
    NewEntry,
    Tracker,
    TrackerError,
    TrackerLockedError,
    normalize_company,
    normalize_link,
)


def make_tracker(path: Path) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Toronto"
    ws["A1"] = "Job search 2026"  # title row above the header
    ws.append(["Empresa", "Puesto", "Link", "Fit %", "Estado", "Fecha", "Notas"])
    ws.append(["Altimetrik", "Senior Java Developer", "https://www.linkedin.com/jobs/view/111/?trk=abc",
               0.8, "Entrevista", date(2026, 8, 1), "Citibank client"])
    ws.append(["Shopify Inc.", "Backend Engineer", "https://jobs.example.com/shopify/42", 0.65,
               "Aplicado", date(2026, 8, 5), None])
    for row in ws.iter_rows(min_row=3, max_row=4):
        row[3].number_format = "0%"
        row[5].number_format = "dd/mm/yyyy"
        row[0].font = Font(bold=True)
    other = wb.create_sheet("Montreal")
    other.append(["Company", "Role", "URL", "Status"])
    other.append(["MM Global", "Développeur Java", "https://mmglobal.example/careers/7", "Rechazado"])
    wb.create_sheet("Resumen")["A1"] = "Totals"  # no recognizable header -> ignored
    wb.save(path)
    return path


@pytest.fixture
def tracker(tmp_path: Path) -> Tracker:
    return Tracker(make_tracker(tmp_path / "tracker.xlsx"))


def test_normalizers() -> None:
    assert normalize_company("Shopify Inc.") == normalize_company("shopify")
    assert normalize_company("Ernst & Young LLC") == "ernst and young"
    assert normalize_link("https://www.linkedin.com/jobs/view/111/?trk=abc") == "linkedin.com/jobs/view/111"
    assert normalize_link("ca.indeed.com/viewjob?jk=ABC&from=x") == "ca.indeed.com/viewjob?jk=abc"
    assert normalize_link("") == ""


def test_reads_all_sheets_with_headers(tracker: Tracker) -> None:
    rows = tracker.rows()
    assert [(r.sheet, r.company) for r in rows] == [
        ("Toronto", "Altimetrik"),
        ("Toronto", "Shopify Inc."),
        ("Montreal", "MM Global"),
    ]
    assert [l.name for l in tracker.layouts()] == ["Toronto", "Montreal"]
    assert tracker.layouts()[0].header_row == 2


def test_duplicate_by_link_ignores_tracking_params(tracker: Tracker) -> None:
    check = tracker.find_duplicates("Other Co", link="linkedin.com/jobs/view/111")
    assert check.is_duplicate and check.duplicates[0].company == "Altimetrik"


def test_duplicate_by_company_and_role(tracker: Tracker) -> None:
    assert tracker.find_duplicates("SHOPIFY", "backend engineer").is_duplicate
    other_role = tracker.find_duplicates("Shopify", "Staff Engineer")
    assert not other_role.is_duplicate
    assert [r.role for r in other_role.same_company] == ["Backend Engineer"]


def test_duplicate_in_second_sheet(tracker: Tracker) -> None:
    assert tracker.find_duplicates("mm global", "Développeur Java").is_duplicate


def test_add_appends_with_sheet_conventions(tracker: Tracker) -> None:
    row = tracker.add(NewEntry(company="Wealthsimple", role="Senior Java Developer",
                               link="https://example.com/ws/1", fit=72, notes="n",
                               date=date(2026, 9, 30)))
    assert (row.sheet, row.row) == ("Toronto", 5)

    ws = load_workbook(tracker.path)["Toronto"]
    assert [c.value for c in ws[5]][:5] == ["Wealthsimple", "Senior Java Developer",
                                           "https://example.com/ws/1", 0.72, "Pending review"]
    assert ws["D5"].number_format == "0%"          # fraction + percent format, like the rows above
    assert ws["F5"].number_format == "dd/mm/yyyy"  # date format copied
    assert ws["A5"].font.bold                      # style copied
    assert ws["C5"].hyperlink.target == "https://example.com/ws/1"
    assert load_workbook(tracker.path)["Montreal"].max_row == 2  # other sheets untouched


def test_add_creates_backup(tracker: Tracker) -> None:
    tracker.add(NewEntry(company="A", role="B"))
    backups = list((tracker.path.parent / "backups").glob("tracker-*.xlsx"))
    assert len(backups) == 1
    assert len(Tracker(backups[0]).rows()) == 3  # backup is the pre-write state


def test_add_to_named_sheet(tmp_path: Path) -> None:
    t = Tracker(make_tracker(tmp_path / "t.xlsx"), target_sheet="Montreal")
    row = t.add(NewEntry(company="Desjardins", role="Développeur Java", fit=90))
    assert (row.sheet, row.row) == ("Montreal", 3)
    with pytest.raises(TrackerError):
        Tracker(t.path, target_sheet="Resumen").add(NewEntry(company="X", role="Y"))


def test_add_refuses_duplicate(tracker: Tracker) -> None:
    with pytest.raises(TrackerError):
        tracker.add(NewEntry(company="Altimetrik", role="Senior Java Developer"))
    assert len(tracker.rows()) == 3


def test_add_refuses_when_open_in_excel(tracker: Tracker, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("job_assistant.tracker.workbook._excel_running", lambda: True)
    (tracker.path.parent / f"~${tracker.path.name}").write_bytes(b"")
    with pytest.raises(TrackerLockedError):
        tracker.add(NewEntry(company="A", role="B"))


def test_stale_lock_is_ignored_when_excel_is_not_running(tracker: Tracker, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("job_assistant.tracker.workbook._excel_running", lambda: False)
    (tracker.path.parent / f"~${tracker.path.name}").write_bytes(b"")
    tracker.add(NewEntry(company="A", role="B"))
    assert any(r.company == "A" for r in tracker.rows())


def _evaluation(decision_not_met: bool) -> PostingEvaluation:
    status = "not_met" if decision_not_met else "unclear"
    return PostingEvaluation(
        fit=FitResult(score=70, found=["Java", "AWS"], matched=["Java"], missing=["AWS"],
                      years_required=None, years_ok=True),
        requirements=RequirementsAssessment(
            company="Acme Corp", role_title="Senior Java Developer", location="Toronto",
            requirements=[Requirement(requirement="Kubernetes", kind="must_have",
                                      status=status, evidence="e")],  # type: ignore[arg-type]
        ),
    )


def test_pipeline_adds_passing_posting_with_notes(tracker: Tracker) -> None:
    row = add_to_tracker(tracker, _evaluation(False), status="Por aplicar")
    assert row.company == "Acme Corp" and row.status == "Por aplicar"
    assert row.notes == "Missing keywords: AWS | Verify must-haves: Kubernetes"


def test_pipeline_refuses_stopped_posting(tracker: Tracker) -> None:
    with pytest.raises(TrackerError):
        add_to_tracker(tracker, _evaluation(True), status="x")
    assert len(tracker.rows()) == 3


def test_template_round_trip(tmp_path: Path) -> None:
    from job_assistant.tracker.template import create_tracker_template

    path = create_tracker_template(tmp_path / "data" / "tracker.xlsx")
    with pytest.raises(FileExistsError):
        create_tracker_template(path)

    t = Tracker(path)
    assert [l.name for l in t.layouts()] == ["Aplicaciones"]  # contacts sheet is not an applications sheet
    assert set(t.layouts()[0].columns) == {
        "company", "role", "end_client", "location", "link", "fit", "status", "date", "source", "notes"
    }
    row = t.add(NewEntry(company="Desjardins", role="Développeur Java", location="Montréal, QC",
                         fit=81, status="Por aplicar"))
    assert row.row == 2
    ws = load_workbook(path)["Aplicaciones"]
    assert ws["F2"].value == 0.81 and ws["F2"].number_format == "0%"
    assert ws["D2"].value == "Montréal, QC"
    assert ws.auto_filter.ref == "A1:J2"
    assert t.rows()[0].location == "Montréal, QC"
