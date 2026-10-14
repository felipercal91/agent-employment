from datetime import date
from pathlib import Path

from openpyxl import load_workbook

from job_assistant.cv import postular
from job_assistant.discovery import daily
from job_assistant.discovery.discover import DiscoveryConfig
from job_assistant.discovery.sources import Board, Posting
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.tracker.excel import NewEntry, Tracker
from job_assistant.tracker.template import create_tracker_template

TODAY = date(2026, 10, 14)
PROFILE = CandidateProfile(name="", title="Dev", years_experience=13,
                           skills=["Java", "Spring Boot", "Kafka", "PostgreSQL", "Microservices"])
TEXT = "We need Java, Spring Boot, Kafka and PostgreSQL for microservices."
CONFIG = DiscoveryConfig(boards=[Board("Acme", "greenhouse", "acme")], min_fit=50, max_age_days=21, daily_max_new=2)


def post(company: str, title: str = "Senior Java Developer", posted: str = "2026-10-10", url: str = "",
         source: str = "greenhouse") -> Posting:
    return Posting(source, company, title, "Toronto, ON", url or f"https://x/{company}", TEXT, posted)


def make_workbook(tmp_path: Path) -> Path:
    path = create_tracker_template(tmp_path / "t.xlsx")
    postular.ensure_sheet(path)
    return path


def search(postings: list[Posting], path: Path, config: DiscoveryConfig = CONFIG) -> daily.DailySearch:
    known = daily.KnownPostings(Tracker(path), path)
    return daily.search_new(PROFILE, config, known, path.parent / "inbox", TODAY,
                            fetch=lambda boards: (postings, ["Beta: HTTP 404"]), read=lambda d: [])


def test_freshness_old_skipped_unknown_date_kept() -> None:
    assert daily.is_fresh(post("A", posted="2026-10-01"), 21, TODAY)
    assert not daily.is_fresh(post("A", posted="2026-08-01"), 21, TODAY)
    assert daily.is_fresh(post("A", posted=""), 21, TODAY)
    assert daily.is_fresh(post("A", posted="2020-01-01", source="inbox"), 21, TODAY)


def test_search_skips_old_tracked_and_postular_rows_and_applies_cap(tmp_path: Path) -> None:
    path = make_workbook(tmp_path)
    Tracker(path).add(NewEntry(company="Applied", role="Senior Java Developer"))
    postular.table(path).append({"link": "https://x/InPostular", "status": "Nuevo"})
    postings = [
        post("Applied"), post("InPostular"), post("Old", posted="2026-07-01"),
        post("New1"), post("New2"), post("New3"),
    ]
    result = search(postings, path)
    assert result.stale == 1
    assert {p.company for p in result.discovery.already_tracked} == {"Applied", "InPostular"}
    assert len(result.to_add) == 2 and result.over_cap == 1
    assert result.errors == ["Beta: HTTP 404"]


def test_add_to_postular_fills_columns_and_never_repeats(tmp_path: Path) -> None:
    path = make_workbook(tmp_path)
    first = search([post("New1"), post("New2")], path)
    rows = daily.add_to_postular(path, first.to_add)
    assert len(rows) == 2
    saved = postular.table(path).rows()
    assert {r["company"] for r in saved} == {"New1", "New2"}
    row = next(r for r in saved if r["company"] == "New1")
    assert row["status"] == "Nuevo" and row["source"] == "greenhouse"
    assert row["description"] == TEXT and row["fit"] >= 50
    assert load_workbook(path)["Postular"].max_row == 3
    # the next day the same postings are already in the sheet
    again = search([post("New1"), post("New2")], path)
    assert again.to_add == [] and len(again.discovery.already_tracked) == 2


def test_long_description_is_cut_for_the_excel_cell_limit(tmp_path: Path) -> None:
    path = make_workbook(tmp_path)
    big = Posting("greenhouse", "Big", "Java Developer", "Toronto", "https://x/big", "Java " * 20000, "2026-10-10")
    result = search([big], path)
    daily.add_to_postular(path, result.to_add)
    assert 29900 < len(postular.table(path).rows()[0]["description"]) <= daily.MAX_DESCRIPTION_CHARS


def test_added_rows_are_picked_up_by_the_cv_step(tmp_path: Path) -> None:
    path = make_workbook(tmp_path)
    daily.add_to_postular(path, search([post("New1")], path).to_add)
    assert [r["company"] for r in postular.pending_rows(path)] == ["New1"]


def test_inbox_files_archived_only_when_handled(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    files = {name: inbox / f"{name}.txt" for name in ("added", "tracked", "held")}
    for f in files.values():
        f.write_text("x", encoding="utf-8")
    def inbox_post(name: str) -> Posting:
        return Posting("inbox", name, "Java Dev", "", f"https://x/{name}", TEXT, "", files[name])
    path = make_workbook(tmp_path)
    postular.table(path).append({"link": "https://x/tracked", "status": "Nuevo"})
    known = daily.KnownPostings(Tracker(path), path)
    config = DiscoveryConfig(daily_max_new=1)
    result = daily.search_new(PROFILE, config, known, inbox, TODAY, fetch=lambda b: ([], []),
                              read=lambda d: [inbox_post("added"), inbox_post("tracked"), inbox_post("held")])
    assert len(result.to_add) == 1 and result.over_cap == 1
    daily.archive_handled_inbox(result)
    remaining = sorted(p.name for p in inbox.glob("*.txt"))
    assert len(remaining) == 1 and remaining[0] in {"added.txt", "held.txt"}  # the one cut by the cap stays
    assert "tracked.txt" not in remaining


def test_report_lists_cvs_stopped_errors_and_next_steps(tmp_path: Path) -> None:
    path = make_workbook(tmp_path)
    s = search([post("New1")], path)
    results = [
        {"status": postular.STATUS_READY, "hr_score": 8.5, "company": "Good", "role": "Dev", "cv_file": "good.pdf"},
        {"status": postular.STATUS_MISMATCH, "company": "Stop", "role": "Dev", "hr_notes": "Needs 10y Rust"},
        {"status": postular.STATUS_NO_JD, "company": "NoJD", "link": "https://nojd", "hr_notes": ""},
    ]
    text = daily.render_daily_report(daily.DailyReport(
        today=TODAY, search=s, added=1, results=results, usage="$0.01", recruiters_due=["Ana (Acme): 12 días"],
    ))
    for expected in ("2026-10-14", "good.pdf", "Needs 10y Rust", "NoJD", "Beta: HTTP 404", "Aprobado = Sí", "Ana (Acme)"):
        assert expected in text


def test_report_paths_do_not_overwrite(tmp_path: Path) -> None:
    first = daily.write_report(tmp_path, TODAY, "a")
    second = daily.write_report(tmp_path, TODAY, "b")
    assert first.name == "Reporte 10-14.md" and second.name == "Reporte 10-14 2.md"
    assert first.read_text(encoding="utf-8") == "a"


def test_todays_results_only_returns_rows_processed_today(tmp_path: Path) -> None:
    path = make_workbook(tmp_path)
    postular.table(path).append({"link": "https://a", "processed": TODAY})
    postular.table(path).append({"link": "https://b", "processed": date(2026, 10, 1)})
    assert [r["link"] for r in daily.todays_results(path, TODAY)] == ["https://a"]
