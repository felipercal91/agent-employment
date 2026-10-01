from datetime import date
from pathlib import Path

import httpx
import pytest

from job_assistant.discovery.discover import (
    DiscoveryConfig,
    discover,
    is_relevant,
    load_config,
    render_report,
    save_posting,
)
from job_assistant.discovery.sources import (
    Board,
    Posting,
    archive_inbox_file,
    fetch_boards,
    html_to_text,
    parse_ashby,
    parse_greenhouse,
    parse_inbox_file,
    parse_lever,
    read_inbox,
)
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.tracker.excel import NewEntry, Tracker
from job_assistant.tracker.template import create_tracker_template

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
PROFILE = CandidateProfile(name="", title="Dev", years_experience=13,
                           skills=["Java", "Spring Boot", "Kafka", "PostgreSQL", "Microservices"])
JAVA_TEXT = "We need Java, Spring Boot, Kafka and PostgreSQL for microservices."

GREENHOUSE = {"jobs": [{
    "title": "Senior Java Developer", "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1",
    "location": {"name": "Toronto, Ontario"}, "updated_at": "2026-09-20T10:00:00-04:00",
    "content": "&lt;p&gt;We need &lt;strong&gt;Java&lt;/strong&gt;&lt;/p&gt;&lt;ul&gt;&lt;li&gt;Kafka&lt;/li&gt;&lt;/ul&gt;",
}]}
LEVER = [{
    "text": "Backend Developer", "hostedUrl": "https://jobs.lever.co/acme/abc",
    "categories": {"location": "Montreal, QC"}, "workplaceType": "hybrid",
    "descriptionPlain": "Intro.", "lists": [{"text": "Requirements", "content": "<li>Java 17</li>"}],
    "additionalPlain": "", "createdAt": 1758000000000,
}]
ASHBY = {"jobs": [{
    "title": "Software Developer, Payments", "location": "Toronto", "isRemote": True,
    "secondaryLocations": [{"location": "Montreal"}], "jobUrl": "https://jobs.ashbyhq.com/acme/x",
    "descriptionPlain": "Java and Spring Boot", "publishedAt": "2026-09-10T00:00:00Z",
}]}


def test_html_to_text() -> None:
    assert html_to_text("&lt;p&gt;A&lt;/p&gt;&lt;ul&gt;&lt;li&gt;B&lt;/li&gt;&lt;/ul&gt;") == "A\n- B"


def test_parse_greenhouse() -> None:
    [p] = parse_greenhouse("Acme", GREENHOUSE)
    assert (p.title, p.location, p.posted_at) == ("Senior Java Developer", "Toronto, Ontario", "2026-09-20")
    assert "Java" in p.text and "<" not in p.text


def test_parse_lever() -> None:
    [p] = parse_lever("Acme", LEVER)
    assert p.title == "Backend Developer" and p.location == "Montreal, QC"
    assert "Requirements" in p.text and "Java 17" in p.text and p.posted_at.startswith("2025")


def test_parse_ashby() -> None:
    [p] = parse_ashby("Acme", ASHBY)
    assert p.location == "Toronto, Montreal (Remote)"
    assert p.posted_at == "2026-09-10"


def test_fetch_boards_with_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "greenhouse" in request.url.host:
            return httpx.Response(200, json=GREENHOUSE)
        if "lever" in request.url.host:
            return httpx.Response(404)
        return httpx.Response(200, json=ASHBY)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    boards = [Board("A", "greenhouse", "a"), Board("B", "lever", "b"), Board("C", "ashby", "c"),
              Board("D", "workday", "d")]
    postings, errors = fetch_boards(boards, client=client, delay=0)
    assert [p.company for p in postings] == ["A", "C"]
    assert errors[0] == "B (lever/b): HTTP 404"
    assert "Unsupported ATS" in errors[1]


def test_parse_inbox_file_with_headers(tmp_path: Path) -> None:
    f = tmp_path / "job.txt"
    f.write_text("Company: Acme\nTitle: Senior Java Dev\nLocation: Toronto\n\nWe need Java.\n"
                 "Apply at https://acme.example/jobs/9", encoding="utf-8")
    p = parse_inbox_file(f)
    assert (p.company, p.title, p.location) == ("Acme", "Senior Java Dev", "Toronto")
    assert p.url == "https://acme.example/jobs/9" and p.text.startswith("We need Java.")


def test_parse_inbox_file_plain(tmp_path: Path) -> None:
    f = tmp_path / "copied.txt"
    f.write_text("Backend Developer - Montreal\n\nJava and Kafka", encoding="utf-8")
    p = parse_inbox_file(f)
    assert p.title == "Backend Developer - Montreal" and p.company == "" and p.url == ""


def test_read_and_archive_inbox(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Java role", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"x")
    [p] = read_inbox(tmp_path)
    moved = archive_inbox_file(p.inbox_file)  # type: ignore[arg-type]
    assert moved.parent.name == "processed" and not (tmp_path / "a.txt").exists()
    assert read_inbox(tmp_path / "missing") == []


def board_posting(title: str, location: str = "Toronto, ON", company: str = "Acme",
                  url: str = "", text: str = JAVA_TEXT) -> Posting:
    return Posting("greenhouse", company, title, location, url or f"https://x/{title}", text)


@pytest.mark.parametrize(
    ("title", "location", "relevant"),
    [
        ("Senior Java Developer", "Toronto, ON", True),
        ("Développeur backend", "Montréal, QC", True),
        ("Backend Engineer", "Remote (Canada)", True),
        ("Junior Java Developer", "Toronto", False),
        ("Frontend Developer", "Toronto", False),
        ("Senior Java Developer", "Remote - Indonesia", False),
        ("Account Executive", "Toronto", False),
    ],
)
def test_is_relevant(title: str, location: str, relevant: bool) -> None:
    assert is_relevant(board_posting(title, location), DiscoveryConfig()) is relevant


def test_board_posting_without_core_keyword_is_filtered() -> None:
    python_role = board_posting("Software Engineer, Integrations", "Toronto", text="Python, LLM APIs")
    no_keywords = board_posting("Software Engineer", "Toronto", text="Great team, build things")
    assert not is_relevant(python_role, DiscoveryConfig())
    assert not is_relevant(no_keywords, DiscoveryConfig())
    assert is_relevant(no_keywords, DiscoveryConfig(require_any_keyword=[]))


def test_inbox_postings_skip_filters() -> None:
    p = Posting("inbox", "", "Whatever title", "", "", "text")
    assert is_relevant(p, DiscoveryConfig())


def test_discover_dedupes_and_ranks(tmp_path: Path) -> None:
    tracker = Tracker(create_tracker_template(tmp_path / "t.xlsx"))
    tracker.add(NewEntry(company="Acme Inc.", role="Senior Java Developer"))
    postings = [
        board_posting("Senior Java Developer"),                                    # tracked (company+role)
        board_posting("Backend Developer", company="Beta", url="https://b/1"),
        board_posting("Backend Developer", company="Beta", url="https://b/1"),      # duplicate in run
        board_posting("Java Developer", company="Gamma", url="https://g/1", text="Java, AWS, Kubernetes, Terraform, Go"),
        board_posting("Java Developer", company="Delta", url="https://d/1", text="Java Spring Boot only"),
        board_posting("Frontend Developer"),                                       # filtered
    ]
    result = discover(postings, PROFILE, DiscoveryConfig(min_fit=50), tracker)
    assert [p.company for p in result.already_tracked] == ["Acme"]
    assert [c.posting.company for c in result.candidates] == ["Beta", "Delta"]
    assert [c.posting.company for c in result.below_min_fit] == ["Gamma"]
    assert result.filtered_out == 1


def test_load_example_config(tmp_path: Path) -> None:
    (tmp_path / "discovery.toml").write_bytes((EXAMPLES / "discovery.example.toml").read_bytes())
    config = load_config(tmp_path)
    assert Board("Wealthsimple", "ashby", "wealthsimple") in config.boards
    assert config.min_fit == 50
    assert load_config(tmp_path / "none").boards == []


def test_save_posting_and_report(tmp_path: Path) -> None:
    p = board_posting("Senior Java Developer", url="https://x/1")
    path = save_posting(p, tmp_path)
    assert path.read_text(encoding="utf-8").startswith("Company: Acme\nTitle: Senior Java Developer")
    result = discover([p], PROFILE, DiscoveryConfig(), None, errors=["X: HTTP 404"])
    report = render_report(result, date(2026, 9, 30), {"https://x/1": path})
    assert "New relevant postings: **1**" in report and "X: HTTP 404" in report
    assert "Senior Java Developer @ Acme" in report
