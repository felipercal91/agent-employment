from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from openpyxl import load_workbook

from job_assistant.agents.hr_reviewer import HRReview, ReviewIssue, Scores, passes
from job_assistant.agents.requirements_agent import Requirement, RequirementsAssessment
from job_assistant.cv import postular
from job_assistant.cv.checks import (
    Finding,
    ai_phrases,
    missing_jd_keywords,
    repeated_openings,
    run_checks,
    unsupported_numbers,
)
from job_assistant.cv.content import Bullet, JobBullets, ResumeContent, normalize
from job_assistant.cv.jd import JobDescription, JobDescriptionError, fetch_job_description
from job_assistant.cv.master import MasterNotFoundError, load_master
from job_assistant.cv.pipeline import generate_resume, render_review_report
from job_assistant.cv.render import build_html
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.tracker.template import create_tracker_template

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
JD = ("Senior Java Developer. We need Java, Spring Boot, Apache Kafka, PostgreSQL and "
      "Docker for our payments microservices. Kubernetes is a plus. " * 3)
TODAY = date(2026, 10, 1)


@pytest.fixture
def master_dir(tmp_path: Path) -> Path:
    (tmp_path / "cv_master.toml").write_bytes((EXAMPLES / "cv_master.example.toml").read_bytes())
    return tmp_path


@pytest.fixture
def master(master_dir: Path) -> Any:
    return load_master(master_dir)


def content(**kw: Any) -> ResumeContent:
    base = dict(
        title="Senior Java Developer",
        summary="Java developer with 12 years of experience building Spring Boot microservices and Kafka integrations.",
        competencies=["Java", "Spring Boot", "Apache Kafka"],
        experience=[
            JobBullets(job_id="bank", bullets=[
                Bullet(lead="Built Spring Boot microservices", rest="for payments, handling 2 million transactions per day."),
                Bullet(lead="Implemented Kafka event streaming", rest="between services, cutting reconciliation time by 40%."),
            ]),
            JobBullets(job_id="insurer", bullets=[
                Bullet(lead="Migrated Oracle PL/SQL batch jobs", rest="to Java services, reducing run time by 25%."),
            ]),
        ],
        tools=["Java", "Spring Boot", "Apache Kafka", "PostgreSQL", "Docker"],
        methodologies=["Agile", "Microservices Architecture"],
    )
    return ResumeContent(**{**base, **kw})


# ---- master -------------------------------------------------------------------------


def test_load_master(master: Any) -> None:
    assert [j.id for j in master.jobs("en")] == ["bank", "insurer"]
    assert master.jobs("fr")[1].location == "Santiago, Chili"
    assert master.jobs("en")[1].location == "Santiago, Chile"
    assert "2 million" in master.source_text("en")


def test_missing_master(tmp_path: Path) -> None:
    with pytest.raises(MasterNotFoundError):
        load_master(tmp_path)


def test_real_master_shape_if_present() -> None:
    real = Path(__file__).resolve().parent.parent / "data"
    if not (real / "cv_master.toml").is_file():
        pytest.skip("no personal master file")
    m = load_master(real)
    for lang in ("en", "fr"):
        assert all(j.bullets and j.dates and j.title for j in m.jobs(lang))  # type: ignore[arg-type]


# ---- normalize ----------------------------------------------------------------------


def test_normalize_orders_fills_and_filters(master: Any) -> None:
    c = content(
        experience=[JobBullets(job_id="insurer", bullets=[Bullet(lead="Did", rest="x")]),
                    JobBullets(job_id="ghost", bullets=[Bullet(lead="Fake", rest="job")])],
        tools=["docker", "Kubernetes", "Java"],
        methodologies=["Scrum"],
    )
    fixed, problems = normalize(c, master, "en")
    assert [j.job_id for j in fixed.experience] == ["bank", "insurer"]
    assert fixed.experience[0].bullets[0].rest.startswith("Built Spring Boot")  # original used
    assert fixed.tools == ["Docker", "Java"]
    assert any("Kubernetes" in p for p in problems) and any("ghost" in p for p in problems)


# ---- deterministic checks -----------------------------------------------------------


def test_unsupported_numbers(master: Any) -> None:
    source = master.source_text("en")
    assert unsupported_numbers(content(), source) == []
    bad = content(summary="Java developer with 15 years who cut costs by 10%.")
    assert unsupported_numbers(bad, source) == ["10", "15"]


def test_ai_phrases_and_repeats() -> None:
    c = content(summary="Results-driven developer who leveraged Kafka.")
    assert set(ai_phrases(c, "en")) == {"results-driven", "leveraged"}
    same = content(experience=[JobBullets(job_id="bank", bullets=[
        Bullet(lead="Built A", rest="."), Bullet(lead="Built B", rest="."), Bullet(lead="Built C", rest=".")])])
    assert repeated_openings(same) == ["built"]
    assert set(ai_phrases(content(summary="Développeur passionné et dynamique."), "fr")) == {"passionné", "dynamique"}


def test_missing_jd_keywords(master: Any) -> None:
    c = content(tools=["Java"], competencies=["Java"], summary="Java developer.",
                experience=[JobBullets(job_id="bank", bullets=[Bullet(lead="Wrote", rest="Java code.")])])
    addable, gaps = missing_jd_keywords(c, JD, master.source_text("en"))
    assert "Apache Kafka" not in addable  # canonical names come from the vocabulary
    assert "Kafka" in addable and "PostgreSQL" in addable
    assert gaps == ["Kubernetes"]


def test_run_checks_blockers(master: Any) -> None:
    source = master.source_text("fr")
    c = content(title="Ingénieur Java sénior", summary="Ingénieur avec Kubernetes et 99 projets.")
    severities = {f.problem.split(":")[0]: f.severity for f in run_checks(c, "fr", source, JD, pages=3)}
    assert severities["Figures not found in your facts"] == "blocker"
    assert severities["Technologies not in your facts"] == "blocker"
    assert any(k.startswith("Uses 'Ingénieur'") for k in severities)
    assert severities["The PDF runs to 3 pages."] == "major"


def test_run_checks_clean(master: Any) -> None:
    assert [f for f in run_checks(content(), "en", master.source_text("en"), JD, pages=2)
            if f.severity == "blocker"] == []


def test_certified_scrum_master_flag(master: Any) -> None:
    c = content(competencies=["Certified Scrum Master (SMPC®)"])
    assert any("CSM" in f.problem for f in run_checks(c, "en", master.source_text("en"), "", 2))


# ---- render -------------------------------------------------------------------------


def test_build_html_uses_fixed_facts_and_escapes(master: Any) -> None:
    html = build_html(master, content(summary="Uses <script> & stuff"), "fr")
    assert "ALEX SAMPLE" in html and "Janvier 2020 – Juin 2026" in html
    assert "Sample Insurance | Santiago, Chili" in html
    assert "SOMMAIRE PROFESSIONNEL" in html and "&lt;script&gt; &amp; stuff" in html
    assert "<b>Built Spring Boot microservices</b>" in html


# ---- job description fetching -------------------------------------------------------


def client_for(routes: dict[str, httpx.Response]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        for key, resp in routes.items():
            if key in str(request.url):
                return resp
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


LONG = "Build Java services. " * 40


def test_fetch_greenhouse_lever_ashby() -> None:
    c = client_for({
        "boards-api.greenhouse.io/v1/boards/acme/jobs/123": httpx.Response(
            200, json={"title": "Java Dev", "content": f"&lt;p&gt;{LONG}&lt;/p&gt;", "location": {"name": "Toronto"}}),
        "api.lever.co/v0/postings/acme/": httpx.Response(
            200, json={"text": "Backend Dev", "descriptionPlain": LONG, "lists": [], "categories": {"location": "Montreal"}}),
        "api.ashbyhq.com/posting-api/job-board/acme": httpx.Response(
            200, json={"jobs": [{"title": "Senior Dev", "jobUrl": "https://jobs.ashbyhq.com/acme/11111111-2222-3333-4444-555555555555",
                                 "descriptionPlain": LONG, "location": "Remote"}]}),
    })
    gh = fetch_job_description("https://job-boards.greenhouse.io/acme/jobs/123", c)
    assert gh.title == "Java Dev" and gh.location == "Toronto" and "<" not in gh.text
    lv = fetch_job_description("https://jobs.lever.co/acme/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", c)
    assert lv.title == "Backend Dev" and lv.source == "lever api"
    ab = fetch_job_description("https://jobs.ashbyhq.com/acme/11111111-2222-3333-4444-555555555555", c)
    assert ab.title == "Senior Dev"


def test_fetch_json_ld_and_page_text() -> None:
    ld = ('<html><script type="application/ld+json">{"@type":"JobPosting","title":"Dev Java",'
          '"description":"<p>' + LONG + '</p>","hiringOrganization":{"name":"Desjardins"},'
          '"jobLocation":{"address":{"addressLocality":"Montréal","addressRegion":"QC"}}}</script></html>')
    c = client_for({"careers.example/a": httpx.Response(200, text=ld),
                    "careers.example/b": httpx.Response(200, text=f"<body><nav>menu</nav><p>{LONG}</p></body>"),
                    "careers.example/c": httpx.Response(200, text="<p>tiny</p>")})
    jd = fetch_job_description("https://careers.example/a", c)
    assert (jd.company, jd.location) == ("Desjardins", "Montréal, QC")
    page = fetch_job_description("https://careers.example/b", c)
    assert "menu" not in page.text and page.source == "page text"
    with pytest.raises(JobDescriptionError):
        fetch_job_description("https://careers.example/c", c)


def test_fetch_refuses_portals_and_bad_links() -> None:
    for url in ("https://www.linkedin.com/jobs/view/1", "https://ca.indeed.com/viewjob?jk=1", "not a link"):
        with pytest.raises(JobDescriptionError):
            fetch_job_description(url, client_for({}))
    with pytest.raises(JobDescriptionError, match="HTTP 404"):
        fetch_job_description("https://careers.example/missing", client_for({}))


# ---- writer <-> reviewer loop ------------------------------------------------------


def review(score: int = 9, issues: list[ReviewIssue] | None = None, fit: int = 8) -> HRReview:
    return HRReview(
        scores=Scores(requirements_coverage=score, truthfulness=score, human_tone=score,
                      clarity=score, ats_keywords=score, candidate_fit=fit),
        issues=issues or [], gaps_cannot_fix=["Kubernetes"],
        likely_interview_questions=["Tell me about Kafka."], notes_for_candidate="Buen calce.",
    )


class FakeWriter:
    def __init__(self, drafts: list[ResumeContent]) -> None:
        self.drafts, self.calls = drafts, []

    def write(self, jd: str, master: Any, lang: str, notes: str = "", previous: Any = None,
              feedback: Any = None, quebec: bool = False) -> ResumeContent:
        self.calls.append({"previous": previous, "feedback": feedback, "notes": notes, "quebec": quebec})
        return self.drafts[min(len(self.calls) - 1, len(self.drafts) - 1)]


class FakeReviewer:
    def __init__(self, reviews: list[HRReview]) -> None:
        self.reviews, self.calls = reviews, 0

    def review(self, *args: Any) -> HRReview:
        self.calls += 1
        return self.reviews[min(self.calls - 1, len(self.reviews) - 1)]


def fake_render(html: str, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return 2


def test_scores_are_clamped() -> None:
    assert Scores(requirements_coverage=14, truthfulness=-2, human_tone=7, clarity=7, ats_keywords=7,
                  candidate_fit=3).minimum() == 0


def test_passes_rule() -> None:
    major = ReviewIssue(severity="major", where="s", problem="p", fix="f")
    blocker = ReviewIssue(severity="blocker", where="s", problem="p", fix="f")
    assert passes(review(9), [])
    assert not passes(review(6), [])                      # average below 7
    assert passes(review(8, [major]), [Finding("major", "p", "f")])  # majors are listed, not blocking
    assert not passes(review(9, [blocker]), [])           # a reviewer blocker blocks
    assert not passes(review(9), [Finding("blocker", "p", "f")])     # so does a deterministic one
    low_truth = review(9)
    low_truth.scores.truthfulness = 6
    assert not passes(low_truth, [])                      # truthfulness must reach 7 on its own
    low_fit = review(9, fit=2)
    assert passes(low_fit, [])                            # candidate_fit is informational


def test_loop_revises_with_feedback_then_passes(master: Any, tmp_path: Path) -> None:
    bad = content(summary="Results-driven developer with 15 years.")
    writer = FakeWriter([bad, content()])
    reviewer = FakeReviewer([review(6, [ReviewIssue(severity="major", where="summary", problem="vague", fix="be specific")]),
                             review(9)])
    result = generate_resume(JD, master, "en", writer, reviewer, tmp_path / "cv.pdf",  # type: ignore[arg-type]
                             candidate_notes="stress Kafka", renderer=fake_render)
    assert result.passed and len(result.rounds) == 2
    second = writer.calls[1]
    assert second["previous"] is not None and second["notes"] == "stress Kafka"
    assert any("15" in line for line in second["feedback"])          # deterministic finding forwarded
    assert any("be specific" in line for line in second["feedback"])  # reviewer issue forwarded
    assert (tmp_path / "cv.pdf").read_text(encoding="utf-8").count("Results-driven") == 0


def test_good_first_round_skips_the_second(master: Any, tmp_path: Path) -> None:
    writer, reviewer = FakeWriter([content()]), FakeReviewer([review(8)])
    result = generate_resume(JD, master, "en", writer, reviewer, tmp_path / "cv.pdf",  # type: ignore[arg-type]
                             max_rounds=3, renderer=fake_render)
    assert result.passed and len(writer.calls) == 1 and reviewer.calls == 1


def test_loop_stops_at_max_rounds(master: Any, tmp_path: Path) -> None:
    writer = FakeWriter([content(summary="Leveraged synergy.")])
    result = generate_resume(JD, master, "en", writer, FakeReviewer([review(5)]),  # type: ignore[arg-type]
                             tmp_path / "cv.pdf", max_rounds=2, renderer=fake_render)
    assert not result.passed and len(writer.calls) == 2
    report = render_review_report(result, "Acme", "Dev", "https://x", ["7+ years Java 11"])
    assert "REVISAR MANUALMENTE" in report and "7+ years Java 11" in report and "Kubernetes" in report


# ---- Postular sheet ----------------------------------------------------------------


def assessment(stop: bool = False) -> RequirementsAssessment:
    return RequirementsAssessment(
        company="Acme", role_title="Senior Java Developer", location="Toronto, ON",
        requirements=[Requirement(requirement="CPO certification" if stop else "Java", kind="must_have",
                                  status="not_met" if stop else "met", evidence="e"),
                      Requirement(requirement="Java 11+", kind="must_have", status="unclear", evidence="no version")],
    )


class FakeRequirements:
    def __init__(self, stop: bool = False) -> None:
        self.stop = stop

    def check(self, *args: Any) -> RequirementsAssessment:
        return assessment(self.stop)


PROFILE = CandidateProfile(name="Alex", title="Dev", years_experience=12, skills=["Java"])


@pytest.fixture
def sheet_path(tmp_path: Path) -> Path:
    path = create_tracker_template(tmp_path / "tracker.xlsx")
    assert postular.ensure_sheet(path) is True
    assert postular.ensure_sheet(path) is False
    return path


def add_row(path: Path, **values: Any) -> int:
    return postular.table(path).append(values)


def run(path: Path, master: Any, tmp_path: Path, row: dict[str, Any], *, stop: bool = False,
        fetch: Any = None) -> postular.RowOutcome:
    return postular.process_row(
        row, path=path, master=master, profile=PROFILE, requirements=FakeRequirements(stop),  # type: ignore[arg-type]
        writer=FakeWriter([content()]), reviewer=FakeReviewer([review(9)]),  # type: ignore[arg-type]
        output_dir=tmp_path / "out", fetch=fetch or (lambda url: JobDescription(text=JD, title="Java Dev")),
        renderer=fake_render, today=TODAY,
    )


def test_sheet_is_first_with_dropdowns(sheet_path: Path) -> None:
    wb = load_workbook(sheet_path)
    assert wb.sheetnames[0] == "Postular"
    assert len(wb["Postular"].data_validations.dataValidation) == 2


def test_pending_rows(sheet_path: Path) -> None:
    add_row(sheet_path, link="https://a")
    add_row(sheet_path, link="https://b", status="CV listo – revisar")
    add_row(sheet_path, link="https://c", status="Regenerar")
    add_row(sheet_path, company="no link or description")
    assert [r["link"] for r in postular.pending_rows(sheet_path)] == ["https://a", "https://c"]
    assert [r["link"] for r in postular.pending_rows(sheet_path, only_row=3)] == ["https://b"]


def test_process_row_success_updates_sheet(sheet_path: Path, master: Any, tmp_path: Path) -> None:
    add_row(sheet_path, link="https://job-boards.greenhouse.io/acme/jobs/1", notes="stress Kafka")
    [row] = postular.pending_rows(sheet_path)
    out = run(sheet_path, master, tmp_path, row)
    assert out.status == postular.STATUS_READY and out.cv_path is not None
    assert out.cv_path.name == "Alex_Sample_Resume_Acme_Java-Dev.pdf"
    folder = out.cv_path.parent
    assert (folder / "job_description.txt").read_text(encoding="utf-8").startswith("https://job-boards")
    assert "Java 11+" in (folder / "revision_hr.md").read_text(encoding="utf-8")
    [saved] = postular.table(sheet_path).rows()
    assert saved["status"] == postular.STATUS_READY and saved["cv_file"] == out.cv_path.name
    assert saved["language"] == "en" and saved["hr_score"] == 9 and saved["company"] == "Acme"
    ws = load_workbook(sheet_path)["Postular"]
    assert ws.cell(saved["_row"], 9).hyperlink.target.startswith("file:")


def test_process_row_without_description(sheet_path: Path, master: Any, tmp_path: Path) -> None:
    add_row(sheet_path, link="https://www.linkedin.com/jobs/view/1")
    [row] = postular.pending_rows(sheet_path)

    def refuse(url: str) -> JobDescription:
        raise JobDescriptionError("linkedin.com does not allow automated reading")

    out = run(sheet_path, master, tmp_path, row, fetch=refuse)
    assert out.status == postular.STATUS_NO_JD
    assert postular.table(sheet_path).rows()[0]["status"] == postular.STATUS_NO_JD


def test_pasted_description_wins_and_french_detected(sheet_path: Path, master: Any, tmp_path: Path) -> None:
    fr = ("Nous cherchons un développeur Java avec de l'expérience pour notre équipe. Vous avez une "
          "maîtrise des microservices Spring Boot et des compétences en Kafka dans les services. ") * 4
    add_row(sheet_path, link="https://www.linkedin.com/jobs/view/1", description=fr, company="Desjardins")

    def must_not_fetch(url: str) -> JobDescription:
        raise AssertionError("pasted text must be used")

    [row] = postular.pending_rows(sheet_path)
    out = run(sheet_path, master, tmp_path, row, fetch=must_not_fetch)
    assert out.cv_path is not None and out.cv_path.name.startswith("Alex_Sample_CV_Desjardins")
    assert postular.table(sheet_path).rows()[0]["language"] == "fr"


def test_hard_mismatch_stops_unless_forced(sheet_path: Path, master: Any, tmp_path: Path) -> None:
    add_row(sheet_path, link="https://x/1")
    [row] = postular.pending_rows(sheet_path)
    out = run(sheet_path, master, tmp_path, row, stop=True)
    assert out.status == postular.STATUS_MISMATCH and out.cv_path is None
    assert "Forzar" in postular.table(sheet_path).rows()[0]["hr_notes"]

    postular.table(sheet_path).update(row["_row"], {"status": "Forzar"})
    [row] = postular.pending_rows(sheet_path)
    forced = run(sheet_path, master, tmp_path, row, stop=True)
    assert forced.status == postular.STATUS_READY
    assert "NO CUMPLES: CPO certification" in (forced.cv_path.parent / "revision_hr.md").read_text(encoding="utf-8")  # type: ignore[union-attr]


def test_choose_language_and_filename() -> None:
    assert postular.choose_language("FR", JD) == "fr"
    assert postular.choose_language("auto", JD) == "en"
    assert postular.cv_filename("Felipe_Rodriguez", "fr", "Banque Nationale", "Développeur Java") == \
        "Felipe_Rodriguez_CV_Banque-Nationale_Developpeur-Java.pdf"


def test_low_candidate_fit_does_not_block() -> None:
    assert passes(review(9, fit=3), [])  # real gaps are informational, not a writing failure


def test_best_round_is_kept(master: Any, tmp_path: Path) -> None:
    good, worse = content(), content(summary="Leveraged synergy with 99 projects.")
    issues = [ReviewIssue(severity="major", where="x", problem="p", fix="f")]
    writer = FakeWriter([good, worse])
    reviewer = FakeReviewer([review(6, issues), review(4, issues + issues)])
    result = generate_resume(JD, master, "en", writer, reviewer, tmp_path / "cv.pdf",  # type: ignore[arg-type]
                             max_rounds=2, renderer=fake_render)
    assert result.content == good and len(result.rounds) == 2
    pdf_html = (tmp_path / "cv.pdf").read_text(encoding="utf-8")
    assert "synergy" not in pdf_html  # the better first round was re-rendered


def test_reviewer_prompt_has_today(master: Any) -> None:
    from job_assistant.agents.hr_reviewer import build_messages as review_messages

    system, _ = review_messages(JD, content(), master, "en", [], today=date(2026, 10, 1))
    assert "Today is 2026-10-01" in system.content


def test_quebec_english_headline_rule(master: Any) -> None:
    from job_assistant.agents.resume_writer import build_messages as writer_messages

    c = content(title="Senior Java Engineer")
    source = master.source_text("en")
    assert any("Quebec" in f.problem for f in run_checks(c, "en", source, JD, 2, quebec=True))
    assert not any("Quebec" in f.problem for f in run_checks(c, "en", source, JD, 2, quebec=False))
    system, _ = writer_messages(JD, master, "en", quebec=True)
    assert "protected title (OIQ) in English too" in system.content


def test_montreal_posting_passes_quebec_flag(sheet_path: Path, master: Any, tmp_path: Path) -> None:
    add_row(sheet_path, link="https://x/1", location="Montreal, QC")
    [row] = postular.pending_rows(sheet_path)
    writer = FakeWriter([content()])
    postular.process_row(
        row, path=sheet_path, master=master, profile=PROFILE, requirements=FakeRequirements(),  # type: ignore[arg-type]
        writer=writer, reviewer=FakeReviewer([review(9)]), output_dir=tmp_path / "out",  # type: ignore[arg-type]
        fetch=lambda url: JobDescription(text=JD), renderer=fake_render, today=TODAY,
    )
    assert writer.calls[0]["quebec"] is True


def test_short_summary_used_when_link_unreadable(sheet_path: Path, master: Any, tmp_path: Path) -> None:
    add_row(sheet_path, link="https://www.linkedin.com/jobs/view/9", company="Cogency",
            description="LinkedIn Easy Apply. 8+ years. Java, Spring Boot, microservices, Kafka, React, AWS.")

    def refuse(url: str) -> JobDescription:
        raise JobDescriptionError("linkedin.com does not allow automated reading")

    [row] = postular.pending_rows(sheet_path)
    out = run(sheet_path, master, tmp_path, row, fetch=refuse)
    assert out.cv_path is not None
    assert postular.table(sheet_path).rows()[0]["hr_notes"].startswith("Basado en un resumen corto")
    assert "Basado en un resumen corto" in (out.cv_path.parent / "revision_hr.md").read_text(encoding="utf-8")


def test_tiny_description_still_missing(sheet_path: Path, master: Any, tmp_path: Path) -> None:
    add_row(sheet_path, link="https://www.linkedin.com/jobs/view/9", description="Java")

    def refuse(url: str) -> JobDescription:
        raise JobDescriptionError("blocked")

    [row] = postular.pending_rows(sheet_path)
    assert run(sheet_path, master, tmp_path, row, fetch=refuse).status == postular.STATUS_NO_JD


def test_review_effort_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from job_assistant.config import load_settings
    from job_assistant.llm import get_chat_model

    def settings_with(value: str | None) -> Any:
        for var in ('REVIEW_EFFORT', 'ANTHROPIC_API_KEY'):  # load_dotenv leaks into os.environ
            monkeypatch.delenv(var, raising=False)
        env = tmp_path / 'effort.env'
        env.write_text('ANTHROPIC_API_KEY=k' + chr(10) + (f'REVIEW_EFFORT={value}' + chr(10) if value is not None else ''),
                       encoding='utf-8')
        return load_settings(env)

    assert settings_with(None).review_effort == 'low'        # not configured: cheap default
    assert settings_with('medium').review_effort == 'medium'
    assert settings_with('').review_effort is None           # empty: the model's default
    s = settings_with('low')
    assert get_chat_model(s, effort='low').reasoning_effort == 'low'
    assert get_chat_model(s).reasoning_effort is None
