from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from openpyxl import load_workbook

from job_assistant.agents.hr_reviewer import HRReview, Scores
from job_assistant.agents.requirements_agent import Requirement, RequirementsAssessment
from job_assistant.cv import postular
from job_assistant.cv.content import Bullet, JobBullets, ResumeContent
from job_assistant.cv.jd import JobDescription
from job_assistant.cv.master import load_master
from job_assistant.profile.candidate import CandidateProfile
from job_assistant.tracker.template import create_tracker_template

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
JD = ("Senior Java Developer. We need Java, Spring Boot, Apache Kafka, PostgreSQL and Docker. " * 3)
TODAY = date(2026, 10, 1)
PROFILE = CandidateProfile(name="Alex", title="Dev", years_experience=12, skills=["Java"])


class FakeWriter:
    def write(self, *args: Any, **kw: Any) -> ResumeContent:
        return ResumeContent(
            title="Senior Java Developer",
            summary="Java developer with 12 years of experience building Spring Boot microservices.",
            competencies=["Java", "Spring Boot"],
            experience=[
                JobBullets(job_id="bank", bullets=[
                    Bullet(lead="Built Spring Boot microservices", rest="for payments."),
                ])
            ],
            tools=["Java", "Spring Boot"],
            methodologies=["Agile"],
        )


def review(score: int = 9, fit: int = 8) -> HRReview:
    return HRReview(
        scores=Scores(requirements_coverage=score, truthfulness=score, human_tone=score,
                      clarity=score, ats_keywords=score, candidate_fit=fit),
        issues=[], gaps_cannot_fix=["Kubernetes"],
        likely_interview_questions=["Tell me about Kafka."], notes_for_candidate="Buen calce.",
    )


class FakeReviewer:
    def review(self, *args: Any, **kw: Any) -> HRReview:
        return review(9)


class FakeRequirements:
    def check(self, *args: Any, **kw: Any) -> RequirementsAssessment:
        return RequirementsAssessment(
            company="Acme", role_title="Senior Java Developer", location="Toronto, ON",
            requirements=[Requirement(requirement="Java", kind="must_have", status="met", evidence="e")],
        )


def fake_render(html: str, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return 2


def test_concurrent_row_processing(tmp_path: Path) -> None:
    (tmp_path / "cv_master.toml").write_bytes((EXAMPLES / "cv_master.example.toml").read_bytes())
    master = load_master(tmp_path, allow_example=True)

    sheet_path = create_tracker_template(tmp_path / "tracker.xlsx")
    postular.ensure_sheet(sheet_path)

    table = postular.table(sheet_path)
    for i in range(1, 11):
        table.append({"link": f"https://company{i}.com/job", "company": f"Company {i}", "role": "Java Dev"})

    rows = postular.pending_rows(sheet_path)
    assert len(rows) == 10

    def process(r: dict[str, Any]) -> postular.RowOutcome:
        return postular.process_row(
            r, path=sheet_path, master=master, profile=PROFILE,
            requirements=FakeRequirements(),  # type: ignore[arg-type]
            writer=FakeWriter(),  # type: ignore[arg-type]
            reviewer=FakeReviewer(),  # type: ignore[arg-type]
            output_dir=tmp_path / "out",
            fetch=lambda url: JobDescription(text=JD, title="Java Dev"),
            renderer=fake_render, today=TODAY,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process, rows))

    assert len(results) == 10
    assert all(res.status == postular.STATUS_READY for res in results)

    # Verify that all 10 rows in tracker.xlsx were written cleanly without corruption
    saved_rows = table.rows()
    assert len(saved_rows) == 10
    for r in saved_rows:
        assert r["status"] == postular.STATUS_READY
        assert r["company"].startswith("Company ")
