from pathlib import Path
from typing import Any

import pytest

from job_assistant.cv import verify
from job_assistant.cv.content import Bullet, JobBullets
from job_assistant.cv.render import build_html
from job_assistant.cv.roles import Role
from job_assistant.cv.verify import FAIL, PASS, REVIEW, LEVEL_WORDS, render_verification, verify_resume
from tests.test_cv import content, master, master_dir  # noqa: F401  (fixtures are used by name)

JD_TEXT = "Company: Acme\nTitle: Java Developer\nLocation: Toronto, ON\n\nWe need a Java developer with Spring Boot."
GUIDEWIRE = Role("gw", "Guidewire Business Analyst", "Guidewire", "BA", ("Guidewire Business Analyst",),
                 ("Guidewire",), "x", verify_terms=("Guidewire",))


def pdf_for(master: Any, extra: str = "") -> str:
    c = master.contact
    return f"{c['name']}\n{c.get('email', '')} {c.get('phone', '')}\nCanadian Permanent Resident\n{extra}"


def check(master: Any, resume: Any = None, pages: int = 2, text: str | None = None, roles: Any = None) -> verify.Verdict:
    return verify_resume("acme", resume or content(), "en", JD_TEXT, pages, pdf_for(master) if text is None else text,
                         master, roles, title="Java Developer", location="Toronto, ON", company="Acme")


def test_a_clean_resume_passes(master: Any) -> None:
    v = check(master)
    assert v.status == PASS, [i.problem for i in v.issues if i.severity != "minor"]


def test_too_many_pages_fails(master: Any) -> None:
    v = check(master, pages=3)
    assert v.status == FAIL and any("3 pages" in i.problem for i in v.issues)


def test_a_language_level_in_the_pdf_fails(master: Any) -> None:
    v = check(master, text=pdf_for(master, "Languages: Spanish (native), English (C1)"))
    assert v.status == FAIL and any("language level" in i.problem for i in v.issues)


@pytest.mark.parametrize("phrase", ["React Native", "cloud-native services", "B2B payments"])
def test_technologies_that_look_like_levels_are_not_flagged(phrase: str) -> None:
    assert LEVEL_WORDS.search(phrase) is None


def test_missing_work_authorization_needs_review(master: Any) -> None:
    v = check(master, text=f"{master.contact['name']}\n{master.contact.get('email', '')} {master.contact.get('phone', '')}")
    assert v.status == REVIEW and any("work authorization" in i.problem for i in v.issues)


def test_missing_contact_in_the_pdf_needs_review(master: Any) -> None:
    v = check(master, text="Canadian Permanent Resident")
    assert v.status == REVIEW and any("email" in i.problem for i in v.issues)


def test_dashes_as_punctuation_need_review(master: Any) -> None:
    v = check(master, resume=content(summary="Java developer — builds Spring Boot services - fast."))
    assert v.status == REVIEW and any("Dashes" in i.problem for i in v.issues)


def test_a_role_claim_without_facts_fails(master: Any) -> None:
    claims = content(summary="Java developer who configured Guidewire for insurers, with Spring Boot.")
    v = check(master, resume=claims, roles=[GUIDEWIRE])
    assert v.status == FAIL and any("Guidewire" in i.problem for i in v.issues)
    assert v.role == "Guidewire Business Analyst"


def test_a_tool_claimed_on_a_job_without_support_fails(master: Any) -> None:
    master.raw["skills"]["tools_en"] = [*master.raw["skills"]["tools_en"], "Postman"]
    resume = content(experience=[JobBullets(job_id="bank", bullets=[
        Bullet(lead="Tested APIs", rest="in Postman for the payments service.")])])
    v = check(master, resume=resume)
    assert v.status == FAIL and any("Postman" in i.problem for i in v.issues)


def test_report_sorts_failures_first_and_counts(master: Any) -> None:
    verdicts = [check(master), check(master, pages=3), check(master, text="Canadian Permanent Resident")]
    verdicts[1].name, verdicts[2].name = "failing", "review-me"
    text = render_verification(verdicts)
    assert "1 pasan · 1 para revisar · 1 fallan" in text
    rows = [line for line in text.splitlines() if line.startswith("| ") and "Resultado" not in line and "---" not in line]
    assert [r.split("|")[1].strip() for r in rows] == [FAIL, REVIEW, PASS]


def test_verify_folder_reads_the_saved_result(master: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "Acme_Java-Developer"
    folder.mkdir()
    (folder / "Alex_Resume_Acme.pdf").write_bytes(b"%PDF")
    (folder / "resume_content.json").write_text(content().model_dump_json(), encoding="utf-8")
    (folder / "job_description.txt").write_text("https://x/1\nSource: pasted\n\n" + JD_TEXT, encoding="utf-8")
    monkeypatch.setattr(verify, "page_count", lambda pdf: 2)
    monkeypatch.setattr(verify, "pdf_text", lambda pdf: pdf_for(master))
    v = verify.verify_folder(folder, master)
    assert v is not None and v.name == "Acme_Java-Developer" and v.status == PASS
    assert verify.verify_folder(tmp_path, master) is None  # a folder with no resume is skipped


def test_the_rendered_header_would_pass_the_authorization_check(master: Any) -> None:
    master.contact["work_authorization"] = "Canadian Permanent Resident"
    assert "Canadian Permanent Resident" in build_html(master, content(), "en")
