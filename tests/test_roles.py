import json
from pathlib import Path
from typing import Any

import pytest

from job_assistant.agents.resume_writer import SYSTEM_PROMPT, build_messages
from job_assistant.cv import postular
from job_assistant.cv.checks import skills_only_claims
from job_assistant.cv.content import Bullet, JobBullets, ResumeContent, normalize, strip_dashes
from job_assistant.cv.jd import JobDescription
from job_assistant.cv.masters import build_masters, load_master_content, render_report
from job_assistant.cv.pipeline import generate_resume
from job_assistant.cv.render import build_html
from job_assistant.cv.roles import (
    Role,
    RoleBase,
    RolesNotFoundError,
    coverage,
    load_roles,
    match_role,
    synthetic_posting,
    ungrounded_terms,
)
from tests.test_cv import (  # noqa: F401  (fixtures are used by name)
    JD,
    PROFILE,
    TODAY,
    FakeRequirements,
    FakeReviewer,
    add_row,
    content,
    fake_render,
    master,
    master_dir,
    review,
    sheet_path,
)

ROLES_TOML = """
[[roles]]
id = "bsa"
name = "Business Systems Analyst"
file_name = "Business-Systems-Analyst"
headline = "Business Systems Analyst"
title_keywords = ["Business Analyst", "Systems Analyst", "Analyste d'affaires"]
ats_keywords = ["Requirements gathering", "UAT", "SQL", "Jira"]
verify_terms = ["Guidewire"]
focus = "Requirements analysis and insurance."

[[roles]]
id = "guidewire-ba"
name = "Guidewire Business Analyst"
file_name = "Guidewire-Business-Analyst"
headline = "Business Analyst"
title_keywords = ["Guidewire Business Analyst", "Guidewire BA"]
ats_keywords = ["Guidewire", "PolicyCenter", "Requirements gathering"]
verify_terms = ["Guidewire", "PolicyCenter"]
focus = "No Guidewire product experience: never claim it."

[[roles]]
id = "java"
name = "Backend Developer (Java)"
file_name = "Backend-Developer-Java"
headline = "Senior Java Backend Developer"
title_keywords = ["Java Developer", "Backend Developer", "Software Engineer"]
ats_keywords = ["Java", "Spring Boot", "Kafka", "Docker"]
focus = "Hands-on Java."

[[roles]]
id = "lead"
name = "Technical Team Lead (Java)"
file_name = "Technical-Team-Lead-Java"
headline = "Technical Team Lead"
title_keywords = ["Tech Lead", "Lead Software Engineer", "Technical Lead"]
ats_keywords = ["Java", "Mentoring", "Code review", "Sprint planning"]
focus = "Leading a team."
"""


@pytest.fixture
def roles(tmp_path: Path) -> list[Role]:
    (tmp_path / "roles.toml").write_text(ROLES_TOML, encoding="utf-8")
    return load_roles(tmp_path)


def by_id(roles: list[Role], role_id: str) -> Role:
    return next(r for r in roles if r.id == role_id)


# ---- loading and matching ----------------------------------------------------------------------


def test_load_roles(roles: list[Role]) -> None:
    assert [r.id for r in roles] == ["bsa", "guidewire-ba", "java", "lead"]
    assert roles[0].pdf_name == "Master_Business-Systems-Analyst_Resume_EN.pdf"
    assert roles[1].verify_terms == ("Guidewire", "PolicyCenter")


def test_missing_roles_file(tmp_path: Path) -> None:
    with pytest.raises(RolesNotFoundError):
        load_roles(tmp_path)


@pytest.mark.parametrize(("title", "expected"), [
    ("Senior Business Analyst, Insurance", "bsa"),
    ("Guidewire Business Analyst", "guidewire-ba"),      # the more specific keyword beats "Business Analyst"
    ("Senior Software Engineer", "java"),
    ("Lead Software Engineer", "lead"),                   # "Lead Software Engineer" beats "Software Engineer"
    ("Analyste d'affaires TI", "bsa"),                    # accents and French titles
])
def test_match_role_by_title(roles: list[Role], title: str, expected: str) -> None:
    assert match_role(title, "", roles).role.id == expected


def test_match_role_explains_itself_and_names_the_runner_up(roles: list[Role]) -> None:
    m = match_role("Lead Software Engineer", "Java mentoring code review", roles)
    assert "Lead Software Engineer" in m.reason and m.runner_up is not None
    assert m.runner_up.id == "java" and m.runner_up.name in m.reason


def test_match_role_falls_back_to_posting_content(roles: list[Role]) -> None:
    m = match_role("Consultant", "We need Java, Spring Boot, Kafka and Docker.", roles)
    assert m.role.id == "java" and "contenido" in m.reason


def test_match_role_with_no_signal_still_answers(roles: list[Role]) -> None:
    m = match_role("Chef de cuisine", "Cook meals.", roles)
    assert m.role.id == "bsa" and "Sin coincidencia" in m.reason


def test_real_roles_file_is_consistent() -> None:
    path = Path(__file__).resolve().parent.parent / "data"
    if not (path / "roles.toml").is_file():
        pytest.skip("data/roles.toml is personal and not in the repository")
    real = load_roles(path)
    assert len(real) == 10
    assert all(len(r.ats_keywords) == 10 and r.title_keywords for r in real)
    assert len({r.file_name for r in real}) == 10
    assert match_role("Guidewire Business Analyst", "", real).role.id == "guidewire-ba"
    assert match_role("Java Developer", "", real).role.id == "java-backend"


# ---- facts coverage and the claims a role resume may not make ---------------------------------


FACTS = "Built Spring Boot services with Java.\nDefined UAT scripts with the business team.\nWrote SQL reports."


def test_coverage_splits_covered_and_gaps(roles: list[Role]) -> None:
    covered, gaps = coverage(by_id(roles, "bsa"), FACTS)
    assert covered == ["UAT", "SQL"] and gaps == ["Requirements gathering", "Jira"]


def test_coverage_accepts_all_words_of_a_phrase_on_one_line(roles: list[Role]) -> None:
    covered, _ = coverage(by_id(roles, "bsa"), "Led requirements analysis and gathering with vendors.")
    assert "Requirements gathering" in covered


def test_coverage_matches_other_forms_of_a_word(roles: list[Role]) -> None:
    lead = by_id(roles, "lead")  # keywords: Java, Mentoring, Code review, Sprint planning
    covered, gaps = coverage(lead, "Mentored 15 developers on Java and reviewed code during sprint planning.")
    assert covered == ["Java", "Mentoring", "Code review", "Sprint planning"] and gaps == []
    covered, gaps = coverage(lead, "Built Java services.")
    assert covered == ["Java"] and gaps == ["Mentoring", "Code review", "Sprint planning"]


def test_coverage_word_forms_do_not_overmatch() -> None:
    role = Role("x", "X", "X", "X", (), ("Java", "REST APIs", "KPI", "Test plans"), "")
    facts = "\n".join(["Built JavaScript apps.", "Shipped RESTful APIs and tracked KPIs.", "Wrote tests."])
    covered, gaps = coverage(role, facts)
    assert covered == ["REST APIs", "KPI"] and gaps == ["Java", "Test plans"]  # JavaScript is not Java


def test_coverage_does_not_mistake_a_product_for_a_common_word() -> None:
    role = Role("x", "X", "X", "X", (), ("ClaimCenter", "ServiceNow", "Integration", "Configuration"), "")
    facts = "\n".join(["Built insurance claims tooling.", "Integrated services and configured environments for the service desk."])
    covered, gaps = coverage(role, facts)
    assert covered == ["Integration", "Configuration"] and gaps == ["ClaimCenter", "ServiceNow"]


def test_ungrounded_terms(roles: list[Role]) -> None:
    role = by_id(roles, "guidewire-ba")
    assert ungrounded_terms(role, "Configured Guidewire PolicyCenter.", FACTS) == ["Guidewire", "PolicyCenter"]
    assert ungrounded_terms(role, "Configured Guidewire PolicyCenter.", FACTS + "\nGuidewire PolicyCenter work") == []
    assert ungrounded_terms(role, "Reviewed insurance requirements.", FACTS) == []


# ---- writing rules enforced in code ------------------------------------------------------------


def test_strip_dashes_keeps_ranges_and_compound_words() -> None:
    assert strip_dashes("Led X — and Y – then Z - done") == "Led X, and Y, then Z, done"
    assert strip_dashes("2021 – 2026, 2021-2026, cloud-native, event-driven") == "2021 – 2026, 2021-2026, cloud-native, event-driven"


def test_normalize_removes_dash_punctuation_from_the_whole_resume(master: Any) -> None:
    dirty = content(title="Java Developer — Insurance", summary="Built services — fast. Then more - done.")
    dirty.experience[0].bullets[0] = Bullet(lead="Built it — well", rest="for payments – at scale.")
    fixed, _ = normalize(dirty, master, "en")
    text = fixed.all_text()
    assert "—" not in text and " – " not in text and " - " not in text
    assert fixed.title == "Java Developer, Insurance"


def test_writer_prompt_carries_the_xyz_and_dash_rules() -> None:
    assert "Google XYZ" in SYSTEM_PROMPT and "never invent a measure" in SYSTEM_PROMPT
    assert "No dashes as punctuation" in SYSTEM_PROMPT and "Canadian spelling" in SYSTEM_PROMPT


def test_writer_prompt_includes_target_role_and_master(master: Any, roles: list[Role]) -> None:
    base = RoleBase(by_id(roles, "java"), '{"title": "MASTER-JSON"}')
    text = build_messages(JD, master, "en", role_base=base)[1].content
    assert "<target_role>" in text and "Spring Boot" in text and "MASTER-JSON" in text
    assert "write the result in French" not in text
    fr = build_messages(JD, master, "fr", role_base=base)[1].content
    assert "write the result in French from the facts" in fr


def test_writer_prompt_without_role_has_no_role_block(master: Any) -> None:
    assert "<target_role>" not in build_messages(JD, master, "en")[1].content


def test_writing_a_master_has_role_guidance_but_no_master_json(master: Any, roles: list[Role]) -> None:
    text = build_messages(synthetic_posting(roles[0]), master, "en", role_base=RoleBase(roles[0]))[1].content
    assert "<target_role>" in text and "starting point" not in text
    assert "Requirements gathering" in synthetic_posting(roles[0])


# ---- layout ------------------------------------------------------------------------------------


def test_render_has_work_authorization_and_no_dashes_in_education(master: Any) -> None:
    master.contact["work_authorization"] = "Canadian Permanent Resident"
    master.contact["work_authorization_fr"] = "Résident permanent du Canada"
    master.raw["education"]["en"] = ["BSc, Computer Science — Some University — 2012"]
    html_en = build_html(master, content(), "en")
    assert "Canadian Permanent Resident" in html_en and "—" not in html_en
    assert "BSc, Computer Science, Some University, 2012" in html_en
    assert "Résident permanent du Canada" in build_html(master, content(), "fr") or not master.has_language("fr")


def test_render_without_work_authorization_adds_no_line(master: Any) -> None:
    master.contact.pop("work_authorization", None)
    assert "Resident" not in build_html(master, content(), "en")


# ---- the role claims check inside the writer <-> reviewer loop --------------------------------


class RecordingWriter:
    def __init__(self, drafts: list[ResumeContent]) -> None:
        self.drafts, self.role_bases = drafts, []

    def write(self, jd: str, master: Any, lang: str, notes: str = "", previous: Any = None,
              feedback: Any = None, quebec: bool = False, role_base: Any = None) -> ResumeContent:
        self.role_bases.append(role_base)
        return self.drafts[min(len(self.role_bases) - 1, len(self.drafts) - 1)]


def test_pipeline_blocks_a_claim_the_facts_do_not_support(master: Any, roles: list[Role], tmp_path: Path) -> None:
    claims = content(summary="Java developer who configured Guidewire PolicyCenter for insurers.")
    honest = content()
    writer = RecordingWriter([claims, honest])
    result = generate_resume(JD, master, "en", writer, FakeReviewer([review(9)]), tmp_path / "a.pdf",  # type: ignore[arg-type]
                             renderer=fake_render, role_base=RoleBase(by_id(roles, "guidewire-ba")))
    assert len(writer.role_bases) == 2 and all(b is not None for b in writer.role_bases)
    assert "Guidewire" not in result.content.all_text()
    assert result.rounds[0].serious_issues >= 1  # round 1 carried the blocker


def test_pipeline_without_a_role_does_not_pass_role_base(master: Any, tmp_path: Path) -> None:
    class Plain:
        def write(self, jd: str, master: Any, lang: str, notes: str = "", previous: Any = None,
                  feedback: Any = None, quebec: bool = False) -> ResumeContent:
            return content()

    generate_resume(JD, master, "en", Plain(), FakeReviewer([review(9)]), tmp_path / "a.pdf",  # type: ignore[arg-type]
                    renderer=fake_render)


# ---- building the masters ----------------------------------------------------------------------


def test_build_masters_writes_pdfs_index_and_report(master: Any, roles: list[Role], tmp_path: Path) -> None:
    out = tmp_path / "master_resumes"
    results = build_masters(roles, master, RecordingWriter([content()]), FakeReviewer([review(9)]), out,  # type: ignore[arg-type]
                            renderer=fake_render)
    assert [r.pdf.name for r in results] == [r.pdf_name for r in roles]
    assert all(r.pdf.is_file() and r.content_file.is_file() and r.pages == 2 for r in results)
    index = json.loads((out / "roles_index.json").read_text(encoding="utf-8"))
    assert [e["id"] for e in index] == ["bsa", "guidewire-ba", "java", "lead"]
    java = next(e for e in index if e["id"] == "java")
    assert java["file"] == "Master_Backend-Developer-Java_Resume_EN.pdf"
    assert "Software Engineer" in java["title_keywords"] and "Kafka" in java["ats_keywords"]
    report = (out / "ROLES_REPORT.md").read_text(encoding="utf-8")
    assert "Guidewire Business Analyst" in report and "**gap**" in report


def test_masters_are_written_with_role_guidance(master: Any, roles: list[Role], tmp_path: Path) -> None:
    writer = RecordingWriter([content()])
    build_masters(roles[:1], master, writer, FakeReviewer([review(9)]), tmp_path, renderer=fake_render)  # type: ignore[arg-type]
    assert writer.role_bases[0].role.id == "bsa" and writer.role_bases[0].master_json == ""


def test_load_master_content(master: Any, roles: list[Role], tmp_path: Path) -> None:
    assert load_master_content(tmp_path, roles[0]) == ""
    build_masters(roles[:1], master, RecordingWriter([content()]), FakeReviewer([review(9)]), tmp_path,  # type: ignore[arg-type]
                  renderer=fake_render)
    assert json.loads(load_master_content(tmp_path, roles[0]))["title"] == "Senior Java Developer"


def test_report_lists_gap_keywords(master: Any, roles: list[Role], tmp_path: Path) -> None:
    results = build_masters(roles, master, RecordingWriter([content()]), FakeReviewer([review(9)]), tmp_path,  # type: ignore[arg-type]
                            renderer=fake_render)
    assert "| Guidewire | **gap** |" in render_report(results)


# ---- phase 2: the tracker row picks its master ------------------------------------------------


def test_process_row_starts_from_the_matching_master_and_records_it(
    sheet_path: Path, master: Any, roles: list[Role], tmp_path: Path,
) -> None:
    masters_dir = tmp_path / "masters"
    build_masters(roles, master, RecordingWriter([content(title="MASTER TITLE")]), FakeReviewer([review(9)]),  # type: ignore[arg-type]
                  masters_dir, renderer=fake_render)
    add_row(sheet_path, link="https://job-boards.greenhouse.io/acme/jobs/1", role="Senior Java Developer")
    [row] = postular.pending_rows(sheet_path)
    writer = RecordingWriter([content()])
    out = postular.process_row(
        row, path=sheet_path, master=master, profile=PROFILE, requirements=FakeRequirements(False),  # type: ignore[arg-type]
        writer=writer, reviewer=FakeReviewer([review(9)]), output_dir=tmp_path / "out",  # type: ignore[arg-type]
        fetch=lambda url: JobDescription(text=JD, title="Java Dev"), renderer=fake_render, today=TODAY,
        roles=roles, masters_dir=masters_dir,
    )
    base = writer.role_bases[0]
    assert base.role.id == "java" and "MASTER TITLE" in base.master_json
    assert out.master.startswith("Backend Developer (Java)") and "coincide" in out.master
    [saved] = [r for r in postular.table(sheet_path).rows() if r["_row"] == row["_row"]]
    assert saved["master_role"] == out.master
    report = (out.cv_path.parent / "revision_hr.md").read_text(encoding="utf-8")  # type: ignore[union-attr]
    assert "**Master usado:** Backend Developer (Java)" in report


def test_process_row_without_roles_behaves_as_before(sheet_path: Path, master: Any, tmp_path: Path) -> None:
    add_row(sheet_path, link="https://job-boards.greenhouse.io/acme/jobs/1")
    [row] = postular.pending_rows(sheet_path)
    writer = RecordingWriter([content()])
    out = postular.process_row(
        row, path=sheet_path, master=master, profile=PROFILE, requirements=FakeRequirements(False),  # type: ignore[arg-type]
        writer=writer, reviewer=FakeReviewer([review(9)]), output_dir=tmp_path / "out",  # type: ignore[arg-type]
        fetch=lambda url: JobDescription(text=JD, title="Java Dev"), renderer=fake_render, today=TODAY,
    )
    assert writer.role_bases == [None] and out.master == ""


# ---- a tool listed only as a skill cannot become an activity on a job --------------------------


def _skills_master() -> Any:
    from job_assistant.cv.master import CVMaster

    raw = {
        "experience": [
            {"id": "a", "company": "A", "location": "X", "title_en": "Dev", "dates_en": "2020",
             "bullets_en": ["Built Spring Boot services with SQL Server.", "Reviewed code for 12 developers."]},
            {"id": "b", "company": "B", "location": "X", "title_en": "Dev", "dates_en": "2019",
             "bullets_en": ["Wrote Postman collections for the payments API."]},
        ],
        "skills": {"tools_en": ["Spring Boot", "SQL Server", "Postman", "Git & GitHub", "Java (J2EE, JSP)"],
                   "methodologies_en": []},
    }
    return CVMaster(contact={"name": "X"}, raw=raw)


def _resume(*bullets_by_job: tuple[str, str]) -> ResumeContent:
    return content(experience=[JobBullets(job_id=j, bullets=[Bullet(lead="", rest=b)]) for j, b in bullets_by_job])


def test_skills_only_tool_claimed_in_a_bullet_is_flagged() -> None:
    bad = _resume(("a", "Performed REST API checks in Postman across SQL Server."))
    assert skills_only_claims(bad, _skills_master(), "en") == ["Postman"]


def test_tool_in_the_jobs_own_facts_is_fine_and_is_judged_per_job() -> None:
    ok = _resume(("a", "Built Spring Boot services on SQL Server."), ("b", "Wrote Postman collections for the API."))
    assert skills_only_claims(ok, _skills_master(), "en") == []
    moved = _resume(("a", "Wrote Postman collections for the API."))   # Postman belongs to job b, not a
    assert skills_only_claims(moved, _skills_master(), "en") == ["Postman"]


def test_listing_the_tool_in_the_tools_line_is_not_a_claim() -> None:
    resume = _resume(("a", "Built services."))
    assert "Postman" in resume.tools or skills_only_claims(resume, _skills_master(), "en") == []


def test_writer_prompt_forbids_turning_a_skill_into_an_activity() -> None:
    assert "stays in the Tools line" in SYSTEM_PROMPT


# ---- rebuilding one role must not erase the others ---------------------------------------------


def test_rebuilding_one_role_keeps_the_rest_of_the_index_and_report(
    master: Any, roles: list[Role], tmp_path: Path,
) -> None:
    out = tmp_path / "masters"
    build_masters(roles, master, RecordingWriter([content()]), FakeReviewer([review(9)]), out,  # type: ignore[arg-type]
                  renderer=fake_render)
    build_masters([roles[2]], master, RecordingWriter([content(title="REBUILT")]), FakeReviewer([review(7)]), out,  # type: ignore[arg-type]
                  renderer=fake_render, all_roles=roles)
    index = json.loads((out / "roles_index.json").read_text(encoding="utf-8"))
    assert [e["id"] for e in index] == ["bsa", "guidewire-ba", "java", "lead"]  # all four, original order
    report = (out / "ROLES_REPORT.md").read_text(encoding="utf-8")
    for role in roles:
        assert f"## {role.name}" in report
    assert report.count("## Backend Developer (Java)") == 1 and "HR score 7.0/10" in report
    assert json.loads(load_master_content(out, roles[2]))["title"] == "REBUILT"
    assert json.loads(load_master_content(out, roles[0]))["title"] == "Senior Java Developer"  # untouched


# ---- title matching tolerates the variants real postings use -----------------------------------


@pytest.mark.parametrize(("title", "expected"), [
    ("Senior Business System Analyst, Enterprise Platforms", "bsa"),   # system, not systems
    ("Sr Consultant - Business System Analyst", "bsa"),
    ("Systems Analyst (Insurance)", "bsa"),
    ("Lead Software Engineering Manager", "lead"),                       # engineering matches engineer
])
def test_match_role_accepts_singular_plural_and_word_forms(roles: list[Role], title: str, expected: str) -> None:
    assert match_role(title, "", roles).role.id == expected


def test_posting_without_any_signal_uses_the_default_role(roles: list[Role]) -> None:
    default = Role("java-backend", "Backend Developer (Java)", "Backend", "Dev", ("Java Developer",), ("Java",), "x")
    m = match_role("Chef de cuisine", "Cook meals.", [*roles, default])
    assert m.role.id == "java-backend" and "Sin coincidencia" in m.reason and m.runner_up is None


def test_pending_rows_can_leave_rows_without_a_description_untouched(sheet_path: Path) -> None:
    add_row(sheet_path, link="https://a")                                            # no description
    add_row(sheet_path, link="https://b", description="x" * 250)                      # full description
    add_row(sheet_path, link="https://c", description="too short")
    add_row(sheet_path, link="https://d", description="y" * 250, status="Aplicado")   # not pending at all
    assert [r["link"] for r in postular.pending_rows(sheet_path)] == ["https://a", "https://b", "https://c"]
    only = postular.pending_rows(sheet_path, require_description=True)
    assert [r["link"] for r in only] == ["https://b"]


def test_skill_names_keep_slash_names_whole() -> None:
    from job_assistant.cv.checks import _skill_names

    assert _skill_names("Oracle PL/SQL") == ["Oracle PL/SQL"]
    assert _skill_names("HTML/CSS") == ["HTML/CSS"]
    assert _skill_names("Git & GitHub") == ["Git", "GitHub"]
    assert _skill_names("Java (J2EE, JSP)") == ["Java"]


def test_a_slash_tool_listed_only_as_a_skill_is_flagged_by_its_full_name() -> None:
    master = _skills_master()
    master.raw["skills"]["tools_en"] = [*master.raw["skills"]["tools_en"], "Oracle PL/SQL"]
    bad = _resume(("a", "Tuned SQL Server and Oracle PL/SQL data layers."))
    assert skills_only_claims(bad, master, "en") == ["Oracle PL/SQL"]


def test_title_with_a_typographic_apostrophe_matches_the_plain_keyword(roles: list[Role]) -> None:
    curly = "Analyste d’affaires en gestion des demandes"
    m = match_role(curly, "", roles)
    assert m.role.id == "bsa" and "Analyste d'affaires" in m.reason
