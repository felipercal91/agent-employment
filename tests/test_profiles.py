import os
from pathlib import Path
from typing import Any

import pytest

from job_assistant import config
from job_assistant.agents.requirements_agent import (
    LOGISTICS_FLEXIBLE,
    LOGISTICS_STRICT,
    Requirement,
    RequirementsAssessment,
    build_messages,
)
from job_assistant.config import ConfigError, load_settings, profile_name
from job_assistant.cv.master import MasterPlaceholderError, load_master
from job_assistant.profile.candidate import (
    CandidateProfile,
    ProfileNotFoundError,
    load_candidate_profile,
)
from job_assistant.setup_profile import SUBFOLDERS, scaffold

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
ENV_VARS = (
    "JOB_ASSISTANT_PROFILE", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_ENABLED",
    "WRITER_PROVIDER", "REVIEW_PROVIDER", "DATA_DIR", "OUTPUT_DIR", "VECTORSTORE_DIR",
    "TRACKER_PATH", "TRACKER_SHEET", "CLAUDE_MODEL",
)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated project root, with a clean environment (load_dotenv leaks into os.environ)."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    yield tmp_path
    for var in ENV_VARS:
        os.environ.pop(var, None)


def write(path: Path, *lines: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    return path


# ---- profile paths ------------------------------------------------------------------------


def test_default_profile_is_unchanged(root: Path) -> None:
    shared = write(root / ".env", "ANTHROPIC_API_KEY=k", "DATA_DIR=data", "TRACKER_PATH=data/t.xlsx")
    s = load_settings(shared)
    assert s.profile is None
    assert s.data_dir == root / "data" and s.tracker_path == root / "data" / "t.xlsx"
    assert s.output_dir == root / "output" and s.vectorstore_dir == root / "vectorstore"


def test_profile_gets_its_own_folders_and_never_the_shared_paths(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The shared .env holds the FIRST person's paths. A profile must not inherit them.
    shared = write(root / ".env", "DEEPSEEK_API_KEY=ds", "WRITER_PROVIDER=deepseek", "DATA_DIR=data",
                   "OUTPUT_DIR=output", "VECTORSTORE_DIR=vectorstore", "TRACKER_PATH=data/tracker.xlsx",
                   "TRACKER_SHEET=Felipe", "CLAUDE_MODEL=claude-x")
    monkeypatch.setenv("JOB_ASSISTANT_PROFILE", "novia")
    s = load_settings(shared)
    base = root / "profiles" / "novia"
    assert s.profile == "novia"
    assert s.data_dir == base / "data" and s.output_dir == base / "output"
    assert s.vectorstore_dir == base / "vectorstore" and s.tracker_path == base / "data" / "tracker.xlsx"
    assert s.tracker_sheet is None
    # ...while the shared non-path settings do apply
    assert s.deepseek_api_key == "ds" and s.writer_provider == "deepseek" and s.model == "claude-x"


def test_profile_env_overrides_shared_settings(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared = write(root / ".env", "DEEPSEEK_API_KEY=shared-key", "WRITER_PROVIDER=deepseek")
    write(root / "profiles" / "novia" / ".env", "DEEPSEEK_API_KEY=her-own-key", "TRACKER_SHEET=Postular")
    monkeypatch.setenv("JOB_ASSISTANT_PROFILE", "novia")
    s = load_settings(shared)
    assert s.deepseek_api_key == "her-own-key" and s.tracker_sheet == "Postular"


@pytest.mark.parametrize("bad", ["../felipe", "a b", "x/y", "-start", "a" * 40, "ñandú"])
def test_invalid_profile_names_are_rejected(bad: str) -> None:
    with pytest.raises(ConfigError, match="Invalid profile name"):
        profile_name(bad)
    assert profile_name("") is None and profile_name("novia-2") == "novia-2"


# ---- API keys ------------------------------------------------------------------------------


def test_only_a_deepseek_key_is_enough(root: Path) -> None:
    s = load_settings(write(root / ".env", "DEEPSEEK_API_KEY=ds"))
    assert s.anthropic_api_key == "" and s.writer_provider == "deepseek" and s.review_provider == "deepseek"
    from job_assistant.llm import get_chat_model

    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY is not set"):
        get_chat_model(s, provider="anthropic")
    os.environ.pop("DEEPSEEK_API_KEY", None)  # load_settings leaked it from the first .env
    with pytest.raises(ConfigError, match="No API key"):
        load_settings(write(root / "empty.env", "CLAUDE_MODEL=x"))


def test_claude_switched_off_defaults_to_deepseek(root: Path) -> None:
    s = load_settings(write(root / ".env", "ANTHROPIC_API_KEY=k", "DEEPSEEK_API_KEY=ds", "ANTHROPIC_ENABLED=false"))
    assert s.writer_provider == "deepseek"


# ---- placeholders: never produce a CV from the example person ----------------------------------


def test_example_data_is_refused(tmp_path: Path) -> None:
    (tmp_path / "cv_master.toml").write_bytes((EXAMPLES / "cv_master.example.toml").read_bytes())
    (tmp_path / "profile.toml").write_bytes((EXAMPLES / "profile.example.toml").read_bytes())
    with pytest.raises(MasterPlaceholderError, match="still has the example data"):
        load_master(tmp_path)
    with pytest.raises(ProfileNotFoundError, match="still has the example data"):
        load_candidate_profile(tmp_path)
    assert load_master(tmp_path, allow_example=True).contact["name"] == "ALEX SAMPLE"
    assert load_candidate_profile(tmp_path, allow_example=True).name == "Your Name"
    real = (EXAMPLES / "cv_master.example.toml").read_text(encoding="utf-8").replace("ALEX SAMPLE", "MARIA REAL")
    (tmp_path / "cv_master.toml").write_text(real, encoding="utf-8")
    assert load_master(tmp_path).contact["name"] == "MARIA REAL"


# ---- setup -----------------------------------------------------------------------------------


def test_scaffold_creates_without_overwriting(tmp_path: Path) -> None:
    data, tracker = tmp_path / "p" / "data", tmp_path / "p" / "data" / "tracker.xlsx"
    env = tmp_path / "p" / ".env"
    first = scaffold(data, tracker, EXAMPLES, env)
    assert {f"{d}/" for d in SUBFOLDERS} <= set(first.created)
    assert {"profile.toml", "cv_master.toml", "discovery.toml", "tracker.xlsx", ".env (profile)"} <= set(first.created)
    assert first.kept == []
    assert "DATA_DIR" in env.read_text(encoding="utf-8")  # only mentioned as a warning, never set
    assert not any(line.startswith("DATA_DIR=") for line in env.read_text(encoding="utf-8").splitlines())

    (data / "profile.toml").write_text('name = "Edited"' + chr(10), encoding="utf-8")
    second = scaffold(data, tracker, EXAMPLES, env)
    assert second.created == [] and "profile.toml" in second.kept
    assert (data / "profile.toml").read_text(encoding="utf-8") == 'name = "Edited"' + chr(10)  # untouched


def test_setup_command_end_to_end(root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from job_assistant.__main__ import main

    (root / "examples").mkdir()
    for f in EXAMPLES.glob("*.toml"):
        (root / "examples" / f.name).write_bytes(f.read_bytes())
    write(root / ".env", "DEEPSEEK_API_KEY=ds", "DATA_DIR=data")  # the first person's path must be ignored
    monkeypatch.setattr(config, "PROJECT_ROOT", root)
    monkeypatch.setattr("job_assistant.config.load_dotenv", lambda *a, **k: False, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds")
    assert main(["--profile", "novia", "setup"]) == 0
    out = capsys.readouterr()
    assert (root / "profiles" / "novia" / "data" / "cv_master.toml").is_file()
    assert (root / "profiles" / "novia" / "data" / "tracker.xlsx").is_file()
    assert not (root / "data").exists()  # nothing was created in the default person's folder
    assert "[profile: novia]" in out.err and "--profile novia postular" in out.out


# ---- per-profile prompt wording --------------------------------------------------------------------


def profile(flexible: bool) -> CandidateProfile:
    return CandidateProfile(name="M", title="Dev", years_experience=5, skills=["Java"], flexible_location=flexible)


def test_logistics_rule_follows_the_profile() -> None:
    flexible = build_messages("posting", profile(True), [])[0].content
    strict = build_messages("posting", profile(False), [])[0].content
    assert LOGISTICS_FLEXIBLE in flexible and LOGISTICS_STRICT not in flexible
    assert LOGISTICS_STRICT in strict and LOGISTICS_FLEXIBLE not in strict
    assert "Toronto" not in flexible and "__LOGISTICS__" not in flexible
    assert "Open to relocating and any work mode: yes" in build_messages("p", profile(True), [])[1].content


def test_logistics_guard_only_for_flexible_profiles() -> None:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.output_parsers import PydanticOutputParser

    from job_assistant.agents.requirements_agent import RequirementsAgent

    payload = RequirementsAssessment(company="A", role_title="B", location="C", requirements=[
        Requirement(requirement="Must work on-site in the Mississauga office", kind="must_have",
                    status="not_met", evidence="e")]).model_dump_json()

    class JsonFake(FakeListChatModel):
        def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
            return self | PydanticOutputParser(pydantic_object=schema)

    assert RequirementsAgent(JsonFake(responses=[payload])).check("p", profile(True)).should_stop is False
    assert RequirementsAgent(JsonFake(responses=[payload])).check("p", profile(False)).should_stop is True


def test_profile_flexible_location_is_read_from_toml(tmp_path: Path) -> None:
    (tmp_path / "profile.toml").write_text(chr(10).join(
        ['name = "M"', 'title = "Dev"', "years_experience = 3", "flexible_location = true", ""]), encoding="utf-8")
    assert load_candidate_profile(tmp_path).flexible_location is True
    (tmp_path / "profile.toml").write_text(chr(10).join(['name = "M"', 'title = "Dev"', "years_experience = 3", ""]),
                                           encoding="utf-8")
    assert load_candidate_profile(tmp_path).flexible_location is False


def test_professional_field_reaches_the_prompts() -> None:
    from job_assistant.agents.hr_reviewer import build_messages as review_messages
    from job_assistant.agents.resume_writer import build_messages as writer_messages
    from job_assistant.cv.content import ResumeContent
    from job_assistant.cv.master import CVMaster

    base = (EXAMPLES / "cv_master.example.toml").read_text(encoding="utf-8")
    import tomllib

    nurse = CVMaster(contact={"name": "X"}, raw={**tomllib.loads(base), "meta": {"field": "registered nurses"}})
    default = CVMaster(contact={"name": "X"}, raw=tomllib.loads(base.replace('field = "senior software developers"', "")))
    assert nurse.field() == "registered nurses"
    assert "writer for registered nurses in Canada" in writer_messages("jd", nurse, "en")[0].content
    empty = ResumeContent(title="t", summary="s", competencies=[], experience=[], tools=[], methodologies=[])
    assert "screens CVs for registered nurses" in review_messages("jd", empty, nurse, "en", [])[0].content
    assert "{field}" not in review_messages("jd", empty, default, "en", [])[0].content


# ---- compliance profile ---------------------------------------------------------------------------


def test_compliance_starter_files_load_and_render(tmp_path: Path) -> None:
    from job_assistant.cv.content import Bullet, JobBullets, ResumeContent
    from job_assistant.cv.render import build_html

    src = EXAMPLES / "compliance"
    (tmp_path / "cv_master.toml").write_bytes((src / "cv_master.example.toml").read_bytes())
    (tmp_path / "profile.toml").write_bytes((src / "profile.example.toml").read_bytes())
    m = load_master(tmp_path, allow_example=True)
    assert m.field() == "compliance professionals" and m.protect_engineer_title() is False
    assert [j.id for j in m.jobs("fr")] == ["bank", "insurer"] and m.jobs("fr")[1].location == "Santiago, Chili"
    assert m.label("tools", "en", "x") == "Systems & Tools:"
    assert load_candidate_profile(tmp_path, allow_example=True).title == "Compliance Analyst"
    jobs = [JobBullets(job_id=j.id, bullets=[Bullet(lead="", rest=b) for b in j.bullets]) for j in m.jobs("en")]
    content = ResumeContent(title="Compliance Analyst", summary="s", competencies=["AML"], experience=jobs,
                            tools=m.tools("en"), methodologies=m.methodologies("en"))
    html = build_html(m, content, "en")
    assert "Systems &amp; Tools:" in html and "Regulatory Frameworks &amp; Methods:" in html
    assert "Tools &amp; Technology:" not in html
    assert "Systèmes et outils :" in build_html(m, content, "fr")


def test_french_title_rule_is_off_for_compliance(tmp_path: Path) -> None:
    from job_assistant.agents.hr_reviewer import language_rule as review_rule
    from job_assistant.agents.resume_writer import language_rule as writer_rule
    from job_assistant.cv.checks import run_checks
    from job_assistant.cv.content import Bullet, JobBullets, ResumeContent

    src = EXAMPLES / "compliance"
    (tmp_path / "cv_master.toml").write_bytes((src / "cv_master.example.toml").read_bytes())
    compliance = load_master(tmp_path, allow_example=True)
    dev_dir = tmp_path / "dev"
    dev_dir.mkdir()
    (dev_dir / "cv_master.toml").write_bytes((EXAMPLES / "cv_master.example.toml").read_bytes())
    developer = load_master(dev_dir, allow_example=True)

    assert "Développeur" not in writer_rule("fr", compliance.protect_engineer_title(), True)
    assert "Développeur" in writer_rule("fr", developer.protect_engineer_title(), True)
    assert "Ingénieur" not in review_rule("fr", compliance) and "Ingénieur" in review_rule("fr", developer)

    c = ResumeContent(title="Ingénieur en conformité", summary="Ingénieur de la conformité.", competencies=[],
                      experience=[JobBullets(job_id="bank", bullets=[Bullet(lead="A revu", rest="des dossiers.")])],
                      tools=[], methodologies=[])
    flagged = [f.problem for f in run_checks(c, "fr", developer.source_text("fr"), "", 2, True, developer)]
    assert any("Ingénieur" in p for p in flagged)  # developers: still blocked
    quiet = [f.problem for f in run_checks(c, "fr", compliance.source_text("fr"), "", 2, True, compliance)]
    assert not any("Ingénieur" in p for p in quiet)  # compliance: not our rule to enforce


def test_compliance_vocabulary_blocks_invented_credentials(tmp_path: Path) -> None:
    from job_assistant.cv.checks import in_progress_certification_claims, run_checks, ungrounded_technologies
    from job_assistant.cv.content import ResumeContent
    from job_assistant.scoring.fit import extract_keywords

    (tmp_path / "cv_master.toml").write_bytes((EXAMPLES / "compliance" / "cv_master.example.toml").read_bytes())
    m = load_master(tmp_path, allow_example=True)
    src = m.source_text("en")
    assert {"AML", "KYC", "FINTRAC", "Actimize", "World-Check"} <= set(extract_keywords(src))

    def cv(summary: str) -> ResumeContent:
        return ResumeContent(title="Compliance Analyst", summary=summary, competencies=[], experience=[],
                             tools=[], methodologies=[])

    invented = cv("CAMS certified analyst with OSFI and GDPR experience, using Fircosoft and Archer.")
    assert set(ungrounded_technologies(invented, src)) >= {"OSFI", "Privacy / data protection", "FircoSoft", "GRC tools"}
    assert "CAMS" not in ungrounded_technologies(invented, src)  # CAMS is in her facts, but only as in progress
    assert in_progress_certification_claims(invented, src) == ["CAMS"]  # ...so writing "CAMS certified" is blocked
    assert any(f.severity == "blocker" for f in run_checks(invented, "en", src, "", 2, master=m))
    honest = cv("Compliance analyst with AML and KYC experience, filing reports with FINTRAC, using Actimize.")
    assert ungrounded_technologies(honest, src) == []
    caveated = cv("Compliance analyst working toward CAMS (in progress), with AML and KYC experience.")
    assert in_progress_certification_claims(caveated, src) == []  # saying it is in progress is fine
    held = src.replace("CAMS (in progress)", "CAMS")
    assert in_progress_certification_claims(invented, held) == []  # once she holds it, claiming it is fine


def test_setup_with_the_compliance_template(root: Path, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    from job_assistant.__main__ import main

    shutil_src = EXAMPLES
    (root / "examples" / "compliance").mkdir(parents=True)
    for f in shutil_src.glob("*.toml"):
        (root / "examples" / f.name).write_bytes(f.read_bytes())
    for f in (shutil_src / "compliance").glob("*.toml"):
        (root / "examples" / "compliance" / f.name).write_bytes(f.read_bytes())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds")
    assert main(["--profile", "novia", "setup", "--template", "compliance"]) == 0
    data = root / "profiles" / "novia" / "data"
    assert 'field = "compliance professionals"' in (data / "cv_master.toml").read_text(encoding="utf-8")
    assert "Compliance Analyst" in (data / "profile.toml").read_text(encoding="utf-8")
    assert "compliance" in (data / "discovery.toml").read_text(encoding="utf-8")
    capsys.readouterr()
    os.environ.pop("JOB_ASSISTANT_PROFILE", None)  # one process per command in real use; here it would stick
    assert main(["setup"]) == 0  # the default person's setup uses the generic examples, not compliance
    default_profile = (root / "data" / "profile.toml").read_text(encoding="utf-8")
    assert "Java" in default_profile and "Compliance Analyst" not in default_profile


# ---- a person with an English-only CV --------------------------------------------------------------


def english_only_master(tmp_path: Path) -> Any:
    import tomllib

    from job_assistant.cv.master import CVMaster

    raw = tomllib.loads((EXAMPLES / 'cv_master.example.toml').read_text(encoding='utf-8'))
    for job in raw['experience']:
        for key in [k for k in job if k.endswith('_fr')]:
            del job[key]
    del raw['education']['fr'], raw['summary_facts']['fr']
    return CVMaster(contact=raw['contact'], raw=raw)


def test_has_language(tmp_path: Path) -> None:
    import tomllib

    from job_assistant.cv.master import CVMaster

    full = CVMaster(contact={}, raw=tomllib.loads((EXAMPLES / 'cv_master.example.toml').read_text(encoding='utf-8')))
    assert full.has_language('en') and full.has_language('fr')
    only_en = english_only_master(tmp_path)
    assert only_en.has_language('en') and not only_en.has_language('fr')


def test_french_posting_falls_back_to_english_without_a_french_cv(tmp_path: Path) -> None:
    from datetime import date

    from job_assistant.agents.hr_reviewer import HRReview, Scores
    from job_assistant.agents.requirements_agent import Requirement
    from job_assistant.cv import postular
    from job_assistant.cv.content import Bullet, JobBullets, ResumeContent
    from job_assistant.cv.jd import JobDescription
    from job_assistant.tracker.template import create_tracker_template

    master = english_only_master(tmp_path)
    path = create_tracker_template(tmp_path / 'tracker.xlsx')
    postular.ensure_sheet(path)
    fr_jd = ('Nous cherchons un analyste en conformité avec de l' + chr(39) + 'expérience pour notre équipe dans les services '
             'financiers. Vous avez une maîtrise des politiques et des contrôles dans les institutions. ') * 4
    postular.table(path).append({'link': 'https://x/fr', 'company': 'Desjardins', 'description': fr_jd})
    [row] = postular.pending_rows(path)

    class Req:
        def check(self, *a: Any) -> RequirementsAssessment:
            return RequirementsAssessment(company='Desjardins', role_title='Analyste', location='Montréal, QC',
                requirements=[Requirement(requirement='r', kind='must_have', status='met', evidence='e')])

    class Writer:
        langs: list[str] = []

        def write(self, jd: str, m: Any, lang: str, *a: Any, **k: Any) -> ResumeContent:
            Writer.langs.append(lang)
            return ResumeContent(title='Analyst', summary='s', competencies=['c'], tools=m.tools(lang),
                methodologies=m.methodologies(lang),
                experience=[JobBullets(job_id=j.id, bullets=[Bullet(lead='', rest=j.bullets[0])]) for j in m.jobs(lang)])

    class Reviewer:
        def review(self, *a: Any) -> HRReview:
            return HRReview(scores=Scores(requirements_coverage=9, truthfulness=9, human_tone=9, clarity=9,
                ats_keywords=9, candidate_fit=9), issues=[], gaps_cannot_fix=[], likely_interview_questions=[],
                notes_for_candidate='ok')

    def fake_render(html: str, out: Path) -> int:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding='utf-8')
        return 2

    out = postular.process_row(row, path=path, master=master,
        profile=CandidateProfile(name='M', title='T', years_experience=5, skills=[]),
        requirements=Req(), writer=Writer(), reviewer=Reviewer(), output_dir=tmp_path / 'out',
        renderer=fake_render, today=date(2026, 10, 4))  # type: ignore[arg-type]
    assert Writer.langs == ['en']  # the posting is French, the CV could not be
    assert out.status == postular.STATUS_READY and out.cv_path is not None and 'Resume' in out.cv_path.name
    saved = postular.table(path).rows()[0]
    assert saved['language'] == 'en' and saved['hr_notes'].startswith('La vacante pide francés')


# ---- per-person standing rules: fixed tools, banned words, base CV ------------------------------------


def master_with(meta: str, skills_extra: str = "") -> Any:
    import tomllib

    from job_assistant.cv.master import CVMaster

    raw = tomllib.loads((EXAMPLES / "cv_master.example.toml").read_text(encoding="utf-8"))
    raw["meta"] = tomllib.loads(meta)["meta"] if meta else raw["meta"]
    if skills_extra:
        raw["skills"].update(tomllib.loads(skills_extra)["skills"])
    return CVMaster(contact=raw["contact"], raw=raw)


def test_keep_all_tools_prints_every_tool_in_order() -> None:
    from job_assistant.cv.content import Bullet, JobBullets, ResumeContent, normalize

    base = master_with("", "[skills]" + chr(10) + "keep_all_tools = true")
    assert base.keep_all_tools() is True
    trimmed = ResumeContent(
        title="t", summary="s", competencies=[], tools=["Docker"], methodologies=["Agile"],
        experience=[JobBullets(job_id=j.id, bullets=[Bullet(lead="", rest="x")]) for j in base.jobs("en")])
    fixed, _ = normalize(trimmed, base, "en")
    assert fixed.tools == base.tools("en") and fixed.methodologies == ["Agile"]
    normal = master_with("")
    assert normal.keep_all_tools() is False
    assert normalize(trimmed, normal, "en")[0].tools == ["Docker"]  # others keep the writer's selection


def test_banned_terms_and_standing_rules() -> None:
    from job_assistant.agents.resume_writer import build_messages as writer_messages
    from job_assistant.agents.resume_writer import standing_rules
    from job_assistant.cv.checks import banned_terms_found, run_checks
    from job_assistant.cv.content import ResumeContent

    meta = chr(10).join(["[meta]", 'field = "x"',
                         'extra_rules = ["Never state a language level."]',
                         'banned_terms = ["proficiency", "native", "fluent"]'])
    m = master_with(meta)
    assert m.extra_rules() == ["Never state a language level."] and m.banned_terms() == ["proficiency", "native", "fluent"]
    assert standing_rules(master_with("")) == ""
    assert "Never state a language level." in writer_messages("jd", m, "en")[0].content
    assert "Standing rules" not in writer_messages("jd", master_with(""), "en")[0].content

    def cv(summary: str) -> ResumeContent:
        return ResumeContent(title="t", summary=summary, competencies=["native tools"], experience=[],
                             tools=[], methodologies=[])

    found = banned_terms_found(cv("Fluent English and Spanish proficiency."), m.banned_terms())
    assert set(found) == {"proficiency", "native", "fluent"}      # "native" comes from the competencies
    assert "native" in banned_terms_found(cv("ok"), m.banned_terms())               # whole word, also in competencies
    assert banned_terms_found(ResumeContent(title="t", summary="Nativeland is fine", competencies=[], experience=[],
                                            tools=[], methodologies=[]), ["native"]) == []   # not inside another word
    findings = run_checks(cv("Fluent in English."), "en", m.source_text("en"), "", 2, master=m)
    assert any(f.severity == "blocker" and "does not want" in f.problem for f in findings)
    clean = ResumeContent(title="t", summary="Bilingual in Spanish and English.", competencies=[], experience=[],
                          tools=[], methodologies=[])
    assert not any("does not want" in f.problem for f in run_checks(clean, "en", m.source_text("en"), "", 2, master=m))


def test_split_lead_cuts_at_natural_boundaries() -> None:
    from job_assistant.cv.base import split_lead

    cases = {
        "Demonstrated high adaptability and strong time-management skills in a fast-paced environment.":
            ("Demonstrated high adaptability", "and strong time-management skills in a fast-paced environment."),
        "Managed inventory preparation, resource allocation, and team coordination.":
            ("Managed inventory preparation,", "resource allocation, and team coordination."),
        "Prepared, analyzed, and monitored commercial campaigns, generating tracking reports.":
            ("Prepared, analyzed, and monitored commercial campaigns,", "generating tracking reports."),
        "Conducted comprehensive process surveys across Accounting and Legal departments.":
            ("Conducted comprehensive process surveys", "across Accounting and Legal departments."),
    }
    for text, (lead, rest) in cases.items():
        b = split_lead(text)
        assert (b.lead, b.rest) == (lead, rest), text
        assert f"{b.lead} {b.rest}" == text  # nothing is lost or reworded
    short = split_lead("Built things.")
    assert (short.lead, short.rest) == ("Built things.", "")


def test_base_cv_uses_only_the_master(tmp_path: Path) -> None:
    from job_assistant.cv.base import base_content
    from job_assistant.cv.render import build_html

    m = master_with("", "[skills]" + chr(10) + "keep_all_tools = true")
    content = base_content(m, "en")
    assert content.summary == m.summary_facts("en") and content.tools == m.tools("en")
    originals = [b for j in m.jobs("en") for b in j.bullets]
    rebuilt = [f"{b.lead} {b.rest}".strip() for j in content.experience for b in j.bullets]
    assert rebuilt == originals                                   # every bullet, verbatim and in order
    html = build_html(m, content, "en")
    assert "Methodologies:" not in html                           # not printed twice (they are the competencies)
    assert "<b>Languages:</b> English (C1), French (B2), Spanish (native)" in html


def test_real_anny_master_if_present() -> None:
    real = Path(__file__).resolve().parent.parent / "profiles" / "anny" / "data"
    if not (real / "cv_master.toml").is_file():
        pytest.skip("no personal profile here")
    m = load_master(real)
    assert m.keep_all_tools() and m.tools("en")[-1] == "Bizagi" and not any("ERP" in t for t in m.tools("en"))
    assert m.languages("en") == "Spanish, English"
    assert "native" in m.banned_terms() and "Native" not in m.summary_facts("en")
