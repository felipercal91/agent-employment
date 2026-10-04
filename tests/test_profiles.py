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
