from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from job_assistant.agents.requirements_agent import (
    Requirement,
    RequirementsAssessment,
    build_messages,
)
from job_assistant.pipeline import evaluate_posting
from job_assistant.profile.candidate import (
    CandidateProfile,
    ProfileNotFoundError,
    load_candidate_profile,
)
from job_assistant.scoring.fit import extract_keywords, extract_years_required, score_fit

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def profile() -> CandidateProfile:
    return CandidateProfile(
        name="Test",
        title="Senior Java Backend Developer",
        years_experience=13,
        skills=["Java 17", "Spring Boot", "Microservices", "Kafka", "PostgreSQL", "REST APIs", "Docker"],
    )


# ---- profile loading -------------------------------------------------------


def test_load_example_profile(tmp_path: Path) -> None:
    (tmp_path / "profile.toml").write_bytes((EXAMPLES / "profile.example.toml").read_bytes())
    loaded = load_candidate_profile(tmp_path)
    assert loaded.years_experience == 13
    assert "Spring Boot" in loaded.skills
    assert loaded.certifications == []


def test_missing_profile_raises(tmp_path: Path) -> None:
    with pytest.raises(ProfileNotFoundError):
        load_candidate_profile(tmp_path)


# ---- keyword extraction ----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected", "absent"),
    [
        ("Java 17 with Spring Boot", {"Java", "Spring Boot"}, set()),
        ("JavaScript and TypeScript", {"JavaScript", "TypeScript"}, {"Java"}),
        ("Apache Kafka, K8s, CI/CD", {"Kafka", "Kubernetes", "CI/CD"}, set()),
        ("C# and .NET services", {"C#/.NET"}, set()),
        ("Node.js backend", {"Node.js"}, set()),
        ("Java/Kotlin in Agile/Scrum teams", {"Java", "Kotlin", "Agile/Scrum"}, set()),
        ("Ingest logs into a warehouse", set(), {"Go", "Git"}),
    ],
)
def test_extract_keywords(text: str, expected: set[str], absent: set[str]) -> None:
    found = set(extract_keywords(text))
    assert expected <= found
    assert not (absent & found)


@pytest.mark.parametrize(
    ("text", "years"),
    [
        ("8+ years of experience", 8),
        ("5-7 years in backend", 5),
        ("Minimum 10 yrs", 10),
        ("au moins 6 ans d'expérience", 6),
        ("Founded 120 years ago, we need 3+ years", 3),
        ("No experience requirement", None),
    ],
)
def test_extract_years(text: str, years: int | None) -> None:
    assert extract_years_required(text) == years


# ---- fit score -------------------------------------------------------------


def test_fit_on_example_posting(profile: CandidateProfile) -> None:
    text = (EXAMPLES / "postings" / "senior_java_toronto.txt").read_text(encoding="utf-8")
    fit = score_fit(text, profile)
    assert {"Java", "Spring Boot", "Kafka", "PostgreSQL"} <= set(fit.matched)
    assert {"Kubernetes", "AWS", "Terraform"} <= set(fit.missing)
    assert fit.years_required == 8 and fit.years_ok
    assert fit.score == round(100 * len(fit.matched) / len(fit.found))


def test_fit_years_penalty(profile: CandidateProfile) -> None:
    junior = CandidateProfile(**{**profile.__dict__, "years_experience": 3})
    text = "Java and Spring Boot, 8+ years required"
    assert score_fit(text, profile).score == 100
    low = score_fit(text, junior)
    assert not low.years_ok and low.score == 85


def test_fit_no_keywords_is_neutral(profile: CandidateProfile) -> None:
    fit = score_fit("Great team, free snacks.", profile)
    assert fit.score == 100 and fit.found == []


# ---- requirements check ----------------------------------------------------


def assessment(*reqs: tuple[str, str]) -> RequirementsAssessment:
    return RequirementsAssessment(
        company="Acme",
        role_title="Role",
        requirements=[
            Requirement(requirement=f"req {i}", kind=kind, status=status, evidence="e")  # type: ignore[arg-type]
            for i, (kind, status) in enumerate(reqs)
        ],
    )


def test_stop_only_on_must_have_not_met() -> None:
    assert assessment(("must_have", "not_met")).should_stop
    assert not assessment(("nice_to_have", "not_met"), ("must_have", "unclear")).should_stop
    a = assessment(("must_have", "met"), ("must_have", "unclear"))
    assert [r.requirement for r in a.unclear_must_haves] == ["req 1"]


def test_build_messages_includes_profile_and_posting(profile: CandidateProfile) -> None:
    _, human = build_messages("Need CPO certification", profile, [])
    assert "Certifications: (none listed)" in human.content
    assert "(no documents indexed)" in human.content
    assert "Need CPO certification" in human.content


class FakeAgent:
    def __init__(self, result: RequirementsAssessment) -> None:
        self.result = result
        self.calls: list[Any] = []

    def check(self, posting_text: str, profile: CandidateProfile, excerpts: Any = None) -> RequirementsAssessment:
        self.calls.append(posting_text)
        return self.result


def test_pipeline_decision(profile: CandidateProfile) -> None:
    stop = evaluate_posting("CPO required", profile, FakeAgent(assessment(("must_have", "not_met"))))  # type: ignore[arg-type]
    assert stop.decision == "stop"
    go = evaluate_posting("Java", profile, FakeAgent(assessment(("must_have", "met"))))  # type: ignore[arg-type]
    assert go.decision == "continue" and go.fit.matched == ["Java"]


def test_requirements_agent_parses_structured_json(profile: CandidateProfile) -> None:
    from job_assistant.agents.requirements_agent import RequirementsAgent

    payload = assessment(("must_have", "not_met")).model_dump_json()

    class JsonFake(FakeListChatModel):
        def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
            from langchain_core.output_parsers import PydanticOutputParser

            return self | PydanticOutputParser(pydantic_object=schema)

    result = RequirementsAgent(JsonFake(responses=[payload])).check("posting", profile)
    assert result.should_stop
