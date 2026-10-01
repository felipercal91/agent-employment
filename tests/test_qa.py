from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from job_assistant.agents.qa_agent import (
    DraftAnswer,
    FormAnswers,
    QAAgent,
    build_messages,
    check_premises,
    flag_counts,
    merge_premise_issues,
    parse_questions,
    render_answers_markdown,
)
from job_assistant.profile.candidate import CandidateProfile, load_candidate_profile
from job_assistant.profile.loaders import SourceDocument

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def profile(reason: str = "layoff", **situation: str) -> CandidateProfile:
    return CandidateProfile(
        name="T", title="Senior Java Developer", years_experience=13, skills=["Java"],
        situation={"employment_end_reason": reason, **situation},
    )


def test_profile_situation_loaded(tmp_path: Path) -> None:
    (tmp_path / "profile.toml").write_bytes((EXAMPLES / "profile.example.toml").read_bytes())
    loaded = load_candidate_profile(tmp_path)
    assert loaded.situation["employment_end_reason"] == "layoff"
    assert "employment_end_reason: layoff" in loaded.as_prompt_text()


@pytest.mark.parametrize(
    ("question", "flagged"),
    [
        ("Suite à votre démission, quelle est votre date de disponibilité?", True),
        ("Why did you resign from your last position?", True),
        ("¿Por qué renunció a su último empleo?", True),
        ("Were you ever fired or terminated for cause?", True),
        ("May we contact your current employer?", True),
        ("Are you currently employed?", False),  # asks, does not assume
        ("Why are you looking for a new role?", False),
        ("What are your salary expectations?", False),
    ],
)
def test_premise_rules_for_layoff(question: str, flagged: bool) -> None:
    assert bool(check_premises([question], profile("layoff"))) is flagged


def test_resignation_question_is_fine_when_candidate_resigned() -> None:
    assert check_premises(["Why did you resign?"], profile("resignation")) == []


def test_no_premise_rules_without_situation() -> None:
    p = CandidateProfile(name="", title="Dev", years_experience=1, skills=[])
    assert check_premises(["Suite à votre démission?"], p) == []


def test_parse_questions() -> None:
    assert parse_questions("Q1. First?\n\n2) Second\nline two?\n\n- Third?") == [
        "First?", "Second\nline two?", "Third?",
    ]
    assert parse_questions("One?\nTwo?") == ["One?", "Two?"]
    assert parse_questions("  ") == []
    assert len(parse_questions((EXAMPLES / "forms" / "formulaire_quebec.txt").read_text(encoding="utf-8"))) == 5


def answer(q: str, **kw: Any) -> DraftAnswer:
    base = {"question": q, "answer": "a", "premise_issue": "", "needs_your_input": "",
            "needs_your_decision": False}
    return DraftAnswer(**{**base, **kw})


def test_merge_keeps_deterministic_issue_when_model_omits_it() -> None:
    issues = check_premises(["Q0?", "Suite à votre démission?"], profile())
    merged = merge_premise_issues(FormAnswers(answers=[answer("Q0?"), answer("Suite…")]), issues)
    assert merged.answers[0].premise_issue == ""
    assert "layoff" in merged.answers[1].premise_issue


def test_build_messages_lists_known_issues_and_french_rule() -> None:
    qs = ["Suite à votre démission?"]
    system, human = build_messages(qs, profile(), [SourceDocument("Salary: 120k", "qa_history/a.md", "qa_history")],
                                   check_premises(qs, profile()), "fr")
    assert "Développeur" in system.content
    assert "Q1: Question assumes you resigned" in human.content
    assert "Salary: 120k" in human.content


def test_render_and_counts() -> None:
    answers = FormAnswers(answers=[
        answer("Salary?", needs_your_input="salary range", answer="[YOUR SALARY RANGE]"),
        answer("Consent?", needs_your_decision=True),
        answer("Démission?", premise_issue="It was a layoff"),
    ])
    md = render_answers_markdown(answers, "Acme", "Dev", "fr")
    assert "PENDING YOUR APPROVAL" in md and "Premise mismatch" in md and "Needs your input" in md
    assert flag_counts(answers) == {"premise": 1, "input": 1, "decision": 1}


def test_agent_end_to_end_with_fake_llm() -> None:
    payload = FormAnswers(answers=[answer("Q1. Suite à votre démission?")]).model_dump_json()

    class JsonFake(FakeListChatModel):
        def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
            from langchain_core.output_parsers import PydanticOutputParser

            return self | PydanticOutputParser(pydantic_object=schema)

    result, issues = QAAgent(JsonFake(responses=[payload])).answer(
        ["Suite à votre démission?"], profile(), [], "fr"
    )
    assert len(issues) == 1
    assert "layoff" in result.answers[0].premise_issue  # model omitted it; merged in
    assert result.answers[0].question == "Suite à votre démission?"  # original wording kept
