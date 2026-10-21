from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel

from job_assistant.__main__ import _is_account_error, _short
from job_assistant.config import ConfigError, load_settings
from job_assistant.llm import JsonModeStructured, get_chat_model, structured_llm
from job_assistant.usage import (
    Usage,
    UsageGroup,
    UsageTracker,
    is_deepseek_peak,
    price_per_mtok,
)


def test_account_errors_stop_the_run() -> None:
    no_credit = Exception(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'Your credit balance is too low to access the Anthropic API.'}}"
    )
    assert _is_account_error(no_credit)
    assert _short(no_credit) == "Your credit balance is too low to access the Anthropic API."
    assert _is_account_error(Exception("Error code: 402 - Insufficient Balance"))
    assert not _is_account_error(ValueError("bad JSON in model output"))


# ---- usage and cost --------------------------------------------------------------


def feed(tracker: UsageTracker, *calls: tuple[int, int]) -> None:
    for tokens_in, tokens_out in calls:
        message = AIMessage(content="x", usage_metadata={
            "input_tokens": tokens_in, "output_tokens": tokens_out, "total_tokens": tokens_in + tokens_out})
        tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="no usage"))]]))


def test_usage_tracker_adds_up_calls_and_prices() -> None:
    claude = UsageTracker("claude-sonnet-5-5")
    feed(claude, (1000, 200), (3000, 800))
    usage = claude.snapshot()
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (2, 4000, 1000)
    assert usage.cost == pytest.approx((4000 * 2 + 1000 * 10) / 1_000_000)
    assert "≈ $0.018" in usage.describe()

    unknown = UsageTracker("some-other-model")
    feed(unknown, (10, 10))
    assert unknown.snapshot().cost is None and "≈" not in unknown.snapshot().describe()

    flash = UsageTracker("deepseek-flash")
    feed(flash, (1_000_000, 1_000_000))
    assert flash.snapshot().cost in (pytest.approx(1.50), pytest.approx(0.75))  # peak or off-peak

    group = UsageGroup([claude, flash])
    assert group.snapshot().calls == 3 and group.snapshot().cost is not None
    assert (group.snapshot() - usage).calls == 1
    assert (usage + unknown.snapshot()).cost is None
    assert Usage().describe().startswith("0 calls")


def test_deepseek_peak_hours() -> None:
    def utc(day: int, hour: int) -> datetime:  # 2026-10-05 is a Monday
        return datetime(2026, 10, day, hour, 30, tzinfo=timezone.utc)

    assert is_deepseek_peak(utc(5, 2)) and is_deepseek_peak(utc(5, 7))
    assert not is_deepseek_peak(utc(5, 12)) and not is_deepseek_peak(utc(5, 5))
    assert not is_deepseek_peak(utc(3, 2))  # Saturday
    assert price_per_mtok("deepseek-flash", utc(5, 12)) == (0.15, 0.60)
    assert price_per_mtok("deepseek-flash", utc(5, 2)) == (0.30, 1.20)
    assert price_per_mtok("nope") is None


def test_default_rounds_is_two() -> None:
    from job_assistant.cv.pipeline import DEFAULT_MAX_ROUNDS

    assert DEFAULT_MAX_ROUNDS == 2


# ---- providers and DeepSeek JSON mode ----------------------------------------------


PROVIDER_VARS = ('ANTHROPIC_API_KEY', 'DEEPSEEK_API_KEY', 'WRITER_PROVIDER', 'REVIEW_PROVIDER',
                 'DEEPSEEK_MODEL', 'DEEPSEEK_THINKING')


def settings_from(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> Any:
    import os

    for var in PROVIDER_VARS:
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / 'providers.env'
    env.write_text('ANTHROPIC_API_KEY=k' + chr(10) + body, encoding='utf-8')
    try:
        return load_settings(env)
    finally:  # load_dotenv writes into os.environ; do not leak it into other tests
        for var in PROVIDER_VARS:
            os.environ.pop(var, None)


def test_provider_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = settings_from(tmp_path, monkeypatch, "")
    assert (s.writer_provider, s.review_provider, s.deepseek_api_key) == ("anthropic", "anthropic", None)
    assert "ds-secret" not in repr(settings_from(tmp_path, monkeypatch, "DEEPSEEK_API_KEY=ds-secret" + chr(10)))

    s = settings_from(tmp_path, monkeypatch,
                      "DEEPSEEK_API_KEY=ds" + chr(10) + "REVIEW_PROVIDER=DeepSeek" + chr(10))
    assert s.review_provider == "deepseek" and s.writer_provider == "anthropic"
    assert s.deepseek_model == "deepseek-flash" and s.deepseek_thinking is False

    with pytest.raises(ConfigError, match="needs DEEPSEEK_API_KEY"):
        settings_from(tmp_path, monkeypatch, "REVIEW_PROVIDER=deepseek" + chr(10))
    with pytest.raises(ConfigError, match="must be"):
        settings_from(tmp_path, monkeypatch, "WRITER_PROVIDER=gpt" + chr(10))


def test_get_chat_model_per_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from langchain_anthropic import ChatAnthropic
    from langchain_deepseek import ChatDeepSeek

    s = settings_from(tmp_path, monkeypatch, "DEEPSEEK_API_KEY=ds" + chr(10))
    assert isinstance(get_chat_model(s), ChatAnthropic)
    ds = get_chat_model(s, provider="deepseek")
    assert isinstance(ds, ChatDeepSeek) and ds.model_name == "deepseek-flash"
    assert ds.extra_body == {"thinking": {"type": "disabled"}}  # reasoning off by default
    assert ds.request_timeout == 150.0 and ds.max_retries == 4  # a stuck request fails fast and is retried
    assert isinstance(structured_llm(ds, Answer), JsonModeStructured)
    with pytest.raises(ConfigError):
        get_chat_model(settings_from(tmp_path, monkeypatch, ""), provider="deepseek")


class Answer(BaseModel):
    city: str
    notes: list[str]


def test_json_mode_validates_and_retries() -> None:
    llm = FakeListChatModel(responses=["", "not json at all", "```json" + chr(10) + '{"city": "Toronto"}' + chr(10) + "```",
                              '{"city": "Toronto", "notes": ["big"]}'])
    result = JsonModeStructured(llm, Answer, retries=3).invoke(
        [SystemMessage("Be brief."), HumanMessage("Give json.")])
    assert result == Answer(city="Toronto", notes=["big"])  # 3 bad replies corrected, 4th accepted


def test_json_mode_strips_fences_and_gives_up() -> None:
    fenced = FakeListChatModel(responses=["```json" + chr(10) + '{"city": "Montreal", "notes": []}' + chr(10) + "```"])
    assert JsonModeStructured(fenced, Answer).invoke([HumanMessage("x")]).city == "Montreal"
    broken = FakeListChatModel(responses=["{oops"])
    with pytest.raises(ValueError, match="did not return valid JSON for Answer"):
        JsonModeStructured(broken, Answer, retries=1).invoke([HumanMessage("x")])


def test_json_mode_puts_schema_in_the_system_prompt() -> None:
    structured = JsonModeStructured(FakeListChatModel(responses=["{}"]), Answer)
    system, human = structured._with_instruction([SystemMessage("Be brief."), HumanMessage("hi")])
    assert system.content.startswith("Be brief.") and "JSON schema" in system.content and '"city"' in system.content
    assert human.content == "hi"
    prepended = structured._with_instruction([HumanMessage("hi")])
    assert isinstance(prepended[0], SystemMessage) and len(prepended) == 2


def test_claude_api_can_be_switched_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    def load(extra: str) -> Any:
        for var in ("ANTHROPIC_ENABLED", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "WRITER_PROVIDER", "REVIEW_PROVIDER"):
            monkeypatch.delenv(var, raising=False)
        env = tmp_path / "switch.env"
        env.write_text("ANTHROPIC_API_KEY=k" + chr(10) + "DEEPSEEK_API_KEY=d" + chr(10) + extra, encoding="utf-8")
        try:
            return load_settings(env)
        finally:
            for var in ("ANTHROPIC_ENABLED", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "WRITER_PROVIDER", "REVIEW_PROVIDER"):
                os.environ.pop(var, None)

    assert load("").anthropic_enabled is True                         # default: allowed
    off = load("ANTHROPIC_ENABLED=false" + chr(10))
    assert off.anthropic_enabled is False
    with pytest.raises(ConfigError, match="Claude API is disabled"):
        get_chat_model(off)                                           # any default call is refused
    with pytest.raises(ConfigError, match="Claude API is disabled"):
        get_chat_model(off, provider="anthropic", effort="low")
    assert get_chat_model(off, provider="deepseek") is not None       # DeepSeek keeps working
    from job_assistant.llm import check_connection

    with pytest.raises(ConfigError):
        check_connection(off)                                         # no network call is made
