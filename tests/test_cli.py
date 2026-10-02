from job_assistant.__main__ import _is_account_error, _short


def test_account_errors_stop_the_run() -> None:
    no_credit = Exception(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'Your credit balance is too low to access the Anthropic API.'}}"
    )
    assert _is_account_error(no_credit)
    assert _short(no_credit) == "Your credit balance is too low to access the Anthropic API."
    assert not _is_account_error(ValueError("bad JSON in model output"))


def test_usage_tracker_adds_up_calls_and_prices() -> None:
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from job_assistant.usage import Usage, UsageTracker

    tracker = UsageTracker()
    for tokens_in, tokens_out in ((1000, 200), (3000, 800)):
        message = AIMessage(content="x", usage_metadata={
            "input_tokens": tokens_in, "output_tokens": tokens_out, "total_tokens": tokens_in + tokens_out})
        tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="no usage"))]]))

    usage = tracker.snapshot()
    assert usage == Usage(calls=2, input_tokens=4000, output_tokens=1000)
    assert usage.cost_usd("claude-sonnet-5-5") == (4000 * 2 + 1000 * 10) / 1_000_000
    assert usage.cost_usd("some-other-model") is None
    assert (usage - Usage(1, 1000, 200)).input_tokens == 3000
    assert "≈ $0.018" in usage.describe("claude-sonnet-5-5")


def test_default_rounds_is_two() -> None:
    from job_assistant.cv.pipeline import DEFAULT_MAX_ROUNDS

    assert DEFAULT_MAX_ROUNDS == 2
