from job_assistant.__main__ import _is_account_error, _short


def test_account_errors_stop_the_run() -> None:
    no_credit = Exception(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'Your credit balance is too low to access the Anthropic API.'}}"
    )
    assert _is_account_error(no_credit)
    assert _short(no_credit) == "Your credit balance is too low to access the Anthropic API."
    assert not _is_account_error(ValueError("bad JSON in model output"))
