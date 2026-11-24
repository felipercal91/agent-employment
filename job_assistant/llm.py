"""Single place where chat models are built, so every agent shares the same configuration.

Two providers are supported: Anthropic (default) and DeepSeek (optional, cheaper). Each agent
role can use a different one (see WRITER_PROVIDER / REVIEW_PROVIDER in .env).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, ValidationError

from job_assistant.config import ConfigError, Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

PROVIDERS = ("anthropic", "deepseek")
DEEPSEEK_TIMEOUT_SECONDS = 150.0
DEEPSEEK_MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 2.0


def _is_transient_error(exc: Exception) -> bool:
    """True for network, timeout, rate-limit, or 5xx errors worth retrying with backoff."""
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout,
                        httpx.RemoteProtocolError, ConnectionError, TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    if any(k in name for k in ("connection", "timeout", "ratelimit", "internalserver", "serviceunavailable")):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in (429, 500, 502, 503, 504):
        return True
    msg = str(exc).lower()
    return any(k in msg for k in ("connection error", "timed out", "read timeout", "temporarily unavailable", "bad gateway", "502", "503", "504"))


def invoke_with_backoff(
    fn: Callable[[], Any],
    *,
    max_attempts: int = DEEPSEEK_MAX_RETRIES,
    base_delay: float = BACKOFF_BASE_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Any:
    """Call `fn()` retrying transient connection/5xx errors with exponential backoff."""
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if not _is_transient_error(e) or attempt >= max_attempts:
                raise
            last_err = e
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Transient LLM error on attempt %d/%d (%s: %s). Retrying in %.1fs...",
                attempt, max_attempts, type(e).__name__, str(e)[:120], delay,
            )
            sleep_fn(delay)
    assert last_err is not None
    raise last_err


def get_chat_model(
    settings: Settings,
    max_tokens: int = 8192,
    callbacks: list[BaseCallbackHandler] | None = None,
    effort: str | None = None,
    provider: str = "anthropic",
    thinking: bool | None = None,
    model: str | None = None,
) -> BaseChatModel:
    """`effort` (low/medium/high) sets how much Claude reasons; None = the model's default.

    For DeepSeek the reasoning mode is a setting (`DEEPSEEK_THINKING`), not an effort level.
    """
    if provider == "deepseek":
        if not settings.deepseek_api_key:
            raise ConfigError("DEEPSEEK_API_KEY is not set in .env.")
        reasoning = settings.deepseek_thinking if thinking is None else thinking
        return ChatDeepSeek(
            model=model or settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            max_tokens=max_tokens,
            callbacks=callbacks,
            extra_body=None if reasoning else {"thinking": {"type": "disabled"}},
            # A stalled request used to hang for the SDK default (10 minutes) before failing the row.
            # A normal call takes 15-60 seconds: give up on a stuck one early and retry it.
            timeout=DEEPSEEK_TIMEOUT_SECONDS,
            max_retries=DEEPSEEK_MAX_RETRIES,
        )
    if provider != "anthropic":
        raise ConfigError(f"Unknown provider '{provider}' (use one of: {', '.join(PROVIDERS)}).")
    if not settings.anthropic_enabled:
        raise ConfigError(
            "The Claude API is disabled (ANTHROPIC_ENABLED=false in .env). "
            "Use DeepSeek, or set ANTHROPIC_ENABLED=true to allow Claude again."
        )
    if not settings.anthropic_api_key:
        raise ConfigError("ANTHROPIC_API_KEY is not set in .env.")
    headers = (
        {"anthropic-workspace-id": settings.workspace_id} if settings.workspace_id else None
    )
    return ChatAnthropic(
        model=settings.model,
        api_key=settings.anthropic_api_key,
        max_tokens=max_tokens,
        default_headers=headers,
        callbacks=callbacks,
        reasoning_effort=effort,
    )


class JsonModeStructured:
    """Structured output for providers that cannot enforce a schema (DeepSeek).

    Uses the provider's JSON mode, puts the JSON schema in the prompt, validates the reply with
    Pydantic and, if it is empty or invalid, asks the model once or twice to correct it.
    """

    def __init__(self, llm: BaseChatModel, schema: type[T], retries: int = 2) -> None:
        self._llm = llm.bind(response_format={"type": "json_object"})
        self._schema = schema
        self._retries = retries

    def _instruction(self) -> str:
        return (
            "Respond ONLY with one JSON object (json, no markdown fences, no commentary) that "
            "validates against this JSON schema. Fill every required field and add no others.\n"
            f"JSON schema:\n{json.dumps(self._schema.model_json_schema(), ensure_ascii=False)}"
        )

    def _with_instruction(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        if messages and isinstance(messages[0], SystemMessage):
            first = SystemMessage(f"{messages[0].content}\n\n{self._instruction()}")
            return [first, *messages[1:]]
        return [SystemMessage(self._instruction()), *messages]

    def _parse(self, text: str) -> BaseModel:
        text = text.strip()
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
        if fenced:
            text = fenced.group(1)
        if not text:
            raise ValueError("the reply was empty")
        # Take the first JSON object and ignore anything after it: a model sometimes appends a second
        # object or a comment, and rejecting a complete valid reply wasted the whole row.
        start = text.find("{")
        value, _end = json.JSONDecoder().raw_decode(text[start:] if start >= 0 else text)
        return self._schema.model_validate(value)

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> BaseModel:
        conversation = self._with_instruction(list(messages))
        error: Exception | None = None
        for _ in range(self._retries + 1):
            reply = invoke_with_backoff(lambda: self._llm.invoke(conversation, **kwargs))
            text = message_text(reply)
            try:
                return self._parse(text)
            except (ValueError, ValidationError) as e:  # JSONDecodeError is a ValueError
                error = e
                conversation = [
                    *conversation,
                    reply,
                    HumanMessage(
                        f"Your previous reply was not valid: {str(e)[:400]}. "
                        "Return the corrected JSON object only."
                    ),
                ]
        raise ValueError(f"The model did not return valid JSON for {self._schema.__name__}: {error}")


def structured_llm(llm: BaseChatModel, schema: type[T]) -> Any:
    """Return a runnable whose `invoke(messages)` gives a validated `schema` instance."""
    if isinstance(llm, ChatDeepSeek):
        return JsonModeStructured(llm, schema)
    return llm.with_structured_output(schema, method="json_schema")


def message_text(message: BaseMessage) -> str:
    """Extract plain text from a model reply, skipping thinking/tool blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def check_connection(settings: Settings, provider: str = "anthropic") -> str:
    """Send a trivial prompt to the provider and return its text reply."""
    model = get_chat_model(settings, max_tokens=2048, provider=provider)
    reply = invoke_with_backoff(lambda: model.invoke("Reply with exactly the word: OK"))
    return message_text(reply).strip()
