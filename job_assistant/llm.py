"""Single place where chat models are built, so every agent shares the same configuration.

Two providers are supported: Anthropic (default) and DeepSeek (optional, cheaper). Each agent
role can use a different one (see WRITER_PROVIDER / REVIEW_PROVIDER in .env).
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, ValidationError

from job_assistant.config import ConfigError, Settings

T = TypeVar("T", bound=BaseModel)

PROVIDERS = ("anthropic", "deepseek")


def get_chat_model(
    settings: Settings,
    max_tokens: int = 8192,
    callbacks: list[BaseCallbackHandler] | None = None,
    effort: str | None = None,
    provider: str = "anthropic",
) -> BaseChatModel:
    """`effort` (low/medium/high) sets how much Claude reasons; None = the model's default.

    For DeepSeek the reasoning mode is a setting (`DEEPSEEK_THINKING`), not an effort level.
    """
    if provider == "deepseek":
        if not settings.deepseek_api_key:
            raise ConfigError("DEEPSEEK_API_KEY is not set in .env.")
        return ChatDeepSeek(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            max_tokens=max_tokens,
            callbacks=callbacks,
            extra_body=None if settings.deepseek_thinking else {"thinking": {"type": "disabled"}},
        )
    if provider != "anthropic":
        raise ConfigError(f"Unknown provider '{provider}' (use one of: {', '.join(PROVIDERS)}).")
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
        return self._schema.model_validate(json.loads(text))

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> BaseModel:
        conversation = self._with_instruction(list(messages))
        error: Exception | None = None
        for _ in range(self._retries + 1):
            reply = self._llm.invoke(conversation, **kwargs)
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
    reply = model.invoke("Reply with exactly the word: OK")
    return message_text(reply).strip()
