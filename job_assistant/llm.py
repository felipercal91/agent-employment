"""Single place where the Claude chat model is built, so every agent shares the same config."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

from job_assistant.config import Settings


def get_chat_model(
    settings: Settings, max_tokens: int = 8192, callbacks: list[BaseCallbackHandler] | None = None
) -> ChatAnthropic:
    headers = (
        {"anthropic-workspace-id": settings.workspace_id} if settings.workspace_id else None
    )
    return ChatAnthropic(
        model=settings.model,
        api_key=settings.anthropic_api_key,
        max_tokens=max_tokens,
        default_headers=headers,
        callbacks=callbacks,
    )


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


def check_connection(settings: Settings) -> str:
    """Send a trivial prompt to Claude and return its text reply."""
    model = get_chat_model(settings, max_tokens=2048)
    reply = model.invoke("Reply with exactly the word: OK")
    return message_text(reply).strip()
