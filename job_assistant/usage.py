"""Token and cost accounting for LLM calls, so spending is measured instead of guessed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# USD per million tokens (input, output). Anthropic first-party list prices; output includes
# the model's thinking tokens, which are billed as output.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


@dataclass(frozen=True)
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __sub__(self, other: Usage) -> Usage:
        return Usage(self.calls - other.calls, self.input_tokens - other.input_tokens,
                     self.output_tokens - other.output_tokens)

    def cost_usd(self, model: str) -> float | None:
        prices = PRICES_PER_MTOK.get(model)
        if prices is None:
            return None
        return (self.input_tokens * prices[0] + self.output_tokens * prices[1]) / 1_000_000

    def describe(self, model: str) -> str:
        cost = self.cost_usd(model)
        money = f" ≈ ${cost:.3f}" if cost is not None else ""
        return f"{self.calls} calls, {self.input_tokens:,} in / {self.output_tokens:,} out tokens{money}"


class UsageTracker(BaseCallbackHandler):
    """Adds up the token usage reported by every model call it is attached to."""

    def __init__(self) -> None:
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                meta = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if meta:
                    self.calls += 1
                    self.input_tokens += int(meta.get("input_tokens", 0))
                    self.output_tokens += int(meta.get("output_tokens", 0))

    def snapshot(self) -> Usage:
        return Usage(self.calls, self.input_tokens, self.output_tokens)
