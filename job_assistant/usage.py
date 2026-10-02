"""Token and cost accounting for LLM calls, so spending is measured instead of guessed."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# USD per million tokens (input, output). Output includes the model's thinking tokens, which
# are billed as output.
ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
# DeepSeek list prices (cache-miss input, output), https://api-docs.deepseek.com/quick_start/pricing
# Off-peak is half of peak. Cache hits are billed far lower; counting every input token as a
# miss makes the estimate slightly high, never low.
DEEPSEEK_PEAK: dict[str, tuple[float, float]] = {
    "deepseek-flash": (0.30, 1.20),
    "deepseek-v4-pro": (1.32, 3.96),
}


def is_deepseek_peak(when: datetime) -> bool:
    """Peak: Monday to Friday, 01:00-04:00 and 06:00-10:00 UTC."""
    utc = when.astimezone(timezone.utc)
    return utc.weekday() < 5 and (1 <= utc.hour < 4 or 6 <= utc.hour < 10)


def price_per_mtok(model: str, when: datetime | None = None) -> tuple[float, float] | None:
    """(input, output) USD per million tokens, or None when the model is not in the table."""
    if model in ANTHROPIC_PRICES:
        return ANTHROPIC_PRICES[model]
    if model in DEEPSEEK_PEAK:
        peak_in, peak_out = DEEPSEEK_PEAK[model]
        if is_deepseek_peak(when or datetime.now(timezone.utc)):
            return peak_in, peak_out
        return peak_in / 2, peak_out / 2
    return None


@dataclass(frozen=True)
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = 0.0  # USD; None once any call used a model without a known price

    def __add__(self, other: Usage) -> Usage:
        cost = None if self.cost is None or other.cost is None else self.cost + other.cost
        return Usage(self.calls + other.calls, self.input_tokens + other.input_tokens,
                     self.output_tokens + other.output_tokens, cost)

    def __sub__(self, other: Usage) -> Usage:
        cost = None if self.cost is None or other.cost is None else self.cost - other.cost
        return Usage(self.calls - other.calls, self.input_tokens - other.input_tokens,
                     self.output_tokens - other.output_tokens, cost)

    def describe(self) -> str:
        money = f" ≈ ${self.cost:.3f}" if self.cost is not None else ""
        return f"{self.calls} calls, {self.input_tokens:,} in / {self.output_tokens:,} out tokens{money}"


class UsageTracker(BaseCallbackHandler):
    """Adds up the token usage (and cost) of every call made by the model it is attached to."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._usage = Usage()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                meta = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if not meta:
                    continue
                tokens_in, tokens_out = int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
                prices = price_per_mtok(self.model)
                cost = None if prices is None else (tokens_in * prices[0] + tokens_out * prices[1]) / 1_000_000
                self._usage = self._usage + Usage(1, tokens_in, tokens_out, cost)

    def snapshot(self) -> Usage:
        return self._usage


class UsageGroup:
    """Combines the trackers of several models (one per provider) into one running total."""

    def __init__(self, trackers: list[UsageTracker]) -> None:
        self.trackers = trackers

    def snapshot(self) -> Usage:
        total = Usage()
        for tracker in self.trackers:
            total = total + tracker.snapshot()
        return total
