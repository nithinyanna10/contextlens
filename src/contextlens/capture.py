from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

ProviderName = Literal["openai", "anthropic"]

UsageSource = Literal[
    "response_usage",  # read from response.usage — authoritative
    "estimated",       # tiktoken pre-flight, ceiling ~10% (higher for code/non-English)
    "exact",           # anthropic messages.count_tokens() opt-in
]

ComponentTag = Literal[
    "system",
    "history",
    "retrieved",
    "tool_output",
    "scratchpad",
    "formatting",  # role markers, BOS, message framing — absorbs gap vs usage.input_tokens
]


class TokenSpan(BaseModel):
    """A labelled segment of the assembled prompt."""

    model_config = ConfigDict(frozen=True)

    component: ComponentTag
    text: str
    token_count: int


class UsageRecord(BaseModel):
    """Token counts as reported by the provider API.

    source="response_usage" is the only authoritative value.
    "estimated" and "exact" are pre-call modes.
    """

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    input_tokens: int
    source: UsageSource


class CapturedCall(BaseModel):
    """Immutable record of one LLM call's assembled context.

    For realized calls (usage is not None), every span's token_count is a
    tiktoken ESTIMATE — not the provider total. The 'formatting' span carries
    the gap between content estimates and usage.input_tokens so that
    sum(all spans) == usage.input_tokens exactly.

    Do NOT assert sum(non-formatting spans) == usage.input_tokens.
    That is always false and the failure is correct.
    """

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    model: str
    call_index: int        # 0-based within the test session
    prompt: str            # full assembled prompt text
    spans: list[TokenSpan]
    usage: UsageRecord | None = None  # None for pre-flight estimates

    @model_validator(mode="after")
    def _validate_span_total(self) -> CapturedCall:
        if self.usage is None:
            return self
        span_total = sum(s.token_count for s in self.spans)
        if span_total != self.usage.input_tokens:
            raise ValueError(
                f"sum(spans)={span_total} != usage.input_tokens={self.usage.input_tokens}. "
                "Add a 'formatting' span to absorb role/framing overhead."
            )
        return self
