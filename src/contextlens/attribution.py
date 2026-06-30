from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, computed_field

from contextlens.capture import CapturedCall

SourceLabel = Literal["exact", "estimate"]

_ALL_TAGS = ("system", "history", "retrieved", "tool_output", "scratchpad", "formatting")
_NON_ACTIONABLE = frozenset({"formatting"})


class ComponentTotal(BaseModel):
    """Token count for one component tag."""

    model_config = ConfigDict(frozen=True)

    token_count: int
    non_actionable: bool  # True for 'formatting' — user cannot trim these tokens


class AttributionResult(BaseModel):
    """Per-component breakdown of a CapturedCall's token budget.

    components: one entry per ComponentTag, always — zero count if no spans.
    source:     "exact"    → realized call, totals from usage.input_tokens.
                "estimate" → pre-flight, totals are tiktoken sums.
    total:      derived, cannot drift — always sum(components.*.token_count).
    """

    model_config = ConfigDict(frozen=True)

    components: dict[str, ComponentTotal]  # keys are ComponentTag values
    source: SourceLabel

    @computed_field
    @property
    def total(self) -> int:
        return sum(c.token_count for c in self.components.values())


def attribute(call: CapturedCall) -> AttributionResult:
    """Return per-component token totals for a CapturedCall.

    Works on both realized calls (source="exact") and pre-flight estimates
    (source="estimate"). Every ComponentTag key is always present; missing
    tags return token_count=0.
    """
    totals: dict[str, int] = {tag: 0 for tag in _ALL_TAGS}
    for span in call.spans:
        totals[span.component] += span.token_count

    source: SourceLabel = "exact" if call.usage is not None else "estimate"

    return AttributionResult(
        components={
            tag: ComponentTotal(
                token_count=totals[tag],
                non_actionable=tag in _NON_ACTIONABLE,
            )
            for tag in _ALL_TAGS
        },
        source=source,
    )
