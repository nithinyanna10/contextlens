import pytest

from contextlens.attribution import attribute, AttributionResult
from contextlens.capture import CapturedCall, TokenSpan, UsageRecord

# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------

def _preflight(spans: list[TokenSpan]) -> CapturedCall:
    return CapturedCall(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        call_index=0,
        prompt="test",
        spans=spans,
        usage=None,
    )


def _realized(spans: list[TokenSpan], api_total: int) -> CapturedCall:
    return CapturedCall(
        provider="openai",
        model="gpt-4",
        call_index=0,
        prompt="test",
        spans=spans,
        usage=UsageRecord(provider="openai", input_tokens=api_total, source="response_usage"),
    )


# ---------------------------------------------------------------------------
# 1. empty-tag test
# ---------------------------------------------------------------------------

def test_missing_tags_return_zero_with_key_present():
    """All six ComponentTag keys always appear in the result."""
    call = _preflight([
        TokenSpan(component="history", text="first turn", token_count=5),
        TokenSpan(component="history", text="second turn", token_count=4),
        TokenSpan(component="retrieved", text="doc chunk", token_count=6),
    ])
    result = attribute(call)

    assert result.components["history"].token_count == 5 + 4
    assert result.components["retrieved"].token_count == 6

    for tag in ("system", "tool_output", "scratchpad", "formatting"):
        assert tag in result.components, f"missing key: {tag}"
        assert result.components[tag].token_count == 0


# ---------------------------------------------------------------------------
# 2. reconciliation test (realized call)
# ---------------------------------------------------------------------------

def test_realized_call_totals_reconcile_to_usage_and_source_is_exact():
    sys_tok, hist_tok, fmt_tok = 6, 5, 8
    api_total = sys_tok + hist_tok + fmt_tok

    call = _realized(
        spans=[
            TokenSpan(component="system", text="s", token_count=sys_tok),
            TokenSpan(component="history", text="h", token_count=hist_tok),
            TokenSpan(component="formatting", text="", token_count=fmt_tok),
        ],
        api_total=api_total,
    )
    result = attribute(call)

    assert result.source == "exact"
    assert result.total == api_total
    assert result.total == call.usage.input_tokens


# ---------------------------------------------------------------------------
# 3. estimate test (pre-flight, no usage, no formatting span)
# ---------------------------------------------------------------------------

def test_preflight_call_uses_estimate_source_and_formatting_is_zero():
    """Pre-flight has no formatting span and no usage.
    Constructing the call here also proves the validator stays quiet on
    pre-flight — if this raises ValidationError, capture.py is misfiring.
    """
    call = _preflight([
        TokenSpan(component="system", text="sys", token_count=3),
        TokenSpan(component="history", text="hi", token_count=2),
    ])
    result = attribute(call)

    assert result.source == "estimate"
    assert result.components["system"].token_count == 3
    assert result.components["history"].token_count == 2
    assert result.components["formatting"].token_count == 0
    assert result.total == 3 + 2


# ---------------------------------------------------------------------------
# 4. non_actionable flag
# ---------------------------------------------------------------------------

def test_formatting_is_non_actionable_all_content_tags_are_actionable():
    call = _preflight([
        TokenSpan(component="system", text="s", token_count=2),
    ])
    result = attribute(call)

    assert result.components["formatting"].non_actionable is True
    for tag in ("system", "history", "retrieved", "tool_output", "scratchpad"):
        assert result.components[tag].non_actionable is False
