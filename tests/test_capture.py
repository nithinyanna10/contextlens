import tiktoken
import pytest
from pydantic import ValidationError

from contextlens.capture import CapturedCall, TokenSpan, UsageRecord

_ENC = tiktoken.encoding_for_model("gpt-4")


def _tok(text: str) -> int:
    return len(_ENC.encode(text))


# --- fixtures -----------------------------------------------------------------

_SYS = "You are a helpful assistant."
_HIST = "What is contextlens?"
_SYS_TOK = _tok(_SYS)
_HIST_TOK = _tok(_HIST)

# Simulated API response total: always > content sum (role markers, framing, BOS)
_API_TOTAL = _SYS_TOK + _HIST_TOK + 12
_FMT_TOK = _API_TOTAL - _SYS_TOK - _HIST_TOK  # gap absorbed by formatting span


def _realized_call() -> CapturedCall:
    return CapturedCall(
        provider="openai",
        model="gpt-4",
        call_index=0,
        prompt=_SYS + "\n" + _HIST,
        spans=[
            TokenSpan(component="system", text=_SYS, token_count=_SYS_TOK),
            TokenSpan(component="history", text=_HIST, token_count=_HIST_TOK),
            TokenSpan(component="formatting", text="", token_count=_FMT_TOK),
        ],
        usage=UsageRecord(provider="openai", input_tokens=_API_TOTAL, source="response_usage"),
    )


# --- schema -------------------------------------------------------------------

def test_captured_call_has_usage_field_with_correct_shape():
    call = _realized_call()
    assert call.usage is not None
    assert call.usage.provider == "openai"
    assert call.usage.input_tokens == _API_TOTAL
    assert call.usage.total_input_tokens == _API_TOTAL  # no cache → total == input_tokens
    assert call.usage.source == "response_usage"


def test_preflight_call_accepts_none_usage():
    call = CapturedCall(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        call_index=0,
        prompt=_SYS,
        spans=[TokenSpan(component="system", text=_SYS, token_count=_SYS_TOK)],
        usage=None,
    )
    assert call.usage is None


# --- attribution invariant ----------------------------------------------------

def test_content_spans_alone_do_not_equal_usage():
    """tiktoken estimates for content never reach provider total — that is expected."""
    call = _realized_call()
    content_sum = sum(s.token_count for s in call.spans if s.component != "formatting")
    assert content_sum < call.usage.total_input_tokens


def test_formatting_span_reconciles_to_usage():
    """content + formatting == usage.total_input_tokens (the accounting identity)."""
    call = _realized_call()
    content_sum = sum(s.token_count for s in call.spans if s.component != "formatting")
    fmt = next(s.token_count for s in call.spans if s.component == "formatting")
    assert content_sum + fmt == call.usage.total_input_tokens


def test_all_spans_sum_to_usage():
    """Validator enforces this; test makes it explicit."""
    call = _realized_call()
    assert sum(s.token_count for s in call.spans) == call.usage.total_input_tokens


# --- validator ----------------------------------------------------------------

def test_validator_rejects_spans_that_do_not_reconcile_with_usage():
    """Omitting the formatting span should fail validation."""
    with pytest.raises(ValidationError, match="formatting"):
        CapturedCall(
            provider="openai",
            model="gpt-4",
            call_index=0,
            prompt=_SYS,
            spans=[TokenSpan(component="system", text=_SYS, token_count=_SYS_TOK)],
            usage=UsageRecord(provider="openai", input_tokens=_API_TOTAL, source="response_usage"),
        )
