"""Stress tests for usage accounting and reconciliation — cached-token path.

All tests are offline: synthetic payloads only, no API calls.
Run with: pytest tests/stress/ -m stress -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from contextlens.assertions import check_rot_risk
from contextlens.attribution import attribute
from contextlens.capture import CapturedCall, TokenSpan, UsageRecord
from contextlens.integrations.langchain import ContextLensCallbackHandler
from contextlens.integrations.langgraph import _build_call, _extract_usage_from_response
from contextlens.integrations.openai_sdk import build_captured_call
from contextlens.report import build_report, dump_call, load_call, render

pytestmark = pytest.mark.stress


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_call(
    *,
    input_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
    content_tokens: int | None = None,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-20241022",
) -> CapturedCall:
    """Build a CapturedCall with explicit cache fields."""
    total = input_tokens + cache_read + cache_creation
    content = content_tokens if content_tokens is not None else max(0, total - 6)
    fmt = total - content
    return CapturedCall(
        provider=provider,  # type: ignore[arg-type]
        model=model,
        call_index=0,
        prompt="x" * content,
        spans=[
            TokenSpan(component="system", text="x" * content, token_count=content),
            TokenSpan(component="formatting", text="", token_count=fmt),
        ],
        usage=UsageRecord(
            provider=provider,  # type: ignore[arg-type]
            input_tokens=input_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
            source="response_usage",
        ),
    )


# ---------------------------------------------------------------------------
# 1a. UsageRecord.total_input_tokens
# ---------------------------------------------------------------------------

class TestTotalInputTokens:
    def test_no_cache_total_equals_input_tokens(self):
        u = UsageRecord(provider="anthropic", input_tokens=500, source="response_usage")
        assert u.total_input_tokens == 500
        assert u.cache_read_input_tokens == 0
        assert u.cache_creation_input_tokens == 0

    def test_cache_read_adds_to_total(self):
        u = UsageRecord(
            provider="anthropic",
            input_tokens=10,
            cache_read_input_tokens=500,
            source="response_usage",
        )
        assert u.total_input_tokens == 510

    def test_cache_creation_adds_to_total(self):
        u = UsageRecord(
            provider="anthropic",
            input_tokens=100,
            cache_creation_input_tokens=400,
            source="response_usage",
        )
        assert u.total_input_tokens == 500

    def test_all_three_fields_sum(self):
        u = UsageRecord(
            provider="anthropic",
            input_tokens=10,
            cache_read_input_tokens=90_000,
            cache_creation_input_tokens=5_000,
            source="response_usage",
        )
        assert u.total_input_tokens == 95_010

    def test_old_jsonl_without_cache_fields_loads_correctly(self):
        """Additive defaults: old JSONL must load with cache fields = 0."""
        old_json = json.dumps({
            "provider": "openai",
            "input_tokens": 200,
            "source": "response_usage",
        })
        u = UsageRecord.model_validate_json(old_json)
        assert u.cache_read_input_tokens == 0
        assert u.cache_creation_input_tokens == 0
        assert u.total_input_tokens == 200


# ---------------------------------------------------------------------------
# 1b. Reconciliation invariant against total_input_tokens
# ---------------------------------------------------------------------------

class TestReconciliationInvariant:
    def test_cached_call_reconciles_against_total(self):
        """The key invariant: sum(spans) == total_input_tokens, NOT input_tokens."""
        call = _make_call(input_tokens=10, cache_read=500, content_tokens=480)
        span_sum = sum(s.token_count for s in call.spans)
        assert span_sum == call.usage.total_input_tokens   # 510
        assert span_sum != call.usage.input_tokens         # 10 — NOT this

    def test_validator_rejects_spans_reconciled_to_raw_input_tokens_only(self):
        """Constructing a call where sum(spans)==input_tokens but !=total triggers error."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="total_input_tokens"):
            CapturedCall(
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                call_index=0,
                prompt="x" * 10,
                spans=[
                    TokenSpan(component="system", text="x" * 10, token_count=10),
                    # No formatting — sum==10 == input_tokens, but total==510
                ],
                usage=UsageRecord(
                    provider="anthropic",
                    input_tokens=10,
                    cache_read_input_tokens=500,
                    source="response_usage",
                ),
            )

    def test_cache_creation_call_reconciles(self):
        call = _make_call(input_tokens=100, cache_creation=4_000, content_tokens=4_060)
        assert sum(s.token_count for s in call.spans) == call.usage.total_input_tokens

    def test_both_cache_fields_present(self):
        # First call writes cache; subsequent call reads it — both fields non-zero
        call = _make_call(input_tokens=5, cache_read=3_000, cache_creation=500, content_tokens=3_460)
        assert sum(s.token_count for s in call.spans) == call.usage.total_input_tokens

    def test_jsonl_roundtrip_preserves_cache_fields(self):
        call = _make_call(input_tokens=10, cache_read=500)
        line = dump_call(call)
        loaded = load_call(line)
        assert loaded.usage.cache_read_input_tokens == 500
        assert loaded.usage.total_input_tokens == 510
        assert sum(s.token_count for s in loaded.spans) == 510


# ---------------------------------------------------------------------------
# 1c. Budget and rot-risk use total_input_tokens
# ---------------------------------------------------------------------------

class TestBudgetAndRotRiskUseTotals:
    def test_rot_risk_uses_total_not_raw_input_tokens(self):
        """The silent-wrong case before the fix: 10/128k passes, 510/128k should too
        but the number must be 510, not 10."""
        call = _make_call(input_tokens=10, cache_read=500)
        violations = check_rot_risk(call, below=0.9)
        assert violations == []
        # Confirm the score was computed against the total
        total_used = call.usage.total_input_tokens  # 510
        assert total_used == 510

    def test_rot_risk_fires_on_large_cached_call(self):
        """A call where cache_read fills the window: must violate at a tight threshold.

        90k cached + 5k new = 95k total; gpt-4o window = 128k; ratio ≈ 74.2%.
        Use below=0.70 so 74.2% >= 70% triggers a violation.
        """
        call = _make_call(
            input_tokens=5_000,
            cache_read=90_000,
            content_tokens=94_940,
            model="gpt-4o",
            provider="openai",
        )
        violations = check_rot_risk(call, below=0.70)
        assert len(violations) == 1
        assert "95000" in violations[0].message or "95,000" in violations[0].message

    def test_attribution_total_matches_total_input_tokens(self):
        call = _make_call(input_tokens=10, cache_read=500)
        attr = attribute(call)
        assert attr.total == call.usage.total_input_tokens


# ---------------------------------------------------------------------------
# 1d. OpenAI SDK adapter — cached path
# ---------------------------------------------------------------------------

class TestOpenAISdkCachedPath:
    def _mock_response(self, prompt_tokens: int, cached_tokens: int) -> MagicMock:
        r = MagicMock()
        r.usage.prompt_tokens = prompt_tokens
        r.usage.prompt_tokens_details.cached_tokens = cached_tokens
        return r

    def test_cached_tokens_split_correctly(self):
        r = self._mock_response(prompt_tokens=510, cached_tokens=500)
        messages = [{"role": "system", "content": "hello"}]
        call = build_captured_call("gpt-4o", messages, r)
        assert call.usage.cache_read_input_tokens == 500
        assert call.usage.input_tokens == 10          # 510 - 500
        assert call.usage.total_input_tokens == 510
        assert sum(s.token_count for s in call.spans) == 510

    def test_no_cached_tokens_details_field_defaults_to_zero(self):
        r = MagicMock()
        r.usage.prompt_tokens = 100
        r.usage.prompt_tokens_details = None
        messages = [{"role": "user", "content": "hello"}]
        call = build_captured_call("gpt-4o", messages, r)
        assert call.usage.cache_read_input_tokens == 0
        assert call.usage.input_tokens == 100
        assert call.usage.total_input_tokens == 100


# ---------------------------------------------------------------------------
# 1e. LangChain adapter — Anthropic cached path
# ---------------------------------------------------------------------------

class TestLangChainAnthropicCachedPath:
    def _make_response(
        self,
        input_tokens: int,
        cache_read: int = 0,
        cache_creation: int = 0,
    ) -> MagicMock:
        response = MagicMock()
        response.llm_output = {
            "usage": {
                "input_tokens": input_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            }
        }
        return response

    def _make_handler(self, serialized_name: str = "ChatAnthropic") -> ContextLensCallbackHandler:
        handler = ContextLensCallbackHandler()
        # Seed pending state as on_chat_model_start would
        msg = MagicMock()
        msg.type = "human"
        msg.content = "What is RAG?"
        handler._pending_messages = [msg]
        handler._pending_serialized = {"name": serialized_name, "kwargs": {"model_name": "claude-3-5-sonnet-20241022"}}
        return handler

    def test_cache_read_captured(self):
        handler = self._make_handler()
        handler.on_llm_end(self._make_response(input_tokens=10, cache_read=500))
        call = handler.calls[0]
        assert call.usage.cache_read_input_tokens == 500
        assert call.usage.input_tokens == 10
        assert call.usage.total_input_tokens == 510
        assert sum(s.token_count for s in call.spans) == 510

    def test_cache_creation_captured(self):
        handler = self._make_handler()
        handler.on_llm_end(self._make_response(input_tokens=100, cache_creation=400))
        call = handler.calls[0]
        assert call.usage.cache_creation_input_tokens == 400
        assert call.usage.total_input_tokens == 500
        assert sum(s.token_count for s in call.spans) == 500

    def test_no_cache_fields_defaults_to_zero(self):
        handler = self._make_handler()
        handler.on_llm_end(self._make_response(input_tokens=200))
        call = handler.calls[0]
        assert call.usage.cache_read_input_tokens == 0
        assert call.usage.total_input_tokens == 200


# ---------------------------------------------------------------------------
# 1f. LangGraph adapter — Anthropic cached path
# ---------------------------------------------------------------------------

class TestLangGraphAnthropicCachedPath:
    def _ai_msg(
        self,
        input_tokens: int,
        cache_read: int = 0,
        cache_creation: int = 0,
    ) -> MagicMock:
        msg = MagicMock()
        msg.response_metadata = {
            "usage": {
                "input_tokens": input_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            }
        }
        # No usage_metadata on this path
        del msg.usage_metadata
        return msg

    def test_extract_returns_all_cache_fields(self):
        msg = self._ai_msg(input_tokens=10, cache_read=500, cache_creation=0)
        result = _extract_usage_from_response(msg)
        assert result is not None
        inp, cache_read, cache_creation, source = result
        assert inp == 10
        assert cache_read == 500
        assert cache_creation == 0
        assert source == "response_usage"

    def test_build_call_reconciles_against_total(self):
        msg = self._ai_msg(input_tokens=10, cache_read=90_000)
        input_messages: list = []  # no input messages to keep test simple
        call = _build_call(
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            input_messages=input_messages,
            response_msg=msg,
            call_index=0,
        )
        assert call.usage is not None
        assert call.usage.total_input_tokens == 90_010
        assert sum(s.token_count for s in call.spans) == 90_010


# ---------------------------------------------------------------------------
# 1g. Report renders cache breakdown when present
# ---------------------------------------------------------------------------

class TestReportCacheDisplay:
    def test_report_shows_cache_breakdown_line(self):
        call = _make_call(input_tokens=5_000, cache_read=90_000, content_tokens=94_940)
        report = build_report([call])
        text = render(report)
        assert "cached read" in text
        assert "90,000" in text

    def test_report_no_cache_line_when_uncached(self):
        call = _make_call(input_tokens=200)
        report = build_report([call])
        text = render(report)
        assert "cached" not in text

    def test_cache_creation_shows_in_report(self):
        call = _make_call(input_tokens=100, cache_creation=4_000, content_tokens=4_060)
        report = build_report([call])
        text = render(report)
        assert "cache write" in text
        assert "4,000" in text
